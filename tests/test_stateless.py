"""Despacho v2 (app/stateless.py): el turno sin estado propio.

Dos niveles, a propósito:
- HTTP (`client.post("/dispatch", ...)`, firmado): enruta por `version`,
  aísla el Store y no filtra secretos a los logs.
- Turno (`stateless.run_turn(...)` directo, como test_tools.py hace con
  `ToolRuntime`): gates, prompt, media, envío y LLM — sin la ceremonia de
  firmar cada payload.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx
import pytest

from app import stateless
from app.crm import CrmClient
from app.llm import LlmAuthFailed, LlmNoCredits, LlmReply, ToolCall
from app.state import AppContext, NullStore, TurnCommit
from tests.conftest import CRM_URL, IDENTITY, FakeLLM, make_ctx, mock_crm_basics

SECRET = "test-key"  # CRM_BOT_API_KEY por defecto en tests.conftest.make_settings
SLOT_ISO = "2026-07-20T16:00:00Z"


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# ------------------------------------------------------------- fixtures ---


def hist(
    role: str,
    text: str | None = None,
    *,
    id: str | None = "m1",  # noqa: A002 - nombre del campo del contrato
    type: str = "text",  # noqa: A002
    pending: bool = False,
    media: dict[str, Any] | None = None,
    at: str = "2026-09-25T12:00:00Z",
) -> dict[str, Any]:
    return {
        "id": id,
        "role": role,
        "type": type,
        "text": text,
        "at": at,
        "pending": pending,
        "media": media,
    }


def v2_raw(
    *,
    organization_id: str | None = "org_a",
    conversation_id: str | None = "cv_v2_1",
    is_test: bool = False,
    dispatch_id: str = "dsp_1",
    attempt: int = 0,
    ai_enabled: bool = True,
    window_open: bool = True,
    agent_has_spoken: bool = True,
    allowlist_enabled: bool = False,
    allowed_wa_ids: list[str] | None = None,
    identity: str = IDENTITY,
    history: list[dict[str, Any]] | None = None,
    offers: list[dict[str, Any]] | None = None,
    llm: dict[str, Any] | None = None,
    ad_headline: str | None = None,
    activation_enabled: bool = False,
    activation_messages: list[str] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "contact": {
            "id": "ct_1",
            "name": "Lead de Prueba",
            "waIdentity": identity,
            "phone": identity,
            "ficha": {},
        },
        "conversation": {
            "id": conversation_id,
            "aiEnabled": ai_enabled,
            "handoffAt": None,
            "handoffReason": None,
            "windowOpen": window_open,
            "windowRemainingMs": 3600000,
            "agentHasSpoken": agent_has_spoken,
        },
        "lead": {"stageName": "Nuevo"},
        "agentAccess": {
            "allowlistEnabled": allowlist_enabled,
            "allowedWaIds": allowed_wa_ids or [],
        },
        "booking": {"next": None},
        "adOrigen": (
            {"headline": ad_headline, "body": None, "sourceId": None, "sourceType": None, "sourceUrl": None}
            if ad_headline
            else None
        ),
    }
    profile = {
        "profile": {
            "name": "Nea",
            "tone": None,
            "instructions": None,
            "escalationRules": None,
            "greeting": None,
            "activationEnabled": activation_enabled,
            "activationMessages": activation_messages or [],
            "timezone": None,
            "businessTimezone": None,
            "businessHours": None,
            "responseMode": None,
        },
        "kb": None,
        "resources": [],
    }
    default_history = (
        history if history is not None else [hist("lead", "hola", pending=True)]
    )
    return {
        "version": 2,
        "dispatchId": dispatch_id,
        "attempt": attempt,
        "organizationId": organization_id,
        "conversationId": conversation_id,
        "isTest": is_test,
        # campos v1, deben ignorarse en v2 (se dejan para probar justamente eso)
        "contact": {"identity": identity, "name": "Lead de Prueba"},
        "messages": [],
        "context": context,
        "profile": profile,
        "history": default_history,
        "offers": offers or [],
        "llm": llm,
    }


def v2_payload(**kwargs: Any) -> stateless.DispatchPayloadV2:
    return stateless.DispatchPayloadV2.model_validate(v2_raw(**kwargs))


def v2_body(**kwargs: Any) -> bytes:
    return json.dumps(v2_raw(**kwargs)).encode("utf-8")


def last_user_content(llm: FakeLLM) -> Any:
    return llm.calls[-1]["messages"][-1]["content"]


# ================================================================ Dispatch ===


async def test_v2_nunca_toca_el_store(ctx: AppContext, client, respx_mock):
    """CRÍTICO: el despacho v2 completo (gates, media, LLM, envío, handoff)
    corre de punta a punta sin llamar NUNCA a ctx.store — se lo certifica con
    un Store que revienta ante cualquier llamada."""
    ctx.store = NullStore()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    body = v2_body()
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["action"] == "replied"
    assert routes["messages"].call_count == 1


async def test_v2_sobre_de_respuesta_trae_llm_source_status_y_answeredwith(client, respx_mock):
    """El sobre HTTP completo del contrato: `llm.source`/`llm.status` son el
    contrato mínimo; `answeredWith` es el extra informativo — el CRM no
    valida el body estrictamente (ver safeResponseBody en nea-dispatch.ts),
    así que el campo de más no rompe nada del otro lado."""
    mock_crm_basics(respx_mock, conv_id="cv_v2_envelope")
    body = v2_body(conversation_id="cv_v2_envelope")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm"] == {"source": "platform", "status": "ok", "answeredWith": "platform"}
    assert data["handoff"] is None


async def test_v1_sin_version_sigue_usando_el_store(ctx: AppContext, client, respx_mock):
    """Contraprueba de la anterior: un despacho SIN `version` (v1) sí toca el
    Store (dedup por mark_processed) — con un NullStore revienta de
    inmediato, lo que demuestra que el ruteo por versión de verdad manda cada
    uno por su camino (si v1 se hubiera colado a la ruta v2, esto pasaría de
    largo sin tocar nada)."""
    from tests.test_dispatch import dispatch_body, sign as sign_v1

    ctx.store = NullStore()
    mock_crm_basics(respx_mock, conv_id="cv_v1_x")
    respx_mock.get(f"{CRM_URL}/api/bot/profile").mock(return_value=httpx.Response(404))
    body = dispatch_body(organization_id="org_a", conversation_id="cv_v1_x", wamid="wamid.v1x")
    with pytest.raises(RuntimeError, match="NullStore"):
        await client.post("/dispatch", content=body, headers={"x-signature": sign_v1(body)})


async def test_v2_apikey_nunca_aparece_en_los_logs(ctx: AppContext, client, caplog):
    """El apiKey de un negocio no debe imprimirse ni siquiera cuando el
    payload es inválido y dispatch.py loguea el motivo del 400."""
    secreto = "sk-super-secreto-de-un-negocio-jamas-debe-aparecer"
    raw = v2_raw(llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": secreto})
    del raw["dispatchId"]  # obligatorio: sin él, pydantic SÍ revienta con ValidationError
    body = json.dumps(raw).encode("utf-8")
    with caplog.at_level(logging.WARNING):
        resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400
    assert secreto not in caplog.text
    # de verdad fue un ValidationError por dispatchId, no el 400 de
    # organizationId/conversationId (que no depende de pydantic) — si esta
    # aserción fallara, el test de arriba dejaría de probar nada.
    assert "dispatchId" in caplog.text


async def test_v2_dispatchid_ausente_400(client):
    """MINOR: dispatchId es obligatorio en el contrato v2 (es la clave de
    idempotencia del envío, ver app/stateless._send) — sin él, 400 claro."""
    raw = v2_raw()
    del raw["dispatchId"]
    body = json.dumps(raw).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_v2_organizationid_o_conversationid_faltante_400(ctx: AppContext, client):
    raw = v2_raw()
    del raw["organizationId"]
    body = json.dumps(raw).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_version_no_soportada_400(client):
    raw = v2_raw()
    raw["version"] = 99
    body = json.dumps(raw).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_v2_llm_provider_mal_escrito_400_no_manda_la_clave_al_proveedor_equivocado(client):
    """MINOR (revisión): un `provider` capitalizado/con typo ("OpenRouter",
    "AWS"...) con `str` suelto pasaba silencioso y `_build_org_llm` caía al
    `else` (base_url=None) — mandando la clave de OpenRouter del negocio a
    api.openai.com. Con `Literal["openrouter","openai"]`, ese payload ya ni
    llega a construir un cliente: 400 claro en la validación."""
    raw = v2_raw(
        llm={"provider": "OpenRouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-x"}
    )
    body = json.dumps(raw).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_v2_llm_provider_valido_en_minusculas_pasa(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_provider_ok")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
    )
    payload = v2_payload(
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-x"}
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"


# ==================================================================== Turn ===


async def test_v2_solo_lo_pendiente_se_responde(respx_mock):
    """Un lead viejo (pending=False) no debe colarse en el mensaje final —
    solo lo que el CRM marcó como pendiente de ESTA ráfaga."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    payload = v2_payload(
        history=[
            hist("lead", "pregunta vieja ya respondida", pending=False),
            hist("agent", "ya te respondí eso"),
            hist("lead", "pregunta nueva", id="m2", pending=True),
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert last_user_content(ctx.llm) == "pregunta nueva"


async def test_v2_noop_sin_pendientes(respx_mock):
    ctx = make_ctx()
    payload = v2_payload(history=[hist("lead", "algo viejo", pending=False)])
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "noop"
    assert ctx.llm.calls == []


async def test_v2_saludo_viene_de_agent_has_spoken_no_de_greeted(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    payload = v2_payload(agent_has_spoken=False)
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    system_prompt = ctx.llm.calls[-1]["messages"][0]["content"]
    assert "PRIMER contacto" in system_prompt

    ctx2 = make_ctx()
    payload2 = v2_payload(agent_has_spoken=True)
    await stateless.run_turn(ctx2, payload2, organization_id="org_a")
    system_prompt2 = ctx2.llm.calls[-1]["messages"][0]["content"]
    assert "PRIMER contacto" not in system_prompt2


async def test_v2_marca_mensajes_de_equipo_y_dueno(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    payload = v2_payload(
        history=[
            hist("lead", "hola, tengo una duda", pending=False),
            hist("team", "ya te contesto en un momento", id="t1"),
            hist("owner", "hola, soy el dueño, dime", id="o1"),
            hist("lead", "gracias, ya me quedó claro", id="m2", pending=True),
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    messages = ctx.llm.calls[-1]["messages"]
    contents = [m["content"] for m in messages]
    assert any("[Respuesta de una persona del negocio]: ya te contesto" in str(c) for c in contents)
    assert any("[Respuesta de una persona del negocio]: hola, soy el dueño" in str(c) for c in contents)
    assert any("Son contexto: no los repitas ni los contradigas" in str(c) for c in contents)


async def test_v2_sin_equipo_ni_dueno_no_agrega_la_nota(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    payload = v2_payload()  # solo un lead pendiente, sin team/owner
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    contents = [m["content"] for m in ctx.llm.calls[-1]["messages"]]
    assert not any("Son contexto: no los repitas" in str(c) for c in contents)


async def test_v2_respuesta_que_repite_el_marcador_se_limpia_antes_de_mandar(respx_mock):
    """MINOR (revisión): el modelo VE el marcador en su propio historial
    (como turno de "assistant") y a veces lo repite en su respuesta — nunca
    debe llegarle al lead tal cual."""
    ctx = make_ctx()
    ctx.llm.replies = [
        LlmReply(
            content=(
                "[Respuesta de una persona del negocio]: Claro, te ayudo con eso."
            )
        )
    ]
    routes = mock_crm_basics(
        respx_mock,
        conv_id="cv_v2_marker_leak",
    )
    payload = v2_payload(
        conversation_id="cv_v2_marker_leak",
        history=[
            hist("lead", "hola", pending=False),
            hist("team", "ya te contesto", id="t1"),
            hist("lead", "gracias", id="m2", pending=True),
        ],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    sent = json.loads(routes["messages"].calls[0].request.content)
    assert "[Respuesta de una persona del negocio]" not in sent["text"]
    assert sent["text"] == "Claro, te ayudo con eso."


async def test_v2_marcador_en_medio_del_texto_no_se_toca(respx_mock):
    """Solo se limpia al INICIO — en medio del texto sería parte legítima de
    la respuesta (p.ej. el lead preguntó por esa frase)."""
    texto = 'Dijiste "[Respuesta de una persona del negocio]: hola" y sí, es correcto.'
    assert stateless._strip_leaked_marker(texto) == texto


async def test_v2_hostilidad_cuenta_rafagas_no_mensajes(respx_mock):
    """Tres RÁFAGAS hostiles seguidas (separadas por respuestas del agente)
    disparan el backstop de handoff — igual que en v1."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_hostil")
    payload = v2_payload(
        history=[
            hist("lead", "eres un estafador", pending=False),
            hist("agent", "lamento que sientas eso"),
            hist("lead", "puros mentirosos, pura basura", id="m2", pending=False),
            hist("agent", "entiendo tu molestia"),
            hist("lead", "pinches bots chafas", id="m3", pending=True),
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert routes["handoff"].call_count == 1
    body = json.loads(routes["handoff"].calls[0].request.content)
    assert body["reason"] == "hostilidad"
    assert result.handoff_reason == "hostilidad"


async def test_v2_hostilidad_dentro_de_una_sola_rafaga_no_dispara_el_backstop(respx_mock):
    """Tres mensajes hostiles SIN respuesta del agente entre medio son UNA
    sola ráfaga — no tres — y no deben disparar el backstop determinista
    (el LLM de prueba no llama handoff por su cuenta)."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_hostil2")
    payload = v2_payload(
        history=[
            hist("lead", "eres un estafador", pending=True, id="m1"),
            hist("lead", "puros mentirosos, pura basura", id="m2", pending=True),
            hist("lead", "pinches bots chafas", id="m3", pending=True),
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert routes["handoff"].call_count == 0


async def test_v2_hostilidad_por_nota_de_voz_asentada_usa_la_transcripcion(respx_mock):
    """MINOR (revisión): una ráfaga hostil dicha por NOTA DE VOZ en historial
    YA resuelto (settled, no pendiente) no tiene `item.text` — sin usar
    `media.transcript` como respaldo, esas ráfagas nunca contaban para
    AC-18, dejando un hueco: la misma hostilidad "no cuenta" si se manda por
    audio en vez de texto."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_hostil_voz")
    payload = v2_payload(
        history=[
            hist(
                "lead", None, id="m1", type="audio", pending=False,
                media={"mediaId": "media-h1", "mime": "audio/ogg", "fileName": None,
                       "caption": None, "transcript": "eres un estafador", "location": None,
                       "contacts": None},
            ),
            hist("agent", "lamento que sientas eso", id="a1"),  # separa las ráfagas
            hist(
                "lead", None, id="m2", type="audio", pending=False,
                media={"mediaId": "media-h2", "mime": "audio/ogg", "fileName": None,
                       "caption": None, "transcript": "puros mentirosos, pura basura",
                       "location": None, "contacts": None},
            ),
            hist("agent", "entiendo tu molestia", id="a2"),
            hist("lead", "pinches bots chafas", id="m3", pending=True),
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert routes["handoff"].call_count == 1
    assert result.handoff_reason == "hostilidad"


async def test_v2_activacion_desde_el_payload(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_act", ai_enabled=False)
    payload = v2_payload(
        ai_enabled=False,
        activation_enabled=True,
        activation_messages=["Quiero agendar"],
        history=[hist("lead", "  QUIERO AGENDAR  ", pending=True)],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert routes["activate"].call_count == 1
    assert result.action == "replied"


async def test_v2_sin_activador_queda_en_silencio(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_noact", ai_enabled=False)
    payload = v2_payload(ai_enabled=False, activation_enabled=False)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert routes["activate"].call_count == 0
    assert ctx.llm.calls == []


async def test_v2_activate_fallido_se_propaga_para_que_el_crm_reintente(respx_mock):
    """MINOR (revisión): un fallo REAL de /api/bot/activate no debe
    degradarse a un 200 silencioso (nada comprometido todavía) — v1 en modo
    estricto ya hace `raise` aquí (app/turn.py); v2 ES ese modo estricto
    siempre. Degradar a silencio dejaría el activador respondido con "nada"
    para siempre, sin que el CRM pueda reintentarlo."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_act_fail", ai_enabled=False)
    respx_mock.post(f"{CRM_URL}/api/bot/activate").mock(return_value=httpx.Response(503))
    payload = v2_payload(
        ai_enabled=False,
        activation_enabled=True,
        activation_messages=["Quiero agendar"],
        history=[hist("lead", "Quiero agendar", pending=True)],
    )
    with pytest.raises(Exception):
        await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert ctx.llm.calls == []  # nunca llegó a conversar


async def test_v2_allowlist_del_payload_bloquea_fuera_de_lista(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_allow")
    payload = v2_payload(allowlist_enabled=True, allowed_wa_ids=["5219999999999"])
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert ctx.llm.calls == []


async def test_v2_allowlist_del_payload_deja_pasar_dentro_de_lista(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_allow2")
    payload = v2_payload(allowlist_enabled=True, allowed_wa_ids=[IDENTITY])
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"


async def test_v2_istest_ignora_la_allowlist(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_lab")
    payload = v2_payload(
        is_test=True,
        allowlist_enabled=True,
        allowed_wa_ids=["5219999999999"],  # no incluye a nadie de esta prueba
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"


async def test_v2_ventana_cerrada_silencio(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_win", window_open=False)
    payload = v2_payload(window_open=False)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"


async def test_v2_reset_llama_al_endpoint_v2_con_notice_y_dispatchid(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_reset")
    reset_route = respx_mock.post(f"{CRM_URL}/api/bot/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    payload = v2_payload(
        allowlist_enabled=True,
        allowed_wa_ids=[IDENTITY],
        dispatch_id="dsp_reset_1",
        history=[hist("lead", "/reset", pending=True)],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "reset"
    assert reset_route.call_count == 1
    sent = json.loads(reset_route.calls[0].request.content)
    assert sent["conversationId"] == "cv_v2_1"
    assert sent["dispatchId"] == "dsp_reset_1"
    assert "memoria reiniciada" in sent["notice"]


async def test_v2_reset_sin_allowlist_no_corre(respx_mock):
    """Igual que v1: sin allowlist activa (o fuera de ella) /reset es un
    mensaje de texto cualquiera, no un comando."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_reset2")
    reset_route = respx_mock.post(f"{CRM_URL}/api/bot/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    payload = v2_payload(history=[hist("lead", "/reset", pending=True)])
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action != "reset"
    assert reset_route.call_count == 0
    assert routes["messages"].call_count == 1  # respondió como a cualquier mensaje


async def test_v2_ad_origen_alimenta_el_prompt(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_ad")
    payload = v2_payload(ad_headline="Agenda tu cita hoy")
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    system_prompt = ctx.llm.calls[-1]["messages"][0]["content"]
    assert "Agenda tu cita hoy" in system_prompt


# =================================================================== Media ===


async def test_v2_transcribe_audio_pendiente_y_escribe_la_transcripcion(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_audio")
    transcript_route = respx_mock.post(
        f"{CRM_URL}/api/bot/messages/m-audio-1/transcript"
    ).mock(return_value=httpx.Response(200, json={"ok": True}))
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-audio-1",
                type="audio",
                pending=True,
                media={"mediaId": "media-1", "mime": "audio/ogg", "fileName": None, "caption": None, "transcript": None, "location": None, "contacts": None},
            )
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert len(ctx.llm.transcriptions) == 1
    assert transcript_route.call_count == 1
    sent = json.loads(transcript_route.calls[0].request.content)
    assert sent["text"] == ctx.llm.transcript_text
    assert ctx.llm.transcript_text in str(last_user_content(ctx.llm))


async def test_v2_no_repite_transcripcion_si_ya_hay_una(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_audio2")
    transcript_route = respx_mock.post(
        f"{CRM_URL}/api/bot/messages/m-audio-2/transcript"
    ).mock(return_value=httpx.Response(200, json={"ok": True}))
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-audio-2",
                type="audio",
                pending=True,
                media={
                    "mediaId": "media-2",
                    "mime": "audio/ogg",
                    "fileName": None,
                    "caption": None,
                    "transcript": "ya vengo transcrito de una corrida anterior",
                    "location": None,
                    "contacts": None,
                },
            )
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert len(ctx.llm.transcriptions) == 0  # NUNCA se volvió a llamar
    assert transcript_route.call_count == 0  # tampoco se reescribió
    assert "ya vengo transcrito de una corrida anterior" in str(last_user_content(ctx.llm))


async def test_v2_falla_al_escribir_transcripcion_se_tolera(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_audio3")
    respx_mock.post(f"{CRM_URL}/api/bot/messages/m-audio-3/transcript").mock(
        return_value=httpx.Response(500)
    )
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-audio-3",
                type="audio",
                pending=True,
                media={"mediaId": "media-3", "mime": "audio/ogg", "fileName": None, "caption": None, "transcript": None, "location": None, "contacts": None},
            )
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"  # el turno sigue pese al 500 en el write-back


async def test_v2_ubicacion_sin_coordenadas_nunca_dice_none(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_loc")
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-loc-1",
                type="location",
                pending=True,
                media={"mediaId": None, "mime": None, "fileName": None, "caption": None, "transcript": None, "location": {}, "contacts": None},
            )
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    texto = str(last_user_content(ctx.llm))
    assert "None" not in texto


async def test_v2_ubicacion_con_coordenadas_se_renderiza(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_loc2")
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-loc-2",
                type="location",
                pending=True,
                media={
                    "mediaId": None, "mime": None, "fileName": None, "caption": None,
                    "transcript": None,
                    "location": {"latitude": 20.69, "longitude": -101.36},
                    "contacts": None,
                },
            )
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    texto = str(last_user_content(ctx.llm))
    assert "20.69" in texto and "-101.36" in texto


async def test_v2_contactos_e_imagen_con_caption_se_renderizan(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_media_mix")
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-img-1",
                type="image",
                pending=True,
                media={
                    "mediaId": "media-img-1", "mime": "image/jpeg", "fileName": None,
                    "caption": "mira mi local", "transcript": None, "location": None,
                    "contacts": None,
                },
            ),
            hist(
                "lead",
                None,
                id="m-contacts-1",
                type="contacts",
                pending=True,
                media={
                    "mediaId": None, "mime": None, "fileName": None, "caption": None,
                    "transcript": None, "location": None,
                    # Shape REAL de Meta (crudo, sin normalizar por el CRM —
                    # ver src/server/inbox/ingest.ts: payload = msg.contacts):
                    # objetos, no strings. Un `list[str]` aquí rechazaba con
                    # 400 CUALQUIER despacho con una tarjeta de contacto.
                    "contacts": [
                        {
                            "name": {"formatted_name": "Juan Pérez", "first_name": "Juan"},
                            "phones": [{"phone": "+521234567890", "type": "CELL"}],
                        }
                    ],
                },
            ),
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    texto = str(last_user_content(ctx.llm))
    assert "mira mi local" in texto
    assert "Juan Pérez" in texto
    assert "+521234567890" in texto


async def test_v2_contacto_con_nombre_string_y_sin_phones_tambien_se_renderiza(respx_mock):
    """El shape real también permite `name` como string suelto (no objeto) y
    faltar `phones` del todo — ver ContactPayload en el propio CRM
    (message-thread.tsx). Nunca debe reventar ni caer en "None"."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_contact_str")
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-contacts-2",
                type="contacts",
                pending=True,
                media={
                    "mediaId": None, "mime": None, "fileName": None, "caption": None,
                    "transcript": None, "location": None,
                    "contacts": [{"name": "Ana Ruiz"}],
                },
            ),
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    texto = str(last_user_content(ctx.llm))
    assert "Ana Ruiz" in texto
    assert "None" not in texto


async def test_v2_contactos_en_historial_asentado_no_revienta_la_validacion(respx_mock):
    """El bug real reportado: una tarjeta de contacto en historial YA
    resuelto (ni siquiera pendiente) con `list[str]` tumbaba con 400 TODO el
    despacho, cada vez que apareciera en los últimos 20 mensajes."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_contact_settled")
    payload = v2_payload(
        history=[
            hist(
                "lead",
                None,
                id="m-contacts-old",
                type="contacts",
                pending=False,
                media={
                    "mediaId": None, "mime": None, "fileName": None, "caption": None,
                    "transcript": None, "location": None,
                    "contacts": [{"name": {"formatted_name": "Viejo Contacto"}}],
                },
            ),
            hist("lead", "hola de nuevo", id="m-new", pending=True),
        ]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"


# ==================================================================== Send ===


async def test_v2_send_manda_dispatchid_y_seq(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_send1")
    payload = v2_payload(dispatch_id="dsp_seq_1")
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    sent = json.loads(routes["messages"].calls[0].request.content)
    assert sent["dispatchId"] == "dsp_seq_1"
    assert sent["seq"] == 0


async def test_v2_send_reintenta_5xx_y_termina_bien(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_send2")
    messages_route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        side_effect=[httpx.Response(502), httpx.Response(200, json={"messageId": "m1"})]
    )
    payload = v2_payload()
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert messages_route.call_count == 2


async def test_v2_send_reintenta_409_send_in_progress(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_send3")
    messages_route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        side_effect=[
            httpx.Response(409, json={"code": "send_in_progress"}),
            httpx.Response(200, json={"messageId": "m1"}),
        ]
    )
    payload = v2_payload()
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert messages_route.call_count == 2


async def test_v2_send_agota_reintentos_y_propaga(respx_mock):
    """Sin 2xx nunca: run_turn debe DEJAR ESCAPAR la excepción (dispatch.py la
    vuelve 5xx para que el CRM reintente el despacho completo)."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_send4")
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(return_value=httpx.Response(502))
    payload = v2_payload()
    with pytest.raises(Exception):
        await stateless.run_turn(ctx, payload, organization_id="org_a")


async def test_v2_send_agota_reintentos_da_5xx_por_http(ctx: AppContext, client, respx_mock):
    mock_crm_basics(respx_mock, conv_id="cv_v2_send5")
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(return_value=httpx.Response(502))
    body = v2_body(conversation_id="cv_v2_send5")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 500


async def test_v2_ai_paused_queda_en_silencio(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_send6")
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(409, json={"code": "ai_paused"})
    )
    payload = v2_payload()
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"


async def test_v2_402_es_final_queda_en_silencio(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_send7")
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(402, json={"code": "payment_required"})
    )
    payload = v2_payload()
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"


async def test_v2_commit_se_marca_solo_tras_2xx():
    """Unidad directa de _send: el commit NO se marca mientras no llegue un
    2xx, ni siquiera durante los reintentos."""
    crm = CrmClient(CRM_URL, "k")
    commit = TurnCommit()
    calls = {"n": 0}

    async def fake_request(method, url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"messageId": "m1"})

    crm._request = fake_request  # type: ignore[method-assign]
    ok = await stateless._send(crm, "cv_1", "hola", dispatch_id="dsp_1", commit=commit)
    assert ok is True
    assert commit.done is True
    assert calls["n"] == 2


async def test_v2_commit_nunca_se_marca_si_nunca_hay_2xx():
    crm = CrmClient(CRM_URL, "k")
    commit = TurnCommit()

    async def fake_request(method, url, **kwargs):
        return httpx.Response(502)

    crm._request = fake_request  # type: ignore[method-assign]
    with pytest.raises(Exception):
        await stateless._send(crm, "cv_1", "hola", dispatch_id="dsp_1", commit=commit)
    assert commit.done is False


async def test_v2_booking_201_mas_envio_agotado_no_marca_comprometido(respx_mock):
    """CRÍTICO (revisión): book_session marca commit ANTES de reservar en
    v1 (necesario ahí: el envío no es idempotente) — en v2 eso perdía la
    confirmación para siempre: create_booking 201, /api/bot/messages 502×3,
    y como el commit YA estaba marcado por la reserva, dispatch.py
    respondía 200 (nada que reintentar) con el lead agendado y SIN avisarle
    nada. Con `commit=None` en el tool runner, solo `_send` compromete — así
    que sin un 2xx del envío, `run_turn` debe DEJAR ESCAPAR la excepción."""
    ctx = make_ctx()
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="book_session", arguments={"start_utc": SLOT_ISO})],
        ),
        LlmReply(content="Listo, tu cita quedó agendada."),
    ]
    mock_crm_basics(respx_mock, conv_id="cv_v2_booking_lost")
    booking_route = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "lunes 10am"})
    )
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": True})
    )
    messages_route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(502)
    )
    payload = v2_payload(
        conversation_id="cv_v2_booking_lost",
        offers=[{"startUtc": SLOT_ISO, "label": "lunes 10am"}],
    )
    with pytest.raises(Exception):
        await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert booking_route.call_count == 1  # la reserva SÍ se hizo
    assert messages_route.call_count == 3  # los 3 reintentos de envío, todos fallidos
    # (dispatch.py, que envuelve run_turn, vería esta excepción con
    # commit.done aún False y respondería 500 — el CRM reintenta el
    # despacho, ver test_v2_send_agota_reintentos_da_5xx_por_http arriba)


async def test_v2_booking_confirmado_en_reintento_no_reserva_dos_veces(respx_mock):
    """La otra mitad del escenario: en el REINTENTO del CRM, `context.
    booking.next` ya refleja la cita creada en el intento anterior. Aunque
    el catálogo de este intento sea nuevo (sin ese horario ofrecido) y el
    LLM insista en llamar book_session sobre el MISMO horario, debe
    confirmar sin volver a tocar /api/bot/bookings."""
    ctx = make_ctx()
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="book_session", arguments={"start_utc": SLOT_ISO})],
        ),
        LlmReply(content="Listo, tu cita quedó agendada para el lunes a las 10am."),
    ]
    mock_crm_basics(respx_mock, conv_id="cv_v2_booking_retry")
    bookings_route = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "lunes 10am"})
    )
    payload = v2_payload(
        conversation_id="cv_v2_booking_retry",
        offers=[],  # el CRM ya no re-ofrece un horario que acaba de ocupar
    )
    payload.context["booking"] = {
        "next": {
            "id": "bk_1",
            "scheduledAtUtc": SLOT_ISO,
            "label": "lunes 10am",
            "meetingLink": "https://zoom.us/j/1",
        }
    }
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert bookings_route.call_count == 0  # NUNCA se reservó de nuevo


# ----------------------------------------------------------- org LLM ---


async def test_v2_llm_del_negocio_401_cae_a_la_plataforma_y_lo_reporta(respx_mock):
    """Contrato: `llm.source` es de QUIÉN ES LA CLAVE reportada, no quién
    terminó contestando — el CRM solo marca una clave inválida cuando
    `source == "org"` (server/ai/credentials.ts:markAiCredentialInvalidIfUnchanged,
    llamado desde el paso 6 SOLO si `source === "org"`). Reportar "platform"
    tras el fallback dejaría la clave rota del negocio viva para siempre."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_llm1")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "code": None}})
    )
    payload = v2_payload(
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-negocio-invalida"}
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"  # la plataforma (FakeLLM) sí respondió
    assert result.llm_source == "org"  # la clave que FALLÓ — para que se marque inválida
    assert result.llm_status == "auth_failed"
    assert result.llm_answered_with == "platform"  # informativo: quién contestó de verdad
    assert len(ctx.llm.calls) == 1  # la plataforma SÍ terminó respondiendo


async def test_v2_llm_del_negocio_ok_source_es_org(respx_mock):
    """Contraprueba: si la clave del negocio responde bien (sin fallback),
    `source` también es "org" — es la que está en juego, tuvo o no problema."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_llm_ok")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "todo bien"}}]}
        )
    )
    payload = v2_payload(
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-negocio-buena"}
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.llm_source == "org"
    assert result.llm_status == "ok"
    assert result.llm_answered_with == "org"


async def test_v2_sin_llm_del_negocio_usa_la_plataforma_directo(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_llm2")
    payload = v2_payload(llm=None)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.llm_source == "platform"
    assert result.llm_status == "ok"
    assert result.llm_answered_with == "platform"


async def test_v2_ambas_claves_fallan_da_resultado_manejado_no_500(respx_mock):
    """MINOR: org y plataforma fallando las dos NO debe escapar como una
    excepción sin clasificar que dispatch.py convertiría en 500 (una
    tormenta de reintentos del CRM contra las MISMAS dos claves rotas) — se
    trata como agotado: silencio + handoff `error`, y el `llm` reportado."""
    ctx = make_ctx()
    ctx.llm.raise_exc = LlmAuthFailed("la plataforma también quedó sin crédito")
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_llm_ambas")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "code": None}})
    )
    payload = v2_payload(
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-negocio-invalida"}
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert result.llm_source == "org"
    assert result.llm_status == "auth_failed"
    assert result.handoff_reason == "error"
    assert routes["handoff"].call_count == 1


async def test_v2_sin_llm_del_negocio_la_plataforma_falla_igual_reporta_status(respx_mock):
    """Sin clave de negocio en juego (source="platform" desde el arranque):
    si la ÚNICA clave (la de plataforma) falla por credenciales, el estado
    debe reflejarlo — no "ok", que sería engañoso en los logs — aunque el
    CRM no actúe sobre esto (solo invalida con source=="org")."""
    ctx = make_ctx()
    ctx.llm.raise_exc = LlmNoCredits("402 de la clave de plataforma")
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_llm_plat_falla")
    payload = v2_payload(llm=None)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert result.llm_source == "platform"
    assert result.llm_status == "no_credits"
    assert result.handoff_reason == "error"
    assert routes["handoff"].call_count == 1


async def test_v2_cliente_del_negocio_se_cierra_al_terminar_el_turno(respx_mock, monkeypatch):
    """El cliente por turno (clave del negocio) se cierra SIEMPRE al acabar —
    éxito o fallo — para no fugar conexiones HTTP."""
    from app.llm import OpenAiLlm

    cerrados = {"n": 0}
    original_aclose = OpenAiLlm.aclose

    async def aclose_contado(self):
        cerrados["n"] += 1
        await original_aclose(self)

    monkeypatch.setattr(OpenAiLlm, "aclose", aclose_contado)

    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_llm3")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "listo"}}]}
        )
    )
    payload = v2_payload(
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-del-negocio"}
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert cerrados["n"] == 1
    # la plataforma (FakeLLM, sin aclose real de OpenAiLlm) sigue viva —
    # solo se cerró el cliente DE ESTE TURNO, nunca ctx.llm.
    assert isinstance(ctx.llm, FakeLLM)


# ============================================== forma real del payload CRM ===
# Fixture que espeja EXACTAMENTE lo que arma
# vocero-crm-dispatch-v2/src/server/ai/nea-payload.ts (`toHistoryItem`,
# `roleFor`): direction "in" -> "lead"; "out"+origin "ai" -> "agent"; origin
# "manual" -> "owner"; origin "operator"/"template" -> "team". `media.location`
# y `media.contacts` son el `payload` CRUDO de Meta (`mediaAsset.payload`,
# ver src/server/inbox/ingest.ts), nunca normalizado.


async def test_v2_payload_con_la_forma_real_del_crm_no_revienta_y_responde(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_real_shape")
    transcript_route = respx_mock.post(
        f"{CRM_URL}/api/bot/messages/msg_audio_settled/transcript"
    ).mock(return_value=httpx.Response(200, json={"ok": True}))
    history = [
        hist("lead", "hola, quiero información", id="msg_1", at="2026-09-25T11:00:00.000Z"),
        hist("agent", "¡Hola! Claro, cuéntame más", id="msg_2", at="2026-09-25T11:00:05.000Z"),
        hist(  # origin "operator"/"template" -> role "team"
            "team", "Un asesor te va a escribir en breve", id="msg_3",
            at="2026-09-25T11:01:00.000Z",
        ),
        hist(  # origin "manual" -> role "owner"
            "owner", "Hola, soy Juan, el dueño — dime en qué te ayudo", id="msg_4",
            at="2026-09-25T11:02:00.000Z",
        ),
        hist(  # nota de voz YA resuelta, con transcripción ya escrita por el CRM
            "lead", None, id="msg_5", type="audio", at="2026-09-25T11:03:00.000Z",
            media={
                "mediaId": "wamid.media1", "mime": "audio/ogg", "fileName": None,
                "caption": None, "transcript": "quiero saber precios",
                "location": None, "contacts": None,
            },
        ),
        hist(  # pendiente: imagen con caption
            "lead", None, id="msg_6", type="image", pending=True, at="2026-09-25T11:05:00.000Z",
            media={
                "mediaId": "wamid.media2", "mime": "image/jpeg", "fileName": None,
                "caption": "aquí está mi negocio", "transcript": None,
                "location": None, "contacts": None,
            },
        ),
        hist(  # pendiente: tarjeta de contacto, shape crudo de Meta
            "lead", None, id="msg_7", type="contacts", pending=True, at="2026-09-25T11:05:30.000Z",
            media={
                "mediaId": None, "mime": None, "fileName": None, "caption": None,
                "transcript": None, "location": None,
                "contacts": [
                    {
                        "name": {"formatted_name": "María López", "first_name": "María"},
                        "phones": [{"phone": "+528112345678", "type": "CELL"}],
                    }
                ],
            },
        ),
        hist(  # pendiente: nota de voz SIN transcripción todavía
            "lead", None, id="msg_audio_settled", type="audio", pending=True,
            at="2026-09-25T11:06:00.000Z",
            media={
                "mediaId": "wamid.media3", "mime": "audio/ogg", "fileName": None,
                "caption": None, "transcript": None, "location": None, "contacts": None,
            },
        ),
        hist("lead", "eso es todo, gracias", id="msg_8", pending=True, at="2026-09-25T11:06:30.000Z"),
    ]
    payload = v2_payload(conversation_id="cv_v2_real_shape", history=history)

    result = await stateless.run_turn(ctx, payload, organization_id="org_a")

    assert result.action == "replied"
    assert routes["messages"].call_count == 1
    assert transcript_route.call_count == 1  # la nota de voz pendiente SÍ se transcribió

    messages = ctx.llm.calls[-1]["messages"]
    contents = [str(m["content"]) for m in messages]
    assert any("[Respuesta de una persona del negocio]: Un asesor" in c for c in contents)
    assert any("[Respuesta de una persona del negocio]: Hola, soy Juan" in c for c in contents)
    assert any("quiero saber precios" in c for c in contents)  # audio settled, vía transcript
    final_text = str(last_user_content(ctx.llm))
    assert "aquí está mi negocio" in final_text
    assert "María López" in final_text and "+528112345678" in final_text
    assert "eso es todo, gracias" in final_text
    assert "None" not in final_text
