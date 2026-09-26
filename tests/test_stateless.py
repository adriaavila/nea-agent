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
from app.state import AppContext, NullStore, TurnCommit
from tests.conftest import CRM_URL, IDENTITY, FakeLLM, make_ctx, mock_crm_basics

SECRET = "test-key"  # CRM_BOT_API_KEY por defecto en tests.conftest.make_settings


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
    del raw["conversationId"]  # fuerza un ValidationError (obligatorio)
    body = json.dumps(raw).encode("utf-8")
    with caplog.at_level(logging.WARNING):
        resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400
    assert secreto not in caplog.text


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
                    "transcript": None, "location": None, "contacts": ["Juan Pérez"],
                },
            ),
        ]
    )
    await stateless.run_turn(ctx, payload, organization_id="org_a")
    texto = str(last_user_content(ctx.llm))
    assert "mira mi local" in texto
    assert "Juan Pérez" in texto


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


# ----------------------------------------------------------- org LLM ---


async def test_v2_llm_del_negocio_401_cae_a_la_plataforma_y_lo_reporta(respx_mock):
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
    assert result.llm_source == "platform"
    assert result.llm_status == "auth_failed"
    assert len(ctx.llm.calls) == 1  # la plataforma SÍ terminó respondiendo


async def test_v2_sin_llm_del_negocio_usa_la_plataforma_directo(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_v2_llm2")
    payload = v2_payload(llm=None)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.llm_source == "platform"
    assert result.llm_status == "ok"


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
