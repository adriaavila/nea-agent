"""Despacho v2 con `followup: true` (app/stateless._run_followup): el ÚNICO
empujón de las 4 h de silencio, ahora disparado por el CRM (dueño del
temporizador) en vez del `FollowupWorker` de v1 (app/followup.py, que vive de
`ctx.store` y por eso dejó de correr para negocios en modo despacho).

Reusa los helpers de tests/test_stateless.py (`v2_raw`/`v2_payload`/`hist`/
`sign`) — mismo contrato v2, un campo más.
"""
from __future__ import annotations

import json

import httpx

from app import stateless
from app.llm import LlmExhausted
from app.prompt import FOLLOWUP_INSTRUCTION
from tests.conftest import make_ctx, mock_crm_basics
from tests.test_stateless import hist, sign, v2_payload, v2_raw


async def test_followup_camino_feliz_un_solo_envio_y_llm_sin_tools(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    payload = v2_payload(
        followup=True,
        dispatch_id="ajfu_cv_v2_1",
        history=[
            hist("lead", "hola, quiero información", id="m1"),
            hist("agent", "¡Claro! ¿Qué necesitas?", id="a1"),
        ],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert routes["messages"].call_count == 1
    sent = json.loads(routes["messages"].calls[0].request.content)
    assert sent["dispatchId"] == "ajfu_cv_v2_1"
    assert sent["seq"] == 0
    assert ctx.llm.calls[-1]["tools"] is None
    assert ctx.llm.calls[-1]["messages"][-1] == {
        "role": "system",
        "content": FOLLOWUP_INSTRUCTION,
    }


async def test_followup_chat_pausado_silencio_incluso_con_activacion(respx_mock):
    """Un empujón NUNCA reactiva un chat pausado — ni siquiera con frases
    activadoras encendidas en el perfil: no es el lead quien está "hablando"."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_pausado", ai_enabled=False)
    payload = v2_payload(
        followup=True,
        conversation_id="cv_v2_pausado",
        ai_enabled=False,
        activation_enabled=True,
        activation_messages=["Quiero agendar"],
        history=[hist("agent", "¿seguimos?", id="a1")],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert ctx.llm.calls == []
    assert routes["messages"].call_count == 0
    assert routes["activate"].call_count == 0


async def test_followup_ventana_cerrada_silencio(respx_mock):
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_ventana", window_open=False)
    payload = v2_payload(
        followup=True,
        conversation_id="cv_v2_ventana",
        window_open=False,
        history=[hist("agent", "¿seguimos?", id="a1")],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert ctx.llm.calls == []
    assert routes["messages"].call_count == 0


async def test_followup_lead_hablo_ultimo_noop_sin_llm(respx_mock):
    """El lead ya contestó (o alguien más) — eso lo atiende el turno normal,
    no un empujón repetido."""
    ctx = make_ctx()
    payload = v2_payload(
        followup=True,
        history=[
            hist("agent", "¿seguimos?", id="a1"),
            hist("lead", "hola de nuevo", id="m2"),
        ],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "noop"
    assert ctx.llm.calls == []


async def test_followup_con_pendiente_sin_resolver_noop_sin_llm(respx_mock):
    """Un `pending: true` en el historial (aunque no sea el último item)
    significa que hay algo sin resolver — eso lo atiende el turno normal."""
    ctx = make_ctx()
    payload = v2_payload(
        followup=True,
        history=[
            hist("lead", "hola", id="m1", pending=True),
            hist("agent", "¿seguimos?", id="a1", pending=False),
        ],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "noop"
    assert ctx.llm.calls == []


async def test_followup_historial_vacio_noop_sin_llm(respx_mock):
    """Sin agente hablando último (no hay historial siquiera): nada que
    empujar."""
    ctx = make_ctx()
    payload = v2_payload(followup=True, history=[])
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "noop"
    assert ctx.llm.calls == []


async def test_followup_istest_noop(respx_mock):
    """Jamás se empuja una conversación del Laboratorio."""
    ctx = make_ctx()
    payload = v2_payload(
        followup=True,
        is_test=True,
        history=[hist("agent", "¿seguimos?", id="a1")],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "noop"
    assert ctx.llm.calls == []


async def test_followup_allowlist_bloquea_fuera_de_lista(respx_mock):
    ctx = make_ctx()
    payload = v2_payload(
        followup=True,
        allowlist_enabled=True,
        allowed_wa_ids=["5219999999999"],
        history=[hist("agent", "¿seguimos?", id="a1")],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert ctx.llm.calls == []


async def test_followup_llm_agotado_silencio_sin_handoff(respx_mock):
    """A diferencia de un turno normal agotado: SIN handoff — un empujón
    fallido no amerita pausar la conversación."""
    ctx = make_ctx()
    ctx.llm.raise_exc = LlmExhausted("boom")
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_followup_agotado")
    payload = v2_payload(
        followup=True,
        conversation_id="cv_v2_followup_agotado",
        history=[hist("agent", "¿seguimos?", id="a1")],
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "silent"
    assert routes["handoff"].call_count == 0
    assert routes["messages"].call_count == 0


async def test_followup_llm_del_negocio_401_cae_a_la_plataforma_y_lo_reporta(
    respx_mock,
):
    """Mismo mecanismo de fallback que un turno normal (mismo `_tool_loop`):
    `llm.source` sigue siendo la clave EN JUEGO, no quién contestó."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_followup_fallback")
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "code": None}})
    )
    payload = v2_payload(
        followup=True,
        conversation_id="cv_v2_followup_fallback",
        history=[hist("agent", "¿seguimos?", id="a1")],
        llm={"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "apiKey": "sk-negocio-invalida"},
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert result.llm_source == "org"
    assert result.llm_status == "auth_failed"
    assert result.llm_answered_with == "platform"
    assert len(ctx.llm.calls) == 1  # la plataforma SÍ terminó respondiendo
    assert routes["messages"].call_count == 1


async def test_followup_con_valor_string_da_400(client):
    """Mismo patrón de validación estricta que `isTest`: un string no se
    coacciona a boolean, 400 claro."""
    raw = v2_raw(history=[hist("agent", "¿seguimos?", id="a1")])
    raw["followup"] = "true"
    body = json.dumps(raw).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_followup_ausente_en_payload_viejo_no_cambia_el_turno_normal(respx_mock):
    """Un CRM que todavía no manda `followup` (payload de antes de este
    cambio): pydantic lo defaultea a False y el turno normal corre exacto
    igual que siempre."""
    ctx = make_ctx()
    routes = mock_crm_basics(respx_mock, conv_id="cv_v2_1")
    raw = v2_raw()
    del raw["followup"]
    payload = stateless.DispatchPayloadV2.model_validate(raw)
    assert payload.followup is False
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert routes["messages"].call_count == 1
