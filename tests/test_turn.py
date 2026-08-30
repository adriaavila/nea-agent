"""Turno completo: handoff después de la despedida; LLM agotado → silencio + handoff error."""
from __future__ import annotations

import asyncio
import json

from app.llm import LlmExhausted, LlmReply, ToolCall
from app.profile import BusinessProfile
from app.sender import SenderWorker
from tests.conftest import CRM_CONV_ID, IDENTITY, mock_crm_basics, wa_body


class ActivationProfile:
    async def get(self):
        return BusinessProfile(
            activation_enabled=True,
            activation_messages=("Quiero agendar",),
        )


async def test_mensaje_configurado_activa_chat_y_conversa(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock, ai_enabled=False)
    ctx.profile = ActivationProfile()

    await client.post("/webhook", content=wa_body(text="  QUIERO AGENDAR "))
    await asyncio.sleep(0.2)
    assert routes["activate"].call_count == 1
    assert routes["messages"].call_count == 1
    assert len(ctx.llm.calls) == 1

    await client.post("/webhook", content=wa_body(text="otra pregunta", wamid="wamid.2"))
    await asyncio.sleep(0.2)
    assert routes["activate"].call_count == 1
    assert routes["messages"].call_count == 1
    assert len(ctx.llm.calls) == 1


async def test_handoff_despedida_primero_pausa_despues(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="handoff", arguments={"reason": "pidió humano"})],
        ),
        LlmReply(content="Va — te paso con el equipo ahora mismo, sin que repitas nada."),
    ]
    await client.post("/webhook", content=wa_body(text="quiero hablar con una persona"))
    await asyncio.sleep(0.25)

    assert routes["messages"].call_count == 1
    assert routes["handoff"].call_count == 1
    # ORDEN CRÍTICO: primero la despedida, luego el handoff (si no, 409 ai_paused)
    llamadas = [
        str(c.request.url.path) for c in respx_mock.calls if "/api/bot/" in str(c.request.url)
    ]
    assert llamadas.index("/api/bot/messages") < llamadas.index("/api/bot/handoff")
    body = json.loads(routes["handoff"].calls[0].request.content)
    # El texto libre del LLM se normaliza al catálogo del CRM (002): un
    # reason fuera de catálogo era 422 y el handoff se perdía en producción.
    assert body["reason"] == "cliente"


async def test_llm_agotado_silencio_mas_handoff_error(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    ctx.llm.raise_exc = LlmExhausted("proveedor caído")

    resp = await client.post("/webhook", content=wa_body(text="hola"))
    assert resp.status_code == 200  # el webhook JAMÁS falla por el LLM
    await asyncio.sleep(0.25)

    assert routes["messages"].call_count == 0  # nada roto al lead
    assert routes["handoff"].call_count == 1
    body = json.loads(routes["handoff"].calls[0].request.content)
    assert body["reason"] == "error"
    # el relay a la bandeja quedó intacto
    assert len(ctx.store.relays) == 1


async def test_turno_programa_seguimiento(ctx, client, respx_mock):
    mock_crm_basics(respx_mock)
    await client.post("/webhook", content=wa_body(text="hola"))
    await asyncio.sleep(0.2)
    conv = next(iter(ctx.store.conversations.values()))
    assert conv.greeted is True
    assert conv.followup_due_at is not None  # empujón agendado a FOLLOWUP_HOURS


async def test_mensaje_nuevo_descarta_respuesta_pendiente_vieja(
    ctx, client, respx_mock
):
    routes = mock_crm_basics(respx_mock)
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    pending_id = await ctx.store.enqueue_pending_send(
        conv.id, CRM_CONV_ID, "respuesta vieja"
    )

    await client.post(
        "/webhook", content=wa_body(text="pregunta nueva", wamid="wamid.new")
    )
    await asyncio.sleep(0.2)
    await SenderWorker(ctx).tick()

    assert ctx.store.pending_sends[pending_id].abandoned_at is not None
    assert routes["messages"].call_count == 1


async def test_turno_con_route_out_cierra_sin_seguimiento(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    ctx.llm.replies = [
        LlmReply(content=None, tool_calls=[ToolCall(id="t1", name="route_out", arguments={})]),
        LlmReply(content="Por ahora no somos el mejor fit — te dejo unos recursos para arrancar por tu cuenta."),
    ]
    await client.post("/webhook", content=wa_body(text="soy estudiante"))
    await asyncio.sleep(0.25)
    assert routes["messages"].call_count == 1
    ficha = json.loads(routes["ficha"].calls[0].request.content)
    assert ficha["ficha"]["resultado"] == "dio_diy"
    conv = next(iter(ctx.store.conversations.values()))
    assert conv.phase == "cerrada"
    assert conv.followup_due_at is None
