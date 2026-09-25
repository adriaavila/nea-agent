"""Envíos que agotan reintentos NO se descartan: se encolan y salen después.

Cubre el fix del incidente 2026-08-03 (respuesta generada perdida cuando Meta
hipa): el turno encola en pending_send y el SenderWorker entrega con backoff.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx

from app import turn
from app.sender import SenderWorker
from app.state import utcnow
from tests.conftest import CRM_CONV_ID, CRM_URL, IDENTITY, make_ctx


async def _sin_esperas(monkeypatch):
    """Anula los sleeps del backoff del turno para no alentar la suite."""

    async def instantaneo(_seconds: float) -> None:
        return None

    monkeypatch.setattr(turn.asyncio, "sleep", instantaneo)


async def test_envio_agotado_se_encola_no_se_descarta(respx_mock, monkeypatch):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(502, json={"code": "meta_unavailable"})
    )
    await _sin_esperas(monkeypatch)

    sent = await turn._send(ctx, conv.id, CRM_CONV_ID, "respuesta importante")

    assert sent is False
    assert route.call_count == turn.SEND_ATTEMPTS  # paciencia: 4 intentos
    pendientes = await ctx.store.due_pending_sends(utcnow())
    assert len(pendientes) == 1
    assert pendientes[0].content == "respuesta importante"
    assert pendientes[0].crm_conversation_id == CRM_CONV_ID
    await ctx.crm.aclose()


async def test_409_no_se_encola(respx_mock):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(409, json={"code": "ai_paused"})
    )

    sent = await turn._send(ctx, conv.id, CRM_CONV_ID, "hola")

    assert sent is False
    assert await ctx.store.due_pending_sends(utcnow()) == []  # silencio, sin cola
    await ctx.crm.aclose()


async def test_sender_entrega_pendiente_y_lo_recuerda(respx_mock):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    pid = await ctx.store.enqueue_pending_send(conv.id, CRM_CONV_ID, "hola tarde")
    route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "m1"})
    )
    worker = SenderWorker(ctx)

    await worker.tick()

    assert route.call_count == 1
    assert ctx.store.pending_sends[pid].delivered_at is not None
    history = await ctx.store.recent_messages(conv.id, 10)
    assert [(m.role, m.content) for m in history] == [("assistant", "hola tarde")]
    # segundo barrido: ya no hay nada que enviar
    await worker.tick()
    assert route.call_count == 1
    await ctx.crm.aclose()


async def test_sender_reintenta_con_backoff(respx_mock):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    pid = await ctx.store.enqueue_pending_send(conv.id, CRM_CONV_ID, "texto")
    route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(502, json={"code": "meta_unavailable"})
    )
    worker = SenderWorker(ctx)

    await worker.tick()

    item = ctx.store.pending_sends[pid]
    assert route.call_count == 1
    assert item.attempts == 1
    assert item.next_retry_at > utcnow()  # reprogramado, no martillado
    # aún no toca reintentar: el barrido siguiente no lo levanta
    await worker.tick()
    assert route.call_count == 1
    await ctx.crm.aclose()


async def test_sender_abandona_por_409_sin_handoff(respx_mock):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    pid = await ctx.store.enqueue_pending_send(conv.id, CRM_CONV_ID, "texto")
    respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(409, json={"code": "window_closed"})
    )
    handoff = respx_mock.post(f"{CRM_URL}/api/bot/handoff").mock(
        return_value=httpx.Response(200, json={})
    )
    worker = SenderWorker(ctx)

    await worker.tick()

    assert ctx.store.pending_sends[pid].abandoned_at is not None
    assert handoff.call_count == 0  # rechazo legítimo: sin alerta
    await ctx.crm.aclose()


async def test_sender_usa_el_client_de_la_organizacion_del_pending(respx_mock):
    """El organization_id viaja con el pending_send (app/turn.py); el
    SenderWorker debe reintentar con el CrmClient de ESA organización, header
    X-Organization-Id incluido, nunca con el legacy."""
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY, organization_id="org_a")
    pid = await ctx.store.enqueue_pending_send(
        conv.id, CRM_CONV_ID, "hola org a", organization_id="org_a"
    )
    route = respx_mock.post(
        f"{CRM_URL}/api/bot/messages", headers={"x-organization-id": "org_a"}
    ).mock(return_value=httpx.Response(200, json={"messageId": "m1"}))
    worker = SenderWorker(ctx)

    await worker.tick()

    assert route.call_count == 1
    assert ctx.store.pending_sends[pid].delivered_at is not None
    await ctx.crm.aclose()
    for org_crm in ctx.crm_clients.values():
        await org_crm.aclose()


async def test_sender_agota_24h_abandona_y_alerta(respx_mock):
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    pid = await ctx.store.enqueue_pending_send(conv.id, CRM_CONV_ID, "texto")
    ctx.store.pending_sends[pid].created_at = utcnow() - timedelta(hours=25)
    messages = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "m1"})
    )
    handoff = respx_mock.post(f"{CRM_URL}/api/bot/handoff").mock(
        return_value=httpx.Response(200, json={})
    )
    worker = SenderWorker(ctx)

    await worker.tick()

    assert messages.call_count == 0  # vencido: ni se intenta
    assert ctx.store.pending_sends[pid].abandoned_at is not None
    assert handoff.call_count == 1  # humano alertado vía handoff error
    await ctx.crm.aclose()
