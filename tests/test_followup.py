"""Seguimiento: UN empujón y solo uno, marcado ANTES de enviar."""
from __future__ import annotations

from datetime import timedelta

import httpx

from app.followup import FollowupWorker
from app.llm import LlmReply
from app.state import utcnow
from tests.conftest import (
    CRM_CONV_ID,
    CRM_URL,
    IDENTITY,
    crm_context,
    make_ctx,
)


async def preparar_conversacion(ctx, hace_horas: float = 5.0):
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    await ctx.store.add_message(conv.id, "user", "me interesa, luego te digo")
    await ctx.store.add_message(conv.id, "assistant", "va, ¿mañana o pasado?")
    await ctx.store.update_conversation(
        conv.id,
        crm_conversation_id=CRM_CONV_ID,
        greeted=True,
        followup_due_at=utcnow() - timedelta(hours=hace_horas - 4),
    )
    return conv


async def test_followup_un_solo_empujon(respx_mock):
    ctx = make_ctx()
    ctx.llm.replies = [LlmReply(content="¿Seguimos donde nos quedamos? 😊")]
    conv = await preparar_conversacion(ctx)
    respx_mock.get(f"{CRM_URL}/api/bot/context").mock(
        return_value=httpx.Response(200, json=crm_context())
    )
    messages = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "m1"})
    )
    worker = FollowupWorker(ctx)

    await worker.tick()
    assert messages.call_count == 1
    assert ctx.store.conversations[conv.id].followup_sent is True

    # segundo barrido, y hasta un due_at fantasma: jamás un segundo empujón
    await ctx.store.update_conversation(
        conv.id, followup_due_at=utcnow() - timedelta(minutes=1)
    )
    await worker.tick()
    await worker.tick()
    assert messages.call_count == 1
    await ctx.crm.aclose()


async def test_followup_ventana_cerrada_se_omite_y_consume(respx_mock):
    ctx = make_ctx()
    conv = await preparar_conversacion(ctx)
    respx_mock.get(f"{CRM_URL}/api/bot/context").mock(
        return_value=httpx.Response(200, json=crm_context(window_open=False))
    )
    messages = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "m1"})
    )
    worker = FollowupWorker(ctx)
    await worker.tick()
    assert messages.call_count == 0  # omitido con log
    # el claim ocurrió ANTES del envío: aunque "crashee" después, a lo sumo uno
    assert ctx.store.conversations[conv.id].followup_sent is True
    await worker.tick()
    assert messages.call_count == 0
    await ctx.crm.aclose()


async def test_followup_no_aplica_a_conversacion_cerrada(respx_mock):
    ctx = make_ctx()
    conv = await preparar_conversacion(ctx)
    await ctx.store.update_conversation(conv.id, phase="cerrada")
    context_route = respx_mock.get(f"{CRM_URL}/api/bot/context").mock(
        return_value=httpx.Response(200, json=crm_context())
    )
    worker = FollowupWorker(ctx)
    await worker.tick()
    assert context_route.call_count == 0  # ni siquiera se consideró
    assert ctx.store.conversations[conv.id].followup_sent is False
    await ctx.crm.aclose()


async def test_followup_usa_el_client_de_la_organizacion(respx_mock):
    """Una conversación con organization_id (modo despacho) debe empujarse
    con el CrmClient de ESA organización — header X-Organization-Id incluido
    — y buscar el contexto por conversationId, no por waIdentity."""
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY, organization_id="org_a")
    await ctx.store.add_message(conv.id, "user", "me interesa, luego te digo")
    await ctx.store.add_message(conv.id, "assistant", "va, ¿mañana o pasado?")
    await ctx.store.update_conversation(
        conv.id,
        crm_conversation_id=CRM_CONV_ID,
        greeted=True,
        followup_due_at=utcnow() - timedelta(hours=1),
    )
    ctx.llm.replies = [LlmReply(content="¿seguimos donde nos quedamos?")]

    context_route = respx_mock.get(
        f"{CRM_URL}/api/bot/context", headers={"x-organization-id": "org_a"}
    ).mock(return_value=httpx.Response(200, json=crm_context()))
    respx_mock.get(
        f"{CRM_URL}/api/bot/profile", headers={"x-organization-id": "org_a"}
    ).mock(return_value=httpx.Response(404))
    messages_route = respx_mock.post(
        f"{CRM_URL}/api/bot/messages", headers={"x-organization-id": "org_a"}
    ).mock(return_value=httpx.Response(200, json={"messageId": "m1"}))
    worker = FollowupWorker(ctx)

    await worker.tick()

    assert context_route.call_count == 1
    assert messages_route.call_count == 1
    await ctx.crm.aclose()
    for org_crm in ctx.crm_clients.values():
        await org_crm.aclose()


async def test_followup_llm_caido_se_omite_sin_reintento_futuro(respx_mock):
    from app.llm import LlmExhausted

    ctx = make_ctx()
    ctx.llm.raise_exc = LlmExhausted("apagado")
    conv = await preparar_conversacion(ctx)
    respx_mock.get(f"{CRM_URL}/api/bot/context").mock(
        return_value=httpx.Response(200, json=crm_context())
    )
    messages = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "m1"})
    )
    worker = FollowupWorker(ctx)
    await worker.tick()
    assert messages.call_count == 0
    assert ctx.store.conversations[conv.id].followup_sent is True  # consumido
    await ctx.crm.aclose()
