"""Seguimiento: UN empujón suave si el lead se quedó callado.

Loop asyncio cada 60 s. Un empujón a las `FOLLOWUP_HOURS` (default 4) si la
fase lo amerita (hubo conversación, sin resultado, sin handoff). `followup_sent`
se marca ANTES de enviar: a lo sumo uno, incluso con crash a media operación.
Ventana cerrada o IA pausada → se omite con log (v1 no maneja plantillas).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from app.crm import CrmConflict, CrmError
from app.llm import LlmExhausted
from app.multiorg import scoped_ctx
from app.profile import resolve_profile
from app.prompt import FOLLOWUP_INSTRUCTION, build_system_prompt
from app.state import AppContext, Conversation, utcnow
from app.turn import _agent_tz

logger = logging.getLogger("nea.followup")


class FollowupWorker:
    INTERVAL = 60.0

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.INTERVAL)
            try:
                await self.tick()
            except Exception:
                logger.exception("followup: fallo en el barrido")

    async def tick(self, now: datetime | None = None) -> None:
        now = now or utcnow()
        for conv in await self._ctx.store.due_followups(now):
            # Claim atómico ANTES de enviar: jamás un segundo empujón.
            if not await self._ctx.store.claim_followup(conv.id):
                continue
            try:
                await self._push(conv)
            except Exception:
                logger.exception(
                    "followup de %s falló — queda consumido (a lo sumo uno)",
                    conv.wa_identity,
                )

    async def _push(self, conv: Conversation) -> None:
        # La organización de la conversación decide con QUIÉN habla el CRM:
        # None = camino legacy (ctx.crm de siempre); con valor, el CrmClient
        # cacheado de esa organización (nunca el global — X-Organization-Id
        # importa en cada llamada de este empujón).
        ctx = scoped_ctx(self._ctx, conv.organization_id)
        try:
            if conv.organization_id is not None:
                # Modo despacho: el CRM enruta por conversationId — no hay
                # una identidad real que buscar (conversaciones de prueba del
                # Laboratorio incluidas).
                context = await ctx.crm.get_context(
                    conversation_id=conv.crm_conversation_id
                )
            else:
                context = await ctx.crm.get_context(conv.wa_identity)
        except CrmError as exc:
            logger.warning("followup %s: CRM inaccesible (%s) — omitido", conv.wa_identity, exc)
            return
        if context is None:
            logger.warning("followup %s: sin contexto — omitido", conv.wa_identity)
            return
        info = context.get("conversation") or {}
        crm_conv_id = info.get("id") or conv.crm_conversation_id
        if not crm_conv_id:
            logger.warning("followup %s: sin conversationId — omitido", conv.wa_identity)
            return
        if not info.get("aiEnabled", False):
            logger.info("followup %s: IA pausada — omitido", conv.wa_identity)
            return
        if not info.get("windowOpen", False):
            logger.info(
                "followup %s: ventana de 24 h cerrada — omitido con registro",
                conv.wa_identity,
            )
            return

        system = build_system_prompt(
            profile=await resolve_profile(ctx),
            context=context,
            conv=conv,
            offered=[],
            tz=_agent_tz(ctx.settings),
        )
        history = await ctx.store.recent_messages(
            conv.id, ctx.settings.history_window
        )
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in history]
            + [{"role": "system", "content": FOLLOWUP_INSTRUCTION}]
        )
        try:
            reply = await ctx.llm.complete(messages, tools=None)
        except LlmExhausted as exc:
            logger.warning("followup %s: LLM agotado (%s) — omitido", conv.wa_identity, exc)
            return
        text = (reply.content or "").strip()
        if not text:
            logger.warning("followup %s: LLM sin texto — omitido", conv.wa_identity)
            return
        try:
            await ctx.crm.send_message(str(crm_conv_id), text)
        except CrmConflict as exc:
            logger.info("followup %s: bloqueado por el CRM (%s)", conv.wa_identity, exc.code)
            return
        except CrmError as exc:
            logger.warning("followup %s: envío falló (%s)", conv.wa_identity, exc)
            return
        await ctx.store.add_message(conv.id, "assistant", text)
        logger.info("followup %s: empujón único enviado", conv.wa_identity)
