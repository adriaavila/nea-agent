"""Orquestación del turno conversacional.

Gate → contexto del CRM → LLM con tools → envío vía CRM → ficha/fase/seguimiento.
Degradación silenciosa: cualquier fallo termina en silencio + log (y handoff
`error` si el LLM se agotó) — jamás texto roto al lead (Constitución IV).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app import media
from app.config import access_policy, canonical_identity
from app.crm import CrmConflict, CrmError
from app.hostility import ALERT as HOSTILITY_ALERT, hostile_streak
from app.llm import LlmExhausted
from app.profile import resolve_profile
from app.prompt import build_system_prompt
from app.state import AppContext, InboundMessage, TurnCommit, utcnow
from app.tools import TOOL_SCHEMAS, ToolRuntime

logger = logging.getLogger("nea.turn")

MAX_TOOL_ROUNDS = 5
CONTEXT_ATTEMPTS = 3  # el relay puede tardar un instante en aterrizar en el CRM

# Comando de pruebas: reinicia la memoria de ESA conversación. Disponible SOLO
# para identidades de ALLOWED_WA_IDS — con la allowlist vacía queda apagado.
RESET_COMMANDS = frozenset({"/reset", "#reset"})


def _normalized_message(text: str) -> str:
    return " ".join(text.split()).casefold()


def _activation_matches(profile: Any, inbound: list[InboundMessage]) -> bool:
    incoming = _normalized_message("\n".join(m.text for m in inbound if m.text))
    return bool(incoming) and any(
        _normalized_message(message) == incoming
        for message in profile.activation_messages
    )


#: Último recurso, y solo eso. Cablear la zona es como se rompe el
#: agendamiento sin que nada falle: el bot dice "mañana a las 3" en una zona y
#: el motor del CRM reserva en otra.
_TZ_ULTIMO_RECURSO = "America/Mexico_City"


def _agent_tz(settings: Any, profile: Any = None) -> ZoneInfo:
    """La zona en la que el agente PIENSA las fechas.

    Orden: la del negocio (la sirve el CRM y es la que etiqueta los huecos) →
    AGENT_TIMEZONE del despliegue → último recurso.

    El primer escalón importa: sin él, un negocio en Caracas con un bot que
    asume Ciudad de México ofrece horarios con una hora de desfase, y el lead
    llega tarde o temprano a su propia cita.
    """
    candidatos = [
        getattr(profile, "timezone", None),
        getattr(settings, "agent_timezone", None),
        _TZ_ULTIMO_RECURSO,
    ]
    for candidato in candidatos:
        nombre = (candidato or "").strip()
        if not nombre:
            continue
        try:
            return ZoneInfo(nombre)
        except Exception:
            logger.warning("zona horaria inválida %r — sigo con la siguiente", nombre)
    return ZoneInfo(_TZ_ULTIMO_RECURSO)


async def handle_flush(ctx: AppContext, identity: str, items: list[Any]) -> None:
    """Callback del coalescer — nunca propaga excepciones."""
    try:
        await run_turn(ctx, identity, items)
    except Exception:
        logger.exception("turno de %s reventó — silencio", identity)


async def run_turn(
    ctx: AppContext,
    identity: str,
    inbound: list[InboundMessage],
    *,
    organization_id: str | None = None,
    crm_conversation_id: str | None = None,
    bypass_allowlist: bool = False,
    strict: bool = False,
    commit: TurnCommit | None = None,
) -> None:
    """Corre un turno completo.

    `organization_id`/`crm_conversation_id`/`bypass_allowlist`/`strict`/
    `commit` solo los usa el modo de despacho multi-organización
    (app/dispatch.py): `ctx` ya viene con el `crm`/`profile` de la
    organización correcta (app/multiorg.scoped_ctx), `organization_id` se
    persiste en bot_conversation/pending_send para que los workers de fondo
    sepan con quién hablar, `crm_conversation_id` evita resolver el contexto
    por waIdentity (el CRM enruta por conversationId) y hace UN solo intento
    (sin esperar al relay, que en este modo no existe), `bypass_allowlist`
    deja pasar a las conversaciones de prueba del Laboratorio aunque la
    organización tenga la allowlist encendida, `strict=True` convierte un
    fallo real del CRM (context, perfil, activate) en una excepción que se
    propaga — dispatch.py la vuelve 5xx para que el CRM reintente el job — en
    vez del silencio+200 del camino legacy, y `commit` (ver
    app/state.TurnCommit) se marca justo antes del primer efecto irreversible
    hacia el CRM (mandar el mensaje, reservar/mover cita): dispatch.py lo usa
    para decidir si un fallo DESPUÉS de ese punto debe reintentarse (nunca —
    correría el LLM de nuevo y mandaría una SEGUNDA respuesta real) o no. El
    camino legacy (webhook de Meta) no pasa ninguno — comportamiento idéntico
    al de siempre.
    """
    settings = ctx.settings
    conv = await ctx.store.get_or_create_conversation(
        identity, organization_id=organization_id
    )
    await ctx.store.abandon_pending_sends(conv.id)

    # --- Gate 1: contexto + allowlist administrada por el CRM -------------
    context = await _fetch_context(
        ctx, identity, crm_conversation_id=crm_conversation_id, strict=strict
    )
    if context is None:
        logger.warning("turno %s: sin contexto del CRM — silencio", identity)
        return
    restricted, allowed = access_policy(settings, context)
    if restricted and not bypass_allowlist and canonical_identity(identity) not in allowed:
        logger.info(
            "allowlist: %s no autorizado — relay sí, respuesta no", identity
        )
        return

    # --- Comando /reset (líneas de prueba) --------------------------------
    # Corre ANTES de los gates de aiEnabled/ventana: un reset también debe
    # sacar la conversación de un handoff activo.
    if (
        (bypass_allowlist or (restricted and canonical_identity(identity) in allowed))
        and any((m.text or "").strip().lower() in RESET_COMMANDS for m in inbound)
    ):
        await _run_reset(
            ctx,
            conv,
            identity,
            crm_conversation_id=crm_conversation_id,
            strict=strict,
            organization_id=organization_id,
            commit=commit,
        )
        return

    # --- Gate 2: aiEnabled y ventana --------------------------------------
    conversation_info = context.get("conversation") or {}
    crm_conv_id = conversation_info.get("id")
    if not crm_conv_id:
        logger.warning("turno %s: contexto sin conversationId — silencio", identity)
        return
    profile = await resolve_profile(ctx, strict=strict)
    if not conversation_info.get("aiEnabled", False):
        if not profile.activation_enabled or not _activation_matches(profile, inbound):
            logger.info("turno %s: chat pausado y sin mensaje activador — silencio", identity)
            return
        try:
            await ctx.crm.post_activate(str(crm_conv_id))
        except CrmError as exc:
            logger.warning("turno %s: no pude activar el chat (%s) — silencio", identity, exc)
            if strict:
                raise
            return
        conversation_info["aiEnabled"] = True
        await ctx.store.update_conversation(
            conv.id, phase="descubrimiento", followup_due_at=None
        )
        logger.info("turno %s: IA activada por mensaje configurado", identity)
    if not conversation_info.get("windowOpen", False):
        logger.info("turno %s: ventana de 24 h cerrada — silencio", identity)
        return

    await ctx.store.update_conversation(
        conv.id,
        crm_conversation_id=str(crm_conv_id),
        last_inbound_at=utcnow(),
        followup_due_at=None,  # el lead habló: se re-agenda al final del turno
    )

    # Señal de vida: leído + "escribiendo…" mientras Nea piensa (007).
    # Best-effort absoluto: un fallo aquí jamás afecta el turno.
    try:
        await ctx.crm.post_typing(str(crm_conv_id))
    except Exception as exc:
        logger.debug("typing de %s falló (%s) — sigo", identity, exc)

    # --- Contenido del turno: texto + multimedia procesada (spec 002) -----
    parts: list[str] = []
    image_uris: list[str] = []
    for m in inbound:
        if m.text:
            parts.append(m.text)
            continue
        if m.type in ("text", "button", "interactive"):
            continue  # texto vacío raro: nada que procesar
        part = await media.describe_item(ctx, m)
        if part.text:
            parts.append(part.text)
        if part.image_data_uri:
            image_uris.append(part.image_data_uri)
    if not parts and not image_uris:
        logger.info("turno %s: nada procesable en la ráfaga — silencio", identity)
        return

    user_text = "\n".join(parts)
    await ctx.store.add_message(
        conv.id, "user", user_text, wa_message_id=inbound[0].wa_message_id
    )

    # --- Armar mensajes para el LLM ---------------------------------------
    referral = next((m.referral_headline for m in inbound if m.referral_headline), None)
    offered = await ctx.store.get_offered_slots(conv.id)
    system = build_system_prompt(
        profile=profile,
        context=context,
        conv=conv,
        referral_headline=referral,
        offered=offered,
        tz=_agent_tz(settings, profile),
    )
    history = await ctx.store.recent_messages(conv.id, settings.history_window)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content} for m in history
    ]
    # Hostilidad sostenida (AC-18): el CONTEO es determinista — el LLM salió
    # flaky contando entre turnos. Al tercer strike: alerta en el turno y
    # handoff garantizado más abajo aunque el modelo no llame la herramienta.
    streak = hostile_streak([m.content for m in history if m.role == "user"])
    if streak >= 3:
        messages.append({"role": "system", "content": HOSTILITY_ALERT})
    if image_uris:
        # El último user message de este turno se vuelve multimodal: el
        # historial persiste solo el texto; las imágenes viven en ESTE turno.
        last = messages[-1]
        last["content"] = [{"type": "text", "text": str(last["content"])}] + [
            {"type": "image_url", "image_url": {"url": uri}} for uri in image_uris
        ]

    # --- LLM con tools ----------------------------------------------------
    runtime = ToolRuntime(ctx, conv, str(crm_conv_id), profile=profile, commit=commit)
    try:
        final_text = await _tool_loop(ctx, messages, runtime)
    except LlmExhausted as exc:
        logger.error(
            "turno %s: LLM agotó reintentos (%s) — silencio + handoff error",
            identity,
            exc,
        )
        await _safe_handoff(ctx, str(crm_conv_id), "error")
        await ctx.store.update_conversation(
            conv.id, phase="cerrada", followup_due_at=None
        )
        return

    # Backstop determinista: al tercer strike el handoff SUCEDE, lo haya
    # llamado el modelo o no (la regla de negocio no depende de su humor).
    if streak >= 3 and runtime.handoff_reason is None:
        runtime.handoff_reason = "hostilidad"

    # --- Enviar la respuesta (SIEMPRE vía el CRM, nunca Meta directo) -----
    sent = False
    if final_text and final_text.strip():
        sent = await _send(
            ctx,
            conv.id,
            str(crm_conv_id),
            final_text.strip(),
            organization_id=organization_id,
            commit=commit,
        )
        if sent:
            await ctx.store.add_message(conv.id, "assistant", final_text.strip())

    # El handoff se ejecuta DESPUÉS de la despedida (si no, el CRM la rechaza
    # con 409 ai_paused).
    if runtime.handoff_reason is not None:
        await _safe_handoff(ctx, str(crm_conv_id), runtime.handoff_reason)

    # --- Fase + seguimiento -----------------------------------------------
    updates: dict[str, Any] = {"greeted": True}
    if runtime.handoff_reason is not None or runtime.booked or runtime.routed_out:
        updates["phase"] = "cerrada"
        updates["followup_due_at"] = None
    else:
        if runtime.proposed:
            updates["phase"] = "agendando"
        # Las conversaciones de prueba del Laboratorio (identidad
        # `test:<conversationId>`) no son un lead real — no tiene sentido
        # empujarlas con un seguimiento a las N horas.
        is_lab_conversation = identity.startswith("test:")
        if sent and not conv.followup_sent and not is_lab_conversation:
            updates["followup_due_at"] = utcnow() + timedelta(
                hours=settings.followup_hours
            )
    await ctx.store.update_conversation(conv.id, **updates)


async def _run_reset(
    ctx: AppContext,
    conv: Any,
    identity: str,
    crm_conversation_id: str | None = None,
    strict: bool = False,
    organization_id: str | None = None,
    commit: TurnCommit | None = None,
) -> None:
    """Reinicio de pruebas: CRM primero (ficha limpia + IA reactivada, para que
    la confirmación no rebote con 409 ai_paused) y luego la memoria local.

    Prefiere el `crm_conversation_id` que trajo ESTE despacho sobre el
    guardado en `conv` — el que acaba de mandar el CRM es, por definición, el
    vigente; el guardado puede haber quedado stale si la conversación se
    recreó del lado del CRM.
    """
    crm_conv_id = crm_conversation_id or conv.crm_conversation_id
    if not crm_conv_id:
        context = await _fetch_context(
            ctx, identity, crm_conversation_id=crm_conversation_id, strict=strict
        )
        crm_conv_id = ((context or {}).get("conversation") or {}).get("id")
    if crm_conv_id:
        try:
            await ctx.crm.post_reset(str(crm_conv_id))
        except CrmError as exc:
            logger.warning("reset %s: el CRM no pudo reiniciar (%s) — sigo", identity, exc)
    await ctx.store.reset_conversation(conv.id)
    logger.info("reset de pruebas ejecutado para %s", identity)
    if crm_conv_id:
        await _send(
            ctx,
            conv.id,
            str(crm_conv_id),
            "🧹 Listo: memoria reiniciada. Te trato como lead nuevo desde tu "
            "próximo mensaje. (Comando de pruebas, solo líneas autorizadas.)",
            organization_id=organization_id,
            commit=commit,
        )


async def _fetch_context(
    ctx: AppContext,
    identity: str,
    *,
    crm_conversation_id: str | None = None,
    strict: bool = False,
) -> dict[str, Any] | None:
    """`strict=True` (modo despacho): UN solo intento — el CRM ya guardó el
    mensaje antes de despachar, no hay relay corriendo en paralelo que
    esperar — y un CrmError real (no un simple "no lo conozco") se PROPAGA en
    vez de volverse silencio, para que dispatch.py responda 5xx."""
    attempts = 1 if strict else CONTEXT_ATTEMPTS
    for attempt in range(attempts):
        try:
            context = await ctx.crm.get_context(
                identity, conversation_id=crm_conversation_id
            )
        except CrmError as exc:
            logger.warning(
                "context de %s: error del CRM (intento %d): %s",
                identity,
                attempt + 1,
                exc,
            )
            if strict:
                raise
            context = None
        # El relay y este turno corren en paralelo. Un chat conocido puede
        # llegar todavía con la ventana vieja cerrada; espera a que el CRM
        # persista el mensaje entrante que acaba de reabrirla.
        if context is not None and (
            (context.get("conversation") or {}).get("windowOpen", False)
            or attempt == attempts - 1
        ):
            return context
        if attempt < attempts - 1:
            await asyncio.sleep(1.0)  # chance a que el relay aterrice en el CRM
    return None


async def _tool_loop(
    ctx: AppContext, messages: list[dict[str, Any]], runtime: ToolRuntime
) -> str | None:
    """Rondas de tool-calling hasta obtener texto final (o rendirse)."""
    for _ in range(MAX_TOOL_ROUNDS):
        reply = await ctx.llm.complete(messages, tools=TOOL_SCHEMAS)
        if not reply.tool_calls:
            return reply.content  # turno de puro texto
        # content vacío con tool_calls es normal (turno solo-herramientas)
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in reply.tool_calls
                ],
            }
        )
        for tc in reply.tool_calls:
            result = await runtime.execute(tc.name, tc.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
    logger.warning("turno: demasiadas rondas de herramientas — corto sin texto")
    return None


SEND_ATTEMPTS = 4  # backoff 1 s, 2 s, 4 s entre intentos (~7 s en el turno)


async def _send(
    ctx: AppContext,
    conv_id: int,
    crm_conv_id: str,
    text: str,
    organization_id: str | None = None,
    commit: TurnCommit | None = None,
) -> bool:
    """Envía vía el CRM. Si el turno agota sus reintentos, la respuesta NO se
    descarta: se encola en pending_send y el SenderWorker la reintenta con
    backoff hasta entregar o agotar 24 h (incidente 2026-08-03).

    Marca `commit` ANTES del primer intento de red, no después de un éxito:
    incluso un intento que "falla" de nuestro lado puede haber llegado al
    otro — y aunque no llegara, un reintento del turno completo generaría una
    respuesta DISTINTA del LLM y la mandaría también. Pasado este punto ya no
    es seguro reintentar el turno entero desde cero.
    """
    if commit is not None:
        commit.mark()
    for attempt in range(SEND_ATTEMPTS):
        try:
            await ctx.crm.send_message(crm_conv_id, text)
            return True
        except CrmConflict as exc:
            # ai_paused / window_closed: silencio respetuoso, sin reintento.
            logger.info("envío bloqueado por el CRM (%s) — silencio", exc.code)
            return False
        except CrmError as exc:
            logger.warning("envío falló (intento %d): %s", attempt + 1, exc)
            if attempt < SEND_ATTEMPTS - 1:
                await asyncio.sleep(2.0**attempt)
    pending_id = await ctx.store.enqueue_pending_send(
        conv_id, crm_conv_id, text, organization_id=organization_id
    )
    logger.error(
        "envío agotó reintentos del turno — encolado como pending_send %d",
        pending_id,
    )
    return False


async def _safe_handoff(ctx: AppContext, crm_conv_id: str, reason: str) -> None:
    try:
        await ctx.crm.post_handoff(crm_conv_id, reason)
        logger.info("handoff registrado en el CRM (reason=%s)", reason)
    except CrmError as exc:
        logger.error("no pude registrar el handoff (%s): %s", reason, exc)
