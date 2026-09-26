"""Despacho v2: el turno SIN estado propio.

Con dispatch v2 el CRM manda TODO lo que Nea necesita para pensar un turno —
`context`/`profile` son los cuerpos EXACTOS de `GET /api/bot/context` y
`GET /api/bot/profile`, `history` es la conversación completa (incluida la voz
del dueño/equipo del negocio, que Nea antes no veía) y `offers` es lo último
que se le ofreció al lead. Este módulo NUNCA usa `ctx.store`: ni
`mark_processed`, ni `bot_conversation`, ni `bot_message`, ni `pending_send`.
El dedup de reintentos del CRM lo resuelve `dispatchId` de su lado (mensajes
idempotentes); aquí el único "estado" que existe vive y muere con el turno
(`app.tools.MemoryOfferBook`).

`app/dispatch.py` es quien decide, por `version`, si un despacho llega aquí o
al camino v1/legacy (`app/turn.py`, sin tocar). Esa frontera la vigila un
test con un Store que revienta ante cualquier llamada (`app.state.NullStore`).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

from app import media
from app.config import access_policy, canonical_identity
from app.crm import CrmClient, CrmConflict, CrmError, CrmPaymentRequired, canonical_handoff_reason
from app.hostility import ALERT as HOSTILITY_ALERT, hostile_streak
from app.llm import Llm, LlmAuthFailed, LlmExhausted, LlmNoCredits, OpenAiLlm
from app.multiorg import crm_for
from app.profile import BusinessProfile, profile_from_payload
from app.prompt import TEAM_OWNER_MARKER_PREFIX, TEAM_OWNER_NOTE, build_system_prompt
from app.state import AppContext, Conversation, InboundMessage, OfferedSlot, TurnCommit
from app.tools import TOOL_SCHEMAS, MemoryOfferBook, ToolRuntime
from app.turn import _agent_tz

logger = logging.getLogger("nea.stateless")

MAX_TOOL_ROUNDS = 5
SEND_ATTEMPTS = 3  # backoff 1 s, 2 s — el despacho entero tiene 75 s (dispatch.py)

RESET_COMMANDS = frozenset({"/reset", "#reset"})
RESET_NOTICE = (
    "🧹 Listo: memoria reiniciada. Te trato como lead nuevo desde tu próximo "
    "mensaje. (Comando de pruebas, solo líneas autorizadas.)"
)

# mismo mapeo mime→extensión que app/media.py, para el nombre de archivo que
# se le manda a la transcripción (algunos proveedores lo usan para elegir el
# decoder).
_AUDIO_EXT = {
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/amr": "amr",
    "audio/wav": "wav",
}


# ------------------------------------------------------------------ tipos ---


class DispatchMediaIn(BaseModel):
    mediaId: str | None = None
    mime: str | None = None
    fileName: str | None = None
    caption: str | None = None
    transcript: str | None = None
    location: dict[str, Any] | None = None
    contacts: list[str] | None = None


class DispatchHistoryItemIn(BaseModel):
    id: str | None = None
    role: str  # lead | agent | team | owner
    type: str = "text"
    text: str | None = None
    at: str | None = None
    pending: bool = False
    media: DispatchMediaIn | None = None


class DispatchOfferIn(BaseModel):
    startUtc: str
    label: str | None = None


class DispatchLlmIn(BaseModel):
    provider: str  # openrouter | openai
    model: str
    apiKey: SecretStr


class DispatchPayloadV2(BaseModel):
    """Valida los TIPOS del contrato v2 (400 si no calzan). Los campos v1
    (`contact`, `messages` a nivel raíz) se ignoran a propósito — pydantic los
    descarta solo (default `extra='ignore'`), nunca se declaran aquí."""

    version: int = 2
    dispatchId: str | None = None
    attempt: int = 0
    organizationId: str | None = None
    conversationId: str | None = None
    isTest: bool = False
    context: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    history: list[DispatchHistoryItemIn] = Field(default_factory=list)
    offers: list[DispatchOfferIn] = Field(default_factory=list)
    llm: DispatchLlmIn | None = None

    @field_validator("isTest", mode="before")
    @classmethod
    def _is_test_debe_ser_bool(cls, v: Any) -> Any:
        if not isinstance(v, bool):
            raise ValueError("isTest debe ser boolean, no string/número")
        return v


@dataclass
class V2Result:
    """Lo que `app/dispatch.py` traduce al sobre de respuesta del contrato."""

    action: str  # replied | silent | noop | reset
    llm_source: str = "platform"
    llm_status: str = "ok"
    handoff_reason: str | None = None
    handoff_applied: bool | None = None


@dataclass
class _LlmState:
    """Qué cliente terminó respondiendo y con qué estado — sobrevive aunque
    `_tool_loop` termine en `LlmExhausted` (la excepción no puede cargar esto
    y el llamador necesita saberlo igual para el `llm` de la respuesta)."""

    source: str = "platform"
    status: str = "ok"


# --------------------------------------------------------------- identidad ---


def turn_identity(payload: DispatchPayloadV2) -> str:
    """La identidad para el lock por (organización, identidad) y la allowlist.

    Pruebas del Laboratorio (`isTest`): una identidad propia por conversación,
    igual que v1 — dos corridas nunca comparten candado ni allowlist. Fuera de
    eso, la del contacto que ya viene en `context` (mismo shape que
    `GET /api/bot/context`): no hace falta el `contact.identity` de nivel
    raíz (campo v1, ignorado en v2)."""
    conversation_id = (payload.conversationId or "").strip()
    if payload.isTest:
        return f"test:{conversation_id}"
    contact = (payload.context or {}).get("contact") or {}
    raw = contact.get("waIdentity") or contact.get("phone") or ""
    return str(raw).strip()


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _activation_matches(profile: BusinessProfile, incoming_text: str) -> bool:
    incoming = _normalized(incoming_text)
    return bool(incoming) and any(
        _normalized(m) == incoming for m in profile.activation_messages
    )


def _is_reset_command(pending: list[DispatchHistoryItemIn]) -> bool:
    return any((m.text or "").strip().lower() in RESET_COMMANDS for m in pending)


# ------------------------------------------------------------------ media ---


def _render_location_text(loc: dict[str, Any] | None) -> str:
    """Mismo blindaje que app/media.py._location: jamás "lat None, long None"."""
    loc = loc or {}
    detalles = ", ".join(str(loc[k]) for k in ("name", "address") if loc.get(k))
    lat, lon = loc.get("latitude"), loc.get("longitude")
    coords = f"lat {lat}, long {lon}" if lat is not None and lon is not None else None
    if coords is None and not detalles:
        return "[Compartió su ubicación, pero no llegaron las coordenadas.]"
    cuerpo = f"{detalles} — {coords}" if detalles and coords else (coords or detalles)
    return f"[Compartió su ubicación: {cuerpo}.]"


def _caption_suffix(media_: DispatchMediaIn | None) -> str:
    if media_ and media_.caption:
        return f' Nota junto al envío: "{media_.caption}".'
    return ""


def _settled_media_text(item: DispatchHistoryItemIn) -> str | None:
    """Media de un mensaje YA resuelto (no pendiente de esta ráfaga): solo
    texto, con lo que el CRM ya trae — nunca se vuelve a descargar nada."""
    media_ = item.media
    if item.type == "audio":
        if media_ and media_.transcript:
            return f"[Nota de voz, transcrita]: {media_.transcript}"
        return f"[Nota de voz — sin transcripción disponible.{_caption_suffix(media_)}]"
    if item.type == "location":
        return _render_location_text(media_.location if media_ else None)
    if item.type == "contacts":
        nombres = ", ".join((media_.contacts if media_ else None) or []) or "alguien"
        return f"[Compartió una tarjeta de contacto de: {nombres}.]"
    etiquetas = {
        "image": "una imagen",
        "document": "un documento",
        "video": "un video",
        "sticker": "un sticker",
    }
    etiqueta = etiquetas.get(item.type, "contenido")
    return f"[Envió {etiqueta}.{_caption_suffix(media_)}]"


async def _transcribe_pending_audio(
    media_ctx: AppContext, item: DispatchHistoryItemIn
) -> str:
    """Nota de voz PENDIENTE: si ya trae transcripción del CRM, se usa tal
    cual (jamás se repite la llamada al LLM). Si no, se baja el binario y se
    transcribe SIEMPRE con `media_ctx.llm` — el llamador (run_turn) garantiza
    que ese es el cliente de PLATAFORMA, nunca la clave de un negocio — y se
    escribe de vuelta al CRM antes de seguir (best-effort: un fallo aquí no
    tumba el turno, ver contrato)."""
    media_ = item.media
    if media_ and media_.transcript:
        return f"[Nota de voz del lead, transcrita]: {media_.transcript}"
    honesto = (
        "[El lead mandó una nota de voz que NO pudiste abrir/procesar. Sé "
        "honesta: dile que no la pudiste abrir y pídele el contenido en "
        "texto.]"
    )
    if not media_ or not media_.mediaId:
        return honesto
    try:
        data, mime = await media_ctx.crm.get_media(media_.mediaId)
        mime_clean = (mime or media_.mime or "audio/ogg").split(";")[0].strip()
        ext = _AUDIO_EXT.get(mime_clean, "ogg")
        transcript = await media_ctx.llm.transcribe(
            data, mime_clean, filename=f"nota-de-voz.{ext}"
        )
    except Exception:
        logger.exception(
            "stateless: transcripción de %s falló", media_.mediaId
        )
        return honesto
    if item.id:
        try:
            await media_ctx.crm.post_transcript(item.id, transcript)
        except Exception as exc:
            logger.warning(
                "stateless: no pude escribir la transcripción de %s: %s",
                item.id,
                exc,
            )
    return f"[Nota de voz del lead, transcrita]: {transcript}"


async def _describe_pending_item(
    media_ctx: AppContext, item: DispatchHistoryItemIn
) -> tuple[str | None, str | None]:
    """Todo lo demás (imagen, documento, sticker, video, ubicación,
    contactos) reutiliza app/media.py tal cual — mismo código probado del
    camino legacy, ninguno de esos handlers toca el LLM."""
    media_ = item.media
    msg = InboundMessage(
        wa_message_id=item.id,
        identity="",
        type=item.type,
        text=item.text,
        media_id=media_.mediaId if media_ else None,
        media_mime=media_.mime if media_ else None,
        media_filename=media_.fileName if media_ else None,
        media_caption=media_.caption if media_ else None,
        media_voice=(item.type == "audio"),
        location=media_.location if media_ else None,
        contact_names=(media_.contacts if media_ else None) or [],
    )
    part = await media.describe_item(media_ctx, msg)
    return part.text, part.image_data_uri


async def _process_pending(
    media_ctx: AppContext, pending: list[DispatchHistoryItemIn]
) -> tuple[list[str], list[str]]:
    """Baja y describe/transcribe lo pendiente, EN ORDEN — como las ráfagas
    de siempre, solo que la fuente es `payload.history` y no el coalescer."""
    texts: list[str] = []
    images: list[str] = []
    for item in pending:
        if item.type in ("text", "button", "interactive"):
            if item.text:
                texts.append(item.text)
            continue
        if item.type == "audio":
            texts.append(await _transcribe_pending_audio(media_ctx, item))
            continue
        text, image_uri = await _describe_pending_item(media_ctx, item)
        if text:
            texts.append(text)
        if image_uri:
            images.append(image_uri)
    return texts, images


# --------------------------------------------------------------- prompt ---


def _history_messages(
    history: list[DispatchHistoryItemIn],
) -> tuple[list[dict[str, Any]], bool]:
    """Historial YA resuelto (sin los pendientes de esta ráfaga, que se unen
    al mensaje final aparte) para el LLM, en orden cronológico. `team`/`owner`
    llevan un marcador — nunca los escribió el lead ni Nea."""
    out: list[dict[str, Any]] = []
    had_team_or_owner = False
    for item in history:
        if item.role == "lead" and item.pending:
            continue
        if item.role == "lead":
            text = (
                item.text
                if item.type in ("text", "button", "interactive")
                else _settled_media_text(item)
            )
            if text:
                out.append({"role": "user", "content": text})
        elif item.role == "agent":
            if item.text:
                out.append({"role": "assistant", "content": item.text})
        else:  # team | owner
            had_team_or_owner = True
            if item.text:
                out.append(
                    {
                        "role": "assistant",
                        "content": f"{TEAM_OWNER_MARKER_PREFIX}: {item.text}",
                    }
                )
    return out, had_team_or_owner


def _bursts(history: list[DispatchHistoryItemIn], pending_text: str) -> list[str]:
    """Agrupa mensajes CONSECUTIVOS del lead en una ráfaga — la hostilidad
    sostenida (AC-18) cuenta ráfagas, no mensajes sueltos. Cualquier mensaje
    de agent/team/owner corta la racha, igual que un mensaje no-hostil."""
    bursts: list[str] = []
    current: list[str] = []
    for item in history:
        if item.role == "lead" and item.pending:
            continue  # el pendiente se agrega aparte, al final
        if item.role == "lead":
            if item.text:
                current.append(item.text)
        elif current:
            bursts.append("\n".join(current))
            current = []
    if current:
        bursts.append("\n".join(current))
    if pending_text:
        bursts.append(pending_text)
    return bursts


def _parse_utc(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _offers_from_payload(offers: list[DispatchOfferIn]) -> list[OfferedSlot]:
    out: list[OfferedSlot] = []
    for raw in offers:
        start = _parse_utc(raw.startUtc)
        if start is None:
            continue
        out.append(
            OfferedSlot(
                conversation_id=0,
                start_utc=start,
                end_utc=None,
                label=raw.label or raw.startUtc,
            )
        )
    return out


# ------------------------------------------------------------------- LLM ---


def _build_org_llm(llm_in: DispatchLlmIn) -> OpenAiLlm:
    """Cliente del NEGOCIO, vivo solo durante este turno — `run_turn` lo
    cierra siempre en su `finally`. Nunca sirve para transcribir (eso es
    SIEMPRE la plataforma, ver app/llm.py)."""
    base_url = "https://openrouter.ai/api/v1" if llm_in.provider == "openrouter" else None
    return OpenAiLlm(llm_in.apiKey.get_secret_value(), llm_in.model, base_url=base_url)


async def _tool_loop(
    messages: list[dict[str, Any]],
    runtime: ToolRuntime,
    org_llm: Llm | None,
    platform_llm: Llm,
    state: _LlmState,
) -> str | None:
    """Rondas de tool-calling hasta obtener texto final (o rendirse) — igual
    que app/turn.py._tool_loop, más el fallback de clave por negocio: si el
    cliente del negocio falla por credenciales/crédito, esta MISMA llamada se
    reintenta YA con la plataforma, y el resto del turno sigue con ella."""
    active = org_llm or platform_llm
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            reply = await active.complete(messages, tools=TOOL_SCHEMAS)
        except (LlmAuthFailed, LlmNoCredits) as exc:
            if active is platform_llm:
                # la plataforma TAMBIÉN falló por credenciales — no hay a
                # dónde más caer; se trata como agotado (silencio + handoff).
                raise LlmExhausted(str(exc)) from exc
            state.status = "auth_failed" if isinstance(exc, LlmAuthFailed) else "no_credits"
            state.source = "platform"
            logger.warning(
                "stateless: LLM del negocio falló (%s) — cae a la plataforma", exc
            )
            active = platform_llm
            reply = await active.complete(messages, tools=TOOL_SCHEMAS)
        if not reply.tool_calls:
            return reply.content
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
    logger.warning("stateless: demasiadas rondas de herramientas — corto sin texto")
    return None


# ----------------------------------------------------------------- envío ---


async def _send(
    crm: CrmClient,
    conversation_id: str,
    text: str,
    *,
    dispatch_id: str | None,
    commit: TurnCommit | None,
) -> bool:
    """Envía vía el CRM con `dispatchId`/`seq` (mensajes idempotentes, PR 2A).

    El commit se marca SOLO tras un 2xx — a diferencia de app/turn.py._send,
    que lo marca antes de intentar (ahí hace falta: sin envío idempotente, un
    reintento del turno completo podría mandar una SEGUNDA respuesta real).
    Aquí no: sin `pending_send` que encole el resultado, una respuesta que
    nunca se confirmó DEBE volver 5xx para que el CRM reintente el despacho —
    reintentar es seguro porque el CRM dedupea por el mismo `dispatchId`.
    """
    last_error: Exception | None = None
    for attempt in range(SEND_ATTEMPTS):
        try:
            await crm.send_message(conversation_id, text, dispatch_id=dispatch_id, seq=0)
        except CrmConflict as exc:
            if exc.code == "send_in_progress" and attempt < SEND_ATTEMPTS - 1:
                logger.warning(
                    "envío v2: send_in_progress (intento %d) — reintento", attempt + 1
                )
                last_error = exc
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            if exc.code == "send_in_progress":
                raise  # agotó reintentos con el CRM en vuelo: 5xx, que reintente el despacho
            # ai_paused / window_closed: silencio respetuoso, sin reintento.
            logger.info("envío v2 bloqueado por el CRM (%s) — silencio", exc.code)
            return False
        except CrmPaymentRequired:
            logger.warning("envío v2: 402 (límite del plan) — final, silencio")
            return False
        except CrmError as exc:
            last_error = exc
            logger.warning("envío v2 falló (intento %d): %s", attempt + 1, exc)
            if attempt < SEND_ATTEMPTS - 1:
                await asyncio.sleep(2.0**attempt)
                continue
            raise
        else:
            if commit is not None:
                commit.mark()
            return True
    if last_error is not None:
        raise last_error
    return False


async def _safe_handoff(crm: CrmClient, conversation_id: str, reason: str) -> bool:
    try:
        await crm.post_handoff(conversation_id, reason)
        logger.info("stateless: handoff registrado (reason=%s)", reason)
        return True
    except CrmError as exc:
        logger.error("stateless: no pude registrar el handoff (%s): %s", reason, exc)
        return False


async def _run_reset(
    crm: CrmClient,
    payload: DispatchPayloadV2,
    *,
    conversation_id: str,
    commit: TurnCommit | None,
) -> V2Result:
    """`/reset` en v2: sin `bot_conversation` que borrar, el CRM es quien
    limpia SU cursor de memoria y manda el aviso — Nea solo se lo pide."""
    await crm.post_reset(conversation_id, notice=RESET_NOTICE, dispatch_id=payload.dispatchId)
    if commit is not None:
        commit.mark()
    logger.info("stateless %s: reset de pruebas ejecutado", conversation_id)
    return V2Result(action="reset")


# ------------------------------------------------------------------ turno ---


async def run_turn(
    ctx: AppContext,
    payload: DispatchPayloadV2,
    *,
    organization_id: str,
    commit: TurnCommit | None = None,
) -> V2Result:
    """Corre UN turno v2 completo. `ctx.store` NUNCA se toca — todo lo que se
    necesita ya viene en `payload` (ver el docstring del módulo)."""
    context = payload.context or {}
    conv_info = dict(context.get("conversation") or {})
    conversation_id = (payload.conversationId or "").strip()

    pending = [m for m in payload.history if m.role == "lead" and m.pending]
    if not pending:
        logger.info("stateless %s: nada pendiente — noop", conversation_id)
        return V2Result(action="noop")

    identity = turn_identity(payload)
    restricted, allowed = access_policy(ctx.settings, context)
    bypass = payload.isTest
    in_allowlist = restricted and canonical_identity(identity) in allowed
    if restricted and not bypass and not in_allowlist:
        logger.info(
            "stateless %s: %s no autorizado — silencio", conversation_id, identity
        )
        return V2Result(action="silent")

    scoped_crm = crm_for(ctx, organization_id)

    if _is_reset_command(pending) and (bypass or in_allowlist):
        return await _run_reset(
            scoped_crm, payload, conversation_id=conversation_id, commit=commit
        )

    profile = profile_from_payload(payload.profile or {}, default_name=ctx.settings.agent_name)

    if not conv_info.get("aiEnabled", False):
        pending_text = "\n".join(m.text for m in pending if m.text)
        if not profile.activation_enabled or not _activation_matches(profile, pending_text):
            logger.info(
                "stateless %s: chat pausado y sin mensaje activador — silencio",
                conversation_id,
            )
            return V2Result(action="silent")
        try:
            await scoped_crm.post_activate(conversation_id)
        except CrmError as exc:
            logger.warning(
                "stateless %s: no pude activar el chat (%s) — silencio",
                conversation_id,
                exc,
            )
            return V2Result(action="silent")
        conv_info["aiEnabled"] = True
        logger.info("stateless %s: IA activada por mensaje configurado", conversation_id)

    if not conv_info.get("windowOpen", False):
        logger.info("stateless %s: ventana de 24 h cerrada — silencio", conversation_id)
        return V2Result(action="silent")

    try:
        await scoped_crm.post_typing(conversation_id)
    except Exception as exc:  # best-effort absoluto (007)
        logger.debug("stateless %s: typing falló (%s) — sigo", conversation_id, exc)

    # ctx de PLATAFORMA para media/transcripción: crm de la organización, llm
    # SIEMPRE el de ctx (nunca la clave de un negocio — ver app/llm.py).
    media_ctx = dataclasses.replace(ctx, crm=scoped_crm)
    pending_texts, image_uris = await _process_pending(media_ctx, pending)
    if not pending_texts and not image_uris:
        logger.info("stateless %s: nada procesable en la ráfaga — silencio", conversation_id)
        return V2Result(action="silent")
    final_user_text = "\n".join(pending_texts)

    history_messages, had_team_or_owner = _history_messages(payload.history)
    streak = hostile_streak(_bursts(payload.history, final_user_text))

    offer_book = MemoryOfferBook(_offers_from_payload(payload.offers))
    fake_conv = Conversation(
        id=0,
        wa_identity=identity or conversation_id,
        greeted=bool(conv_info.get("agentHasSpoken")),
    )
    system = build_system_prompt(
        profile=profile,
        context=context,
        conv=fake_conv,
        offered=await offer_book.get(),
        tz=_agent_tz(ctx.settings, profile),
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if had_team_or_owner:
        messages.append({"role": "system", "content": TEAM_OWNER_NOTE})
    messages += history_messages
    if streak >= 3:
        messages.append({"role": "system", "content": HOSTILITY_ALERT})
    final_content: Any = final_user_text
    if image_uris:
        final_content = [{"type": "text", "text": final_user_text}] + [
            {"type": "image_url", "image_url": {"url": uri}} for uri in image_uris
        ]
    messages.append({"role": "user", "content": final_content})

    runtime = ToolRuntime(
        ctx=media_ctx,
        conv=None,
        crm_conversation_id=conversation_id,
        profile=profile,
        commit=commit,
        offers=offer_book,
    )
    org_llm = _build_org_llm(payload.llm) if payload.llm is not None else None
    state = _LlmState(source="org" if org_llm is not None else "platform")
    try:
        try:
            final_text = await _tool_loop(messages, runtime, org_llm, ctx.llm, state)
        except LlmExhausted as exc:
            logger.error(
                "stateless %s: LLM agotó reintentos (%s) — silencio + handoff error",
                conversation_id,
                exc,
            )
            applied = await _safe_handoff(scoped_crm, conversation_id, "error")
            return V2Result(
                action="silent",
                llm_source=state.source,
                llm_status=state.status,
                handoff_reason="error",
                handoff_applied=applied,
            )
    finally:
        if org_llm is not None:
            await org_llm.aclose()

    if streak >= 3 and runtime.handoff_reason is None:
        runtime.handoff_reason = "hostilidad"

    sent = False
    if final_text and final_text.strip():
        sent = await _send(
            scoped_crm,
            conversation_id,
            final_text.strip(),
            dispatch_id=payload.dispatchId,
            commit=commit,
        )

    # El handoff se ejecuta DESPUÉS de la despedida (si no, el CRM la rechaza
    # con 409 ai_paused) — igual que en app/turn.py.
    handoff_applied: bool | None = None
    handoff_reason: str | None = None
    if runtime.handoff_reason is not None:
        handoff_reason = canonical_handoff_reason(runtime.handoff_reason)
        handoff_applied = await _safe_handoff(scoped_crm, conversation_id, runtime.handoff_reason)

    return V2Result(
        action="replied" if sent else "silent",
        llm_source=state.source,
        llm_status=state.status,
        handoff_reason=handoff_reason,
        handoff_applied=handoff_applied,
    )
