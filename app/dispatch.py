"""Ruta de despacho: el CRM multi-tenant le entrega el turno ya armado a Nea.

Es la inversión del webhook clásico. Con un solo negocio, Meta manda el
webhook a Nea y Nea lo releva al CRM. Con un Vocero multitenant, el CRM ya
recibió el webhook de Meta para TODAS sus organizaciones, guardó el mensaje,
hizo su propio debounce, y ahora le DESPACHA el turno a un Nea compartido
entre organizaciones.

Eso cambia tres cosas:

- **No hay relay.** El CRM ya tiene el mensaje guardado; reenviárselo lo
  duplicaría.
- **No hay coalesce.** El CRM ya esperó a que el lead terminara de escribir y
  entrega la ráfaga junta — sumarle otra espera aquí solo le agrega latencia a
  un cliente que ya esperó la del CRM.
- **Es síncrono.** La respuesta HTTP solo llega después de correr el turno
  completo: si algo revienta (o tarda más de lo que el CRM espera), se
  responde 5xx para que el CRM marque el job fallido y reintente — y el
  `mark_processed` que se reclamó para ESTE intento se SUELTA antes de
  contestar, o el reintento encontraría los ids ya "procesados" y el mensaje
  se perdería en silencio (200 sin turno).

Esta ruta se monta SIEMPRE (no hay bandera de modo) — la protege la firma, y
un secreto vacío aquí SIEMPRE rechaza (a diferencia del webhook de Meta, donde
un secreto vacío es "dev, no verifiques"). El webhook de Meta, el relay y el
coalescer de siempre no la ven ni ella los toca.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from app import stateless
from app.config import canonical_identity
from app.multiorg import scoped_ctx
from app.state import AppContext, InboundMessage, TurnCommit
from app.turn import run_turn

logger = logging.getLogger("nea.dispatch")

router = APIRouter()

# El CRM da por fallido el job de despacho alrededor de los 90 s; respondemos
# ANTES de eso para poder soltar los ids reclamados nosotros mismos en vez de
# dejar la conexión colgada hasta que el CRM decida que se cortó.
DISPATCH_TIMEOUT_SECONDS = 75.0


def _error_locations(exc: ValidationError) -> list[str]:
    """Solo las UBICACIONES del error de validación (p.ej. "llm.apiKey"),
    NUNCA el valor — un payload v2 con un `apiKey` mal tipado no debe
    terminar impreso en el log por loguear el ValidationError completo."""
    return [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()]


# ------------------------------------------------------------------ tipos ---


class DispatchContact(BaseModel):
    identity: str | None = None
    name: str | None = None


class DispatchMessageIn(BaseModel):
    id: str | None = None
    type: str = "text"
    text: str | None = None
    mediaId: str | None = None
    timestamp: str | None = None


class DispatchPayload(BaseModel):
    """Valida los TIPOS del contrato fijo con el CRM (400 si no calzan). La
    presencia de organizationId/conversationId/contact.identity se revisa
    aparte en el handler — necesita un mensaje de error específico por campo,
    no el genérico de un ValidationError."""

    organizationId: str | None = None
    conversationId: str | None = None
    isTest: bool = False
    contact: DispatchContact = Field(default_factory=DispatchContact)
    messages: list[DispatchMessageIn] = Field(default_factory=list)

    @field_validator("isTest", mode="before")
    @classmethod
    def _is_test_debe_ser_bool(cls, v: Any) -> Any:
        # pydantic en modo laxo acepta "true"/1/"yes" como bool; el contrato
        # pide un boolean de verdad — una CRM con un bug de serialización que
        # mande "false" (string, truthy en JS) no debe activar el Laboratorio
        # ni saltarse la allowlist de una organización real.
        if not isinstance(v, bool):
            raise ValueError("isTest debe ser boolean, no string/número")
        return v


def verify_dispatch_signature(body: bytes, header: str | None, secret: str) -> bool:
    """HMAC-SHA256 del cuerpo CRUDO con la CRM_BOT_API_KEY compartida.

    A diferencia de `webhook.verify_signature` (secreto vacío = no se exige
    firma, modo dev de un solo negocio), aquí un secreto vacío SIEMPRE
    rechaza: es la única prueba de que quien despacha es el CRM y no
    cualquiera que conozca la URL. Se verifica sobre el cuerpo crudo, ANTES de
    parsear nada — un byte de diferencia en el re-serializado y la firma jamás
    coincide.

    Compara BYTES, no strings: `hmac.compare_digest` truena con `TypeError`
    si le pasas un `str` con caracteres no-ASCII (un header corrupto o con
    basura lo dispara), y esa excepción sin capturar se volvía un 500 en vez
    de un 401 — un atacante podía tumbar el endpoint mandando un header raro.
    """
    if not secret or not header:
        return False
    if not header.startswith("sha256="):
        return False
    try:
        received = bytes.fromhex(header[len("sha256="):].strip())
    except ValueError:
        return False  # no era hex válido (incluye cualquier basura no-ASCII)
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return hmac.compare_digest(received, expected)


def messages_from_dispatch(
    payload: DispatchPayload, identity: str
) -> list[InboundMessage]:
    """Traduce los mensajes ya validados del despacho al formato que consume
    el turno. `id: null` es el caso normal de una conversación de prueba del
    Laboratorio — jamás se manda a `mark_processed` (ver el handler), así que
    nunca "se dedupea" a sí mismo."""
    return [
        InboundMessage(
            wa_message_id=raw.id,
            identity=identity,
            type=raw.type,
            text=raw.text,
            profile_name=payload.contact.name,
            media_id=raw.mediaId,
        )
        for raw in payload.messages
    ]


def _lock_for(ctx: AppContext, organization_id: str, identity: str) -> asyncio.Lock:
    """Lock por (organización, identidad): serializa despachos concurrentes
    de la MISMA conversación en este proceso (ver AppContext.dispatch_locks
    — nota de escalado ahí). `ctx.dispatch_locks` es un WeakValueDictionary:
    basta con NO guardar una referencia fuerte propia más allá del request
    para que se limpie solo en cuanto nadie más lo esté usando."""
    key = (organization_id, identity)
    lock = ctx.dispatch_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        ctx.dispatch_locks[key] = lock
    return lock


async def _release_claimed_ids(ctx: AppContext, claimed_ids: list[str]) -> None:
    """Suelta el claim de mark_processed. Se blinda con su propio try/except:
    si ESTO falla, los ids quedan atascados como "procesados" y el CRM jamás
    podrá reintentarlos solo — hace falta intervención manual, así que se
    loguea fuerte en vez de dejarlo pasar en silencio."""
    if not claimed_ids:
        return
    try:
        await asyncio.shield(ctx.store.release_processed(claimed_ids))
    except Exception:
        logger.exception(
            "dispatch: release_processed FALLÓ para %s — quedan atascados "
            "como 'procesados'; el CRM NO va a poder reintentarlos solo, "
            "hace falta intervención manual",
            claimed_ids,
        )


def _log_background_outcome(organization_id: str, conversation_id: str):
    """Callback para el turno que sigue corriendo en segundo plano tras un
    timeout post-commit (ver dispatch()): ya respondimos 200, pero si termina
    en error igual queremos que quede en el log — nunca "Task exception was
    never retrieved" silencioso."""

    def _callback(task: "asyncio.Task[None]") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "dispatch %s/%s: el turno en segundo plano (post-commit) "
                "terminó con error DESPUÉS de responder 200: %s",
                organization_id,
                conversation_id,
                exc,
            )

    return _callback


@router.post("/dispatch")
async def dispatch(request: Request) -> Any:
    """Enruta por `version`: sin ella, o `1`, es el camino v1/legacy de
    siempre (`_dispatch_v1`, sin tocar); `2` es el despacho sin estado
    (`_dispatch_v2` → app/stateless.py, que NUNCA usa `ctx.store`). La firma,
    el parseo del cuerpo y el límite de tamaño de versión soportada son
    comunes a las dos."""
    ctx: AppContext = request.app.state.ctx
    body = await request.body()
    signature = request.headers.get("x-signature")
    if not verify_dispatch_signature(body, signature, ctx.settings.crm_bot_api_key):
        logger.warning("dispatch: firma inválida o ausente — 401")
        return JSONResponse({"error": "firma inválida"}, status_code=401)

    try:
        raw_payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        logger.warning("dispatch: cuerpo ilegible — 400")
        return JSONResponse({"error": "cuerpo ilegible"}, status_code=400)
    if not isinstance(raw_payload, dict):
        return JSONResponse({"error": "cuerpo inesperado"}, status_code=400)

    version = raw_payload.get("version")
    if version not in (None, 1, 2):
        logger.warning("dispatch: version %r no soportada — 400", version)
        return JSONResponse({"error": "version no soportada"}, status_code=400)
    if version == 2:
        return await _dispatch_v2(ctx, raw_payload)
    return await _dispatch_v1(ctx, raw_payload)


async def _dispatch_v1(ctx: AppContext, raw_payload: dict[str, Any]) -> Any:
    """Camino v1/legacy — comportamiento IDÉNTICO al de antes de dispatch v2,
    solo que ahora vive en su propia función. Usa `ctx.store` (dedup,
    `bot_conversation`, `bot_message`, `pending_send`): con `DISPATCH_ONLY=true`
    y sin `DATABASE_URL` (ver app/main.py) no hay Store real que tocar, así
    que se corta ANTES de intentarlo — 503, claro, en vez de un error de
    conexión opaco contra una base que no existe."""
    if ctx.settings.dispatch_only and not ctx.settings.database_url:
        logger.warning(
            "dispatch v1: instancia sin base de datos (DISPATCH_ONLY sin "
            "DATABASE_URL) — 503, el CRM debe mandar version=2"
        )
        return JSONResponse(
            {"error": "v1 no disponible: esta instancia corre sin base de datos"},
            status_code=503,
        )

    try:
        payload = DispatchPayload.model_validate(raw_payload)
    except ValidationError as exc:
        logger.warning(
            "dispatch: cuerpo con tipos inválidos — 400 (%s)", _error_locations(exc)
        )
        return JSONResponse({"error": "cuerpo con tipos inválidos"}, status_code=400)

    organization_id = (payload.organizationId or "").strip()
    conversation_id = (payload.conversationId or "").strip()
    if not organization_id or not conversation_id:
        logger.warning("dispatch: sin organizationId/conversationId — 400")
        return JSONResponse(
            {"error": "organizationId y conversationId son obligatorios"},
            status_code=400,
        )

    if payload.isTest:
        # Laboratorio: identidad propia por conversación de prueba para que
        # dos corridas nunca compartan historial, y sin allowlist (una
        # organización con allowlist encendida jamás podría probarse).
        identity = f"test:{conversation_id}"
    else:
        raw_identity = (payload.contact.identity or "").strip()
        if not raw_identity:
            logger.warning("dispatch %s: sin contact.identity — 400", conversation_id)
            return JSONResponse(
                {"error": "contact.identity es obligatorio (isTest=false)"},
                status_code=400,
            )
        identity = canonical_identity(raw_identity)

    inbound = messages_from_dispatch(payload, identity)

    # Dedup por wa_message_id (mismo gate que el webhook de Meta). Un mensaje
    # sin id (Laboratorio) nunca se marca — siempre se procesa, igual que en
    # webhook.py. Los ids que SÍ reclamamos aquí se recuerdan: si el turno
    # falla más abajo, hay que soltarlos (ver el except) o el reintento del
    # CRM los encuentra ya "procesados" y el mensaje se pierde en silencio.
    fresh: list[InboundMessage] = []
    claimed_ids: list[str] = []
    for msg in inbound:
        if msg.wa_message_id:
            is_new = await ctx.store.mark_processed(msg.wa_message_id)
            if not is_new:
                logger.info(
                    "dispatch: %s ya procesado — descartado", msg.wa_message_id
                )
                continue
            claimed_ids.append(msg.wa_message_id)
        fresh.append(msg)
    if not fresh:
        # Ráfaga vacía o TODOS duplicados: nada que correr, pero 200 — el CRM
        # no debe reintentar un job que ya se cumplió.
        return {"ok": True}

    # El ctx con el que va a pensar y a hablar ESTE turno: CrmClient (header
    # X-Organization-Id) y ProfileProvider cacheados de esta organización
    # (app/multiorg.py), nunca los legacy de ctx.crm/ctx.profile.
    turn_ctx = scoped_ctx(ctx, organization_id)
    lock = _lock_for(ctx, organization_id, identity)
    # Se marca justo antes del primer efecto irreversible (send_message,
    # create_booking, reschedule — ver app/turn.py y app/tools.py). Antes de
    # ese punto un fallo es seguro de reintentar desde cero; después, NO — un
    # reintento correría el LLM de nuevo y mandaría una respuesta DISTINTA.
    commit = TurnCommit()

    async def _run() -> None:
        async with lock:
            await run_turn(
                turn_ctx,
                identity,
                fresh,
                organization_id=organization_id,
                crm_conversation_id=conversation_id,
                bypass_allowlist=payload.isTest,
                strict=True,
                commit=commit,
            )

    # asyncio.shield: un TimeoutError de wait_for cancela la ESPERA de
    # dispatch() por esta tarea, NO la tarea en sí — así, si el timeout llega
    # DESPUÉS del commit, el turno sigue corriendo hasta terminar (mandar el
    # handoff, actualizar la fase…) en vez de que la cancelación interrumpa a
    # medias justo después de que el lead ya recibió una respuesta real.
    task: "asyncio.Task[None]" = asyncio.create_task(_run())
    task.add_done_callback(_log_background_outcome(organization_id, conversation_id))

    # `handled` + el `finally`: si algo escapa de los dos `except` de abajo
    # (típicamente un CancelledError — el propio request se cortó, cliente
    # desconectado, servidor reiniciando; ni TimeoutError ni Exception lo
    # capturan) el `finally` sigue corriendo igual, y decide si soltar los
    # ids con la MISMA regla (nunca si ya se comprometió algo). Sin esto, esa
    # ruta de salida dejaría los ids atascados como "procesados" para
    # siempre.
    handled = False
    try:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=DISPATCH_TIMEOUT_SECONDS)
        except TimeoutError:
            if commit.done:
                logger.error(
                    "dispatch %s/%s: tardó más de %.0f s DESPUÉS de "
                    "comprometerse (ya se le respondió/reservó al lead) — "
                    "200, el turno sigue en segundo plano, ids se quedan "
                    "reclamados",
                    organization_id,
                    conversation_id,
                    DISPATCH_TIMEOUT_SECONDS,
                )
                handled = True
                return {"ok": True}
            logger.error(
                "dispatch %s/%s: tardó más de %.0f s ANTES de comprometerse "
                "— 500, ids liberados",
                organization_id,
                conversation_id,
                DISPATCH_TIMEOUT_SECONDS,
            )
            task.cancel()
            await _release_claimed_ids(ctx, claimed_ids)
            handled = True
            return JSONResponse({"error": "el turno tardó demasiado"}, status_code=500)
        except Exception:
            if commit.done:
                logger.exception(
                    "dispatch %s/%s: falló DESPUÉS de comprometerse (ya se "
                    "le respondió/reservó al lead) — 200 para NO reintentar "
                    "y duplicar, ids se quedan reclamados",
                    organization_id,
                    conversation_id,
                )
                handled = True
                return {"ok": True}
            logger.exception(
                "dispatch %s/%s: el turno reventó ANTES de comprometerse — "
                "500 para que el CRM reintente, ids liberados",
                organization_id,
                conversation_id,
            )
            await _release_claimed_ids(ctx, claimed_ids)
            handled = True
            return JSONResponse({"error": "el turno falló"}, status_code=500)
        handled = True
        return {"ok": True}
    finally:
        if not handled and not commit.done:
            await _release_claimed_ids(ctx, claimed_ids)


def _v2_body(result: "stateless.V2Result | None") -> dict[str, Any]:
    """El sobre de respuesta del contrato v2. `None` (timeout/fallo tras
    comprometerse, ver `_dispatch_v2`): no sabemos en qué quedó el resto del
    turno en segundo plano — se contesta lo mínimo honesto, `ok` a secas."""
    if result is None:
        return {"ok": True}
    return {
        "ok": True,
        "action": result.action,
        "llm": {"source": result.llm_source, "status": result.llm_status},
        "handoff": (
            {"reason": result.handoff_reason, "applied": bool(result.handoff_applied)}
            if result.handoff_reason is not None
            else None
        ),
    }


async def _dispatch_v2(ctx: AppContext, raw_payload: dict[str, Any]) -> Any:
    """Despacho v2: mismo candado por (organización, identidad) y el mismo
    límite de 75 s que v1 — pero SIN dedup por mark_processed/release_processed:
    el CRM ya dedupea sus reintentos por `dispatchId` (mensajes idempotentes,
    PR 2A), así que aquí no hay ids que reclamar ni soltar. El turno en sí
    (app/stateless.run_turn) NUNCA toca `ctx.store`."""
    try:
        payload = stateless.DispatchPayloadV2.model_validate(raw_payload)
    except ValidationError as exc:
        logger.warning(
            "dispatch v2: cuerpo con tipos inválidos — 400 (%s)", _error_locations(exc)
        )
        return JSONResponse({"error": "cuerpo con tipos inválidos"}, status_code=400)

    organization_id = (payload.organizationId or "").strip()
    conversation_id = (payload.conversationId or "").strip()
    if not organization_id or not conversation_id:
        logger.warning("dispatch v2: sin organizationId/conversationId — 400")
        return JSONResponse(
            {"error": "organizationId y conversationId son obligatorios"},
            status_code=400,
        )

    identity = stateless.turn_identity(payload)
    lock = _lock_for(ctx, organization_id, identity)
    commit = TurnCommit()

    async def _run() -> "stateless.V2Result":
        async with lock:
            return await stateless.run_turn(
                ctx, payload, organization_id=organization_id, commit=commit
            )

    task: "asyncio.Task[stateless.V2Result]" = asyncio.create_task(_run())
    task.add_done_callback(_log_background_outcome(organization_id, conversation_id))

    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=DISPATCH_TIMEOUT_SECONDS)
    except TimeoutError:
        if commit.done:
            logger.error(
                "dispatch v2 %s/%s: tardó más de %.0f s DESPUÉS de "
                "comprometerse (ya se le respondió al lead) — 200, el turno "
                "sigue en segundo plano",
                organization_id,
                conversation_id,
                DISPATCH_TIMEOUT_SECONDS,
            )
            return _v2_body(None)
        logger.error(
            "dispatch v2 %s/%s: tardó más de %.0f s ANTES de comprometerse "
            "— 500",
            organization_id,
            conversation_id,
            DISPATCH_TIMEOUT_SECONDS,
        )
        task.cancel()
        return JSONResponse({"error": "el turno tardó demasiado"}, status_code=500)
    except Exception:
        if commit.done:
            logger.exception(
                "dispatch v2 %s/%s: falló DESPUÉS de comprometerse (ya se le "
                "respondió al lead) — 200 para NO reintentar y duplicar",
                organization_id,
                conversation_id,
            )
            return _v2_body(None)
        logger.exception(
            "dispatch v2 %s/%s: el turno reventó ANTES de comprometerse — "
            "500 para que el CRM reintente",
            organization_id,
            conversation_id,
        )
        return JSONResponse({"error": "el turno falló"}, status_code=500)
    return _v2_body(result)
