"""Estado propio del bot: modelos, contrato de almacenamiento y fake en memoria.

El acceso a datos está abstraído en el protocolo `Store` para que los unit
tests corran con `MemoryStore` (sin Postgres). La implementación real con
asyncpg vive en `app/db.py` (`PgStore`).
"""
from __future__ import annotations

import asyncio
import itertools
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.config import Settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- modelos ---


@dataclass
class Conversation:
    id: int
    wa_identity: str
    crm_conversation_id: str | None = None
    phase: str = "descubrimiento"  # descubrimiento|insight|salida|agendando|cerrada
    greeted: bool = False
    media_notice_sent: bool = False
    followup_due_at: datetime | None = None
    followup_sent: bool = False
    last_inbound_at: datetime | None = None
    #: None = camino legacy de un solo negocio (webhook de Meta + relay). Con
    #: valor, esta conversación vino del modo de despacho multi-organización:
    #: FollowupWorker/SenderWorker la usan para hablarle al CRM con el
    #: CrmClient de ESA organización, nunca el global.
    organization_id: str | None = None


@dataclass
class BotMessage:
    id: int
    conversation_id: int
    role: str  # user | assistant
    content: str
    wa_message_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class OfferedSlot:
    conversation_id: int
    start_utc: datetime
    end_utc: datetime | None
    label: str
    offered_at: datetime = field(default_factory=utcnow)


@dataclass
class RelayItem:
    id: int
    body: bytes
    signature: str | None
    attempts: int
    created_at: datetime
    next_retry_at: datetime
    delivered_at: datetime | None = None
    abandoned_at: datetime | None = None


@dataclass
class PendingSend:
    """Respuesta generada cuyo envío al CRM agotó los reintentos del turno.
    Se persiste para reintento diferido — jamás se descarta (incidente 2026-08-03)."""

    id: int
    conversation_id: int
    crm_conversation_id: str
    content: str
    attempts: int
    created_at: datetime
    next_retry_at: datetime
    delivered_at: datetime | None = None
    abandoned_at: datetime | None = None
    #: Misma semántica que Conversation.organization_id — el SenderWorker la
    #: necesita para reintentar con el CrmClient correcto.
    organization_id: str | None = None


@dataclass
class InboundMessage:
    """Mensaje entrante ya parseado del webhook de Meta."""

    wa_message_id: str | None
    identity: str
    type: str
    text: str | None = None
    referral_headline: str | None = None
    profile_name: str | None = None
    # Multimedia (spec 002): lo mínimo para procesar el contenido.
    media_id: str | None = None
    media_mime: str | None = None
    media_filename: str | None = None
    media_caption: str | None = None
    media_voice: bool = False
    location: dict[str, Any] | None = None
    contact_names: list[str] = field(default_factory=list)


class TurnCommit:
    """Marca el punto de no-retorno de UN turno de despacho.

    Antes del primer efecto irreversible hacia el CRM (mandar el mensaje,
    reservar/mover una cita) un fallo es seguro de reintentar desde cero: el
    CRM no vio nada todavía. Después, un reintento correría el LLM de nuevo
    (con una respuesta probablemente DISTINTA) y la mandaría — un segundo
    mensaje real al lead. `run_turn`/`ToolRuntime` llaman `mark()` justo antes
    de ESE primer intento (`app/turn.py._send`, `app/tools.py._book_session`);
    `app/dispatch.py` lo consulta para decidir si un fallo posterior a ese
    punto debe volverse un 5xx (reintentable) o un 200 silencioso (ya se dijo
    algo real; solo queda loguear y dejar los ids reclamados como están).

    `None` en cualquier llamador (webhook legacy, tests viejos) es un no-op:
    el camino de siempre no tiene concepto de "reintento del llamador".
    """

    def __init__(self) -> None:
        self.done = False

    def mark(self) -> None:
        self.done = True


# --------------------------------------------------------------- contrato ---


class Store(Protocol):
    """Contrato de persistencia del bot (Postgres real o memoria en tests)."""

    # dedup
    async def mark_processed(self, wa_message_id: str) -> bool:
        """True si el mensaje es nuevo (gana el INSERT); False si ya se procesó."""
        ...
    async def release_processed(self, wa_message_ids: list[str]) -> None:
        """Revierte el claim de mark_processed. Solo para el modo de despacho:
        si el turno revienta después de reclamar los ids (respuesta 5xx al
        CRM), hay que soltarlos — si no, el reintento del CRM los encuentra
        YA procesados y el mensaje se pierde en silencio (200 sin turno)."""
        ...

    # cola de relay
    async def enqueue_relay(self, body: bytes, signature: str | None) -> int: ...
    async def due_relays(self, now: datetime) -> list[RelayItem]: ...
    async def mark_relay_delivered(self, relay_id: int) -> None: ...
    async def mark_relay_abandoned(self, relay_id: int) -> None: ...
    async def reschedule_relay(
        self, relay_id: int, attempts: int, next_retry_at: datetime
    ) -> None: ...

    # conversaciones
    async def get_or_create_conversation(
        self, wa_identity: str, organization_id: str | None = None
    ) -> Conversation:
        """`organization_id` None = namespace legacy de un solo negocio. Con
        valor, la identidad se vuelve única por organización — la MISMA
        identidad en dos organizaciones son dos conversaciones distintas."""
        ...
    async def update_conversation(self, conversation_id: int, **fields: Any) -> None: ...
    async def reset_conversation(self, conversation_id: int) -> None:
        """Borra historial + slots y regresa la conversación a estado inicial
        (comando /reset de la línea de pruebas)."""
        ...

    # historial LLM
    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        wa_message_id: str | None = None,
    ) -> None: ...
    async def recent_messages(
        self, conversation_id: int, limit: int
    ) -> list[BotMessage]: ...

    # slots ofrecidos
    async def replace_offered_slots(
        self, conversation_id: int, slots: list[OfferedSlot]
    ) -> None: ...
    async def get_offered_slots(self, conversation_id: int) -> list[OfferedSlot]: ...
    async def clear_offered_slots(self, conversation_id: int) -> None: ...

    # cola de envíos pendientes (respuestas que no pudieron salir en el turno)
    async def enqueue_pending_send(
        self,
        conversation_id: int,
        crm_conversation_id: str,
        content: str,
        organization_id: str | None = None,
    ) -> int: ...
    async def due_pending_sends(self, now: datetime) -> list[PendingSend]: ...
    async def abandon_pending_sends(self, conversation_id: int) -> None: ...
    async def mark_pending_send_delivered(self, pending_id: int) -> None: ...
    async def mark_pending_send_abandoned(self, pending_id: int) -> None: ...
    async def reschedule_pending_send(
        self, pending_id: int, attempts: int, next_retry_at: datetime
    ) -> None: ...

    # seguimiento
    async def due_followups(self, now: datetime) -> list[Conversation]: ...
    async def claim_followup(self, conversation_id: int) -> bool:
        """Marca followup_sent=True atómicamente. True si ESTA llamada lo ganó."""
        ...

    # corte a modo despacho (LEGACY_ORGANIZATION_ID, ver app/main.py)
    async def adopt_legacy_rows(self, organization_id: str) -> tuple[int, int]:
        """Adopta TODAS las filas del namespace legacy (organization_id NULL)
        hacia `organization_id`, en bot_conversation y pending_send. Devuelve
        (conversaciones movidas, pending_send movidos). Idempotente: correrlo
        de nuevo tras la primera adopción no mueve nada más (ya no queda NULL
        que adoptar)."""
        ...

    async def ping(self) -> None: ...
    async def aclose(self) -> None: ...


# ------------------------------------------------------- fake en memoria ---


class MemoryStore:
    """Implementación en memoria del Store — solo para tests."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.processed: set[str] = set()
        self.relays: dict[int, RelayItem] = {}
        self.conversations: dict[int, Conversation] = {}
        # Clave (organization_id o "", wa_identity): espeja el índice único de
        # Postgres sobre (COALESCE(organization_id, ''), wa_identity).
        self._conv_by_identity: dict[tuple[str, str], int] = {}
        self.messages: list[BotMessage] = []
        self.offered: dict[int, list[OfferedSlot]] = {}
        self.pending_sends: dict[int, PendingSend] = {}

    async def mark_processed(self, wa_message_id: str) -> bool:
        if wa_message_id in self.processed:
            return False
        self.processed.add(wa_message_id)
        return True

    async def release_processed(self, wa_message_ids: list[str]) -> None:
        for wid in wa_message_ids:
            self.processed.discard(wid)

    async def enqueue_relay(self, body: bytes, signature: str | None) -> int:
        rid = next(self._ids)
        now = utcnow()
        self.relays[rid] = RelayItem(
            id=rid, body=body, signature=signature, attempts=0,
            created_at=now, next_retry_at=now,
        )
        return rid

    async def due_relays(self, now: datetime) -> list[RelayItem]:
        return [
            r for r in sorted(self.relays.values(), key=lambda r: r.id)
            if r.delivered_at is None and r.abandoned_at is None
            and r.next_retry_at <= now
        ]

    async def mark_relay_delivered(self, relay_id: int) -> None:
        self.relays[relay_id].delivered_at = utcnow()

    async def mark_relay_abandoned(self, relay_id: int) -> None:
        self.relays[relay_id].abandoned_at = utcnow()

    async def reschedule_relay(
        self, relay_id: int, attempts: int, next_retry_at: datetime
    ) -> None:
        item = self.relays[relay_id]
        item.attempts = attempts
        item.next_retry_at = next_retry_at

    async def get_or_create_conversation(
        self, wa_identity: str, organization_id: str | None = None
    ) -> Conversation:
        key = (organization_id or "", wa_identity)
        cid = self._conv_by_identity.get(key)
        if cid is not None:
            return self.conversations[cid]
        cid = next(self._ids)
        conv = Conversation(id=cid, wa_identity=wa_identity, organization_id=organization_id)
        self.conversations[cid] = conv
        self._conv_by_identity[key] = cid
        return conv

    async def update_conversation(self, conversation_id: int, **fields: Any) -> None:
        conv = self.conversations[conversation_id]
        for key, value in fields.items():
            setattr(conv, key, value)

    async def reset_conversation(self, conversation_id: int) -> None:
        self.messages = [
            m for m in self.messages if m.conversation_id != conversation_id
        ]
        self.offered.pop(conversation_id, None)
        conv = self.conversations[conversation_id]
        conv.phase = "descubrimiento"
        conv.greeted = False
        conv.media_notice_sent = False
        conv.followup_due_at = None
        conv.followup_sent = False

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        wa_message_id: str | None = None,
    ) -> None:
        self.messages.append(
            BotMessage(
                id=next(self._ids),
                conversation_id=conversation_id,
                role=role,
                content=content,
                wa_message_id=wa_message_id,
            )
        )

    async def recent_messages(
        self, conversation_id: int, limit: int
    ) -> list[BotMessage]:
        msgs = [m for m in self.messages if m.conversation_id == conversation_id]
        return msgs[-limit:]

    async def replace_offered_slots(
        self, conversation_id: int, slots: list[OfferedSlot]
    ) -> None:
        self.offered[conversation_id] = list(slots)

    async def get_offered_slots(self, conversation_id: int) -> list[OfferedSlot]:
        return list(self.offered.get(conversation_id, []))

    async def clear_offered_slots(self, conversation_id: int) -> None:
        self.offered.pop(conversation_id, None)

    async def enqueue_pending_send(
        self,
        conversation_id: int,
        crm_conversation_id: str,
        content: str,
        organization_id: str | None = None,
    ) -> int:
        pid = next(self._ids)
        now = utcnow()
        self.pending_sends[pid] = PendingSend(
            id=pid, conversation_id=conversation_id,
            crm_conversation_id=crm_conversation_id, content=content,
            attempts=0, created_at=now, next_retry_at=now,
            organization_id=organization_id,
        )
        return pid

    async def due_pending_sends(self, now: datetime) -> list[PendingSend]:
        return [
            p for p in sorted(self.pending_sends.values(), key=lambda p: p.id)
            if p.delivered_at is None and p.abandoned_at is None
            and p.next_retry_at <= now
        ]

    async def abandon_pending_sends(self, conversation_id: int) -> None:
        now = utcnow()
        for item in self.pending_sends.values():
            if (
                item.conversation_id == conversation_id
                and item.delivered_at is None
                and item.abandoned_at is None
            ):
                item.abandoned_at = now

    async def mark_pending_send_delivered(self, pending_id: int) -> None:
        self.pending_sends[pending_id].delivered_at = utcnow()

    async def mark_pending_send_abandoned(self, pending_id: int) -> None:
        self.pending_sends[pending_id].abandoned_at = utcnow()

    async def reschedule_pending_send(
        self, pending_id: int, attempts: int, next_retry_at: datetime
    ) -> None:
        item = self.pending_sends[pending_id]
        item.attempts = attempts
        item.next_retry_at = next_retry_at

    async def due_followups(self, now: datetime) -> list[Conversation]:
        return [
            c for c in self.conversations.values()
            if c.followup_due_at is not None
            and c.followup_due_at <= now
            and not c.followup_sent
            and c.phase != "cerrada"
        ]

    async def claim_followup(self, conversation_id: int) -> bool:
        conv = self.conversations[conversation_id]
        if conv.followup_sent:
            return False
        conv.followup_sent = True
        return True

    async def adopt_legacy_rows(self, organization_id: str) -> tuple[int, int]:
        # El índice por identidad va con clave (organization_id o "", wa_identity)
        # — no basta con mutar Conversation.organization_id, hay que MOVER la
        # entrada del índice o get_or_create_conversation("org_a", identity)
        # nunca encontraría la fila adoptada y crearía una NUEVA vacía.
        moved_conv = 0
        reindexed: dict[tuple[str, str], int] = {}
        for (org_key, identity), cid in self._conv_by_identity.items():
            conv = self.conversations[cid]
            if conv.organization_id is None:
                conv.organization_id = organization_id
                reindexed[(organization_id, identity)] = cid
                moved_conv += 1
            else:
                reindexed[(org_key, identity)] = cid
        self._conv_by_identity = reindexed

        moved_pending = 0
        for pending in self.pending_sends.values():
            if pending.organization_id is None:
                pending.organization_id = organization_id
                moved_pending += 1
        return moved_conv, moved_pending

    async def ping(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


# ------------------------------------------------------------- contexto ---


@dataclass
class AppContext:
    """Dependencias vivas de la app — inyectables en tests."""

    settings: "Settings"
    store: Store
    crm: Any  # CrmClient
    llm: Any  # OpenAiLlm o fake con .complete()
    profile: Any | None = None  # ProfileProvider; None en tests = perfil mínimo
    coalescer: Any | None = None
    relay_wake: asyncio.Event = field(default_factory=asyncio.Event)
    # Modo de despacho multi-organización (app/multiorg.py): un CrmClient y un
    # ProfileProvider por organización, cacheados aquí para que el turno y los
    # workers de fondo reutilicen el mismo (el perfil tiene su propio TTL —
    # sin cache por organización, una IA compartida serviría el perfil de la
    # organización A a la B en cuanto las dos pidieran perfil en la misma
    # ventana). `crm`/`profile` arriba siguen siendo el camino legacy
    # (organization_id=None) y no se tocan.
    crm_clients: dict[str, Any] = field(default_factory=dict)
    profile_providers: dict[str, Any] = field(default_factory=dict)
    # ponytail: lock por proceso — si Nea llega a correr con más de una
    # réplica hace falta un lock distribuido (p. ej. pg_advisory_xact_lock)
    # en vez de este dict en memoria. Serializa dos despachos concurrentes de
    # la MISMA (organización, identidad) — sin esto, dos turnos corriendo a
    # la vez para el mismo lead pueden contestar los dos y pisarse el
    # abandon_pending_sends el uno al otro.
    #
    # WeakValueDictionary a propósito: sin esto, cada (organización,
    # identidad) que alguna vez despachó deja un Lock vivo para siempre — una
    # fuga lenta en un proceso de larga vida con miles de leads. Con la
    # referencia débil, el Lock desaparece solo en cuanto nadie lo sigue
    # usando (el `lock` local de app/dispatch.dispatch() sale de scope al
    # terminar el request) — cero mantenimiento manual, cero "última fecha de
    # uso" que llevar.
    dispatch_locks: weakref.WeakValueDictionary[tuple[str | None, str], asyncio.Lock] = field(
        default_factory=weakref.WeakValueDictionary
    )
