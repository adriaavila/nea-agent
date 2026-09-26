"""Herramientas del LLM: update_ficha, propose_slots, book_session, route_out, handoff.

La validación es server-side: `book_session` SOLO acepta slots previamente
ofrecidos (tabla offered_slots, comparación por epoch exacto). Un fallo del
CRM dentro de una tool regresa `{"ok": false, ...}` al LLM — nunca tumba el
turno.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from app.crm import AgendaUnavailable, CrmError, SlotTaken
from app.profile import BusinessProfile
from app.state import AppContext, Conversation, OfferedSlot, TurnCommit

logger = logging.getLogger("nea.tools")

MAX_OFFERED = 3


class OfferBook(Protocol):
    """Dónde vive "lo que ya se le ofreció a este lead" — el catálogo que
    `book_session` valida por epoch exacto (ver `_book_session`). Dos
    implementaciones: una respaldada por el Store (v1/legacy, la de
    siempre) y una puramente en memoria (despacho v2, sembrada de
    `payload.offers` — sin base de datos que tocar)."""

    async def get(self) -> list[OfferedSlot]: ...
    async def replace(self, slots: list[OfferedSlot]) -> None: ...
    async def clear(self) -> None: ...


class StoreOfferBook:
    """Adaptador delgado sobre el Store — el camino de siempre (v1/legacy)."""

    def __init__(self, store: Any, conversation_id: int) -> None:
        self._store = store
        self._conversation_id = conversation_id

    async def get(self) -> list[OfferedSlot]:
        return await self._store.get_offered_slots(self._conversation_id)

    async def replace(self, slots: list[OfferedSlot]) -> None:
        await self._store.replace_offered_slots(self._conversation_id, slots)

    async def clear(self) -> None:
        await self._store.clear_offered_slots(self._conversation_id)


class MemoryOfferBook:
    """Catálogo en memoria, vivo solo durante ESTE turno (despacho v2): se
    siembra de `payload.offers` y `propose_slots` lo reemplaza — jamás toca
    el Store (Nea no guarda nada en v2)."""

    def __init__(self, initial: list[OfferedSlot] | None = None) -> None:
        self._slots = list(initial or [])

    async def get(self) -> list[OfferedSlot]:
        return list(self._slots)

    async def replace(self, slots: list[OfferedSlot]) -> None:
        self._slots = list(slots)

    async def clear(self) -> None:
        self._slots = []

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_ficha",
            "description": (
                "Guarda o actualiza la ficha del lead en el CRM (merge: solo los "
                "campos que mandes). Llámala en cuanto descubras un dato nuevo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rubro": {"type": "string"},
                    "rol": {
                        "type": "string",
                        "description": "dueno | hijo_del_dueno | empleado | otro",
                    },
                    "tamano_aprox": {"type": "string"},
                    "sistemas": {"type": "string"},
                    "dolor_principal": {"type": "string"},
                    "geo": {"type": "string"},
                    "calificado": {"type": "boolean"},
                    "resultado": {
                        "type": "string",
                        "description": "agendo | dio_diy | handoff | sin_respuesta",
                    },
                    "notas": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_slots",
            "description": (
                "Consulta la disponibilidad real de la agenda del negocio y te "
                "regresa hasta 3 horarios para ofrecer al lead (con etiqueta en "
                "español). SOLO estos horarios serán reservables después."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_session",
            "description": (
                "Reserva la cita en uno de los horarios previamente ofrecidos. "
                "start_utc debe ser EXACTAMENTE el start_utc de un slot ofrecido "
                "en esta conversación."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_utc": {
                        "type": "string",
                        "description": "ISO 8601 UTC del slot elegido, tal cual se ofreció",
                    }
                },
                "required": ["start_utc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_session",
            "description": (
                "Mueve la cita YA agendada de este lead a otro horario. Úsala "
                "solo cuando el lead pide cambiar una cita que ya tiene. Igual "
                "que book_session, start_utc debe ser de un slot que ofreciste "
                "en esta conversación: llama antes a propose_slots."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_utc": {
                        "type": "string",
                        "description": "ISO 8601 UTC del nuevo slot, tal cual se ofreció",
                    }
                },
                "required": ["start_utc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_out",
            "description": (
                "Marca al lead como no calificado (hoy). Después despídete con "
                "honestidad, compartiendo los recursos alternativos del negocio "
                "si existen, puerta abierta."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff",
            "description": (
                "Pasa la conversación a un humano del negocio y pausa la IA. Tu "
                "mensaje de despedida se envía ANTES de la pausa — salvo en el "
                "handoff por hostilidad, donde cierras sobrio sin anunciarlo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Motivo breve (p.ej. 'pidió humano', 'duda fuera del conocimiento')",
                    }
                },
            },
        },
    },
]


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _slots_from_payload(
    conversation_id: int, raw_slots: list[dict[str, Any]]
) -> list[OfferedSlot]:
    """Convierte slots del CRM ({startUtc,endUtc,label}) a OfferedSlot, tolerante."""
    out: list[OfferedSlot] = []
    for raw in raw_slots[:MAX_OFFERED]:
        start = _parse_utc(str(raw.get("startUtc") or ""))
        if start is None:
            continue
        end = _parse_utc(str(raw.get("endUtc") or "")) if raw.get("endUtc") else None
        out.append(
            OfferedSlot(
                conversation_id=conversation_id,
                start_utc=start,
                end_utc=end,
                label=str(raw.get("label") or _iso_z(start)),
            )
        )
    return out


def _slots_for_llm(slots: list[OfferedSlot]) -> list[dict[str, str]]:
    return [{"start_utc": _iso_z(s.start_utc), "label": s.label} for s in slots]


class ToolRuntime:
    """Ejecuta las tool-calls de UN turno y acumula sus efectos."""

    def __init__(
        self,
        ctx: AppContext,
        conv: Conversation | None,
        crm_conversation_id: str,
        profile: BusinessProfile | None = None,
        commit: TurnCommit | None = None,
        offers: OfferBook | None = None,
    ) -> None:
        self._ctx = ctx
        self._conv = conv
        self._crm_conv_id = crm_conversation_id
        self._profile = profile or BusinessProfile()
        self._commit = commit
        # v1/legacy (offers=None, comportamiento de SIEMPRE): respaldado por
        # el Store, requiere un `conv` real. v2 (despacho sin estado) siempre
        # pasa su propio MemoryOfferBook — `conv` puede venir None.
        conv_id = conv.id if conv is not None else 0
        self._offers: OfferBook = offers or StoreOfferBook(ctx.store, conv_id)
        # Solo para ESTAMPAR OfferedSlot.conversation_id (metadato informativo
        # de _slots_from_payload; las búsquedas reales van por self._offers).
        self._conv_id_for_slots = conv_id
        # Efectos observables por turn.py:
        self.handoff_reason: str | None = None  # se ejecuta DESPUÉS de la despedida
        self.booked = False
        self.routed_out = False
        self.proposed = False

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "update_ficha":
                return await self._update_ficha(args)
            if name == "propose_slots":
                return await self._propose_slots()
            if name == "book_session":
                return await self._book_session(args)
            if name == "reschedule_session":
                return await self._book_session(args, mover=True)
            if name == "route_out":
                return await self._route_out()
            if name == "handoff":
                return self._handoff(args)
            logger.warning("tools: herramienta desconocida %r", name)
            return {"ok": False, "error": f"herramienta desconocida: {name}"}
        except CrmError as exc:
            logger.warning("tools: %s falló contra el CRM: %s", name, exc)
            return {
                "ok": False,
                "error": "crm_error",
                "detalle": "no pude completar la acción; continúa la conversación o haz handoff",
            }

    async def _update_ficha(self, args: dict[str, Any]) -> dict[str, Any]:
        # Tolera el drift del LLM: manda lo que haya, el CRM normaliza flojo.
        ficha = {k: v for k, v in args.items() if v is not None}
        if not ficha:
            return {"ok": True, "nota": "sin campos nuevos"}
        await self._ctx.crm.put_ficha(self._crm_conv_id, ficha)
        return {"ok": True}

    async def _propose_slots(self) -> dict[str, Any]:
        try:
            payload = await self._ctx.crm.get_availability(self._crm_conv_id)
        except AgendaUnavailable:
            # El CRM de este cliente no tiene agenda: no es un fallo, es que
            # aquí no se agenda. El agente coordina con un humano.
            return {
                "ok": False,
                "error": "sin_agenda",
                "detalle": "esta instancia no agenda; usa handoff para coordinar directo",
            }
        slots = _slots_from_payload(self._conv_id_for_slots, payload["slots"])
        if not slots:
            return {
                "ok": False,
                "error": "sin_disponibilidad",
                "detalle": "no hay horarios abiertos; ofrece handoff para coordinar directo",
            }
        await self._offers.replace(slots)
        self.proposed = True
        return {
            "ok": True,
            "slots": _slots_for_llm(slots),
            # El CRM ya registró estos horarios como "lo ofrecido"; el catálogo
            # es más ancho que el menú a propósito (ver crm.get_availability).
            "dias_con_agenda": payload["dias_con_agenda"],
            "instrucciones": (
                "ofrece MÁXIMO 3 con su etiqueta tal cual. Si el lead pide otro "
                "día, mira dias_con_agenda: un día que no esté ahí, el negocio "
                "lo tiene cerrado — dilo, no lo prometas."
            ),
        }

    async def _book_session(
        self, args: dict[str, Any], *, mover: bool = False
    ) -> dict[str, Any]:
        """
        Reserva (`mover=False`) o mueve (`mover=True`) la cita.

        Es el mismo camino a propósito: las dos operaciones tienen las mismas
        reglas —solo un horario ofrecido, y el hueco tiene que seguir libre— y
        los mismos 409. Duplicarlo sería duplicar también los errores.
        """
        wanted = _parse_utc(str(args.get("start_utc") or ""))
        offered = await self._offers.get()
        if wanted is None:
            return {
                "ok": False,
                "error": "start_utc_invalido",
                "slots_ofrecidos": _slots_for_llm(offered),
            }
        # Validación server-side por epoch exacto: solo lo ofrecido es reservable.
        chosen = next(
            (s for s in offered if int(s.start_utc.timestamp()) == int(wanted.timestamp())),
            None,
        )
        if chosen is None:
            logger.info(
                "tools: book_session rechazado — %s no está entre los ofrecidos",
                args.get("start_utc"),
            )
            return {
                "ok": False,
                "error": "slot_no_ofrecido",
                "detalle": "solo puedes reservar un horario que ya ofreciste",
                "slots_ofrecidos": _slots_for_llm(offered),
            }
        # Marca el commit ANTES del intento, no después de un 201/200: incluso
        # un intento que "falla" de nuestro lado (timeout, red) puede haber
        # llegado al CRM, y reintentar el TURNO completo desde cero podría
        # generar una segunda reserva o un mensaje de confirmación duplicado.
        if self._commit is not None:
            self._commit.mark()
        try:
            result = await (
                self._ctx.crm.reschedule_booking(
                    self._crm_conv_id, _iso_z(chosen.start_utc)
                )
                if mover
                else self._ctx.crm.create_booking(
                    self._crm_conv_id, _iso_z(chosen.start_utc)
                )
            )
        except AgendaUnavailable:
            return {
                "ok": False,
                "error": "sin_agenda",
                "detalle": "esta instancia no agenda; usa handoff para coordinar directo",
            }
        except SlotTaken as exc:
            # Se ocupó entre oferta y elección, o el CRM no reconoce el
            # instante como ofrecido. Misma salida: alternativas frescas del
            # propio CRM, nunca discutir con el lead.
            fresh = _slots_from_payload(self._conv_id_for_slots, exc.slots)
            await self._offers.replace(fresh)
            return {
                "ok": False,
                "error": exc.code,
                "detalle": (
                    "ese horario se acaba de ocupar; discúlpate breve y ofrece estas alternativas"
                    if exc.code == "slot_taken"
                    else "ese horario no está entre los que ofreciste; ofrece estos"
                ),
                "slots": _slots_for_llm(fresh),
            }
        await self._offers.clear()
        self.booked = True
        if not mover:
            try:
                await self._ctx.crm.put_ficha(
                    self._crm_conv_id, {"calificado": True, "resultado": "agendo"}
                )
            except CrmError as exc:  # best-effort: la cita ya existe
                logger.warning("tools: no pude actualizar ficha tras booking: %s", exc)
        return {
            "ok": True,
            "label": result.get("label") or chosen.label,
            # `meetingLink` es el nombre del CRM desde el motor de agenda
            # universal; `zoomJoinUrl` era del conector único de antes.
            "meeting_url": result.get("meetingLink") or result.get("zoomJoinUrl"),
            "link_pendiente": bool(result.get("linkPending")),
            "movida": mover,
            "instrucciones": (
                "confirma día y hora de la cita y menciona lo que el negocio "
                "pida para llegar preparado. Si hay meeting_url, compártelo. Si "
                "link_pendiente es true, la cita EXISTE pero el enlace todavía "
                "no: di que se lo mandas en un momento — nunca inventes uno."
            ),
        }

    async def _route_out(self) -> dict[str, Any]:
        # "dio_diy" es el valor del enum `resultado` en el gateway del CRM
        # (006); el nombre de la herramienta es genérico, el cable no cambia.
        await self._ctx.crm.put_ficha(
            self._crm_conv_id, {"calificado": False, "resultado": "dio_diy"}
        )
        self.routed_out = True
        out: dict[str, Any] = {"ok": True}
        if self._profile.resources:
            out["recursos"] = self._profile.resources
            out["instrucciones"] = "comparte estos recursos al despedirte, puerta abierta"
        return out

    def _handoff(self, args: dict[str, Any]) -> dict[str, Any]:
        self.handoff_reason = str(args.get("reason") or "lead_request")
        return {
            "ok": True,
            "nota": (
                "el pase a humano se ejecutará después de tu mensaje de despedida"
            ),
        }
