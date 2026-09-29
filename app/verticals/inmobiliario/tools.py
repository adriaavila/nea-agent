"""Herramientas del vertical inmobiliario (Rei CRM).

Nuevas de verdad: `guardar_requerimiento`, `ver_propiedad`, `enviar_ficha`.
`propose_slots` y `handoff` se REUTILIZAN tal cual del chasis por defecto
(mismo esquema exportado por `app.tools`, mismo código en
`app.tools.ToolRuntime.execute` — ese código no sabe que este vertical
existe). `book_session`/`reschedule_session` TAMBIÉN son el mismo código:
`ToolRuntime._book_session` ya lee `property_id` de los args si `allow_
property_id=True` (ver `app/tools.py`) — aquí solo se anuncia ese parámetro
opcional en el esquema.

`update_ficha` (campos B2B de allok) y `route_out` NO existen en este
vertical: ver `DISABLED_TOOLS`, que `ToolRuntime` usa para rechazarlas antes
de ejecutarlas aunque el modelo alucine el nombre.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from app.crm import (
    CrmConflict,
    CrmError,
    CrmPaymentRequired,
    RealtyInvalidRequirement,
    RealtyPropertyNotFound,
    RealtyPropertyUnavailable,
    RealtyVerticalDisabled,
)
from app.tools import TOOL_SCHEMAS as _DEFAULT_TOOL_SCHEMAS
from app.tools import ExtraToolHandler
from app.verticals.inmobiliario import catalog

logger = logging.getLogger("nea.verticals.inmobiliario")

#: Ver el docstring del módulo. `ToolRuntime.execute` (app/tools.py) rechaza
#: estas ANTES de tocar el CRM — no basta con no anunciarlas en el esquema.
DISABLED_TOOLS = frozenset({"update_ficha", "route_out"})

#: Reintentos de `enviar_ficha` ante un fallo transitorio del CRM — mismo
#: número que `app.stateless.SEND_ATTEMPTS` (es un SEND más, mismo
#: presupuesto), duplicado aquí para no importar de `app.stateless` y evitar
#: cualquier acoplamiento circular o de merge con ese módulo.
_FICHA_SEND_ATTEMPTS = 3


def _default_schema(name: str) -> dict[str, Any]:
    return next(t for t in _DEFAULT_TOOL_SCHEMAS if t["function"]["name"] == name)


def _with_property_id(schema: dict[str, Any], *, extra_desc: str) -> dict[str, Any]:
    """Copia el esquema por defecto de book_session/reschedule_session y le
    agrega `property_id` opcional — el resto (start_utc, validación
    server-side por epoch exacto) es EXACTAMENTE el mismo código."""
    out = copy.deepcopy(schema)
    out["function"]["description"] = f"{out['function']['description']} {extra_desc}"
    out["function"]["parameters"]["properties"]["property_id"] = {
        "type": "string",
        "description": "Id de la propiedad de la visita, si aplica (opcional).",
    }
    return out


REALTY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "guardar_requerimiento",
            "description": (
                "Guarda o actualiza lo que el comprador/arrendatario busca "
                "(merge: solo los campos que mandes). Llámala en cuanto sepas "
                "CUALQUIER dato nuevo, no esperes a tener todo. Te regresa las "
                "candidatas más frescas. Si un valor no es válido, el resultado "
                "trae el campo exacto para que lo corrijas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": list(catalog.OPERATIONS),
                        "description": (
                            'el lead puede decir "alquiler": ese valor es '
                            '"renta" en esta tool'
                        ),
                    },
                    "budgetMin": {"type": "number"},
                    "budgetMax": {"type": "number"},
                    "currency": {"type": "string", "enum": list(catalog.CURRENCIES)},
                    "zones": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": list(catalog.PROPERTY_KINDS)},
                    "minBedrooms": {"type": "integer"},
                    "minBathrooms": {
                        "type": "number",
                        "description": "medios baños son válidos, p.ej. 1.5",
                    },
                    "amenities": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(catalog.AMENITIES)},
                    },
                    "paymentMethod": {
                        "type": "string",
                        "enum": list(catalog.PAYMENT_METHODS),
                    },
                    "needsGuarantor": {"type": "boolean"},
                    "urgency": {"type": "string", "enum": list(catalog.URGENCIES)},
                    "notes": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ver_propiedad",
            "description": (
                "Trae el detalle completo de UNA propiedad (para responder algo "
                "puntual que no tengas, incluida su dirección). Usa el id de "
                "una candidata que ya viste."
            ),
            "parameters": {
                "type": "object",
                "properties": {"property_id": {"type": "string"}},
                "required": ["property_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_ficha",
            "description": (
                "Manda la ficha de la propiedad (foto + datos) por WhatsApp. "
                "Solo cuando el lead mostró interés real en ESA propiedad. "
                "Llamarla dos veces para la misma propiedad en la misma "
                "conversación nunca la reenvía: te regresa el mismo resultado."
            ),
            "parameters": {
                "type": "object",
                "properties": {"property_id": {"type": "string"}},
                "required": ["property_id"],
            },
        },
    },
    _default_schema("propose_slots"),
    _with_property_id(
        _default_schema("book_session"),
        extra_desc="Si la cita es una visita a una propiedad, manda su property_id.",
    ),
    _with_property_id(
        _default_schema("reschedule_session"),
        extra_desc="Si la visita es a una propiedad, manda su property_id.",
    ),
    _default_schema("handoff"),
]


def _vertical_disabled_result() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "vertical_disabled",
        "detalle": "este negocio no tiene el módulo inmobiliario activo, haz handoff",
    }


async def _guardar_requerimiento(
    crm: Any, conversation_id: str, args: dict[str, Any]
) -> dict[str, Any]:
    requirement = {k: v for k, v in args.items() if v is not None}
    if not requirement:
        return {"ok": True, "nota": "sin campos nuevos"}
    try:
        data = await crm.put_realty_requirement(conversation_id, requirement)
    except RealtyVerticalDisabled:
        return _vertical_disabled_result()
    except RealtyInvalidRequirement as exc:
        return {
            "ok": False,
            "error": "invalid_requirement",
            "field": exc.field,
            "detalle": (
                f"el valor que mandaste para {exc.field} no es válido, usa "
                "uno de los permitidos en el esquema de esta herramienta"
                if exc.field
                else "hay un valor inválido en el requerimiento, revisa los permitidos"
            ),
        }
    return {
        "ok": True,
        "requirement": data.get("requirement") or requirement,
        "candidates": list(data.get("candidates") or []),
        "instrucciones": (
            "propón MÁXIMO 3 de estas candidatas, con una razón corta cada "
            "una; si falta algo del requerimiento, sigue preguntando de a uno"
        ),
    }


async def _ver_propiedad(crm: Any, args: dict[str, Any]) -> dict[str, Any]:
    property_id = str(args.get("property_id") or "").strip()
    if not property_id:
        return {"ok": False, "error": "property_id_requerido"}
    try:
        data = await crm.get_realty_property(property_id)
    except RealtyVerticalDisabled:
        return _vertical_disabled_result()
    except RealtyPropertyNotFound:
        return {
            "ok": False,
            "error": "property_not_found",
            "detalle": "esa propiedad ya no existe, usa una de las candidatas actuales",
        }
    return {"ok": True, "propiedad": data}


async def _post_ficha_con_reintentos(
    crm: Any, conversation_id: str, property_id: str, dispatch_id: str
) -> dict[str, Any]:
    """Intenta el SEND con backoff ante fallos transitorios. Nunca deja
    escapar una excepción: `ToolRuntime.execute` atraparía cualquier
    CrmError de una tool con el mismo mensaje genérico igual (a diferencia
    de `app.stateless._send`, que corre FUERA de ese try/except), así que
    regresar aquí un fallo ya informativo es estrictamente mejor."""
    for attempt in range(_FICHA_SEND_ATTEMPTS):
        try:
            data = await crm.post_realty_ficha(
                conversation_id, property_id, dispatch_id=dispatch_id
            )
        except CrmConflict as exc:
            if exc.code == "send_in_progress" and attempt < _FICHA_SEND_ATTEMPTS - 1:
                logger.warning(
                    "enviar_ficha: send_in_progress (intento %d), reintento",
                    attempt + 1,
                )
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            if exc.code == "send_in_progress":
                return {
                    "ok": False,
                    "error": "send_in_progress",
                    "detalle": "la ficha está en proceso, no la reenvíes",
                }
            logger.info("enviar_ficha bloqueada por el CRM (%s), sin enviar", exc.code)
            return {
                "ok": False,
                "error": exc.code,
                "detalle": "no se pudo enviar la ficha ahora mismo",
            }
        except CrmPaymentRequired:
            logger.warning("enviar_ficha: 402 (límite del plan), sin enviar")
            return {"ok": False, "error": "payment_required"}
        except RealtyVerticalDisabled:
            return _vertical_disabled_result()
        except RealtyPropertyNotFound:
            return {
                "ok": False,
                "error": "property_not_found",
                "detalle": "esa propiedad ya no existe, usa una de las candidatas actuales",
            }
        except RealtyPropertyUnavailable:
            return {
                "ok": False,
                "error": "property_unavailable",
                "detalle": "esa propiedad ya no está disponible, ofrece otra candidata",
            }
        except CrmError as exc:
            logger.warning("enviar_ficha falló (intento %d): %s", attempt + 1, exc)
            if attempt < _FICHA_SEND_ATTEMPTS - 1:
                await asyncio.sleep(2.0**attempt)
                continue
            return {
                "ok": False,
                "error": "crm_error",
                "detalle": "no pude mandar la ficha, continúa la conversación o haz handoff",
            }
        else:
            # NUNCA marca ningún commit aquí (solo `_send` compromete, ver el
            # contrato): si la respuesta de TEXTO de este turno falla
            # después, el despacho completo debe poder forzar un 500/retry,
            # y en ESE retry el CRM dedupea esta ficha por (dispatchId,
            # propertyId) — nunca por un contador de Nea.
            return {
                "ok": True,
                "enviado": True,
                "duplicado": bool(data.get("duplicate")),
                "foto_enviada": bool(data.get("photoSent")),
                "instrucciones": (
                    "la ficha le llega al lead ANTES que tu respuesta de "
                    "texto: no la anuncies ni describas la foto que ya vio. "
                    "Si foto_enviada es false, se mandó solo como texto sin "
                    "foto: no digas que mandaste una imagen ni la reenvíes."
                ),
            }
    raise AssertionError("inalcanzable: cada rama del for anterior retorna o continúa")


async def _enviar_ficha(
    crm: Any,
    conversation_id: str,
    dispatch_id: str,
    ficha_cache: dict[str, dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    property_id = str(args.get("property_id") or "").strip()
    if not property_id:
        return {"ok": False, "error": "property_id_requerido"}
    if property_id in ficha_cache:
        # Memoizado por property_id DENTRO de este turno: una segunda
        # tool-call para la MISMA propiedad (el modelo insiste, o el
        # despacho reordena sus tool-calls) nunca dispara un segundo POST,
        # sin importar en qué orden se haya llamado la primera vez.
        return ficha_cache[property_id]
    result = await _post_ficha_con_reintentos(
        crm, conversation_id, property_id, dispatch_id
    )
    ficha_cache[property_id] = result
    return result


def build_realty_tools(
    *,
    crm: Any,
    conversation_id: str,
    dispatch_id: str,
) -> dict[str, ExtraToolHandler]:
    """Los handlers de las tools NUEVAS de este vertical, listos para
    `ToolRuntime(..., extra_tools=...)`. `propose_slots`, `book_session`,
    `reschedule_session` y `handoff` NO están aquí: los ejecuta el código
    genérico de siempre en `app.tools.ToolRuntime` (book_session/
    reschedule_session ya leen `property_id` de los args si viene, ver ahí).

    `ficha_cache` (un dict propio por turno, `{property_id: resultado}`)
    memoiza `enviar_ficha`: dos tool-calls para la MISMA propiedad en el
    MISMO turno hacen UN solo POST. Un RETRY del despacho completo (mismo
    `dispatchId`, turno nuevo, `ficha_cache` nuevo) sí vuelve a intentar el
    POST — pero el CRM lo reconoce como duplicado comparando (dispatchId,
    propertyId) de su lado, así que el lead nunca recibe la ficha dos veces.
    Eso es también por qué esta tool NUNCA marca ningún commit por su cuenta
    (ver `_post_ficha_con_reintentos`): dejar que una respuesta de texto
    fallida siga pudiendo forzar un 5xx/retry del despacho es justamente lo
    que hace segura esa deduplicación del lado del CRM."""
    ficha_cache: dict[str, dict[str, Any]] = {}

    async def _guardar(args: dict[str, Any]) -> dict[str, Any]:
        return await _guardar_requerimiento(crm, conversation_id, args)

    async def _ver(args: dict[str, Any]) -> dict[str, Any]:
        return await _ver_propiedad(crm, args)

    async def _enviar(args: dict[str, Any]) -> dict[str, Any]:
        return await _enviar_ficha(crm, conversation_id, dispatch_id, ficha_cache, args)

    return {
        "guardar_requerimiento": _guardar,
        "ver_propiedad": _ver,
        "enviar_ficha": _enviar,
    }
