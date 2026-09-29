"""Cliente httpx del API de servicio del CRM (bot gateway de vocero-crm).

Endpoints:
  GET  /api/bot/profile                               → agent profile + KB (404 = sin perfil)
  GET  /api/bot/context?waIdentity=...  o  ?conversationId=...  (modo despacho)
  POST /api/bot/messages   {conversationId, text}   → 409 ai_paused|window_closed
  PUT  /api/bot/ficha      {conversationId, ficha}
  POST /api/bot/handoff    {conversationId, reason}
  GET  /api/bot/availability?conversationId=...&limit=&perDay=&days=
       → {slots, diasConAgenda}. REGISTRA la oferta: sin esta llamada el CRM
         rechaza cualquier reserva. 404 = esta instancia no tiene agenda.
  POST /api/bot/bookings   {conversationId, startUtc} → 201; 409 slot_taken |
       slot_not_offered, con `slots` frescos al lado del sobre de error
  PATCH /api/bot/bookings  {conversationId, startUtc} → 200, mueve la cita
  GET  /api/bot/media/{mediaId}                       → binario + content-type
  POST /api/bot/reset      {conversationId}           → reinicio de pruebas (002)
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("nea.crm")


class CrmError(Exception):
    """Fallo genérico hablando con el CRM (red, 5xx, 401...)."""


class CrmConflict(CrmError):
    """409 tipado del CRM: ai_paused | window_closed | slot_taken."""

    def __init__(self, code: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.payload = payload or {}


class AgendaUnavailable(CrmError):
    """Esta instancia del CRM no tiene agenda encendida (bandera AGENDA)."""


class CrmPaymentRequired(CrmError):
    """402 de /api/bot/messages (despacho v2): límite del plan del negocio.
    A diferencia de un 409 (ai_paused/window_closed), esto no es un estado de
    la conversación — es final igual: no se reintenta, no hay nada que un
    reintento del turno pudiera arreglar."""

    def __init__(self, code: str = "payment_required") -> None:
        super().__init__(code)
        self.code = code


class SlotTaken(CrmConflict):
    """El slot se ocupó entre oferta y confirmación; trae alternativas frescas."""

    def __init__(
        self, payload: dict[str, Any] | None = None, code: str = "slot_taken"
    ) -> None:
        super().__init__(code, payload)
        self.slots: list[dict[str, Any]] = list((payload or {}).get("slots") or [])


# --- Vertical inmobiliario (Rei CRM) -----------------------------------------
#
# Ver app/verticals/inmobiliario/ para el chasis y las tools que usan estos
# errores. Cualquier ruta `/api/bot/realty/*` (y `bookings`/`availability` con
# un `propertyId` inválido) puede darlos; viven aquí y no en el paquete del
# vertical porque son forma del CONTRATO con el CRM, igual que
# AgendaUnavailable/SlotTaken arriba.


class RealtyVerticalDisabled(CrmError):
    """404 `vertical_disabled`: la organización no tiene el vertical
    inmobiliario activo. Cualquier ruta `/api/bot/realty/*` lo devuelve para
    una organización que no sea de bienes raíces."""


class RealtyPropertyNotFound(CrmError):
    """La propiedad no existe: 404 `property_not_found` (ficha/detalle) o 422
    `property_not_found` (bookings/availability con un `propertyId` inválido)."""


class RealtyPropertyUnavailable(CrmError):
    """422 `property_unavailable`: la propiedad existe pero ya no se puede
    ofrecer ni enviar (vendida, pausada, etc.)."""


class RealtyInvalidRequirement(CrmError):
    """422 `invalid_requirement` de `PUT /api/bot/realty/requirement`: un
    valor no está en el catálogo cerrado (`src/lib/realty/catalog.ts` del
    CRM — operation/kind/currency/amenities/paymentMethod/urgency). Trae el
    campo señalado (sobre PLANO `{error, field}`, no el `{error:{code}}`
    anidado de los demás 409/422 de aquí) para que el modelo lo corrija sin
    adivinar cuál."""

    def __init__(self, field: str | None = None) -> None:
        super().__init__(f"invalid_requirement: {field}")
        self.field = field


def _conflict_code(response: httpx.Response) -> str:
    """
    El código del 409, venga en el sobre plano o en el anidado.

    Vocero pasó de `{"code": "..."}` a `{"error": {"code": "..."}}` con `slots`
    de hermano. Se leen los dos: un cliente que solo entendiera uno se queda
    sin la rama de re-oferta y el lead ve al agente insistir con un horario
    ocupado.
    """
    try:
        payload = response.json()
    except Exception:
        return "conflict"
    if not isinstance(payload, dict):
        return "conflict"
    anidado = payload.get("error")
    if isinstance(anidado, dict) and anidado.get("code"):
        return str(anidado["code"])
    return str(payload.get("code") or "conflict")


# Catálogo cerrado del CRM para handoff.reason (006). El LLM escribe motivos
# libres ("pidió humano", "duda técnica") — aquí se normalizan SIEMPRE:
# certificación 002 cazó en vivo que un reason fuera de catálogo era 422 y el
# handoff se perdía con la IA aún activa.
HANDOFF_REASONS = frozenset({"cliente", "modelo", "error", "ventana", "hostilidad"})


def canonical_handoff_reason(reason: str | None) -> str:
    r = (reason or "").strip().lower()
    if r in HANDOFF_REASONS:
        return r
    if "hostil" in r or "groser" in r or "insult" in r:
        return "hostilidad"
    if any(k in r for k in ("humano", "persona", "pidi", "lead_request", "hablar")):
        return "cliente"
    if "error" in r:
        return "error"
    return "modelo"


class CrmClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
        organization_id: str | None = None,
    ) -> None:
        """`organization_id` arma un cliente para el modo de despacho
        multi-organización: cada llamada a /api/bot/* lleva el header
        `X-Organization-Id` además del `X-API-Key` de siempre — es lo que le
        deja al CRM enrutar la petición a la organización correcta. `None`
        (default) es el camino legacy de un solo negocio: sin ese header."""
        self.organization_id = organization_id
        headers = {"X-API-Key": api_key}
        if organization_id:
            headers["X-Organization-Id"] = organization_id
        self._http = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise CrmError(f"error de red hacia el CRM: {exc}") from exc

    async def get_context(
        self, wa_identity: str | None = None, *, conversation_id: str | None = None
    ) -> dict[str, Any] | None:
        """Contexto conversacional; None si el CRM aún no conoce la identidad
        o la conversación (404).

        Dos caminos: `waIdentity` (webhook clásico) o `conversationId` (modo
        despacho — el CRM ya conoce la conversación de antemano, y las
        conversaciones de prueba del Laboratorio no tienen una identidad real
        que buscar; `conversation_id`, si se pasa, siempre gana).
        """
        if conversation_id is not None:
            params = {"conversationId": conversation_id}
        elif wa_identity is not None:
            params = {"waIdentity": wa_identity}
        else:
            raise ValueError("get_context necesita wa_identity o conversation_id")
        resp = await self._request("GET", "/api/bot/context", params=params)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise CrmError(f"context devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def get_profile(self) -> dict[str, Any] | None:
        """Agent profile + knowledge base del negocio; None si el CRM no lo
        expone todavía (404) — el bot cae al brief local (app/profile.py)."""
        resp = await self._request("GET", "/api/bot/profile")
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise CrmError(f"profile devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def send_message(
        self,
        conversation_id: str,
        text: str,
        *,
        dispatch_id: str | None = None,
        seq: int | None = None,
    ) -> dict[str, Any]:
        """`dispatch_id`/`seq` (despacho v2, PR 2A): el CRM arma con ellos un
        id de mensaje DETERMINISTA y reserva el envío antes de tocar Graph —
        reintentar con el MISMO par es seguro, nunca duplica el mensaje real.
        `None` (default, camino v1/legacy) no manda esos campos — cuerpo
        idéntico al de siempre."""
        body: dict[str, Any] = {"conversationId": conversation_id, "text": text}
        if dispatch_id is not None:
            body["dispatchId"] = dispatch_id
        if seq is not None:
            body["seq"] = seq
        resp = await self._request("POST", "/api/bot/messages", json=body)
        if resp.status_code == 409:
            raise CrmConflict(_conflict_code(resp))
        if resp.status_code == 402:
            raise CrmPaymentRequired(_conflict_code(resp))
        if resp.status_code != 200:
            raise CrmError(f"messages devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def post_transcript(self, message_id: str, text: str) -> None:
        """Escribe de vuelta la transcripción de un audio (despacho v2, antes
        de llamar al LLM): `POST /api/bot/messages/{messageId}/transcript`.
        Gana la primera escritura del lado del CRM; aquí no hay reintentos —
        el llamador (app/stateless.py) lo trata como best-effort."""
        resp = await self._request(
            "POST",
            f"/api/bot/messages/{message_id}/transcript",
            json={"text": text},
        )
        if resp.status_code != 200:
            raise CrmError(f"transcript devolvió {resp.status_code}")

    async def put_ficha(
        self, conversation_id: str, ficha: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await self._request(
            "PUT",
            "/api/bot/ficha",
            json={"conversationId": conversation_id, "ficha": ficha},
        )
        if resp.status_code != 200:
            raise CrmError(f"ficha devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def post_handoff(self, conversation_id: str, reason: str | None = None) -> None:
        resp = await self._request(
            "POST",
            "/api/bot/handoff",
            json={
                "conversationId": conversation_id,
                "reason": canonical_handoff_reason(reason),
            },
        )
        if resp.status_code != 200:
            raise CrmError(f"handoff devolvió {resp.status_code}")

    async def get_availability(
        self,
        conversation_id: str,
        *,
        limit: int = 12,
        per_day: int = 3,
        days: int = 5,
    ) -> dict[str, Any]:
        """
        Los horarios que se le van a ofrecer al lead, Y su registro en el CRM.

        `conversationId` es obligatorio y no es burocracia: el CRM guarda lo que
        devuelve como "lo ofrecido a esta conversación", y `POST /bookings`
        rechaza cualquier instante que no esté en esa lista. Sin pasar por aquí,
        reservar es imposible.

        Se piden MÁS de los que se enseñan (12 reservables, 3 por día): si el
        lead pide "mejor el jueves", el agente tiene alternativas legítimas que
        aceptar en vez de tener que re-ofrecer.
        """
        resp = await self._request(
            "GET",
            "/api/bot/availability",
            params={
                "conversationId": conversation_id,
                "limit": limit,
                "perDay": per_day,
                "days": days,
            },
        )
        if resp.status_code == 404:
            raise AgendaUnavailable("la instancia no tiene agenda encendida")
        if resp.status_code != 200:
            raise CrmError(f"availability devolvió {resp.status_code}")
        data = resp.json()
        return {
            "slots": list(data.get("slots") or []),
            # Los días que NO están aquí no tienen agenda. Es lo que evita que
            # el modelo prometa un jueves que el negocio tiene cerrado.
            "dias_con_agenda": list(data.get("diasConAgenda") or []),
        }

    async def create_booking(
        self, conversation_id: str, start_utc: str, *, property_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"conversationId": conversation_id, "startUtc": start_utc}
        if property_id is not None:
            # Vertical inmobiliario: la visita es a ESTA propiedad. Ausente
            # (default) el cuerpo queda IDÉNTICO al de siempre (allok/B2B).
            body["propertyId"] = property_id
        resp = await self._request("POST", "/api/bot/bookings", json=body)
        return self._booking_response(resp)

    async def reschedule_booking(
        self, conversation_id: str, start_utc: str, *, property_id: str | None = None
    ) -> dict[str, Any]:
        """
        Mueve la cita viva de esta conversación a otro horario ofrecido.

        PATCH y 200, no POST y 201: mover no crea nada. Los mismos 409 que
        crear, así que comparte el manejo.
        """
        body: dict[str, Any] = {"conversationId": conversation_id, "startUtc": start_utc}
        if property_id is not None:
            body["propertyId"] = property_id
        resp = await self._request("PATCH", "/api/bot/bookings", json=body)
        return self._booking_response(resp)

    @staticmethod
    def _booking_response(resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code == 404:
            raise AgendaUnavailable("la instancia no tiene agenda encendida")
        if resp.status_code == 409:
            payload: dict[str, Any] = {}
            try:
                payload = resp.json()
            except Exception:
                pass
            code = _conflict_code(resp)
            # `slot_not_offered` se trata como `slot_taken`: en los dos casos la
            # salida es la misma —re-ofrecer con datos reales— y el CRM manda
            # los horarios frescos en el mismo sobre.
            if code in ("slot_taken", "slot_not_offered"):
                raise SlotTaken(payload, code=code)
            raise CrmConflict(code, payload)
        if resp.status_code == 422:
            # Vertical inmobiliario: `propertyId` inválido en la reserva. Un
            # 422 fuera de ese caso (nunca lo manda el contrato de siempre)
            # sigue siendo un CrmError genérico.
            code = _conflict_code(resp)
            if code == "property_not_found":
                raise RealtyPropertyNotFound(code)
            raise CrmError(f"bookings devolvió 422 ({code})")
        # El CRM real responde 201 Created (REST); los mocks viejos daban 200.
        if resp.status_code not in (200, 201):
            raise CrmError(f"bookings devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    # --- Vertical inmobiliario (Rei CRM) -------------------------------------

    async def get_realty_property(self, property_id: str) -> dict[str, Any]:
        """Detalle completo de una propiedad (tool `ver_propiedad`, ver
        app/verticals/inmobiliario/tools.py)."""
        resp = await self._request(
            "GET", "/api/bot/realty/property", params={"id": property_id}
        )
        if resp.status_code == 404:
            code = _conflict_code(resp)
            if code == "vertical_disabled":
                raise RealtyVerticalDisabled(code)
            raise RealtyPropertyNotFound(code or "property_not_found")
        if resp.status_code != 200:
            raise CrmError(f"realty/property devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def put_realty_requirement(
        self, conversation_id: str, requirement: dict[str, Any]
    ) -> dict[str, Any]:
        """Guarda (merge parcial) lo que el comprador/arrendatario busca y
        regresa `{requirement, candidates}` ya frescos (tool
        `guardar_requerimiento`)."""
        resp = await self._request(
            "PUT",
            "/api/bot/realty/requirement",
            json={"conversationId": conversation_id, "requirement": requirement},
        )
        if resp.status_code == 404:
            raise RealtyVerticalDisabled(_conflict_code(resp) or "vertical_disabled")
        if resp.status_code == 422:
            # Sobre PLANO `{error: "invalid_requirement", field}` — a
            # propósito distinto del `{error: {code}}` anidado de los demás
            # 409/422 de este archivo (ver RealtyInvalidRequirement).
            payload: dict[str, Any] = {}
            try:
                payload = resp.json()
            except Exception:
                pass
            raise RealtyInvalidRequirement(
                payload.get("field") if isinstance(payload, dict) else None
            )
        if resp.status_code != 200:
            raise CrmError(f"realty/requirement devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def post_realty_ficha(
        self,
        conversation_id: str,
        property_id: str,
        *,
        dispatch_id: str,
    ) -> dict[str, Any]:
        """Manda la ficha (foto + datos) de una propiedad — un SEND más,
        idempotente por (dispatchId, propertyId) del lado del CRM (tool
        `enviar_ficha`). A diferencia de `send_message`, NO lleva `seq`: el
        vertical inmobiliario memoiza por `property_id` DENTRO del turno
        (ver app/verticals/inmobiliario/tools.py), y un RETRY del despacho
        completo repite el MISMO `dispatchId` — el CRM dedupea por el par
        (dispatchId, propertyId), nunca por un contador de Nea."""
        resp = await self._request(
            "POST",
            "/api/bot/realty/ficha",
            json={
                "conversationId": conversation_id,
                "propertyId": property_id,
                "dispatchId": dispatch_id,
            },
        )
        if resp.status_code == 409:
            raise CrmConflict(_conflict_code(resp))
        if resp.status_code == 402:
            raise CrmPaymentRequired(_conflict_code(resp))
        if resp.status_code == 404:
            code = _conflict_code(resp)
            if code == "vertical_disabled":
                raise RealtyVerticalDisabled(code)
            raise RealtyPropertyNotFound(code or "property_not_found")
        if resp.status_code == 422:
            raise RealtyPropertyUnavailable(_conflict_code(resp) or "property_unavailable")
        if resp.status_code != 200:
            raise CrmError(f"realty/ficha devolvió {resp.status_code}")
        data: dict[str, Any] = resp.json()
        return data

    async def post_typing(self, conversation_id: str) -> None:
        """Marca leído + "escribiendo…" (007). Best-effort: sin reintentos."""
        resp = await self._request(
            "POST",
            "/api/bot/typing",
            json={"conversationId": conversation_id},
            timeout=8.0,
        )
        if resp.status_code != 200:
            raise CrmError(f"typing devolvió {resp.status_code}")

    async def post_reset(
        self,
        conversation_id: str,
        *,
        notice: str | None = None,
        dispatch_id: str | None = None,
    ) -> None:
        """Reinicio de pruebas (spec 002): ficha limpia + IA reactivada + etapa
        al inicio en el CRM. Solo lo dispara el comando /reset de la allowlist.

        `notice`/`dispatch_id` (despacho v2): sin `bot_conversation` local que
        reiniciar, Nea no tiene de dónde mandar el aviso ella misma — se lo
        pasa al CRM para que LO envíe él. `None` (default, v1/legacy) deja el
        cuerpo idéntico al de siempre."""
        body: dict[str, Any] = {"conversationId": conversation_id}
        if notice is not None:
            body["notice"] = notice
        if dispatch_id is not None:
            body["dispatchId"] = dispatch_id
        resp = await self._request("POST", "/api/bot/reset", json=body)
        if resp.status_code != 200:
            raise CrmError(f"reset devolvió {resp.status_code}")

    async def post_activate(self, conversation_id: str) -> None:
        """Reactiva solo la IA del chat; conserva ficha, etapa e historial."""
        resp = await self._request(
            "POST", "/api/bot/activate", json={"conversationId": conversation_id}
        )
        if resp.status_code != 200:
            raise CrmError(f"activate devolvió {resp.status_code}")

    async def get_media(self, media_id: str) -> tuple[bytes, str]:
        """Descarga un binario de Meta A TRAVÉS del CRM (el token vive allá).

        Devuelve (bytes, mime). Timeout amplio: los adjuntos pueden pesar.
        """
        resp = await self._request(
            "GET", f"/api/bot/media/{media_id}", timeout=60.0
        )
        if resp.status_code != 200:
            raise CrmError(f"media devolvió {resp.status_code}")
        mime = resp.headers.get("content-type") or "application/octet-stream"
        return resp.content, mime

    async def aclose(self) -> None:
        await self._http.aclose()
