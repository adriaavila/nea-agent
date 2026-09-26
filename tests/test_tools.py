"""Tools: book_session SOLO acepta slots ofrecidos; slot_taken trae alternativas."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from app.state import OfferedSlot
from app.tools import ToolRuntime
from tests.conftest import CRM_CONV_ID, CRM_URL, IDENTITY, make_ctx

SLOT_ISO = "2026-07-20T16:00:00Z"
SLOT_DT = datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc)
SLOT2_ISO = "2026-07-21T16:00:00Z"
SLOT2_DT = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)


@pytest.fixture
async def runtime_y_ctx():
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    await ctx.store.replace_offered_slots(
        conv.id,
        [
            OfferedSlot(
                conversation_id=conv.id,
                start_utc=SLOT_DT,
                end_utc=None,
                label="lunes 20 de julio, 10:00 am",
            )
        ],
    )
    runtime = ToolRuntime(ctx, conv, CRM_CONV_ID)
    yield runtime, ctx, conv
    await ctx.crm.aclose()


async def test_book_rechaza_slot_no_ofrecido(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    bookings = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(200, json={"bookingId": "bk_1", "label": "x"})
    )
    result = await runtime.execute(
        "book_session", {"start_utc": "2026-07-20T17:00:00Z"}  # nunca ofrecido
    )
    assert result["ok"] is False
    assert result["error"] == "slot_no_ofrecido"
    assert bookings.call_count == 0  # jamás llegó al CRM
    assert runtime.booked is False


async def test_book_acepta_slot_ofrecido_epoch_exacto(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    bookings = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        # 201 Created: el código REAL del CRM (route.ts responde 201, no 200 —
        # el mock infiel escondió este bug hasta la certificación 002).
        return_value=httpx.Response(
            201,
            json={
                "bookingId": "bk_1",
                "zoomJoinUrl": "https://zoom.us/j/1",
                "label": "lunes 20 de julio, 10:00 am",
            },
        )
    )
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": True})
    )
    # mismo instante escrito con offset en vez de Z — el epoch es lo que cuenta
    result = await runtime.execute(
        "book_session", {"start_utc": "2026-07-20T16:00:00+00:00"}
    )
    assert result["ok"] is True
    assert runtime.booked is True
    body = json.loads(bookings.calls[0].request.content)
    assert body == {"conversationId": CRM_CONV_ID, "startUtc": SLOT_ISO}
    # al reservar se limpian los ofrecidos
    assert await ctx.store.get_offered_slots(conv.id) == []


async def test_book_slot_taken_ofrece_alternativas_frescas(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    frescos = [
        {"startUtc": "2026-07-21T16:00:00Z", "endUtc": None, "label": "martes 21, 10:00 am"},
        {"startUtc": "2026-07-21T17:00:00Z", "endUtc": None, "label": "martes 21, 11:00 am"},
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(409, json={"code": "slot_taken", "slots": frescos})
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is False
    assert result["error"] == "slot_taken"
    assert [s["label"] for s in result["slots"]] == [s["label"] for s in frescos]
    # los frescos quedan como los nuevos (y únicos) reservables
    offered = await ctx.store.get_offered_slots(conv.id)
    assert [s.label for s in offered] == [s["label"] for s in frescos]
    assert runtime.booked is False


async def test_propose_slots_maximo_3_y_persistidos(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    seis = [
        {
            "startUtc": f"2026-07-2{d}T16:00:00Z",
            "endUtc": f"2026-07-2{d}T16:30:00Z",
            "label": f"día 2{d}, 10:00 am",
        }
        for d in range(6)
    ]
    respx_mock.get(f"{CRM_URL}/api/bot/availability").mock(
        return_value=httpx.Response(200, json={"slots": seis})
    )
    result = await runtime.execute("propose_slots", {})
    assert result["ok"] is True
    assert len(result["slots"]) == 3  # máx 3 por mensaje
    offered = await ctx.store.get_offered_slots(conv.id)
    assert len(offered) == 3
    assert runtime.proposed is True


async def test_update_ficha_manda_lo_que_haya(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    ficha_route = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": False})
    )
    result = await runtime.execute(
        "update_ficha",
        {"rubro": "clínica dental", "rol": "el dueño mero mero", "campo_raro": "x"},
    )
    assert result["ok"] is True
    body = json.loads(ficha_route.calls[0].request.content)
    # drift tolerado: se manda tal cual, el CRM normaliza flojo
    assert body["ficha"]["rol"] == "el dueño mero mero"
    assert body["ficha"]["campo_raro"] == "x"


async def test_handoff_se_difiere_al_final_del_turno(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    handoff_route = respx_mock.post(f"{CRM_URL}/api/bot/handoff").mock(
        return_value=httpx.Response(200, json={})
    )
    result = await runtime.execute("handoff", {"reason": "pidió humano"})
    assert result["ok"] is True
    assert runtime.handoff_reason == "pidió humano"
    # la tool NO llama al CRM: turn.py lo hace después de la despedida
    assert handoff_route.call_count == 0


async def test_crm_caido_en_tool_no_tumba_el_turno(runtime_y_ctx, respx_mock):
    runtime, ctx, conv = runtime_y_ctx
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(500)
    )
    result = await runtime.execute("update_ficha", {"rubro": "ferretería"})
    assert result["ok"] is False
    assert result["error"] == "crm_error"


# ---------------------------------------------- reintento tras reserva viva ---
# PR 2B, revisión: create_booking 201 pero el envío de la confirmación nunca
# llegó (v2 no tiene pending_send que lo rescate) → el turno se reintenta
# ENTERO con un catálogo de horarios NUEVO, que ya no ofrece el que se acaba
# de ocupar. Sin `already_booked`, book_session lo rechazaría como
# "no ofrecido" e intentaría reservar una SEGUNDA vez.


async def test_book_session_sobre_lo_ya_agendado_es_idempotente_sin_tocar_el_crm():
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    # Catálogo de ESTE intento vacío a propósito: el horario ya está ocupado,
    # el CRM no lo re-ofrece.
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(
            start_utc=SLOT_DT, label="lunes 20 de julio, 10:00 am", meeting_url="https://zoom.us/j/1"
        ),
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["label"] == "lunes 20 de julio, 10:00 am"
    assert result["meeting_url"] == "https://zoom.us/j/1"
    assert result["link_pendiente"] is False  # default: el CRM no manda linkPending en booking.next hoy
    assert runtime.booked is True
    await ctx.crm.aclose()


async def test_book_session_idempotente_propaga_link_pending_si_viene(runtime_y_ctx):
    """Ronda 3: si `context.booking.next` SÍ trae `linkPending` (el CRM no lo
    manda hoy, pero por si lo agrega — ver AlreadyBooked), el atajo
    idempotente debe reflejarlo, no un `False` fijo."""
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(
            start_utc=SLOT_DT, label="lunes 10am", meeting_url=None, link_pending=True
        ),
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["link_pendiente"] is True
    await ctx.crm.aclose()


async def test_book_session_sobre_otro_horario_distinto_al_ya_agendado_sigue_rechazando(
    respx_mock,
):
    """`already_booked` no es un pase libre: SOLO protege el horario exacto
    que ya está agendado — cualquier otro sigue exigiendo estar ofrecido."""
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    bookings = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "x"})
    )
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(start_utc=SLOT_DT, label="lunes 10am", meeting_url=None),
    )
    otro_horario = "2026-07-21T16:00:00Z"  # un día distinto al ya agendado
    result = await runtime.execute("book_session", {"start_utc": otro_horario})
    assert result["ok"] is False
    assert result["error"] == "slot_no_ofrecido"
    assert bookings.call_count == 0
    await ctx.crm.aclose()


async def test_reschedule_sobre_lo_ya_agendado_tambien_es_idempotente():
    """El mismo camino cubre reschedule_session: si el reintento pide mover
    a un horario que YA es el vigente (la mudanza anterior sí llegó al CRM),
    confirma sin volver a llamarlo."""
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(start_utc=SLOT_DT, label="lunes 10am", meeting_url=None),
    )
    result = await runtime.execute("reschedule_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["movida"] is True


async def test_v1_sin_already_booked_se_comporta_igual_que_siempre(runtime_y_ctx, respx_mock):
    """v1/legacy nunca pasa `already_booked` (default None): un horario no
    ofrecido sigue rechazándose exactamente como antes — sin este parámetro
    nuevo, nada cambia para el camino de siempre."""
    runtime, ctx, conv = runtime_y_ctx
    result = await runtime.execute(
        "book_session", {"start_utc": "2026-07-20T17:00:00Z"}  # nunca ofrecido
    )
    assert result["ok"] is False
    assert result["error"] == "slot_no_ofrecido"


# --------------------------------- regresión: already_booked queda viejo ---
# Revisión de PR 2B, ronda 3: sin refrescar `_already_booked` tras una
# reserva/movida REAL de este turno, un reschedule T1→T2 (real, exitoso)
# seguido de otro T2→T1 en el MISMO turno comparaba T1 contra el snapshot
# VIEJO (seguía en T1 desde el arranque) y devolvía "ok, movida a T1" SIN
# tocar el CRM — el lead se enteraba de la hora equivocada.


async def test_reschedule_a_otro_horario_y_de_vuelta_nunca_da_un_ok_falso(respx_mock):
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    # Arranca YA agendado en SLOT (snapshot de context.booking.next).
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(start_utc=SLOT_DT, label="lunes 10am", meeting_url=None),
    )
    reschedule_route = respx_mock.patch(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(200, json={"label": "martes 10am", "meetingLink": None})
    )
    # El modelo mueve la cita a SLOT2 — real, ofrecido, debe pasar por el CRM.
    await ctx.store.replace_offered_slots(
        conv.id,
        [OfferedSlot(conversation_id=conv.id, start_utc=SLOT2_DT, end_utc=None, label="martes 10am")],
    )
    result1 = await runtime.execute("reschedule_session", {"start_utc": SLOT2_ISO})
    assert result1["ok"] is True
    assert reschedule_route.call_count == 1

    # El modelo (confundido, o el lead cambió de opinión) pide volver a SLOT
    # (el horario ORIGINAL) EN EL MISMO turno. SLOT ya no está ofrecido (se
    # limpió tras el éxito de arriba) y el snapshot de "ya agendado" ahora
    # es SLOT2 (recién actualizado) — SLOT no debe dar un "ok" gratis.
    result2 = await runtime.execute("reschedule_session", {"start_utc": SLOT_ISO})
    assert result2["ok"] is False
    assert result2["error"] == "slot_no_ofrecido"
    assert reschedule_route.call_count == 1  # NO se llamó al CRM una segunda vez para esto

    await ctx.crm.aclose()


async def test_reschedule_de_vuelta_al_horario_original_si_se_reofrece_es_un_patch_real(
    respx_mock,
):
    """La otra rama del mismo escenario: si SLOT (el original) se vuelve a
    ofrecer de verdad tras la primera movida, el segundo reschedule_session
    SÍ debe pasar por el CRM — nunca el atajo, porque ya no es el snapshot
    de arranque."""
    from app.tools import AlreadyBooked

    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    runtime = ToolRuntime(
        ctx,
        conv,
        CRM_CONV_ID,
        already_booked=AlreadyBooked(start_utc=SLOT_DT, label="lunes 10am", meeting_url=None),
    )
    reschedule_route = respx_mock.patch(f"{CRM_URL}/api/bot/bookings").mock(
        side_effect=[
            httpx.Response(200, json={"label": "martes 10am", "meetingLink": None}),
            httpx.Response(200, json={"label": "lunes 10am de nuevo", "meetingLink": None}),
        ]
    )
    await ctx.store.replace_offered_slots(
        conv.id,
        [OfferedSlot(conversation_id=conv.id, start_utc=SLOT2_DT, end_utc=None, label="martes 10am")],
    )
    result1 = await runtime.execute("reschedule_session", {"start_utc": SLOT2_ISO})
    assert result1["ok"] is True

    # SLOT se re-ofrece de verdad (p.ej. el lead pidió volver y el modelo
    # llamó propose_slots otra vez).
    await ctx.store.replace_offered_slots(
        conv.id,
        [OfferedSlot(conversation_id=conv.id, start_utc=SLOT_DT, end_utc=None, label="lunes 10am")],
    )
    result2 = await runtime.execute("reschedule_session", {"start_utc": SLOT_ISO})
    assert result2["ok"] is True
    assert result2["label"] == "lunes 10am de nuevo"
    assert reschedule_route.call_count == 2  # las DOS movidas pasaron por el CRM

    await ctx.crm.aclose()


async def test_segunda_tool_call_sobre_lo_recien_reservado_en_el_mismo_turno_no_usa_el_atajo(
    respx_mock,
):
    """Sin `already_booked` de arranque (una reserva NUEVA, no un reintento):
    reservar y luego, en el MISMO turno, pedir el mismo horario otra vez no
    debe dar un "ok" gratis — el catálogo ya lo limpió, así que cae a
    "no ofrecido", no a un atajo de reintento que nunca aplicó aquí."""
    ctx = make_ctx()
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    runtime = ToolRuntime(ctx, conv, CRM_CONV_ID)  # sin already_booked
    booking_route = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"label": "lunes 10am", "meetingLink": None})
    )
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": True})
    )
    await ctx.store.replace_offered_slots(
        conv.id,
        [OfferedSlot(conversation_id=conv.id, start_utc=SLOT_DT, end_utc=None, label="lunes 10am")],
    )
    result1 = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result1["ok"] is True
    assert booking_route.call_count == 1

    result2 = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result2["ok"] is False
    assert result2["error"] == "slot_no_ofrecido"
    assert booking_route.call_count == 1  # nunca una segunda reserva

    await ctx.crm.aclose()
