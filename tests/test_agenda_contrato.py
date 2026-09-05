"""
El contrato de agenda del CRM cambió con el motor universal (015) de Vocero.

Lo que rompió y por qué cada caso está aquí:
  · `GET /api/bot/availability` ahora EXIGE conversationId — sin él es 422 y el
    agente se queda mudo justo cuando el lead dijo que sí.
  · Es esa llamada la que REGISTRA la oferta. Reservar sin haber pasado por ahí
    es 409 `slot_not_offered`, siempre.
  · Crear responde 201 (no 200) y el 409 trae el código ANIDADO en
    `{"error":{"code":…}}` con `slots` de hermano. Un cliente que leyera el
    sobre plano trataría todo 409 como genérico y nunca re-ofrecería.
  · La agenda es opcional (bandera AGENDA): apagada, la superficie es 404 y el
    agente tiene que coordinar por humano en vez de romperse.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from app.state import OfferedSlot
from app.tools import ToolRuntime
from tests.conftest import CRM_CONV_ID, CRM_URL, IDENTITY, make_ctx

SLOT_ISO = "2026-07-20T16:00:00Z"
SLOT_DT = datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc)
OTRO_ISO = "2026-07-21T16:00:00Z"


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


async def test_propose_slots_manda_la_conversacion(runtime_y_ctx, respx_mock):
    """Sin conversationId el CRM no registra la oferta y reservar es imposible."""
    runtime, _ctx, _conv = runtime_y_ctx
    ruta = respx_mock.get(f"{CRM_URL}/api/bot/availability").mock(
        return_value=httpx.Response(
            200,
            json={
                "slots": [
                    {
                        "startUtc": SLOT_ISO,
                        "endUtc": "2026-07-20T16:30:00Z",
                        "label": "lunes 20, 10:00 am",
                    }
                ],
                "diasConAgenda": ["2026-07-20", "2026-07-21"],
            },
        )
    )
    result = await runtime.execute("propose_slots", {})
    assert result["ok"] is True
    pedido = ruta.calls.last.request.url
    assert pedido.params["conversationId"] == CRM_CONV_ID
    # Se piden más de los que se enseñan: el catálogo reservable es más ancho.
    assert int(pedido.params["limit"]) > 3
    assert result["dias_con_agenda"] == ["2026-07-20", "2026-07-21"]


async def test_agenda_apagada_no_rompe_el_turno(runtime_y_ctx, respx_mock):
    """AGENDA=off ⇒ 404. El agente coordina por humano, no se cae."""
    runtime, _ctx, _conv = runtime_y_ctx
    respx_mock.get(f"{CRM_URL}/api/bot/availability").mock(
        return_value=httpx.Response(404)
    )
    result = await runtime.execute("propose_slots", {})
    assert result["ok"] is False
    assert result["error"] == "sin_agenda"


async def test_reserva_acepta_201(runtime_y_ctx, respx_mock):
    """El CRM responde 201 Created. Un cliente que exija 200 no agenda nunca."""
    runtime, _ctx, _conv = runtime_y_ctx
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(
            201,
            json={
                "bookingId": "bk_1",
                "label": "lunes 20, 10:00 am",
                "meetingLink": "https://meet.example/x",
                "linkPending": False,
            },
        )
    )
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}})
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["meeting_url"] == "https://meet.example/x"
    assert runtime.booked is True


async def test_link_pendiente_no_promete_enlace(runtime_y_ctx, respx_mock):
    """La cita existe y el enlace no: se confirma la cita, no se inventa el link."""
    runtime, _ctx, _conv = runtime_y_ctx
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(
            201,
            json={
                "bookingId": "bk_1",
                "label": "lunes 20, 10:00 am",
                "meetingLink": None,
                "linkPending": True,
            },
        )
    )
    respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}})
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["meeting_url"] is None
    assert result["link_pendiente"] is True


@pytest.mark.parametrize("code", ["slot_taken", "slot_not_offered"])
async def test_409_anidado_reofrece(runtime_y_ctx, respx_mock, code):
    """El sobre anidado se entiende y las alternativas del CRM sustituyen la oferta."""
    runtime, ctx, conv = runtime_y_ctx
    frescos = [
        {"startUtc": OTRO_ISO, "endUtc": "2026-07-21T16:30:00Z", "label": "martes 21, 10:00 am"}
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(
            409,
            json={"error": {"code": code, "message": "no"}, "slots": frescos},
        )
    )
    result = await runtime.execute("book_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is False
    assert result["error"] == code
    assert [s["label"] for s in result["slots"]] == ["martes 21, 10:00 am"]
    guardados = await ctx.store.get_offered_slots(conv.id)
    assert [s.label for s in guardados] == ["martes 21, 10:00 am"]
    assert runtime.booked is False


async def test_reagendar_usa_patch_y_no_recalifica(runtime_y_ctx, respx_mock):
    """Mover una cita no vuelve a marcar la ficha como 'agendó'."""
    runtime, _ctx, _conv = runtime_y_ctx
    patch = respx_mock.patch(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(
            200,
            json={
                "bookingId": "bk_1",
                "label": "lunes 20, 10:00 am",
                "meetingLink": None,
                "linkPending": False,
            },
        )
    )
    ficha = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}})
    )
    result = await runtime.execute("reschedule_session", {"start_utc": SLOT_ISO})
    assert result["ok"] is True
    assert result["movida"] is True
    assert patch.call_count == 1
    assert ficha.call_count == 0


async def test_reagendar_tambien_exige_slot_ofrecido(runtime_y_ctx, respx_mock):
    runtime, _ctx, _conv = runtime_y_ctx
    patch = respx_mock.patch(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(200, json={"bookingId": "bk_1", "label": "x"})
    )
    result = await runtime.execute("reschedule_session", {"start_utc": OTRO_ISO})
    assert result["ok"] is False
    assert result["error"] == "slot_no_ofrecido"
    assert patch.call_count == 0
