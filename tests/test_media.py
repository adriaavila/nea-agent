"""app/media.py: ubicacion entrante -- nunca "lat None, long None" (PR 2B)."""
from __future__ import annotations

from app.media import describe_item
from app.state import InboundMessage
from tests.conftest import make_ctx


async def test_ubicacion_sin_coordenadas_no_dice_none():
    ctx = make_ctx()
    msg = InboundMessage(
        wa_message_id="w1", identity="52155", type="location", location={}
    )
    part = await describe_item(ctx, msg)
    assert "None" not in (part.text or "")
    assert "coordenadas" in (part.text or "")


async def test_ubicacion_con_coordenadas_se_renderiza_igual_que_antes():
    ctx = make_ctx()
    msg = InboundMessage(
        wa_message_id="w2",
        identity="52155",
        type="location",
        location={"latitude": 20.69, "longitude": -101.36},
    )
    part = await describe_item(ctx, msg)
    assert "lat 20.69, long -101.36" in (part.text or "")


async def test_ubicacion_con_nombre_sin_coordenadas_muestra_el_nombre():
    ctx = make_ctx()
    msg = InboundMessage(
        wa_message_id="w3",
        identity="52155",
        type="location",
        location={"name": "Consultorio Centro"},
    )
    part = await describe_item(ctx, msg)
    assert "Consultorio Centro" in (part.text or "")
    assert "None" not in (part.text or "")
