"""El reloj del agente.

Este test existe por un fallo del piloto: "el agente no entendía bien las
fechas". Eran dos causas, ninguna del modelo:

1. `strftime("%A ... %B")` traduce según el locale del PROCESO, y la imagen no
   fija ninguno. En el contenedor salía el locale C y el prompt decía
   "Thursday 05 de March" dentro de un texto en español — el modelo tenía tres
   vocabularios para un mismo día y elegía mal.
2. La zona horaria estaba cableada a America/Mexico_City. Un negocio en
   Caracas quedaba una hora corrido respecto del motor de agenda del CRM, que
   etiqueta los huecos en la zona del negocio.
"""

import locale
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.profile import profile_from_payload
from app.prompt import _fmt_local
from app.turn import _agent_tz

AHORA = datetime(2026, 3, 5, 21, 30, tzinfo=timezone.utc)


def test_dia_y_mes_en_espanol():
    salida = _fmt_local(AHORA, ZoneInfo("America/Mexico_City"))
    assert "jueves" in salida
    assert "marzo" in salida
    assert "15:30" in salida
    assert "2026" in salida
    # Lo que salía antes en el contenedor.
    assert "Thursday" not in salida
    assert "March" not in salida


def test_no_depende_del_locale_del_proceso():
    """El locale C es el del contenedor: ahí es donde se rompía."""
    previo = locale.setlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, "C")
        salida = _fmt_local(AHORA, ZoneInfo("America/Mexico_City"))
    finally:
        locale.setlocale(locale.LC_TIME, previo)
    assert "jueves 5 de marzo" in salida


def test_la_zona_cambia_el_dia():
    """Misma marca de tiempo, otro día: si la zona no manda, se ofrece mal."""
    assert "jueves 5" in _fmt_local(AHORA, ZoneInfo("America/Mexico_City"))
    assert "viernes 6" in _fmt_local(AHORA, ZoneInfo("Asia/Tokyo"))


class _Settings:
    agent_timezone = "America/Mexico_City"


def test_manda_la_zona_del_negocio_sobre_la_del_despliegue():
    perfil = profile_from_payload(
        {"profile": {"name": "Nea", "timezone": "America/Caracas"}}, "Nea"
    )
    assert perfil.timezone == "America/Caracas"
    assert _agent_tz(_Settings(), perfil).key == "America/Caracas"


def test_sin_zona_del_crm_manda_el_despliegue():
    perfil = profile_from_payload({"profile": {"name": "Nea"}}, "Nea")
    assert perfil.timezone is None
    assert _agent_tz(_Settings(), perfil).key == "America/Mexico_City"


def test_una_zona_basura_no_tumba_el_turno():
    class Rota:
        agent_timezone = "No/Existe"

    perfil = profile_from_payload({"profile": {"timezone": "Tampoco/Existe"}}, "Nea")
    # Degrada al último recurso en vez de reventar el turno del lead.
    assert _agent_tz(Rota(), perfil).key == "America/Mexico_City"
