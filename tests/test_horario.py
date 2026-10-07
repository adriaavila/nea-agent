"""El agente sabe el horario del equipo y nota cuando el lead vuelve días después.

Antes, el horario que el dueño pone en «Tu negocio» llegaba en el perfil del
despacho y Nea lo tiraba: "¿a qué hora abren?" solo tenía respuesta si el
dueño lo repetía en el conocimiento, y al pasar a una persona fuera de
horario el agente prometía una respuesta que nadie iba a dar hasta el día
siguiente.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.horario import describe_hours, describe_next_open, team_status
from app.profile import profile_from_payload
from app.prompt import build_system_prompt
from app.state import Conversation
from app.stateless import DispatchHistoryItemIn, _history_messages

CDMX = ZoneInfo("America/Mexico_City")
SEMANA = {
    "mon": [{"start": "09:00", "end": "18:00"}],
    "tue": [{"start": "09:00", "end": "18:00"}],
    "wed": [{"start": "09:00", "end": "18:00"}],
    "thu": [{"start": "09:00", "end": "18:00"}],
    "fri": [{"start": "09:00", "end": "18:00"}],
    "sat": [{"start": "10:00", "end": "14:00"}],
}


def _profile(hours=SEMANA):
    return profile_from_payload(
        {
            "profile": {
                "name": "Sofi",
                "instructions": "Clínica dental.",
                "businessHours": hours,
                "businessTimezone": "America/Mexico_City",
            },
            "kb": "",
        },
        "Nea",
    )


def test_el_perfil_lee_el_horario_y_descarta_lo_mal_formado():
    p = _profile({**SEMANA, "sun": [{"start": "9", "end": "x"}], "xyz": []})
    assert set(p.business_hours) == {"mon", "tue", "wed", "thu", "fri", "sat"}
    assert p.business_timezone == "America/Mexico_City"
    assert _profile(None).business_hours == {}


def test_horario_dicho_como_persona():
    texto = describe_hours(_profile().business_hours)
    assert texto == (
        "lunes a viernes de 09:00 a 18:00; sábado de 10:00 a 14:00; domingo cerrado"
    )


def test_dia_entero_y_varios_tramos():
    p = _profile({
        "mon": [{"start": "09:00", "end": "13:00"}, {"start": "15:00", "end": "19:00"}],
        "tue": [{"start": "00:00", "end": "00:00"}],
    })
    texto = describe_hours(p.business_hours)
    assert "lunes de 09:00 a 13:00 y 15:00 a 19:00" in texto
    assert "martes abierto todo el día" in texto
    assert "miércoles a domingo cerrado" in texto


def test_estado_del_equipo_y_proxima_apertura():
    hours = _profile().business_hours
    # Jueves 5 de marzo de 2026, 12:00 en CDMX → abierto.
    abierto, _ = team_status(hours, CDMX, datetime(2026, 3, 5, 18, 0, tzinfo=timezone.utc))
    assert abierto
    # Viernes 21:00 CDMX → vuelve el sábado a las 10:00 ("mañana").
    ahora = datetime(2026, 3, 7, 3, 0, tzinfo=timezone.utc)
    abierto, proxima = team_status(hours, CDMX, ahora)
    assert not abierto
    assert describe_next_open(proxima, ahora) == "mañana a las 10:00"
    # Sábado 16:00 CDMX → el lunes a las 09:00.
    ahora = datetime(2026, 3, 7, 22, 0, tzinfo=timezone.utc)
    _, proxima = team_status(hours, CDMX, ahora)
    assert describe_next_open(proxima, ahora) == "el lunes a las 09:00"
    # Jueves 07:30 CDMX → hoy a las 09:00.
    ahora = datetime(2026, 3, 5, 13, 30, tzinfo=timezone.utc)
    _, proxima = team_status(hours, CDMX, ahora)
    assert describe_next_open(proxima, ahora) == "hoy a las 09:00"


def test_turno_que_cruza_la_medianoche():
    hours = _profile({"fri": [{"start": "20:00", "end": "02:00"}]}).business_hours
    # Sábado 01:00 CDMX: sigue el turno del viernes.
    abierto, _ = team_status(hours, CDMX, datetime(2026, 3, 7, 7, 0, tzinfo=timezone.utc))
    assert abierto


def test_el_prompt_lleva_horario_y_si_el_equipo_atiende():
    p = _profile()
    fuera = build_system_prompt(
        profile=p,
        context={},
        conv=Conversation(id=0, wa_identity="x", greeted=True),
        now=datetime(2026, 3, 7, 3, 0, tzinfo=timezone.utc),
        tz=CDMX,
    )
    assert "lunes a viernes de 09:00 a 18:00" in fuera
    assert "FUERA de horario" in fuera and "mañana a las 10:00" in fuera
    dentro = build_system_prompt(
        profile=p,
        context={},
        conv=Conversation(id=0, wa_identity="x", greeted=True),
        now=datetime(2026, 3, 5, 18, 0, tzinfo=timezone.utc),
        tz=CDMX,
    )
    assert "está en horario de atención" in dentro
    sin = build_system_prompt(
        profile=_profile({}),
        context={},
        conv=Conversation(id=0, wa_identity="x", greeted=True),
        tz=CDMX,
    )
    assert "Horario de atención" not in sin and "FUERA de horario" not in sin


def test_cita_existente_se_mueve_no_se_escala():
    texto = build_system_prompt(
        profile=_profile(),
        context={"booking": {"next": {"scheduledAtUtc": "2026-03-09T16:00:00Z", "label": None}}},
        conv=Conversation(id=0, wa_identity="x", greeted=True),
        tz=CDMX,
    )
    assert "2026-03-09T16:00:00Z" in texto
    assert "reschedule_session" in texto
    assert "si quiere cambiarla, handoff" not in texto


def test_el_historial_marca_cuando_el_lead_vuelve_dias_despues():
    history = [
        DispatchHistoryItemIn(role="lead", text="hola, precio de limpieza?", at="2026-03-01T15:00:00Z"),
        DispatchHistoryItemIn(role="agent", text="Cuesta $600.", at="2026-03-01T15:00:20Z"),
        DispatchHistoryItemIn(role="lead", text="ok gracias", at="2026-03-01T15:02:00Z"),
        DispatchHistoryItemIn(role="lead", text="hola de nuevo", at="2026-03-05T18:00:00Z", pending=True),
    ]
    msgs, _ = _history_messages(history)
    notas = [m for m in msgs if m["role"] == "system"]
    assert len(notas) == 1
    assert "4 días" in notas[0]["content"]
    # La nota queda al final: justo antes del mensaje pendiente del lead.
    assert msgs[-1] is notas[0]


def test_sin_pausa_larga_no_hay_nota():
    history = [
        DispatchHistoryItemIn(role="lead", text="hola", at="2026-03-01T15:00:00Z"),
        DispatchHistoryItemIn(role="agent", text="¡Hola!", at="2026-03-01T15:00:20Z"),
        DispatchHistoryItemIn(role="lead", text="precio?", at="2026-03-01T17:00:00Z", pending=True),
        DispatchHistoryItemIn(role="lead", text="sin fecha"),
    ]
    msgs, _ = _history_messages(history)
    assert all(m["role"] != "system" for m in msgs)
