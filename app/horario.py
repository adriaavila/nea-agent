"""El horario del equipo, dicho como lo diría una persona.

El dueño lo pone en «Tu negocio» del CRM (`agent_profile.business_hours`) y
llega en el perfil del despacho v2. Con él el agente contesta "¿a qué hora
abren?" sin que el dueño tenga que repetirlo en el conocimiento, y cuando pasa
una conversación a una persona fuera de horario dice CUÁNDO le van a
contestar en vez de prometer un "en un momento" que nadie va a cumplir.

Mismo contrato que `src/server/business-hours.ts` del CRM: intervalos HH:MM
por día (`mon`..`sun`), `00:00–00:00` es el día entero y un intervalo que
termina antes de empezar cruza la medianoche.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.profile import WEEKDAY_KEYS

Hours = dict[str, tuple[tuple[str, str], ...]]

_NOMBRES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _all_day(start: str, end: str) -> bool:
    return start == "00:00" and end == "00:00"


def _render_day(intervals: tuple[tuple[str, str], ...]) -> str:
    if any(_all_day(s, e) for s, e in intervals):
        return "abierto todo el día"
    return " y ".join(f"{s} a {e}" for s, e in intervals)


def describe_hours(hours: Hours) -> str | None:
    """"lunes a viernes de 09:00 a 18:00; sábado de 10:00 a 14:00; domingo
    cerrado". Agrupa días seguidos con el mismo horario. None sin horario."""
    if not hours:
        return None
    per_day = [hours.get(d) for d in WEEKDAY_KEYS]
    groups: list[tuple[int, int, tuple[tuple[str, str], ...] | None]] = []
    for i, iv in enumerate(per_day):
        if groups and groups[-1][2] == iv and groups[-1][1] == i - 1:
            groups[-1] = (groups[-1][0], i, iv)
        else:
            groups.append((i, i, iv))
    parts: list[str] = []
    for first, last, iv in groups:
        if first == last:
            dias = _NOMBRES[first]
        elif last == first + 1:
            dias = f"{_NOMBRES[first]} y {_NOMBRES[last]}"
        else:
            dias = f"{_NOMBRES[first]} a {_NOMBRES[last]}"
        if iv is None:
            parts.append(f"{dias} cerrado")
        else:
            rendered = _render_day(iv)
            parts.append(f"{dias} {rendered}" if rendered.startswith("abierto") else f"{dias} de {rendered}")
    return "; ".join(parts)


def _open_at(hours: Hours, local: datetime) -> bool:
    day = WEEKDAY_KEYS[local.weekday()]
    prev = WEEKDAY_KEYS[(local.weekday() - 1) % 7]
    minute = local.hour * 60 + local.minute
    for s, e in hours.get(day, ()):
        if _all_day(s, e):
            return True
        a, b = _minutes(s), _minutes(e)
        if a < b and a <= minute < b:
            return True
        if a > b and minute >= a:  # cruza la medianoche: la parte de hoy
            return True
    for s, e in hours.get(prev, ()):
        a, b = _minutes(s), _minutes(e)
        if not _all_day(s, e) and a > b and minute < b:
            return True
    return False


def team_status(hours: Hours, tz: ZoneInfo, now: datetime) -> tuple[bool, datetime | None]:
    """(¿el equipo atiende ahora?, próxima apertura en hora local si no).

    La próxima apertura se busca minuto a minuto durante 8 días (~11 mil
    comparaciones: microsegundos al lado de una llamada al modelo)."""
    local = now.astimezone(tz)
    if _open_at(hours, local):
        return True, None
    # Empezar en el siguiente minuto exacto que el dueño puede haber escrito.
    cursor = local.replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = cursor + timedelta(days=8)
    while cursor < end:
        if _open_at(hours, cursor):
            return False, cursor
        cursor += timedelta(minutes=1)
    return False, None


def describe_next_open(next_open: datetime, now: datetime) -> str:
    """"hoy a las 16:00", "mañana a las 09:00" o "el lunes a las 09:00"."""
    local_now = now.astimezone(next_open.tzinfo)
    days = (next_open.date() - local_now.date()).days
    hora = next_open.strftime("%H:%M")
    if days == 0:
        return f"hoy a las {hora}"
    if days == 1:
        return f"mañana a las {hora}"
    return f"el {_NOMBRES[next_open.weekday()]} a las {hora}"
