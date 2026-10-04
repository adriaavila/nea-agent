"""Rastro de decisión del turno v2 (`decision` en la respuesta de /dispatch).

El CRM le enseña al dueño qué hizo el agente en cada turno y por qué ("Ver
cómo decidió"): modelo, versión del prompt, herramientas que llamó (en orden,
con un resumen en español) y cuánto tardó. Es un campo ADITIVO y opcional del
sobre — un CRM que lo ignore sigue funcionando igual (ver
`app/dispatch.py:_v2_body`).

Dos reglas que este módulo hace cumplir, porque el resumen se muestra tal cual
en una pantalla del negocio:

- **Nada sensible.** Los resúmenes salen de plantillas fijas y de listas
  cerradas: de la ficha del lead solo van nombres de campo CONOCIDOS (jamás
  valores ni claves que invente el modelo), del `handoff` solo el motivo
  canónico (nunca el texto libre del modelo, que puede traer nombres, cédulas
  o direcciones), de un fallo solo un código conocido. Lo único que entra de
  fuera es la etiqueta de un horario que arma el CRM; todo el resumen pasa
  igual por `_clean`, que tacha teléfonos, correos, enlaces y claves.
- **Jamás tumba un turno.** Armar el rastro es observabilidad: si el resumen
  de un paso falla, se cae a uno genérico en vez de propagar la excepción.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.crm import canonical_handoff_reason
from app.llm import LlmUsage
from app.tools import TOOL_SCHEMAS

logger = logging.getLogger("nea.decision")

MAX_STEPS = 20
MAX_SUMMARY_CHARS = 200
UNKNOWN_MODEL = "unknown"
UNKNOWN_TOOL = "desconocida"

# Solo las herramientas que el modelo puede pedir de verdad: cualquier otro
# nombre (inventado, o con datos escondidos) se reporta como "desconocida".
_KNOWN_TOOLS = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)

# Cuántos nombres de campo / horarios se listan antes de resumir con "y N más".
_MAX_LISTED = 6

# Nombres legibles de los campos de la ficha (app/tools.py:TOOL_SCHEMAS).
_FICHA_LABELS = {
    "rubro": "rubro",
    "rol": "rol",
    "tamano_aprox": "tamaño",
    "sistemas": "sistemas",
    "dolor_principal": "dolor principal",
    "geo": "zona",
    "calificado": "calificación",
    "resultado": "resultado",
    "notas": "notas",
}

# Motivo canónico del handoff (app/crm.py:canonical_handoff_reason, el mismo
# catálogo cerrado que recibe el CRM) → frase fija. Jamás el texto libre.
_HANDOFF_PHRASES = {
    "cliente": "pidió humano",
    "hostilidad": "mensajes hostiles",
    "modelo": "decisión del agente",
    "error": "error del agente",
    "ventana": "ventana de 24 h cerrada",
}

# Razón corta (en español) de los errores que devuelve `ToolRuntime.execute`.
_FAILURE_REASONS = {
    "crm_error": "el CRM no respondió",
    "sin_agenda": "este negocio no tiene agenda",
    "sin_disponibilidad": "no había horarios abiertos",
    "slot_no_ofrecido": "ese horario no se había ofrecido",
    "slot_not_offered": "ese horario no se había ofrecido",
    "slot_taken": "el horario ya estaba ocupado",
    "start_utc_invalido": "horario inválido",
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(?:sk|pk|rk|whsec|ghp)[-_][A-Za-z0-9_-]{8,}|\b[A-Za-z0-9_-]{32,}\b"
)

# Fechas y horas (las etiquetas de horarios que arma el CRM: "lun 6 oct,
# 10:00", "06-10-2026 10:00", "2026-10-06T10:00:00") NO son teléfonos. La
# alternativa `when` va primero para ganar en cada posición; el lookbehind
# obliga a que empiece en un borde de número (así "0414-12-3456" no se lee
# como una fecha a partir de su segundo grupo).
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?"
_DATE = r"\d{4}-\d{2}-\d{2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
_NUMBER_RE = re.compile(
    rf"(?P<when>(?<![\d+])(?P<date>{_DATE})(?:[ T](?P<dtime>{_TIME}))?(?!\d)"
    rf"|(?<![\d:])(?P<time>{_TIME})(?![\d:]))"
    r"|(?P<phone>\+?\d[\d\s().-]{5,}\d)"
)


def _valid_time(value: str) -> bool:
    parts = [int(p) for p in value.split(":")]
    return parts[0] <= 23 and all(p <= 59 for p in parts[1:])


def _valid_date(value: str) -> bool:
    first, second, third = re.split(r"[-/.]", value)
    if len(first) == 4:  # año-mes-día
        return 1 <= int(second) <= 12 and 1 <= int(third) <= 31
    day_or_month, other = int(first), int(second)  # d-m-a o m-d-a
    return 1 <= day_or_month <= 31 and 1 <= other <= 31 and min(day_or_month, other) <= 12


def _is_when(match: re.Match[str]) -> bool:
    """¿Es de verdad una fecha/hora y no un teléfono con la misma forma (p.ej.
    "12-34-5678")? Se exige que mes/día y hora/minuto quepan."""
    date, dtime, time = match.group("date"), match.group("dtime"), match.group("time")
    if date is not None and not _valid_date(date):
        return False
    return all(_valid_time(t) for t in (dtime, time) if t is not None)


def _redact_number(match: re.Match[str]) -> str:
    found = match.group()
    if match.group("when") is not None and _is_when(match):
        return found
    return "[número]" if sum(c.isdigit() for c in found) >= 7 else found


def _clean(text: str) -> str:
    """Texto seguro para mostrar: sin teléfonos, correos, enlaces ni claves,
    espacios colapsados y acotado a `MAX_SUMMARY_CHARS`."""
    text = _URL_RE.sub("[enlace]", text)
    text = _EMAIL_RE.sub("[correo]", text)
    text = _SECRET_RE.sub("[clave]", text)
    text = _NUMBER_RE.sub(_redact_number, text)
    text = " ".join(text.split())
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return text


def model_id(llm: Any) -> str:
    """Id del modelo con el que responde `llm` (`OpenAiLlm.model`); un cliente
    sin ese atributo (fakes, implementaciones ajenas) queda como "unknown"."""
    model = getattr(llm, "model", None)
    return model if isinstance(model, str) and model.strip() else UNKNOWN_MODEL


def _list_phrase(items: list[str]) -> str:
    shown = items[:_MAX_LISTED]
    extra = len(items) - len(shown)
    return ", ".join(shown) + (f" y {extra} más" if extra > 0 else "")


def _ficha_fields(args: dict[str, Any]) -> list[str]:
    """Nombres (NUNCA valores) de los campos que `update_ficha` mandó, con los
    mismos criterios que `ToolRuntime._update_ficha` (los `None` no cuentan)."""
    names: list[str] = []
    unknown = False
    for key, value in args.items():
        if value is None:
            continue
        label = _FICHA_LABELS.get(str(key))
        if label is None:
            # El modelo puede inventar claves ("juan_perez", "pw_hunter2"):
            # un nombre de campo desconocido jamás se imprime.
            unknown = True
        elif label not in names:
            names.append(label)
    if unknown:
        names.append("otros campos")
    return names


def _failure_reason(result: dict[str, Any]) -> str:
    error = str(result.get("error") or "")
    if error in _FAILURE_REASONS:
        return _FAILURE_REASONS[error]
    if error.startswith("herramienta desconocida"):
        return "herramienta desconocida"
    return "error"  # cualquier otro código (o texto libre) no se muestra


def _step_summary(tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    ok = bool(result.get("ok"))
    if tool == "update_ficha":
        names = _ficha_fields(args)
        if ok:
            return "actualizó lead: " + (_list_phrase(names) if names else "sin campos nuevos")
        return "intentó actualizar lead: " + (_list_phrase(names) or "sin campos")
    if tool == "propose_slots":
        slots = [s for s in (result.get("slots") or []) if isinstance(s, dict)]
        if ok and slots:
            labels = [str(s.get("label")) for s in slots if s.get("label")]
            noun = "horario" if len(slots) == 1 else "horarios"
            phrase = f"ofreció {len(slots)} {noun}"
            return f"{phrase}: {_list_phrase(labels)}" if labels else phrase
        return "buscó horarios"
    if tool in ("book_session", "reschedule_session"):
        moved = tool == "reschedule_session"
        if ok:
            label = str(result.get("label") or "").strip()
            verb = "movió la cita a" if moved else "agendó"
            return f"{verb}: {label}" if label else ("movió la cita" if moved else "agendó la cita")
        return "intentó mover la cita" if moved else "intentó agendar"
    if tool == "route_out":
        if ok:
            extra = ", compartió recursos alternativos" if result.get("recursos") else ""
            return "marcó al lead como no calificado" + extra
        return "intentó marcar al lead como no calificado"
    if tool == "handoff":
        # Mismo motivo que recibe el CRM (catálogo cerrado), jamás el texto
        # libre del modelo. Sin motivo, el runtime usa "lead_request".
        reason = canonical_handoff_reason(str(args.get("reason") or "lead_request"))
        return f"pasó a una persona: {_HANDOFF_PHRASES[reason]}"
    if tool == UNKNOWN_TOOL:
        return "intentó usar una herramienta desconocida"
    return f"usó {tool}"


def summarize_step(tool: str, args: dict[str, Any], result: Any) -> tuple[str, bool]:
    """(resumen, ok) de UNA llamada a herramienta. `result` es lo que devolvió
    `ToolRuntime.execute`; el resumen ya sale limpio y de ≤200 caracteres."""
    res = result if isinstance(result, dict) else {}
    ok = bool(res.get("ok"))
    tool = tool if tool in _KNOWN_TOOLS else UNKNOWN_TOOL
    summary = _step_summary(tool, args if isinstance(args, dict) else {}, res)
    if not ok and tool != UNKNOWN_TOOL:
        summary += f" (falló: {_failure_reason(res)})"
    return _clean(summary), ok


@dataclass
class DecisionTrace:
    """Acumula, durante UN turno v2, lo que `decision` reporta. Se crea al
    arrancar `run_turn` (el reloj de `latencyMs` empieza ahí) y se serializa
    con `to_dict()` justo antes de devolver el resultado."""

    prompt_version: str = ""
    model: str = UNKNOWN_MODEL
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    has_usage: bool = False
    started_at: float = field(default_factory=time.monotonic)

    def use_model(self, llm: Any) -> None:
        """El cliente que va a contestar (o que acaba de fallar): tras un
        fallback del negocio a la plataforma, `model` es el de la plataforma."""
        self.model = model_id(llm)

    def add_usage(self, usage: LlmUsage | None) -> None:
        """Suma el uso de una respuesta. Sin uso reportado en NINGUNA ronda,
        `tokens` se omite del `decision` (no se inventa un 0)."""
        if usage is None:
            return
        self.tokens_in += usage.input
        self.tokens_out += usage.output
        self.has_usage = True

    def add_step(self, tool: str, args: dict[str, Any], result: Any) -> None:
        if len(self.steps) >= MAX_STEPS:
            return
        # Solo herramientas de TOOL_SCHEMAS: un nombre inventado puede cargar
        # datos, así que se reporta como "desconocida".
        name = tool if tool in _KNOWN_TOOLS else UNKNOWN_TOOL
        try:
            summary, ok = summarize_step(name, args, result)
        except Exception:  # observabilidad: jamás tumba el turno
            logger.exception("decision: no pude resumir el paso %s", name)
            summary = (
                "intentó usar una herramienta desconocida"
                if name == UNKNOWN_TOOL
                else f"usó {name}"
            )
            ok = bool(isinstance(result, dict) and result.get("ok"))
        self.steps.append({"tool": name, "summary": summary, "ok": ok})

    def to_dict(self) -> dict[str, Any]:
        decision: dict[str, Any] = {
            "model": self.model,
            "promptVersion": self.prompt_version,
            "steps": list(self.steps),
            "latencyMs": max(0, int((time.monotonic() - self.started_at) * 1000)),
        }
        if self.has_usage:
            decision["tokens"] = {"input": self.tokens_in, "output": self.tokens_out}
        return decision
