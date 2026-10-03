"""Rastro de decisión del turno v2 (`decision` en la respuesta de /dispatch).

El CRM le enseña al dueño qué hizo el agente en cada turno y por qué ("Ver
cómo decidió"): modelo, versión del prompt, herramientas que llamó (en orden,
con un resumen en español) y cuánto tardó. Es un campo ADITIVO y opcional del
sobre — un CRM que lo ignore sigue funcionando igual (ver
`app/dispatch.py:_v2_body`).

Dos reglas que este módulo hace cumplir, porque el resumen se muestra tal cual
en una pantalla del negocio:

- **Nada sensible.** Los resúmenes salen de plantillas fijas; lo poco que
  entra de fuera (etiquetas de horarios del CRM, el motivo libre del
  `handoff`) pasa por `_clean`, que tacha teléfonos, correos, enlaces y
  claves. De la ficha del lead solo van los NOMBRES de los campos, jamás los
  valores. Un fallo de herramienta lleva una razón corta, nunca un stack.
- **Jamás tumba un turno.** Armar el rastro es observabilidad: si el resumen
  de un paso falla, se cae a uno genérico en vez de propagar la excepción.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.llm import LlmUsage

logger = logging.getLogger("nea.decision")

MAX_STEPS = 20
MAX_SUMMARY_CHARS = 200
UNKNOWN_MODEL = "unknown"

# Cuántos nombres de campo / horarios se listan antes de resumir con "y N más".
_MAX_LISTED = 6
_MAX_REASON_CHARS = 80

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
# El LLM puede inventar claves: solo pasan las que parecen un identificador
# corto y sin rachas de dígitos (un nombre de campo jamás debe poder cargar
# un teléfono o un id).
_SAFE_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}")
_DIGIT_RUN_RE = re.compile(r"\d{4,}")

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
_SAFE_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,39}")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{5,}\d")
_SECRET_RE = re.compile(
    r"\b(?:sk|pk|rk|whsec|ghp)[-_][A-Za-z0-9_-]{8,}|\b[A-Za-z0-9_-]{32,}\b"
)
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _redact_phone(match: re.Match[str]) -> str:
    found = match.group()
    if _ISO_DATE_RE.fullmatch(found):  # una fecha no es un teléfono
        return found
    return "[número]" if sum(c.isdigit() for c in found) >= 7 else found


def _clean(text: str) -> str:
    """Texto seguro para mostrar: sin teléfonos, correos, enlaces ni claves,
    espacios colapsados y acotado a `MAX_SUMMARY_CHARS`."""
    text = _URL_RE.sub("[enlace]", text)
    text = _EMAIL_RE.sub("[correo]", text)
    text = _SECRET_RE.sub("[clave]", text)
    text = _PHONE_RE.sub(_redact_phone, text)
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
    unknown = 0
    for key, value in args.items():
        if value is None:
            continue
        key = str(key)
        label = _FICHA_LABELS.get(key)
        if label is None:
            if _SAFE_FIELD_RE.fullmatch(key) and not _DIGIT_RUN_RE.search(key):
                label = key.replace("_", " ")
            else:
                unknown += 1
                continue
        if label not in names:
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
    if _SAFE_CODE_RE.fullmatch(error):
        return error.replace("_", " ")
    return "error"


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
        reason = " ".join(str(args.get("reason") or "").split())[:_MAX_REASON_CHARS]
        return f"pasó a una persona: {reason}" if reason else "pasó a una persona"
    return f"usó {tool}"


def summarize_step(tool: str, args: dict[str, Any], result: Any) -> tuple[str, bool]:
    """(resumen, ok) de UNA llamada a herramienta. `result` es lo que devolvió
    `ToolRuntime.execute`; el resumen ya sale limpio y de ≤200 caracteres."""
    res = result if isinstance(result, dict) else {}
    ok = bool(res.get("ok"))
    summary = _step_summary(tool, args if isinstance(args, dict) else {}, res)
    if not ok:
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
        name = _TOOL_NAME_RE.sub("", str(tool))[:64] or "desconocida"
        try:
            summary, ok = summarize_step(name, args, result)
        except Exception:  # observabilidad: jamás tumba el turno
            logger.exception("decision: no pude resumir el paso %s", name)
            summary, ok = f"usó {name}", bool(isinstance(result, dict) and result.get("ok"))
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
