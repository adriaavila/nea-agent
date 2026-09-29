"""Lectura y render de `context.realty` (vertical inmobiliario).

`context.realty` es EXCLUSIVO de organizaciones con `profile.vertical ==
"inmobiliario"` — el resto del contrato de `GET /api/bot/context` no cambia.
Este módulo NUNCA llama al CRM: solo lee el bloque que ya vino en el
despacho y lo compacta para el prompt (`app/verticals/inmobiliario/
prompt.py`). Las candidatas que aquí se leen son la ÚNICA fuente de verdad
que el chasis tiene permitido mencionar sin haber llamado antes a una tool
(`guardar_requerimiento`/`ver_propiedad`) que regrese las suyas propias,
frescas — ver `app/verticals/inmobiliario/tools.py`.

Blindado contra un `context.realty` malformado: CUALQUIER pieza que no tenga
el tipo esperado (no un dict, no una lista) se trata como ausente. Nunca
revienta el turno por un contrato que el CRM todavía no sirve del todo.

Todo texto que pudo haber escrito un humano (notas del requerimiento, zonas,
títulos de propiedad) se renderiza con `json.dumps` — igual que
`app.prompt.build_system_prompt` hace con la ficha del lead — para que un
salto de línea o una comilla dentro de ese texto no pueda simular una nueva
línea de sistema (inyección de prompt).
"""
from __future__ import annotations

import json
from typing import Any

from app.verticals.inmobiliario import catalog


def realty_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """El bloque `context.realty` tal cual, tolerante a su ausencia o a que
    no sea un dict (el CRM aún no lo manda en este turno, o la migración de
    la organización está a medias)."""
    if not isinstance(context, dict):
        return {}
    realty = context.get("realty")
    return realty if isinstance(realty, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    """Una lista de dicts, tolerante: cualquier otra cosa (None, un dict
    suelto, un string) se vuelve una lista vacía; los elementos que no sean
    dict dentro de una lista sí válida se descartan uno por uno."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v not in (None, "")]


def realty_requirement(context: dict[str, Any] | None) -> dict[str, Any]:
    requirement = realty_context(context).get("requirement")
    return requirement if isinstance(requirement, dict) else {}


def realty_candidates(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    return _dict_list(realty_context(context).get("candidates"))


def candidate_ids(context: dict[str, Any] | None) -> frozenset[str]:
    """Los ids de las candidatas que el CRM ya ofreció en `context.realty` —
    útil para pruebas/instrumentación que quieran verificar que el chasis
    nunca menciona una propiedad fuera de este catálogo (o de una tool)."""
    return frozenset(
        str(c["id"]) for c in realty_candidates(context) if c.get("id") not in (None, "")
    )


def _jsonify(value: Any) -> str:
    """Texto potencialmente escrito por un humano (nota, zona, título),
    listo para pegarse dentro de una línea del prompt sin que un salto de
    línea o una comilla adentro finja ser una línea de sistema nueva."""
    return json.dumps(str(value), ensure_ascii=False)


def _fmt_money(value: Any, currency: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        amount = f"{float(value):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        amount = str(value)
    return f"{amount} {currency}" if currency else amount


_REQUIREMENT_LABELS = {
    "operation": "operación",
    "kind": "tipo",
    "zones": "zona(s)",
    "minBedrooms": "dormitorios mínimos",
    "minBathrooms": "baños mínimos",
    "paymentMethod": "forma de pago",
    "needsGuarantor": "necesita garante",
    "urgency": "urgencia",
    "amenities": "amenidades",
}

#: Campos que son texto potencialmente escrito por un humano — se renderizan
#: con `_jsonify` (ver el docstring del módulo). El resto (enums cerrados,
#: números, booleanos) no puede traer una inyección: son valores del
#: catálogo o numéricos, así que se muestran tal cual.
_FREE_TEXT_FIELDS = frozenset({"zones", "notes"})


def _requirement_lines(requirement: dict[str, Any]) -> list[str]:
    if not requirement:
        return ["- Requerimiento: todavía no se sabe nada, empieza a calificar."]
    lines = ["- Requerimiento guardado hasta ahora:"]
    for key, label in _REQUIREMENT_LABELS.items():
        value = requirement.get(key)
        if value in (None, "", []):
            continue
        if key in _FREE_TEXT_FIELDS:
            value_str = (
                ", ".join(_jsonify(v) for v in value)
                if isinstance(value, list)
                else _jsonify(value)
            )
        elif isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        lines.append(f"  - {label}: {value_str}")
    notes = requirement.get("notes")
    if notes not in (None, ""):
        lines.append(f"  - notas: {_jsonify(notes)}")
    presupuesto_min = _fmt_money(requirement.get("budgetMin"), requirement.get("currency"))
    presupuesto_max = _fmt_money(requirement.get("budgetMax"), requirement.get("currency"))
    if presupuesto_min or presupuesto_max:
        rango = " a ".join(p for p in (presupuesto_min, presupuesto_max) if p)
        lines.append(f"  - presupuesto: {rango}")
    missing = _str_list(requirement.get("missing"))
    if missing:
        lines.append(
            "  - AÚN FALTA (pregunta esto, uno a la vez): " + ", ".join(missing)
        )
    elif requirement.get("complete"):
        lines.append("  - requerimiento completo: ya puedes proponer candidatas")
    return lines


def _candidate_line(c: dict[str, Any]) -> str:
    precio = _fmt_money(c.get("price"), c.get("currency")) or "precio no disponible"
    zona = c.get("zone") or c.get("city") or "zona sin dato"
    reasons = _str_list(c.get("reasons"))
    razones = "; ".join(_jsonify(r) for r in reasons) if reasons else "sin razón registrada"
    titulo_raw = c.get("title") or c.get("kind") or "propiedad"
    titulo = _jsonify(titulo_raw)
    estado_raw = c.get("status")
    estado = catalog.PROPERTY_STATUS_LABELS.get(str(estado_raw)) if estado_raw else None
    estado_txt = f" · estado: {estado}" if estado else ""
    return f"  - id={c.get('id')} · {titulo} · {precio} · {zona}{estado_txt} · {razones}"


def _candidates_lines(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return [
            "- Candidatas: ninguna todavía. NO inventes propiedades, guarda "
            "el requerimiento (guardar_requerimiento) y espera candidatas reales."
        ]
    lines = [
        "- Candidatas disponibles (SOLO estas; máximo 3 por mensaje, una "
        "razón corta cada una; si el estado es reservada o cerrada, dilo "
        "con honestidad antes de insistir en ella):"
    ]
    lines.extend(_candidate_line(c) for c in candidates)
    return lines


def _viewings_lines(viewings: list[dict[str, Any]]) -> list[str]:
    if not viewings:
        return []
    lines = ["- Visitas ya agendadas:"]
    for v in viewings:
        titulo = _jsonify(v.get("propertyTitle") or v.get("propertyId") or "propiedad")
        cuando = v.get("label") or v.get("startUtc") or "sin fecha"
        estado = v.get("status") or "confirmada"
        lines.append(f"  - {titulo}: {cuando} ({estado}).")
    return lines


def render_realty_block(context: dict[str, Any] | None) -> str:
    """Bloque compacto para CONTEXTO ACTUAL: requerimiento + faltantes,
    candidatas con id/precio/zona/estado/razones, propiedad en foco y
    visitas agendadas. Tolerante a `context.realty` ausente o malformado
    (ver el docstring del módulo): el modelo debe saber que sigue en modo
    inmobiliario (calificando) aunque el CRM todavía no haya mandado nada."""
    realty = realty_context(context)
    requirement = realty_requirement(context)
    candidates = realty_candidates(context)
    viewings = _dict_list(realty.get("viewings"))
    focus_raw = realty.get("focusPropertyId")
    focus_id = str(focus_raw) if isinstance(focus_raw, (str, int)) and focus_raw else None

    lines: list[str] = ["", "PROPIEDADES (contexto vivo de esta conversación):"]
    lines += _requirement_lines(requirement)
    lines += _candidates_lines(candidates)
    if focus_id:
        lines.append(f"- Propiedad en foco ahora mismo: id={focus_id}.")
    lines += _viewings_lines(viewings)
    return "\n".join(lines)
