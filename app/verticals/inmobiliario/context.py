"""Lectura y render de `context.realty` (vertical inmobiliario).

`context.realty` es EXCLUSIVO de organizaciones con `profile.vertical ==
"inmobiliario"` — el resto del contrato de `GET /api/bot/context` no cambia.
Este módulo NUNCA llama al CRM: solo lee el bloque que ya vino en el
despacho y lo compacta para el prompt (`app/verticals/inmobiliario/
prompt.py`). Las candidatas que aquí se leen son la ÚNICA fuente de verdad
que el chasis tiene permitido mencionar sin haber llamado antes a una tool
(`guardar_requerimiento`/`ver_propiedad`) que regrese las suyas propias,
frescas — ver `app/verticals/inmobiliario/tools.py`.
"""
from __future__ import annotations

from typing import Any


def realty_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """El bloque `context.realty` tal cual, tolerante a su ausencia (el CRM
    aún no lo manda en este turno, o la migración de la organización está a
    medias)."""
    realty = (context or {}).get("realty")
    return realty if isinstance(realty, dict) else {}


def candidate_ids(context: dict[str, Any] | None) -> frozenset[str]:
    """Los ids de las candidatas que el CRM ya ofreció en `context.realty` —
    útil para pruebas/instrumentación que quieran verificar que el chasis
    nunca menciona una propiedad fuera de este catálogo (o de una tool)."""
    candidates = realty_context(context).get("candidates") or []
    return frozenset(
        str(c["id"]) for c in candidates if isinstance(c, dict) and c.get("id")
    )


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
    "minBedrooms": "recámaras mínimas",
    "minBathrooms": "baños mínimos",
    "paymentMethod": "forma de pago",
    "urgency": "urgencia",
    "amenities": "amenidades",
    "notes": "notas",
}


def _requirement_lines(requirement: dict[str, Any]) -> list[str]:
    if not requirement:
        return ["- Requerimiento: todavía no se sabe nada — empieza a calificar."]
    lines = ["- Requerimiento guardado hasta ahora:"]
    for key, label in _REQUIREMENT_LABELS.items():
        value = requirement.get(key)
        if value in (None, "", []):
            continue
        value_str = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"  - {label}: {value_str}")
    presupuesto_min = _fmt_money(requirement.get("budgetMin"), requirement.get("currency"))
    presupuesto_max = _fmt_money(requirement.get("budgetMax"), requirement.get("currency"))
    if presupuesto_min or presupuesto_max:
        rango = " a ".join(p for p in (presupuesto_min, presupuesto_max) if p)
        lines.append(f"  - presupuesto: {rango}")
    missing = requirement.get("missing") or []
    if missing:
        lines.append(
            "  - AÚN FALTA (pregunta esto, uno a la vez): " + ", ".join(str(m) for m in missing)
        )
    elif requirement.get("complete"):
        lines.append("  - requerimiento completo: ya puedes proponer candidatas")
    return lines


def _candidates_lines(candidates: list[Any]) -> list[str]:
    if not candidates:
        return [
            "- Candidatas: ninguna todavía. NO inventes propiedades — guarda "
            "el requerimiento (guardar_requerimiento) y espera candidatas reales."
        ]
    lines = [
        "- Candidatas disponibles (SOLO estas; máximo 3 por mensaje, una "
        "razón corta cada una):"
    ]
    for c in candidates:
        if not isinstance(c, dict):
            continue
        precio = _fmt_money(c.get("price"), c.get("currency")) or "precio no disponible"
        zona = c.get("zone") or c.get("city") or "zona sin dato"
        razones = "; ".join(str(r) for r in (c.get("reasons") or [])) or "sin razón registrada"
        titulo = c.get("title") or c.get("kind") or "propiedad"
        lines.append(f"  - id={c.get('id')} · {titulo} · {precio} · {zona} · {razones}")
    return lines


def _viewings_lines(viewings: list[Any]) -> list[str]:
    if not viewings:
        return []
    lines = ["- Visitas ya agendadas:"]
    for v in viewings:
        if not isinstance(v, dict):
            continue
        titulo = v.get("propertyTitle") or v.get("propertyId") or "propiedad"
        cuando = v.get("label") or v.get("startUtc") or "sin fecha"
        estado = v.get("status") or "confirmada"
        lines.append(f"  - {titulo}: {cuando} ({estado}).")
    return lines


def render_realty_block(context: dict[str, Any] | None) -> str:
    """Bloque compacto para CONTEXTO ACTUAL: requerimiento + faltantes,
    candidatas con id/precio/zona/razones, propiedad en foco y visitas
    agendadas. Tolerante a `context.realty` ausente — el modelo debe saber
    que sigue en modo inmobiliario (calificando) aunque el CRM todavía no
    haya mandado nada."""
    realty = realty_context(context)
    requirement = realty.get("requirement") or {}
    candidates = realty.get("candidates") or []
    focus_id = realty.get("focusPropertyId")
    viewings = realty.get("viewings") or []

    lines: list[str] = ["", "PROPIEDADES (contexto vivo de esta conversación):"]
    lines += _requirement_lines(requirement)
    lines += _candidates_lines(candidates)
    if focus_id:
        lines.append(f"- Propiedad en foco ahora mismo: id={focus_id}.")
    lines += _viewings_lines(viewings)
    return "\n".join(lines)
