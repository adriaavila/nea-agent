"""Vertical inmobiliario (Rei CRM): chasis, tools y contexto para comprar,
alquilar o anticretizar una propiedad, en vez del agendamiento B2B de allok.

Se activa cuando `profile.vertical == VERTICAL_NAME` (ver
`app/stateless.run_turn`, que es el ÚNICO lugar que importa este paquete).
Ver `README.md` → "Vertical inmobiliario" para el contrato completo con el
CRM, y `app/verticals/inmobiliario/{context,prompt,tools}.py` para el
detalle de cada pieza.
"""
from __future__ import annotations

from app.verticals.inmobiliario.prompt import build_system_prompt as build_prompt
from app.verticals.inmobiliario.tools import (
    DISABLED_TOOLS,
    REALTY_TOOL_SCHEMAS,
    build_realty_tools,
)

#: El valor de `profile.vertical` que activa este paquete (ver
#: `app/profile.py::profile_from_payload`).
VERTICAL_NAME = "inmobiliario"

__all__ = [
    "VERTICAL_NAME",
    "build_prompt",
    "REALTY_TOOL_SCHEMAS",
    "DISABLED_TOOLS",
    "build_realty_tools",
]
