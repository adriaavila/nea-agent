"""Verticales de negocio de Nea.

Un vertical es un chasis (`app/prompt.py`) y un set de tools (`app/tools.py`)
ALTERNOS a los de allok/B2B, para un tipo de negocio distinto — hoy, bienes
raíces (`app/verticals/inmobiliario/`, Rei CRM).

Se activa por `profile.vertical` (ver `app/profile.py`, `app/stateless.py`):
sin él (default), TODO el comportamiento es exactamente el de siempre. Cada
vertical vive en su propio paquete y se engancha al turno con unas pocas
líneas en `app/stateless.run_turn` — nunca al revés (el chasis/tools por
defecto no saben que los verticales existen).
"""
from __future__ import annotations
