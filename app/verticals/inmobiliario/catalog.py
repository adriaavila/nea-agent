"""Espejo Python de los catálogos cerrados del vertical inmobiliario.

Fuente de verdad: `src/lib/realty/catalog.ts` de vocero-inmobiliario (Rei
CRM). Son constantes, no lógica — mismo principio que ese archivo: ampliar
un catálogo es una migración consciente compartida con el CRM, nunca un
string suelto en el esquema de una tool.

Las CLAVES son las que viaja en el JSON (idénticas al TS); las etiquetas en
español son solo para RENDERIZAR en el prompt (ver `context.py`).
"""
from __future__ import annotations

OPERATIONS: tuple[str, ...] = ("renta", "venta", "anticretico")
PROPERTY_KINDS: tuple[str, ...] = (
    "casa",
    "departamento",
    "local",
    "terreno",
    "oficina",
    "bodega",
)
CURRENCIES: tuple[str, ...] = ("USD", "BOB", "MXN")
AMENITIES: tuple[str, ...] = (
    "estacionamiento",
    "alberca",
    "jardin",
    "roof_garden",
    "seguridad",
    "elevador",
    "gimnasio",
    "amueblado",
    "acepta_mascotas",
    "bodega",
    "terraza",
    "cocina_integral",
    "aire_acondicionado",
    "cisterna",
)
PAYMENT_METHODS: tuple[str, ...] = (
    "contado",
    "credito_bancario",
    "credito_vis",
    "infonavit",
    "fovissste",
    "otro",
)
URGENCIES: tuple[str, ...] = ("alta", "media", "baja")
PROPERTY_STATUSES: tuple[str, ...] = ("disponible", "apartada", "cerrada")

#: Etiqueta en español SOLO para el bloque de contexto del prompt (ver
#: context.py) — la clave en base/JSON no cambia de mercado, la etiqueta sí
#: (mismo criterio que PROPERTY_STATUS_LABELS del CRM: "apartada" siempre se
#: LEE "apartada" en la base, pero se MUESTRA "reservada").
PROPERTY_STATUS_LABELS: dict[str, str] = {
    "disponible": "disponible",
    "apartada": "reservada",
    "cerrada": "cerrada",
}
