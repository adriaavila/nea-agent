"""Recursos por organización para el modo de despacho multi-tenant.

Con un CRM multitenant (vocero-crm sirviendo a varias organizaciones) una sola
Nea compartida no puede tener un CrmClient ni un ProfileProvider fijos: cada
organización necesita que sus llamadas /api/bot/* lleven su propio
`X-Organization-Id`, y su perfil de negocio (nombre, tono, KB) NUNCA debe
mezclarse con el de otra.

A diferencia de un modelo con credenciales por organización, aquí todas
comparten `CRM_BASE_URL` y `CRM_BOT_API_KEY` (es el contrato del despacho: el
CRM y Nea comparten un solo secreto para firmar/verificar `/dispatch`) — lo
único que cambia por organización es el header que le dice al CRM a quién
enrutar, y el perfil que se cachea.

`organization_id=None` es siempre el camino legacy de un solo negocio
(webhook de Meta + relay): las funciones de aquí lo devuelven sin tocar,
apuntando al `crm`/`profile` de siempre en el AppContext.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from app.crm import CrmClient
from app.profile import ProfileProvider
from app.state import AppContext


def crm_for(ctx: AppContext, organization_id: str | None) -> Any:
    """El CrmClient de ESA organización (cacheado), o el legacy si es None."""
    if organization_id is None:
        return ctx.crm
    client = ctx.crm_clients.get(organization_id)
    if client is None:
        client = CrmClient(
            ctx.settings.crm_base_url,
            ctx.settings.crm_bot_api_key,
            organization_id=organization_id,
        )
        ctx.crm_clients[organization_id] = client
    return client


def profile_for(ctx: AppContext, organization_id: str | None) -> Any:
    """El ProfileProvider de ESA organización (cacheado), o el legacy si es
    None. Nunca comparte el TTL/cache con otra organización."""
    if organization_id is None:
        return ctx.profile
    provider = ctx.profile_providers.get(organization_id)
    if provider is None:
        provider = ProfileProvider(
            crm_for(ctx, organization_id),
            default_name=ctx.settings.agent_name,
            brief_path=ctx.settings.brief_path or None,
        )
        ctx.profile_providers[organization_id] = provider
    return provider


def scoped_ctx(ctx: AppContext, organization_id: str | None) -> AppContext:
    """Copia superficial del AppContext con el `crm`/`profile` de ESA
    organización. Todo lo demás (store, llm, settings, workers) es
    compartido: solo cambian las credenciales/caché con las que el turno
    habla con el CRM.

    `organization_id=None` regresa el MISMO ctx (cero costo, cero
    comportamiento nuevo para el webhook de Meta de siempre).
    """
    if organization_id is None:
        return ctx
    return dataclasses.replace(
        ctx,
        crm=crm_for(ctx, organization_id),
        profile=profile_for(ctx, organization_id),
    )
