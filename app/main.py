"""App FastAPI de Nea: lifespan (migraciones + workers) y /health.

`create_app()` sin argumentos es el camino de producción (uvicorn app.main:app):
el lifespan conecta Postgres, aplica migraciones y arranca los workers de relay
y seguimiento. Los tests inyectan un `AppContext` ya armado (MemoryStore, LLM
fake, CRM contra respx) y manejan los workers a mano.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.coalesce import Coalescer
from app.config import Settings
from app.crm import CrmClient
from app.db import PgStore
from app.dispatch import router as dispatch_router
from app.followup import FollowupWorker
from app.llm import OpenAiLlm
from app.profile import ProfileProvider
from app.relay import RelayWorker
from app.sender import SenderWorker
from app.state import AppContext
from app.turn import handle_flush
from app.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nea.main")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _wire_coalescer(ctx: AppContext) -> None:
    if ctx.coalescer is None:
        ctx.coalescer = Coalescer(
            ctx.settings.coalesce_seconds, partial(handle_flush, ctx)
        )


def create_app(ctx: AppContext | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        own_resources = app.state.ctx is None
        if own_resources:
            settings = Settings()
            if settings.database_url and not settings.meta_app_secret:
                if not settings.dispatch_only:
                    # CRM_BOT_API_KEY casi siempre está configurado (TODO
                    # despliegue de un solo negocio también le habla al CRM),
                    # así que NO basta como señal de "esto es solo-despacho" —
                    # un despliegue clásico que se quedó sin META_APP_SECRET
                    # arrancaría en verde y le respondería 401 a cada entrega
                    # de Meta sin que nada avise. DISPATCH_ONLY tiene que ser
                    # explícito.
                    raise RuntimeError(
                        "META_APP_SECRET es obligatorio con persistencia habilitada, "
                        "salvo que DISPATCH_ONLY=true (despliegue de solo-despacho: "
                        "/webhook queda deshabilitado y solo corre /dispatch)"
                    )
                logger.warning(
                    "DISPATCH_ONLY=true sin META_APP_SECRET: /webhook va a "
                    "RECHAZAR TODAS las peticiones de Meta con 401 — esta "
                    "instancia solo atiende /dispatch. Si esto es un "
                    "despliegue de UN solo negocio, es una falla silenciosa: "
                    "revisa META_APP_SECRET/DISPATCH_ONLY ahora."
                )
            store = PgStore(settings.database_url)
            await store.connect()
            await store.migrate(MIGRATIONS_DIR)
            logger.info("migraciones aplicadas — DB lista")
            if settings.legacy_organization_id:
                # Cutover: adopta las filas que quedaron en el namespace
                # legacy (organization_id NULL) hacia esta organización.
                # Idempotente — en arranques posteriores no hay nada NULL que
                # mover y esto es un no-op instantáneo.
                moved_conv, moved_pending = await store.adopt_legacy_rows(
                    settings.legacy_organization_id
                )
                logger.warning(
                    "LEGACY_ORGANIZATION_ID=%s: %d conversaciones y %d "
                    "pending_send adoptados del namespace legacy",
                    settings.legacy_organization_id,
                    moved_conv,
                    moved_pending,
                )
            crm = CrmClient(settings.crm_base_url, settings.crm_bot_api_key)
            app.state.ctx = AppContext(
                settings=settings,
                store=store,
                crm=crm,
                llm=OpenAiLlm(
                    settings.openrouter_api_token or settings.openai_api_key,
                    settings.openrouter_model or settings.openai_model,
                    transcribe_model=settings.openai_transcribe_model,
                    base_url=(
                        "https://openrouter.ai/api/v1"
                        if settings.openrouter_api_token
                        else None
                    ),
                ),
                profile=ProfileProvider(
                    crm,
                    default_name=settings.agent_name,
                    brief_path=settings.brief_path or None,
                ),
            )
        c: AppContext = app.state.ctx
        _wire_coalescer(c)

        relay_worker = RelayWorker(c.store, c.settings.crm_webhook_url, c.relay_wake)
        followup_worker = FollowupWorker(c)
        sender_worker = SenderWorker(c)
        workers = [
            asyncio.create_task(relay_worker.run(), name="relay-worker"),
            asyncio.create_task(followup_worker.run(), name="followup-worker"),
            asyncio.create_task(sender_worker.run(), name="sender-worker"),
        ]
        logger.info("Nea arriba: relay + followup + sender corriendo")
        try:
            yield
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await relay_worker.aclose()
            if c.coalescer is not None:
                await c.coalescer.aclose()
            # Los CrmClient por organización (modo despacho, app/multiorg.py)
            # los abre este proceso bajo demanda — nadie más los cierra, a
            # diferencia del `c.crm` legacy que en tests gestiona el fixture.
            for org_crm in c.crm_clients.values():
                await org_crm.aclose()
            if own_resources:
                await c.crm.aclose()
                await c.store.aclose()

    app = FastAPI(title="Nea — agente de agendamiento para WhatsApp", lifespan=lifespan)
    app.state.ctx = ctx
    if ctx is not None:
        _wire_coalescer(ctx)
    app.include_router(webhook_router)
    app.include_router(dispatch_router)

    @app.get("/health")
    async def health(request: Request):  # type: ignore[no-untyped-def]
        c: AppContext | None = request.app.state.ctx
        if c is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            await c.store.ping()
        except Exception:
            logger.exception("health: la DB no responde")
            return JSONResponse(
                {"status": "degraded", "db": "error"}, status_code=503
            )
        return {"status": "ok", "db": "ok"}

    return app


app = create_app()
