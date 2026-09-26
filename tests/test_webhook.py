"""Webhook: challenge del GET, firma del POST, dedup y multimedia."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

import httpx
import pytest

from app.main import create_app
from app.webhook import extract_inbound, verify_signature
from tests.conftest import (
    CRM_CONV_ID,
    CRM_URL,
    IDENTITY,
    FakeLLM,
    make_ctx,
    make_settings,
    mock_crm_basics,
    wa_body,
)

SECRET = "super-secreto"


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ------------------------------------------------------------ GET verify ---


async def test_get_verify_challenge_ok(client):
    resp = await client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "vtoken",
            "hub.challenge": "reto-123",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "reto-123"


async def test_get_verify_token_incorrecto(client):
    resp = await client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "otro",
            "hub.challenge": "reto-123",
        },
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------- firma ---


def test_verify_signature_pura():
    body = b'{"hola": 1}'
    ok = sign(body)
    assert verify_signature(body, ok, SECRET) is True
    assert verify_signature(body, "sha256=" + "0" * 64, SECRET) is False
    assert verify_signature(body, None, SECRET) is False
    assert verify_signature(body, "malformada", SECRET) is False
    # Sin secret configurado no se exige firma:
    assert verify_signature(body, None, None) is True
    assert verify_signature(body, "sha256=basura", None) is True


# ------------------------------------------------------------- identidad ---


def test_extract_inbound_canonicaliza_mx_521():
    """Meta manda `from: 521XXXXXXXXXX`; el CRM guarda 52XXXXXXXXXX — la
    identidad debe salir canónica desde el parseo (bug real de producción:
    el contexto del CRM daba 404 y Nea guardaba silencio)."""
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"profile": {"name": "Aurelio"}}],
                            "messages": [
                                {
                                    "id": "wamid.canon.1",
                                    "from": "5215550001111",
                                    "type": "text",
                                    "text": {"body": "hola"},
                                },
                                {
                                    "id": "wamid.canon.2",
                                    "from_user_id": "bsuid-abc",
                                    "type": "text",
                                    "text": {"body": "hola"},
                                },
                            ],
                        }
                    }
                ]
            }
        ]
    }
    inbound = extract_inbound(payload)
    assert [m.identity for m in inbound] == ["525550001111", "bsuid-abc"]


def test_echo_de_coexistence_no_abre_turno():
    """008: un `smb_message_echoes` (mensaje que el dueño mandó A MANO desde
    la app del teléfono) jamás es un entrante de lead — solo relay al CRM.
    Sin este guard, un echo bajo la clave `messages` abriría un turno de Nea
    hacia el propio número del negocio."""
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "field": "smb_message_echoes",
                        "value": {
                            "message_echoes": [
                                {
                                    "id": "wamid.echo.1",
                                    "from": "5215550009999",
                                    "to": "5215550002222",
                                    "type": "text",
                                    "text": {"body": "te contesto yo"},
                                }
                            ],
                            # variante defensiva: mismo contenido bajo `messages`
                            "messages": [
                                {
                                    "id": "wamid.echo.2",
                                    "from": "5215550009999",
                                    "type": "text",
                                    "text": {"body": "te contesto yo"},
                                }
                            ],
                        },
                    }
                ]
            }
        ]
    }
    assert extract_inbound(payload) == []


@pytest.fixture
def signed_ctx():
    return make_ctx(make_settings(meta_app_secret=SECRET))


@pytest.fixture
async def signed_client(signed_ctx):
    app = create_app(ctx=signed_ctx)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bot.test") as c:
        yield c
    await signed_ctx.crm.aclose()


async def test_post_firma_valida(signed_ctx, signed_client, respx_mock):
    mock_crm_basics(respx_mock)
    body = wa_body()
    resp = await signed_client.post(
        "/webhook", content=body, headers={"x-hub-signature-256": sign(body)}
    )
    assert resp.status_code == 200
    await asyncio.sleep(0.02)
    assert len(signed_ctx.store.relays) == 1  # el relay quedó encolado


async def test_post_firma_invalida(signed_ctx, signed_client):
    body = wa_body()
    resp = await signed_client.post(
        "/webhook", content=body, headers={"x-hub-signature-256": "sha256=" + "0" * 64}
    )
    assert resp.status_code == 401
    await asyncio.sleep(0.02)
    assert len(signed_ctx.store.relays) == 0


async def test_post_firma_ausente(signed_client):
    resp = await signed_client.post("/webhook", content=wa_body())
    assert resp.status_code == 401


async def test_post_firma_no_ascii_es_401_no_500(signed_client):
    """`hmac.compare_digest` truena con TypeError si se le pasa un string con
    caracteres no-ASCII sin blindarlo antes — un header corrupto no debe
    tumbar el endpoint con un 500."""
    body = wa_body()
    firma_no_ascii = ("sha256=" + "é" * 64).encode("utf-8")
    resp = await signed_client.post(
        "/webhook", content=body, headers={"x-hub-signature-256": firma_no_ascii}
    )
    assert resp.status_code == 401


async def test_post_sin_secret_no_exige_firma(ctx, client, respx_mock):
    mock_crm_basics(respx_mock)
    resp = await client.post("/webhook", content=wa_body())
    assert resp.status_code == 200


async def test_post_pide_reintento_si_no_puede_persistir(ctx, client, monkeypatch):
    async def db_down(body, signature):
        raise RuntimeError("db down")

    monkeypatch.setattr(ctx.store, "enqueue_relay", db_down)
    resp = await client.post("/webhook", content=wa_body())
    assert resp.status_code == 503


async def test_relay_only_releva_pero_no_corre_turno(ctx, client, respx_mock):
    """RELAY_ONLY=true (corte a modo despacho): el payload se releva al CRM
    exactamente igual que siempre, pero NO se corre turno — ni LLM ni ninguna
    llamada /api/bot/* (typing/context/messages), porque el CRM va a
    despachar ese mismo turno por /dispatch. Sin este corte, el lead recibiría
    dos respuestas para el mismo mensaje durante la transición."""
    routes = mock_crm_basics(respx_mock)
    ctx.settings.relay_only = True

    resp = await client.post("/webhook", content=wa_body())
    assert resp.status_code == 200
    await asyncio.sleep(0.2)  # tiempo de sobra para que un turno (si corriera) actuara

    assert len(ctx.store.relays) == 1  # el relay SÍ se encoló
    assert ctx.llm.calls == []  # pero ningún turno llamó al LLM
    for route in (routes["context"], routes["typing"], routes["messages"]):
        assert route.call_count == 0


async def test_arranque_persistente_exige_firma_de_meta(monkeypatch):
    """Sin META_APP_SECRET, un arranque con persistencia real truena — SIEMPRE,
    aunque CRM_BOT_API_KEY esté configurado. Un despliegue de un solo negocio
    tiene CRM_BOT_API_KEY por defecto (le habla al CRM igual); si la sola
    presencia de esa key bastara para dejar arrancar sin META_APP_SECRET, un
    despliegue clásico que se quedó sin el secreto bootearía en verde y le
    respondería 401 a cada entrega de Meta sin que nada avise (el guard
    relajado que esto reemplaza). Solo DISPATCH_ONLY=true, explícito, lo
    permite (ver el siguiente test)."""
    from app import main

    monkeypatch.setattr(
        main,
        "Settings",
        lambda: make_settings(
            database_url="postgresql://db/nea",
            meta_app_secret="",
            crm_bot_api_key="k",  # presente, como en CUALQUIER despliegue normal
            dispatch_only=False,
        ),
    )
    app = main.create_app()

    with pytest.raises(RuntimeError, match="META_APP_SECRET"):
        async with app.router.lifespan_context(app):
            pass


async def test_arranque_persistente_dispatch_only_no_exige_meta_app_secret(monkeypatch):
    """Con DISPATCH_ONLY=true, un despliegue de solo-despacho puede arrancar
    sin META_APP_SECRET (no habla con Meta — solo con el CRM)."""
    from app import main

    monkeypatch.setattr(
        main,
        "Settings",
        lambda: make_settings(
            database_url="postgresql://db/nea",
            meta_app_secret="",
            crm_bot_api_key="k",
            dispatch_only=True,
        ),
    )

    class FakePgStore:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        async def connect(self) -> None:
            pass

        async def migrate(self, migrations_dir) -> None:  # type: ignore[no-untyped-def]
            pass

        async def ping(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(main, "PgStore", FakePgStore)

    app = main.create_app()
    async with app.router.lifespan_context(app):
        pass  # no debe lanzar RuntimeError


async def test_arranque_dispatch_only_sin_database_url_no_usa_postgres(monkeypatch):
    """PR 2B: `DISPATCH_ONLY=true` SIN `DATABASE_URL` arranca igual — sin
    PgStore, sin migraciones, sin workers de fondo y sin /webhook. Solo queda
    /dispatch (v2 — v1 responde 503, ver test_dispatch_only_sin_db_v1_da_503)
    y /health, que debe contestar sin tocar ninguna base."""
    from app import main

    monkeypatch.setattr(
        main,
        "Settings",
        lambda: make_settings(
            database_url="",
            meta_app_secret="",
            crm_bot_api_key="k",
            dispatch_only=True,
        ),
    )

    def _revienta(*_a, **_kw):
        raise AssertionError("PgStore no debía construirse sin DATABASE_URL")

    monkeypatch.setattr(main, "PgStore", _revienta)

    app = main.create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bot.test") as c:
            resp = await c.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok", "db": "none"}
            # /webhook no se monta en este modo: 404, no 401 (ni verificado).
            webhook_resp = await c.get(
                "/webhook",
                params={"hub.mode": "subscribe", "hub.verify_token": "x", "hub.challenge": "y"},
            )
            assert webhook_resp.status_code == 404


async def test_dispatch_only_sin_db_v1_da_503(monkeypatch):
    """El mismo arranque sin base: un despacho v1 (sin `version`, o `1`) no
    tiene Store real que tocar — 503 claro, en vez de reventar contra una
    base inexistente. v2 sigue funcionando (test_stateless.py lo cubre)."""
    from app import main

    monkeypatch.setattr(
        main,
        "Settings",
        lambda: make_settings(
            database_url="",
            meta_app_secret="",
            crm_bot_api_key="test-key",
            dispatch_only=True,
        ),
    )
    monkeypatch.setattr(
        main, "PgStore", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("sin DB"))
    )

    app = main.create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bot.test") as c:
            from tests.test_dispatch import dispatch_body, sign

            body = dispatch_body(organization_id="org_a", conversation_id="cv_1")
            resp = await c.post("/dispatch", content=body, headers={"x-signature": sign(body)})
            assert resp.status_code == 503


async def test_webhook_deshabilitado_en_despliegue_dispatch_only(ctx, client):
    """Con DISPATCH_ONLY=true y sin META_APP_SECRET, el webhook de Meta
    rechaza TODO en vez de aceptar payloads sin firma."""
    ctx.settings.dispatch_only = True
    ctx.settings.meta_app_secret = ""

    resp = await client.post("/webhook", content=wa_body())
    assert resp.status_code == 401


async def test_webhook_sin_dispatch_only_sigue_aceptando_sin_firma(ctx, client, respx_mock):
    """Sin DISPATCH_ONLY (el default), un secreto vacío sigue siendo el modo
    dev de siempre — el webhook de Meta NO se deshabilita solo porque haya
    persistencia real configurada. La bandera es la única señal."""
    mock_crm_basics(respx_mock)
    ctx.settings.database_url = "postgresql://db/nea"
    ctx.settings.meta_app_secret = ""
    ctx.settings.dispatch_only = False

    resp = await client.post("/webhook", content=wa_body())
    assert resp.status_code == 200


# ---------------------------------------------------------------- dedup ---


async def test_dedup_mismo_wamid_una_sola_respuesta(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    fake_llm: FakeLLM = ctx.llm
    body = wa_body(wamid="wamid.dup")
    await client.post("/webhook", content=body)
    await client.post("/webhook", content=body)  # re-entrega de Meta
    await asyncio.sleep(0.25)
    assert len(fake_llm.calls) == 1
    assert routes["messages"].call_count == 1
    # ambos payloads sí se relevan al CRM (bandeja íntegra)
    assert len(ctx.store.relays) == 2


async def test_fantasma_unsupported_no_bloquea_la_reentrega(ctx, client, respx_mock):
    """Incidente real (2026-07-30, lead Felipe): Meta entrega primero un
    `unsupported` con errors 131060 ("message unavailable") y ~200 ms después
    RE-ENTREGA el mensaje real (texto + referral) con el MISMO wamid. El
    fantasma no debe abrir turno ni marcar dedup — la re-entrega es la buena."""
    routes = mock_crm_basics(respx_mock)
    fake_llm: FakeLLM = ctx.llm

    from tests.conftest import wa_payload

    fantasma = wa_payload(wamid="wamid.ghost")
    msg = fantasma["entry"][0]["changes"][0]["value"]["messages"][0]
    del msg["text"]
    msg["type"] = "unsupported"
    msg["unsupported"] = {"type": "unknown"}
    msg["errors"] = [{"code": 131060, "title": "This message is unavailable."}]

    await client.post("/webhook", content=json.dumps(fantasma).encode())
    await asyncio.sleep(0.15)
    assert len(fake_llm.calls) == 0  # el fantasma no abre turno

    # re-entrega real de Meta: mismo wamid, ahora con texto
    await client.post("/webhook", content=wa_body(wamid="wamid.ghost", text="¡Hola! Quiero más información"))
    await asyncio.sleep(0.25)
    assert len(fake_llm.calls) == 1  # la re-entrega SÍ se responde
    assert routes["messages"].call_count == 1
    assert len(ctx.store.relays) == 2  # ambos payloads relevados al CRM


async def test_unsupported_sin_errors_si_abre_turno(ctx, client, respx_mock):
    """Un `unsupported` genuino (sin errors) conserva el fallback amable de
    Nea ("no puedo ver eso") — solo el fantasma con errors se ignora."""
    mock_crm_basics(respx_mock)
    from tests.conftest import wa_payload

    payload = wa_payload(wamid="wamid.unsup")
    msg = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    del msg["text"]
    msg["type"] = "unsupported"
    msg["unsupported"] = {"type": "unknown"}

    await client.post("/webhook", content=json.dumps(payload).encode())
    await asyncio.sleep(0.25)
    assert len(ctx.llm.calls) == 1


# ----------------------------------------------------- payloads raros ------


async def test_payload_sin_identidad_no_truena(ctx, client, respx_mock):
    mock_crm_basics(respx_mock)
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"id": "wamid.x", "type": "text", "text": {"body": "hola"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    resp = await client.post("/webhook", content=json.dumps(payload).encode())
    assert resp.status_code == 200
    await asyncio.sleep(0.15)
    assert len(ctx.llm.calls) == 0  # descartado con log, sin crash
    assert len(ctx.store.relays) == 1  # pero relevado


async def test_payload_de_statuses_solo_relay(ctx, client, respx_mock):
    mock_crm_basics(respx_mock)
    payload = {
        "entry": [
            {"changes": [{"value": {"statuses": [{"id": "wamid.s", "status": "read"}]}}]}
        ]
    }
    resp = await client.post("/webhook", content=json.dumps(payload).encode())
    assert resp.status_code == 200
    await asyncio.sleep(0.1)
    assert len(ctx.llm.calls) == 0
    assert len(ctx.store.relays) == 1


# ----------------------------------------------------------- multimedia ---


async def test_imagen_entra_al_turno_con_vision(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    await client.post("/webhook", content=wa_body(msg_type="image", wamid="wamid.img1"))
    await asyncio.sleep(0.3)
    assert routes["media"].call_count == 1  # descargó vía CRM
    assert len(ctx.llm.calls) == 1
    last_user = ctx.llm.calls[0]["messages"][-1]
    kinds = {p.get("type") for p in last_user["content"]}
    assert kinds == {"text", "image_url"}  # turno multimodal
    assert routes["messages"].call_count == 1  # Nea respondió al contenido


async def test_audio_se_transcribe_y_responde(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    await client.post("/webhook", content=wa_body(msg_type="audio", wamid="wamid.aud1"))
    await asyncio.sleep(0.3)
    assert len(ctx.llm.transcriptions) == 1  # se transcribió de verdad
    user_texts = " ".join(
        str(m["content"])
        for m in ctx.llm.calls[0]["messages"]
        if m["role"] == "user"
    )
    assert "transcrita" in user_texts
    assert ctx.llm.transcript_text in user_texts
    assert routes["messages"].call_count == 1


async def test_sticker_sigue_natural_sin_descarga(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    await client.post("/webhook", content=wa_body(msg_type="sticker", wamid="wamid.stk1"))
    await asyncio.sleep(0.3)
    assert routes["media"].call_count == 0  # el sticker no se descarga
    assert len(ctx.llm.calls) == 1
    assert routes["messages"].call_count == 1


async def test_typing_temprano_llega_antes_de_que_cierre_el_coalesce(respx_mock):
    """El "escribiendo…" no espera la ráfaga: sale ~medio segundo tras recibir
    (aquí acelerado), y la respuesta llega después, al cerrar el coalesce."""
    settings = make_settings(coalesce_seconds=0.4, typing_delay_seconds=0.01)
    ctx = make_ctx(settings)
    routes = mock_crm_basics(respx_mock)
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    await ctx.store.update_conversation(conv.id, crm_conversation_id=CRM_CONV_ID)

    app = create_app(ctx=ctx)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bot.test") as c:
        await c.post("/webhook", content=wa_body(text="hola", wamid="wamid.typ2"))
        await asyncio.sleep(0.15)
        # A mitad de la ventana de coalesce: typing YA salió, respuesta AÚN no.
        assert routes["typing"].call_count == 1
        assert routes["messages"].call_count == 0
        await asyncio.sleep(0.7)
    await ctx.crm.aclose()

    assert routes["messages"].call_count == 1  # el turno respondió al cerrar
    assert routes["typing"].call_count == 2  # temprano + refresco del turno


async def test_typing_se_dispara_y_su_fallo_no_afecta(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    routes["typing"].mock(return_value=httpx.Response(500))  # CRM/Meta caídos
    await client.post("/webhook", content=wa_body(text="hola", wamid="wamid.typ1"))
    await asyncio.sleep(0.3)
    assert routes["typing"].call_count >= 1  # se intentó la señal de vida
    assert routes["messages"].call_count == 1  # y el turno respondió igual


async def test_reaccion_no_abre_turno(ctx, client, respx_mock):
    mock_crm_basics(respx_mock)
    await client.post("/webhook", content=wa_body(msg_type="reaction", wamid="wamid.rx1"))
    await asyncio.sleep(0.2)
    assert len(ctx.llm.calls) == 0  # solo relay, sin turno
    assert len(ctx.store.relays) == 1


# ------------------------------------------------------- comando /reset ---


async def test_reset_de_linea_de_pruebas_borra_memoria(respx_mock):
    settings = make_settings(allowed_wa_ids=IDENTITY)
    ctx = make_ctx(settings)
    routes = mock_crm_basics(respx_mock)
    reset_route = respx_mock.post(f"{CRM_URL}/api/bot/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    conv = await ctx.store.get_or_create_conversation(IDENTITY)
    await ctx.store.add_message(conv.id, "user", "hola")
    await ctx.store.add_message(conv.id, "assistant", "¡Hola!")
    await ctx.store.update_conversation(
        conv.id, crm_conversation_id=CRM_CONV_ID, greeted=True, phase="agendando"
    )

    app = create_app(ctx=ctx)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bot.test") as c:
        await c.post("/webhook", content=wa_body(text="/reset", wamid="wamid.rst1"))
        await asyncio.sleep(0.3)
    await ctx.crm.aclose()

    assert reset_route.call_count == 1  # el CRM también se reinició
    assert await ctx.store.recent_messages(conv.id, 50) == []  # memoria fuera
    fresh = await ctx.store.get_or_create_conversation(IDENTITY)
    assert fresh.greeted is False and fresh.phase == "descubrimiento"
    assert len(ctx.llm.calls) == 0  # el comando jamás llega al LLM
    assert routes["messages"].call_count == 1  # confirmación al tester


async def test_reset_sin_allowlist_es_texto_normal(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    reset_route = respx_mock.post(f"{CRM_URL}/api/bot/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    await client.post("/webhook", content=wa_body(text="/reset", wamid="wamid.rst2"))
    await asyncio.sleep(0.3)
    assert reset_route.call_count == 0  # con la allowlist vacía no hay comando
    assert len(ctx.llm.calls) == 1  # se trató como un mensaje cualquiera
    assert routes["messages"].call_count == 1


async def test_media_caida_degrada_honesto(ctx, client, respx_mock):
    routes = mock_crm_basics(respx_mock)
    routes["media"].mock(return_value=httpx.Response(500))
    await client.post("/webhook", content=wa_body(msg_type="audio", wamid="wamid.aud2"))
    await asyncio.sleep(0.3)
    assert routes["messages"].call_count == 1  # responde igual, sin romperse
    user_texts = " ".join(
        str(m["content"])
        for m in ctx.llm.calls[0]["messages"]
        if m["role"] == "user"
    )
    assert "NO pudiste abrir" in user_texts  # marcador honesto
