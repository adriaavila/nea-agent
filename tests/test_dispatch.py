"""Modo de despacho multi-organización (POST /dispatch).

Cubre: firma HMAC sobre el cuerpo crudo (y el rechazo con secreto vacío),
aislamiento estricto entre organizaciones (CrmClient, ProfileProvider,
bot_conversation), el atajo de conversaciones de prueba del Laboratorio, y el
dedup de mensajes ya procesados. El camino legacy (webhook de Meta) se
verifica sin cambios en tests/test_webhook.py y tests/test_turn.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Any

import httpx

from app.multiorg import profile_for
from tests.conftest import CRM_URL, IDENTITY, crm_context, mock_crm_basics

SECRET = "test-key"  # CRM_BOT_API_KEY por defecto en tests.conftest.make_settings


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def dispatch_payload(
    *,
    organization_id: str | None = "org_a",
    conversation_id: str | None = "cv_dispatch_1",
    identity: str = IDENTITY,
    text: str = "hola",
    wamid: str | None = "wamid.dsp1",
    is_test: bool = False,
    name: str | None = "Ana",
) -> dict[str, Any]:
    """Un cuerpo de despacho fiel al contrato fijo con el CRM."""
    payload: dict[str, Any] = {
        "organizationId": organization_id,
        "conversationId": conversation_id,
        "isTest": is_test,
        "contact": {"identity": identity, "name": name},
        "messages": [
            {
                "id": wamid,
                "type": "text",
                "text": text,
                "mediaId": None,
                "timestamp": "2026-09-25T12:00:00Z",
            }
        ],
    }
    return payload


def dispatch_body(**kwargs: Any) -> bytes:
    return json.dumps(dispatch_payload(**kwargs)).encode("utf-8")


def mock_profile_404(respx_mock: Any) -> None:
    """mock_crm_basics no cubre /api/bot/profile; 404 = perfil mínimo, que es
    justo lo que necesitan las pruebas que no versan sobre el perfil."""
    respx_mock.get(f"{CRM_URL}/api/bot/profile").mock(return_value=httpx.Response(404))


# ------------------------------------------------------------------ firma ---


async def test_dispatch_sin_firma_401(client):
    resp = await client.post("/dispatch", content=dispatch_body())
    assert resp.status_code == 401


async def test_dispatch_firma_invalida_401(client):
    body = dispatch_body()
    resp = await client.post(
        "/dispatch", content=body, headers={"x-signature": "sha256=" + "0" * 64}
    )
    assert resp.status_code == 401


async def test_dispatch_api_key_vacia_rechaza_aunque_la_firma_sea_valida(ctx, client):
    body = dispatch_body()
    firma = sign(body)  # firmada con la key ORIGINAL, antes de vaciarla
    ctx.settings.crm_bot_api_key = ""
    resp = await client.post("/dispatch", content=body, headers={"x-signature": firma})
    assert resp.status_code == 401


async def test_dispatch_cuerpo_malformado_400(client):
    payload = dispatch_payload()
    del payload["organizationId"]
    body = json.dumps(payload).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


# ------------------------------------------------------------- turno real ---


async def test_dispatch_valido_corre_el_turno_con_header_de_organizacion(
    ctx, client, respx_mock
):
    routes = mock_crm_basics(respx_mock, conv_id="cv_dispatch_1")
    mock_profile_404(respx_mock)
    body = dispatch_body(organization_id="org_a", conversation_id="cv_dispatch_1")

    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert routes["messages"].call_count == 1
    # TODAS las llamadas /api/bot/* de este turno llevan X-Organization-Id.
    for route in (routes["context"], routes["typing"], routes["messages"]):
        assert route.calls, f"{route} nunca se llamó"
        for call in route.calls:
            assert call.request.headers["x-organization-id"] == "org_a"
    # Modo despacho: sin relay (el CRM ya tenía el mensaje).
    assert ctx.store.relays == {}


async def test_ids_null_del_laboratorio_nunca_se_dedupean(ctx, client, respx_mock):
    """Contrato: el CRM manda `id: null` para mensajes de Lab/prueba (wamids
    reales en todo lo demás). mark_processed jamás se llama con un id nulo —
    dos despachos de prueba distintos NUNCA se pisan entre sí por dedup."""
    mock_crm_basics(respx_mock, conv_id="cv_lab_null")
    mock_profile_404(respx_mock)

    body = dispatch_body(
        organization_id="org_a", conversation_id="cv_lab_null", is_test=True, wamid=None
    )
    resp1 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp1.status_code == 200
    resp2 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp2.status_code == 200
    assert len(ctx.llm.calls) == 2  # ninguno de los dos se dedupeó


async def test_dispatch_ids_duplicados_no_corren_segundo_turno(ctx, client, respx_mock):
    mock_crm_basics(respx_mock, conv_id="cv_dup_1")
    mock_profile_404(respx_mock)
    body = dispatch_body(organization_id="org_a", conversation_id="cv_dup_1", wamid="wamid.dup1")

    resp1 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp1.status_code == 200
    assert len(ctx.llm.calls) == 1

    resp2 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp2.status_code == 200
    assert len(ctx.llm.calls) == 1  # el segundo despacho es puro duplicado


async def test_dispatch_isTest_usa_identidad_de_laboratorio_e_ignora_allowlist(
    ctx, client, respx_mock
):
    # Allowlist encendida y sin incluir a nadie de esta prueba: si NO se
    # ignorara, el turno se quedaría en silencio.
    ctx.settings.allowed_wa_ids = "5219999999999"
    mock_crm_basics(respx_mock, conv_id="cv_lab_1")
    mock_profile_404(respx_mock)

    body = dispatch_body(
        organization_id="org_a",
        conversation_id="cv_lab_1",
        identity="cualquier-cosa",
        is_test=True,
        wamid="wamid.lab1",
    )
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})

    assert resp.status_code == 200
    assert len(ctx.llm.calls) == 1  # el turno SÍ corrió pese a la allowlist
    convs = list(ctx.store.conversations.values())
    assert len(convs) == 1
    assert convs[0].wa_identity == "test:cv_lab_1"
    assert convs[0].organization_id == "org_a"
    # MENOR: una conversación de prueba no es un lead real — sin seguimiento.
    assert convs[0].followup_due_at is None


# ------------------------------------------------------- aislamiento ---


async def test_dos_organizaciones_misma_identidad_son_conversaciones_separadas(
    ctx, client, respx_mock
):
    mock_crm_basics(respx_mock, conv_id="cv_org_a")
    mock_profile_404(respx_mock)
    body_a = dispatch_body(
        organization_id="org_a", conversation_id="cv_org_a", identity=IDENTITY, wamid="wamid.a1"
    )
    body_b = dispatch_body(
        organization_id="org_b", conversation_id="cv_org_b", identity=IDENTITY, wamid="wamid.b1"
    )

    resp_a = await client.post("/dispatch", content=body_a, headers={"x-signature": sign(body_a)})
    resp_b = await client.post("/dispatch", content=body_b, headers={"x-signature": sign(body_b)})
    assert resp_a.status_code == 200 and resp_b.status_code == 200

    same_identity = [c for c in ctx.store.conversations.values() if c.wa_identity == IDENTITY]
    assert len(same_identity) == 2
    assert {c.organization_id for c in same_identity} == {"org_a", "org_b"}

    # Cada conversación tiene SU propio historial — nada se mezcla.
    conv_a = next(c for c in same_identity if c.organization_id == "org_a")
    conv_b = next(c for c in same_identity if c.organization_id == "org_b")
    hist_a_ids = {m.id for m in ctx.store.messages if m.conversation_id == conv_a.id}
    hist_b_ids = {m.id for m in ctx.store.messages if m.conversation_id == conv_b.id}
    assert hist_a_ids and hist_b_ids
    assert hist_a_ids.isdisjoint(hist_b_ids)


async def test_perfil_de_una_organizacion_no_se_sirve_a_otra(ctx, client, respx_mock):
    mock_crm_basics(respx_mock, conv_id="cv_org_a")
    profile_a = respx_mock.get(
        f"{CRM_URL}/api/bot/profile", headers={"x-organization-id": "org_a"}
    ).mock(return_value=httpx.Response(200, json={"profile": {"name": "Agente A"}}))
    profile_b = respx_mock.get(
        f"{CRM_URL}/api/bot/profile", headers={"x-organization-id": "org_b"}
    ).mock(return_value=httpx.Response(200, json={"profile": {"name": "Agente B"}}))

    body_a = dispatch_body(organization_id="org_a", conversation_id="cv_org_a", wamid="wamid.pa")
    body_b = dispatch_body(organization_id="org_b", conversation_id="cv_org_b", wamid="wamid.pb")
    await client.post("/dispatch", content=body_a, headers={"x-signature": sign(body_a)})
    await client.post("/dispatch", content=body_b, headers={"x-signature": sign(body_b)})

    assert profile_a.call_count == 1
    assert profile_b.call_count == 1

    prof_a = await profile_for(ctx, "org_a").get()
    prof_b = await profile_for(ctx, "org_b").get()
    assert prof_a.agent_name == "Agente A"
    assert prof_b.agent_name == "Agente B"


async def test_perfil_de_organizacion_no_usa_el_brief_del_despliegue(
    ctx, client, respx_mock, tmp_path
):
    """BRIEF_PATH es el brief local de ESTE despliegue (un solo negocio, para
    correr sin CRM) — filtrarlo a una organización del despacho cuyo perfil
    diera 404 sería servirle el negocio del dueño del despliegue a un
    tercero."""
    brief = tmp_path / "brief.md"
    brief.write_text("Somos Clínica Ejemplo — brief del DESPLIEGUE legacy.")
    ctx.settings.brief_path = str(brief)

    mock_crm_basics(respx_mock, conv_id="cv_brief_1")
    respx_mock.get(f"{CRM_URL}/api/bot/profile").mock(return_value=httpx.Response(404))

    body = dispatch_body(organization_id="org_a", conversation_id="cv_brief_1", wamid="wamid.brief1")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 200

    prof = await profile_for(ctx, "org_a").get()
    assert prof.instructions is None  # NO absorbió el brief del deployment
    assert prof.agent_name == ctx.settings.agent_name  # perfil mínimo, no el brief


# --------------------------------------------------- claim/release de ids ---


async def test_dispatch_falla_libera_ids_y_el_reintento_corre_el_turno(
    ctx, client, respx_mock
):
    """CRÍTICO: si el turno revienta, el id reclamado por mark_processed se
    suelta — si no, el reintento del CRM con el MISMO body encuentra el
    mensaje "ya procesado" y responde 200 sin correr nada: el mensaje se
    pierde en silencio."""
    mock_crm_basics(respx_mock, conv_id="cv_retry_1")
    mock_profile_404(respx_mock)
    ctx.llm.raise_exc = RuntimeError("boom no manejado")

    body = dispatch_body(
        organization_id="org_a", conversation_id="cv_retry_1", wamid="wamid.retry1"
    )
    resp1 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp1.status_code == 500
    assert len(ctx.llm.calls) == 1

    ctx.llm.raise_exc = None  # el CRM reintenta el job con el MISMO body
    resp2 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp2.status_code == 200
    assert len(ctx.llm.calls) == 2  # el turno SÍ corrió en el reintento


async def test_dispatch_context_503_es_5xx_y_el_reintento_corre_el_turno(
    ctx, client, respx_mock
):
    """IMPORTANTE: un CrmError real en /api/bot/context (no un simple "no lo
    conozco") no puede volverse silencio+200 — dispatch.py debe verlo y
    responder 5xx para que el CRM reintente."""
    mock_profile_404(respx_mock)
    context_route = respx_mock.get(f"{CRM_URL}/api/bot/context").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=crm_context(conv_id="cv_503_1")),
        ]
    )
    messages_route = respx_mock.post(f"{CRM_URL}/api/bot/messages").mock(
        return_value=httpx.Response(200, json={"messageId": "msg_1"})
    )
    respx_mock.post(f"{CRM_URL}/api/bot/typing").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    body = dispatch_body(
        organization_id="org_a", conversation_id="cv_503_1", wamid="wamid.503a"
    )
    resp1 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp1.status_code == 500
    assert context_route.call_count == 1  # modo despacho: UN solo intento
    assert messages_route.call_count == 0  # nunca llegó a correr el turno

    resp2 = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp2.status_code == 200
    assert context_route.call_count == 2
    assert messages_route.call_count == 1  # el reintento SÍ corrió el turno


async def test_dispatch_timeout_es_5xx_y_libera_ids(ctx, client, respx_mock, monkeypatch):
    """MENOR: un turno que se cuelga (CRM real da por muerto el job a los
    ~90 s) no debe dejar la conexión colgada — responde 5xx antes, y libera
    los ids reclamados como cualquier otra falla."""
    from app import dispatch as dispatch_module

    monkeypatch.setattr(dispatch_module, "DISPATCH_TIMEOUT_SECONDS", 0.05)
    mock_crm_basics(respx_mock, conv_id="cv_timeout_1")
    mock_profile_404(respx_mock)

    original_complete = ctx.llm.complete

    async def lenta(messages: Any, tools: Any = None) -> Any:
        await asyncio.sleep(0.3)
        return await original_complete(messages, tools=tools)

    ctx.llm.complete = lenta  # type: ignore[method-assign]

    body = dispatch_body(
        organization_id="org_a", conversation_id="cv_timeout_1", wamid="wamid.timeout1"
    )
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 500
    assert "wamid.timeout1" not in ctx.store.processed  # liberado


# ---------------------------------------------------------- concurrencia ---


async def test_dispatches_concurrentes_misma_identidad_se_serializan(
    ctx, client, respx_mock
):
    """El lock por (organización, identidad) evita que dos despachos
    concurrentes de la MISMA conversación corran el turno a la vez — sin él,
    los dos podrían contestar y pisarse el abandon_pending_sends."""
    mock_crm_basics(respx_mock, conv_id="cv_lock_1")
    mock_profile_404(respx_mock)

    order: list[str] = []
    original_complete = ctx.llm.complete

    async def lenta(messages: Any, tools: Any = None) -> Any:
        order.append("start")
        await asyncio.sleep(0.05)
        order.append("end")
        return await original_complete(messages, tools=tools)

    ctx.llm.complete = lenta  # type: ignore[method-assign]

    body1 = dispatch_body(
        organization_id="org_a", conversation_id="cv_lock_1", wamid="wamid.lock1"
    )
    body2 = dispatch_body(
        organization_id="org_a", conversation_id="cv_lock_1", wamid="wamid.lock2"
    )

    results = await asyncio.gather(
        client.post("/dispatch", content=body1, headers={"x-signature": sign(body1)}),
        client.post("/dispatch", content=body2, headers={"x-signature": sign(body2)}),
    )
    assert [r.status_code for r in results] == [200, 200]
    # Serializado: nunca dos "start" seguidos — el segundo turno espera a que
    # el primero termine antes de empezar.
    assert order == ["start", "end", "start", "end"]


# ---------------------------------------------------------------- /reset ---


async def test_reset_usa_el_conversationid_del_despacho_no_el_guardado(
    ctx, client, respx_mock
):
    """MENOR: /reset debe preferir el conversationId que trajo ESTE despacho
    sobre el guardado en la conversación — el guardado puede haber quedado
    stale si la conversación se recreó del lado del CRM."""
    ctx.settings.allowed_wa_ids = IDENTITY  # /reset exige allowlist (legacy)
    mock_crm_basics(respx_mock, conv_id="cv_reset_new")
    mock_profile_404(respx_mock)
    reset_route = respx_mock.post(f"{CRM_URL}/api/bot/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    # Conversación ya existente con un crm_conversation_id VIEJO.
    conv = await ctx.store.get_or_create_conversation(IDENTITY, organization_id="org_a")
    await ctx.store.update_conversation(conv.id, crm_conversation_id="cv_reset_old")

    body = dispatch_body(
        organization_id="org_a",
        conversation_id="cv_reset_new",
        identity=IDENTITY,
        text="/reset",
        wamid="wamid.reset1",
    )
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})

    assert resp.status_code == 200
    assert reset_route.call_count == 1
    sent = json.loads(reset_route.calls[0].request.content)
    assert sent["conversationId"] == "cv_reset_new"  # el del despacho, no el viejo


# ------------------------------------------------------------- apagado ---


async def test_apagado_cierra_los_crmclients_por_organizacion(ctx):
    from app.main import create_app

    class FakeOrgCrm:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake = FakeOrgCrm()
    ctx.crm_clients["org_a"] = fake
    app = create_app(ctx=ctx)

    async with app.router.lifespan_context(app):
        pass

    assert fake.closed is True


# -------------------------------------------------------- entrada dura ---


async def test_dispatch_firma_no_ascii_es_401_no_500(client):
    """MENOR: un X-Signature con caracteres no-ASCII (basura o corrupción de
    red) no debe tumbar el endpoint — `hmac.compare_digest` truena con
    TypeError si se le pasan strings no-ASCII sin blindarlo antes."""
    body = dispatch_body()
    firma_no_ascii = ("sha256=" + "é" * 64).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": firma_no_ascii})
    assert resp.status_code == 401


async def test_dispatch_contact_no_es_objeto_400(client):
    payload = dispatch_payload()
    payload["contact"] = "no-es-un-objeto"
    body = json.dumps(payload).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_dispatch_messages_no_es_lista_400(client):
    payload = dispatch_payload()
    payload["messages"] = "no-es-una-lista"
    body = json.dumps(payload).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_dispatch_text_no_es_str_ni_null_400(client):
    payload = dispatch_payload()
    payload["messages"][0]["text"] = 12345
    body = json.dumps(payload).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400


async def test_dispatch_istest_no_bool_400(client):
    payload = dispatch_payload()
    payload["isTest"] = "true"  # string, no boolean real
    body = json.dumps(payload).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 400
