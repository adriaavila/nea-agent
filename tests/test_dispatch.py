"""Modo de despacho multi-organización (POST /dispatch).

Cubre: firma HMAC sobre el cuerpo crudo (y el rechazo con secreto vacío),
aislamiento estricto entre organizaciones (CrmClient, ProfileProvider,
bot_conversation), el atajo de conversaciones de prueba del Laboratorio, y el
dedup de mensajes ya procesados. El camino legacy (webhook de Meta) se
verifica sin cambios en tests/test_webhook.py y tests/test_turn.py.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx

from app.multiorg import profile_for
from tests.conftest import CRM_URL, IDENTITY, mock_crm_basics

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
