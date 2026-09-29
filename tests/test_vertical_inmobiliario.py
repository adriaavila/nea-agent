"""Vertical inmobiliario (Rei CRM): selección por `profile.vertical`, el
esquema de tools nuevo, el redondeo de guardar_requerimiento/ver_propiedad,
la idempotencia de enviar_ficha, book/reschedule con `property_id`, el 404
`vertical_disabled` manejado sin reventar, y que el camino allok/B2B de
siempre NO CAMBIA EN NADA.

Helpers propios (no se toca tests/test_stateless.py ni ningún otro archivo
existente) para no competir con las dos PR abiertas que tocan app/prompt.py
y app/stateless.py (feat/chasis-ux-whatsapp, feat/followup-dispatch).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from app import stateless
from app.llm import LlmReply, ToolCall
from app.profile import profile_from_payload
from app.state import AppContext, TurnCommit
from app.tools import TOOL_SCHEMAS
from app.verticals import inmobiliario
from app.verticals.inmobiliario.context import render_realty_block
from app.verticals.inmobiliario.tools import REALTY_TOOL_SCHEMAS
from tests.conftest import CRM_URL, IDENTITY, make_ctx, mock_crm_basics

SLOT_ISO = "2026-07-20T16:00:00Z"
SECRET = "test-key"  # CRM_BOT_API_KEY por defecto en tests.conftest.make_settings


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def realty_raw(
    *,
    organization_id: str = "org_rei",
    conversation_id: str = "cv_rei_1",
    dispatch_id: str = "dsp_rei_1",
    vertical: str | None = "inmobiliario",
    identity: str = IDENTITY,
    history: list[dict[str, Any]] | None = None,
    offers: list[dict[str, Any]] | None = None,
    llm: dict[str, Any] | None = None,
    realty: dict[str, Any] | None = None,
    ai_enabled: bool = True,
    window_open: bool = True,
    agent_has_spoken: bool = True,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "contact": {
            "id": "ct_1",
            "name": "Compradora",
            "waIdentity": identity,
            "phone": identity,
            "ficha": {},
        },
        "conversation": {
            "id": conversation_id,
            "aiEnabled": ai_enabled,
            "handoffAt": None,
            "handoffReason": None,
            "windowOpen": window_open,
            "windowRemainingMs": 3600000,
            "agentHasSpoken": agent_has_spoken,
        },
        "lead": {"stageName": "Nuevo"},
        "agentAccess": {"allowlistEnabled": False, "allowedWaIds": []},
        "booking": {"next": None},
        "adOrigen": None,
    }
    if realty is not None:
        context["realty"] = realty
    profile: dict[str, Any] = {
        "profile": {
            "name": "Rei",
            "tone": None,
            "instructions": None,
            "escalationRules": None,
            "greeting": None,
            "activationEnabled": False,
            "activationMessages": [],
            "timezone": None,
        },
        "kb": None,
        "resources": [],
    }
    if vertical is not None:
        profile["vertical"] = vertical
    default_history = (
        history
        if history is not None
        else [
            {
                "id": "m1",
                "role": "lead",
                "type": "text",
                "text": "hola, busco depa",
                "at": "2026-09-25T12:00:00Z",
                "pending": True,
                "media": None,
            }
        ]
    )
    return {
        "version": 2,
        "dispatchId": dispatch_id,
        "attempt": 0,
        "organizationId": organization_id,
        "conversationId": conversation_id,
        "isTest": False,
        "context": context,
        "profile": profile,
        "history": default_history,
        "offers": offers or [],
        "llm": llm,
    }


def realty_payload(**kwargs: Any) -> stateless.DispatchPayloadV2:
    return stateless.DispatchPayloadV2.model_validate(realty_raw(**kwargs))


# ======================================================== selección de vertical ===


def test_perfil_sin_vertical_es_default():
    prof = profile_from_payload({"profile": {"name": "Nea"}}, default_name="Nea")
    assert prof.vertical is None


def test_perfil_con_vertical_top_level():
    prof = profile_from_payload(
        {"profile": {"name": "Rei"}, "vertical": "inmobiliario"}, default_name="Nea"
    )
    assert prof.vertical == "inmobiliario"


def test_perfil_con_vertical_anidado_tambien_se_lee():
    """Tolerancia extra si el CRM termina mandándolo DENTRO del objeto
    `profile` en vez de al nivel del cuerpo (ver la duda de contrato en el
    reporte del builder: "profile.vertical" admite las dos lecturas)."""
    prof = profile_from_payload(
        {"profile": {"name": "Rei", "vertical": "inmobiliario"}}, default_name="Nea"
    )
    assert prof.vertical == "inmobiliario"


async def test_run_turn_realty_usa_el_esquema_y_chasis_inmobiliario(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_rei_sel")
    payload = realty_payload(conversation_id="cv_rei_sel")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert ctx.llm.calls[-1]["tools"] == REALTY_TOOL_SCHEMAS
    system = ctx.llm.calls[0]["messages"][0]["content"]
    assert "inmobiliaria" in system
    assert "guardar_requerimiento" in system
    assert "route_out" not in system  # tool B2B: no existe en este chasis


async def test_dispatch_http_real_con_perfil_inmobiliario(
    ctx: AppContext, client, respx_mock
):
    """Camino REAL de punta a punta (no `run_turn` directo): firma HMAC,
    `POST /dispatch` de verdad contra la app ASGI, ruteo por `version`, y
    solo entonces `run_turn` — igual que `test_v2_nunca_toca_el_store` en
    tests/test_stateless.py, pero con `profile.vertical` encendido."""
    mock_crm_basics(respx_mock, conv_id="cv_http_rei")
    respx_mock.put(f"{CRM_URL}/api/bot/realty/requirement").mock(
        return_value=httpx.Response(
            200,
            json={
                "requirement": {"operation": "venta"},
                "candidates": [
                    {
                        "id": "p1",
                        "title": "Depto Sopocachi",
                        "price": 90000,
                        "currency": "USD",
                        "zone": "Sopocachi",
                        "reasons": ["dentro de presupuesto"],
                    }
                ],
            },
        )
    )
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(id="t1", name="guardar_requerimiento", arguments={"operation": "venta"})
            ],
        ),
        LlmReply(content="Encontré una opción: Depto Sopocachi, 90000 USD."),
    ]
    body = json.dumps(realty_raw(conversation_id="cv_http_rei")).encode("utf-8")
    resp = await client.post("/dispatch", content=body, headers={"x-signature": sign(body)})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["action"] == "replied"


async def test_run_turn_default_sigue_exactamente_igual(respx_mock):
    """Regresión explícita: perfil SIN vertical usa TOOL_SCHEMAS (el MISMO
    objeto, no una copia) y el chasis de siempre."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_default_sel")
    payload = realty_payload(conversation_id="cv_default_sel", vertical=None)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert ctx.llm.calls[-1]["tools"] is TOOL_SCHEMAS  # identidad, no solo igualdad
    system = ctx.llm.calls[0]["messages"][0]["content"]
    assert "route_out" in system
    assert "guardar_requerimiento" not in system


# ============================================================== esquemas de tools ===


def test_realty_tool_schemas_sin_b2b():
    names = {t["function"]["name"] for t in REALTY_TOOL_SCHEMAS}
    assert names == {
        "guardar_requerimiento",
        "ver_propiedad",
        "enviar_ficha",
        "propose_slots",
        "book_session",
        "reschedule_session",
        "handoff",
    }
    assert "update_ficha" not in names
    assert "route_out" not in names


def test_realty_book_y_reschedule_tienen_property_id_opcional():
    for name in ("book_session", "reschedule_session"):
        schema = next(t for t in REALTY_TOOL_SCHEMAS if t["function"]["name"] == name)
        params = schema["function"]["parameters"]
        assert "property_id" in params["properties"]
        assert "property_id" not in params.get("required", [])
        assert "start_utc" in params["required"]  # el resto del esquema no cambió


def test_default_tool_schemas_no_cambiaron():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {
        "update_ficha",
        "propose_slots",
        "book_session",
        "reschedule_session",
        "route_out",
        "handoff",
    }
    book = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "book_session")
    assert "property_id" not in book["function"]["parameters"]["properties"]


# ==================================================== guardar_requerimiento ===


async def test_guardar_requerimiento_roundtrip(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_req_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="guardar_requerimiento",
                    arguments={
                        "operation": "alquiler",
                        "zones": ["Equipetrol"],
                        "budgetMax": 3500,
                        "currency": "BOB",
                    },
                )
            ],
        ),
        LlmReply(content="Perfecto, ¿cuántas recámaras buscas?"),
    ]
    req_route = respx_mock.put(f"{CRM_URL}/api/bot/realty/requirement").mock(
        return_value=httpx.Response(
            200,
            json={
                "requirement": {
                    "operation": "alquiler",
                    "zones": ["Equipetrol"],
                    "budgetMax": 3500,
                    "currency": "BOB",
                    "missing": ["minBedrooms"],
                },
                "candidates": [
                    {
                        "id": "p1",
                        "title": "Depto Equipetrol",
                        "price": 3200,
                        "currency": "BOB",
                        "zone": "Equipetrol",
                        "reasons": ["dentro de presupuesto"],
                    }
                ],
            },
        )
    )
    payload = realty_payload(conversation_id="cv_req_1")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert req_route.call_count == 1
    body = json.loads(req_route.calls[0].request.content)
    assert body == {
        "conversationId": "cv_req_1",
        "requirement": {
            "operation": "alquiler",
            "zones": ["Equipetrol"],
            "budgetMax": 3500,
            "currency": "BOB",
        },
    }
    # las candidatas frescas le llegaron al modelo en el MISMO turno
    tool_msg = ctx.llm.calls[-1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "Depto Equipetrol" in tool_msg["content"]


async def test_guardar_requerimiento_sin_campos_no_llama_al_crm(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_req_vacio")
    req_route = respx_mock.put(f"{CRM_URL}/api/bot/realty/requirement").mock(
        return_value=httpx.Response(200, json={"requirement": {}, "candidates": []})
    )
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="guardar_requerimiento", arguments={})],
        ),
        LlmReply(content="Cuéntame más."),
    ]
    payload = realty_payload(conversation_id="cv_req_vacio")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert req_route.call_count == 0


# =============================================================== ver_propiedad ===


async def test_ver_propiedad_trae_detalle(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ver_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="ver_propiedad", arguments={"property_id": "p1"})],
        ),
        LlmReply(content="Tiene 120 m2 construidos y acepta anticrético."),
    ]
    prop_route = respx_mock.get(f"{CRM_URL}/api/bot/realty/property").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "p1",
                "title": "Casa Urubó",
                "builtArea": 120,
                "acceptedPayments": ["anticretico"],
            },
        )
    )
    payload = realty_payload(conversation_id="cv_ver_1")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert prop_route.call_count == 1
    assert prop_route.calls[0].request.url.params["id"] == "p1"


async def test_ver_propiedad_404_no_revienta_el_turno(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ver_404")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="ver_propiedad", arguments={"property_id": "ghost"})],
        ),
        LlmReply(content="Uy, esa ya no está disponible; te muestro otra."),
    ]
    respx_mock.get(f"{CRM_URL}/api/bot/realty/property").mock(
        return_value=httpx.Response(404, json={"error": {"code": "property_not_found"}})
    )
    payload = realty_payload(conversation_id="cv_ver_404")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"


async def test_ver_propiedad_sin_id_no_llama_al_crm(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ver_sinid")
    prop_route = respx_mock.get(f"{CRM_URL}/api/bot/realty/property").mock(
        return_value=httpx.Response(200, json={"id": "x"})
    )
    ctx.llm.replies = [
        LlmReply(content=None, tool_calls=[ToolCall(id="t1", name="ver_propiedad", arguments={})]),
        LlmReply(content="¿Cuál propiedad te interesa?"),
    ]
    payload = realty_payload(conversation_id="cv_ver_sinid")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert prop_route.call_count == 0


# ================================================================ enviar_ficha ===


async def test_enviar_ficha_manda_dispatchid_y_seq(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ficha_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="enviar_ficha", arguments={"property_id": "p1"})],
        ),
        LlmReply(content="Ahí te la mando."),
    ]
    ficha_route = respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(
        return_value=httpx.Response(
            200, json={"sent": True, "messageId": "wamid.f1", "photoSent": True}
        )
    )
    payload = realty_payload(conversation_id="cv_ficha_1", dispatch_id="dsp_f1")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert ficha_route.call_count == 1
    body = json.loads(ficha_route.calls[0].request.content)
    assert body == {
        "conversationId": "cv_ficha_1",
        "propertyId": "p1",
        "dispatchId": "dsp_f1",
        "seq": 1,
    }


async def test_enviar_ficha_idempotente_en_un_reintento_del_despacho(respx_mock):
    """Simula un RETRY del despacho completo (mismo dispatchId, turno nuevo):
    Nea debe mandar el MISMO (dispatchId, seq) las dos veces — la dedup real
    la hace el CRM (contrato: idempotente por dispatchId+seq), pero Nea tiene
    que ser determinista para que funcione. La segunda vez el CRM contesta
    `duplicate: true`; eso no debe tratarse como error."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ficha_retry")
    seqs_recibidos: list[tuple[str, int]] = []

    def _responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seqs_recibidos.append((body["dispatchId"], body["seq"]))
        duplicate = len(seqs_recibidos) > 1
        return httpx.Response(
            200,
            json={
                "sent": True,
                "messageId": "wamid.f1",
                "photoSent": True,
                "duplicate": duplicate,
            },
        )

    respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(side_effect=_responder)

    for _ in range(2):
        ctx.llm.replies = [
            LlmReply(
                content=None,
                tool_calls=[ToolCall(id="t1", name="enviar_ficha", arguments={"property_id": "p1"})],
            ),
            LlmReply(content="Ahí te la mando."),
        ]
        payload = realty_payload(conversation_id="cv_ficha_retry", dispatch_id="dsp_retry_1")
        result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
        assert result.action == "replied"

    assert seqs_recibidos == [("dsp_retry_1", 1), ("dsp_retry_1", 1)]


async def test_enviar_ficha_marca_el_commit_solo_tras_2xx(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ficha_commit")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="enviar_ficha", arguments={"property_id": "p1"})],
        ),
        LlmReply(content="Ahí te la mando."),
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(
        return_value=httpx.Response(200, json={"sent": True, "messageId": "m1", "photoSent": False})
    )
    commit = TurnCommit()
    payload = realty_payload(conversation_id="cv_ficha_commit")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei", commit=commit)
    assert result.action == "replied"
    assert commit.done is True


async def test_enviar_ficha_sin_2xx_nunca_marca_el_commit(respx_mock):
    """Unidad directa del handler (igual que test_v2_commit_nunca_se_marca_si_
    nunca_hay_2xx hace con `_send`): agotar los 3 reintentos sin un solo 2xx
    no debe marcar el commit, y el resultado es un fallo GRACIOSO de la tool
    (nunca una excepción: `ToolRuntime.execute` atraparía cualquier CrmError
    de una tool de todas formas — a diferencia de `_send`, que corre FUERA de
    ese try/except y por eso sí puede dejar escapar la excepción)."""
    ctx = make_ctx()
    ficha_route = respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(
        return_value=httpx.Response(502)
    )
    commit = TurnCommit()
    handlers = inmobiliario.build_realty_tools(
        crm=ctx.crm, conversation_id="cv_ficha_fail", dispatch_id="dsp_1", commit=commit
    )
    result = await handlers["enviar_ficha"]({"property_id": "p1"})
    assert result["ok"] is False
    assert result["error"] == "crm_error"
    assert commit.done is False
    assert ficha_route.call_count == 3  # agotó los reintentos, nunca un 2xx
    await ctx.crm.aclose()


async def test_enviar_ficha_falla_gracil_dentro_de_execute_nunca_revienta_el_turno(respx_mock):
    """El mismo agotamiento, pero visto DESDE run_turn: el modelo se entera
    del fallo por el resultado de la tool y sigue la conversación con
    naturalidad — el turno completa normal, nunca un 500."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ficha_fail_turno")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="enviar_ficha", arguments={"property_id": "p1"})],
        ),
        LlmReply(content="Uy, no pude mandarte la ficha; dame un momento."),
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(return_value=httpx.Response(502))
    payload = realty_payload(conversation_id="cv_ficha_fail_turno")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    tool_msg = ctx.llm.calls[-1]["messages"][-1]
    assert json.loads(tool_msg["content"])["error"] == "crm_error"


async def test_enviar_ficha_402_no_revienta(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_ficha_402")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="enviar_ficha", arguments={"property_id": "p1"})],
        ),
        LlmReply(content="Dame un segundo."),
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/realty/ficha").mock(
        return_value=httpx.Response(402, json={"code": "payment_required"})
    )
    payload = realty_payload(conversation_id="cv_ficha_402")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"


# ============================================================ book/reschedule ===


async def test_book_session_con_property_id(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_book_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="book_session",
                    arguments={"start_utc": SLOT_ISO, "property_id": "p1"},
                )
            ],
        ),
        LlmReply(content="Quedó tu visita."),
    ]
    bookings_route = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "lunes 10am"})
    )
    payload = realty_payload(
        conversation_id="cv_book_1", offers=[{"startUtc": SLOT_ISO, "label": "lunes 10am"}]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    body = json.loads(bookings_route.calls[0].request.content)
    assert body == {"conversationId": "cv_book_1", "startUtc": SLOT_ISO, "propertyId": "p1"}


async def test_book_session_realty_no_escribe_ficha_b2b(respx_mock):
    """A diferencia del camino default, reservar en este vertical NO debe
    tocar PUT /api/bot/ficha — calificado/resultado son vocabulario B2B sin
    sentido aquí (update_ficha ni siquiera existe en este vertical)."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_book_noficha")
    ficha_route = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": False})
    )
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="book_session", arguments={"start_utc": SLOT_ISO})],
        ),
        LlmReply(content="Quedó tu visita."),
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "lunes 10am"})
    )
    payload = realty_payload(
        conversation_id="cv_book_noficha", offers=[{"startUtc": SLOT_ISO, "label": "lunes 10am"}]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert ficha_route.call_count == 0


async def test_reschedule_session_con_property_id(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_resch_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="reschedule_session",
                    arguments={"start_utc": SLOT_ISO, "property_id": "p1"},
                )
            ],
        ),
        LlmReply(content="Movida tu visita."),
    ]
    patch_route = respx_mock.patch(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(200, json={"label": "martes 10am"})
    )
    payload = realty_payload(
        conversation_id="cv_resch_1", offers=[{"startUtc": SLOT_ISO, "label": "martes 10am"}]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    body = json.loads(patch_route.calls[0].request.content)
    assert body == {"conversationId": "cv_resch_1", "startUtc": SLOT_ISO, "propertyId": "p1"}


async def test_book_session_422_property_not_found_no_revienta(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_book_422")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="book_session",
                    arguments={"start_utc": SLOT_ISO, "property_id": "ghost"},
                )
            ],
        ),
        LlmReply(content="Esa propiedad ya no está; te muestro otra."),
    ]
    respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(422, json={"error": {"code": "property_not_found"}})
    )
    payload = realty_payload(
        conversation_id="cv_book_422", offers=[{"startUtc": SLOT_ISO, "label": "lunes 10am"}]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"


async def test_book_session_sin_property_id_sigue_funcionando(respx_mock):
    """El property_id es OPCIONAL: una visita sin propiedad puntual (p.ej.
    una cita general con el agente) reserva igual, sin la llave de más."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_book_sin_prop")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="book_session", arguments={"start_utc": SLOT_ISO})],
        ),
        LlmReply(content="Quedó agendado."),
    ]
    bookings_route = respx_mock.post(f"{CRM_URL}/api/bot/bookings").mock(
        return_value=httpx.Response(201, json={"bookingId": "bk_1", "label": "lunes 10am"})
    )
    payload = realty_payload(
        conversation_id="cv_book_sin_prop", offers=[{"startUtc": SLOT_ISO, "label": "lunes 10am"}]
    )
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    body = json.loads(bookings_route.calls[0].request.content)
    assert body == {"conversationId": "cv_book_sin_prop", "startUtc": SLOT_ISO}
    assert "propertyId" not in body


# ============================================================ vertical_disabled ===


async def test_vertical_disabled_en_guardar_requerimiento_no_revienta(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_vd_1")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[
                ToolCall(id="t1", name="guardar_requerimiento", arguments={"operation": "venta"})
            ],
        ),
        LlmReply(content="Dame un segundo, te conecto con alguien del equipo."),
    ]
    respx_mock.put(f"{CRM_URL}/api/bot/realty/requirement").mock(
        return_value=httpx.Response(404, json={"error": {"code": "vertical_disabled"}})
    )
    payload = realty_payload(conversation_id="cv_vd_1")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"  # nunca crashea; el turno sigue


async def test_vertical_disabled_el_modelo_puede_optar_por_handoff(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_vd_2")
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="ver_propiedad", arguments={"property_id": "p1"})],
        ),
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t2", name="handoff", arguments={"reason": "quiere hablar con alguien"})],
        ),
        LlmReply(content="Ya te conecto con el equipo."),
    ]
    respx_mock.get(f"{CRM_URL}/api/bot/realty/property").mock(
        return_value=httpx.Response(404, json={"error": {"code": "vertical_disabled"}})
    )
    payload = realty_payload(conversation_id="cv_vd_2")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert result.handoff_applied is True


# ==================================================================== prompt ===


def test_render_realty_block_solo_candidatas_del_contexto():
    context = {
        "realty": {
            "requirement": {"operation": "venta", "missing": ["budgetMax"]},
            "candidates": [
                {
                    "id": "p1",
                    "title": "Casa Urubó",
                    "price": 250000,
                    "currency": "USD",
                    "zone": "Urubó",
                    "reasons": ["3 dormitorios"],
                }
            ],
            "focusPropertyId": "p1",
            "viewings": [],
        }
    }
    block = render_realty_block(context)
    assert "Casa Urubó" in block
    assert "id=p1" in block
    assert "Depto Inventado" not in block  # jamás algo que no vino en el contexto


def test_render_realty_block_sin_context_realty_no_inventa_nada():
    block = render_realty_block(None)
    assert "PROPIEDADES" in block
    assert "ninguna todavía" in block or "todavía no se sabe nada" in block


async def test_prompt_inmobiliario_solo_menciona_candidatas_del_contexto(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_prompt_1")
    payload = realty_payload(
        conversation_id="cv_prompt_1",
        realty={
            "requirement": {"operation": "alquiler"},
            "candidates": [
                {
                    "id": "p9",
                    "title": "Depto Real de verdad",
                    "price": 4000,
                    "currency": "BOB",
                    "zone": "Norte",
                    "reasons": ["cerca del trabajo"],
                }
            ],
            "focusPropertyId": None,
            "viewings": [],
        },
    )
    await stateless.run_turn(ctx, payload, organization_id="org_rei")
    system = ctx.llm.calls[0]["messages"][0]["content"]
    assert "Depto Real de verdad" in system
    assert "id=p9" in system


# =============================================================== DISABLED_TOOLS ===


async def test_update_ficha_deshabilitada_en_inmobiliario(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_disabled_1")
    ficha_route = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={})
    )
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="update_ficha", arguments={"rubro": "x"})],
        ),
        LlmReply(content="Sigamos con lo tuyo."),
    ]
    payload = realty_payload(conversation_id="cv_disabled_1")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert ficha_route.call_count == 0  # jamás llegó al CRM


async def test_route_out_deshabilitada_en_inmobiliario(respx_mock):
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_disabled_2")
    ficha_route = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={})
    )
    ctx.llm.replies = [
        LlmReply(content=None, tool_calls=[ToolCall(id="t1", name="route_out", arguments={})]),
        LlmReply(content="Gracias por tu tiempo."),
    ]
    payload = realty_payload(conversation_id="cv_disabled_2")
    result = await stateless.run_turn(ctx, payload, organization_id="org_rei")
    assert result.action == "replied"
    assert ficha_route.call_count == 0


# ============================================================ camino default ===


async def test_default_path_sigue_llamando_update_ficha_normalmente(respx_mock):
    """Contraprueba: fuera de este vertical, update_ficha sigue funcionando
    exactamente igual (disabled_tools vacío por default)."""
    ctx = make_ctx()
    mock_crm_basics(respx_mock, conv_id="cv_default_ficha")
    ficha_route = respx_mock.put(f"{CRM_URL}/api/bot/ficha").mock(
        return_value=httpx.Response(200, json={"ficha": {}, "stageMoved": False})
    )
    ctx.llm.replies = [
        LlmReply(
            content=None,
            tool_calls=[ToolCall(id="t1", name="update_ficha", arguments={"rubro": "clínica"})],
        ),
        LlmReply(content="Perfecto, ¿algo más?"),
    ]
    payload = realty_payload(conversation_id="cv_default_ficha", vertical=None)
    result = await stateless.run_turn(ctx, payload, organization_id="org_a")
    assert result.action == "replied"
    assert ficha_route.call_count == 1
