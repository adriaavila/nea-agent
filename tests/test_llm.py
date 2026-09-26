"""OpenAiLlm: fallback de credenciales (401/402/429), moderación (403) y las
dos rutas de transcripción (whisper real vs. por chat) — PR 2B, voz e
identidad de negocio en el despacho v2.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm import LlmAuthFailed, LlmExhausted, LlmNoCredits, OpenAiLlm

OPENAI_CHAT = "https://api.openai.com/v1/chat/completions"
OPENAI_TRANSCRIBE = "https://api.openai.com/v1/audio/transcriptions"
OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"

MSGS = [{"role": "user", "content": "hola"}]


def _err(code: str | None = None, status: int = 400, message: str = "boom") -> httpx.Response:
    body = {"error": {"message": message, "type": "error", "param": None, "code": code}}
    return httpx.Response(status, json=body)


# --------------------------------------------------------- fallo sin reintento ---


async def test_401_marca_auth_failed_sin_reintentar(respx_mock):
    route = respx_mock.post(OPENAI_CHAT).mock(return_value=_err(status=401))
    llm = OpenAiLlm("sk-mala", "gpt-4o-mini", max_retries=0)
    with pytest.raises(LlmAuthFailed):
        await llm.complete(MSGS)
    assert route.call_count == 1  # sin reintentos: la MISMA clave no cambia
    await llm.aclose()


async def test_402_marca_no_credits_sin_reintentar(respx_mock):
    route = respx_mock.post(OPENROUTER_CHAT).mock(return_value=_err(status=402))
    llm = OpenAiLlm(
        "sk-sin-credito", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    with pytest.raises(LlmNoCredits):
        await llm.complete(MSGS)
    assert route.call_count == 1
    await llm.aclose()


async def test_429_insufficient_quota_marca_no_credits(respx_mock):
    route = respx_mock.post(OPENAI_CHAT).mock(
        return_value=_err(code="insufficient_quota", status=429)
    )
    llm = OpenAiLlm("sk-x", "gpt-4o-mini", max_retries=0)
    with pytest.raises(LlmNoCredits):
        await llm.complete(MSGS)
    assert route.call_count == 1
    await llm.aclose()


async def test_429_sin_insufficient_quota_reintenta_como_error_normal(respx_mock):
    route = respx_mock.post(OPENAI_CHAT).mock(return_value=_err(code="rate_limit_exceeded", status=429))
    llm = OpenAiLlm("sk-x", "gpt-4o-mini", max_retries=0)
    with pytest.raises(LlmExhausted):
        await llm.complete(MSGS)
    assert route.call_count == llm.RETRIES + 1  # SÍ reintentó — no es un fallo de clave
    await llm.aclose()


# ------------------------------------------------------- 403 y 5xx normales ---


async def test_403_moderacion_no_cae_reintenta_como_error_normal(respx_mock):
    """403 de OpenRouter = moderación de contenido, NO credenciales: sin
    marcar, sin fallback — un error cualquiera que reintenta y, agotado,
    termina en LlmExhausted (silencio + handoff error, igual que cualquier
    otro fallo del LLM)."""
    route = respx_mock.post(OPENROUTER_CHAT).mock(return_value=_err(status=403))
    llm = OpenAiLlm(
        "sk-x", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    with pytest.raises(LlmExhausted):
        await llm.complete(MSGS)
    assert route.call_count == llm.RETRIES + 1
    await llm.aclose()


async def test_403_limite_de_gasto_de_la_clave_marca_no_credits_sin_reintentar(respx_mock):
    """Revisión ronda 3: OpenRouter no tiene un código de error aparte para
    "esta clave puntual se quedó sin presupuesto" (a diferencia del
    `insufficient_quota` de OpenAI) — lo dice en el MENSAJE del 403
    ("Key limit exceeded"). Debe caer a la plataforma igual que un 402."""
    route = respx_mock.post(OPENROUTER_CHAT).mock(
        return_value=_err(status=403, message="Key limit exceeded")
    )
    llm = OpenAiLlm(
        "sk-x", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    with pytest.raises(LlmNoCredits):
        await llm.complete(MSGS)
    assert route.call_count == 1  # sin reintentos: la MISMA clave no cambia
    await llm.aclose()


async def test_403_menciona_credito_tambien_marca_no_credits(respx_mock):
    route = respx_mock.post(OPENROUTER_CHAT).mock(
        return_value=_err(status=403, message="Insufficient credit balance for this key")
    )
    llm = OpenAiLlm(
        "sk-x", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    with pytest.raises(LlmNoCredits):
        await llm.complete(MSGS)
    assert route.call_count == 1
    await llm.aclose()


async def test_403_moderacion_de_verdad_no_menciona_limite_ni_credito_sigue_normal(respx_mock):
    """Contraprueba: un 403 cuyo mensaje de verdad es moderación (nada de
    "limit"/"credit") sigue el camino normal — reintenta, nunca marca ni cae
    a la plataforma."""
    route = respx_mock.post(OPENROUTER_CHAT).mock(
        return_value=_err(status=403, message="Content flagged by moderation")
    )
    llm = OpenAiLlm(
        "sk-x", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    with pytest.raises(LlmExhausted):
        await llm.complete(MSGS)
    assert route.call_count == llm.RETRIES + 1
    await llm.aclose()


async def test_5xx_no_cae_reintenta_como_error_normal(respx_mock):
    route = respx_mock.post(OPENAI_CHAT).mock(return_value=httpx.Response(500))
    llm = OpenAiLlm("sk-x", "gpt-4o-mini", max_retries=0)
    with pytest.raises(LlmExhausted):
        await llm.complete(MSGS)
    assert route.call_count == llm.RETRIES + 1
    await llm.aclose()


# --------------------------------------------------- alcance de max_retries ---


async def test_max_retries_por_default_deja_el_default_del_sdk(respx_mock):
    """El cliente COMPARTIDO de plataforma (app/main.py) NO pasa
    `max_retries` — v1/legacy no debe cambiar de comportamiento con este PR.
    El guardia de 75 s (asyncio.wait_for en app/dispatch.py) es lo que
    protege a v2 si esto tarda de más, no este parámetro."""
    llm = OpenAiLlm("sk-x", "gpt-4o-mini")  # sin max_retries: el de siempre
    assert llm._client.max_retries == 2  # default del SDK, sin tocar
    await llm.aclose()


async def test_max_retries_cero_es_explicito_para_el_cliente_por_turno(respx_mock):
    """app/stateless._build_org_llm SÍ lo pasa en 0 — un cliente de usar y
    cerrar dentro del presupuesto de 75 s del despacho."""
    llm = OpenAiLlm("sk-x", "z-ai/glm-5.3-flash", base_url="https://openrouter.ai/api/v1", max_retries=0)
    assert llm._client.max_retries == 0
    await llm.aclose()


async def test_max_retries_cero_tambien_alcanza_al_cliente_de_whisper_aparte(respx_mock):
    llm = OpenAiLlm(
        "sk-openrouter",
        "z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key="sk-openai-real",
        max_retries=0,
    )
    assert llm._whisper_client.max_retries == 0
    await llm.aclose()


# ------------------------------------------------------------- aclose() ---


async def test_aclose_cierra_el_cliente_unico_cuando_no_hay_whisper_aparte(respx_mock):
    llm = OpenAiLlm("sk-x", "gpt-4o-mini")  # base_url=None: un solo cliente
    assert llm._client._client.is_closed is False
    await llm.aclose()
    assert llm._client._client.is_closed is True


async def test_aclose_cierra_tambien_el_cliente_de_whisper_aparte(respx_mock):
    llm = OpenAiLlm(
        "sk-openrouter",
        "z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key="sk-openai-real",
    )
    assert llm._client is not llm._whisper_client
    await llm.aclose()
    assert llm._client._client.is_closed is True
    assert llm._whisper_client._client.is_closed is True


# --------------------------------------------------- transcripción: combos ---


async def test_combo_nea_agent_openrouter_mas_openai_transcribe_usa_whisper_real(respx_mock):
    """nea-agent: OPENROUTER_API_TOKEN + OPENROUTER_MODEL (conversa) y
    OPENAI_API_KEY + OPENAI_TRANSCRIBE_MODEL=whisper-1 (transcribe). El chat
    va por OpenRouter; la transcripción, por un cliente de OpenAI real
    aparte — nunca por el endpoint de OpenRouter, que no existe."""
    whisper_route = respx_mock.post(OPENAI_TRANSCRIBE).mock(
        return_value=httpx.Response(200, json={"text": "hola desde whisper real"})
    )
    llm = OpenAiLlm(
        "sk-openrouter-token",
        "z-ai/glm-5.3-flash",
        transcribe_model="whisper-1",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key="sk-openai-real",
    )
    text = await llm.transcribe(b"audio-bytes", "audio/ogg")
    assert text == "hola desde whisper real"
    assert whisper_route.call_count == 1
    assert whisper_route.calls[0].request.headers["authorization"] == "Bearer sk-openai-real"
    await llm.aclose()


async def test_combo_nea_mistica_solo_openrouter_transcribe_por_chat(respx_mock):
    """nea-mistica: solo OPENROUTER_API_TOKEN + OPENROUTER_MODEL, sin
    OPENAI_API_KEY. No hay endpoint de transcripción — se manda el audio
    dentro del chat (input_audio) al modelo default que sí oye."""
    chat_route = respx_mock.post(OPENROUTER_CHAT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "hola desde el chat"}}
                ]
            },
        )
    )
    llm = OpenAiLlm(
        "sk-openrouter-token",
        "z-ai/glm-5.3-flash",
        transcribe_model="whisper-1",  # el default de siempre; NO tiene "/"
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=None,
    )
    text = await llm.transcribe(b"audio-bytes", "audio/ogg")
    assert text == "hola desde el chat"
    assert chat_route.call_count == 1
    sent = json.loads(chat_route.calls[0].request.content)
    assert sent["model"] == "google/gemini-2.5-flash-lite"  # default probado, GLM no oye
    parts = sent["messages"][0]["content"]
    assert any(p["type"] == "input_audio" for p in parts)
    await llm.aclose()


async def test_transcribe_model_explicito_proveedor_modelo_se_respeta(respx_mock):
    chat_route = respx_mock.post(OPENROUTER_CHAT).mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
    )
    llm = OpenAiLlm(
        "sk-openrouter-token",
        "z-ai/glm-5.3-flash",
        transcribe_model="google/gemini-2.5-flash",  # explícito, "proveedor/modelo"
        base_url="https://openrouter.ai/api/v1",
    )
    await llm.transcribe(b"audio-bytes", "audio/ogg")
    sent = json.loads(chat_route.calls[0].request.content)
    assert sent["model"] == "google/gemini-2.5-flash"
    await llm.aclose()


async def test_solo_openai_sin_openrouter_transcribe_con_el_mismo_cliente(respx_mock):
    """Deploy 100% OpenAI (sin OPENROUTER_API_TOKEN): base_url=None desde el
    arranque — el MISMO cliente ya es OpenAI real, sin necesidad de uno
    aparte (comportamiento de siempre, sin cambios)."""
    whisper_route = respx_mock.post(OPENAI_TRANSCRIBE).mock(
        return_value=httpx.Response(200, json={"text": "listo"})
    )
    llm = OpenAiLlm("sk-openai", "gpt-4o-mini", transcribe_model="whisper-1")
    assert llm._whisper_client is llm._client
    text = await llm.transcribe(b"audio-bytes", "audio/ogg")
    assert text == "listo"
    assert whisper_route.call_count == 1
    await llm.aclose()
