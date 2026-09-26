"""Capa LLM: OpenAI chat.completions con tool-calling y extracción tolerante.

Gotchas del brief que se honran aquí:
- `content` vacío con tool_calls es NORMAL (turno solo-herramientas).
- Respuesta vacía de verdad (sin content ni tool_calls) o excepción → reintento
  con backoff (2 reintentos). Agotado → `LlmExhausted` y el turno degrada en
  silencio + handoff error (Constitución IV).
- Los `arguments` de las tools pueden venir malformados: JSON inválido → {}.
- 401/402 (o 429 `insufficient_quota`) NO reintentan aquí: son responsabilidad
  del LLAMADOR (app/stateless.py, PR 2B) que decide caer a la clave de la
  plataforma y reportarlo — reintentar contra la MISMA clave rota solo quema
  tiempo. El 403 de OpenRouter es moderación de contenido, no credenciales:
  se trata como cualquier otro error (reintenta, no cae a otra clave).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import APIStatusError, AsyncOpenAI

logger = logging.getLogger("nea.llm")


class LlmExhausted(Exception):
    """El LLM falló todos los reintentos — el turno debe degradar en silencio."""


class LlmAuthFailed(LlmExhausted):
    """401: la clave del proveedor no sirve. Sin reintento — el llamador decide
    el fallback a la clave de la plataforma.

    Subclase de `LlmExhausted` a propósito: app/turn.py (v1/legacy) solo
    atrapa `LlmExhausted` alrededor de `_tool_loop` — sin esta herencia, un
    401/402/429 contra la clave de la plataforma (que hoy SÍ puede pasar,
    nadie está a salvo de una clave que vence) escaparía sin degradar
    (silencio + handoff `error`, fase `cerrada`), rompiendo el camino legacy.
    app/stateless.py (v2) sigue distinguiéndola con su propio
    `except (LlmAuthFailed, LlmNoCredits)`, evaluado ANTES de llegar a
    cualquier `except LlmExhausted` más externo — el orden de los `except`
    de Python ya lo garantiza, esta herencia no lo cambia."""


class LlmNoCredits(LlmExhausted):
    """402, o 429 con código `insufficient_quota`: sin crédito. Sin reintento —
    el llamador decide el fallback a la clave de la plataforma. Ver el
    docstring de `LlmAuthFailed`: misma razón para heredar de `LlmExhausted`."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class Llm(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LlmReply: ...

    async def transcribe(
        self, data: bytes, mime: str, filename: str = "audio.ogg"
    ) -> str: ...

    async def aclose(self) -> None: ...


# Los proveedores compatibles (OpenRouter) piden el formato del audio aparte
# del binario. WhatsApp manda notas de voz en OGG/Opus; el resto se deduce
# del mime y, si no se reconoce, se manda como ogg en vez de fallar antes de
# intentarlo.
_FORMATOS_AUDIO = {
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/aac": "aac",
}

_TRANSCRIBE_CHAT_DEFAULT = "google/gemini-2.5-flash-lite"


def _formato_de_audio(mime: str) -> str:
    return _FORMATOS_AUDIO.get((mime or "").split(";")[0].strip().lower(), "ogg")


def _sin_reintento(exc: APIStatusError) -> type[Exception] | None:
    """401/402, o 429 `insufficient_quota`: la clave no sirve — reintentar
    contra ELLA MISMA no cambia nada. `None` = error normal (reintenta como
    cualquier otro, incluido el 403 de moderación de OpenRouter, que NO es
    un problema de credenciales)."""
    if exc.status_code == 401:
        return LlmAuthFailed
    if exc.status_code == 402:
        return LlmNoCredits
    if exc.status_code == 429 and exc.code == "insufficient_quota":
        return LlmNoCredits
    return None


class OpenAiLlm:
    RETRIES = 2  # además del intento inicial

    def __init__(
        self,
        api_key: str,
        model: str,
        transcribe_model: str = "whisper-1",
        base_url: str | None = None,
        *,
        openai_api_key: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        # base_url ≠ None → proveedor OpenAI-compatible (p. ej. OpenRouter,
        # para el bench de modelos del 002 y para la clave por negocio del
        # despacho v2).
        #
        # `max_retries`: el SDK reintenta 429/5xx por su cuenta (2 veces, por
        # default) ANTES de que este código vea la excepción — sin apagarlo,
        # un fallo "agotado" hace 3 (SDK) x 3 (RETRIES de aquí) = 9 llamadas
        # HTTP reales, no las ~2-3 que dicen los comentarios. `None` (default)
        # deja el default del SDK intacto — es lo que usa el cliente
        # COMPARTIDO de plataforma (app/main.py): v1/legacy no cambia de
        # comportamiento con este PR. `0` es lo que pasa app/stateless.py al
        # construir el cliente POR TURNO de un negocio — ahí sí importa: el
        # despacho entero tiene 75 s (app/dispatch.DISPATCH_TIMEOUT_SECONDS) y
        # una clave rota reintentando 9 veces se come ese presupuesto por las
        # puras. El guardia de 75 s protege a los DOS casos igual si algo se
        # escapa; esto es solo para no desperdiciarlo en el camino más común.
        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._transcribe_model = transcribe_model
        # Transcripción: SIEMPRE con credenciales de la plataforma, nunca las
        # de un negocio (ver app/stateless.py). Tres casos:
        # - base_url es None: este cliente YA es OpenAI real — se reutiliza.
        # - base_url tiene valor pero hay una `openai_api_key` real: cliente
        #   APARTE, sin base_url, para el endpoint de whisper de verdad.
        # - ninguna de las anteriores (solo OpenRouter, sin clave de OpenAI):
        #   no hay endpoint de transcripción — se transcribe POR CHAT (audio
        #   en base64 como `input_audio`) con un modelo que oiga.
        if base_url is None:
            self._whisper_client: AsyncOpenAI | None = self._client
        elif openai_api_key:
            whisper_kwargs: dict[str, Any] = {"api_key": openai_api_key}
            if max_retries is not None:
                whisper_kwargs["max_retries"] = max_retries
            self._whisper_client = AsyncOpenAI(**whisper_kwargs)
        else:
            self._whisper_client = None
        # Modelo que OYE cuando se transcribe por chat: un `proveedor/modelo`
        # explícito (p.ej. viene de OPENAI_TRANSCRIBE_MODEL), o el default
        # probado — GLM no oye, este sí. Un id pelado de OpenAI (`whisper-1`,
        # el default de siempre) NO cuenta como "proveedor/modelo".
        self._chat_transcribe_model = (
            transcribe_model
            if transcribe_model and "/" in transcribe_model
            else _TRANSCRIBE_CHAT_DEFAULT
        )
        # Contadores de uso (para el bench de costos del 002): tokens reales
        # reportados por el proveedor, acumulados por instancia.
        self.usage = {"prompt": 0, "cached": 0, "completion": 0, "llamadas": 0}

    async def transcribe(
        self, data: bytes, mime: str, filename: str = "audio.ogg"
    ) -> str:
        """Audio → texto. Vacío o fallo → LlmExhausted.

        Dos caminos, porque los proveedores no ofrecen lo mismo: OpenAI real
        (endpoint de whisper) o un compatible sin ese endpoint (por chat,
        `input_audio`) — ver `__init__`.
        """
        if self._whisper_client is not None:
            return await self._transcribe_whisper(data, mime, filename)
        return await self._transcribe_por_chat(data, mime)

    async def _transcribe_whisper(
        self, data: bytes, mime: str, filename: str
    ) -> str:
        assert self._whisper_client is not None
        last_error: Exception | None = None
        content_type = (mime or "audio/ogg").split(";")[0].strip()
        for attempt in range(2):
            try:
                resp = await self._whisper_client.audio.transcriptions.create(
                    model=self._transcribe_model,
                    file=(filename, data, content_type),
                    language="es",
                )
                text = (getattr(resp, "text", None) or "").strip()
                if text:
                    return text
                last_error = ValueError("transcripción vacía")
                logger.warning("transcribe: texto vacío, intento %d", attempt + 1)
            except Exception as exc:
                last_error = exc
                logger.warning("transcribe: fallo en intento %d: %s", attempt + 1, exc)
            if attempt == 0:
                await asyncio.sleep(1.0)
        raise LlmExhausted(str(last_error))

    async def _transcribe_por_chat(self, data: bytes, mime: str) -> str:
        """El audio, en base64, dentro de un mensaje de chat — así transcriben
        los proveedores compatibles: no hay endpoint de audio, hay modelos que
        oyen. Se le pide el texto y NADA más: sin esa instrucción, un modelo
        servicial contesta a lo que dijo el cliente en vez de transcribirlo, y
        Nea acabaría respondiendo a su propia paráfrasis."""
        fmt = _formato_de_audio(mime)
        b64 = base64.b64encode(data).decode()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._chat_transcribe_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Transcribe este audio al español. "
                                        "Responde SOLO con la transcripción "
                                        "literal, sin comillas, sin comentarios "
                                        "y sin responder a lo que dice."
                                    ),
                                },
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": b64, "format": fmt},
                                },
                            ],
                        }
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                last_error = ValueError("transcripción vacía")
                logger.warning("transcribe(chat): vacío, intento %d", attempt + 1)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "transcribe(chat) con %s: fallo en intento %d: %s",
                    self._chat_transcribe_model,
                    attempt + 1,
                    exc,
                )
            if attempt == 0:
                await asyncio.sleep(1.0)
        raise LlmExhausted(str(last_error))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LlmReply:
        last_error: Exception | None = None
        for attempt in range(self.RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = await self._client.chat.completions.create(
                    model=self._model, messages=messages, **kwargs
                )
                u = getattr(resp, "usage", None)
                if u is not None:
                    det = getattr(u, "prompt_tokens_details", None)
                    self.usage["llamadas"] += 1
                    self.usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["completion"] += getattr(u, "completion_tokens", 0) or 0
                    self.usage["cached"] += getattr(det, "cached_tokens", 0) or 0
                reply = self._parse(resp)
                if reply.content or reply.tool_calls:
                    return reply
                last_error = ValueError("respuesta vacía del LLM (sin content ni tools)")
                logger.warning("llm: respuesta vacía, intento %d", attempt + 1)
            except APIStatusError as exc:
                sin_reintento = _sin_reintento(exc)
                if sin_reintento is not None:
                    raise sin_reintento(str(exc)) from exc
                last_error = exc
                logger.warning("llm: fallo en intento %d: %s", attempt + 1, exc)
            except Exception as exc:  # red, parseo — todo reintenta
                last_error = exc
                logger.warning("llm: fallo en intento %d: %s", attempt + 1, exc)
            if attempt < self.RETRIES:
                await asyncio.sleep(2**attempt)  # 1 s, 2 s
        raise LlmExhausted(str(last_error))

    async def aclose(self) -> None:
        """Cierra el/los cliente(s) HTTP subyacentes. Para el cliente de
        plataforma (larga vida) se llama al apagar el proceso; para un
        cliente por turno (clave de un negocio, ver app/stateless.py) se
        llama SIEMPRE al terminar ESE turno, éxito o fallo."""
        await self._client.close()
        if self._whisper_client is not None and self._whisper_client is not self._client:
            await self._whisper_client.close()

    @staticmethod
    def _parse(resp: Any) -> LlmReply:
        """Extracción tolerante: nunca truena por formato inesperado."""
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return LlmReply(content=None)
        message = getattr(choices[0], "message", None)
        if message is None:
            return LlmReply(content=None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            content = content.strip() or None
        else:
            content = None
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            if not name:
                continue
            raw_args = getattr(fn, "arguments", None) or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except (TypeError, ValueError):
                logger.warning("llm: arguments malformados en %s — uso {}", name)
                args = {}
            tool_calls.append(
                ToolCall(id=getattr(tc, "id", "") or "", name=name, arguments=args)
            )
        return LlmReply(content=content, tool_calls=tool_calls)
