import asyncio
import logging
from typing import AsyncIterator

import httpx

from config import settings

log = logging.getLogger("chatlm.ollama")


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, host: str, timeout: int):
        # `timeout` is treated as the per-request CEILING. We override the
        # READ timeout to None: large reasoning models in think mode can sit
        # silent for 30+ s between streamed tokens, and a fixed read timeout
        # would kill the SSE connection mid-thought (manifests as
        # "network error" in the browser). Connect/write/pool stay finite
        # so a wedged daemon still surfaces quickly.
        self._client = httpx.AsyncClient(
            base_url=host,
            timeout=httpx.Timeout(
                connect=10.0,
                read=None,
                write=float(timeout),
                pool=10.0,
            ),
        )
        # Cache of /api/show capabilities per model. Capabilities are a
        # property of the installed model layer, so a process-lifetime
        # cache is correct: pulling a new tag would require a restart to
        # see updated caps, which is rare enough not to design around.
        self._capabilities: dict[str, frozenset[str]] = {}
        self._capabilities_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def capabilities(self, model: str) -> frozenset[str]:
        """Return Ollama's reported capabilities for `model` (e.g. {'completion',
        'tools', 'thinking', 'vision'}). Empty set if /api/show fails — caller
        should treat 'unknown' the same as 'unsupported' to avoid sending
        param flags the model can't handle."""
        cached = self._capabilities.get(model)
        if cached is not None:
            return cached
        async with self._capabilities_lock:
            cached = self._capabilities.get(model)
            if cached is not None:
                return cached
            try:
                info = await self._post_json("/api/show", {"model": model})
                caps = frozenset(info.get("capabilities") or [])
            except Exception as err:
                log.debug(f"capabilities probe failed for {model!r}: {err}")
                caps = frozenset()
            self._capabilities[model] = caps
            return caps

    async def _resolve_think(self, model: str, think: bool) -> bool:
        """Downgrade `think=True` to False when the model lacks the
        'thinking' capability — Ollama otherwise returns 400 and the chat
        request fails outright. Lets the UI keep a single 'think' toggle
        without per-model gating."""
        if not think:
            return False
        if "thinking" in await self.capabilities(model):
            return True
        log.info(f"think=True requested for {model!r} but model lacks 'thinking' capability — downgrading to False")
        return False

    async def supports_tools(self, model: str) -> bool:
        """True iff the model self-reports a `tools` capability. Used by
        callers to decide whether to send `tools=[...]` to this model or
        re-route to a tool-capable fallback. Ollama returns 400
        ('does not support tools') if you send tools to a model that
        can't use them — same failure shape as `think`."""
        return "tools" in await self.capabilities(model)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        think: bool = False,
        format: str | dict | None = None,
        tools: list[dict] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": await self._resolve_think(model, think),
            "options": _build_options(temperature, max_tokens, top_p, top_k),
        }
        if format is not None:
            payload["format"] = format
        if tools:
            payload["tools"] = tools
        return await self._post_json("/api/chat", payload)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        think: bool = False,
        tools: list[dict] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": await self._resolve_think(model, think),
            "options": _build_options(temperature, max_tokens, top_p, top_k),
        }
        if tools:
            payload["tools"] = tools
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            if response.is_error:
                # Streaming responses don't have .text populated until you
                # explicitly read the body; doing it inline gives a real
                # error message instead of `httpx.ResponseNotRead`.
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise OllamaError(f"Ollama {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": _build_options(temperature, max_tokens),
        }
        return await self._post_json("/api/generate", payload)

    async def list_models(self) -> dict:
        return await self._get_json("/api/tags")

    async def list_resident(self) -> list[dict]:
        """Models currently loaded in the daemon (vs. just installed on disk)."""
        try:
            data = await self._get_json("/api/ps")
        except Exception:
            return []
        return data.get("models", []) or []

    async def unload(self, model: str) -> None:
        """Force the daemon to evict a loaded model immediately by issuing
        a zero-token generate with keep_alive=0. Used to free unified
        memory before loading a heavy diffusion pipe."""
        payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        try:
            await self._post_json("/api/generate", payload)
        except Exception:
            pass  # best-effort; unload errors should never break the caller

    async def unload_all(self) -> int:
        """Unload every model currently resident in the daemon. Returns count."""
        resident = await self.list_resident()
        for m in resident:
            await self.unload(m.get("name") or m.get("model"))
        return len(resident)

    async def _post_json(self, path: str, payload: dict) -> dict:
        response = await self._client.post(path, json=payload)
        _raise_for_status(response)
        return response.json()

    async def _get_json(self, path: str) -> dict:
        response = await self._client.get(path)
        _raise_for_status(response)
        return response.json()


def _build_options(
    temperature: float,
    max_tokens: int | None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    return options


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise OllamaError(f"Ollama {response.status_code}: {response.text}")


client = OllamaClient(settings.ollama_host, settings.request_timeout)
