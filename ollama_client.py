from typing import AsyncIterator

import httpx

from config import settings


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, host: str, timeout: int):
        self._client = httpx.AsyncClient(base_url=host, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        think: bool = False,
        format: str | dict | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": _build_options(temperature, max_tokens),
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
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": _build_options(temperature, max_tokens),
        }
        if tools:
            payload["tools"] = tools
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            _raise_for_status(response)
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

    async def _post_json(self, path: str, payload: dict) -> dict:
        response = await self._client.post(path, json=payload)
        _raise_for_status(response)
        return response.json()

    async def _get_json(self, path: str) -> dict:
        response = await self._client.get(path)
        _raise_for_status(response)
        return response.json()


def _build_options(temperature: float, max_tokens: int | None) -> dict:
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    return options


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise OllamaError(f"Ollama {response.status_code}: {response.text}")


client = OllamaClient(settings.ollama_host, settings.request_timeout)
