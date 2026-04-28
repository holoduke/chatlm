"""End-to-end test for the auto-fallback path in `/chat/stream`.

Mocks the OllamaClient and LlamaServerClient at the singleton level so
we can exercise main.py's fallback orchestration without spinning up
either real backend. Covers three scenarios that pure-function tests
can't: happy path, fallback-on-fingerprint, and non-fingerprint errors
that should NOT trigger a fallback.
"""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import llama_server_client
import main
from ollama_client import OllamaError, client as ollama


pytestmark = pytest.mark.asyncio


def _ollama_done_chunk(model: str) -> str:
    """Final NDJSON frame Ollama emits — keeps the chatlm response
    pipeline's `if evt.get('done'): final = evt` branch satisfied."""
    return json.dumps({
        "model": model,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    })


def _content_chunk(model: str, text: str) -> str:
    return json.dumps({
        "model": model,
        "message": {"role": "assistant", "content": text},
        "done": False,
    })


@pytest.fixture
def reset_fallback_cache():
    """Don't let stale entries from one test bleed into the next —
    `_OLLAMA_INCAPABLE` is module-global."""
    main._OLLAMA_INCAPABLE.clear()
    yield
    main._OLLAMA_INCAPABLE.clear()


@pytest_asyncio.fixture
async def http():
    """ASGITransport bypasses the lifespan handler, so warmup +
    MCP-restore don't make outbound calls during tests."""
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_happy_path_stays_on_ollama(monkeypatch, http, reset_fallback_cache):
    """No error, no fallback — chunks come from Ollama, the cache stays
    empty, and the response forwards the Ollama-emitted NDJSON."""
    model = "huihui_ai/qwen3-abliterated:8b"

    async def fake_ollama_stream(**kwargs):
        yield _content_chunk(model, "hi")
        yield _ollama_done_chunk(model)

    async def boom_llama_stream(**kwargs):
        raise AssertionError("llama-server must not be called on a happy Ollama path")
        yield  # pragma: no cover — keeps this an async generator

    monkeypatch.setattr(ollama, "chat_stream", fake_ollama_stream)
    monkeypatch.setattr(llama_server_client.client, "chat_stream", boom_llama_stream)

    r = await http.post("/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
    })
    assert r.status_code == 200
    assert '"content": "hi"' in r.text
    assert model not in main._OLLAMA_INCAPABLE


async def test_fallback_fires_on_unable_to_load_fingerprint(
    monkeypatch, http, reset_fallback_cache
):
    """The exact failure mode that prompted this whole feature:
    Ollama 500 'unable to load model' on a GGUF its bundled llama.cpp
    can't parse. The streaming endpoint must catch it before yielding,
    swap to llama-server, and replay from the start."""
    model = "hf.co/test/broken-arch:Q4"

    async def boom_ollama_stream(**kwargs):
        raise OllamaError(
            'Ollama 500: {"error":"unable to load model: /path/to/blob"}'
        )
        yield  # pragma: no cover

    async def fake_llama_stream(**kwargs):
        # Verify the model name was rewritten with the `llama:` prefix
        # before being handed to the llama-server backend.
        assert kwargs["model"] == llama_server_client.PREFIX + model
        yield _content_chunk(kwargs["model"], "fallback-ok")
        yield _ollama_done_chunk(kwargs["model"])

    monkeypatch.setattr(ollama, "chat_stream", boom_ollama_stream)
    monkeypatch.setattr(llama_server_client.client, "chat_stream", fake_llama_stream)

    r = await http.post("/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
    })
    assert r.status_code == 200
    assert '"content": "fallback-ok"' in r.text
    # And the next request for this model should skip Ollama entirely.
    assert model in main._OLLAMA_INCAPABLE


async def test_unknown_architecture_fingerprint_also_triggers_fallback(
    monkeypatch, http, reset_fallback_cache
):
    """Second known fingerprint — the older error string Ollama returned
    before recent versions normalised to 'unable to load model'."""
    model = "hf.co/test/gemma4-arch:Q4"

    async def boom_ollama_stream(**kwargs):
        raise OllamaError(
            'Ollama 500: {"error":"unknown model architecture: \'gemma4\'"}'
        )
        yield  # pragma: no cover

    async def fake_llama_stream(**kwargs):
        yield _content_chunk(kwargs["model"], "via-llama")
        yield _ollama_done_chunk(kwargs["model"])

    monkeypatch.setattr(ollama, "chat_stream", boom_ollama_stream)
    monkeypatch.setattr(llama_server_client.client, "chat_stream", fake_llama_stream)

    r = await http.post("/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
    })
    assert r.status_code == 200
    assert '"content": "via-llama"' in r.text
    assert model in main._OLLAMA_INCAPABLE


async def test_non_fingerprint_error_does_not_fallback(
    monkeypatch, http, reset_fallback_cache
):
    """A 503 / 400 / connect error has nothing to do with arch
    incompatibility, so silently switching to llama-server would mask
    real outages. The endpoint should surface the error in-band and
    leave the cache empty."""
    model = "huihui_ai/qwen3-abliterated:8b"

    async def boom_ollama_stream(**kwargs):
        raise OllamaError("Ollama 503: connection refused")
        yield  # pragma: no cover

    async def boom_llama_stream(**kwargs):
        raise AssertionError("llama-server must not run for a non-fingerprint error")
        yield  # pragma: no cover

    monkeypatch.setattr(ollama, "chat_stream", boom_ollama_stream)
    monkeypatch.setattr(llama_server_client.client, "chat_stream", boom_llama_stream)

    r = await http.post("/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
    })
    assert r.status_code == 200  # streaming endpoint puts errors in-band
    # The error should appear as a JSON `error` field in the body, not
    # as a fallback completion.
    assert '"error"' in r.text
    assert "connection refused" in r.text
    assert model not in main._OLLAMA_INCAPABLE
