"""main.py routing helpers — backend dispatch and fallback fingerprinting.

Pure logic only. Endpoint behaviour is covered by the live-server
validation script."""
from __future__ import annotations

import llama_server_client
import main
import mlx_client
from ollama_client import client as ollama_client


def test_should_fallback_matches_known_fingerprints():
    assert main._should_fallback('Ollama 500: {"error":"unknown model architecture: gemma4"}')
    assert main._should_fallback('Ollama 500: unable to load model: /path/to/blob')


def test_should_fallback_ignores_unrelated_errors():
    assert not main._should_fallback("Ollama 503: connection refused")
    assert not main._should_fallback("Ollama 400: invalid request")
    assert not main._should_fallback("")


def test_dispatch_mlx_prefix_routes_to_mlx():
    assert main._dispatch("mlx:gemma-4-e2b-it-4bit") is mlx_client


def test_dispatch_llama_prefix_routes_to_llama_server():
    assert main._dispatch("llama:hf.co/foo/bar:tag") is llama_server_client.client


def test_dispatch_default_routes_to_ollama():
    assert main._dispatch("huihui_ai/qwen3-abliterated:8b") is ollama_client
    assert main._dispatch("gemma4:e2b") is ollama_client


def test_dispatch_uses_fallback_cache_to_skip_failed_ollama_models():
    """Models that previously failed with 'unable to load' should
    short-circuit straight to llama-server next time."""
    bad = "hf.co/test/broken:Q4"
    main._OLLAMA_INCAPABLE.add(bad)
    try:
        assert main._dispatch(bad) is llama_server_client.client
    finally:
        main._OLLAMA_INCAPABLE.discard(bad)


def test_resolve_model_for_dispatch_rewrites_cached_failures():
    bad = "hf.co/test/broken:Q4"
    main._OLLAMA_INCAPABLE.add(bad)
    try:
        assert main._resolve_model_for_dispatch(bad) == "llama:" + bad
    finally:
        main._OLLAMA_INCAPABLE.discard(bad)


def test_resolve_model_for_dispatch_leaves_uncached_names_alone():
    name = "huihui_ai/qwen3-abliterated:8b"
    assert main._resolve_model_for_dispatch(name) == name


def test_resolve_model_for_dispatch_doesnt_double_prefix():
    """If the name already has the llama: prefix, don't add a second one."""
    name = "llama:hf.co/foo:tag"
    main._OLLAMA_INCAPABLE.add(name)  # contrived but defensive
    try:
        assert main._resolve_model_for_dispatch(name) == name
    finally:
        main._OLLAMA_INCAPABLE.discard(name)


def test_backend_label_returns_correct_string_per_backend():
    assert main._backend_label(mlx_client) == "mlx"
    assert main._backend_label(llama_server_client.client) == "llama-server"
    assert main._backend_label(ollama_client) == "ollama"
