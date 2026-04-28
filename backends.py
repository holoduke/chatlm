"""Backend dispatch + auto-fallback policy.

Three backends serve chat:
- **Ollama** (default) — daemon at :11434
- **MLX-VLM** (`mlx:` prefix) — Apple Silicon native
- **llama-server** (`llama:` prefix) — latest llama.cpp; covers GGUFs
  Ollama's bundled llama.cpp can't parse (e.g. `gemma4` arch).

The fallback cache below tracks Ollama-incapable model names. Once an
attempt fails with a known fingerprint, the same model rerouted to
llama-server on subsequent calls — no second failed roundtrip.

This module is the single place where "which backend serves this name"
is decided. Endpoint code in `main.py` calls into here and never
reaches into `_OLLAMA_INCAPABLE` directly. Tests under
`tests/test_main_routing.py` exercise the policy in isolation.
"""
from __future__ import annotations

import llama_server_client
import mlx_client
from ollama_client import client as _ollama_client


# Models we've discovered Ollama can't load. Once a name lands here,
# `_dispatch` returns the llama-server backend on subsequent calls.
# Lifetime: process-resident; restart picks up Ollama upgrades.
#
# Concurrency: this is a plain set, mutated under the asyncio event
# loop in a single thread. CPython's `set.add` is atomic at the
# bytecode level, and the duplicate-add case (two requests for the
# same model both losing the Ollama probe) is benign — both add the
# same value, and `LlamaServerClient._lock` serialises the actual
# spawn. No asyncio.Lock needed.
OLLAMA_INCAPABLE: set[str] = set()


# Substrings in an OllamaError message that mean "Ollama's bundled
# llama.cpp doesn't understand this GGUF". We retry against
# llama-server. Fingerprints kept tight — we don't want to mask
# unrelated 5xx errors (daemon down, OOM, etc.) by silently switching
# backends.
FALLBACK_FINGERPRINTS = (
    "unknown model architecture",
    "unable to load model",
)


def should_fallback(err_text: str) -> bool:
    return any(fp in err_text for fp in FALLBACK_FINGERPRINTS)


def dispatch(model_name: str):
    """Pick the right chat backend based on model name prefix."""
    if mlx_client.is_mlx_name(model_name):
        return mlx_client
    if llama_server_client.is_llama_name(model_name):
        return llama_server_client.client
    if model_name in OLLAMA_INCAPABLE:
        return llama_server_client.client
    return _ollama_client


def resolve_model_for_dispatch(model_name: str) -> str:
    """If a name is in the fallback cache, transparently rewrite it to
    its `llama:` form so the downstream backend gets the prefix it
    expects. No-op for names that already carry an explicit prefix."""
    if (
        model_name in OLLAMA_INCAPABLE
        and not llama_server_client.is_llama_name(model_name)
        and not mlx_client.is_mlx_name(model_name)
    ):
        return llama_server_client.PREFIX + model_name
    return model_name


def label(backend) -> str:
    """Human-readable backend tag for logs / /stats responses."""
    if backend is mlx_client:
        return "mlx"
    if backend is llama_server_client.client:
        return "llama-server"
    return "ollama"


def mark_incapable(model_name: str) -> None:
    """Record that Ollama can't load this model. Future calls skip the
    failing roundtrip and go straight to llama-server."""
    OLLAMA_INCAPABLE.add(model_name)
