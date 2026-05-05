"""talkie-lm backend — wraps the `talkie` PyTorch inference package as
a chatlm-compatible chat backend.

Why this exists
---------------
The talkie-lm 13B models (`talkie-1930-13b-base`, `talkie-1930-13b-it`,
`talkie-web-13b-base`) ship as raw PyTorch pickles with a custom
architecture (Llama-shape + per-layer/per-head gain params + an
embedding-skip residual that no GGUF runtime supports). Until a
community GGUF mirror exists, the only way to run them is the team's
own Python inference stack — which we wrap here.

Routing
-------
External names use the `talkie:` prefix. The full set the upstream
package recognises is in `talkie.MODELS`:

  - talkie:1930-13b-it     (instruction-tuned; the only chat-usable one)
  - talkie:1930-13b-base   (base; raw text completion)
  - talkie:web-13b-base    (modern-corpus control variant; base)

Strip the prefix and pass `talkie-` + the suffix to the upstream package.

Process model
-------------
Single-resident: one model loaded at a time. ~26 GB resident in bf16,
so the 36 GB M3 Pro can hold it comfortably as long as nothing else is
hogging unified memory. Switching models drops the previous one and
loads the new one; model loading takes ~30-60 s from local cache.

Wire format
-----------
The talkie package exposes `model.chat(messages, ...)` (returns a
`GenerationResult`) and `model.chat_stream(messages, ...)` (yields
token strings). We translate these to the Ollama-shaped envelope the
rest of chatlm consumes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

log = logging.getLogger("chatlm.talkie")

PREFIX = "talkie:"


def is_talkie_name(name: str) -> bool:
    return name.startswith(PREFIX)


def strip_prefix(name: str) -> str:
    return name.removeprefix(PREFIX)


def _canonical_model_name(stripped: str) -> str:
    """Map the chatlm-style suffix to the upstream package's model id.

    Accepts both bare suffixes (`1930-13b-it`) and already-prefixed names
    (`talkie-1930-13b-it`) so caller code doesn't need to know which
    convention was used."""
    if stripped.startswith("talkie-"):
        return stripped
    return f"talkie-{stripped}"


# ── singleton state ─────────────────────────────────────────────────


# `_active` holds (canonical_name, Talkie_instance) when a model is
# loaded. Guarded by `_load_lock` for swap-safety; inference runs
# serialised under `_infer_lock` so concurrent requests can't trample
# the model's KV cache (talkie's `chat`/`chat_stream` are stateful
# during a generation but not reentrant).
_active: tuple[str, Any] | None = None
_load_lock = asyncio.Lock()
_infer_lock = asyncio.Lock()


def _pick_device() -> str:
    """Apple Silicon prefers MPS; the talkie package's auto-detect only
    knows CUDA / CPU. Pick MPS explicitly when available so the model
    actually uses the Metal backend instead of falling back to CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_messages(messages: list[dict]):
    """Convert chatlm-shaped message dicts into talkie's `Message`
    dataclass list. Drops fields talkie doesn't understand
    (`images`, `audios`, `tool_calls`) — the talkie models are
    text-only."""
    from talkie import Message
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant", "system"):
            # Tool messages and other vocabularies aren't supported;
            # surface them as system context so the model still sees
            # something rather than silently dropping content.
            role = "system"
        out.append(Message(role=role, content=m.get("content", "") or ""))
    return out


async def _ensure_loaded(canonical: str):
    """Load (or reuse) the talkie model. Blocking — runs in a worker
    thread so the asyncio loop stays responsive."""
    global _active
    async with _load_lock:
        if _active is not None and _active[0] == canonical:
            return _active[1]

        if _active is not None:
            log.info(f"unloading previous talkie model {_active[0]!r}")
            _active = None  # drops the old reference; gc reclaims weights
            import gc; gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        device = _pick_device()
        log.info(f"loading talkie model {canonical!r} on device={device}")
        t0 = time.perf_counter()

        def _load():
            from talkie import Talkie
            return Talkie(canonical, device=device)

        model = await asyncio.to_thread(_load)
        log.info(f"talkie {canonical!r} ready in {time.perf_counter()-t0:.1f}s")
        _active = (canonical, model)
        return model


# ── public chat surface (mirrors OllamaClient) ──────────────────────


async def capabilities(model: str) -> frozenset[str]:
    """All talkie variants are text-only, no tool calling, no thinking
    channel. Return the same flat set everywhere."""
    return frozenset({"completion"})


async def chat(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    think: bool = False,  # accepted for surface parity; ignored
    format: Any = None,
    tools: list[dict] | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    canonical = _canonical_model_name(strip_prefix(model))
    talkie_model = await _ensure_loaded(canonical)
    msg_list = _build_messages(messages)

    def _run():
        # talkie expects integer max_tokens; default to a reasonable
        # cap when the caller leaves it None (matches OllamaClient).
        kwargs: dict = {"temperature": temperature, "max_tokens": max_tokens or 512}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        return talkie_model.chat(msg_list, **kwargs)

    t0 = time.perf_counter()
    async with _infer_lock:
        result = await asyncio.to_thread(_run)
    elapsed_ns = int((time.perf_counter() - t0) * 1e9)

    return {
        "model": model,
        "message": {
            "role": "assistant",
            "content": getattr(result, "text", "") or "",
            "tool_calls": None,
        },
        "done": True,
        "done_reason": getattr(result, "finish_reason", "stop") or "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,  # talkie doesn't expose this
        "prompt_eval_duration": 0,
        "eval_count": getattr(result, "token_count", 0) or 0,
        "eval_duration": elapsed_ns,
    }


async def chat_stream(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    think: bool = False,  # accepted for surface parity; ignored
    tools: list[dict] | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> AsyncIterator[str]:
    canonical = _canonical_model_name(strip_prefix(model))
    talkie_model = await _ensure_loaded(canonical)
    msg_list = _build_messages(messages)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    total_tokens = 0
    t0 = time.perf_counter()

    def _produce():
        nonlocal total_tokens
        try:
            kwargs: dict = {"temperature": temperature, "max_tokens": max_tokens or 512}
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
            for tok in talkie_model.chat_stream(msg_list, **kwargs):
                if not tok:
                    continue
                total_tokens += 1
                evt = {
                    "model": model,
                    "message": {"role": "assistant", "content": tok},
                    "done": False,
                }
                asyncio.run_coroutine_threadsafe(
                    queue.put(json.dumps(evt)), loop
                )
            elapsed_ns = int((time.perf_counter() - t0) * 1e9)
            final = {
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": elapsed_ns,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": total_tokens,
                "eval_duration": elapsed_ns,
            }
            asyncio.run_coroutine_threadsafe(queue.put(json.dumps(final)), loop)
        except Exception as err:
            log.error(f"talkie stream error: {err}")
            asyncio.run_coroutine_threadsafe(
                queue.put(json.dumps({"error": str(err)})), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    await _infer_lock.acquire()
    try:
        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        _infer_lock.release()


async def list_models() -> dict:
    """Surface every talkie model the upstream package knows about, so
    they appear in chatlm's dropdown without manual configuration."""
    try:
        from talkie import MODELS
    except Exception as err:
        log.debug(f"talkie list_models failed: {err}")
        return {"models": []}
    return {
        "models": [
            {
                "name": f"{PREFIX}{name.removeprefix('talkie-')}",
                "details": {"family": "talkie"},
                "size": 0,
            }
            for name in sorted(MODELS)
        ]
    }


async def close() -> None:
    """Drop the resident model on shutdown so we don't hold ~26 GB
    longer than necessary if the process keeps running for cleanup."""
    global _active
    _active = None
    import gc; gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
