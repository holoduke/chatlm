"""Backend that drives a `llama-server` (latest llama.cpp) subprocess for
GGUF models that Ollama 0.21.2's bundled llama.cpp can't load.

Why this exists
---------------
Ollama vendors a llama.cpp build that lags upstream by weeks/months. Newer
GGUFs (e.g. anything with `general.architecture=gemma4`, merged upstream
2026-04-02 in PR #21309) fail with "unknown model architecture: 'gemma4'".
We bridge that gap by spawning the brew-installed `llama-server` (b8920+
at time of writing) on a side port and routing requests to it.

Routing
-------
External model names are prefixed with `llama:`. Strip the prefix, look
up the GGUF blob in Ollama's local cache (no double-download), spawn or
re-aim a single llama-server child process at it.

Process model
-------------
Single-resident: one llama-server process at a time. Model swap kills
and respawns (~5s ready). Cheaper than running N parallel servers on a
36 GB box, simpler than a process pool, and matches the existing
single-resident pattern in `txt2img.py` / `mlx_client.py`.

Wire format
-----------
llama-server speaks OpenAI-compatible /v1/chat/completions. We translate
to/from the Ollama-shaped JSON the rest of chatlm expects so callers
don't need a third codepath.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("chatlm.llama_server")

PREFIX = "llama:"
PORT = 11436
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"

OLLAMA_ROOT = Path.home() / ".ollama" / "models"
MANIFEST_DIR = OLLAMA_ROOT / "manifests"
BLOB_DIR = OLLAMA_ROOT / "blobs"

# Spawn args. Tuned for Apple Silicon: -ngl 99 offloads everything to Metal,
# --jinja honours the GGUF's own chat template (matches the model's expected
# format instead of the generic OpenAI default). 16K context covers chat
# without paying the full 131K KV-cache cost on smaller boxes; bump via
# CHATLM_LLAMA_CTX env var for long-context workloads.
_DEFAULT_CTX = int(os.environ.get("CHATLM_LLAMA_CTX", "16384"))
_SPAWN_ARGS = ["-ngl", "99", "--jinja", "-c", str(_DEFAULT_CTX)]


def is_llama_name(name: str) -> bool:
    return name.startswith(PREFIX)


def strip_prefix(name: str) -> str:
    return name.removeprefix(PREFIX)


def _split_name(name: str) -> tuple[str, str, str]:
    """Decompose `[registry/]repo[:tag]` into (registry, repo, tag).

    - If the part before the first `/` contains a dot, it's treated as a
      registry hostname (e.g. `hf.co/...`).
    - Otherwise the default Ollama registry is assumed. A bare `model:tag`
      (no slash) is rewritten to `library/model` to match Ollama's path
      layout for the implicit library namespace.
    """
    base, _, tag = name.partition(":")
    if not tag:
        tag = "latest"
    if "/" in base:
        head, _, rest = base.partition("/")
        if "." in head:
            return head, rest, tag
        return "registry.ollama.ai", base, tag
    return "registry.ollama.ai", f"library/{base}", tag


_MEDIA_MODEL = "application/vnd.ollama.image.model"
_MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"


def _read_manifest(name: str) -> tuple[Path, dict]:
    registry, repo, tag = _split_name(strip_prefix(name))
    manifest_path = MANIFEST_DIR / registry / repo / tag
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no Ollama manifest at {manifest_path} — pull the model first "
            f"(`ollama pull {strip_prefix(name)}`)"
        )
    return manifest_path, json.loads(manifest_path.read_text())


def _layer_blob(manifest: dict, mediatype: str) -> Path | None:
    """Return the blob path for the first layer matching `mediatype`, or
    None if absent. Existence of the on-disk blob is verified."""
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == mediatype:
            digest = layer["digest"].removeprefix("sha256:")
            blob = BLOB_DIR / f"sha256-{digest}"
            return blob if blob.is_file() else None
    return None


def resolve_blob_path(name: str) -> Path:
    """Find the GGUF model blob for an Ollama-cached model.

    Raises FileNotFoundError if the manifest doesn't exist (model not
    pulled) or no model-mediatype layer is present (manifest malformed).
    """
    manifest_path, manifest = _read_manifest(name)
    blob = _layer_blob(manifest, _MEDIA_MODEL)
    if blob is None:
        raise FileNotFoundError(f"manifest {manifest_path} has no model-mediatype layer")
    return blob


def resolve_projector_path(name: str) -> Path | None:
    """Return the multimodal `mmproj` projector blob for the model, if
    one was pulled. Required for vision/audio in models like HauhauCS's
    Gemma 4 GGUFs — passed to llama-server via `--mmproj`. Returns None
    when the model is text-only or no projector layer was published."""
    try:
        _, manifest = _read_manifest(name)
    except FileNotFoundError:
        return None
    return _layer_blob(manifest, _MEDIA_PROJECTOR)


# ── Translation helpers ────────────────────────────────────────────────


def _build_payload(
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    tools: list[dict] | None,
    top_p: float | None,
    top_k: int | None,
    *,
    stream: bool,
) -> dict:
    """Shape the OpenAI /v1/chat/completions request body. Single helper
    so streaming and non-streaming codepaths can't drift on which
    sampling fields they include."""
    payload: dict = {
        "messages": _normalise_messages(messages),
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        # llama-server accepts `top_k` directly (extension over OpenAI spec).
        payload["top_k"] = top_k
    return payload


def _normalise_messages(messages: list[dict]) -> list[dict]:
    """Reshape Ollama-style messages into the OpenAI form llama-server's
    `/v1/chat/completions` validates against. Three known mismatches:

    1. tool_calls produced by Ollama / chatlm history lack `type: "function"`.
       OpenAI's spec requires it; llama-server returns 500
       ('Missing tool call type') if it's absent.
    2. Images & audios: Ollama puts sibling `images: [b64,...]` and
       `audios: [b64,...]` arrays on user messages. OpenAI wants them
       as content parts:
       `[{type:"text",...},
         {type:"image_url", image_url:{url}},
         {type:"input_audio", input_audio:{data, format}}]`.
       llama-server only sees attachments via the OpenAI shape.
    3. Tool-result messages need role `tool` plus `tool_call_id` — Ollama
       sometimes uses `name` as the linkage. Best-effort fill-in.
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        copy = dict(msg)
        tcs = copy.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            copy["tool_calls"] = [_normalise_tool_call(tc) for tc in tcs]
        imgs = copy.pop("images", None)
        auds = copy.pop("audios", None)
        if imgs or auds:
            copy["content"] = _attach_media(copy.get("content", ""), imgs, auds)
        out.append(copy)
    return out


def _attach_media(
    text_content: str,
    images: list[str] | None,
    audios: list[str] | None,
) -> list[dict]:
    """Build the OpenAI content-parts array from a plain string plus
    Ollama-style base64 image/audio lists. Each base64 is wrapped in a
    data URL (images) or `input_audio` block (audios) so llama-server
    accepts them without a network round-trip."""
    parts: list[dict] = []
    if text_content:
        parts.append({"type": "text", "text": text_content})
    for b64 in images or []:
        if not isinstance(b64, str):
            continue
        # Ollama accepts raw base64; OpenAI wants `data:image/...;base64,...`.
        # Default to image/png — almost all real images we get are PNG/JPG
        # and llama-server's mime sniff handles the latter.
        url = b64 if b64.startswith("data:") else f"data:image/png;base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    for b64 in audios or []:
        if not isinstance(b64, str):
            continue
        # OpenAI's input_audio expects raw base64 (no data: prefix) plus
        # an explicit `format`. WAV is the safest default — ffmpeg-based
        # multimodal decoders inside llama.cpp probe the magic bytes
        # anyway, so a minor mislabel is recoverable.
        parts.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
    return parts


def _normalise_tool_call(tc: dict) -> dict:
    """Force the OpenAI-required `type: "function"` and stringify the
    arguments payload — OpenAI expects `arguments` to be a JSON-encoded
    string, not a dict."""
    if not isinstance(tc, dict):
        return tc
    out = dict(tc)
    out.setdefault("type", "function")
    fn = out.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            fn["arguments"] = json.dumps(args)
        out["function"] = fn
    return out


def _openai_to_ollama_chat(model: str, payload: dict) -> dict:
    """Reshape an OpenAI /v1/chat/completions response into Ollama's
    /api/chat envelope, which is what main.py consumes.

    llama-server splits "thinking" output into a sibling
    `reasoning_content` field (gemma 4 / qwen3 / deepseek-r1 emit there
    when the runtime detects a thinking model). We surface it as
    Ollama's `thinking` field so the existing UI replay path picks it up.
    Also: when only `reasoning_content` is populated (e.g. the model ran
    out of tokens before the final answer), fall back to it as `content`
    so users at least see *something* instead of an empty bubble.
    """
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    elapsed_ns = int((timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) * 1e6)
    content = msg.get("content") or ""
    thinking = msg.get("reasoning_content") or ""
    if not content and thinking:
        content = thinking
    return {
        "model": model,
        "message": {
            "role": msg.get("role", "assistant"),
            "content": content,
            "thinking": thinking or None,
            "tool_calls": msg.get("tool_calls"),
        },
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": int(timings.get("prompt_ms", 0) * 1e6),
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": int(timings.get("predicted_ms", 0) * 1e6),
    }


def _openai_stream_to_ollama_chunk(model: str, sse_line: str) -> str | None:
    """Translate one OpenAI SSE `data:` line into one Ollama NDJSON line.
    Returns None for keepalive / `[DONE]` markers (callers skip those)."""
    if not sse_line.startswith("data:"):
        return None
    body = sse_line[len("data:"):].strip()
    if body == "[DONE]":
        return None
    try:
        evt = json.loads(body)
    except json.JSONDecodeError:
        return None
    choice = (evt.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    finish = choice.get("finish_reason")
    if finish:
        return json.dumps({
            "model": model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": finish,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        })
    # llama-server emits `reasoning_content` deltas separately from
    # `content` deltas. The chatlm UI reads `message.thinking` for the
    # collapsed reasoning panel (matches the Ollama qwen3 path), so map
    # it through. If neither field is present in this delta, drop the
    # frame.
    content = delta.get("content")
    thinking = delta.get("reasoning_content")
    if content is None and thinking is None:
        return None
    msg: dict = {"role": "assistant", "content": content or ""}
    if thinking:
        msg["thinking"] = thinking
    return json.dumps({"model": model, "message": msg, "done": False})


# ── Subprocess + HTTP client ───────────────────────────────────────────


class LlamaServerClient:
    def __init__(self) -> None:
        # READ timeout None mirrors OllamaClient — long thinking pauses must
        # not kill an in-flight stream.
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=10.0, read=None, write=120.0, pool=10.0),
        )
        self._proc: subprocess.Popen | None = None
        self._loaded_model: str | None = None  # canonical (no prefix) name
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()
        self._kill_proc()

    # ----- lifecycle -------------------------------------------------

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            log.info(f"stopping llama-server pid={self._proc.pid}")
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._loaded_model = None

    async def _spawn(self, canonical: str) -> None:
        blob = resolve_blob_path(canonical)
        mmproj = resolve_projector_path(canonical)
        extra: list[str] = []
        if mmproj is not None:
            extra.extend(["--mmproj", str(mmproj)])
            log.info(
                f"spawning llama-server for {canonical!r} blob={blob.name} "
                f"mmproj={mmproj.name} (multimodal enabled)"
            )
        else:
            log.info(f"spawning llama-server for {canonical!r} blob={blob.name}")
        self._proc = subprocess.Popen(
            [
                "llama-server",
                "-m", str(blob),
                "--host", HOST,
                "--port", str(PORT),
                *_SPAWN_ARGS,
                *extra,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # No `preexec_fn=os.setsid`: putting the child in a new session
            # would shield it from the parent's terminal SIGTERM, leaving an
            # orphan on :11436 whenever uvicorn is hard-killed. With the
            # default we share the parent's process group, so SIGINT/SIGTERM
            # at the shell delivers to both. Graceful shutdown is still done
            # explicitly via `_kill_proc()` from the lifespan handler — and
            # an `atexit` hook below covers Python-level exits that bypass
            # the lifespan (e.g. an unhandled exception during startup).
        )
        # Poll /health until 200. First-load Metal init can take ~8s.
        # llama-server returns 503 *during* model load (expected), then
        # 200 once ready — so we only treat 200 as healthy and wait
        # through everything else. Connect/timeout errors are swallowed
        # by httpx's exception handling (we do narrow the catch here
        # though, so a misconfigured tls/dns failure surfaces fast
        # instead of burning the full 60s deadline silently).
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited prematurely (rc={self._proc.returncode})"
                )
            try:
                r = await self._client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    log.info(f"llama-server ready on :{PORT}")
                    self._loaded_model = canonical
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                # Process is up but socket isn't accepting yet (or first
                # /health is slow). Loop and try again.
                pass
            await asyncio.sleep(0.5)
        self._kill_proc()
        raise RuntimeError("llama-server failed to become healthy within 60s")

    async def _ensure_model(self, canonical: str) -> None:
        async with self._lock:
            if self._loaded_model == canonical and self._proc and self._proc.poll() is None:
                return
            self._kill_proc()
            await self._spawn(canonical)

    # ----- public surface (mirrors OllamaClient) ---------------------

    async def capabilities(self, model: str) -> frozenset[str]:
        # llama-server doesn't expose Ollama-style capabilities. Most
        # models loaded this way are vanilla GGUFs; report only what we
        # can promise. Tools/thinking detection would need extra probes.
        return frozenset({"completion"})

    async def chat(
        self,
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
        canonical = strip_prefix(model)
        await self._ensure_model(canonical)
        payload: dict = _build_payload(
            messages, temperature, max_tokens, tools, top_p, top_k, stream=False
        )
        r = await self._client.post("/v1/chat/completions", json=payload)
        if r.is_error:
            raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
        return _openai_to_ollama_chat(model, r.json())

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        think: bool = False,  # accepted for surface parity; ignored
        tools: list[dict] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        canonical = strip_prefix(model)
        await self._ensure_model(canonical)
        payload: dict = _build_payload(
            messages, temperature, max_tokens, tools, top_p, top_k, stream=True
        )
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            if response.is_error:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if not line:
                    continue
                out = _openai_stream_to_ollama_chunk(model, line)
                if out is not None:
                    yield out

    async def list_models(self) -> dict:
        """Catalogue: every Ollama-cached model is *potentially* loadable
        here, but exposing them all would clutter the dropdown. Surface
        only the currently-resident model (if any). Auto-fallback adds
        more entries dynamically as it discovers them."""
        if not self._loaded_model:
            return {"models": []}
        return {
            "models": [
                {
                    "name": f"{PREFIX}{self._loaded_model}",
                    "details": {"family": "llama-server"},
                    "size": 0,
                }
            ]
        }


client = LlamaServerClient()


def _kill_on_exit() -> None:
    """atexit safety net: if the FastAPI lifespan handler doesn't get to
    run (startup error, sys.exit during init), still tear down the child
    so we don't leave a zombie holding :11436 across restarts."""
    try:
        client._kill_proc()
    except Exception:
        pass


atexit.register(_kill_on_exit)
