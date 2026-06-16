"""Llama Guard 3 — a separate-classifier safety layer.

Sits *outside* the chat model so an adversarial user prompt cannot
disable it (the chat model never sees the guard's rules; the guard is
not asked to follow instructions, only to classify). Mirrors the
Bedrock Guardrails architecture at small-scale: pre-check on user
input, optional post-check on assistant output.

Memory model
------------
Disabled by default. When the runtime flag is off, no Ollama call is
ever made and the guard model never loads. When the flag is flipped to
off, the guard model is explicitly unloaded (`keep_alive=0`) so unified
memory is reclaimed immediately. While enabled, the model stays warm
for `CHATLM_GUARD_KEEP_ALIVE` (default 2m) between turns so back-to-back
classifications don't pay re-load latency.

Failure mode
------------
Fail-open: if the guard call errors (Ollama crashed, guard model not
pulled, HTTP timeout), we log a warning and let the message through.
This matches "don't break the user's workflow when our safety net
itself broke" — the alternative (fail-closed) would mean a flaky guard
silently blocks every chat. Override with CHATLM_GUARD_FAIL_CLOSED=1.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from config import settings

log = logging.getLogger("chatlm.guard")

# Llama Guard 3 1B is the right balance for local: <2 GB resident,
# >200 tok/s, covers 14 hazard categories + jailbreak detection.
GUARD_MODEL = os.environ.get("CHATLM_GUARD_MODEL", "llama-guard3:1b")

# Ollama keep_alive for the guard model. Format follows Ollama's:
# `2m`, `30s`, `0` (immediate unload), `-1` (forever). 2m is a good
# active-conversation default — back-to-back turns hit warm cache,
# idle releases memory automatically.
GUARD_KEEP_ALIVE = os.environ.get("CHATLM_GUARD_KEEP_ALIVE", "2m")

# Default-off so an unconfigured chatlm install doesn't surprise the
# user with extra resident memory. Operator opts in via env or via
# the runtime POST /guard/toggle.
_DEFAULT_ENABLED = os.environ.get("CHATLM_GUARD_ENABLED", "0").lower() in ("1", "true", "yes", "on")

# Fail-closed flag: when the guard call itself errors, treat the
# message as unsafe rather than letting it through.
_FAIL_CLOSED = os.environ.get("CHATLM_GUARD_FAIL_CLOSED", "0").lower() in ("1", "true", "yes", "on")


class GuardClient:
    """Thin wrapper over Ollama's /api/chat for the guard model.

    Two public methods drive the integration:
    - `classify_user(text)`   — pre-check before the chat model runs
    - `classify_assistant(text)` — post-check after the chat model replies

    Both return the same shape so a caller can branch on `safe`:

        {"safe": bool, "categories": [str], "raw": str, "skipped": bool}

    `skipped=True` means guard was disabled or input was empty — the
    classifier was never invoked.
    """

    __slots__ = ("_client", "_enabled", "_lock")

    def __init__(self) -> None:
        # Long timeout for cold-load on first turn; subsequent calls
        # finish in ~50 ms. We prefer waiting over false-block.
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
        self._enabled = _DEFAULT_ENABLED
        # Serialise concurrent toggles so an unload-in-flight can't be
        # raced by a re-enable that would silently leak.
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model(self) -> str:
        return GUARD_MODEL

    async def set_enabled(self, value: bool) -> None:
        """Toggle the guard at runtime. Switching to OFF unloads the
        model from Ollama's cache so unified memory is reclaimed."""
        async with self._lock:
            was_on = self._enabled
            self._enabled = bool(value)
            if was_on and not self._enabled:
                # Best-effort unload — log on failure but don't bubble
                # up; the toggle itself succeeded.
                try:
                    await self._unload_locked()
                    log.info(f"guard disabled — unloaded {GUARD_MODEL}")
                except Exception as err:
                    log.warning(f"guard disable: unload failed: {err}")
            elif not was_on and self._enabled:
                log.info(f"guard enabled — model={GUARD_MODEL} keep_alive={GUARD_KEEP_ALIVE}")

    async def classify_user(self, text: str) -> dict:
        """Classify a user message — call before sending to the chat model."""
        return await self._classify("user", text)

    async def classify_assistant(self, text: str) -> dict:
        """Classify an assistant response — call after the chat model replies."""
        return await self._classify("assistant", text)

    async def _classify(self, role: str, text: str) -> dict:
        if not self._enabled:
            return {"safe": True, "categories": [], "raw": "", "skipped": True}
        if not text or not text.strip():
            return {"safe": True, "categories": [], "raw": "", "skipped": True}
        payload = {
            "model": GUARD_MODEL,
            "messages": [{"role": role, "content": text}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
            "keep_alive": GUARD_KEEP_ALIVE,
            # Guard models don't have a thinking branch; explicit False
            # keeps Ollama from inserting <think> markers.
            "think": False,
        }
        try:
            r = await self._client.post("/api/chat", json=payload)
            r.raise_for_status()
            raw_text = (r.json().get("message", {}).get("content") or "").strip()
            return self._parse(raw_text)
        except Exception as err:
            log.warning(f"guard classify ({role}) failed: {err}")
            if _FAIL_CLOSED:
                return {
                    "safe": False,
                    "categories": ["GUARD_ERROR"],
                    "raw": str(err),
                    "skipped": False,
                }
            return {"safe": True, "categories": [], "raw": "", "skipped": True}

    @staticmethod
    def _parse(raw: str) -> dict:
        """Llama Guard 3 emits either `safe` or `unsafe\\nS1,S5`.
        Categories are strings like S1..S14 — see the model card for
        the mapping. We surface the raw token list and let the UI
        render human-readable labels if desired."""
        first = raw.split("\n", 1)[0].strip().lower()
        if first == "safe":
            return {"safe": True, "categories": [], "raw": raw, "skipped": False}
        cats: list[str] = []
        if "\n" in raw:
            cats = [c.strip() for c in raw.split("\n", 1)[1].split(",") if c.strip()]
        return {"safe": False, "categories": cats, "raw": raw, "skipped": False}

    async def _unload_locked(self) -> None:
        """Caller must hold `self._lock`. Posts an empty chat with
        keep_alive=0 — Ollama interprets that as 'unload now'."""
        try:
            await self._client.post(
                "/api/chat",
                json={"model": GUARD_MODEL, "messages": [], "keep_alive": 0},
                timeout=httpx.Timeout(10.0),
            )
        except Exception as err:
            log.debug(f"guard unload ignored: {err}")

    async def close(self) -> None:
        """Lifecycle hook called from FastAPI's lifespan teardown."""
        async with self._lock:
            if self._enabled:
                try:
                    await self._unload_locked()
                except Exception:
                    pass
            await self._client.aclose()


# Singleton — main.py imports `client` directly.
client = GuardClient()


# Stable category labels for UI rendering. Llama Guard 3's hazard
# taxonomy as of model card v1.0:
CATEGORY_LABELS: dict[str, str] = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


def label_categories(codes: list[str]) -> list[str]:
    """Human-readable labels for a list of `S1`..`S14` codes."""
    return [CATEGORY_LABELS.get(c, c) for c in codes]
