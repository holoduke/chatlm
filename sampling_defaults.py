"""Per-model recommended sampling values surfaced by model authors.

Used as a fallback when the request doesn't supply a value. User-set
fields always win. Keep entries minimal and authoritative — only add
what the model card explicitly recommends.

Matching is substring-based against the model name so a single entry
(e.g. "gemma-4") covers every quant/variant/finetune of that family
(`huihui_ai/qwen3-abliterated:8b`, `mlx:gemma-4-e4b-it-4bit`,
`hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M`).
First match wins — order substrings from most-specific to most-generic.
"""
from __future__ import annotations


# (substring, defaults). Lower-case substring match against model name.
_RECOMMENDATIONS: list[tuple[str, dict]] = [
    # Google Gemma 4 authors recommend t=1.0, top_p=0.95, top_k=64
    # (https://huggingface.co/google/gemma-4-e4b-it / model cards of
    # community quants reproduce this).
    ("gemma-4", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    ("gemma4", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    # Qwen 3.6 (thinking mode is the default; README:
    # https://huggingface.co/Qwen/Qwen3.6-27B § Best Practices).
    # Listed before the broader `qwen3` so it wins on substring match.
    ("qwen3.6", {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
    # Qwen 3.x reasoning models recommend top_p=0.8, top_k=20.
    ("qwen3", {"top_p": 0.8, "top_k": 20}),
]


def lookup(model: str) -> dict:
    """Return the merged recommendation dict for `model`. Empty dict
    means no recommendations on file (caller should leave defaults
    untouched)."""
    needle = (model or "").lower()
    for sub, rec in _RECOMMENDATIONS:
        if sub in needle:
            return dict(rec)
    return {}


def fill_defaults(
    model: str,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> tuple[float | None, float | None, int | None]:
    """Fill in any unset (None) sampling fields from the model's
    recommendations. Returns a possibly-replaced (temperature, top_p,
    top_k) triple. Use a sentinel of None for "user didn't set it" so
    that explicit zeros from the UI still pass through unchanged."""
    rec = lookup(model)
    if temperature is None and "temperature" in rec:
        temperature = rec["temperature"]
    if top_p is None and "top_p" in rec:
        top_p = rec["top_p"]
    if top_k is None and "top_k" in rec:
        top_k = rec["top_k"]
    return temperature, top_p, top_k
