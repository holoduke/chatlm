"""Catalogue of MLX-VLM models served by `mlx_client.py`.

Lives in its own module so the runtime (loading, locking, streaming)
stays focused. To add a new MLX model, add a row here — runtime code
discovers it transparently via `MLX_MODELS`.

Per-preset fields:
- `repo`             Hugging Face repo id passed to mlx_vlm.load
- `label`            human-readable string for the dropdown
- `parameter_size`   reported as-is to /stats (display only)
- `quantization`     short tag: q4 / q8 / q4-ud
- `size_gb`          approximate on-disk size, also used by /models
                     ordering and the model-fits-in-RAM hint

Naming: external key is `mlx:<short-name>`; the `mlx:` prefix is what
`mlx_client.is_mlx_name()` uses to route a request to the MLX backend.
"""
from __future__ import annotations

MLX_MODELS: dict[str, dict] = {
    # ── Gemma 4 ────────────────────────────────────────────────────────
    "mlx:gemma-4-e2b-it-4bit": {
        "repo": "mlx-community/gemma-4-e2b-it-4bit",
        "label": "MLX · Gemma 4 E2B-IT 4bit",
        "parameter_size": "5.1B",
        "quantization": "q4",
        "size_gb": 3.6,
    },
    "mlx:gemma-4-e4b-it-4bit": {
        "repo": "mlx-community/gemma-4-e4b-it-4bit",
        "label": "MLX · Gemma 4 E4B-IT 4bit",
        "parameter_size": "8.0B",
        "quantization": "q4",
        "size_gb": 5.8,
    },
    "mlx:gemma-4-e2b-it-8bit": {
        "repo": "mlx-community/gemma-4-e2b-it-8bit",
        "label": "MLX · Gemma 4 E2B-IT 8bit",
        "parameter_size": "5.1B",
        "quantization": "q8",
        "size_gb": 6.1,
    },
    "mlx:gemma-4-e4b-it-8bit": {
        "repo": "mlx-community/gemma-4-E4B-it-8bit",
        "label": "MLX · Gemma 4 E4B-IT 8bit",
        "parameter_size": "8.0B",
        "quantization": "q8",
        "size_gb": 8.4,
    },
    "mlx:gemma-4-e4b-it-ud-4bit": {
        "repo": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        "label": "MLX · Gemma 4 E4B-IT UD 4bit (Unsloth)",
        "parameter_size": "8.0B",
        "quantization": "q4-ud",
        "size_gb": 6.2,
    },
    "mlx:gemma-4-26b-a4b-it-4bit": {
        "repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "label": "MLX · Gemma 4 26B-A4B-IT 4bit (MoE)",
        "parameter_size": "26B MoE (~3.8B active)",
        "quantization": "q4",
        "size_gb": 15.0,
    },
    "mlx:gemma-4-26b-a4b-it-ud-4bit": {
        "repo": "unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit",
        "label": "MLX · Gemma 4 26B-A4B-IT UD 4bit (Unsloth MoE)",
        "parameter_size": "26B MoE (~3.8B active)",
        "quantization": "q4-ud",
        "size_gb": 15.0,
    },

    # ── Qwen 3.5 ──────────────────────────────────────────────────────
    # Too new for llama.cpp / Ollama (custom `qwen35` arch token); MLX has
    # its own arch dispatch and supports it via mlx-community.
    "mlx:qwen3.5-4b-4bit": {
        "repo": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "label": "MLX · Qwen 3.5 4B 4bit",
        "parameter_size": "4.2B",
        "quantization": "q4",
        "size_gb": 2.5,
    },
    "mlx:qwen3.5-9b-4bit": {
        "repo": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "label": "MLX · Qwen 3.5 9B 4bit",
        "parameter_size": "8.95B",
        "quantization": "q4",
        "size_gb": 5.5,
    },
    "mlx:qwen3.5-9b-8bit": {
        "repo": "mlx-community/Qwen3.5-9B-MLX-8bit",
        "label": "MLX · Qwen 3.5 9B 8bit",
        "parameter_size": "8.95B",
        "quantization": "q8",
        "size_gb": 9.5,
    },
    "mlx:qwen3.5-27b-4bit": {
        "repo": "mlx-community/Qwen3.5-27B-4bit",
        "label": "MLX · Qwen 3.5 27B 4bit",
        "parameter_size": "27B",
        "quantization": "q4",
        "size_gb": 15.5,
    },
}
