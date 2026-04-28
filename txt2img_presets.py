"""Catalogue of text-to-image diffusion presets.

Lives in its own module so the runtime in `txt2img.py` (loading,
inference, eviction) stays focused. To add a new model, add a row here
and — only if it introduces a new pipeline kind — extend the
`_build_pipeline` dispatch in `txt2img.py`.

Per-preset fields:
- `label`         human-readable string for the dropdown
- `model_id`      Hugging Face repo id (or repo + override; see notes)
- `pipeline`      kind tag dispatched in `_build_pipeline`:
                    "sdxl"            — StableDiffusionXLPipeline
                    "sdxl-lightning"  — SDXL base + Lightning UNet swap
                    "flux"            — FluxPipeline (FLUX.1)
                    "flux2"           — Flux2Pipeline (FLUX.2)
                    "sd3"             — StableDiffusion3Pipeline
- `steps`         default num_inference_steps
- `guidance`      default guidance_scale (CFG)
- `width/height`  default native generation size
- `weight_gb`     measured peak resident memory at the chosen dtype.
                  Used by the memory guardrail to refuse loads that
                  would push macOS into hard swap (kernel-panic territory
                  on the 36 GB M3 Pro). When in doubt, overestimate.
- `notes` (opt)   short freeform comment that surfaces in the UI tooltip.
"""
from __future__ import annotations

# Reserve memory for the OS, Chrome, FastAPI, etc. on top of model weights.
# Each preset's `weight_gb` is the peak resident footprint we measured, so
# this headroom only needs to cover the rest of the system, not generation
# spikes. Tuned down from 6 → 4 after measuring real usage.
SAFETY_HEADROOM_GB = 4.0


TXT2IMG_PRESETS: dict[str, dict] = {
    # ── Stable Diffusion XL ────────────────────────────────────────────
    "sdxl-base": {
        "label": "SDXL base 1.0 · 25 steps · ~20s",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "sdxl",
        "steps": 25,
        "guidance": 7.0,
        "width": 1024,
        "height": 1024,
        # SDXL needs fp32 on MPS — fp16 produces NaN black output even
        # with the fp16-fix VAE.
        "weight_gb": 14.0,
    },
    "sdxl-turbo": {
        "label": "SDXL Turbo · 4 steps · ~4s",
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline": "sdxl",
        "steps": 4,
        "guidance": 0.0,
        "width": 512,
        "height": 512,
        "weight_gb": 8.0,  # measured peak ~7.5 GB at 512x512 fp32 on M3 Pro
    },
    "sdxl-lightning-4step": {
        "label": "SDXL Lightning · 4 steps · ~5s (UNet swap on SDXL base)",
        # Loaded as: SDXL base 1.0 + Lightning UNet weights from
        # ByteDance/SDXL-Lightning · sdxl_lightning_4step_unet.safetensors.
        # `model_id` carries the base; the UNet override is hard-wired in
        # `_build_pipeline` for the "sdxl-lightning" kind to keep this
        # catalogue declarative.
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "sdxl-lightning",
        "lightning_repo": "ByteDance/SDXL-Lightning",
        "lightning_unet": "sdxl_lightning_4step_unet.safetensors",
        "steps": 4,
        "guidance": 0.0,
        "width": 1024,
        "height": 1024,
        "weight_gb": 14.0,  # full SDXL base footprint
    },

    # ── FLUX (Black Forest Labs) ──────────────────────────────────────
    "flux-schnell": {
        "label": "FLUX.1-schnell · 4 steps · ~20s (gated, accept license on HF)",
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "flux",
        "steps": 4,
        "guidance": 0.0,
        "width": 1024,
        "height": 1024,
        "weight_gb": 24.0,  # fp16
    },
    "flux2-klein": {
        "label": "FLUX.2 [klein] · 28 steps · ~30s (gated, distilled small)",
        "model_id": "black-forest-labs/FLUX.2-klein",
        "pipeline": "flux2",
        "steps": 28,
        "guidance": 3.5,
        "width": 1024,
        "height": 1024,
        "weight_gb": 16.0,  # bf16, distilled
    },
    "flux2-dev": {
        "label": "FLUX.2 [dev] · 28 steps · ~60s (gated, non-commercial)",
        "model_id": "black-forest-labs/FLUX.2-dev",
        "pipeline": "flux2",
        "steps": 28,
        "guidance": 3.5,
        "width": 1024,
        "height": 1024,
        # 24 GB bf16 — exceeds 36 GB unified after headroom only on edge;
        # `_ensure_pipeline` will refuse load if not enough free memory.
        "weight_gb": 24.0,
    },

    # ── Stable Diffusion 3.5 ──────────────────────────────────────────
    "sd35-medium": {
        "label": "SD 3.5 Medium · 28 steps · ~20s (gated, accept license on HF)",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline": "sd3",
        "steps": 28,
        "guidance": 4.5,
        "width": 1024,
        "height": 1024,
        "weight_gb": 12.0,  # fp16
    },
    "sd35-large": {
        "label": "SD 3.5 Large · 28 steps · ~40s (gated, free <$1M revenue)",
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": "sd3",
        "steps": 28,
        "guidance": 4.5,
        "width": 1024,
        "height": 1024,
        # 8B MMDiT in bf16. Triple text encoder (CLIP-L, CLIP-G, T5-XXL);
        # T5 alone is ~9 GB. Budget conservatively.
        "weight_gb": 18.0,
    },
}
