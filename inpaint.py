"""Fast single-frame inpainting on Apple Silicon.

Stack: SD 1.5 inpainting + LCM LoRA + MPS + fp16 + attention slicing.
Typical latency on M3 Pro (36GB): 3-6s per 512px frame at 4 steps.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("chatlm.inpaint")

_pipe: Any = None
_i2i_pipe: Any = None
_pipe_name: str = ""
_lock = threading.Lock()
_device: str = "cpu"

DEFAULT_STRENGTH = 0.999

# Each preset describes one swappable inpainting stack.
INPAINT_PRESETS: dict[str, dict] = {
    "sd15-lcm-fast": {
        "label": "SD 1.5 + LCM · 4 steps · ~2.5s",
        "model_id": "runwayml/stable-diffusion-inpainting",
        "lora_id": "latent-consistency/lcm-lora-sdv1-5",
        "pipeline": "sd",
        "steps": 4,
        "guidance": 4.0,
    },
    "sd15-lcm-ultra": {
        "label": "SD 1.5 + LCM · 2 steps · ~1.3s",
        "model_id": "runwayml/stable-diffusion-inpainting",
        "lora_id": "latent-consistency/lcm-lora-sdv1-5",
        "pipeline": "sd",
        "steps": 2,
        "guidance": 2.0,
    },
    "lcm-dreamshaper": {
        "label": "LCM-Dreamshaper v7 · 4 steps · ~2.5s",
        "model_id": "SimianLuo/LCM_Dreamshaper_v7",
        "lora_id": None,  # already LCM-baked
        "pipeline": "sd",
        "steps": 4,
        "guidance": 1.0,
    },
    "sdxl-lcm": {
        "label": "SDXL Inpaint + LCM · 4 steps · ~6s",
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "lora_id": "latent-consistency/lcm-lora-sdxl",
        "pipeline": "sdxl",
        "steps": 4,
        "guidance": 2.0,
    },
}
_current_preset: str = "sd15-lcm-fast"


def _pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_pipeline(preset_name: str):
    import torch
    from diffusers import AutoPipelineForInpainting, LCMScheduler

    global _device
    _device = _pick_device()
    preset = INPAINT_PRESETS[preset_name]
    # fp16 on MPS produces black (VAE NaN); fp32 e2e is reliable at ~2.5s.
    dtype = torch.float16 if _device == "cuda" else torch.float32
    log.info(f"loading {preset['model_id']} on {_device} ({dtype}, preset={preset_name})...")
    t0 = time.perf_counter()
    kwargs: dict = dict(
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForInpainting.from_pretrained(preset["model_id"], **kwargs)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if preset.get("lora_id"):
        pipe.load_lora_weights(preset["lora_id"])
        pipe.fuse_lora()
    pipe = pipe.to(_device)
    pipe.set_progress_bar_config(disable=True)
    if _device in {"mps", "cuda"}:
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    log.info(f"inpaint pipeline ready in {time.perf_counter() - t0:.1f}s (device={_device})")
    return pipe


def _ensure_pipeline():
    global _pipe, _pipe_name, _i2i_pipe
    if _pipe is None or _pipe_name != _current_preset:
        if _pipe is not None:
            log.info(f"swapping inpaint pipeline: {_pipe_name} -> {_current_preset}")
            _pipe = None
            _i2i_pipe = None
        _pipe = _build_pipeline(_current_preset)
        _pipe_name = _current_preset
    return _pipe


I2I_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"


def _ensure_i2i_pipeline():
    """Image-to-image pipeline. Inpainting UNets have 9 input channels, so
    they can't be reused for img2img (which needs 4). We load a separate,
    pre-baked LCM model for fast full-frame img2img."""
    global _i2i_pipe, _device
    if _i2i_pipe is not None:
        return _i2i_pipe
    import torch
    from diffusers import AutoPipelineForImage2Image, LCMScheduler

    _device = _pick_device()
    dtype = torch.float16 if _device == "cuda" else torch.float32
    log.info(f"loading img2img model {I2I_MODEL_ID} on {_device} ({dtype})...")
    t0 = time.perf_counter()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        I2I_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(_device)
    pipe.set_progress_bar_config(disable=True)
    if _device in {"mps", "cuda"}:
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    _i2i_pipe = pipe
    log.info(f"img2img pipeline ready in {time.perf_counter() - t0:.1f}s")
    return _i2i_pipe


def set_preset(name: str) -> None:
    if name not in INPAINT_PRESETS:
        raise ValueError(f"unknown inpaint preset: {name}")
    global _current_preset
    with _lock:
        _current_preset = name
        # Don't load here; next /inpaint call will lazy-load so dropdown
        # switching is instant and the download cost is deferred until use.
        global _pipe
        if _pipe is not None and _pipe_name != name:
            _pipe = None


def current_preset() -> str:
    return _current_preset


def default_params_for_current() -> dict:
    p = INPAINT_PRESETS[_current_preset]
    return {"steps": p["steps"], "guidance": p["guidance"]}


def _round_to(value: int, multiple: int = 8) -> int:
    return max(multiple, (value // multiple) * multiple)


def _prepare_mask(mask_image: Image.Image, feather: int) -> Image.Image:
    """Slight blur+dilate softens mask edges so inpainting blends cleanly."""
    mask = mask_image.convert("L")
    if feather > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=max(3, feather | 1)))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))
    return mask


def inpaint(
    image_b64: str,
    mask_b64: str,
    prompt: str,
    negative_prompt: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    max_size: int = 640,
    feather: int = 9,
) -> dict:
    pipe = _ensure_pipeline()
    defaults = default_params_for_current()
    if steps is None:
        steps = defaults["steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")

    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    nw, nh = _round_to(int(w * scale)), _round_to(int(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
        mask = mask.resize((nw, nh), Image.LANCZOS)

    prepared_mask = _prepare_mask(mask, feather)

    timings: dict[str, float] = {"prep_ms": 0.0, "inference_ms": 0.0}
    t_prep = time.perf_counter()
    timings["prep_ms"] = round((time.perf_counter() - t_prep) * 1000, 1)

    with _lock:
        ti = time.perf_counter()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=img,
            mask_image=prepared_mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=DEFAULT_STRENGTH,
            width=nw,
            height=nh,
        ).images[0]
        timings["inference_ms"] = round((time.perf_counter() - ti) * 1000, 1)

    out_buf = io.BytesIO()
    result.save(out_buf, format="JPEG", quality=88)
    out_b64 = base64.b64encode(out_buf.getvalue()).decode("ascii")

    return {
        "image": out_b64,
        "width": nw,
        "height": nh,
        "steps": steps,
        "guidance": guidance,
        "timings_ms": timings,
    }


def img2img(
    image_b64: str,
    prompt: str,
    negative_prompt: str | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    strength: float = 0.7,
    max_size: int = 640,
) -> dict:
    """Generate a new image from a source frame + text prompt.
    `strength` (0..1) controls how much to transform the source.
    """
    pipe = _ensure_i2i_pipeline()
    defaults = default_params_for_current()
    if steps is None:
        steps = defaults["steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    nw, nh = _round_to(int(w * scale)), _round_to(int(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)

    # SD img2img reduces effective steps by strength; when LCM-baked the model
    # wants at least a couple of actual steps.  Boost steps to compensate.
    effective_steps = max(steps, int(round(steps / max(strength, 0.1))))
    timings = {"inference_ms": 0.0}

    with _lock:
        ti = time.perf_counter()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=img,
            num_inference_steps=effective_steps,
            guidance_scale=guidance,
            strength=strength,
        ).images[0]
        timings["inference_ms"] = round((time.perf_counter() - ti) * 1000, 1)

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=88)
    return {
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": nw,
        "height": nh,
        "steps": effective_steps,
        "guidance": guidance,
        "strength": strength,
        "timings_ms": timings,
    }


def mask_from_polygons(
    polygons: list[list[list[int]]],
    width: int,
    height: int,
) -> str:
    """Render a list of polygon coordinates into a binary mask PNG (base64)."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([(x, y) for x, y in poly], fill=255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_from_boxes(
    boxes: list[list[int]],
    width: int,
    height: int,
) -> str:
    from PIL import Image, ImageDraw

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for b in boxes:
        if len(b) >= 4:
            draw.rectangle([b[0], b[1], b[2], b[3]], fill=255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
