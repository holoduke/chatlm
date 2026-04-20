# CHATLM

A local, cyberpunk-themed multimodal playground on Apple Silicon. Chat with Gemma 4, stream your webcam into it, detect and segment objects by name, inpaint regions or restyle the whole frame, hear replies aloud and talk back — all running on your own Mac.

![CHATLM in video mode](docs/screenshot.png)

---

## What it does

Two modes, one page.

**Chat mode** — streaming text/vision chat against Gemma 4 (Ollama *or* Apple-native MLX), with:
- optional **deep-thinking** mode (Gemma's native channel)
- optional **tool use**: the model can request shell commands (`run_shell`) which you approve one-by-one
- image attach + clipboard paste
- push-to-talk **🎤 mic** for voice input (Whisper)
- **♪ SPEAK** on every reply (Kokoro TTS)

**Video mode** — live webcam pipeline with a whole toolbox:
- **LIVE** one-line descriptions of each frame (editable prompt)
- **SCAN / AUTO** object-list extraction → clickable chips → **TRACK**
- **TRACK** open-vocabulary detection with stable `#id`s, optional pixel masks
- **REPLACE** an object with a text prompt (Stable Diffusion inpainting)
- **GENERATE** re-style the whole frame (img2img)
- **POSE** · **DEPTH** · **CUTOUT** · **OCR** · **FACE** (landmarks + emotion)

Full telemetry pane shows per-pipeline latency, tokens/sec, memory, and total per-frame cost.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Browser (static/) — cyberpunk SPA, canvas overlays, mic/cam │
└────────────────────────────┬─────────────────────────────────┘
                             │  HTTP + streaming NDJSON
┌────────────────────────────┴─────────────────────────────────┐
│  FastAPI (main.py)                                           │
│   /chat /chat/stream /scan /detect /inpaint /img2img         │
│   /pose /depth /remove-bg /ocr /face                         │
│   /transcribe /speak /voices                                 │
│   /tools/exec /models/*                                      │
└────────┬─────────┬─────────┬─────────┬───────────┬───────────┘
         │         │         │         │           │
    ┌────┴───┐ ┌──┴────┐ ┌──┴────┐ ┌──┴────┐ ┌────┴────┐
    │ Ollama │ │  MLX  │ │ ultra-│ │  HF   │ │ kokoro/ │
    │ Gemma4 │ │ Gemma │ │lytics │ │ trans-│ │ whisper │
    │ llama  │ │ VLM   │ │YOLO/  │ │formers│ │ rembg   │
    │        │ │       │ │ SAM   │ │       │ │ easyocr │
    └────────┘ └───────┘ └───────┘ └───────┘ └─────────┘
```

Everything is lazy-loaded on first use and stays resident. Models are swappable at runtime via header dropdowns; selections persist in localStorage.

---

## Models used

### LLMs (chat, scan, live-describe)

| Model | Backend | Size | Notes |
|---|---|---|---|
| [`gemma4:e2b`](https://ollama.com/library/gemma4) | Ollama · llama.cpp + Metal | 6.7 GB (Q4_K_M) | Default. Vision-capable, ~52 tok/s on M3 Pro. |
| [`gemma4:e4b`](https://ollama.com/library/gemma4) | Ollama | 9 GB | Bigger Gemma 4 variant. |
| [`mlx-community/gemma-4-e2b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit) | Apple **MLX-VLM** | 3.6 GB | Text + vision. ~15–30 % faster than Ollama on the same model. |
| [`mlx-community/gemma-4-e4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) | MLX-VLM | 5.8 GB | |
| [`mlx-community/gemma-4-e2b-it-8bit`](https://huggingface.co/mlx-community/gemma-4-e2b-it-8bit) | MLX-VLM | 6.1 GB | Higher-precision MLX. |
| [`llama3.2:3b`](https://ollama.com/library/llama3.2) | Ollama | 2 GB | Good tool-caller (Gemma 4 is flaky with Ollama's tool API). |

Switch via the **EMMA** / **SCAN** dropdowns. `mlx:` prefix dispatches to MLX; anything else goes to Ollama.

### Open-vocabulary detectors (DETECT dropdown)

| Model | Source | Strength |
|---|---|---|
| [YOLO-World S/S-v2/M/L/X](https://docs.ultralytics.com/models/yolo-world/) | ultralytics | Fast (≈40 ms @ 480 px). Best when prompts are common nouns. |
| [Grounding DINO tiny/base](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | HF Transformers | Much stronger on bare nouns (`shirt`, `background`) and phrases, ~1.9 s per frame. |
| [OWLv2 base](https://huggingface.co/google/owlv2-base-patch16-ensemble) | HF Transformers | Google's open-vocab; similar tier to Grounding DINO. |

### Segmenters (SEGMENT dropdown)

| Model | Source | Notes |
|---|---|---|
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | ultralytics | Default. ~300 ms @ 480 px. |
| [FastSAM s / x](https://docs.ultralytics.com/models/fast-sam/) | ultralytics | Alternative architecture. |
| [SAM 2.0 tiny / small / base](https://github.com/facebookresearch/sam2) | ultralytics | Meta's v2 family. |
| [SAM 2.1 tiny / small / base / large](https://github.com/facebookresearch/sam2) | ultralytics | Newer iteration; better edges. |
| [SAM 1 base / large](https://github.com/facebookresearch/segment-anything) | ultralytics | High quality but slow. |

Pipeline: DETECT → box prompts → SEGMENT → polygon overlays + optional YOLO bbox shadow. Tracking uses ultralytics built-in **ByteTrack** for stable `#id` labels across frames.

### Stable Diffusion (REPLACE / GENERATE)

Inpainting presets (INPAINT dropdown):

| Preset | Base | Steps | Latency (M3 Pro, 640 px) |
|---|---|---|---|
| SD 1.5 + [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) · 4 steps | [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) | 4 | ~2.5 s |
| SD 1.5 + LCM · 2 steps (ultra) | same | 2 | ~1.3 s |
| [LCM-Dreamshaper v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | pre-baked LCM | 4 | ~2.5 s |
| [SDXL Inpaint + LCM-SDXL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | 4 | ~6 s · sharper |

Img2img uses LCM-Dreamshaper v7 (≈2.3 s warm).

Runs on MPS in **fp32** — MPS fp16 produced NaN-clamped black outputs in the VAE. Uses `enable_attention_slicing()` + VAE slicing to keep peak RAM manageable.

### Vision extras

| Feature | Model | Source |
|---|---|---|
| POSE | [`yolov8n-pose.pt`](https://docs.ultralytics.com/tasks/pose/) | ultralytics, 17 COCO keypoints |
| DEPTH | [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | HF Transformers, ~25 M params |
| CUTOUT | [rembg / U²-Net](https://github.com/danielgatis/rembg) | ONNX Runtime, ~180 MB |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | PyTorch, English by default |
| FACE mesh | [MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | Google, 478 landmarks, up to 5 faces |
| FACE emotion | [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) | HF ViT classifier |

### Audio

| Feature | Model | Source |
|---|---|---|
| STT | [`mlx-community/whisper-small.en-mlx`](https://huggingface.co/mlx-community/whisper-small.en-mlx) | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple-native, ~260 ms warm |
| TTS | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), ~380 ms warm |

Audio in: WebM/Opus from the browser → ffmpeg → 16 kHz mono WAV → Whisper.
Audio out: Kokoro 24 kHz mono → WAV → `<audio>` element.

### Tool use

Gemma (or any tool-capable Ollama model, e.g. llama3.2) can request shell commands through a single `run_shell(command)` tool. The UI shows an orange approval card with the editable command and **APPROVE / DENY** buttons — the command only runs after you click.

---

## Setup

```bash
git clone https://github.com/holoduke/chatlm.git
cd chatlm

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # MPS wheels on macOS-arm64
# NVIDIA: first install the CUDA torch wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install + start Ollama and pull a Gemma model:
brew install ollama
brew services start ollama
ollama pull gemma4:e2b
# Optional extras:
ollama pull llama3.2:3b        # reliable tool caller

# Recommended env-vars for Ollama on Apple Silicon:
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_KEEP_ALIVE 24h
launchctl setenv OLLAMA_NUM_PARALLEL 1
brew services restart ollama

# Run the app:
cp .env.example .env
.venv/bin/uvicorn main:app --port 8000
```

Open <http://localhost:8000>.

All weights (YOLO, SAM, Depth-Anything, RMBG, EasyOCR, MediaPipe, SD 1.5 inpainting, LCM LoRA, Dreamshaper, Whisper, Kokoro, FaceExpression, Grounding DINO, OWLv2…) are downloaded **lazily on first use**. Nothing is bundled in the repo.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | The SPA |
| `GET /health` | Ollama reachable? default model? |
| `GET /stats` | RSS, CPU, MEM, current model details |
| `GET /models` | All available Ollama + MLX + detector + segmenter + inpaint presets |
| `POST /chat` · `/chat/stream` | Streaming chat with tool + think + images |
| `POST /scan` | Gemma-vision SCAN → `{description, objects[]}` |
| `POST /detect` | Open-vocab detection + optional segmentation, ByteTrack IDs |
| `POST /inpaint` | SD inpaint with polygon/box mask |
| `POST /img2img` | Full-frame SD transform |
| `POST /pose` | 17-keypoint COCO skeleton |
| `POST /depth` | Coloured depth map |
| `POST /remove-bg` | U²-Net cutout (RGBA or mask) |
| `POST /ocr` | Text items with polygons |
| `POST /face` | FaceMesh landmarks + optional emotion |
| `POST /transcribe` | Whisper (any ffmpeg-readable audio) |
| `POST /speak` | Kokoro TTS |
| `GET /voices` | Kokoro voices |
| `POST /tools/exec` | Run a shell command (frontend MUST gate this with user approval) |
| `POST /models/emma` · `/models/scan` · `/models/detector` · `/models/segmenter` · `/models/inpaint` | Live-swap the active model |

---

## Performance notes (M3 Pro, 36 GB)

| Pipeline | Warm latency |
|---|---|
| Gemma 4 E2B text (Ollama) | 52 tok/s, ~240 ms for short replies |
| Gemma 4 E2B text (MLX) | ~15 % faster, ~200 ms for short replies |
| YOLO-World boxes (480 px) | ~40 ms (~24 fps track loop) |
| MobileSAM (480 px) | ~300 ms |
| Grounding DINO tiny | ~1.9 s |
| SD 1.5 inpaint (LCM 4 steps, 640 px) | ~2.5 s |
| SD img2img (LCM 4 steps) | ~2.3 s |
| Depth Anything V2 Small | ~1.2 s |
| rembg U²-Net | ~380 ms |
| Whisper small.en MLX | ~260 ms for ~2 s clips |
| Kokoro TTS | ~380 ms for one sentence |

---

## Stack

- **Runtime**: Python 3.13, FastAPI, httpx, pydantic
- **LLM**: [Ollama](https://ollama.com), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- **Vision**: [ultralytics](https://github.com/ultralytics/ultralytics) (YOLO-World, YOLO-pose, SAM/SAM2, FastSAM, ByteTrack), [transformers](https://github.com/huggingface/transformers) (Grounding DINO, OWLv2, Depth Anything V2, face-expression), [mediapipe](https://github.com/google-ai-edge/mediapipe) (face mesh), [easyocr](https://github.com/JaidedAI/EasyOCR), [rembg](https://github.com/danielgatis/rembg)
- **Generative**: [diffusers](https://github.com/huggingface/diffusers), [peft](https://github.com/huggingface/peft), SD 1.5 inpainting, SDXL inpainting, LCM-LoRA, LCM-Dreamshaper
- **Audio**: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), ffmpeg

## License

MIT. Model weights retain their own licenses (Gemma terms, Apache 2.0, CC-BY-NC, etc.) — check each model's card before commercial use.
