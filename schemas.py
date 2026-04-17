from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    model: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    stream: bool = False
    think: bool = False
    tools: list[dict[str, Any]] | None = None


class InpaintRequest(BaseModel):
    image: str = Field(..., description="Base64 JPEG/PNG of the original frame")
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: str | None = Field(None, max_length=300)
    # Caller supplies either a pre-rendered mask PNG, or the detect output
    # (targets + polygons) which we rasterise server-side. At least one required.
    mask: str | None = Field(None, description="Base64 PNG binary mask (white=replace)")
    polygons: list[list[list[int]]] | None = Field(None, description="Per-region polygon coords in image pixel space")
    boxes: list[list[int]] | None = Field(None, description="Per-region xyxy boxes (fallback when no polygons)")
    width: int | None = Field(None, ge=64, le=2048)
    height: int | None = Field(None, ge=64, le=2048)
    steps: int = Field(4, ge=1, le=30)
    guidance: float = Field(4.0, ge=0.0, le=15.0)
    max_size: int = Field(640, ge=256, le=1024)
    feather: int = Field(9, ge=0, le=40)


class InpaintResponse(BaseModel):
    image: str
    width: int
    height: int
    steps: int
    guidance: float
    latency_ms: int
    timings_ms: dict[str, float]


class Img2ImgRequest(BaseModel):
    image: str = Field(..., description="Base64 JPEG/PNG of the source frame")
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: str | None = Field(None, max_length=300)
    strength: float = Field(0.7, ge=0.1, le=1.0, description="How much to deviate from the source")
    steps: int | None = Field(None, ge=1, le=30)
    guidance: float | None = Field(None, ge=0.0, le=15.0)
    max_size: int = Field(640, ge=256, le=1024)


class Img2ImgResponse(BaseModel):
    image: str
    width: int
    height: int
    steps: int
    guidance: float
    strength: float
    latency_ms: int
    timings_ms: dict[str, float]


class ToolExecRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4000)
    cwd: str | None = Field(None, max_length=500)
    timeout: int = Field(30, ge=1, le=300)


class ToolExecResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool


class ChatResponse(BaseModel):
    model: str
    message: Message
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: int | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)


class GenerateResponse(BaseModel):
    model: str
    response: str


class ScanRequest(BaseModel):
    image: str = Field(..., description="Base64 JPEG/PNG (no data-url prefix)")
    max_objects: int = Field(10, ge=1, le=30)
    prompt: str | None = Field(None, max_length=500, description="Optional custom prompt. If set, JSON schema still enforced.")


class ScanResponse(BaseModel):
    description: str
    objects: list[str]
    latency_ms: int


class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64 JPEG/PNG (no data-url prefix)")
    prompt: str = Field(..., min_length=1)
    conf: float = Field(0.08, ge=0.01, le=0.9)
    masks: bool = True
    imgsz: int = Field(480, ge=256, le=1024)


class SetModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class DetectResponse(BaseModel):
    targets: list[str]
    polygons: list[list[list[int]]]
    boxes: list[list[int]]
    labels: list[str]
    confidences: list[float]
    w: int
    h: int
    latency_ms: int
    timings_ms: dict[str, float]
