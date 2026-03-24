import base64
import json
import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8091")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8092"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "./voices"))
VOICES_META = VOICES_DIR / "voices.json"

http_client: httpx.AsyncClient


def load_voices() -> dict:
    if VOICES_META.exists():
        return json.loads(VOICES_META.read_text())
    return {}


def save_voices(voices: dict):
    VOICES_META.write_text(json.dumps(voices, indent=2, ensure_ascii=False))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    if not VOICES_META.exists():
        save_voices({})
    http_client = httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=300)
    yield
    await http_client.aclose()


app = FastAPI(
    title="Qwen3-TTS Voice Proxy",
    description="Proxy for Qwen3-TTS with voice management. "
    "Register voices once and reuse them by name.",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────


class ResponseFormat(str, Enum):
    wav = "wav"
    pcm = "pcm"
    flac = "flac"
    mp3 = "mp3"
    aac = "aac"
    opus = "opus"


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        "default",
        description="Name of a registered voice, or a built-in voice name",
    )
    model: str = Field(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", description="Model identifier"
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.wav, description="Output audio format"
    )
    language: Optional[str] = Field(
        None,
        description="Target language (e.g. 'Portuguese', 'English'). "
        "Default: auto-detect",
    )
    speed: float = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed. WARNING: values != 1.0 may degrade quality",
    )
    temperature: float = Field(
        0.9,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more consistent, "
        "higher = more varied. Recommended: 0.3-0.5 for voice cloning",
    )
    top_k: int = Field(
        50,
        ge=1,
        description="Top-K sampling. Lower = more deterministic. "
        "Recommended: 10-20 for voice cloning",
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Top-P (nucleus) sampling",
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible output"
    )
    repetition_penalty: Optional[float] = Field(
        None, ge=0.0, description="Repetition penalty (default: 1.05)"
    )
    max_tokens: Optional[int] = Field(
        None, description="Max tokens to generate"
    )
    instructions: Optional[str] = Field(
        None, description="Style/emotion directives (e.g. 'Speak calmly')"
    )
    stream: bool = Field(False, description="Enable chunked streaming response")

    model_config = {"extra": "allow"}


class VoiceOut(BaseModel):
    name: str
    ref_text: str


class VoiceListOut(BaseModel):
    voices: list[VoiceOut]


# ── Voice management ──────────────────────────────────────────────


@app.get("/v1/audio/voices", response_model=VoiceListOut)
async def list_voices():
    """List all registered voices."""
    voices = load_voices()
    return {
        "voices": [
            {"name": v["name"], "ref_text": v.get("ref_text", "")}
            for v in voices.values()
        ]
    }


@app.post("/v1/audio/voices", summary="Register a voice")
async def upload_voice(
    file: UploadFile = File(..., description="Reference audio file (WAV, max 10MB)"),
    name: str = Form(..., description="Unique voice identifier"),
    ref_text: str = Form(
        "",
        description="Transcript of the reference audio. "
        "Improves cloning quality significantly (ICL mode). "
        "If empty, x-vector-only mode is used.",
    ),
):
    """Upload a reference audio to register a new voice.

    The audio is stored as base64 and reused automatically when you
    pass `voice: "<name>"` in the speech endpoint.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 10MB)"}, status_code=413)

    audio_b64 = base64.b64encode(audio_bytes).decode()
    ref_audio_uri = f"data:audio/wav;base64,{audio_b64}"

    voices = load_voices()
    voices[name] = {"name": name, "ref_text": ref_text, "ref_audio": ref_audio_uri}
    save_voices(voices)

    return {"name": name, "ref_text": ref_text, "created": True}


@app.delete("/v1/audio/voices/{name}", summary="Delete a voice")
async def delete_voice(name: str):
    """Remove a registered voice."""
    voices = load_voices()
    if name not in voices:
        return JSONResponse({"error": "Voice not found"}, status_code=404)

    del voices[name]
    save_voices(voices)
    return {"name": name, "deleted": True}


# ── TTS proxy with voice injection ───────────────────────────────


@app.post(
    "/v1/audio/speech",
    summary="Generate speech",
    response_class=StreamingResponse,
)
async def tts_speech(request: Request):
    """Generate speech from text.

    If `voice` matches a registered voice name, the proxy automatically
    injects the stored reference audio and transcript.

    - **With ref_text**: uses ICL mode (better quality)
    - **Without ref_text**: uses x-vector-only mode

    **Example request:**
    ```json
    {
      "input": "Olá! Seja bem-vindo ao atendimento do Banco do Povo. Como posso te ajudar hoje? Gostaria de um empréstimo pra pagar o agiota?",
      "response_format": "wav",
      "instructions": "Use um sotaque neutro do brasil",
      "voice": "ref_nova",
      "speed": 1,
      "stream_format": "audio",
      "stream": false,
      "task_type": "Base",
      "language": "Portuguese",
      "temperature": 0.3,
      "top_k": 10
    }
    ```
    """
    content_type = request.headers.get("content-type", "")

    if "json" in content_type:
        body = await request.json()
    else:
        # Handle form-data: read fields into a dict
        form = await request.form()
        body = {}
        for key in form:
            val = form[key]
            if isinstance(val, str):
                if val.lower() == "true":
                    body[key] = True
                elif val.lower() == "false":
                    body[key] = False
                else:
                    try:
                        body[key] = float(val) if "." in val else int(val)
                    except ValueError:
                        body[key] = val
            else:
                body[key] = val

    voice_name = body.get("voice", "")

    voices = load_voices()
    if voice_name in voices:
        voice = voices[voice_name]
        body["ref_audio"] = voice["ref_audio"]
        if voice["ref_text"]:
            body["ref_text"] = voice["ref_text"]
        else:
            body.setdefault("x_vector_only_mode", True)
        body.setdefault("task_type", "Base")

    upstream = await http_client.post("/v1/audio/speech", json=body)

    return StreamingResponse(
        upstream.iter_bytes(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/octet-stream"),
    )


# ── Catch-all proxy ──────────────────────────────────────────────


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    upstream = await http_client.request(
        method=request.method,
        url=f"/{path}",
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        params=request.query_params,
        content=await request.body(),
    )
    return StreamingResponse(
        upstream.iter_bytes(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
