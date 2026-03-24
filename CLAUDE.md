# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Docker-based deployment of Qwen3-TTS voice synthesis with a FastAPI proxy for voice management. Two containers: a vLLM-Omni TTS backend (GPU) and a lightweight voice proxy.

## Architecture

```
Client :8092 → [tts-proxy (FastAPI)] :8091 → [qwen3-tts (vLLM-Omni)]
                     │                              │
               voices.json (base64)          Stage 0: Talker (text→tokens)
                                                    │ SharedMemory
                                             Stage 1: Code2Wav (tokens→audio)
```

- **tts-proxy** (`proxy.py`, `Dockerfile.proxy`): Manages voice registration (stores ref audio as base64), injects `ref_audio`/`ref_text` into requests when a registered voice name is used. Port 8092.
- **qwen3-tts** (`Dockerfile`, `entrypoint.sh`): vLLM-Omni server with 2-stage pipeline. Port 8091 (internal only, not exposed to host).
- **stage_configs/**: YAML config for the 2-stage pipeline + `generate_config.py` that generates runtime config from env vars.

## Common Commands

```bash
# Start everything
docker compose up -d

# Rebuild proxy only (doesn't restart TTS / reload model)
docker compose up -d --build tts-proxy --no-deps

# Rebuild everything (TTS model reload takes ~2 min)
docker compose up -d --build

# Check if TTS model finished loading (look for "Application startup complete")
docker compose logs --tail 10 qwen3-tts

# Check proxy logs
docker compose logs --tail 20 tts-proxy

# Expose via ngrok (point to proxy, not TTS directly)
ngrok http 8092
```

## Key Design Decisions

- **Port 8091 is internal only** (`expose`, not `ports`). All external traffic goes through the proxy on 8092.
- **Voice audio stored as base64 in `voices.json`**, not as .wav files. Avoids re-encoding on every TTS request.
- **Two voice cloning modes**: ICL (with `ref_text` — better quality) and x-vector-only (without `ref_text` — the proxy sets `x_vector_only_mode: true` automatically).
- **Stage 0 sampling** (temperature, top_k) heavily affects voice consistency. Lower values (temp 0.3, top_k 10) produce more consistent cloning.
- **`speed` != 1.0** degrades audio quality noticeably — avoid it.

## Environment Configuration

All config is via env vars (see `.env.example`). Key ones:
- `MODEL`: Qwen3-TTS model variant (Base for cloning, Instruct for style control)
- `GPU_MEMORY_UTILIZATION_STAGE0/1`: VRAM allocation per stage (default 0.3 each)
- `TEMPERATURE`, `TOP_K`: Sampling defaults for Stage 0 (talker)
- `PROXY_PORT`: Proxy port (default 8092)

## Proxy API Endpoints

- `POST /v1/audio/voices` — Register voice (multipart: `file`, `name`, `ref_text`)
- `GET /v1/audio/voices` — List registered voices
- `DELETE /v1/audio/voices/{name}` — Remove a voice
- `POST /v1/audio/speech` — Generate TTS (pass `voice: "<name>"` to use registered voice)
- `GET /docs` — Swagger UI with full documentation
