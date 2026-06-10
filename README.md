# OmniVoice-Triton-API

[中文文档](README_zh.md)

OpenAI-compatible TTS API powered by [OmniVoice](https://github.com/k2-fsa/OmniVoice) with [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton) acceleration.

3.4x faster than stock OmniVoice. ~170ms per utterance. 646 languages. Voice cloning + voice design. No flash-attn needed.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint (JSON body)
- **Triton kernel fusion + CUDA Graph** — 3.4x faster than baseline
- 646 languages (broadest coverage among open TTS models)
- Voice cloning from 3-second reference audio
- Voice design via natural language description
- Built-in voice presets (auto, female, male, child, elderly, etc.)
- ~2GB model, ~2GB VRAM
- Supports RTX 50-series (Blackwell) GPUs

## Quick Start

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  -v /mnt/user/appdata/omnivoice-triton-api/models:/root/.cache/huggingface \
  --shm-size=4g \
  --name omnivoice-triton-api \
  ghcr.io/hsiang-han/omnivoice-triton-api:latest
```

Or with docker compose:

```bash
docker compose -f docker/gpu/docker-compose.yml up -d
```

First start downloads model (~2GB). China users: set `HF_ENDPOINT=https://hf-mirror.com`.

## Usage Examples

```bash
# Auto voice (default)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "auto"}' \
  --output output.wav

# Voice design (describe the voice you want)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "young female, warm and gentle", "language": "zh"}' \
  --output designed.wav

# Use preset
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Testing male voice.", "voice": "male_en"}' \
  --output male.wav

# Voice cloning
curl -X POST http://localhost:8080/v1/audio/speech/clone \
  -F "input=This is my cloned voice." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=The text spoken in the reference audio." \
  --output cloned.wav

# Faster inference (16 steps instead of 32)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Fast mode!", "voice": "female", "num_step": 16}' \
  --output fast.wav

# List voices
curl http://localhost:8080/v1/voices
```

## Voice Presets

| Voice | Description |
|-------|-------------|
| auto | Auto-generated voice (no instruct) |
| female | Generic female voice |
| male | Generic male voice |
| female_en | Female, American accent |
| male_en | Male, American accent |
| female_zh | Female, Chinese |
| male_zh | Male, Chinese |
| child | Child voice |
| elderly | Elderly female voice |

Or pass any free-form description as `voice` (e.g. "warm baritone, British accent, slow pace").

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Text-to-speech (JSON, OpenAI-compatible) |
| `/v1/audio/speech/clone` | POST | Voice cloning (Form + file upload) |
| `/v1/voices` | GET | List available voice presets |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger documentation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_ID | k2-fsa/OmniVoice | HuggingFace model ID |
| RUNNER_MODE | triton | Inference mode (see below) |
| NUM_STEPS | 32 | Diffusion steps (16 for speed, 32 for quality) |
| DTYPE | fp16 | Model precision |
| DEVICE | cuda:0 | CUDA device |
| PORT | 8080 | API server port |
| HF_ENDPOINT | https://huggingface.co | HuggingFace mirror |

## Runner Modes

| Mode | Speedup | VRAM | Status |
|------|---------|------|--------|
| base | 1.0x | ~2-3GB | Stable |
| **triton** (default) | ~1.5x | ~3-4GB | Stable |
| triton+sage | ~1.5-1.7x | ~3-4GB | Stable |
| faster | ~2.3x | ~5-6GB | ⚠️ VRAM leak |
| hybrid | ~3.4x | ~7GB+ | ⚠️ VRAM leak |
| hybrid+sage | ~3.4x | ~7GB+ | ⚠️ VRAM leak |

> **Warning:** `hybrid` and `faster` modes have a known VRAM leak — memory grows with each request until OOM. See [omnivoice-triton#8](https://github.com/newgrit1004/omnivoice-triton/issues/8). We are tracking the fix. Use `triton` or `triton+sage` for production.

## Hardware Requirements

- NVIDIA GPU with 2GB+ VRAM
- NVIDIA driver 550+ (Ampere/Ada) or 570+ (Blackwell)
- Docker with NVIDIA Container Toolkit

## Credits

- [OmniVoice](https://github.com/k2-fsa/OmniVoice) by k2-fsa (Next-gen Kaldi) — the model (646 languages, Apache-2.0)
- [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton) by [@newgrit1004](https://github.com/newgrit1004) — Triton kernel fusion + CUDA Graph acceleration

## License

Apache-2.0
