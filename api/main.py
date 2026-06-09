import io
import os
import struct
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
DTYPE = os.getenv("DTYPE", "fp16")
DEVICE = os.getenv("DEVICE", "cuda:0")
NUM_STEPS = int(os.getenv("NUM_STEPS", "32"))
RUNNER_MODE = os.getenv("RUNNER_MODE", "hybrid")

SAMPLE_RATE = 24000
_runner = None
_model_lock = threading.Lock()

DEFAULT_VOICE_INSTRUCT = os.getenv("DEFAULT_VOICE", "female, young adult")

VOICE_PRESETS = {
    "auto": DEFAULT_VOICE_INSTRUCT,
    "female": "female",
    "male": "male",
    "female_en": "female, american accent",
    "male_en": "male, american accent",
    "female_zh": "female, chinese",
    "male_zh": "male, chinese",
    "child": "child",
    "elderly": "elderly, female",
    # OpenAI standard voice names → mapped to instruct descriptions
    "alloy": "female, neutral tone",
    "echo": "male, deep voice",
    "fable": "male, warm storyteller",
    "onyx": "male, deep authoritative",
    "nova": "female, young energetic",
    "shimmer": "female, soft gentle",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runner, SAMPLE_RATE
    from omnivoice_triton import create_runner

    _runner = create_runner(RUNNER_MODE, device=DEVICE, model_id=MODEL_ID, dtype=DTYPE)
    _runner.load_model()
    SAMPLE_RATE = 24000

    yield
    if _runner:
        _runner.unload_model()
        _runner = None


app = FastAPI(title="OmniVoice-Triton-API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _runner else "loading",
        "model": MODEL_ID,
        "dtype": DTYPE,
        "runner_mode": RUNNER_MODE,
        "num_steps": NUM_STEPS,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "k2-fsa",
                "modes": ["auto", "voice_clone", "voice_design"],
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": list(VOICE_PRESETS.keys()),
        "note": "Use any preset name, or pass a free-form description as voice (e.g. 'young female, warm tone')",
    }


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: str = Field(default="auto", description="Preset name or free-form voice description")
    language: Optional[str] = Field(default=None, description="ISO language code (auto-detected if omitted)")
    response_format: str = "wav"
    speed: float = Field(default=1.0, description="Accepted for OpenAI compatibility but not yet applied")
    num_step: int = Field(default=0, description="Diffusion steps (0 = use server default)")
    guidance_scale: float = Field(default=2.0)


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if not _runner:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    instruct = _resolve_voice(req.voice)
    steps = req.num_step if req.num_step > 0 else NUM_STEPS

    with _model_lock:
        if instruct:
            result = _runner.generate_voice_design(
                text=req.input,
                instruct=instruct,
                language=req.language,
                num_step=steps,
                guidance_scale=req.guidance_scale,
            )
        else:
            result = _runner.generate(
                text=req.input,
                language=req.language,
                num_step=steps,
                guidance_scale=req.guidance_scale,
            )

    audio = result["audio"]
    sr = result["sample_rate"]

    return _format_audio_response(audio, sr, req.response_format)


@app.post("/v1/audio/speech/clone")
async def clone_speech(
    input: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(default=""),
    language: Optional[str] = Form(default=None),
    num_step: int = Form(default=0),
):
    if not _runner:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    steps = num_step if num_step > 0 else NUM_STEPS

    try:
        with _model_lock:
            result = _runner.generate_voice_clone(
                text=input,
                ref_audio=tmp_path,
                ref_text=ref_text,
                language=language,
                num_step=steps,
            )
    finally:
        os.unlink(tmp_path)

    audio = result["audio"]
    sr = result["sample_rate"]
    return _format_audio_response(audio, sr, "wav")


def _format_audio_response(audio: np.ndarray, sr: int, fmt: str) -> Response:
    if fmt == "pcm":
        return Response(content=_to_pcm16(audio), media_type="audio/pcm")

    if fmt in ("flac", "opus", "ogg"):
        try:
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format=fmt.upper())
            buf.seek(0)
            media = {"flac": "audio/flac", "opus": "audio/ogg", "ogg": "audio/ogg"}
            return Response(content=buf.read(), media_type=media.get(fmt, "audio/ogg"))
        except Exception:
            pass

    if fmt == "mp3":
        try:
            import subprocess
            wav_bytes = _to_wav(audio, sr)
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-ab", "192k", "pipe:1"],
                input=wav_bytes, capture_output=True, check=True,
            )
            return Response(content=result.stdout, media_type="audio/mpeg")
        except Exception:
            pass

    if fmt == "aac":
        try:
            import subprocess
            wav_bytes = _to_wav(audio, sr)
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", "adts", "-ab", "192k", "pipe:1"],
                input=wav_bytes, capture_output=True, check=True,
            )
            return Response(content=result.stdout, media_type="audio/aac")
        except Exception:
            pass

    return Response(content=_to_wav(audio, sr), media_type="audio/wav")


VALID_INSTRUCT_KEYWORDS_EN = {
    "american accent", "australian accent", "british accent", "canadian accent",
    "child", "chinese accent", "elderly", "female", "high pitch", "indian accent",
    "japanese accent", "korean accent", "low pitch", "male", "middle-aged",
    "moderate pitch", "portuguese accent", "russian accent", "teenager",
    "very high pitch", "very low pitch", "whisper", "young adult",
}


def _resolve_voice(voice: str) -> Optional[str]:
    if voice in VOICE_PRESETS:
        return VOICE_PRESETS[voice]
    if not voice or voice == "auto":
        return None
    # Check if it looks like a valid instruct (contains known keywords)
    voice_lower = voice.lower()
    for keyword in VALID_INSTRUCT_KEYWORDS_EN:
        if keyword in voice_lower:
            return voice
    # Chinese instruct check
    if any(c in voice for c in "男女老少年儿童耳语"):
        return voice
    # Unknown voice name — fallback to default (no instruct)
    return None


def _to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int) -> bytes:
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_len))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(audio)
    return _wav_header(sample_rate, len(raw)) + raw
