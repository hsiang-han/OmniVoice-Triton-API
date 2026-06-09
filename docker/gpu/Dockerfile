FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git sox libsox-dev ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packages --no-deps \
    omnivoice-triton omnivoice && \
    pip install --no-cache-dir --break-system-packages \
    triton sageattention transformers accelerate huggingface-hub \
    soundfile numpy sentencepiece \
    fastapi uvicorn[standard] python-multipart

COPY api/ ./api/
COPY docker/scripts/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_ID=k2-fsa/OmniVoice \
    DTYPE=fp16 \
    DEVICE=cuda:0 \
    NUM_STEPS=32 \
    RUNNER_MODE=hybrid \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://huggingface.co \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
