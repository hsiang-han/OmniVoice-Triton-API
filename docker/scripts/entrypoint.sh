#!/bin/bash
set -e

echo "=== OmniVoice-Triton-API ==="
echo "Model:  ${MODEL_ID}"
echo "Runner: ${RUNNER_MODE}"
echo "Steps:  ${NUM_STEPS}"
echo "Dtype:  ${DTYPE}"
echo "Port:   ${PORT}"
echo "============================="

exec python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info \
    --timeout-keep-alive 65
