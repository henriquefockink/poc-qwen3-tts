#!/bin/bash
set -e

echo "=== Qwen3-TTS vLLM-Omni Server ==="
echo "Model: ${MODEL}"
echo "Port: ${PORT:-8091}"

# Generate stage configs YAML from env vars
python3 /app/stage_configs/generate_config.py

STAGE_CONFIGS_PATH="/tmp/stage_configs.yaml"

exec vllm serve "${MODEL}" \
  --stage-configs-path "${STAGE_CONFIGS_PATH}" \
  --omni \
  --port "${PORT:-8091}" \
  --host 0.0.0.0 \
  $([ "${TRUST_REMOTE_CODE}" = "true" ] && echo "--trust-remote-code") \
  $([ "${ENFORCE_EAGER}" = "true" ] && echo "--enforce-eager") \
  "$@"
