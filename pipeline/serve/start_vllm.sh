#!/usr/bin/env bash
# Serve one base model plus every client's LoRA adapter with vLLM (OpenAI-compatible API).
#
#   BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 ADAPTERS_DIR=/data/adapters VLLM_API_KEY=... ./start_vllm.sh
#
# Each sub-folder of ADAPTERS_DIR that holds an adapter_config.json is served under the
# folder's name, e.g. /data/adapters/acme-support -> model "acme-support".
# New adapters can be added without a restart: python register_adapter.py acme-faq /data/adapters/acme-faq
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTERS_DIR="${ADAPTERS_DIR:-/data/adapters}"
PORT="${PORT:-8000}"
MAX_LORAS="${MAX_LORAS:-8}"            # adapters kept on the GPU at once (others swap in from CPU)
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"   # must be >= the rank used in training
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM="${GPU_MEM:-0.90}"
: "${VLLM_API_KEY:?set VLLM_API_KEY - Tokenwise stores it as the upstream key for clients on this server}"

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

modules=()
if [[ -d "$ADAPTERS_DIR" ]]; then
  for d in "$ADAPTERS_DIR"/*/; do
    [[ -f "$d/adapter_config.json" ]] && modules+=("$(basename "$d")=${d%/}")
  done
fi
echo "base: $BASE_MODEL | adapters: ${modules[*]:-(none yet)}"

args=(serve "$BASE_MODEL" --port "$PORT" --api-key "$VLLM_API_KEY"
      --enable-lora --max-loras "$MAX_LORAS" --max-lora-rank "$MAX_LORA_RANK"
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM")
[[ ${#modules[@]} -gt 0 ]] && args+=(--lora-modules "${modules[@]}")
# GPUs without bfloat16 (T4, V100) need half precision.
python - <<'PY' || args+=(--dtype half)
import sys, torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)
PY
exec vllm "${args[@]}"
