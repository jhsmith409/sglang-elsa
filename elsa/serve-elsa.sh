#!/usr/bin/env bash
# Qwen3.6-35B-A3B NVFP4 on one RTX 5090 (SM120, 32 GB): 256K context, vision,
# tool calling, NEXTN MTP with a 32K FR-Spec draft vocabulary.
# Configuration: elsa/elsa.env (see elsa.env.example) or ELSA_ENV=/path/to/env.
# Extra arguments are appended to the sglang command line and override defaults.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_FILE="${ELSA_ENV:-$HERE/elsa.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH (in $ENV_FILE) to the local Qwen3.6-35B-A3B-NVFP4 checkpoint}"
VENV="${VENV:-$REPO/.venv}"
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-moe-nvfp4}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
TOKEN_MAP="${TOKEN_MAP:-$HERE/frspec/qwen3.6-35b-a3b-32k.pt}"

[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/hf_quant_config.json" ]] || {
  echo "Incomplete checkpoint at $MODEL_PATH (need config.json and hf_quant_config.json)" >&2; exit 1; }
[[ -x "$VENV/bin/python" ]] || { echo "No environment at $VENV; run elsa/build-venv.sh" >&2; exit 1; }
[[ "$ENABLE_THINKING" == true || "$ENABLE_THINKING" == false ]] || {
  echo "ENABLE_THINKING must be true or false" >&2; exit 1; }

export PATH="$VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# NVML answers device queries in the HTTP process, so CUDA and NVML must agree on order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
# JIT-compiled kernels (FlashInfer, SGLang, Torch extensions) target SM120 only.
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
if [[ -n "${CACHE_DIR:-}" ]]; then
  mkdir -p "$CACHE_DIR"/{triton,flashinfer,sglang,torchinductor}
  export TRITON_CACHE_DIR="$CACHE_DIR/triton" FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
  export SGLANG_CACHE_DIR="$CACHE_DIR/sglang" TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Keep image decode/preprocessing (and so any CUDA context) out of the HTTP process.
export SGLANG_MM_PREPROCESS_DEVICE="${SGLANG_MM_PREPROCESS_DEVICE:-cpu}"
export SGLANG_FLASHINFER_MOE_FUSED_FINALIZE="${SGLANG_FLASHINFER_MOE_FUSED_FINALIZE:-1}"
# FP8 MTP draft: same acceptance as BF16, +80K tokens of KV pool.
export SGLANG_QWEN35_MTP_ONLINE_FP8="${SGLANG_QWEN35_MTP_ONLINE_FP8:-1}"
# Bound the MTP draft's prompt-pass attention to 4K tokens: long-prompt prefill +7-8%,
# decode unchanged (post-verify draft extends keep full attention).
export SGLANG_QWEN35_MTP_DRAFT_WINDOW="${SGLANG_QWEN35_MTP_DRAFT_WINDOW:-4096}"
export TOKENIZERS_PARALLELISM=false

args=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --context-length 262144
  --kv-cache-dtype fp8_e4m3
  # 0.90 passes benchmarks but OOMs under 7 sampled requests + a 200K prefill.
  --mem-fraction-static 0.88
  --moe-runner-backend flashinfer_cutlass
  --mamba-ssm-dtype bfloat16
  --max-running-requests 8 --max-mamba-cache-size 40
  --speculative-algorithm NEXTN
  --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
  --speculative-token-map "$TOKEN_MAP"
  --enable-linear-replayssm-spec
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --default-chat-template-kwargs "{\"enable_thinking\": $ENABLE_THINKING}"
  --watchdog-timeout 1800
)
exec "$VENV/bin/python" -m sglang.launch_server "${args[@]}" "$@"
