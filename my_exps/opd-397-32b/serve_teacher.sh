#!/bin/bash
set -euo pipefail

DEBUG=0
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--debug]"
  exit 1
fi

MODEL_HOME="${MODEL_HOME:-$HOME/ckpt/hf_models/Qwen}"
PROD_MODEL_ID="${PROD_MODEL_ID:-Qwen/Qwen3.5-397B-A17B-FP8}"
PROD_MODEL_DIR="${PROD_MODEL_DIR:-${MODEL_HOME}/Qwen3.5-397B-A17B-FP8}"
DEBUG_MODEL_DIR="${DEBUG_MODEL_DIR:-${MODEL_HOME}/Qwen3-4B}"

if [[ ${DEBUG} -eq 1 ]]; then
  MODEL_PATH="${TEACHER_MODEL:-${DEBUG_MODEL_DIR}}"
else
  if [[ -n "${TEACHER_MODEL:-}" ]]; then
    MODEL_PATH="${TEACHER_MODEL}"
  elif [[ -d "${PROD_MODEL_DIR}" ]]; then
    MODEL_PATH="${PROD_MODEL_DIR}"
  else
    MODEL_PATH="${PROD_MODEL_ID}"
  fi
fi

TEACHER_HOST="${TEACHER_HOST:-0.0.0.0}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_TP_FLAG="${TEACHER_TP_FLAG:---tp}"
TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.8}"

EXTRA_ARGS=()
if [[ -n "${TEACHER_CONTEXT_LENGTH:-}" ]]; then
  EXTRA_ARGS+=(--context-length "${TEACHER_CONTEXT_LENGTH}")
fi

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${TEACHER_HOST}" \
  --port "${TEACHER_PORT}" \
  "${TEACHER_TP_FLAG}" "${TEACHER_TP}" \
  --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
  --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
  "${EXTRA_ARGS[@]}"
