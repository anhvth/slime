#!/bin/bash
set -euo pipefail

best_effort_kill() {
  local pattern="$1"
  pkill -f "${pattern}" >/dev/null 2>&1 || true
}

# Ray actors run this script under `set -e`; do not fail if no old server exists.
best_effort_kill "sglang.launch_server"
best_effort_kill "fast_sglang"

echo "Waiting for existing SGLang processes to exit..."
sleep 3
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEFAULT_UV_PYTHON="${SCRIPT_DIR}/teacher-qwen-35/.venv/bin/python"
DEFAULT_REPO_VENV_PYTHON="/home/anhvth8/projects/slime/my_exps/opd-397-32b/teacher-qwen-35/.venv/bin/python"
DEFAULT_SYSTEM_UV_PYTHON="/home/anhvth8/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
DEFAULT_FAST_SGLANG_BIN="/home/anhvth8/dotfiles/mybins/fast_sglang"

python_has_runtime() {
  local py="$1"
  "${py}" - <<'PY' >/dev/null 2>&1
import importlib.util
if not importlib.util.find_spec("sglang"):
    raise SystemExit(1)
if not importlib.util.find_spec("transformers"):
    raise SystemExit(1)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "qwen3_5_moe" not in CONFIG_MAPPING:
    raise SystemExit(1)
PY
}

pick_teacher_python() {
  local candidate resolved

  if [[ -n "${TEACHER_PYTHON:-}" ]]; then
    if python_has_runtime "${TEACHER_PYTHON}"; then
      echo "${TEACHER_PYTHON}"
      return 0
    fi
    echo "[serve_teacher] TEACHER_PYTHON is set but missing sglang/transformers: ${TEACHER_PYTHON}" >&2
    return 1
  fi

  local candidates=(
    "${DEFAULT_REPO_VENV_PYTHON}"
    "${DEFAULT_UV_PYTHON}"
    "${DEFAULT_SYSTEM_UV_PYTHON}"
    "python3"
  )

  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" == *"/"* ]]; then
      [[ -x "${candidate}" ]] || continue
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
      [[ -n "${resolved}" ]] || continue
    fi

    if python_has_runtime "${resolved}"; then
      echo "${resolved}"
      return 0
    fi
  done

  echo "[serve_teacher] no usable Python found (need sglang + transformers)." >&2
  return 1
}

TEACHER_PYTHON="$(pick_teacher_python)"
echo "[serve_teacher] using python: ${TEACHER_PYTHON}" >&2

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
TEACHER_PORT="${TEACHER_PORT:-13142}"
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_TP_FLAG="${TEACHER_TP_FLAG:---tp}"
TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.8}"

EXTRA_ARGS=()
if [[ -n "${TEACHER_CONTEXT_LENGTH:-}" ]]; then
  EXTRA_ARGS+=(--context-length "${TEACHER_CONTEXT_LENGTH}")
fi

FAST_SGLANG_BIN="${TEACHER_FAST_SGLANG_BIN:-}"
if [[ -z "${FAST_SGLANG_BIN}" ]]; then
  if [[ -x "${DEFAULT_FAST_SGLANG_BIN}" ]]; then
    FAST_SGLANG_BIN="${DEFAULT_FAST_SGLANG_BIN}"
  else
    FAST_SGLANG_BIN="$(command -v fast_sglang || true)"
  fi
fi

if [[ -n "${FAST_SGLANG_BIN}" && -x "${FAST_SGLANG_BIN}" ]]; then
  echo "[serve_teacher] using fast_sglang: ${FAST_SGLANG_BIN}" >&2
  # fast_sglang stages the model via symlinks so SGLang starts immediately
  # while the real files are copied in the background.
  exec env PYTHONPATH="" "${TEACHER_PYTHON}" "${FAST_SGLANG_BIN}" \
    "${MODEL_PATH}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    "${TEACHER_TP_FLAG}" "${TEACHER_TP}" \
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
    "${EXTRA_ARGS[@]}"
fi

echo "[serve_teacher] fast_sglang not found; falling back to sglang.launch_server." >&2
exec env PYTHONPATH="" "${TEACHER_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${TEACHER_HOST}" \
  --port "${TEACHER_PORT}" \
  "${TEACHER_TP_FLAG}" "${TEACHER_TP}" \
  --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
  --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
  "${EXTRA_ARGS[@]}"
