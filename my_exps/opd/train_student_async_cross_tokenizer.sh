#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_student_async_custom_loss.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

export OPD_CROSS_TOKENIZER_ENABLE="${OPD_CROSS_TOKENIZER_ENABLE:-1}"
export OPD_TEACHER_TOKENIZER_PATH="${OPD_TEACHER_TOKENIZER_PATH:-$HOME/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8}"
export OPD_STUDENT_TOKENIZER_PATH="${OPD_STUDENT_TOKENIZER_PATH:-}"

export MODEL_CONFIG_REL_DEBUG="${MODEL_CONFIG_REL_DEBUG:-scripts/models/qwen3-4B.sh}"
export MODEL_CONFIG_REL_TRAIN="${MODEL_CONFIG_REL_TRAIN:-scripts/models/qwen3-32B.sh}"

export STUDENT_HF_DEFAULT_DEBUG="${STUDENT_HF_DEFAULT_DEBUG:-$HOME/ckpt/hf_models/Qwen/Qwen3-4B}"
export STUDENT_HF_DEFAULT_TRAIN="${STUDENT_HF_DEFAULT_TRAIN:-$HOME/home-trained-model/Stage3_SFT_Epoch3/}"
export RESUME_MODEL_ROOT_DEBUG="${RESUME_MODEL_ROOT_DEBUG:-${STUDENT_HF_DEFAULT_DEBUG}}"
export RESUME_MODEL_ROOT_TRAIN="${RESUME_MODEL_ROOT_TRAIN:-${STUDENT_HF_DEFAULT_TRAIN}}"

if [[ -z "${RESUME_MODEL_ROOT:-}" ]]; then
  RESUME_MODEL_ROOT_SELECTED="${RESUME_MODEL_ROOT_TRAIN}"
  for arg in "$@"; do
    if [[ "${arg}" == "--debug" ]]; then
      RESUME_MODEL_ROOT_SELECTED="${RESUME_MODEL_ROOT_DEBUG}"
      break
    fi
  done
  export RESUME_MODEL_ROOT="${RESUME_MODEL_ROOT_SELECTED}"
fi
if [[ -z "${RESUME_SAVE_TAG:-}" ]]; then
  export RESUME_SAVE_TAG="$(basename "${RESUME_MODEL_ROOT%/}")"
fi

if [[ -z "${DISTILL_LOSS_MODE:-}" ]]; then
  export DISTILL_LOSS_MODE="fkl"
fi

exec "${TARGET_SCRIPT}" "$@"
