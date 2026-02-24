#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_student_async_distill.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

has_torch_dist_payload() {
  local root="$1"

  [[ -f "${root}/common.pt" ]] && return 0
  [[ -f "${root}/latest_checkpointed_iteration.txt" ]] && return 0
  [[ -f "${root}/release/common.pt" ]] && return 0

  local candidate=""
  for candidate in "${root}"/iter_*; do
    [[ -d "${candidate}" ]] || continue
    [[ -f "${candidate}/common.pt" ]] && return 0
  done

  return 1
}

export RESUME_MODEL_ROOT="${RESUME_MODEL_ROOT:-$HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned}"
DEFAULT_RESUME_BASENAME="$(basename "${RESUME_MODEL_ROOT%/}")"
export RESUME_SAVE_TAG="${RESUME_SAVE_TAG:-${DEFAULT_RESUME_BASENAME}}"

if [[ -z "${RESUME_HF_DIR:-}" ]]; then
  if [[ -f "${RESUME_MODEL_ROOT%/}/config.json" ]]; then
    export RESUME_HF_DIR="${RESUME_MODEL_ROOT%/}"
  else
    export RESUME_HF_DIR="${RESUME_MODEL_ROOT%/}/hf"
  fi
fi

if [[ -z "${RESUME_DIST_DIR:-}" ]]; then
  if [[ -d "${RESUME_MODEL_ROOT%/}/dist" ]]; then
    export RESUME_DIST_DIR="${RESUME_MODEL_ROOT%/}/dist"
  elif [[ -d "${RESUME_MODEL_ROOT%/}_torch_dist" ]]; then
    export RESUME_DIST_DIR="${RESUME_MODEL_ROOT%/}_torch_dist"
  else
    export RESUME_DIST_DIR="${RESUME_MODEL_ROOT%/}/dist"
  fi
fi

export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/datasets/200k_prompt_for_distillation.jsonl}"
export STUDENT_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT:-${RESUME_HF_DIR}}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:-${RESUME_DIST_DIR}}"
export STUDENT_SAVE="${STUDENT_SAVE:-${REPO_ROOT}/outputs/opd-397-32b/student_async_forward_kl_from_${RESUME_SAVE_TAG}}"
export RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-1}"

export DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-fkl}"
export OPD_PRIVILEGED_ENABLE="${OPD_PRIVILEGED_ENABLE:-0}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"

if [[ -z "${STUDENT_LOAD:-}" ]]; then
  if has_torch_dist_payload "${STUDENT_SAVE}"; then
    export STUDENT_LOAD="${STUDENT_SAVE}"
    export RESUME_FROM_SAVE=1
  else
    export STUDENT_LOAD="${RESUME_DIST_DIR}"
  fi
fi

echo "[wrapper] DISTILL_LOSS_MODE=${DISTILL_LOSS_MODE}"
echo "[wrapper] resume_root=${RESUME_MODEL_ROOT}"
echo "[wrapper] prompt_data=${PROMPT_DATA}"
echo "[wrapper] hf_checkpoint=${STUDENT_HF_CHECKPOINT}"
echo "[wrapper] ref_load=${STUDENT_REF_LOAD}"
echo "[wrapper] load=${STUDENT_LOAD}"
echo "[wrapper] save=${STUDENT_SAVE}"
exec "${TARGET_SCRIPT}" "$@"
