#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_student_async_distill.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

export RESUME_ITER_TAG="${RESUME_ITER_TAG:-iter_0001599}"
export RESUME_HF_DIR="${RESUME_HF_DIR:-${REPO_ROOT}/outputs/opd-397-32b/student_async_hf/${RESUME_ITER_TAG}}"
export RESUME_DIST_DIR="${RESUME_DIST_DIR:-${REPO_ROOT}/outputs/opd-397-32b/student_async_hf/${RESUME_ITER_TAG}_torch_dist}"

export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/datasets/50k_prompt_for_distillation_privileged.jsonl}"
export STUDENT_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT:-${RESUME_HF_DIR}}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:-${RESUME_DIST_DIR}}"
export STUDENT_LOAD="${STUDENT_LOAD:-${RESUME_DIST_DIR}}"
export STUDENT_SAVE="${STUDENT_SAVE:-${REPO_ROOT}/outputs/opd-397-32b/student_async_distill_privileged_from_${RESUME_ITER_TAG}}"

export DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-jsd}"
export OPD_JSD_BETA="${OPD_JSD_BETA:-0.5}"
export OPD_TOP_LOGPROBS_NUM="${OPD_TOP_LOGPROBS_NUM:-16}"

export OPD_PRIVILEGED_ENABLE="${OPD_PRIVILEGED_ENABLE:-1}"
export OPD_PRIVILEGED_METADATA_KEY="${OPD_PRIVILEGED_METADATA_KEY:-privileged_context}"
export OPD_PRIVILEGED_FALLBACK_LABEL="${OPD_PRIVILEGED_FALLBACK_LABEL:-1}"
export OPD_PRIVILEGED_OPEN_TAG="${OPD_PRIVILEGED_OPEN_TAG:-[PRIVILEGED_CONTEXT]}"
export OPD_PRIVILEGED_CLOSE_TAG="${OPD_PRIVILEGED_CLOSE_TAG:-[/PRIVILEGED_CONTEXT]}"

if [[ -z "${OPD_PRIVILEGED_TOKENIZER_PATH:-}" && -n "${STUDENT_HF_CHECKPOINT:-}" ]]; then
  export OPD_PRIVILEGED_TOKENIZER_PATH="${STUDENT_HF_CHECKPOINT%/}"
fi

export LABEL_KEY="${LABEL_KEY:-label}"
export METADATA_KEY="${METADATA_KEY:-metadata}"

[[ -f "${PROMPT_DATA}" ]] || {
  echo "Missing prompt dataset: ${PROMPT_DATA}" >&2
  exit 1
}
[[ -e "${STUDENT_HF_CHECKPOINT}" ]] || {
  echo "Missing student HF checkpoint path: ${STUDENT_HF_CHECKPOINT}" >&2
  exit 1
}
[[ -e "${STUDENT_REF_LOAD}" ]] || {
  echo "Missing student ref-load path: ${STUDENT_REF_LOAD}" >&2
  exit 1
}
[[ -e "${STUDENT_LOAD}" ]] || {
  echo "Missing student load path: ${STUDENT_LOAD}" >&2
  exit 1
}

echo "[wrapper] DISTILL_LOSS_MODE=${DISTILL_LOSS_MODE}, privileged_key=${OPD_PRIVILEGED_METADATA_KEY}, resume_iter=${RESUME_ITER_TAG}"
echo "[wrapper] prompt_data=${PROMPT_DATA}"
echo "[wrapper] hf_checkpoint=${STUDENT_HF_CHECKPOINT}"
echo "[wrapper] ref_load=${STUDENT_REF_LOAD}"
echo "[wrapper] load=${STUDENT_LOAD}"
echo "[wrapper] save=${STUDENT_SAVE}"
exec "${TARGET_SCRIPT}" "$@"
