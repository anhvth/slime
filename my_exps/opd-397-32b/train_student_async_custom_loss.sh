#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_student_async_distill.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

DEBUG_MODE=0
CLI_LOSS_MODE=""
TARGET_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG_MODE=1
      TARGET_ARGS+=("$1")
      shift
      ;;
    --loss)
      [[ $# -ge 2 ]] || {
        echo "--loss requires a value: rkl|fkl|mixed|jsd" >&2
        exit 1
      }
      CLI_LOSS_MODE="$2"
      shift 2
      ;;
    --loss=*)
      CLI_LOSS_MODE="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--debug] [--loss rkl|fkl|mixed|jsd]"
      exit 0
      ;;
    *)
      TARGET_ARGS+=("$1")
      shift
      ;;
  esac
done

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

next_run_name() {
  local run_root="$1"
  local max_idx=0
  local d=""
  local idx=""

  if [[ -d "${run_root}" ]]; then
    for d in "${run_root}"/run_*; do
      [[ -d "${d}" ]] || continue
      idx="${d##*/run_}"
      [[ "${idx}" =~ ^[0-9]+$ ]] || continue
      if (( idx > max_idx )); then
        max_idx="${idx}"
      fi
    done
  fi

  echo "run_$((max_idx + 1))"
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

if [[ -n "${CLI_LOSS_MODE}" ]]; then
  export DISTILL_LOSS_MODE="${CLI_LOSS_MODE}"
else
  export DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-fkl}"
fi
DISTILL_LOSS_MODE="$(echo "${DISTILL_LOSS_MODE}" | tr '[:upper:]' '[:lower:]')"
case "${DISTILL_LOSS_MODE}" in
  rkl|fkl|mixed|jsd) ;;
  *)
    echo "DISTILL_LOSS_MODE must be one of: rkl, fkl, mixed, jsd. Got '${DISTILL_LOSS_MODE}'" >&2
    exit 1
    ;;
esac
export DISTILL_LOSS_MODE

LOSS_DIR_TAG="${DISTILL_LOSS_MODE}"
case "${DISTILL_LOSS_MODE}" in
  fkl) LOSS_DIR_TAG="forward_kl" ;;
  rkl) LOSS_DIR_TAG="reverse_kl" ;;
  mixed) LOSS_DIR_TAG="mixed_kl" ;;
  jsd) LOSS_DIR_TAG="jsd" ;;
esac

if (( DEBUG_MODE == 1 )); then
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/debug/opd-397-4"
  RUN_MODE_TAG="debug"
else
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/opd-397-32b"
  RUN_MODE_TAG="train"
fi

normalize_bool_01() {
  local raw="${1:-0}"
  raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|t|yes|y|on) echo "1" ;;
    0|false|f|no|n|off|"") echo "0" ;;
    *)
      echo "OPD_PRIVILEGED_ENABLE must be a boolean value, got '${1}'" >&2
      exit 1
      ;;
  esac
}

OPD_PRIVILEGED_ENABLE_NORM="$(normalize_bool_01 "${OPD_PRIVILEGED_ENABLE:-0}")"
export OPD_PRIVILEGED_ENABLE="${OPD_PRIVILEGED_ENABLE_NORM}"
if [[ "${OPD_PRIVILEGED_ENABLE_NORM}" == "1" ]]; then
  PRIV_NAME_TAG="privileged"
else
  PRIV_NAME_TAG="standard"
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
RUN_FAMILY="${RUN_FAMILY:-student_async_${LOSS_DIR_TAG}_${PRIV_NAME_TAG}_from_${RESUME_SAVE_TAG}}"
RUN_ROOT="${OUTPUT_ROOT%/}/${RUN_FAMILY}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ -n "${RUN_INDEX:-}" ]]; then
    [[ "${RUN_INDEX}" =~ ^[1-9][0-9]*$ ]] || {
      echo "RUN_INDEX must be a positive integer, got '${RUN_INDEX}'" >&2
      exit 1
    }
    RUN_NAME="run_${RUN_INDEX}"
  else
    RUN_NAME="$(next_run_name "${RUN_ROOT}")"
  fi
fi
export RUN_NAME

export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/datasets/200k_prompt_for_distillation.jsonl}"
export STUDENT_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT:-${RESUME_HF_DIR}}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:-${RESUME_DIST_DIR}}"
if [[ -z "${STUDENT_SAVE:-}" ]]; then
  export STUDENT_SAVE="${RUN_ROOT}/${RUN_NAME}"
else
  export STUDENT_SAVE="${STUDENT_SAVE%/}"
fi
if [[ -z "${RESUME_FROM_SAVE:-}" ]]; then
  if (( DEBUG_MODE == 1 )); then
    export RESUME_FROM_SAVE=0
  else
    export RESUME_FROM_SAVE=1
  fi
else
  export RESUME_FROM_SAVE
fi
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export WANDB_GROUP="${WANDB_GROUP:-opd-397-32b-${RUN_MODE_TAG}-${LOSS_DIR_TAG}-${PRIV_NAME_TAG}-${RUN_NAME}}"

if [[ -z "${STUDENT_LOAD:-}" ]]; then
  if has_torch_dist_payload "${STUDENT_SAVE}"; then
    export STUDENT_LOAD="${STUDENT_SAVE}"
    export RESUME_FROM_SAVE=1
  else
    export STUDENT_LOAD="${RESUME_DIST_DIR}"
  fi
fi

echo "[wrapper] DISTILL_LOSS_MODE=${DISTILL_LOSS_MODE}"
echo "[wrapper] run_name=${RUN_NAME}"
echo "[wrapper] run_mode=${RUN_MODE_TAG}"
echo "[wrapper] resume_root=${RESUME_MODEL_ROOT}"
echo "[wrapper] prompt_data=${PROMPT_DATA}"
echo "[wrapper] hf_checkpoint=${STUDENT_HF_CHECKPOINT}"
echo "[wrapper] ref_load=${STUDENT_REF_LOAD}"
echo "[wrapper] load=${STUDENT_LOAD}"
echo "[wrapper] save=${STUDENT_SAVE}"
echo "[wrapper] wandb_group=${WANDB_GROUP}"
exec "${TARGET_SCRIPT}" "${TARGET_ARGS[@]}"
