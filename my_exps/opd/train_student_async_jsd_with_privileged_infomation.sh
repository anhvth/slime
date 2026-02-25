#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/train_student_async_distill.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Missing required command: ${cmd}" >&2
    exit 1
  }
}

require_file() {
  local path="$1"
  local message="$2"
  [[ -f "${path}" ]] || {
    echo "${message}: ${path}" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  local message="$2"
  [[ -d "${path}" ]] || {
    echo "${message}: ${path}" >&2
    exit 1
  }
}

resolve_torch_dist_payload_dir() {
  local root="$1"
  local marker=""
  local candidate=""

  if [[ -f "${root}/common.pt" ]]; then
    printf '%s\n' "${root}"
    return 0
  fi

  if [[ -f "${root}/latest_checkpointed_iteration.txt" ]]; then
    marker="$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt" || true)"
    if [[ -n "${marker}" && -f "${root}/${marker}/common.pt" ]]; then
      printf '%s\n' "${root}/${marker}"
      return 0
    fi
  fi

  for candidate in "${root}/release" "${root}"/iter_*; do
    [[ -d "${candidate}" ]] || continue
    if [[ -f "${candidate}/common.pt" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

validate_torch_dist_dir() {
  local root="$1"
  local label="$2"
  local payload_dir=""

  require_dir "${root}" "Missing student ${label} directory"
  if ! payload_dir="$(resolve_torch_dist_payload_dir "${root}")"; then
    echo "Missing student ${label} manifest: ${root}/common.pt (or release/iter_*/common.pt)" >&2
    exit 1
  fi
  if [[ ! -f "${payload_dir}/.metadata" && ! -f "${payload_dir}/metadata.json" ]]; then
    echo "student ${label} appears incomplete: missing .metadata/metadata.json in ${payload_dir}" >&2
    exit 1
  fi
}

wait_http_healthy() {
  local url="$1"
  local name="$2"
  local max_attempts="${3:-15}"
  local sleep_seconds="${4:-2}"
  local connect_timeout="${TEACHER_CONNECT_TIMEOUT:-2}"
  local max_time="${TEACHER_MAX_TIME:-8}"
  local attempt

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if curl -sf --connect-timeout "${connect_timeout}" --max-time "${max_time}" "${url}" >/dev/null; then
      return 0
    fi
    if (( attempt < max_attempts )); then
      sleep "${sleep_seconds}"
    fi
  done

  echo "Teacher preflight failed after ${max_attempts} attempts: ${name} (${url})" >&2
  exit 1
}

export RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://100.96.4.38:8265}"
export TEACHER_URL="${TEACHER_URL:-http://worker-30:13142/generate}"

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
  if [[ -d "${RESUME_MODEL_ROOT%/}_torch_dist" ]]; then
    export RESUME_DIST_DIR="${RESUME_MODEL_ROOT%/}_torch_dist"
  else
    export RESUME_DIST_DIR="${RESUME_MODEL_ROOT%/}/dist"
  fi
fi

export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/datasets/50k_prompt_for_distillation_privileged.jsonl}"
export STUDENT_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT:-${RESUME_HF_DIR}}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:-${RESUME_DIST_DIR}}"
export STUDENT_SAVE="${STUDENT_SAVE:-${REPO_ROOT}/outputs/opd/student_async_distill_privileged_from_${RESUME_SAVE_TAG}}"
export RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-1}"

# 15-node async layout defaults (120 GPUs total): 48 train (6 nodes) + 72 rollout (9 nodes).
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-6}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-72}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"

# Reward HTTP client tuning for high-concurrency teacher calls.
export OPD_RM_CONNECT_TIMEOUT_S="${OPD_RM_CONNECT_TIMEOUT_S:-2.0}"
export OPD_RM_READ_TIMEOUT_S="${OPD_RM_READ_TIMEOUT_S:-120.0}"
export OPD_RM_TOTAL_TIMEOUT_S="${OPD_RM_TOTAL_TIMEOUT_S:-180.0}"
export OPD_RM_MAX_CONNECTIONS="${OPD_RM_MAX_CONNECTIONS:-512}"
export OPD_RM_MAX_CONNECTIONS_PER_HOST="${OPD_RM_MAX_CONNECTIONS_PER_HOST:-256}"
export OPD_RM_RETRY_ATTEMPTS="${OPD_RM_RETRY_ATTEMPTS:-6}"
export OPD_RM_RETRY_BASE_SLEEP_S="${OPD_RM_RETRY_BASE_SLEEP_S:-0.15}"
export OPD_RM_RETRY_MAX_SLEEP_S="${OPD_RM_RETRY_MAX_SLEEP_S:-2.0}"

export DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-jsd}"
export OPD_JSD_BETA="${OPD_JSD_BETA:-0.5}"
export OPD_TOP_LOGPROBS_NUM="${OPD_TOP_LOGPROBS_NUM:-16}"

export OPD_PRIVILEGED_ENABLE="${OPD_PRIVILEGED_ENABLE:-1}"
export OPD_PRIVILEGED_METADATA_KEY="${OPD_PRIVILEGED_METADATA_KEY:-privileged_context}"
export OPD_PRIVILEGED_FALLBACK_LABEL="${OPD_PRIVILEGED_FALLBACK_LABEL:-1}"

if [[ -z "${OPD_PRIVILEGED_TOKENIZER_PATH:-}" && -n "${STUDENT_HF_CHECKPOINT:-}" ]]; then
  export OPD_PRIVILEGED_TOKENIZER_PATH="${STUDENT_HF_CHECKPOINT%/}"
fi

export LABEL_KEY="${LABEL_KEY:-label}"
export METADATA_KEY="${METADATA_KEY:-metadata}"

if [[ -z "${STUDENT_LOAD:-}" ]]; then
  if resolve_torch_dist_payload_dir "${STUDENT_SAVE}" >/dev/null 2>&1; then
    export STUDENT_LOAD="${STUDENT_SAVE}"
    export RESUME_FROM_SAVE=1
  else
    export STUDENT_LOAD="${RESUME_DIST_DIR}"
  fi
fi

require_cmd ray
require_cmd curl

TEACHER_BASE="${TEACHER_URL%/generate}"
[[ "${TEACHER_BASE}" != "${TEACHER_URL}" ]] || {
  echo "TEACHER_URL must end with /generate: ${TEACHER_URL}" >&2
  exit 1
}

require_file "${PROMPT_DATA}" "Missing prompt dataset"
if ! [[ -s "${PROMPT_DATA}" ]]; then
  echo "Prompt dataset is empty: ${PROMPT_DATA}" >&2
  exit 1
fi

require_dir "${STUDENT_HF_CHECKPOINT}" "Missing student HF checkpoint directory"
require_file "${STUDENT_HF_CHECKPOINT}/config.json" "Missing HF checkpoint config"
if [[ ! -f "${STUDENT_HF_CHECKPOINT}/model.safetensors.index.json" && ! -f "${STUDENT_HF_CHECKPOINT}/model.safetensors" \
   && ! -f "${STUDENT_HF_CHECKPOINT}/pytorch_model.bin.index.json" && ! -f "${STUDENT_HF_CHECKPOINT}/pytorch_model.bin" ]]; then
  echo "HF checkpoint appears incomplete: expected model index or weight file under ${STUDENT_HF_CHECKPOINT}" >&2
  exit 1
fi

validate_torch_dist_dir "${STUDENT_REF_LOAD}" "ref-load"
validate_torch_dist_dir "${STUDENT_LOAD}" "load"

if ! ray job list --address="${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
  echo "Ray preflight failed: unable to reach ${RAY_JOB_ADDRESS}" >&2
  exit 1
fi

wait_http_healthy "${TEACHER_BASE}/health_generate" "health_generate"
wait_http_healthy "${TEACHER_BASE}/get_model_info" "get_model_info"

echo "[wrapper] DISTILL_LOSS_MODE=${DISTILL_LOSS_MODE}, privileged_key=${OPD_PRIVILEGED_METADATA_KEY}, resume_root=${RESUME_MODEL_ROOT}"
echo "[wrapper] ray_job_address=${RAY_JOB_ADDRESS}"
echo "[wrapper] teacher_url=${TEACHER_URL}"
echo "[wrapper] prompt_data=${PROMPT_DATA}"
echo "[wrapper] hf_checkpoint=${STUDENT_HF_CHECKPOINT}"
echo "[wrapper] ref_load=${STUDENT_REF_LOAD}"
echo "[wrapper] load=${STUDENT_LOAD}"
echo "[wrapper] save=${STUDENT_SAVE}"
echo "[wrapper] async_layout actor_nodes=${ACTOR_NUM_NODES} actor_gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE} rollout_gpus=${ROLLOUT_NUM_GPUS} rollout_gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "[wrapper] rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}"
exec "${TARGET_SCRIPT}" "$@"
