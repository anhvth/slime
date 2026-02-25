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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LIB_SCRIPT="${SCRIPT_DIR}/train_student_async_distill_lib.sh"
[[ -f "${LIB_SCRIPT}" ]] || { echo "Missing distill helper library: ${LIB_SCRIPT}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${LIB_SCRIPT}"

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Missing required command: ${cmd}" >&2
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

path_uses_qwen35_converted_vocab() {
  local path_value="${1:-}"
  [[ -n "${path_value}" ]] || return 1
  local lowered="${path_value,,}"
  [[ "${lowered}" == *"as-qwen35"* ]]
}

validate_hf_checkpoint_dir() {
  local hf_dir="$1"
  require_dir "${hf_dir}" "Missing student HF checkpoint directory"
  require_file "${hf_dir}/config.json" "Missing student HF config"
  if [[ ! -f "${hf_dir}/model.safetensors.index.json" && ! -f "${hf_dir}/model.safetensors" \
     && ! -f "${hf_dir}/pytorch_model.bin.index.json" && ! -f "${hf_dir}/pytorch_model.bin" ]]; then
    echo "HF checkpoint appears incomplete: missing model index/weights under ${hf_dir}" >&2
    exit 1
  fi
}

validate_torch_dist_dir() {
  local dist_dir="$1"
  local name="$2"
  local payload_dir=""

  require_dir "${dist_dir}" "Missing ${name} torch_dist directory"
  if ! payload_dir="$(resolve_torch_dist_payload_dir "${dist_dir}")"; then
    echo "Missing ${name} torch_dist manifest: ${dist_dir}/common.pt (or release/iter_*/common.pt)" >&2
    exit 1
  fi
  if [[ ! -f "${payload_dir}/.metadata" && ! -f "${payload_dir}/metadata.json" ]]; then
    echo "${name} torch_dist appears incomplete: missing .metadata/metadata.json in ${payload_dir}" >&2
    exit 1
  fi
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

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/training_async_distill_active.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') train_student_async_distill start ====="

export PYTHONBUFFERED=1
if [[ ${DEBUG} -eq 1 ]]; then
  MODEL_CONFIG_REL="${MODEL_CONFIG_REL_DEBUG:-${MODEL_CONFIG_REL:-scripts/models/qwen3-4B-as-qwen35.sh}}"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT_DEBUG:-${STUDENT_HF_DEFAULT:-/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35}}"
  # Force debug-model paths; override any wrapper-supplied 32B paths
  STUDENT_HF_CHECKPOINT="${STUDENT_HF_DEFAULT}"
  unset STUDENT_REF_LOAD
  unset STUDENT_LOAD
else
  MODEL_CONFIG_REL="${MODEL_CONFIG_REL_TRAIN:-${MODEL_CONFIG_REL:-scripts/models/qwen3-32B-as-qwen35.sh}}"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT_TRAIN:-${STUDENT_HF_DEFAULT:-$HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned/}}"
fi
MODEL_CONFIG_SCRIPT="${REPO_ROOT}/${MODEL_CONFIG_REL}"
require_file "${MODEL_CONFIG_SCRIPT}" "Missing model config script"
source "${MODEL_CONFIG_SCRIPT}"

TRAIN_PY_PATH="${TRAIN_PY_PATH:-${REPO_ROOT}/train_async.py}"
require_file "${TRAIN_PY_PATH}" "Missing train entrypoint"
require_cmd ray
require_cmd curl

STUDENT_HF_CHECKPOINT_PATH="${STUDENT_HF_CHECKPOINT:-${STUDENT_HF_DEFAULT}}"
STUDENT_HF_CHECKPOINT_PATH="${STUDENT_HF_CHECKPOINT_PATH%/}"
STUDENT_REF_LOAD_PATH="${STUDENT_REF_LOAD:-${STUDENT_HF_CHECKPOINT_PATH}_torch_dist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/opd-397-32b}"
STUDENT_SAVE_PATH="${STUDENT_SAVE:-${OUTPUT_ROOT}/student_async_distill}"
RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-0}"
if [[ -n "${STUDENT_LOAD:-}" ]]; then
  STUDENT_LOAD_PATH="${STUDENT_LOAD}"
elif [[ "${RESUME_FROM_SAVE}" == "1" ]]; then
  STUDENT_LOAD_PATH="${STUDENT_SAVE_PATH}"
else
  STUDENT_LOAD_PATH=""
fi
mkdir -p "${OUTPUT_ROOT}"
PROMPT_DATA_PATH="${PROMPT_DATA:-${REPO_ROOT}/datasets/200k_prompt_for_distillation.jsonl}"
require_file "${PROMPT_DATA_PATH}" "Missing prompt dataset"
if ! [[ -s "${PROMPT_DATA_PATH}" ]]; then
  echo "Prompt dataset is empty: ${PROMPT_DATA_PATH}" >&2
  exit 1
fi

OPD_CROSS_TOKENIZER_ENABLE_NORM="$(normalize_bool_flag "${OPD_CROSS_TOKENIZER_ENABLE:-0}" "OPD_CROSS_TOKENIZER_ENABLE")"
if [[ "${OPD_CROSS_TOKENIZER_ENABLE_NORM}" == "1" ]]; then
  if path_uses_qwen35_converted_vocab "${STUDENT_HF_CHECKPOINT_PATH}"; then
    echo "Invalid cross-tokenizer config: STUDENT_HF_CHECKPOINT='${STUDENT_HF_CHECKPOINT_PATH}'" >&2
    echo "Cross-tokenizer distillation must use native Qwen3 checkpoints (no '*-As-Qwen35*')." >&2
    exit 1
  fi
  if path_uses_qwen35_converted_vocab "${STUDENT_REF_LOAD_PATH}"; then
    echo "Invalid cross-tokenizer config: STUDENT_REF_LOAD='${STUDENT_REF_LOAD_PATH}'" >&2
    echo "Cross-tokenizer distillation must use native Qwen3 checkpoints (no '*-As-Qwen35*')." >&2
    exit 1
  fi
  if [[ -n "${STUDENT_LOAD_PATH}" ]] && path_uses_qwen35_converted_vocab "${STUDENT_LOAD_PATH}"; then
    echo "Invalid cross-tokenizer config: STUDENT_LOAD='${STUDENT_LOAD_PATH}'" >&2
    echo "Cross-tokenizer distillation must use native Qwen3 checkpoints (no '*-As-Qwen35*')." >&2
    exit 1
  fi
fi

validate_hf_checkpoint_dir "${STUDENT_HF_CHECKPOINT_PATH}"

setup_distill_mode
if [[ "${OPD_CROSS_TOKENIZER_ENABLE}" == "1" && -z "${OPD_STUDENT_TOKENIZER_PATH}" ]]; then
  OPD_STUDENT_TOKENIZER_PATH="${STUDENT_HF_CHECKPOINT_PATH%/}"
  echo "Cross-tokenizer default: OPD_STUDENT_TOKENIZER_PATH=${OPD_STUDENT_TOKENIZER_PATH}"
fi
preflight_cross_tokenizer_vocab_match \
  "${OPD_CROSS_TOKENIZER_ENABLE}" \
  "${MODEL_CONFIG_REL}" \
  "${STUDENT_HF_CHECKPOINT_PATH}" \
  MODEL_ARGS

ENSURE_REF_MODEL_SCRIPT="${SCRIPT_DIR}/ensure_ref_model.sh"
require_file "${ENSURE_REF_MODEL_SCRIPT}" "Missing ref model helper script"
bash "${ENSURE_REF_MODEL_SCRIPT}" \
  --repo-root "${REPO_ROOT}" \
  --model-config-rel "${MODEL_CONFIG_REL}" \
  --hf-checkpoint "${STUDENT_HF_CHECKPOINT_PATH}" \
  --ref-load "${STUDENT_REF_LOAD_PATH}" \
  --megatron-pythonpath "${MEGATRON_PYTHONPATH:-/root/Megatron-LM}"
validate_torch_dist_dir "${STUDENT_REF_LOAD_PATH}" "ref-load"
if [[ -n "${STUDENT_LOAD_PATH}" ]]; then
  validate_torch_dist_dir "${STUDENT_LOAD_PATH}" "load"
fi

MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"
CHECKPOINT_PRUNE_INTERVAL_SEC="${CHECKPOINT_PRUNE_INTERVAL_SEC:-300}"
PRUNE_CKPT_SCRIPT="${SCRIPT_DIR}/prune_old_ckpt.sh"
require_file "${PRUNE_CKPT_SCRIPT}" "Missing checkpoint pruner script"
require_non_negative_int "${MAX_CHECKPOINTS_TO_KEEP}" "MAX_CHECKPOINTS_TO_KEEP"
require_positive_int "${CHECKPOINT_PRUNE_INTERVAL_SEC}" "CHECKPOINT_PRUNE_INTERVAL_SEC"

TEACHER_HOST="${TEACHER_HOST:-worker-30}"
TEACHER_PORT="${TEACHER_PORT:-13142}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
TEACHER_BASE="${TEACHER_URL%/generate}"
[[ "${TEACHER_BASE}" != "${TEACHER_URL}" ]] || { echo "TEACHER_URL must end with /generate" >&2; exit 1; }

RAY_JOB_ADDRESS="$(require_ray_job_address)"
echo "Using Ray Job server: ${RAY_JOB_ADDRESS}"

TEACHER_HEALTH_MAX_ATTEMPTS="${TEACHER_HEALTH_MAX_ATTEMPTS:-15}"
TEACHER_HEALTH_RETRY_SEC="${TEACHER_HEALTH_RETRY_SEC:-2}"
require_positive_int "${TEACHER_HEALTH_MAX_ATTEMPTS}" "TEACHER_HEALTH_MAX_ATTEMPTS"
require_positive_int "${TEACHER_HEALTH_RETRY_SEC}" "TEACHER_HEALTH_RETRY_SEC"

wait_http_healthy "${TEACHER_BASE}/health_generate" "health_generate" "${TEACHER_HEALTH_MAX_ATTEMPTS}" "${TEACHER_HEALTH_RETRY_SEC}"
wait_http_healthy "${TEACHER_BASE}/get_model_info" "get_model_info" "${TEACHER_HEALTH_MAX_ATTEMPTS}" "${TEACHER_HEALTH_RETRY_SEC}"

# Opinionated async layout for a 120-GPU cluster (15x8): 56 train (7 nodes) + 64 rollout (8 nodes).
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-7}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
(( WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE == 0 )) || {
  echo "Invalid parallelism: world_size=${WORLD_SIZE} not divisible by TP=${TENSOR_MODEL_PARALLEL_SIZE}" >&2
  exit 1
}
DP_SIZE=$((WORLD_SIZE / TENSOR_MODEL_PARALLEL_SIZE))

BASE_PROMPTS_PER_DP="${BASE_PROMPTS_PER_DP:-8}"
if [[ -z "${ROLLOUT_BATCH_SIZE:-}" ]]; then
  ROLLOUT_BATCH_SIZE=$((BASE_PROMPTS_PER_DP * DP_SIZE))
fi
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
if [[ -z "${GLOBAL_BATCH_SIZE:-}" ]]; then
  GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
fi
TARGET_PROMPTS="${TARGET_PROMPTS:-200000}"

# Enforce:
# 1) rollout_batch_size * n_samples_per_prompt == global_batch_size
# 2) global_batch_size divisible by dp_size
GCD_DP_NSAMPLES="$(gcd "${DP_SIZE}" "${N_SAMPLES_PER_PROMPT}")"
ROLLOUT_BATCH_GRANULARITY=$((DP_SIZE / GCD_DP_NSAMPLES))
if (( ROLLOUT_BATCH_SIZE % ROLLOUT_BATCH_GRANULARITY != 0 )); then
  RAW_ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}"
  ROLLOUT_BATCH_SIZE=$(( (ROLLOUT_BATCH_SIZE / ROLLOUT_BATCH_GRANULARITY) * ROLLOUT_BATCH_GRANULARITY ))
  (( ROLLOUT_BATCH_SIZE > 0 )) || ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_GRANULARITY}"
  echo "Adjust rollout_batch_size ${RAW_ROLLOUT_BATCH_SIZE} -> ${ROLLOUT_BATCH_SIZE} (granularity=${ROLLOUT_BATCH_GRANULARITY} for dp_size=${DP_SIZE}, n_samples_per_prompt=${N_SAMPLES_PER_PROMPT})"
fi
TOTAL_SAMPLES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if (( GLOBAL_BATCH_SIZE != TOTAL_SAMPLES )); then
  echo "Adjust global_batch_size ${GLOBAL_BATCH_SIZE} -> ${TOTAL_SAMPLES} to satisfy one train step per rollout"
  GLOBAL_BATCH_SIZE="${TOTAL_SAMPLES}"
fi
(( GLOBAL_BATCH_SIZE % DP_SIZE == 0 )) || {
  echo "Invalid batching after auto-adjust: global_batch_size=${GLOBAL_BATCH_SIZE} not divisible by dp_size=${DP_SIZE}" >&2
  exit 1
}
NUM_ROLLOUT_DEFAULT=$(((TARGET_PROMPTS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
if [[ -z "${ROLLOUT_MAX_CONTEXT_LEN:-}" ]]; then
  ROLLOUT_MAX_CONTEXT_LEN="$(detect_rollout_context_len "${STUDENT_HF_CHECKPOINT_PATH}" || true)"
fi
[[ -n "${ROLLOUT_MAX_CONTEXT_LEN:-}" ]] || {
  echo "Unable to determine ROLLOUT_MAX_CONTEXT_LEN from ${STUDENT_HF_CHECKPOINT_PATH}/config.json." >&2
  echo "Set ROLLOUT_MAX_CONTEXT_LEN explicitly." >&2
  exit 1
}
if [[ -z "${ROLLOUT_MAX_PROMPT_LEN:-}" ]]; then
  ROLLOUT_MAX_PROMPT_LEN=$((ROLLOUT_MAX_CONTEXT_LEN - ROLLOUT_MAX_RESPONSE_LEN))
fi
validate_rollout_lengths
echo "Rollout length limits: context=${ROLLOUT_MAX_CONTEXT_LEN}, prompt<=${ROLLOUT_MAX_PROMPT_LEN}, response<=${ROLLOUT_MAX_RESPONSE_LEN}"

LR="${LR:-5e-7}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-20}"
CLIP_GRAD="${CLIP_GRAD:-0.5}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=0
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1

PLUGIN_PYTHONPATH="${REPO_ROOT}/my_exps/opd-397-32b"
RUNTIME_PYTHONPATH="${MEGATRON_PYTHONPATH:-/root/Megatron-LM/}:${PLUGIN_PYTHONPATH}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

CKPT_PRUNER_PID=""
if (( MAX_CHECKPOINTS_TO_KEEP > 0 )); then
  bash "${PRUNE_CKPT_SCRIPT}" \
    --ckpt-root "${STUDENT_SAVE_PATH}" \
    --keep "${MAX_CHECKPOINTS_TO_KEEP}" \
    --interval-sec "${CHECKPOINT_PRUNE_INTERVAL_SEC}" &
  CKPT_PRUNER_PID=$!
fi

cleanup() {
  if [[ -n "${CKPT_PRUNER_PID}" ]]; then
    kill "${CKPT_PRUNER_PID}" >/dev/null 2>&1 || true
    wait "${CKPT_PRUNER_PID}" 2>/dev/null || true
  fi
  cleanup_distill_temp_file
}
trap cleanup EXIT

LOAD_ARGS=()
if [[ -n "${STUDENT_LOAD_PATH}" ]]; then
  LOAD_ARGS+=(--load "${STUDENT_LOAD_PATH}")
  LOAD_ARGS+=(--no-load-optim)
fi

build_distill_args

echo "Distillation mode: ${DISTILL_LOSS_MODE} (topk=${OPD_TOP_LOGPROBS_NUM}, mixed_weight=${OPD_MIXED_KL_WEIGHT}, jsd_beta=${OPD_JSD_BETA}, coef=${OPD_DISTILL_COEF})"

DATASET_KEY_ARGS=(
  --input-key "${INPUT_KEY:-prompt}"
)
if [[ -n "${LABEL_KEY:-}" ]]; then
  DATASET_KEY_ARGS+=(--label-key "${LABEL_KEY}")
fi
if [[ -n "${METADATA_KEY:-}" ]]; then
  DATASET_KEY_ARGS+=(--metadata-key "${METADATA_KEY}")
fi

ray job submit --address="${RAY_JOB_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${TRAIN_PY_PATH}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${STUDENT_HF_CHECKPOINT_PATH}" \
  --ref-load "${STUDENT_REF_LOAD_PATH}" \
  "${LOAD_ARGS[@]}" \
  --save "${STUDENT_SAVE_PATH}" \
  --save-interval "${SAVE_INTERVAL:-100}" \
  --prompt-data "${PROMPT_DATA_PATH}" \
  "${DATASET_KEY_ARGS[@]}" \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout 1 \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" \
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}" \
  --rollout-top-p "${ROLLOUT_TOP_P}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL:-1}" \
  --balance-data \
  --optimizer adam \
  --lr "${LR}" \
  --lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --lr-decay-style constant \
  --clip-grad "${CLIP_GRAD}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --adam-beta1 "${ADAM_BETA1:-0.9}" \
  --adam-beta2 "${ADAM_BETA2:-0.98}" \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}" \
  --use-kl-loss \
  --kl-loss-coef "${KL_LOSS_COEF:-0.0}" \
  --kl-loss-type low_var_kl \
  --entropy-coef "${ENTROPY_COEF:-0.0}" \
  "${DISTILL_ARGS[@]}" \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" \
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}" \
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}" \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --sglang-mem-fraction-static "${ROLLOUT_SGLANG_MEM_FRACTION_STATIC:-0.7}" \
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 "${SGLANG_CUDA_GRAPH_BS_MAX:-256}") \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT:-slime-opd}" \
  --wandb-group "${WANDB_GROUP:-opd-397-32b-student-async-distill}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --rm-url "${TEACHER_URL}"
