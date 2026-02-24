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

export PYTHONBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
RAY_ADDR_UTIL="${SCRIPT_DIR}/ray_job_address_utils.sh"
[[ -f "${RAY_ADDR_UTIL}" ]] || {
  echo "Missing Ray address helper script: ${RAY_ADDR_UTIL}" >&2
  exit 1
}
# shellcheck source=/dev/null
source "${RAY_ADDR_UTIL}"

MODEL_HOME="${MODEL_HOME:-$HOME/ckpt/hf_models/Qwen}"
DATA_HOME="${DATA_HOME:-$HOME/ckpt}"

if [[ ${DEBUG} -eq 1 ]]; then
  MODEL_CONFIG_SCRIPT="${REPO_ROOT}/scripts/models/qwen3-4B-as-qwen35.sh"
  source "${MODEL_CONFIG_SCRIPT}"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT:-/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35}"
else
  MODEL_CONFIG_SCRIPT="${REPO_ROOT}/scripts/models/qwen3-32B-as-qwen35.sh"
  source "${MODEL_CONFIG_SCRIPT}"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT:-/home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35}"
fi

STUDENT_HF_CHECKPOINT_PATH="${STUDENT_HF_CHECKPOINT:-${STUDENT_HF_DEFAULT}}"
if [[ -n "${STUDENT_REF_LOAD:-}" ]]; then
  STUDENT_REF_LOAD_PATH="${STUDENT_REF_LOAD}"
else
  STUDENT_REF_LOAD_PATH="${STUDENT_HF_DEFAULT}_torch_dist"
  if [[ ${DEBUG} -eq 1 && ! -e "${STUDENT_REF_LOAD_PATH}" ]]; then
    # In debug, converted HF weights often reuse the original Qwen3-4B torch_dist reference path.
    DEBUG_REF_FALLBACK="${MODEL_HOME}/Qwen3-4B-As-Qwen35_torch_dist"
    if [[ -e "${DEBUG_REF_FALLBACK}" ]]; then
      STUDENT_REF_LOAD_PATH="${DEBUG_REF_FALLBACK}"
      echo "Using debug ref-load fallback: ${STUDENT_REF_LOAD_PATH}"
    fi
  fi
fi
DEBUG_FLAG=""
if [[ ${DEBUG} -eq 1 ]]; then
  DEBUG_FLAG=" --debug"
fi

if [[ ! -e "${STUDENT_HF_CHECKPOINT_PATH}" ]]; then
  echo "Missing student HF checkpoint path: ${STUDENT_HF_CHECKPOINT_PATH}" >&2
  echo "Use tools/convert_torch_dist_to_hf.py first if you only have torch_dist checkpoints." >&2
  exit 1
fi
if [[ ! -e "${STUDENT_REF_LOAD_PATH}" ]]; then
  echo "Missing student ref-load path: ${STUDENT_REF_LOAD_PATH}" >&2
  echo "Run this exact command to build it:" >&2
  echo "source ${MODEL_CONFIG_SCRIPT} && PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \${MODEL_ARGS[@]} --hf-checkpoint \"${STUDENT_HF_CHECKPOINT_PATH}\" --save \"${STUDENT_REF_LOAD_PATH}\"" >&2
  echo "Or override with an existing path: STUDENT_REF_LOAD=/path/to/torch_dist bash $0${DEBUG_FLAG}" >&2
  exit 1
fi

TRAIN_PY_PATH="${TRAIN_PY_PATH:-${REPO_ROOT}/train.py}"
if [[ ! -f "${TRAIN_PY_PATH}" ]]; then
  echo "Missing train entrypoint: ${TRAIN_PY_PATH}" >&2
  exit 1
fi

# Non-async layout defaults: 7 Ray nodes x 8 GPUs.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-7}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
(( WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE == 0 )) || {
  echo "Invalid parallelism: world_size=${WORLD_SIZE} not divisible by TP=${TENSOR_MODEL_PARALLEL_SIZE}" >&2
  exit 1
}
DP_SIZE=$((WORLD_SIZE / TENSOR_MODEL_PARALLEL_SIZE))

gcd() {
  local a="$1"
  local b="$2"
  while (( b != 0 )); do
    local t="$b"
    b=$((a % b))
    a="$t"
  done
  echo "$a"
}

N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
BASE_PROMPTS_PER_DP="${BASE_PROMPTS_PER_DP:-4}"
if [[ -z "${ROLLOUT_BATCH_SIZE:-}" ]]; then
  ROLLOUT_BATCH_SIZE=$((BASE_PROMPTS_PER_DP * DP_SIZE))
fi

GCD_DP_NSAMPLES="$(gcd "${DP_SIZE}" "${N_SAMPLES_PER_PROMPT}")"
ROLLOUT_BATCH_GRANULARITY=$((DP_SIZE / GCD_DP_NSAMPLES))
if (( ROLLOUT_BATCH_SIZE % ROLLOUT_BATCH_GRANULARITY != 0 )); then
  RAW_ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}"
  ROLLOUT_BATCH_SIZE=$(( (ROLLOUT_BATCH_SIZE / ROLLOUT_BATCH_GRANULARITY) * ROLLOUT_BATCH_GRANULARITY ))
  (( ROLLOUT_BATCH_SIZE > 0 )) || ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_GRANULARITY}"
  echo "Adjust rollout_batch_size ${RAW_ROLLOUT_BATCH_SIZE} -> ${ROLLOUT_BATCH_SIZE} (granularity=${ROLLOUT_BATCH_GRANULARITY} for dp_size=${DP_SIZE}, n_samples_per_prompt=${N_SAMPLES_PER_PROMPT})"
fi
TOTAL_SAMPLES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ -z "${GLOBAL_BATCH_SIZE:-}" ]]; then
  GLOBAL_BATCH_SIZE="${TOTAL_SAMPLES}"
elif (( GLOBAL_BATCH_SIZE != TOTAL_SAMPLES )); then
  echo "Adjust global_batch_size ${GLOBAL_BATCH_SIZE} -> ${TOTAL_SAMPLES} to satisfy one train step per rollout"
  GLOBAL_BATCH_SIZE="${TOTAL_SAMPLES}"
fi
(( GLOBAL_BATCH_SIZE % DP_SIZE == 0 )) || {
  echo "Invalid batching after auto-adjust: global_batch_size=${GLOBAL_BATCH_SIZE} not divisible by dp_size=${DP_SIZE}" >&2
  exit 1
}
TARGET_PROMPTS="${TARGET_PROMPTS:-200000}"
if [[ ! "${TARGET_PROMPTS}" =~ ^[0-9]+$ ]] || (( TARGET_PROMPTS <= 0 )); then
  echo "TARGET_PROMPTS must be a positive integer, got '${TARGET_PROMPTS}'" >&2
  exit 1
fi
NUM_ROLLOUT_DEFAULT=$(((TARGET_PROMPTS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/opd-397-32b}"
STUDENT_LOAD_PATH="${STUDENT_LOAD:-${OUTPUT_ROOT}/student_sync}"
STUDENT_SAVE_PATH="${STUDENT_SAVE:-${OUTPUT_ROOT}/student_sync}"
mkdir -p "${OUTPUT_ROOT}"
PROMPT_DATA_PATH="${PROMPT_DATA:-${REPO_ROOT}/datasets/200k_prompt_for_distillation.jsonl}"
if [[ ! -f "${PROMPT_DATA_PATH}" ]]; then
  echo "Missing prompt dataset: ${PROMPT_DATA_PATH}" >&2
  exit 1
fi

detect_rollout_context_len() {
  local hf_path="$1"
  local config_path="${hf_path%/}/config.json"
  [[ -f "${config_path}" ]] || return 1

  local context_len=""
  context_len="$(awk -F: '/"max_position_embeddings"[[:space:]]*:/ {gsub(/[^0-9]/, "", $2); if (length($2) > 0) {print $2; exit}}' "${config_path}")"
  if [[ -z "${context_len}" ]]; then
    context_len="$(awk -F: '/"model_max_length"[[:space:]]*:/ {gsub(/[^0-9]/, "", $2); if (length($2) > 0) {print $2; exit}}' "${config_path}")"
  fi

  [[ -n "${context_len}" ]] || return 1
  echo "${context_len}"
}

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
for len_var in ROLLOUT_MAX_CONTEXT_LEN ROLLOUT_MAX_PROMPT_LEN ROLLOUT_MAX_RESPONSE_LEN; do
  [[ "${!len_var}" =~ ^[0-9]+$ ]] || {
    echo "${len_var} must be a positive integer, got '${!len_var}'" >&2
    exit 1
  }
done
(( ROLLOUT_MAX_PROMPT_LEN > 0 )) || {
  echo "ROLLOUT_MAX_PROMPT_LEN must be > 0, got ${ROLLOUT_MAX_PROMPT_LEN}" >&2
  exit 1
}
(( ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN <= ROLLOUT_MAX_CONTEXT_LEN )) || {
  echo "Invalid rollout length limits:" >&2
  echo "  rollout_max_prompt_len(${ROLLOUT_MAX_PROMPT_LEN}) + rollout_max_response_len(${ROLLOUT_MAX_RESPONSE_LEN})" >&2
  echo "  exceeds rollout_max_context_len(${ROLLOUT_MAX_CONTEXT_LEN})" >&2
  exit 1
}
echo "Rollout length limits: context=${ROLLOUT_MAX_CONTEXT_LEN}, prompt<=${ROLLOUT_MAX_PROMPT_LEN}, response<=${ROLLOUT_MAX_RESPONSE_LEN}"


TEACHER_HOST="${TEACHER_HOST:-worker-29}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
RAY_JOB_ADDRESS="$(require_ray_job_address)"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-15}"

TEACHER_BASE="${TEACHER_URL%/generate}"
if [[ "${TEACHER_BASE}" == "${TEACHER_URL}" ]]; then
  echo "TEACHER_URL must end with /generate"
  exit 1
fi

echo "Preflight: checking teacher endpoints at ${TEACHER_BASE} ..."
curl -sf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" "${TEACHER_BASE}/health_generate" >/dev/null
curl -sf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" "${TEACHER_BASE}/get_model_info" >/dev/null
echo "Preflight: teacher endpoints OK"
echo "Using Ray Job server: ${RAY_JOB_ADDRESS}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_HF_CHECKPOINT_PATH}"
  --ref-load "${STUDENT_REF_LOAD_PATH}"
  --load "${STUDENT_LOAD_PATH}"
  --save "${STUDENT_SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA_PATH}"
  --input-key "${INPUT_KEY:-prompt}"
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout 1
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL:-5}"
  --balance-data
)

RM_ARGS=(
  --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
  --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
  --rm-url "${TEACHER_URL}"
)

EVAL_ARGS=()
if [[ -n "${EVAL_PROMPT_DATA:-}" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL:-20}"
    --eval-prompt-data aime "${EVAL_PROMPT_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-16}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-16384}"
    --eval-top-p 1
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)

GRPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
  --use-opd
  --opd-type sglang
  --opd-kl-coef "${OPD_KL_COEF:-1.0}"
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF:-0.0}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.98}"
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
  --sglang-mem-fraction-static "${ROLLOUT_SGLANG_MEM_FRACTION_STATIC:-0.7}"
  --sglang-cuda-graph-bs 1 2 4 8
)
for bs in $(seq 16 8 "${SGLANG_CUDA_GRAPH_BS_MAX:-256}"); do
  SGLANG_ARGS+=("${bs}")
done

WANDB_GROUP_DEFAULT="opd-397-32b-student"
if [[ ${DEBUG} -eq 1 ]]; then
  WANDB_GROUP_DEFAULT="${WANDB_GROUP_DEFAULT}-debug"
fi

WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT:-slime-opd}"
  --wandb-group "${WANDB_GROUP:-${WANDB_GROUP_DEFAULT}}"
)

WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS+=(--wandb-key "${WANDB_KEY_VALUE}")
fi
if [[ -n "${WANDB_HOST:-}" ]]; then
  WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
fi
if [[ -n "${WANDB_TEAM:-}" ]]; then
  WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
fi
if [[ "${WANDB_DISABLE_RANDOM_SUFFIX:-0}" == "1" ]]; then
  WANDB_ARGS+=(--disable-wandb-random-suffix)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PYTHONPATH:-/root/Megatron-LM/}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

LAYOUT_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --colocate
)

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
RUN_NAME="train_student"
if [[ ${DEBUG} -eq 1 ]]; then
  RUN_NAME="${RUN_NAME}_debug"
fi
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"
echo "Tee logging to: ${LOG_FILE}"
echo "Launch config: actor=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE} (world=${WORLD_SIZE}, dp=${DP_SIZE}, tp=${TENSOR_MODEL_PARALLEL_SIZE}), rollout_batch=${ROLLOUT_BATCH_SIZE}, n_samples=${N_SAMPLES_PER_PROMPT}, global_batch=${GLOBAL_BATCH_SIZE}"
echo "Length config: rollout_max_context_len=${ROLLOUT_MAX_CONTEXT_LEN}, rollout_max_prompt_len=${ROLLOUT_MAX_PROMPT_LEN}, rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}"
echo "Submitting Ray job to ${RAY_JOB_ADDRESS}"

ray job submit --address="${RAY_JOB_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${TRAIN_PY_PATH}" \
  "${LAYOUT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
