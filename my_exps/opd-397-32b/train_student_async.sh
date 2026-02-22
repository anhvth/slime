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
cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/training_async_active.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') train_student_async start ====="

export PYTHONBUFFERED=1
if [[ ${DEBUG} -eq 1 ]]; then
  source "${REPO_ROOT}/scripts/models/qwen3-4B-as-qwen35.sh"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT:-/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35}"
else
  source "${REPO_ROOT}/scripts/models/qwen3-32B-as-qwen35.sh"
  STUDENT_HF_DEFAULT="${STUDENT_HF_DEFAULT:-/home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35}"
fi

TRAIN_PY_PATH="${TRAIN_PY_PATH:-${REPO_ROOT}/train_async.py}"
[[ -f "${TRAIN_PY_PATH}" ]] || { echo "Missing train entrypoint: ${TRAIN_PY_PATH}" >&2; exit 1; }

STUDENT_HF_CHECKPOINT_PATH="${STUDENT_HF_CHECKPOINT:-${STUDENT_HF_DEFAULT}}"
STUDENT_REF_LOAD_PATH="${STUDENT_REF_LOAD:-${STUDENT_HF_CHECKPOINT_PATH}_torch_dist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/opd-397-32b}"
STUDENT_SAVE_PATH="${STUDENT_SAVE:-${OUTPUT_ROOT}/student_async}"
RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-0}"
if [[ -n "${STUDENT_LOAD:-}" ]]; then
  STUDENT_LOAD_PATH="${STUDENT_LOAD}"
elif [[ "${RESUME_FROM_SAVE}" == "1" ]]; then
  STUDENT_LOAD_PATH="${STUDENT_SAVE_PATH}"
else
  STUDENT_LOAD_PATH=""
fi
mkdir -p "${OUTPUT_ROOT}"

[[ -e "${STUDENT_HF_CHECKPOINT_PATH}" ]] || {
  echo "Missing student HF checkpoint path: ${STUDENT_HF_CHECKPOINT_PATH}" >&2
  exit 1
}
[[ -e "${STUDENT_REF_LOAD_PATH}" ]] || {
  echo "Missing student ref-load path: ${STUDENT_REF_LOAD_PATH}" >&2
  exit 1
}

MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-5}"
CHECKPOINT_PRUNE_INTERVAL_SEC="${CHECKPOINT_PRUNE_INTERVAL_SEC:-300}"

prune_old_checkpoints() {
  local ckpt_root="$1"
  local keep_n="$2"
  [[ -d "${ckpt_root}" ]] || return 0
  [[ "${keep_n}" -gt 0 ]] || return 0

  local ckpts=()
  mapfile -t ckpts < <(find "${ckpt_root}" -maxdepth 1 -mindepth 1 -type d -name 'iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' | sort)
  local count="${#ckpts[@]}"
  (( count > keep_n )) || return 0

  local delete_n=$((count - keep_n))
  for ((i = 0; i < delete_n; i++)); do
    rm -rf -- "${ckpts[$i]}"
  done
}

start_checkpoint_pruner() {
  [[ "${MAX_CHECKPOINTS_TO_KEEP}" -gt 0 ]] || return 0
  (
    while true; do
      prune_old_checkpoints "${STUDENT_SAVE_PATH}" "${MAX_CHECKPOINTS_TO_KEEP}" || true
      sleep "${CHECKPOINT_PRUNE_INTERVAL_SEC}"
    done
  ) &
  CKPT_PRUNER_PID=$!
}

stop_checkpoint_pruner() {
  if [[ -n "${CKPT_PRUNER_PID:-}" ]]; then
    kill "${CKPT_PRUNER_PID}" >/dev/null 2>&1 || true
  fi
}

TEACHER_HOST="${TEACHER_HOST:-worker-29}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
TEACHER_BASE="${TEACHER_URL%/generate}"
[[ "${TEACHER_BASE}" != "${TEACHER_URL}" ]] || { echo "TEACHER_URL must end with /generate" >&2; exit 1; }

curl -sf "${TEACHER_BASE}/health_generate" >/dev/null
curl -sf "${TEACHER_BASE}/get_model_info" >/dev/null

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}"

# Opinionated async layout for a 56-GPU cluster: 24 train (3 nodes) + 32 rollout (4 nodes).
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-3}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-32}"
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

# Opinionated async script enforces:
# 1) rollout_batch_size * n_samples_per_prompt == global_batch_size
# 2) global_batch_size divisible by dp_size
# Auto-correct to nearest valid rollout batch granularity if needed.
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

LR="${LR:-5e-7}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-20}"
CLIP_GRAD="${CLIP_GRAD:-0.5}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=0
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PYTHONPATH:-/root/Megatron-LM/}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

start_checkpoint_pruner
trap stop_checkpoint_pruner EXIT

LOAD_ARGS=()
if [[ -n "${STUDENT_LOAD_PATH}" ]]; then
  LOAD_ARGS+=(--load "${STUDENT_LOAD_PATH}")
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
  --prompt-data "${PROMPT_DATA:-${REPO_ROOT}/datasets/200k_prompt_for_distillation.jsonl}" \
  --input-key "${INPUT_KEY:-prompt}" \
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
  --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL:-5}" \
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
  --use-opd \
  --opd-type sglang \
  --opd-kl-coef "${OPD_KL_COEF:-1.0}" \
  --use-kl-loss \
  --kl-loss-coef "${KL_LOSS_COEF:-0.0}" \
  --kl-loss-type low_var_kl \
  --entropy-coef "${ENTROPY_COEF:-0.0}" \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" \
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}" \
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
  --wandb-group "${WANDB_GROUP:-opd-397-32b-student-async}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func \
  --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards \
  --rm-url "${TEACHER_URL}"
