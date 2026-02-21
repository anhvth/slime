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
STUDENT_LOAD_PATH="${STUDENT_LOAD:-${STUDENT_SAVE_PATH}}"
mkdir -p "${OUTPUT_ROOT}"

[[ -e "${STUDENT_HF_CHECKPOINT_PATH}" ]] || {
  echo "Missing student HF checkpoint path: ${STUDENT_HF_CHECKPOINT_PATH}" >&2
  exit 1
}
[[ -e "${STUDENT_REF_LOAD_PATH}" ]] || {
  echo "Missing student ref-load path: ${STUDENT_REF_LOAD_PATH}" >&2
  exit 1
}

TEACHER_HOST="${TEACHER_HOST:-worker-15}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
TEACHER_BASE="${TEACHER_URL%/generate}"
[[ "${TEACHER_BASE}" != "${TEACHER_URL}" ]] || { echo "TEACHER_URL must end with /generate" >&2; exit 1; }

curl -sf "${TEACHER_BASE}/health_generate" >/dev/null
curl -sf "${TEACHER_BASE}/get_model_info" >/dev/null

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}"

# Opinionated async layout for a 24-GPU cluster: 8 train + 16 rollout.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
(( WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE == 0 )) || {
  echo "Invalid parallelism: world_size=${WORLD_SIZE} not divisible by TP=${TENSOR_MODEL_PARALLEL_SIZE}" >&2
  exit 1
}
DP_SIZE=$((WORLD_SIZE / TENSOR_MODEL_PARALLEL_SIZE))

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-96}"
TOTAL_SAMPLES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
(( GLOBAL_BATCH_SIZE % DP_SIZE == 0 )) || {
  echo "Invalid batching: global_batch_size=${GLOBAL_BATCH_SIZE} not divisible by dp_size=${DP_SIZE}" >&2
  exit 1
}
(( TOTAL_SAMPLES == GLOBAL_BATCH_SIZE )) || {
  echo "Opinionated async script expects one train step per rollout:" >&2
  echo "  rollout_batch_size*n_samples_per_prompt=${TOTAL_SAMPLES} != global_batch_size=${GLOBAL_BATCH_SIZE}" >&2
  echo "Override ROLLOUT_BATCH_SIZE/N_SAMPLES_PER_PROMPT/GLOBAL_BATCH_SIZE to match." >&2
  exit 1
}

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
  --load "${STUDENT_LOAD_PATH}" \
  --save "${STUDENT_SAVE_PATH}" \
  --save-interval "${SAVE_INTERVAL:-20}" \
  --prompt-data "${PROMPT_DATA:-${REPO_ROOT}/datasets/dapo-math-17k.jsonl}" \
  --input-key "${INPUT_KEY:-prompt}" \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT:-300}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout 1 \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL:-5}" \
  --balance-data \
  --optimizer adam \
  --lr "${LR:-1e-6}" \
  --lr-decay-style constant \
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
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}" \
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
