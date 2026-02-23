#!/usr/bin/env bash
set -euo pipefail

LM_HEAD_ONLY=0
if [[ "${1:-}" == "--lm_head_only" ]]; then
  LM_HEAD_ONLY=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--lm_head_only]"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export PYTHONBUFFERED=1

source "${REPO_ROOT}/scripts/models/qwen3-32B-as-qwen35.sh"

STUDENT_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT:-/home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35}"
STUDENT_REF_LOAD="${STUDENT_REF_LOAD:-${STUDENT_HF_CHECKPOINT}_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/home/anhvth8/projects/SFT/data/SFT_merged_2.9M}"
INPUT_KEY="${INPUT_KEY:-messages}"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/opd-397-32b}"
if [[ ${LM_HEAD_ONLY} -eq 1 ]]; then
  SAVE_PATH_DEFAULT="${SAVE_ROOT}/sft_qwen35_lm_head_only"
else
  SAVE_PATH_DEFAULT="${SAVE_ROOT}/sft_qwen35_full"
fi
SAVE_PATH="${SAVE_PATH:-${SAVE_PATH_DEFAULT}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-7}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-112}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-112}"
NUM_EPOCH="${NUM_EPOCH:-1}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}"

[[ -e "${STUDENT_HF_CHECKPOINT}" ]] || { echo "Missing HF checkpoint: ${STUDENT_HF_CHECKPOINT}" >&2; exit 1; }
[[ -e "${PROMPT_DATA}" ]] || { echo "Missing prompt data path: ${PROMPT_DATA}" >&2; exit 1; }

HAS_REF_LOAD=0
if [[ -e "${STUDENT_REF_LOAD}" ]]; then
  HAS_REF_LOAD=1
fi

LOAD_PATH="${LOAD_PATH:-}"
if [[ -z "${LOAD_PATH}" ]]; then
  if [[ ${HAS_REF_LOAD} -eq 1 ]]; then
    LOAD_PATH="${STUDENT_REF_LOAD}"
  else
    LOAD_PATH="${STUDENT_HF_CHECKPOINT}"
  fi
fi
[[ -e "${LOAD_PATH}" ]] || { echo "Missing load checkpoint path: ${LOAD_PATH}" >&2; exit 1; }

if [[ -z "${MEGATRON_TO_HF_MODE:-}" ]]; then
  if [[ ${HAS_REF_LOAD} -eq 1 ]]; then
    MEGATRON_TO_HF_MODE="raw"
  else
    MEGATRON_TO_HF_MODE="bridge"
  fi
fi
if [[ ${HAS_REF_LOAD} -eq 0 && "${MEGATRON_TO_HF_MODE}" != "bridge" ]]; then
  echo "Ref checkpoint not found at ${STUDENT_REF_LOAD}; forcing --megatron-to-hf-mode bridge for HF load."
  MEGATRON_TO_HF_MODE="bridge"
fi

if (( ROLLOUT_BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
  echo "For SFT, rollout_batch_size should equal global_batch_size. Got ${ROLLOUT_BATCH_SIZE} vs ${GLOBAL_BATCH_SIZE}." >&2
  exit 1
fi

WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
if (( WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Invalid parallelism: world_size=${WORLD_SIZE} is not divisible by TP=${TENSOR_MODEL_PARALLEL_SIZE}" >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}" "${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/train_sft_qwen35_hf_$(date +%Y%m%d_%H%M%S).log}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=0
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PYTHONPATH:-/root/Megatron-LM/}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_HF_CHECKPOINT}"
  --load "${LOAD_PATH}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)
if [[ ${HAS_REF_LOAD} -eq 1 ]]; then
  CKPT_ARGS+=(--ref-load "${STUDENT_REF_LOAD}")
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA}"
  --input-key "${INPUT_KEY}"
  --metadata-key "${METADATA_KEY:-metadata}"
  --tool-key "${TOOL_KEY:-tools}"
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

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
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

EXTRA_TRAIN_ARGS=()
if [[ ${LM_HEAD_ONLY} -eq 1 ]]; then
  EXTRA_TRAIN_ARGS+=(--only-train-params-name-list "output_layer|lm_head")
fi

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--use-wandb)
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT:-slime-sft}")
  WANDB_ARGS+=(--wandb-group "${WANDB_GROUP:-qwen35-sft}")
  if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
fi

echo "Submitting SFT Ray job to ${RAY_JOB_ADDRESS}"
echo "  model: ${STUDENT_HF_CHECKPOINT}"
if [[ ${HAS_REF_LOAD} -eq 1 ]]; then
  echo "  ref-load: ${STUDENT_REF_LOAD}"
else
  echo "  ref-load: (not found, optional for SFT)"
fi
echo "  load: ${LOAD_PATH}"
echo "  megatron_to_hf_mode: ${MEGATRON_TO_HF_MODE}"
echo "  prompt-data: ${PROMPT_DATA}"
echo "  save: ${SAVE_PATH}"
echo "  actors: ${ACTOR_NUM_NODES} nodes x ${ACTOR_NUM_GPUS_PER_NODE} GPUs"
echo "  lm_head_only: ${LM_HEAD_ONLY}"
echo "Logging to: ${LOG_FILE}"

ray job submit --address="${RAY_JOB_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${REPO_ROOT}/train_async.py" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
