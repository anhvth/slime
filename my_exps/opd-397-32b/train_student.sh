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
    DEBUG_REF_FALLBACK="${MODEL_HOME}/Qwen3-4B_torch_dist"
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


TEACHER_HOST="${TEACHER_HOST:-worker-15}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}"

TEACHER_BASE="${TEACHER_URL%/generate}"
if [[ "${TEACHER_BASE}" == "${TEACHER_URL}" ]]; then
  echo "TEACHER_URL must end with /generate"
  exit 1
fi

curl -sf "${TEACHER_BASE}/health_generate" >/dev/null
curl -sf "${TEACHER_BASE}/get_model_info" >/dev/null

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_HF_CHECKPOINT_PATH}"
  --ref-load "${STUDENT_REF_LOAD_PATH}"
  --load "${STUDENT_LOAD:-${STUDENT_HF_DEFAULT}_slime}"
  --save "${STUDENT_SAVE:-${STUDENT_HF_DEFAULT}_slime}"
  --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA:-${REPO_ROOT}/datasets/dapo-math-17k.jsonl}"
  --input-key "${INPUT_KEY:-prompt}"
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT:-300}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
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
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-8}"
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
  --actor-num-nodes "${ACTOR_NUM_NODES:-1}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}"
  --colocate
)

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
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}"
