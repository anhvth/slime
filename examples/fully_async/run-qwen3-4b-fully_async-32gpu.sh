#!/bin/bash
# 32-GPU version (4 nodes x 8 GPUs each)
# Assumes Ray cluster is already running across all 4 nodes.
# Setup: on head node run:  ray start --head --node-ip-address <NODE0_IP> --num-gpus 8
#        on each worker run: ray start --address <NODE0_IP>:6379 --num-gpus 8
# Verify with: ray status

set -ex

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/qwen3-4b-fully-async-32gpu_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

echo "Starting training at $(date)"
echo "Log file: $LOG_FILE"
echo "====================================="

# Redirect stdout and stderr to log file while also showing in terminal
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

HF_CKPT=/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-Instruct-2507

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}_torch_dist"
   --load "${HF_CKPT}_slime_${TIMESTAMP}/"
   --save "${HF_CKPT}_slime_${TIMESTAMP}/"
   --save-interval 20
)

PROMPT_SET="${SCRIPT_DIR}/../../datasets/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   # Scaled up 4x from the 8-GPU version
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 960
   --balance-data
)

PERF_ARGS=(
   # TP=2 is sufficient for 4B model; keeps inter-GPU communication low
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name qwen3-4b-fully-async
   --tb-experiment-name 32gpu
)

# Load environment variables from .env file
if [ -f "${SCRIPT_DIR}/../../.env" ]; then
    export $(cat "${SCRIPT_DIR}/../../.env" | xargs)
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group fully-async-32gpu
   --wandb-key ${WANDB_API_KEY}
)

# Ray cluster is already running — do NOT call ray start here.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Resource split:
#   Actor (Megatron training): 3 nodes x 8 GPUs = 24 GPUs
#   Rollout (SGLang inference):                    8 GPUs
#   Total:                                        32 GPUs

# W&B Configuration:
# To enable W&B logging:
# 1. Uncomment --use-wandb in WANDB_ARGS above
# 2. Set your W&B API key: export WANDB_API_KEY=your_api_key_here
# 3. Set custom project/group names as needed
# 4. Optional: Add --wandb-team your_team_name if using teams
# 5. Optional: Set --wandb-dir for custom log directory

echo "Submitting job to Ray cluster..."
echo "MASTER_ADDR: ${MASTER_ADDR}"
ray job submit --address="http://${MASTER_ADDR}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PWD}/train_async.py \
   --actor-num-nodes 3 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${WANDB_ARGS[@]}

echo "Training job submitted successfully!"
echo "Log file: $LOG_FILE"
echo "Monitor training with: tail -f $LOG_FILE"
