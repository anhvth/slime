#!/bin/bash
set -euo pipefail

cd /home/anhvth8/projects/slime
export PYTHONBUFFERED=1

source /home/anhvth8/projects/slime/scripts/models/qwen3-32B-as-qwen35.sh

TEACHER_URL="${TEACHER_URL:-http://worker-15:13141/generate}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PYTHONPATH:-/root/Megatron-LM/}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="${RAY_JOB_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 /home/anhvth8/projects/slime/train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES:-3}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35 \
  --ref-load /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35_torch_dist \
  --load /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35_slime \
  --save /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35_slime \
  --save-interval 20 \
  --prompt-data /home/anhvth8/projects/slime/datasets/dapo-math-17k.jsonl \
  --input-key prompt \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout 300 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 63 \
  --use-dynamic-global-batch-size \
  --balance-data \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --advantage-estimator grpo \
  --use-opd \
  --opd-type sglang \
  --opd-kl-coef 1.0 \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --tensor-model-parallel-size 8 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 16384 \
  --rollout-num-gpus-per-engine 8 \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func \
  --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards \
  --rm-url "${TEACHER_URL}"
