#!/bin/bash
# Convert Qwen3-4B-Instruct-2507 HF checkpoint to torch_dist format
# required before first training run
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

source "${SCRIPT_DIR}/../scripts/models/qwen3-4B-Instruct-2507.sh"

HF_CKPT=/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-Instruct-2507
SAVE_DIR=/home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-Instruct-2507_torch_dist

PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node 1 \
    "${SCRIPT_DIR}/../tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"
