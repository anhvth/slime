cd /home/anhvth8/projects/slime

# =============================================================================
# 32B Model
# =============================================================================

# Step 1: Convert vocab
python3 my_exps/opd-397-32b/convert_qwen3_to_qwen35_vocab.py \
    --src      /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3 \
    --out      /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35 \
    --teacher-tokenizer /home/anhvth8/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
    --csv      my_exps/opd-397-32b/mapping_qwen_35.csv

# Step 2: Convert to torch_dist format for training
source scripts/models/qwen3-32B-as-qwen35.sh && \
    PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
        --hf-checkpoint /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35 \
        --save          /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35_torch_dist

# =============================================================================
# 4B Model (debug)
# =============================================================================

# Step 1: Convert vocab
python3 my_exps/opd-397-32b/convert_qwen3_to_qwen35_vocab.py \
    --src      /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B \
    --out      /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35 \
    --teacher-tokenizer /home/anhvth8/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
    --csv      my_exps/opd-397-32b/mapping_qwen_35.csv

# Step 2: Convert to torch_dist format for training
source scripts/models/qwen3-4B-as-qwen35.sh && \
    PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
        --hf-checkpoint /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35 \
        --save          /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35_torch_dist