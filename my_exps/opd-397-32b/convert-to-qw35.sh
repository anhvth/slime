cd /home/anhvth8/projects/slime
# For 32B Model
python3 my_exps/opd-397-32b/convert_qwen3_to_qwen35_vocab.py \
    --src /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3 \
    --out /home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35 \
    --teacher-tokenizer /home/anhvth8/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
    --csv my_exps/opd-397-32b/mapping_qwen_35.csv

# For 4B Model (debug)
python3 my_exps/opd-397-32b/convert_qwen3_to_qwen35_vocab.py \
    --src /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B \
    --out /home/anhvth8/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35 \
    --teacher-tokenizer /home/anhvth8/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
    --csv my_exps/opd-397-32b/mapping_qwen_35.csv