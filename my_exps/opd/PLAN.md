# OPD 397B -> 32B Plan

## Objective
Train a `Qwen3-32B` student (finetuned, vocab-converted to Qwen3.5) with on-policy distillation
from teacher `Qwen/Qwen3.5-397B-A17B-FP8` via remote SGLang (`--rm-url`).

The teacher (Qwen3.5) and student (Qwen3) have **different vocabularies**.
We solve this by converting the student's `embed_tokens` and `lm_head` to Qwen3.5 vocab space
offline, using the shared-token mapping (`mapping_qwen_35.csv`). After conversion, teacher
logprobs can be used directly — no runtime mapping needed.

## Teacher
- Model: `Qwen/Qwen3.5-397B-A17B-FP8`
- **Hosted at: `worker-15:13141`** (SGLang)

## Student
- Source checkpoint (Qwen3-32B finetune): `~/home-trained-model/Stage3_SFT_Epoch3/`
- Vocab-converted checkpoint: `$MODEL_HOME/Qwen3-32B-as-Qwen35/`
- Megatron config: `scripts/models/qwen3-32B-as-qwen35.sh` (vocab_size=248320)

## Modes
1. Normal mode
- Teacher: `Qwen/Qwen3.5-397B-A17B-FP8` on `worker-15:13141`
- Student: `Qwen3-32B-as-Qwen35` (vocab-converted)

2. Debug mode (`--debug`)
- Teacher: `Qwen3-4B`
- Student: `Qwen3-4B`
- Same GPU topology, Ray layout, and training knobs as normal mode.

## Ready Path In Repo
- `docs/en/advanced/dedicated-teacher-server.md`
- `examples/on_policy_distillation/on_policy_distillation.py`
- `examples/on_policy_distillation/run-qwen3-8B-opd.sh`

## Local Paths (worker-15)
- Teacher 397B: `~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8`
- Teacher debug 4B: `~/ckpt/hf_models/Qwen/Qwen3-4B`
- Student source: `~/home-trained-model/Stage3_SFT_Epoch3/`
- Student converted: `~/ckpt/hf_models/Qwen/Qwen3-32B-as-Qwen35/`

## Scripts In This Folder
- `my_exps/opd/serve_teacher.sh` — start teacher SGLang server
- `my_exps/opd/train_student.sh` — launch OPD training
- `my_exps/opd/convert_qwen3_to_qwen35_vocab.py` — one-time vocab conversion
- `my_exps/opd/mapping_qwen_35.csv` — shared-token mapping (131612 pairs)

## Runbook

### 0. Convert student vocab (one-time)
```bash
cd ~/slime
python my_exps/opd/convert_qwen3_to_qwen35_vocab.py \
    --src ~/home-trained-model/Stage3_SFT_Epoch3 \
    --out ~/ckpt/hf_models/Qwen/Qwen3-32B-as-Qwen35 \
    --teacher-tokenizer ~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
    --csv my_exps/opd/mapping_qwen_35.csv
```

### 1. Teacher is already running on worker-15:13141.
Validate endpoint from trainer node:
```bash
curl -sf http://worker-15:13141/health_generate
curl http://worker-15:13141/get_model_info
```

### 2. Start OPD training on trainer node.
```bash
cd ~/slime
bash my_exps/opd/train_student.sh
```
(Default TEACHER_HOST is `worker-15`, no override needed.)

## Debug Smoke Test
1. Start debug teacher.
```bash
cd ~/slime
bash my_exps/opd/serve_teacher.sh --debug
```

2. Run debug student.
```bash
cd ~/slime
TEACHER_HOST=127.0.0.1 \
  bash my_exps/opd/train_student.sh --debug
```

## Important Defaults
- `MODEL_HOME=$HOME/ckpt/hf_models/Qwen`
- `DATA_HOME=$HOME/ckpt`
- `TEACHER_HOST=worker-15`
- `TEACHER_PORT=13141`
- `TENSOR_MODEL_PARALLEL_SIZE=8`
- `ACTOR_NUM_GPUS_PER_NODE=8`
- `ROLLOUT_NUM_GPUS_PER_ENGINE=8`

## Override Examples
```bash
TEACHER_TP=8 TEACHER_MEM_FRACTION_STATIC=0.8 \
  bash my_exps/opd/serve_teacher.sh
```

```bash
PROMPT_DATA=$HOME/ckpt/dapo-math-17k/dapo-math-17k.jsonl \
EVAL_PROMPT_DATA=$HOME/ckpt/aime-2024/aime-2024.jsonl \
  bash my_exps/opd/train_student.sh
```
