# OPD 397B -> 32B Plan

## Objective
Train a `Qwen3-32B` student with on-policy distillation from teacher `Qwen/Qwen3.5-397B-A17B-FP8` via remote SGLang (`--rm-url`).

## Modes
1. Normal mode
- Teacher: `Qwen/Qwen3.5-397B-A17B-FP8`
- Student: `Qwen3-32B`

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
- Student 32B: `~/ckpt/hf_models/Qwen/Qwen3-32B`

## Scripts In This Folder
- `my_exps/opd-397-32b/serve_teacher.sh`
- `my_exps/opd-397-32b/train_student.sh`

## Runbook
1. Start teacher service on teacher node.
```bash
cd ~/slime
bash my_exps/opd-397-32b/serve_teacher.sh
```

2. Validate endpoint from trainer node.
```bash
curl -sf http://<teacher-host>:13141/health_generate
curl http://<teacher-host>:13141/get_model_info
```

3. Start OPD training on trainer node.
```bash
cd ~/slime
TEACHER_URL=http://<teacher-host>:13141/generate \
  bash my_exps/opd-397-32b/train_student.sh
```

## Debug Smoke Test
1. Start debug teacher.
```bash
cd ~/slime
bash my_exps/opd-397-32b/serve_teacher.sh --debug
```

2. Run debug student.
```bash
cd ~/slime
TEACHER_URL=http://<teacher-host>:13141/generate \
  bash my_exps/opd-397-32b/train_student.sh --debug
```

## Important Defaults
- `MODEL_HOME=$HOME/ckpt/hf_models/Qwen`
- `DATA_HOME=$HOME/ckpt`
- `TEACHER_PORT=13141`
- `TENSOR_MODEL_PARALLEL_SIZE=8`
- `ACTOR_NUM_GPUS_PER_NODE=8`
- `ROLLOUT_NUM_GPUS_PER_ENGINE=8`

## Override Examples
```bash
TEACHER_TP=8 TEACHER_MEM_FRACTION_STATIC=0.8 \
  bash my_exps/opd-397-32b/serve_teacher.sh
```

```bash
TEACHER_URL=http://worker-15:13141/generate \
PROMPT_DATA=$HOME/ckpt/dapo-math-17k/dapo-math-17k.jsonl \
EVAL_PROMPT_DATA=$HOME/ckpt/aime-2024/aime-2024.jsonl \
  bash my_exps/opd-397-32b/train_student.sh
```
