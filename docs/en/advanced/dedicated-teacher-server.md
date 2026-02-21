# Dedicated Teacher Server (One Path)

If your teacher is very large, run it as a **dedicated serving system** and keep training separate.

This is the only recommended path in this guide:

- Teacher runs on an external SGLang server (`--opd-type sglang`)
- Student trains in slime and queries teacher logprobs over HTTP (`--rm-url`)
- Teacher stays in serving format (HF/SGLang-compatible), not Megatron dist checkpoints

## Non-Negotiable Rules

1. Do **not** load a huge teacher inside the training process.
2. Do **not** convert the teacher to torch dist just for OPD serving.
3. Do **not** colocate teacher with student training GPUs in production runs.

## Architecture

- Teacher server: long-lived process, pinned GPU set, stable endpoint.
- Training job: calls teacher endpoint for logprobs during rollout.
- Contract: only network API between training and teacher.

This gives predictable memory behavior and avoids training instability from teacher lifecycle issues.

## Canonical Setup

### 1) Launch the teacher server

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path /models/teacher-hf \
  --host 0.0.0.0 \
  --port 13141 \
  --tp 4 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.6
```

Health checks:

```bash
curl -sf http://<teacher-host>:13141/health_generate
curl http://<teacher-host>:13141/get_model_info
```

### 2) Train the student with OPD over SGLang

Use these OPD settings in your slime run:

```bash
--use-opd \
--opd-type sglang \
--opd-kl-coef 1.0 \
--custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func \
--custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards \
--rm-url http://<teacher-host>:13141/generate
```

Reference script: `examples/on_policy_distillation/run-qwen3-8B-opd.sh`.

## Operational Policy

- Treat teacher server like infrastructure, not an experiment process.
- Keep endpoint stable for the full training run.
- Version teacher weights and server args explicitly.
- Fail fast if health check fails; never silently continue without teacher.

## Bottom Line

For very large teachers, the correct design is:

- **teacher-as-a-service**
- **student-only training process**
- **OPD via `sglang` endpoint**

Anything else is unnecessary complexity for this use case.
