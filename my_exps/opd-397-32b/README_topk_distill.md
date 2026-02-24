# Top-k Distillation Extension (SGLang External Teacher)

This experiment folder now supports four distillation modes via `train_student_async_distill.sh` (original `train_student_async.sh` remains unchanged):

- `rkl` (default): existing on-policy distillation path (reverse-KL-style advantage shaping).
- `fkl`: forward KL on teacher top-k support.
- `mixed`: weighted blend of forward + reverse(top-k) KL.
- `jsd`: top-k renormalized Jensen-Shannon divergence.

It also supports optional **privileged teacher context** (teacher-only side information)
for top-k modes through `opd_topk_reward_plugin.py`.

## What changed

- Added API-contract tests for teacher server (`worker-30:13141`) in:
  - `my_exps/opd-397-32b/tests/test_teacher_api_contract.py`
  - `my_exps/opd-397-32b/tests/probe_teacher_api.py`
- Added top-k reward plugin:
  - `my_exps/opd-397-32b/opd_topk_reward_plugin.py`
- Added top-k custom loss plugin:
  - `my_exps/opd-397-32b/opd_topk_loss_plugin.py`
- Added parser + unit tests:
  - `my_exps/opd-397-32b/opd_topk_parser.py`
  - `my_exps/opd-397-32b/tests/test_topk_parser.py`
- Wired mode switch in:
  - `my_exps/opd-397-32b/train_student_async_distill.sh`
  - `my_exps/opd-397-32b/train_student_async_distill_lib.sh`
- Added privileged JSD wrapper launcher:
  - `my_exps/opd-397-32b/train_student_async_jsd_with_privileged_infomation.sh`

## Comparison with existing default OPD

- Existing `rkl` path in SLIME OPD:
  - Uses sampled student-token logprob and applies reverse-KL-style penalty in advantages.
  - Implemented through existing OPD reward/post-process + `--use-opd`.
- New `fkl` / `mixed` / `jsd` path here:
  - Uses teacher top-k token distribution from SGLang (`input_top_logprobs`).
  - Optimizes direct custom distillation loss (`--loss-type custom_loss`).
  - Keeps external teacher server workflow (no vLLM/Megatron teacher forward dependency).

## Objective definitions

Teacher provides per-token top-k rows `(log p_t(i), token_id_i)` for response positions.

Let `S` be top-k support, `p_t(i)=exp(log p_t(i))`, `p_s(i)=exp(log p_s(i))`:

- `FKL`: `sum_{i in S} p_t(i) * (log p_t(i) - log p_s(i))`
- `RKL_topk`: `sum_{i in S} p_s(i) * (log p_s(i) - log p_t(i))`
- `Mixed`: `alpha * FKL + (1 - alpha) * RKL_topk`
- `JSD_topk` (renormalized): compute `q_t`, `q_s` by renormalizing teacher/student top-k masses over `S`, then
  `JSD = beta * KL(q_t || q_m) + (1 - beta) * KL(q_s || q_m)`, with `q_m = beta*q_t + (1-beta)*q_s`.

`mixed` is not JSD: mixed combines `KL(T||S)` and `KL(S||T)` directly; JSD uses a mixture distribution `M`.

## Mode switch interface

Set these env vars when running `my_exps/opd-397-32b/train_student_async_distill.sh`:

- `DISTILL_LOSS_MODE=rkl|fkl|mixed|jsd` (default: `rkl`)
- `OPD_TOP_LOGPROBS_NUM` (default: `16`)
- `OPD_MIXED_KL_WEIGHT` (default: `0.5`, only used by `mixed`)
- `OPD_JSD_BETA` (default: `0.5`, only used by `jsd`)
- `OPD_DISTILL_COEF` (default: `1.0`, scales custom distillation loss)
- `OPD_PRIVILEGED_ENABLE` (default: `0`; when `1`, enable privileged teacher context path)
- `OPD_PRIVILEGED_METADATA_KEY` (default: `privileged_context`)
- `OPD_PRIVILEGED_FALLBACK_LABEL` (default: `1`; fallback to `label` when metadata key is missing/empty)
- `OPD_PRIVILEGED_OPEN_TAG` (default: `[PRIVILEGED_CONTEXT]`)
- `OPD_PRIVILEGED_CLOSE_TAG` (default: `[/PRIVILEGED_CONTEXT]`)
- `OPD_PRIVILEGED_TOKENIZER_PATH` (default: empty; fallback to `--hf-checkpoint`)
- `OPD_RM_CONNECT_TIMEOUT_S` (default: `2.0`)
- `OPD_RM_READ_TIMEOUT_S` (default: `120.0`)
- `OPD_RM_TOTAL_TIMEOUT_S` (default: `180.0`)
- `OPD_RM_MAX_CONNECTIONS` (default: `512`)
- `OPD_RM_MAX_CONNECTIONS_PER_HOST` (default: `256`)

Examples:

```bash
# Default existing behavior (unchanged)
DISTILL_LOSS_MODE=rkl bash my_exps/opd-397-32b/train_student_async_distill.sh

# Forward KL with teacher top-16
DISTILL_LOSS_MODE=fkl OPD_TOP_LOGPROBS_NUM=16 OPD_DISTILL_COEF=1.0 \
  bash my_exps/opd-397-32b/train_student_async_distill.sh

# Mixed KL
DISTILL_LOSS_MODE=mixed OPD_TOP_LOGPROBS_NUM=16 OPD_MIXED_KL_WEIGHT=0.5 \
  OPD_DISTILL_COEF=1.0 bash my_exps/opd-397-32b/train_student_async_distill.sh

# Top-k renormalized JSD
DISTILL_LOSS_MODE=jsd OPD_TOP_LOGPROBS_NUM=16 OPD_JSD_BETA=0.5 \
  OPD_DISTILL_COEF=1.0 bash my_exps/opd-397-32b/train_student_async_distill.sh

# Privileged top-k JSD wrapper (defaults to jsd + privileged enabled)
bash my_exps/opd-397-32b/train_student_async_jsd_with_privileged_infomation.sh
```

## Resume from iter checkpoint with privileged 50k dataset

Use this flow to continue training from `outputs/opd-397-32b/student_async/iter_0001599`
with privileged context enabled by default.

1. Convert checkpoint `iter_0001599` to HF and prepare `_torch_dist`:

```bash
bash my_exps/opd-397-32b/prepare_student_async_iter_resume.sh
```

2. Launch privileged JSD async training:

```bash
bash my_exps/opd-397-32b/train_student_async_jsd_with_privileged_infomation.sh
```

Wrapper defaults in this resume flow:

- `PROMPT_DATA=$REPO_ROOT/datasets/50k_prompt_for_distillation_privileged.jsonl`
- `RESUME_ITER_TAG=iter_0001599`
- `RESUME_HF_DIR=$REPO_ROOT/outputs/opd-397-32b/student_async_hf/iter_0001599`
- `RESUME_DIST_DIR=$REPO_ROOT/outputs/opd-397-32b/student_async_hf/iter_0001599_torch_dist`
- `STUDENT_HF_CHECKPOINT=$RESUME_HF_DIR`
- `STUDENT_REF_LOAD=$RESUME_DIST_DIR`
- `STUDENT_LOAD=$RESUME_DIST_DIR`
- `STUDENT_SAVE=$REPO_ROOT/outputs/opd-397-32b/student_async_distill_privileged_from_iter_0001599`

Preflight checks in wrapper now fail fast if any required path is missing:

- prompt dataset (`PROMPT_DATA`)
- HF checkpoint directory (`STUDENT_HF_CHECKPOINT`)
- ref checkpoint (`STUDENT_REF_LOAD`)
- load checkpoint (`STUDENT_LOAD`)

Backward-compatible alias:
`my_exps/opd-397-32b/train_student_async_forward_kl.sh` now forwards to `train_student_async_distill.sh`.

## Dataset contract for privileged context

To provide teacher-only privileged information, pass dataset keys through:

- `LABEL_KEY` -> `--label-key` (optional fallback source)
- `METADATA_KEY` -> `--metadata-key` (default: `metadata`)

Expected row shape for preferred path:

```json
{
  "prompt": "...",
  "label": "... optional ...",
  "metadata": {
    "privileged_context": "... teacher-only text ..."
  }
}
```

Resolution order in plugin:

1. `sample.metadata[opd_privileged_metadata_key]`
2. `sample.label` (if `opd_privileged_fallback_label=1`)
3. Prompt-only scoring fallback (no privileged append)

## Current limitations (v1)

- Requires `CONTEXT_PARALLEL_SIZE=1` for `fkl`/`mixed`/`jsd`.
- Teacher API contract assumes SGLang `input_top_logprobs` row entries like:
  - `[float_logprob, int_token_id, text_or_none]`
- Uses teacher top-k from `input_top_logprobs` only (ignores output-side top-logprobs).
- JSD mode is top-k renormalized approximation, not exact full-vocab JSD.

## Metrics

The custom loss logs:

- `train/distill_kl`
- `train/distill_kl_forward`
- `train/distill_kl_reverse`
- `train/distill_jsd`
- `train/distill_mode` (`1=fkl`, `2=mixed`, `3=jsd`)
- `train/distill_topk`

## Performance debugging runbook

Use these tools while privileged JSD training is running:

1. Continuous health + bottleneck monitor (tmux-friendly):

```bash
bash my_exps/opd-397-32b/monitor_privileged_training.sh
```

Useful env overrides:

- `ENABLE_PERF_PROBE=1` (default)
- `PERF_PROBE_EVERY_LOOPS=6` (emit perf analysis every ~3 min at 30s loop)
- `GPU_SNAPSHOT_EVERY_LOOPS=12`
- `PERF_ALERT_WAIT_RATIO=0.65`
- `PERF_ALERT_TPS=180`
- `PERF_ALERT_TEACHER_P95=10`

2. One-shot detailed report with artifacts:

```bash
bash my_exps/opd-397-32b/perf_tools/run_perf_report.sh
```

Outputs:

- `my_exps/opd-397-32b/logs/perf_reports/<timestamp>/summary.md`
- `.../training_perf.txt`, `.../training_perf.json`
- `.../ray_gpu_snapshot.txt`, `.../ray_gpu_snapshot.json`

3. Quick direct analyzer call:

```bash
python3 my_exps/opd-397-32b/perf_tools/analyze_training_perf.py \
  --log my_exps/opd-397-32b/logs/training_async_distill_active.log \
  --tail-lines 8000
```

Interpretation guide:

- `wait_time_ratio` high (`>=0.65`) => train side is waiting on rollout/reward path.
- `tokens_per_gpu_per_sec` low (`<180`) + `token usage` near zero => rollout engines under-filled.
- `teacher_e2e_latency p95` high (`>10s`) => teacher endpoint tail latency bottleneck.
- `truncated_ratio` high (`>0.6`) => many responses hit max length (costly rollout).
