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
  - `my_exps/opd/tests/test_teacher_api_contract.py`
  - `my_exps/opd/tests/probe_teacher_api.py`
- Added top-k reward plugin:
  - `my_exps/opd/opd_topk_reward_plugin.py`
- Added top-k custom loss plugin:
  - `my_exps/opd/opd_topk_loss_plugin.py`
- Added parser + unit tests:
  - `my_exps/opd/opd_topk_parser.py`
  - `my_exps/opd/tests/test_topk_parser.py`
- Wired mode switch in:
  - `my_exps/opd/train_student_async_distill.sh`
  - `my_exps/opd/train_student_async_distill_lib.sh`
- Added privileged JSD wrapper launcher:
  - `my_exps/opd/train_student_async_jsd_with_privileged_infomation.sh`
- Added cross-tokenizer launcher (native Qwen3 student, no vocab conversion):
  - `my_exps/opd/train_student_async_cross_tokenizer.sh`

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

Set these env vars when running `my_exps/opd/train_student_async_distill.sh`:

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
- `OPD_CROSS_TOKENIZER_ENABLE` (default: `0`; when `1`, enable GOLD-style cross-tokenizer grouping for `fkl`/`mixed`/`jsd`)
- `OPD_TEACHER_TOKENIZER_PATH` (required when cross-tokenizer enabled)
- `OPD_STUDENT_TOKENIZER_PATH` (default: empty; fallback to `--hf-checkpoint`)
- `OPD_RM_CONNECT_TIMEOUT_S` (default: `2.0`)
- `OPD_RM_READ_TIMEOUT_S` (default: `120.0`)
- `OPD_RM_TOTAL_TIMEOUT_S` (default: `180.0`)
- `OPD_RM_MAX_CONNECTIONS` (default: `512`)
- `OPD_RM_MAX_CONNECTIONS_PER_HOST` (default: `256`)
- `OPD_DEBUG_DUMP_ENABLE` (default: `1`)
- `OPD_DEBUG_DUMP_DIR` (default: empty -> `${STUDENT_SAVE}/distill_debug_dumps`)
- `OPD_DEBUG_DUMP_MAX_TOTAL_MB` (default: `5120`)
- `OPD_DEBUG_DUMP_MAX_FILES` (default: `20000`)
- `OPD_DEBUG_DUMP_MAX_FILE_MB` (default: `64`, soft cap)
- `OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE` (default: `8`)
- `OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE` (default: `0`, meaning all response positions)
- `OPD_DEBUG_DUMP_SEED` (default: `${SEED:-1234}`)

Examples:

```bash
# Default existing behavior (unchanged)
DISTILL_LOSS_MODE=rkl bash my_exps/opd/train_student_async_distill.sh

# Forward KL with teacher top-16
DISTILL_LOSS_MODE=fkl OPD_TOP_LOGPROBS_NUM=16 OPD_DISTILL_COEF=1.0 \
  bash my_exps/opd/train_student_async_distill.sh

# Mixed KL
DISTILL_LOSS_MODE=mixed OPD_TOP_LOGPROBS_NUM=16 OPD_MIXED_KL_WEIGHT=0.5 \
  OPD_DISTILL_COEF=1.0 bash my_exps/opd/train_student_async_distill.sh

# Top-k renormalized JSD
DISTILL_LOSS_MODE=jsd OPD_TOP_LOGPROBS_NUM=16 OPD_JSD_BETA=0.5 \
  OPD_DISTILL_COEF=1.0 bash my_exps/opd/train_student_async_distill.sh

# Privileged top-k JSD wrapper (defaults to jsd + privileged enabled)
bash my_exps/opd/train_student_async_jsd_with_privileged_infomation.sh

# Cross-tokenizer top-k distillation (Qwen3.5 teacher -> native Qwen3 student)
bash my_exps/opd/train_student_async_cross_tokenizer.sh --loss jsd
```

## Cross-tokenizer (no vocab conversion)

Use this launcher to distill from a Qwen3.5 teacher into native Qwen3 checkpoints without `*-as-qwen35` conversion:

```bash
# 32B default student:
#   ~/home-trained-model/Stage3_SFT_Epoch3/
# 4B debug student:
#   ~/ckpt/hf_models/Qwen/Qwen3-4B
bash my_exps/opd/train_student_async_cross_tokenizer.sh --loss fkl
```

Defaults set by this launcher:

- `OPD_CROSS_TOKENIZER_ENABLE=1`
- `OPD_TEACHER_TOKENIZER_PATH=~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8`
- `MODEL_CONFIG_REL_DEBUG=scripts/models/qwen3-4B.sh`
- `MODEL_CONFIG_REL_TRAIN=scripts/models/qwen3-32B.sh`
- `STUDENT_HF_DEFAULT_DEBUG=~/ckpt/hf_models/Qwen/Qwen3-4B`
- `STUDENT_HF_DEFAULT_TRAIN=~/home-trained-model/Stage3_SFT_Epoch3/`
- `RESUME_MODEL_ROOT_DEBUG=~/ckpt/hf_models/Qwen/Qwen3-4B`
- `RESUME_MODEL_ROOT_TRAIN=~/home-trained-model/Stage3_SFT_Epoch3/`

This avoids `*-As-Qwen35*` resume tags when you are training native Qwen3 cross-tokenizer runs.

Cross-mode launcher/tokenizer precedence now is:

1. `STUDENT_HF_CHECKPOINT` (explicit env override) if provided.
2. In cross mode (`OPD_CROSS_TOKENIZER_ENABLE=1`): native defaults
   `STUDENT_HF_DEFAULT_DEBUG` / `STUDENT_HF_DEFAULT_TRAIN`.
3. Non-cross fallback: `RESUME_HF_DIR`.

In cross mode, when `OPD_STUDENT_TOKENIZER_PATH` is empty, it is auto-set to the resolved
`STUDENT_HF_CHECKPOINT_PATH` before custom config generation.

Strict preflight in `train_student_async_distill.sh` now aborts before Ray submit when:

- `MODEL_ARGS --vocab-size` (from `MODEL_CONFIG_REL_*`) does not match
  `${STUDENT_HF_CHECKPOINT_PATH}/config.json:vocab_size`.

Mismatch error prints both detected values and the exact env/model-config knobs to fix.

## Live viewer tokenizer resolution (legacy dumps)

`debug_dump_live_server_v2.py` now uses dump recipe effective fields first:

- `opd_student_tokenizer_path_effective`
- `opd_teacher_tokenizer_path_effective`
- `hf_checkpoint`

For legacy dumps without these fields, it backfills by token-id capacity:

- student side requires capacity for max id seen in
  `student_input_ids` and `teacher_topk_token_ids`
- teacher side requires capacity for max id seen in `teacher_input_ids`

Tokenizers whose `len(tokenizer)` cannot represent those ids are rejected automatically.
The UI summary shows required max ids, selected tokenizer capacities, and warnings when
capacity fallback was needed.

Manual override examples:

```bash
python my_exps/opd/debug_dump_live_server_v2.py \
  --runs-root outputs/opd-397-32b \
  --student-tokenizer-path ~/ckpt/hf_models/Qwen/Qwen3-4B-As-Qwen35 \
  --teacher-tokenizer-path ~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8
```

## Terminal dump inspector

Use this CLI tool to inspect a single `distill_debug_*.pt` directly in terminal:

```bash
python my_exps/opd/debug_terminal.py /path/to/distill_debug_*.pt
```

Optional overrides:

```bash
python my_exps/opd/debug_terminal.py /path/to/distill_debug_*.pt \
  --sample 0 \
  --student-tokenizer-path ~/ckpt/hf_models/Qwen/Qwen3-4B \
  --teacher-tokenizer-path ~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8
```

## Resume from iter checkpoint with privileged 50k dataset

Use this flow to continue training from `outputs/opd/student_async/iter_0001599`
with privileged context enabled by default.

1. Convert checkpoint `iter_0001599` to HF and prepare `_torch_dist`:

```bash
bash my_exps/opd/prepare_student_async_iter_resume.sh
```

2. Launch privileged JSD async training:

```bash
bash my_exps/opd/train_student_async_jsd_with_privileged_infomation.sh
```

Wrapper defaults in this resume flow:

- `PROMPT_DATA=$REPO_ROOT/datasets/50k_prompt_for_distillation_privileged.jsonl`
- `RESUME_ITER_TAG=iter_0001599`
- `RESUME_HF_DIR=$REPO_ROOT/outputs/opd/student_async_hf/iter_0001599`
- `RESUME_DIST_DIR=$REPO_ROOT/outputs/opd/student_async_hf/iter_0001599_torch_dist`
- `STUDENT_HF_CHECKPOINT=$RESUME_HF_DIR`
- `STUDENT_REF_LOAD=$RESUME_DIST_DIR`
- `STUDENT_LOAD=$RESUME_DIST_DIR`
- `STUDENT_SAVE=$REPO_ROOT/outputs/opd/student_async_distill_privileged_from_iter_0001599`

Preflight checks in wrapper now fail fast if any required path is missing:

- prompt dataset (`PROMPT_DATA`)
- HF checkpoint directory (`STUDENT_HF_CHECKPOINT`)
- ref checkpoint (`STUDENT_REF_LOAD`)
- load checkpoint (`STUDENT_LOAD`)

Backward-compatible alias:
`my_exps/opd/train_student_async_custom_loss.sh` now forwards to `train_student_async_distill.sh`.

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
- Cross-tokenizer support is only implemented for `fkl`/`mixed`/`jsd` (not `rkl`).
- Teacher API contract assumes SGLang `input_top_logprobs` row entries like:
  - `[float_logprob, int_token_id, text_or_none]`
- Cross-tokenizer mode also requires `meta_info.input_token_logprobs` for continuation-chain merging.
- JSD mode is top-k renormalized approximation, not exact full-vocab JSD.

## Metrics

The custom loss logs:

- `train/distill_kl`
- `train/distill_kl_forward`
- `train/distill_kl_reverse`
- `train/distill_jsd`
- `train/distill_mode` (`1=fkl`, `2=mixed`, `3=jsd`)
- `train/distill_topk`
- `train/distill_valid_groups`
- `train/distill_skipped_groups`
- `train/distill_support_coverage`

## Performance debugging runbook

Use these tools while privileged JSD training is running:

1. Continuous health + bottleneck monitor (tmux-friendly):

```bash
bash my_exps/opd/monitor_privileged_training.sh
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
bash my_exps/opd/perf_tools/run_perf_report.sh
```

Outputs:

- `my_exps/opd/logs/perf_reports/<timestamp>/summary.md`
- `.../training_perf.txt`, `.../training_perf.json`
- `.../ray_gpu_snapshot.txt`, `.../ray_gpu_snapshot.json`

3. Quick direct analyzer call:

```bash
python3 my_exps/opd/perf_tools/analyze_training_perf.py \
  --log my_exps/opd/logs/training_async_distill_active.log \
  --tail-lines 8000
```

Interpretation guide:

- `wait_time_ratio` high (`>=0.65`) => train side is waiting on rollout/reward path.
- `tokens_per_gpu_per_sec` low (`<180`) + `token usage` near zero => rollout engines under-filled.
- `teacher_e2e_latency p95` high (`>10s`) => teacher endpoint tail latency bottleneck.
- `truncated_ratio` high (`>0.6`) => many responses hit max length (costly rollout).
