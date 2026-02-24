# Top-k Distillation Extension (SGLang External Teacher)

This experiment folder now supports four distillation modes via `train_student_async_distill.sh` (original `train_student_async.sh` remains unchanged):

- `rkl` (default): existing on-policy distillation path (reverse-KL-style advantage shaping).
- `fkl`: forward KL on teacher top-k support.
- `mixed`: weighted blend of forward + reverse(top-k) KL.
- `jsd`: top-k renormalized Jensen-Shannon divergence.

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
```

Backward-compatible alias:
`my_exps/opd-397-32b/train_student_async_forward_kl.sh` now forwards to `train_student_async_distill.sh`.

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
