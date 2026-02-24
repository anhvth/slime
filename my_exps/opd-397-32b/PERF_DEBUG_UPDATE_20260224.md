# Privileged JSD Perf Debug Update (2026-02-24)

## Scope

Run under analysis: `raysubmit_2cWUC5RHBLbLhVWT`  
Mode: async distill + `DISTILL_LOSS_MODE=jsd` + privileged context (`50k_prompt_for_distillation_privileged.jsonl`)

## Snapshot metrics (last ~8k log lines)

Source: `my_exps/opd-397-32b/perf_tools/analyze_training_perf.py`

- `perf/wait_time_ratio` mean: `0.726` (p95 `0.801`)
- `perf/step_time` mean: `47.33s`
- `perf/train_wait_time` mean: `34.48s`
- `perf/actor_train_time` mean: `12.84s`
- `perf/tokens_per_gpu_per_sec` mean: `149.6`
- `perf/rollout_time` mean: `41.78s`
- `rollout/truncated_ratio` mean: `0.753`
- `teacher_e2e_latency` mean: `8.57s`, p95 `23.52s`, max `25.22s`
- SGLang log `token usage` mean: `0.007`, `#queue-req` p95: `0`

Cluster-level logical usage (from `ray status` samples):

- GPU used/reserved/total around `24 / 120 / 120`

## Hypotheses and verification

1. Hypothesis: training GPUs are mostly waiting on rollout/reward pipeline.
Status: verified.
Evidence: `wait_time_ratio=0.726`, `train_wait_time` dominates `step_time`.

2. Hypothesis: rollout throughput is below what 64 rollout GPUs should deliver.
Status: verified.
Evidence: `tokens_per_gpu_per_sec=149.6` with frequent low SGLang `token usage` and zero queue backlog.

3. Hypothesis: teacher endpoint latency tail is a major contributor.
Status: verified.
Evidence: teacher `e2e_latency` p95 `23.52s`; long-tail outliers up to `25.22s`.

4. Hypothesis: generation length distribution inflates rollout cost.
Status: verified.
Evidence: `truncated_ratio=0.753`, `response_len_median=2048` (hitting cap on most samples).

## Bottleneck ranking

1. Rollout/reward path dominates end-to-end step time.
2. Teacher latency tail (request/response path) adds long stalls.
3. Sequence length profile (many max-length responses) increases rollout wall-time.
4. Scheduler is not queue-bound; under-fill is upstream of engine saturation.

## Process update (what was implemented)

1. Added reusable perf tooling:
   - `my_exps/opd-397-32b/perf_tools/analyze_training_perf.py`
   - `my_exps/opd-397-32b/perf_tools/ray_gpu_snapshot.py`
   - `my_exps/opd-397-32b/perf_tools/run_perf_report.sh`

2. Upgraded monitor:
   - `my_exps/opd-397-32b/monitor_privileged_training.sh`
   - periodic perf summaries + threshold alerts (`wait_ratio`, rollout TPS, teacher p95)
   - periodic GPU/logical usage snapshots

3. Reward plugin optimization for next run:
   - `my_exps/opd-397-32b/opd_topk_reward_plugin.py`
   - switched from per-request `aiohttp.ClientSession()` to pooled persistent session
   - added configurable timeout/connection pool knobs

4. Distill config env wiring for reward HTTP tuning:
   - `my_exps/opd-397-32b/train_student_async_distill_lib.sh`
   - new vars: `OPD_RM_CONNECT_TIMEOUT_S`, `OPD_RM_READ_TIMEOUT_S`, `OPD_RM_TOTAL_TIMEOUT_S`,
     `OPD_RM_MAX_CONNECTIONS`, `OPD_RM_MAX_CONNECTIONS_PER_HOST`

## Immediate tuning suggestions for next restart

1. Keep current training running for now (stable), but for next launch:
   - increase reward HTTP pool (`OPD_RM_MAX_CONNECTIONS`, `OPD_RM_MAX_CONNECTIONS_PER_HOST`)
   - consider shorter max response length (or stricter stop conditions) to reduce truncation tail
   - rebalance train/rollout GPU split if rollout remains dominant

2. Use the new report loop:
   - `bash my_exps/opd-397-32b/perf_tools/run_perf_report.sh`
   - monitor for trend changes instead of single-point snapshots
