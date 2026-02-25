# OPD Training Bottleneck Analysis Report

**Date:** 2026-02-21  
**Training Script:** `my_exps/opd/train_student.sh`  
**WandB Run:** https://wandb.ai/anhvth/slime-opd/runs/zssfduek

---

## Executive Summary

| Metric | Value | Issue |
|--------|-------|-------|
| `train_wait_time` | ~110-125s | **HIGH** |
| `train_time` | ~50-70s | Normal |
| `wait_time_ratio` | ~0.62-0.70 | **Too high** - GPU idle 60-70% of time |
| `tokens_per_gpu_per_sec` | ~300 | Healthy GPU compute |

**Root Cause:** The training uses synchronous `train.py` which blocks on every rollout generation step. Rollout includes:
1. **Student generation** (SGLang engines, 8192 max tokens × 96 samples)
2. **Teacher inference** (HTTP calls to external 397B teacher on `worker-15:13141`)
3. **Offload/onload** (colocated mode memory swapping)

---

## Configuration Summary

### Rollout Parameters
| Parameter | Value | Impact |
|-----------|-------|--------|
| `rollout-batch-size` | 24 | Prompts per batch |
| `n-samples-per-prompt` | 4 | Samples per prompt |
| `rollout-max-response-len` | 8192 | Max generated tokens |
| `global-batch-size` | 96 | Total samples per rollout |
| **Total tokens per rollout** | **786,432** | 24 × 4 × 8192 |

### Parallelism (3 nodes × 8 GPUs = 24 GPUs, colocated)
| Parameter | Value |
|-----------|-------|
| `tensor-model-parallel-size` | 8 |
| `actor-num-nodes` | 3 |
| `actor-num-gpus-per-node` | 8 |
| `colocate` | **Enabled** (training + inference share GPUs) |
| `sglang-mem-fraction-static` | 0.7 |

### Teacher Model (OPD)
| Parameter | Value |
|-----------|-------|
| Teacher URL | `http://worker-15:13141/generate` |
| Teacher Model | Qwen3.5-397B-A17B (external, on separate GPU) |
| OPD KL Coef | 1.0 |
| Custom RM Path | `examples.on_policy_distillation.on_policy_distillation.reward_func` |

---

## Precise Rollout Flow (Step-by-Step)

### In `train.py` (Lines 56-95)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ROLLOUT PHASE (inside train_wait_time)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. rollout_manager.generate.remote(rollout_id)                              │
│    └── RolloutManager.generate() [slime/ray/rollout.py:141]                │
│        └── _get_rollout_data()                                              │
│            └── call_rollout_fn() → SGLang generation                        │
│                ├── Student model generates 96 samples                       │
│                ├── Each sample up to 8192 tokens                            │
│                └── Custom reward_func called per sample                     │
│                    └── HTTP POST to teacher (worker-15:13141)              │
│                        └── Returns teacher logprobs for KL penalty          │
│                                                                              │
│ 2. ray.get() - BLOCKS until all generation completes (~100s)               │
│                                                                              │
│ 3. IF offload_rollout: rollout_manager.offload.remote()                     │
│    └── Releases KV cache to make room for training weights                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PHASE (inside train_time)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. actor_model.async_train(rollout_id, rollout_data_ref)                    │
│    └── MegatronActor.train_actor() [slime/backends/megatron_utils/actor.py]│
│        ├── process_rollout_data() - preprocess samples                      │
│        ├── compute_log_prob() - student forward pass                        │
│        ├── compute_advantages_and_returns() - OPD KL penalty applied here   │
│        └── actor_train() - backward pass + optimizer step                   │
│                                                                              │
│ 5. ray.get() - BLOCKS until training completes (~50s)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ WEIGHT SYNC PHASE (inside train_wait_time)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. IF offload_rollout: onload_weights.remote()                              │
│    └── Reload student weights to SGLang engines                             │
│                                                                              │
│ 7. actor_model.update_weights()                                             │
│    └── Distributed broadcast of updated weights                             │
│    └── dist.barrier() if new engines connected                              │
│                                                                              │
│ 8. IF offload_rollout: onload_kv.remote()                                   │
│    └── Reload KV cache for next generation                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Blocking Points

| Location | File:Line | Blocks On |
|----------|-----------|-----------|
| Rollout generation | `train.py:69` | `ray.get(rollout_manager.generate.remote())` |
| Offload | `train.py:72` | `ray.get(rollout_manager.offload.remote())` |
| Training | `train.py:77` | `ray.get(actor_model.async_train())` |
| Onload weights | `train.py:87` | `ray.get(rollout_manager.onload_weights.remote())` |
| Update weights | `train.py:88` | `actor_model.update_weights()` |
| Onload KV | `train.py:90` | `ray.get(rollout_manager.onload_kv.remote())` |

---

## Where `train_wait_time` is Computed

### Implementation

**File:** `slime/backends/megatron_utils/actor.py`

```python
# Line 46 - Starts "train_wait" timer AFTER init() completes
@with_defer(lambda: Timer().start("train_wait"))
def init(self, args, role, ...):
    ...

# Line 402 - train_actor() STOPS wait, starts train, then restarts wait on exit
def train_actor(self, ...):
    with inverse_timer("train_wait"), timer("train"):
        # All training operations
        ...
```

### What's Included in `train_wait_time`

| Activity | In Wait Time? |
|----------|---------------|
| Rollout generation (student + teacher) | ✅ YES |
| Offload weights/KV | ✅ YES |
| Weight sync to rollout engines | ✅ YES |
| Onload weights/KV | ✅ YES |
| Idle time between steps | ✅ YES |
| Checkpoint saving | ✅ YES |

### Metric Calculation

**File:** `slime/utils/train_metric_utils.py:38-42`

```python
if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
    total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
    log_dict["perf/step_time"] = total_time
    log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time
```

---

## Primary Bottlenecks

### 1. **Synchronous Rollout Generation** (CRITICAL)

The main loop in `train.py` is fully sequential:
```python
# Line 69 - BLOCKS for ~100s during rollout
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```

**Impact:** GPU sits idle while waiting for:
- Student SGLang generation (96 samples × 8192 tokens)
- Teacher HTTP inference (96 HTTP calls to worker-15)

### 2. **Colocated Mode Memory Swapping**

With `--colocate`, training and inference share the same 24 GPUs. This requires:
- `offload()` before training (KV cache → CPU)
- `onload_weights()` after training (weights → GPU)
- `onload_kv()` before next rollout (KV cache → GPU)

**File:** `train.py:70-90`
```python
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())      # ~2-5s
    ...
    ray.get(rollout_manager.onload_weights.remote())  # ~2-5s
    ...
    ray.get(rollout_manager.onload_kv.remote())       # ~2-5s
```

### 3. **Teacher Inference Latency**

**File:** `examples/on_policy_distillation/on_policy_distillation.py`

```python
async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            return await resp.json()
```

**Impact:** 
- 96 sequential HTTP calls to teacher server
- Teacher is 397B model (slow even with A17B active params)
- No batching of teacher requests

### 4. **Distributed Barriers During Weight Update**

**File:** `slime/backends/megatron_utils/actor.py:70,75,539,550`

```python
if num_new_engines > 0:
    dist.barrier(group=get_gloo_group())  # BLOCKS ALL TRAINERS
```

---

## Estimated Per-Step Timing Breakdown

| Phase | Time (est.) | Included In |
|-------|-------------|-------------|
| **WAIT PHASE** | **~110-125s** | `train_wait_time` |
| ├─ Student generation | ~60-80s | wait |
| ├─ Teacher inference (96 samples) | ~20-40s | wait |
| ├─ Offload KV | ~3-5s | wait |
| ├─ Onload weights | ~3-5s | wait |
| └─ Onload KV | ~3-5s | wait |
| **TRAIN PHASE** | **~50-70s** | `train_time` |
| ├─ Process data | ~2s | train |
| ├─ Log probs forward | ~15s | train |
| ├─ Advantage compute | ~5s | train |
| └─ Backward + optimizer | ~30s | train |
| **TOTAL** | **~160-195s** | `step_time` |

---

## Comparison: Sync vs Async Training

### Current: `train.py` (Synchronous)

```
Step N:  [ROLLOUT N] ──────────────► [TRAIN N] ──► [SYNC] ──►
Step N+1:                                              [ROLLOUT N+1] ───► ...
```
- Wait time includes full rollout duration
- `wait_time_ratio ≈ 0.65`

### Alternative: `train_async.py` (Pipelined)

```
Step N:  [ROLLOUT N] ──────────────► [TRAIN N] ──►
Step N+1:              [ROLLOUT N+1] ────────────► [TRAIN N+1] ──►
```
- Rollout N+1 overlaps with Train N
- **But:** Requires `--colocate=False` (disaggregated mode)

---

## Recommendations

### High Impact

| # | Recommendation | Expected Improvement | Effort |
|---|----------------|---------------------|--------|
| 1 | **Use `train_async.py`** | 30-50% reduction in wait_time | Low (but needs disaggregated GPUs) |
| 2 | **Batch teacher inference** | 2-3x faster teacher phase | Medium |
| 3 | **Reduce max_response_len** (e.g., 4096) | Direct reduction in generation time | Low |
| 4 | **Disable offload_rollout** if memory allows | Eliminates offload/onload overhead | Low |

### Medium Impact

| # | Recommendation | Expected Improvement | Effort |
|---|----------------|---------------------|--------|
| 5 | Increase `rollout-batch-size` with `n-samples-per-prompt` reduction | Better GPU utilization during generation | Low |
| 6 | Use local teacher (same cluster) | Reduce network latency | Medium |
| 7 | Enable `--use-dynamic-batch-size` (already enabled) | Better padding efficiency | N/A |

### Architecture Changes

| # | Recommendation | Expected Improvement | Effort |
|---|----------------|---------------------|--------|
| 8 | Switch to disaggregated mode (separate rollout GPUs) | Enable async training | High |
| 9 | Implement `examples/fully_async/` pattern | Maximum overlap | High |

---

## Code Locations

| Component | File | Key Functions |
|-----------|------|---------------|
| Main training loop | `train.py` | `train()`, lines 56-95 |
| Async training loop | `train_async.py` | `train()`, lines 18-66 |
| Rollout manager | `slime/ray/rollout.py` | `RolloutManager.generate()`, line 141 |
| Actor (Megatron) | `slime/backends/megatron_utils/actor.py` | `train_actor()`, line 402 |
| Timer/tracking | `slime/utils/timer.py` | `Timer`, `inverse_timer`, `with_defer` |
| Metrics | `slime/utils/train_metric_utils.py` | `log_perf_data_raw()`, line 38 |
| OPD reward | `examples/on_policy_distillation/on_policy_distillation.py` | `reward_func()`, `post_process_rewards()` |

---

## Conclusion

The high `wait_time_ratio (~0.65)` is primarily caused by:

1. **Synchronous rollout** - Every step blocks on `ray.get(generate.remote())`
2. **Colocated mode overhead** - Memory swapping between training/inference
3. **Sequential teacher inference** - 96 individual HTTP calls

**Quick wins:**
- Reduce `rollout-max-response-len` from 8192 to 4096
- Batch teacher inference calls

**Long-term solution:**
- Switch to `train_async.py` with disaggregated GPU allocation
- Or implement fully async pattern from `examples/fully_async/`
