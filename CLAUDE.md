# slime — Copilot Workspace Instructions

## Project Overview
**slime** is an LLM post-training framework for RL scaling, combining **Megatron-LM** (training) and **SGLang** (inference/rollout) via Ray. Code is edited locally and synced to a remote multi-GPU cluster to run. Do NOT attempt to run training commands locally.

## Key Architecture
- `train.py` — synchronous training loop (rollout → train → repeat)
- `train_async.py` — async training loop (rollout and training overlap); required for fully-async examples
- `slime/` — core library: `ray/`, `rollout/`, `backends/`, `utils/`, `router/`
- `slime_plugins/` — optional plugins: `megatron_bridge/`, `mbridge/`, `models/`, `rollout_buffer/`
- `scripts/models/` — MODEL_ARGS shell snippets per model architecture (source these in run scripts)
- `examples/` — self-contained example directories, each with their own run script(s)
- `datasets/` — local dataset files (JSONL, parquet); root-level, not inside examples/

## Argument Categories
1. **Megatron args** — passed directly (e.g. `--tensor-model-parallel-size 2`)
2. **SGLang args** — must be prefixed with `--sglang-` (e.g. `--sglang-mem-fraction-static 0.8`)
3. **slime-specific args** — defined in `slime/utils/arguments.py`

## Run Script Conventions
- Always use `SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"` for relative paths
- Source the correct model config: `source "${SCRIPT_DIR}/../../scripts/models/<model>.sh"` — this sets `MODEL_ARGS`
- Dataset path: `"${SCRIPT_DIR}/../../datasets/<name>.jsonl"` (root-level `datasets/`)
- Checkpoint layout: `--hf-checkpoint` (HF weights) + `--ref-load` (`_torch_dist`) + `--load`/`--save` (`_slime/`)
- SGLang args passed to rollout engine: prefix with `--sglang-`; engine-count args (e.g. `--rollout-num-gpus-per-engine`) are slime args

## Model Script Notes
- `qwen3-4B.sh` — base rotary base 1000000
- `qwen3-4B-Instruct-2507.sh` — sets `MODEL_ARGS_ROTARY_BASE=5000000` then sources `qwen3-4B.sh`; **use this for Qwen3-4B-Instruct-2507 checkpoints**

## GPU / Ray Cluster Layout
| Mode | Actor (Megatron) | Rollout (SGLang) |
|---|---|---|
| Single node 8 GPU | 1 node × 4 GPU | 4 GPU |
| Multi-node 32 GPU | 3 nodes × 8 GPU | 8 GPU |

- Single-node scripts kill existing processes and call `ray start --head`
- Multi-node scripts assume the cluster is **already running** — never call `ray start` or `pkill` inside them
- `MASTER_ADDR` defaults to `127.0.0.1`; override with env var for multi-node

## Scaling Batch Sizes (proportional to GPU count)
When going from 8→32 GPUs (4×), scale:
- `--rollout-batch-size`: ×4
- `--global-batch-size`: ×4
- TP size can stay at 2 for ≤7B models (within-node, low overhead)

## Fully-Async Pattern (`examples/fully_async/`)
- Use `train_async.py` (not `train.py`)
- Set `--rollout-function-path fully_async_rollout.generate_rollout_fully_async`
- The `PYTHONPATH` in `RUNTIME_ENV_JSON` must include the example directory so `fully_async_rollout` is importable
- Colocation (`--colocate`) is not supported with `train_async.py`

## Code Style
- Line length: 119 (black/isort); ruff handles linting
- Use `pre-commit run --all-files` before committing
- Python 3.10+ target; type hints encouraged in new code
- Tests live in `tests/`; run with `pytest`; `examples/` is excluded from test discovery

## Common Pitfalls
- Wrong model script → wrong rotary base → silent train divergence. Always match checkpoint variant to the right `scripts/models/*.sh`
- `--sglang-*` prefix missing → SGLang args silently ignored
- Multi-node script that calls `ray stop` will destroy the shared cluster
- `PYTHONPATH` in `RUNTIME_ENV_JSON` must include the example dir for custom rollout functions to be importable by Ray workers

## Quick Log Checks
To find wandb links or other info from remote training logs:
```bash
# Find wandb link from latest log
ssh login-node "cd ~/projects/slime && grep -i 'https://wandb' logs/qwen3-4b-fully-async-32gpu_*/training.log | tail -5"

# Or use ssh-tmux-ctl for interactive inspection
./ssh-tmux-ctl.sh out <window_name>
```

## Code Sync Reminder
ALWAYS sync local code changes to remote before running on the cluster. Use:
```bash
rs ./ login-node:/home/anhvth8/projects/slime/
```
Where `rs` is the alias for `rsync -av --progress`. This ensures all local changes are propagated to the remote execution environment.



# Remote Tmux Control (ssh-tmux-ctl)

## Context

We work on a **MacBook** (local). All training runs on a remote GPU cluster reachable as `login-node`.  
The single self-contained script `ssh-tmux-ctl.sh` lives at the **root of this repo** (`/Users/anhvth/projects/slime/ssh-tmux-ctl.sh`).

**How it works:**
1. Run the script locally from the MacBook — no manual SSH needed.
2. On first invocation it `rsync`s itself to `login-node:~/ssh-tmux-ctl.sh`.
3. It then `ssh`es into `login-node` and re-executes itself there with `_REMOTE_EXEC=1`, which skips the sync step and runs the tmux logic directly.
4. All tmux work happens inside the **`main`** session only. No split panes — each window has exactly one terminal with a meaningful name.

---

## When to Use

- Check what is running on the remote right now → `snap` or `ls`
- Launch a training job in a named window → `run`
- Read the output of a running job → `out`
- Clean up a window → `kill`
- Any time Copilot needs to inspect or drive `login-node` state — just call the script locally.

---

## Script Location & Invocation

```bash
# Always run from the MacBook, inside the slime repo root:
./ssh-tmux-ctl.sh <action> [args...]
```

The script auto-syncs itself to the remote on every call, so edits made locally are always reflected immediately.

---

## Actions

| Action | Args | Description |
|---|---|---|
| `snap` | — | Snapshot last 20 lines of every window (default action) |
| `ls` | — | List all windows with current process name |
| `run` | `<win_name> "<command>"` | Open (or reuse) a named window and run the command |
| `out` | `<win_name>` | Print last 50 lines from a window |
| `kill` | `<win_name>` | Kill a window by name |

---

## Common Workflows

### What's running right now?
```bash
./ssh-tmux-ctl.sh snap
./ssh-tmux-ctl.sh ls
```

### Launch a training job
```bash
# Sync code first, then launch
rs ./ login-node:/home/anhvth8/projects/slime/
./ssh-tmux-ctl.sh run train "cd ~/projects/slime && bash examples/my_exp/run.sh"
```

### Check job output
```bash
./ssh-tmux-ctl.sh out train
```

### Kill a window
```bash
./ssh-tmux-ctl.sh kill train
```

---

## Quick Log Checks
To find wandb links or other info from remote training logs:
```bash
# Find wandb link from latest log
ssh login-node "cd ~/projects/slime && grep -i 'https://wandb' logs/qwen3-4b-fully-async-32gpu_*/training.log | tail -5"

# Or use ssh-tmux-ctl for interactive inspection
./ssh-tmux-ctl.sh out <window_name>
```

---

## Tips

- **Self-syncing** — every invocation rsyncs the script to remote first; edits are always live.
- **Named windows only** — never use window indices directly; always use names.
- **Session `main` is fixed** — do not change it; all remote work lives there.
- **No interactive SSH** — all operations are single-shot from the MacBook, Copilot-friendly.
- **Re-use existing windows** — `run` on an already-open window sends the command into it rather than opening a new one.
- **ALWAYS** use remote enviroment:
  - Examples: 
    User ask to find an output file then use ssh-tmux-ctl as starting point
    User ask where is the output dir: translate it to: I need to find the output dir on the remote cluster!
       - Check log of currently training job 