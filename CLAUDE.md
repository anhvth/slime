# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLIME is an LLM post-training framework for RL (Reinforcement Learning) scaling. It connects Megatron (training) with SGLang (inference) through a Ray-based distributed architecture. SLIME has been used to train GLM-5, GLM-4.7, GLM-4.6, GLM-4.5 and supports Qwen3, DeepSeek V3, and Llama 3 models.

## Common Commands

### Training

```bash
# Standard synchronous training (single node, colocated mode)
python train.py --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate ...

# Asynchronous training
python train_async.py --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate ...

# Run example training scripts (refer to scripts/ directory)
bash scripts/run-qwen3-4B.sh
```

### Model Conversion

```bash
# Convert HuggingFace to Megatron torch_dist format
source scripts/models/<model-name>.sh  # Load model config
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/hf/model \
    --save /path/to/torch_dist/checkpoint

# Convert Megatron checkpoint back to HuggingFace format
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/torch_dist/ckpt/iter_xxx/ \
    --output-dir /path/to/output \
    --origin-hf-dir /path/to/origin/hf

# For large models, use torchrun for multi-GPU conversion
torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py ...
```

**Note**: Always set `PYTHONPATH=/root/Megatron-LM` when running tools.

### Code Quality

```bash
# Run pre-commit hooks (includes Black, isort, ruff)
pre-commit run --all-files --show-diff-on-failure --color=always

# Manual formatting (line length: 119)
black slime/
isort slime/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test with markers
pytest -m "unit"              # Unit tests only
pytest -m "integration"       # Integration tests only
pytest -m "not skipduringci"  # Exclude CI-skipped tests

# Run a specific test file
pytest tests/test_qwen3_4B_fsdp.py
```

## Architecture

### Three-Module Design

1. **Training Module (Megatron/FSDP)**: Main training process, reads from Data Buffer, synchronizes parameters to Rollout
2. **Rollout Module (SGLang + Router)**: Generates new data with rewards/verifier outputs, stores in Data Buffer
3. **Data Buffer**: Bridge between training and rollout, manages prompt initialization and custom data

### Key Directories

- `slime/backends/` - Backend implementations (Megatron/FSDP training, SGLang inference)
- `slime/rollout/` - Data generation with reward model support
- `slime/ray/` - Ray-based distributed training (actor groups, placement)
- `slime/router/` - Request routing middleware
- `slime/utils/` - Utilities including argument parsing, logging, evaluation configs
- `scripts/models/` - Model configuration files (source these to load MODEL_ARGS)
- `examples/` - Training patterns (on-policy, multi-agent, tool use, async)
- `slime_plugins/` - Plugin system for extending functionality

### Argument Categories

1. **Megatron arguments**: Pass directly (e.g., `--tensor-model-parallel-size 2`)
2. **SGLang arguments**: Prefix with `--sglang-` (e.g., `--sglang-mem-fraction-static 0.7`)
3. **slime-specific arguments**: See `slime/utils/arguments.py`

### Training Modes

- **Colocated mode** (`--colocate`): Training and inference share GPUs; adjust `--sglang-mem-fraction-static` to avoid OOM (typically 0.7-0.8)
- **Disaggregated mode**: Separate GPU allocation for `--actor-num-gpus-per-node` and `--rollout-num-gpus`

### Data Flow Constraint

The rollout and training phases must balance: `(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`

## Model Configurations

Model configs are in `scripts/models/*.sh`. Always source the appropriate config before conversion/training:

```bash
source scripts/models/qwen3-4B.sh  # Sets MODEL_ARGS
source scripts/models/glm4-9B.sh   # Sets MODEL_ARGS
```

Verify config parameters match your model version (especially `--rotary-base`).

## Supported Algorithms

- GRPO (default)
- GSPO, Reinforce++, Reinforce++ Baseline, PPO
- Set via `--advantage-estimator`

## Dynamic Batching

Enable `--use-dynamic-batch-size` with `--max-tokens-per-gpu` for efficient token-based batching. This is the recommended approach and does not affect loss calculation.

## Checkpoint Management

- `--no-save-optim`: Save model weights only (no optimizer state), reducing checkpoint size by ~70-80%. Note: Cannot resume training from these checkpoints.
- `--save-hf <path>`: Save in HuggingFace format directly during training.

## Custom Extensions

For multi-turn or tool-use scenarios:
1. Prepare data with a `metadata` column containing JSON-structured additional info
2. Implement custom generate function: `async def generate(args, sample: Sample, sampling_params) -> Sample`
3. Implement custom reward function: `async def reward_func(args, sample: Sample, **kwargs) -> float`
4. Configure via `--custom-generate-function-path` and `--custom-rm-path`

## Hardware Support

- **B200/H-series**: Supported via Docker image `slimerl/slime:latest`
- **AMD**: Refer to `docs/en/platform_support/amd_tutorial.md`
- H-series has CI coverage; B-series is stable but lacks CI protection

## Development Tips

- For debugging, see `docs/en/developer_guide/debug.md`
- For profiling, see `docs/en/developer_guide/profiling.md`
- Use `PYTHONBUFFERED=16` to prevent Ray from buffering stdout/stderr
- For multi-node training, start Ray cluster first: `ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8`

### Debug Flags

- `--debug-rollout-only`: Initialize SGLang only (no Megatron) for debugging inference
- `--debug-train-only`: Initialize Megatron only (no SGLang) for debugging training
- `--save-debug-rollout-data <path>`: Save rollout data for later replay
- `--load-debug-rollout-data <path>`: Load saved rollout data for reproducible training debug
- `CUDA_LAUNCH_BLOCKING=1`: Enable for debugging SGLang illegal memory access issues
