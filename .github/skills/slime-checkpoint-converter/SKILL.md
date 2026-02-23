---
name: slime-checkpoint-converter
description: Convert SLIME training checkpoints (torch_dist/Megatron format) to HuggingFace format for deployment and inference. Use when the user asks to convert, export, or transform checkpoints from SLIME training outputs to HF format. Handles both FSDP and Megatron backends with automatic model config detection.
---

# SLIME Checkpoint Converter

Convert SLIME training checkpoints to HuggingFace format.

## Quick Start

**Standard conversion (Megatron torch_dist → HF):**

```bash
# 1. Source the model config
source scripts/models/<model-name>.sh

# 2. Run conversion
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir <checkpoint-path>/iter_XXXXXX \
    --output-dir <output-path> \
    --origin-hf-dir <original-hf-model-path> \
    --vocab-size <vocab-size>
```

**Example (Qwen3-32B):**

```bash
source scripts/models/qwen3-32B-as-qwen35.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir outputs/opd-397-32b/student_async/iter_0000699 \
    --output-dir outputs/opd-397-32b/student_async_hf/iter_0000699 \
    --origin-hf-dir $HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned/ \
    --vocab-size 248320
```

## Required Parameters

- `--input-dir`: Path to torch_dist checkpoint (contains `.distcp` files and `metadata.json`)
- `--output-dir`: Destination for HF checkpoint (must not exist, or use `-f` to overwrite)
- `--origin-hf-dir`: Original HF model directory (for tokenizer, config.json, etc.)
- `--vocab-size`: Vocabulary size from model config (removes embedding padding)

## Model Configurations

Always source the appropriate model config before conversion:

| Model | Config File | Vocab Size |
|-------|-------------|------------|
| Qwen3-4B | `scripts/models/qwen3-4B.sh` | 151936 |
| Qwen3-4B (as Qwen3.5) | `scripts/models/qwen3-4B-as-qwen35.sh` | 248320 |
| Qwen3-32B (as Qwen3.5) | `scripts/models/qwen3-32B-as-qwen35.sh` | 248320 |
| GLM4-9B | `scripts/models/glm4-9B.sh` | 151936 |

**Model config sets `MODEL_ARGS` array with:**
- Architecture parameters (num-layers, hidden-size, etc.)
- Vocabulary size
- Rotary base and normalization settings

## Conversion Process

The conversion tool (`tools/convert_torch_dist_to_hf.py`):

1. **Loads distributed checkpoint** using `torch.distributed.checkpoint`
2. **Converts parameter names** from Megatron to HuggingFace format
3. **Removes vocabulary padding** if `--vocab-size` is specified
4. **Shards weights** into ~5GB chunks (configurable via `--chunk-size`)
5. **Copies tokenizer and config** from origin HF directory

## Output Structure

```
<output-dir>/
├── model-00001-of-000XX.safetensors
├── model-00002-of-000XX.safetensors
├── ...
├── model.safetensors.index.json
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── ... (other HF assets)
```

## Alternative: FSDP to HF

For FSDP checkpoints, use `tools/convert_fsdp_to_hf.py`:

```bash
python tools/convert_fsdp_to_hf.py \
    --input-dir <fsdp-checkpoint-path> \
    --output-dir <output-path> \
    --origin-hf-dir <original-hf-model-path>
```

## Troubleshooting

**"Output directory already exists"**: Add `-f` flag to overwrite

**Missing Megatron-LM**: Ensure `PYTHONPATH=/root/Megatron-LM` is set

**Wrong vocab size**: Check model config file for correct `--vocab-size` value

**Conversion is slow**: Large models (32B+) take 10-30 minutes to load and convert

## Workflow

When user requests checkpoint conversion:

1. **Identify checkpoint path** - Usually in `outputs/<exp-name>/` with `iter_XXXXXX` format
2. **Determine model type** - Check training script or checkpoint metadata
3. **Source model config** - Load appropriate config from `scripts/models/`
4. **Find origin HF model** - From training script's `--hf-checkpoint` parameter
5. **Run conversion** - Execute with all required parameters
6. **Verify output** - Check for safetensors files and copied assets
