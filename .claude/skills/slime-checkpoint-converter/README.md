# SLIME Checkpoint Converter Skill

This skill converts SLIME training checkpoints to HuggingFace format.

## Usage

### Option 1: Direct Claude Request

Simply ask Claude to convert a checkpoint:

```
Convert outputs/opd/student_async/iter_0000699 to HF
```

Claude will automatically:
- Detect the model type
- Source the appropriate model config
- Find the original HF model
- Run the conversion

### Option 2: Using the Helper Script

```bash
.github/slime-checkpoint-converter/scripts/convert_checkpoint.sh \
    outputs/opd/student_async/iter_0000699 \
    outputs/opd/hf/iter_0000699
```

The script auto-detects:
- Model type (from checkpoint path)
- Vocab size (from model config)
- Origin HF directory (from common locations)

### Manual Conversion

```bash
# 1. Source model config
source scripts/models/qwen3-32B-as-qwen35.sh

# 2. Run conversion
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir outputs/opd/student_async/iter_0000699 \
    --output-dir outputs/opd/hf/iter_0000699 \
    --origin-hf-dir $HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned/ \
    --vocab-size 248320
```

## Supported Checkpoint Formats

- **torch_dist** (Megatron format): Use `convert_torch_dist_to_hf.py`
- **FSDP**: Use `convert_fsdp_to_hf.py`

## Output

The conversion creates:
- Sharded safetensors files (`model-0000X-of-0000Y.safetensors`)
- Weight index (`model.safetensors.index.json`)
- Config and tokenizer files (copied from origin HF directory)

## Files

- `SKILL.md` - Main skill documentation for Claude
- `scripts/convert_checkpoint.sh` - Helper script with auto-detection
