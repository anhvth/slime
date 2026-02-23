#!/bin/bash
# Convert SLIME torch_dist checkpoint to HuggingFace format
# Usage: ./convert_checkpoint.sh <checkpoint-path> <output-path> [model-config]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# Arguments
CHECKPOINT_PATH="${1:-}"
OUTPUT_PATH="${2:-}"
MODEL_CONFIG="${3:-}"
ORIGIN_HF_DIR="${4:-}"

# Validation
if [[ -z "${CHECKPOINT_PATH}" ]] || [[ -z "${OUTPUT_PATH}" ]]; then
    echo "Usage: $0 <checkpoint-path> <output-path> [model-config] [origin-hf-dir]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint-path  Path to torch_dist checkpoint (e.g., outputs/exp/iter_0000699)"
    echo "  output-path      Destination for HF checkpoint"
    echo "  model-config     Model config script (default: auto-detect)"
    echo "  origin-hf-dir    Original HF model directory (default: auto-detect)"
    echo ""
    echo "Examples:"
    echo "  $0 outputs/opd-397-32b/student_async/iter_0000699 outputs/opd-397-32b/hf/iter_0000699"
    echo "  $0 outputs/exp/iter_0000100 outputs/hf scripts/models/qwen3-32B-as-qwen35.sh"
    exit 1
fi

# Auto-detect model config if not provided
if [[ -z "${MODEL_CONFIG}" ]]; then
    # Try to find parent directory name containing model hints
    PARENT_DIR=$(basename "$(dirname "${CHECKPOINT_PATH}")")
    if [[ "${PARENT_DIR}" =~ 32[bB] ]] || [[ "${CHECKPOINT_PATH}" =~ 32[bB] ]]; then
        MODEL_CONFIG="scripts/models/qwen3-32B-as-qwen35.sh"
    elif [[ "${PARENT_DIR}" =~ 4[bB] ]] || [[ "${CHECKPOINT_PATH}" =~ 4[bB] ]]; then
        MODEL_CONFIG="scripts/models/qwen3-4B-as-qwen35.sh"
    else
        echo "Could not auto-detect model config. Please specify manually."
        exit 1
    fi
    echo "Auto-detected model config: ${MODEL_CONFIG}"
fi

# Resolve to absolute path if relative
if [[ "${MODEL_CONFIG}" != /* ]]; then
    MODEL_CONFIG="${REPO_ROOT}/${MODEL_CONFIG}"
fi

# Source model config
if [[ -f "${MODEL_CONFIG}" ]]; then
    source "${MODEL_CONFIG}"
else
    echo "Model config not found: ${MODEL_CONFIG}"
    exit 1
fi

# Extract vocab size from MODEL_ARGS
VOCAB_SIZE=""
prev_arg=""
for arg in "${MODEL_ARGS[@]}"; do
    if [[ "${arg}" == "--vocab-size" ]]; then
        :
    elif [[ "${prev_arg}" == "--vocab-size" ]]; then
        VOCAB_SIZE="${arg}"
        break
    fi
    prev_arg="${arg}"
done

if [[ -z "${VOCAB_SIZE}" ]]; then
    echo "Could not extract vocab-size from MODEL_ARGS"
    exit 1
fi
echo "Vocabulary size: ${VOCAB_SIZE}"

# Auto-detect origin HF dir if not provided
if [[ -z "${ORIGIN_HF_DIR}" ]]; then
    # Try common locations
    for candidate in \
        "$HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned" \
        "$HOME/models/Qwen/Qwen3-32B" \
        "$HOME/models/Qwen/Qwen3-4B"; do
        if [[ -d "${candidate}" ]]; then
            ORIGIN_HF_DIR="${candidate}"
            echo "Auto-detected origin HF dir: ${ORIGIN_HF_DIR}"
            break
        fi
    done
fi

if [[ -z "${ORIGIN_HF_DIR}" ]]; then
    echo "Could not auto-detect origin HF directory. Please specify manually."
    exit 1
fi

# Build conversion command
CONVERT_CMD="PYTHONPATH=/root/Megatron-LM python ${REPO_ROOT}/tools/convert_torch_dist_to_hf.py \
    --input-dir ${CHECKPOINT_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --origin-hf-dir ${ORIGIN_HF_DIR} \
    --vocab-size ${VOCAB_SIZE}"

# Check if output exists
if [[ -d "${OUTPUT_PATH}" ]]; then
    echo "Warning: Output directory exists. Adding -f flag to overwrite."
    CONVERT_CMD="${CONVERT_CMD} -f"
fi

# Run conversion
echo "Running conversion..."
echo "  Input:  ${CHECKPOINT_PATH}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Config: ${MODEL_CONFIG}"
echo "  Origin: ${ORIGIN_HF_DIR}"
echo ""

eval "${CONVERT_CMD}"

echo ""
echo "Conversion complete!"
echo "Output saved to: ${OUTPUT_PATH}"
