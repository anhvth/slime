#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MEGATRON_PYTHONPATH="${MEGATRON_PYTHONPATH:-/root/Megatron-LM}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-${REPO_ROOT}/scripts/models/qwen3-32B-as-qwen35.sh}"

INPUT_ITER_DIR="${INPUT_ITER_DIR:-${REPO_ROOT}/outputs/opd/student_async/iter_0001599}"
RESUME_ITER_TAG="${RESUME_ITER_TAG:-iter_0001599}"
OUTPUT_HF_DIR="${OUTPUT_HF_DIR:-${REPO_ROOT}/outputs/opd/student_async_hf/${RESUME_ITER_TAG}}"
OUTPUT_DIST_DIR="${OUTPUT_DIST_DIR:-${REPO_ROOT}/outputs/opd/student_async_hf/${RESUME_ITER_TAG}_torch_dist}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-$HOME/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned}"

OVERWRITE_OUTPUT_DIST="${OVERWRITE_OUTPUT_DIST:-0}"

require_file() {
  local path="$1"
  local message="$2"
  [[ -f "${path}" ]] || {
    echo "${message}: ${path}" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  local message="$2"
  [[ -d "${path}" ]] || {
    echo "${message}: ${path}" >&2
    exit 1
  }
}

require_dir "${INPUT_ITER_DIR}" "Missing input torch_dist checkpoint directory"
require_file "${INPUT_ITER_DIR}/common.pt" "Missing torch_dist manifest file"
require_dir "${ORIGIN_HF_DIR}" "Missing origin HF directory for tokenizer/config assets"
require_file "${ORIGIN_HF_DIR}/config.json" "Missing origin HF config file"
require_file "${MODEL_CONFIG_SCRIPT}" "Missing model config script"

if [[ -e "${OUTPUT_DIST_DIR}" ]]; then
  if [[ "${OVERWRITE_OUTPUT_DIST}" == "1" ]]; then
    echo "Removing existing torch_dist output directory: ${OUTPUT_DIST_DIR}"
    rm -rf "${OUTPUT_DIST_DIR}"
  else
    echo "Output torch_dist directory already exists: ${OUTPUT_DIST_DIR}" >&2
    echo "Set OVERWRITE_OUTPUT_DIST=1 to remove and rebuild it." >&2
    exit 1
  fi
fi

mkdir -p "$(dirname -- "${OUTPUT_HF_DIR}")" "$(dirname -- "${OUTPUT_DIST_DIR}")"

echo "[prepare_resume] repo_root=${REPO_ROOT}"
echo "[prepare_resume] input_iter_dir=${INPUT_ITER_DIR}"
echo "[prepare_resume] output_hf_dir=${OUTPUT_HF_DIR}"
echo "[prepare_resume] output_dist_dir=${OUTPUT_DIST_DIR}"
echo "[prepare_resume] origin_hf_dir=${ORIGIN_HF_DIR}"
echo "[prepare_resume] model_config=${MODEL_CONFIG_SCRIPT}"

echo "[prepare_resume] [1/2] Converting torch_dist -> HF ..."
"${PYTHON_BIN}" tools/convert_torch_dist_to_hf.py \
  --input-dir "${INPUT_ITER_DIR}" \
  --output-dir "${OUTPUT_HF_DIR}" \
  --origin-hf-dir "${ORIGIN_HF_DIR}" \
  -f

require_file "${OUTPUT_HF_DIR}/config.json" "Missing HF output config"
require_file "${OUTPUT_HF_DIR}/model.safetensors.index.json" "Missing HF output model index"

echo "[prepare_resume] [2/2] Converting HF -> torch_dist ..."
# shellcheck source=/dev/null
source "${MODEL_CONFIG_SCRIPT}"
if [[ "${#MODEL_ARGS[@]}" -eq 0 ]]; then
  echo "MODEL_ARGS is empty after sourcing ${MODEL_CONFIG_SCRIPT}" >&2
  exit 1
fi
PYTHONPATH="${MEGATRON_PYTHONPATH}" "${PYTHON_BIN}" tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${OUTPUT_HF_DIR}" \
  --save "${OUTPUT_DIST_DIR}"

require_file "${OUTPUT_DIST_DIR}/common.pt" "Missing torch_dist output common.pt"
if [[ ! -f "${OUTPUT_DIST_DIR}/.metadata" && ! -f "${OUTPUT_DIST_DIR}/metadata.json" ]]; then
  echo "Missing torch_dist output metadata (.metadata or metadata.json): ${OUTPUT_DIST_DIR}" >&2
  exit 1
fi

SUGGESTED_SAVE="${REPO_ROOT}/outputs/opd/student_async_distill_privileged_from_${RESUME_ITER_TAG}"
DEFAULT_PROMPT_DATA="${REPO_ROOT}/datasets/50k_prompt_for_distillation_privileged.jsonl"

echo
echo "[prepare_resume] Conversion complete."
echo "[prepare_resume] Ready-to-run exports:"
cat <<EOF
export RESUME_ITER_TAG="${RESUME_ITER_TAG}"
export RESUME_HF_DIR="${OUTPUT_HF_DIR}"
export RESUME_DIST_DIR="${OUTPUT_DIST_DIR}"
export STUDENT_HF_CHECKPOINT="${OUTPUT_HF_DIR}"
export STUDENT_REF_LOAD="${OUTPUT_DIST_DIR}"
export STUDENT_LOAD="${OUTPUT_DIST_DIR}"
export STUDENT_SAVE="${SUGGESTED_SAVE}"
export PROMPT_DATA="${DEFAULT_PROMPT_DATA}"
bash my_exps/opd/train_student_async_jsd_with_privileged_infomation.sh
EOF
