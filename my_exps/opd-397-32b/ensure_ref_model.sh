#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ensure_ref_model.sh --repo-root PATH --model-config-rel PATH --hf-checkpoint PATH --ref-load PATH [--megatron-pythonpath PATH]

Ensures torch_dist ref model exists. If missing, prints conversion command and prompts to run it.
EOF
}

REPO_ROOT=""
MODEL_CONFIG_REL=""
HF_CHECKPOINT=""
REF_LOAD_PATH=""
MEGATRON_PYTHONPATH_VAL="${MEGATRON_PYTHONPATH:-/root/Megatron-LM}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="${2:-}"
      shift 2
      ;;
    --model-config-rel)
      MODEL_CONFIG_REL="${2:-}"
      shift 2
      ;;
    --hf-checkpoint)
      HF_CHECKPOINT="${2:-}"
      shift 2
      ;;
    --ref-load)
      REF_LOAD_PATH="${2:-}"
      shift 2
      ;;
    --megatron-pythonpath)
      MEGATRON_PYTHONPATH_VAL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "${REPO_ROOT}" ]] || {
  echo "--repo-root is required" >&2
  usage >&2
  exit 1
}
[[ -n "${MODEL_CONFIG_REL}" ]] || {
  echo "--model-config-rel is required" >&2
  usage >&2
  exit 1
}
[[ -n "${HF_CHECKPOINT}" ]] || {
  echo "--hf-checkpoint is required" >&2
  usage >&2
  exit 1
}
[[ -n "${REF_LOAD_PATH}" ]] || {
  echo "--ref-load is required" >&2
  usage >&2
  exit 1
}

[[ -e "${HF_CHECKPOINT}" ]] || {
  echo "Missing student HF checkpoint path: ${HF_CHECKPOINT}" >&2
  exit 1
}

if [[ -e "${REF_LOAD_PATH}" ]]; then
  exit 0
fi

echo "Missing student ref-load path: ${REF_LOAD_PATH}" >&2
printf -v CONVERT_CMD 'source %q && PYTHONPATH=%q python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint %q --save %q' \
  "${MODEL_CONFIG_REL}" "${MEGATRON_PYTHONPATH_VAL}" "${HF_CHECKPOINT}" "${REF_LOAD_PATH}"
echo "Run this command to build it:"
echo "${CONVERT_CMD}"

[[ -t 0 ]] || {
  echo "No interactive stdin available; refusing to auto-run conversion." >&2
  exit 1
}
if ! read -r -p "Run conversion now? [y/N]: " RUN_CONVERT; then
  echo "Failed to read confirmation; exiting." >&2
  exit 1
fi

case "${RUN_CONVERT,,}" in
  y|yes)
    (cd "${REPO_ROOT}" && bash -lc "${CONVERT_CMD}")
    ;;
  *)
    echo "Exit without conversion."
    exit 1
    ;;
esac

[[ -e "${REF_LOAD_PATH}" ]] || {
  echo "Conversion finished but ref-load path is still missing: ${REF_LOAD_PATH}" >&2
  exit 1
}
