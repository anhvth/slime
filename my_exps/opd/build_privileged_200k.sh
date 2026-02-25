#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

INPUT_DATASET="${INPUT_DATASET:-/home/anhvth8/projects/SFT/data/SFT_merged_2.9M}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/datasets/50k_prompt_for_distillation_privileged.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
SEED="${SEED:-33}"

# Endpoint discovery:
# - gateway (default): use Ray Serve gateway (http://<head-ip>:8000/teacher[/v1])
# - per-node: direct fan-out to each alive node using TEACHER_PORT (legacy)
TEACHER_ENDPOINT_MODE="${TEACHER_ENDPOINT_MODE:-gateway}"
TEACHER_HOST="${TEACHER_HOST:-}"
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_ROUTE_PREFIX="${TEACHER_ROUTE_PREFIX:-/teacher}"
BASE_URL="${OPENAI_BASE_URL:-}"
BASE_URLS="${OPENAI_BASE_URLS:-}"
API_KEY="${OPENAI_API_KEY:-abc}"
MODEL_ID="${MODEL_ID:-}"

WORKERS="${WORKERS:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
RETRIES="${RETRIES:-2}"
TIMEOUT="${TIMEOUT:-120}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

extract_url_host() {
  local url="$1"
  "${PYTHON_BIN}" - "${url}" <<'PY'
import sys
from urllib.parse import urlparse

raw = sys.argv[1].strip()
if not raw:
    print("")
    raise SystemExit(0)
if "://" not in raw:
    raw = f"http://{raw}"
p = urlparse(raw)
print(p.hostname or "")
PY
}

detect_head_host_from_ray_job_address() {
  local helper="${SCRIPT_DIR}/ray_job_address_utils.sh"
  if [[ ! -f "${helper}" ]]; then
    echo ""
    return 0
  fi
  # shellcheck source=/dev/null
  source "${helper}"
  local addr=""
  addr="$(resolve_ray_job_address 2>/dev/null || true)"
  if [[ -z "${addr}" ]]; then
    echo ""
    return 0
  fi
  extract_url_host "${addr}"
}

build_per_node_urls() {
  TEACHER_PORT="${TEACHER_PORT}" ray list nodes --format json | "${PYTHON_BIN}" -c '
import json, os, sys
port = os.environ["TEACHER_PORT"]
nodes = json.load(sys.stdin)
ips = sorted({n.get("node_ip") for n in nodes if n.get("state") == "ALIVE" and n.get("node_ip")})
print(",".join(f"http://{ip}:{port}/v1" for ip in ips))
  '
}

if [[ -z "${BASE_URLS}" ]]; then
  if [[ -n "${BASE_URL}" ]]; then
    BASE_URLS="${BASE_URL}"
  else
    case "${TEACHER_ENDPOINT_MODE}" in
      gateway)
        if [[ -z "${TEACHER_HOST}" ]]; then
          TEACHER_HOST="$(detect_head_host_from_ray_job_address)"
        fi
        TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
        BASE_URLS="http://${TEACHER_HOST}:${TEACHER_PORT}${TEACHER_ROUTE_PREFIX}"
        ;;
      per-node)
        BASE_URLS="$(build_per_node_urls)"
        ;;
      *)
        echo "Unsupported TEACHER_ENDPOINT_MODE='${TEACHER_ENDPOINT_MODE}'. Use 'gateway' or 'per-node'." >&2
        exit 1
        ;;
    esac
  fi
fi

if [[ -z "${BASE_URLS}" ]]; then
  echo "Failed to resolve teacher endpoints. Set OPENAI_BASE_URLS or OPENAI_BASE_URL." >&2
  exit 1
fi
echo "Resolved teacher endpoint(s): ${BASE_URLS}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_privileged_dataset.py"
  --input-dataset "${INPUT_DATASET}"
  --output "${OUTPUT_PATH}"
  --num-samples "${NUM_SAMPLES}"
  --seed "${SEED}"
  --base-urls "${BASE_URLS}"
  --api-key "${API_KEY}"
  --workers "${WORKERS}"
  --chunk-size "${CHUNK_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --retries "${RETRIES}"
  --timeout "${TIMEOUT}"
)

if [[ -n "${MODEL_ID}" ]]; then
  CMD+=(--model "${MODEL_ID}")
fi

echo "Running privileged dataset build:"
printf '  %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
