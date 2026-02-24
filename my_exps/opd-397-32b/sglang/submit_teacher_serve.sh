#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXP_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${EXP_DIR}/../.." &>/dev/null && pwd)"
RAY_ADDR_UTIL="${EXP_DIR}/ray_job_address_utils.sh"
[[ -f "${RAY_ADDR_UTIL}" ]] || {
  echo "Missing Ray address helper script: ${RAY_ADDR_UTIL}" >&2
  exit 1
}
# shellcheck source=/dev/null
source "${RAY_ADDR_UTIL}"

RAY_JOB_ADDRESS="$(require_ray_job_address)"
RUNTIME_ENV_REF_EXPIRATION_S="${RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S:-7200}"

export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S="${RUNTIME_ENV_REF_EXPIRATION_S}"

echo "Submitting teacher serve job to ${RAY_JOB_ADDRESS}"
echo "working_dir=${EXP_DIR}"
echo "RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=${RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S}"

cd "${REPO_ROOT}"

RUNTIME_ENV_JSON=$(
  cat <<'JSON'
{
  "excludes": [
    "teacher-qwen-35/.venv/**",
    "teacher-qwen-35/.cache/**",
    "logs/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc"
  ]
}
JSON
)

ray job submit \
  --address="${RAY_JOB_ADDRESS}" \
  --working-dir "${EXP_DIR}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python sglang/lauch.py
