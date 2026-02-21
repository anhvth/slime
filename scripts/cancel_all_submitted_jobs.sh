#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Cancel all active Ray submission jobs.

Usage:
  bash scripts/cancel_all_submitted_jobs.sh [--all] [--dry-run]

Options:
  --all      Include terminal jobs (SUCCEEDED/FAILED/STOPPED) in the listing.
  --dry-run  Print matching jobs without stopping them.

Environment:
  RAY_JOB_ADDRESS   Ray job server address (default: http://127.0.0.1:8265)
EOF
}

INCLUDE_TERMINAL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      INCLUDE_TERMINAL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS%/}"
JOBS_ENDPOINT="${RAY_JOB_ADDRESS}/api/jobs/"

if ! command -v ray >/dev/null 2>&1; then
  echo "ray CLI not found in PATH." >&2
  exit 1
fi

if ! JOBS_JSON="$(curl -sf "${JOBS_ENDPOINT}")"; then
  echo "Failed to fetch jobs from ${JOBS_ENDPOINT}" >&2
  exit 1
fi

mapfile -t JOBS < <(
  JOBS_JSON="${JOBS_JSON}" python3 - "${INCLUDE_TERMINAL}" <<'PY'
import json
import os
import sys

include_terminal = sys.argv[1] == "1"
terminal = {"SUCCEEDED", "FAILED", "STOPPED"}

jobs = json.loads(os.environ["JOBS_JSON"])
for job in jobs:
    if job.get("type") != "SUBMISSION":
        continue
    submission_id = job.get("submission_id")
    status = job.get("status", "UNKNOWN")
    if not submission_id:
        continue
    if not include_terminal and status in terminal:
        continue
    print(f"{submission_id}\t{status}")
PY
)

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "No matching submitted jobs found at ${RAY_JOB_ADDRESS}."
  exit 0
fi

echo "Found ${#JOBS[@]} matching submitted job(s) at ${RAY_JOB_ADDRESS}:"
for row in "${JOBS[@]}"; do
  IFS=$'\t' read -r submission_id status <<<"${row}"
  echo "  ${submission_id} (${status})"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run only. No jobs were stopped."
  exit 0
fi

STOPPED=0
FAILED=0
for row in "${JOBS[@]}"; do
  IFS=$'\t' read -r submission_id status <<<"${row}"

  if [[ "${status}" == "SUCCEEDED" || "${status}" == "FAILED" || "${status}" == "STOPPED" ]]; then
    continue
  fi

  echo "Stopping ${submission_id} ..."
  if ray job stop "${submission_id}" --address="${RAY_JOB_ADDRESS}" --no-wait >/dev/null 2>&1; then
    STOPPED=$((STOPPED + 1))
  else
    echo "  Failed to stop ${submission_id}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "Stop requests sent: ${STOPPED}, failures: ${FAILED}"
exit 0
