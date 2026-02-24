#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Cancel all active Ray submission jobs.

Usage:
  bash scripts/cancel_all_submitted_jobs.sh [--all] [--dry-run] [--serve]

Options:
  --all      Include terminal jobs (SUCCEEDED/FAILED/STOPPED) in the listing.
  --dry-run  Print matching jobs without stopping them.
  --serve    Also delete the Ray Serve teacher-gateway deployment (releases GPU memory).

Environment:
  RAY_JOB_ADDRESS   Ray job server address (auto-detected from 'ray job list' when unset)
EOF
}

INCLUDE_TERMINAL=0
DRY_RUN=0
STOP_SERVE=0

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
    --serve)
      STOP_SERVE=1
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

if ! command -v ray >/dev/null 2>&1; then
  echo "ray CLI not found in PATH." >&2
  exit 1
fi

resolve_ray_job_address() {
  if [[ -n "${RAY_JOB_ADDRESS:-}" ]]; then
    echo "${RAY_JOB_ADDRESS}"
    return 0
  fi

  local job_list_output=""
  job_list_output="$(ray job list 2>&1 | sed -E $'s/\x1B\\[[0-9;]*[[:alpha:]]//g')" || {
    echo "Failed to run 'ray job list' to detect RAY_JOB_ADDRESS." >&2
    exit 1
  }
  local detected_addr=""
  detected_addr="$(printf '%s\n' "${job_list_output}" | grep -Eo 'https?://[^[:space:]]+' | head -n1 || true)"
  if [[ -z "${detected_addr}" ]]; then
    echo "Could not detect RAY_JOB_ADDRESS from 'ray job list' output." >&2
    echo "Set RAY_JOB_ADDRESS explicitly, e.g. RAY_JOB_ADDRESS=http://<head-ip>:8265" >&2
    exit 1
  fi
  echo "${detected_addr}"
}

RAY_JOB_ADDRESS="$(resolve_ray_job_address)"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS%/}"
if ! ray job list --address="${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
  echo "Unable to reach Ray Job server at ${RAY_JOB_ADDRESS}." >&2
  echo "Set RAY_JOB_ADDRESS explicitly, e.g. RAY_JOB_ADDRESS=http://<head-ip>:8265" >&2
  exit 1
fi

JOBS_ENDPOINT="${RAY_JOB_ADDRESS}/api/jobs/"

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
  if [[ "${STOP_SERVE}" -eq 1 ]]; then
    # Fall through to serve deletion below
    :
  else
    exit 0
  fi
fi

if [[ ${#JOBS[@]} -gt 0 ]]; then
  echo "Found ${#JOBS[@]} matching submitted job(s) at ${RAY_JOB_ADDRESS}:"
  for row in "${JOBS[@]}"; do
    IFS=$'\t' read -r submission_id status <<<"${row}"
    echo "  ${submission_id} (${status})"
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run only. No jobs were stopped."
  else
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
  fi
fi

if [[ "${STOP_SERVE}" -eq 1 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would delete all Ray Serve applications and kill orphan sglang processes"
  else
    echo "Deleting all Ray Serve applications..."
    HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X DELETE "${RAY_JOB_ADDRESS}/api/serve/applications/")"
    if [[ "${HTTP_CODE}" == "200" ]]; then
      echo "Ray Serve applications deleted."
    else
      echo "Warning: DELETE /api/serve/applications/ returned HTTP ${HTTP_CODE}" >&2
    fi

    echo "Killing orphan sglang/fast_sglang processes on all nodes..."
    # Base64-encode the kill script so we can pass it cleanly via Ray job REST API
    KILL_B64="$(base64 -w0 <<'PYEOF'
import ray, subprocess, os, socket, signal
ray.init(address="auto")

@ray.remote(num_cpus=0.01)
class K:
    def kill(self):
        killed = []
        for pat in ["sglang.launch_server", "fast_sglang"]:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            if r.returncode == 0:
                for s in r.stdout.strip().split("\n"):
                    try:
                        os.kill(int(s), signal.SIGKILL)
                        killed.append(int(s))
                    except Exception:
                        pass
        h = socket.gethostname()
        return f"{h}: killed {killed}" if killed else f"{h}: clean"

nodes = [n for n in ray.nodes() if n.get("Alive")]
ws = [K.options(scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote() for n in nodes]
for r in ray.get([w.kill.remote() for w in ws], timeout=60):
    print(r)
print(f"Cleaned {len(ws)} nodes")
PYEOF
)"
    KILL_JOB_ID="$(curl -sf -X POST "${RAY_JOB_ADDRESS}/api/jobs/" \
      -H "Content-Type: application/json" \
      -d "{\"entrypoint\": \"python3 -c \\\"import base64;exec(base64.b64decode('${KILL_B64}').decode())\\\"\", \"entrypoint_num_cpus\": 0}" \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('submission_id',''))")"
    if [[ -n "${KILL_JOB_ID}" ]]; then
      echo "Waiting for kill job ${KILL_JOB_ID}..."
      for i in $(seq 1 30); do
        STATUS="$(curl -sf "${RAY_JOB_ADDRESS}/api/jobs/${KILL_JOB_ID}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))")"
        if [[ "${STATUS}" == "SUCCEEDED" ]]; then
          echo "Kill job succeeded."
          ray job logs "${KILL_JOB_ID}" --address="${RAY_JOB_ADDRESS}" 2>/dev/null | tail -20
          break
        elif [[ "${STATUS}" == "FAILED" ]]; then
          echo "Kill job failed. Logs:" >&2
          ray job logs "${KILL_JOB_ID}" --address="${RAY_JOB_ADDRESS}" 2>/dev/null | tail -20
          break
        fi
        sleep 2
      done
    fi
  fi
fi

exit 0
