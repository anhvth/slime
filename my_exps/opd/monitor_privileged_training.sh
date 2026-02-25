#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://100.96.4.38:8265}"
TEACHER_URL="${TEACHER_URL:-http://worker-30:13142/generate}"
TRAIN_LOG="${TRAIN_LOG:-${SCRIPT_DIR}/logs/training_async_distill_active.log}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/opd/student_async_distill_privileged_from_onpolicy-distill-qwen3.5-step1600}"
MONITOR_LOG="${MONITOR_LOG:-${LOG_DIR}/monitor_privileged_training.log}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
MAX_LOG_STALE_SEC="${MAX_LOG_STALE_SEC:-1200}"
MIN_ALIVE_NODES="${MIN_ALIVE_NODES:-15}"
TAIL_LINES="${TAIL_LINES:-500}"
JOB_SUBMISSION_ID="${JOB_SUBMISSION_ID:-}"
RUN_ONCE="${RUN_ONCE:-0}"
ENABLE_PERF_PROBE="${ENABLE_PERF_PROBE:-1}"
PERF_PROBE_EVERY_LOOPS="${PERF_PROBE_EVERY_LOOPS:-6}"
PERF_PROBE_TAIL_LINES="${PERF_PROBE_TAIL_LINES:-8000}"
GPU_SNAPSHOT_EVERY_LOOPS="${GPU_SNAPSHOT_EVERY_LOOPS:-12}"
PERF_ALERT_WAIT_RATIO="${PERF_ALERT_WAIT_RATIO:-0.65}"
PERF_ALERT_TPS="${PERF_ALERT_TPS:-180}"
PERF_ALERT_TEACHER_P95="${PERF_ALERT_TEACHER_P95:-10}"
PERF_ANALYZER_PY="${SCRIPT_DIR}/perf_tools/analyze_training_perf.py"
GPU_SNAPSHOT_PY="${SCRIPT_DIR}/perf_tools/ray_gpu_snapshot.py"

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Missing required command: ${cmd}" >&2
    exit 1
  }
}

require_positive_int() {
  local value="$1"
  local name="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "${name} must be a positive integer, got '${value}'" >&2
    exit 1
  }
  (( value > 0 )) || {
    echo "${name} must be > 0, got '${value}'" >&2
    exit 1
  }
}

log_line() {
  local level="$1"
  local message="$2"
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "${message}" | tee -a "${MONITOR_LOG}"
}

select_submission() {
  python3 - "${JOB_SUBMISSION_ID}" "${SAVE_ROOT}" <<'PY'
import json
import subprocess
import sys

requested_id = sys.argv[1]
save_root = sys.argv[2]
rows = json.loads(subprocess.check_output(["ray", "list", "jobs", "--format", "json"], text=True))

if requested_id:
    for row in rows:
        if row.get("submission_id") == requested_id:
            print(f"{row.get('submission_id','')}|{row.get('status','')}|{row.get('entrypoint','')}")
            raise SystemExit(0)
    print(f"{requested_id}|NOT_FOUND|")
    raise SystemExit(0)

running = [row for row in rows if row.get("status") == "RUNNING"]

def score(row):
    ep = row.get("entrypoint", "")
    s = 0
    if "train_async.py" in ep:
        s += 4
    if "opd_topk_loss_plugin.distill_topk_custom_loss" in ep:
        s += 3
    if save_root and save_root in ep:
        s += 3
    if "--rm-url" in ep:
        s += 1
    return s

if not running:
    print("||")
    raise SystemExit(0)

best = max(running, key=score)
print(f"{best.get('submission_id','')}|{best.get('status','')}|{best.get('entrypoint','')}")
PY
}

alive_nodes_count() {
  python3 - <<'PY'
import json
import subprocess

rows = json.loads(subprocess.check_output(["ray", "list", "nodes", "--format", "json"], text=True))
alive = sum(1 for row in rows if row.get("state") == "ALIVE")
total = len(rows)
gpus = sum(float(row.get("resources_total", {}).get("GPU", 0)) for row in rows if row.get("state") == "ALIVE")
print(f"{alive}|{total}|{int(gpus)}")
PY
}

checkpoint_progress() {
  local root="$1"
  if [[ -f "${root}/latest_checkpointed_iteration.txt" ]]; then
    tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt"
    return 0
  fi

  local latest_iter
  latest_iter="$(find "${root}" -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${latest_iter}" ]]; then
    basename "${latest_iter}"
    return 0
  fi

  echo "none-yet"
}

recent_critical_matches() {
  rg -n -i \
    "Traceback \\(most recent call last\\)|OutOfMemoryError|CUDA out of memory|NCCL error|ConnectionRefusedError|Segmentation fault|ray.exceptions|Job '.*' failed|RuntimeError:.*(CUDA|NCCL|CUBLAS)" \
    || true
}

require_cmd ray
require_cmd curl
require_cmd python3
require_cmd rg
require_cmd stat
require_cmd tail
require_positive_int "${INTERVAL_SEC}" "INTERVAL_SEC"
require_positive_int "${MAX_LOG_STALE_SEC}" "MAX_LOG_STALE_SEC"
require_positive_int "${MIN_ALIVE_NODES}" "MIN_ALIVE_NODES"
require_positive_int "${TAIL_LINES}" "TAIL_LINES"
if [[ "${ENABLE_PERF_PROBE}" == "1" ]]; then
  require_positive_int "${PERF_PROBE_EVERY_LOOPS}" "PERF_PROBE_EVERY_LOOPS"
  require_positive_int "${PERF_PROBE_TAIL_LINES}" "PERF_PROBE_TAIL_LINES"
  require_positive_int "${GPU_SNAPSHOT_EVERY_LOOPS}" "GPU_SNAPSHOT_EVERY_LOOPS"
  [[ -f "${PERF_ANALYZER_PY}" ]] || {
    echo "Missing perf analyzer: ${PERF_ANALYZER_PY}" >&2
    exit 1
  }
  [[ -f "${GPU_SNAPSHOT_PY}" ]] || {
    echo "Missing GPU snapshot tool: ${GPU_SNAPSHOT_PY}" >&2
    exit 1
  }
fi

TEACHER_BASE="${TEACHER_URL%/generate}"
[[ "${TEACHER_BASE}" != "${TEACHER_URL}" ]] || {
  echo "TEACHER_URL must end with /generate: ${TEACHER_URL}" >&2
  exit 1
}

log_line "INFO" "monitor started (ray=${RAY_JOB_ADDRESS}, teacher=${TEACHER_URL}, save_root=${SAVE_ROOT})"

LAST_SUB_ID=""
LAST_SCANNED_LINE=0
LOOP_COUNT=0
if [[ -f "${TRAIN_LOG}" ]]; then
  LAST_SCANNED_LINE="$(wc -l < "${TRAIN_LOG}" | tr -d '[:space:]')"
fi
while true; do
  LOOP_COUNT=$((LOOP_COUNT + 1))
  line="$(select_submission 2>/dev/null || true)"
  SUB_ID="${line%%|*}"
  rest="${line#*|}"
  JOB_STATUS="${rest%%|*}"
  ENTRYPOINT="${rest#*|}"

  if [[ -z "${SUB_ID}" ]]; then
    log_line "ALERT" "no RUNNING submission job found for distill workload"
  elif [[ "${JOB_STATUS}" == "NOT_FOUND" ]]; then
    log_line "ALERT" "submission ${SUB_ID} not found"
  elif [[ "${JOB_STATUS}" != "RUNNING" ]]; then
    log_line "ALERT" "submission ${SUB_ID} status=${JOB_STATUS}"
  else
    log_line "OK" "submission ${SUB_ID} RUNNING"
  fi

  if ! ray job list --address="${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
    log_line "ALERT" "ray job server unreachable: ${RAY_JOB_ADDRESS}"
  fi

  NODES_LINE="$(alive_nodes_count 2>/dev/null || echo "0|0|0")"
  ALIVE_NODES="${NODES_LINE%%|*}"
  rest="${NODES_LINE#*|}"
  TOTAL_NODES="${rest%%|*}"
  ALIVE_GPUS="${rest#*|}"
  if (( ALIVE_NODES < MIN_ALIVE_NODES )); then
    log_line "ALERT" "cluster health: alive_nodes=${ALIVE_NODES}/${TOTAL_NODES}, alive_gpus=${ALIVE_GPUS} (threshold=${MIN_ALIVE_NODES})"
  else
    log_line "OK" "cluster health: alive_nodes=${ALIVE_NODES}/${TOTAL_NODES}, alive_gpus=${ALIVE_GPUS}"
  fi

  HEALTH_CODE="$(curl -sS --connect-timeout 2 --max-time 8 -o /dev/null -w '%{http_code}' "${TEACHER_BASE}/health_generate" || true)"
  INFO_CODE="$(curl -sS --connect-timeout 2 --max-time 8 -o /dev/null -w '%{http_code}' "${TEACHER_BASE}/get_model_info" || true)"
  if [[ "${HEALTH_CODE}" != "200" || "${INFO_CODE}" != "200" ]]; then
    log_line "ALERT" "teacher health failed: health_generate=${HEALTH_CODE}, get_model_info=${INFO_CODE}"
  else
    log_line "OK" "teacher health: health_generate=${HEALTH_CODE}, get_model_info=${INFO_CODE}"
  fi

  if [[ -f "${TRAIN_LOG}" ]]; then
    LOG_MTIME="$(stat -c %Y "${TRAIN_LOG}" 2>/dev/null || echo 0)"
    NOW_TS="$(date +%s)"
    LOG_AGE="$((NOW_TS - LOG_MTIME))"
    if (( LOG_AGE > MAX_LOG_STALE_SEC )); then
      log_line "ALERT" "train log stale: ${TRAIN_LOG} (age=${LOG_AGE}s)"
    else
      log_line "OK" "train log fresh: ${TRAIN_LOG} (age=${LOG_AGE}s)"
    fi

    CURRENT_LINES="$(wc -l < "${TRAIN_LOG}" | tr -d '[:space:]')"
    if [[ -z "${CURRENT_LINES}" || "${CURRENT_LINES}" -lt 0 ]]; then
      CURRENT_LINES=0
    fi
    if (( CURRENT_LINES < LAST_SCANNED_LINE )); then
      LAST_SCANNED_LINE=0
    fi
    if (( CURRENT_LINES > LAST_SCANNED_LINE )); then
      NEW_LOG_TEXT="$(sed -n "$((LAST_SCANNED_LINE + 1)),${CURRENT_LINES}p" "${TRAIN_LOG}" || true)"
      CRITICAL_LINES="$(printf '%s\n' "${NEW_LOG_TEXT}" | recent_critical_matches)"
      if [[ -n "${CRITICAL_LINES}" ]]; then
        log_line "ALERT" "critical patterns detected in newly appended log lines:"
        while IFS= read -r matched; do
          [[ -n "${matched}" ]] && log_line "ALERT" "  ${matched}"
        done <<< "${CRITICAL_LINES}"
      fi
      LAST_SCANNED_LINE="${CURRENT_LINES}"
    fi
  else
    log_line "ALERT" "train log missing: ${TRAIN_LOG}"
  fi

  if [[ -d "${SAVE_ROOT}" ]]; then
    CKPT="$(checkpoint_progress "${SAVE_ROOT}")"
    log_line "OK" "checkpoint progress: ${CKPT}"
  else
    log_line "OK" "checkpoint progress: save root not created yet (${SAVE_ROOT})"
  fi

  if [[ "${SUB_ID}" != "" && "${ENTRYPOINT}" != "" && "${SUB_ID}" != "${LAST_SUB_ID}" ]]; then
    log_line "INFO" "entrypoint: ${ENTRYPOINT:0:320}..."
    LAST_SUB_ID="${SUB_ID}"
  fi

  if [[ "${ENABLE_PERF_PROBE}" == "1" ]]; then
    if [[ -f "${TRAIN_LOG}" ]] && (( LOOP_COUNT % PERF_PROBE_EVERY_LOOPS == 0 )); then
      PERF_JSON="$(python3 "${PERF_ANALYZER_PY}" --log "${TRAIN_LOG}" --tail-lines "${PERF_PROBE_TAIL_LINES}" --json 2>/dev/null || true)"
      if [[ -n "${PERF_JSON}" ]]; then
        PERF_LINE="$(python3 -c '
import json
import sys

def pick(d, section, key, stat):
    row = d.get(section, {}).get(key)
    return None if not row else row.get(stat)

data = json.load(sys.stdin)
wait_ratio = pick(data, "train_summary", "perf/wait_time_ratio", "mean")
tps = pick(data, "rollout_summary", "perf/tokens_per_gpu_per_sec", "mean")
teacher_p95 = pick(data, "teacher_summary", "teacher_e2e_latency", "p95")
trunc = pick(data, "rollout_summary", "rollout/truncated_ratio", "mean")
if all(v is not None for v in (wait_ratio, tps, teacher_p95, trunc)):
    print(f"wait_ratio={wait_ratio:.3f} rollout_tps={tps:.1f} teacher_p95={teacher_p95:.2f}s trunc_ratio={trunc:.3f}")
else:
    print("insufficient samples")
' <<< "${PERF_JSON}")"
        log_line "INFO" "perf summary: ${PERF_LINE}"

        ALERT_LINES="$(python3 -c '
import json
import sys

data = json.load(sys.stdin)
wait_th = float(sys.argv[1])
tps_th = float(sys.argv[2])
teacher_th = float(sys.argv[3])

def pick(section, key, stat):
    row = data.get(section, {}).get(key)
    return None if not row else row.get(stat)

wait_ratio = pick("train_summary", "perf/wait_time_ratio", "mean")
rollout_tps = pick("rollout_summary", "perf/tokens_per_gpu_per_sec", "mean")
teacher_p95 = pick("teacher_summary", "teacher_e2e_latency", "p95")

if wait_ratio is not None and float(wait_ratio) >= wait_th:
    print(f"wait_time_ratio={float(wait_ratio):.3f} >= {wait_th:.3f}")
if rollout_tps is not None and float(rollout_tps) < tps_th:
    print(f"rollout_tps={float(rollout_tps):.1f} < {tps_th:.1f}")
if teacher_p95 is not None and float(teacher_p95) > teacher_th:
    print(f"teacher_p95={float(teacher_p95):.2f}s > {teacher_th:.2f}s")
' "${PERF_ALERT_WAIT_RATIO}" "${PERF_ALERT_TPS}" "${PERF_ALERT_TEACHER_P95}" <<< "${PERF_JSON}")"
        if [[ -n "${ALERT_LINES}" ]]; then
          while IFS= read -r alert_line; do
            [[ -n "${alert_line}" ]] && log_line "ALERT" "perf bottleneck: ${alert_line}"
          done <<< "${ALERT_LINES}"
        fi
      fi
    fi

    if (( LOOP_COUNT % GPU_SNAPSHOT_EVERY_LOOPS == 0 )); then
      GPU_JSON="$(python3 "${GPU_SNAPSHOT_PY}" --samples 1 --interval-sec 1 --json 2>/dev/null || true)"
      if [[ -n "${GPU_JSON}" ]]; then
        GPU_LINE="$(python3 -c '
import json
import sys

data = json.load(sys.stdin)
agg = data.get("aggregate", {})
local_gpu_mean = agg.get("local_gpu_util_mean_across_samples")
local_mem_mean = agg.get("local_mem_used_gb_mean_across_samples")
cluster_gpu_used = agg.get("cluster_gpu_used_mean_across_samples")
cluster_gpu_total = agg.get("cluster_gpu_total_mean_across_samples")
if local_gpu_mean is None or local_mem_mean is None:
    print("insufficient samples")
else:
    cluster = "cluster_gpu=n/a"
    if cluster_gpu_used is not None and cluster_gpu_total is not None:
        cluster = f"cluster_gpu={cluster_gpu_used:.1f}/{cluster_gpu_total:.1f}"
    print(f"local_gpu_util_mean={local_gpu_mean:.1f}% local_mem_used_gb_mean={local_mem_mean:.1f} {cluster}")
' <<< "${GPU_JSON}")"
        log_line "INFO" "gpu snapshot: ${GPU_LINE}"
      fi
    fi
  fi

  if [[ "${RUN_ONCE}" == "1" ]]; then
    log_line "INFO" "run_once=1 completed; exiting"
    break
  fi
  sleep "${INTERVAL_SEC}"
done
