#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
EXP_DIR="${REPO_ROOT}/my_exps/opd-397-32b"

TRAIN_LOG="${TRAIN_LOG:-${EXP_DIR}/logs/training_async_distill_active.log}"
OUT_PARENT="${OUT_PARENT:-${EXP_DIR}/logs/perf_reports}"
TAIL_LINES="${TAIL_LINES:-8000}"
GPU_SAMPLES="${GPU_SAMPLES:-3}"
GPU_INTERVAL_SEC="${GPU_INTERVAL_SEC:-8}"

[[ -f "${TRAIN_LOG}" ]] || {
  echo "Missing training log: ${TRAIN_LOG}" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required" >&2
  exit 1
}

TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${OUT_PARENT}/${TS}"
mkdir -p "${OUT_DIR}"

ANALYZE_PY="${SCRIPT_DIR}/analyze_training_perf.py"
GPU_PY="${SCRIPT_DIR}/ray_gpu_snapshot.py"
[[ -f "${ANALYZE_PY}" ]] || { echo "Missing analyzer: ${ANALYZE_PY}" >&2; exit 1; }
[[ -f "${GPU_PY}" ]] || { echo "Missing GPU snapshot tool: ${GPU_PY}" >&2; exit 1; }

echo "[perf] writing report to ${OUT_DIR}"

python3 "${ANALYZE_PY}" --log "${TRAIN_LOG}" --tail-lines "${TAIL_LINES}" > "${OUT_DIR}/training_perf.txt"
python3 "${ANALYZE_PY}" --log "${TRAIN_LOG}" --tail-lines "${TAIL_LINES}" --json > "${OUT_DIR}/training_perf.json"

python3 "${GPU_PY}" --samples "${GPU_SAMPLES}" --interval-sec "${GPU_INTERVAL_SEC}" > "${OUT_DIR}/ray_gpu_snapshot.txt"
python3 "${GPU_PY}" --samples "${GPU_SAMPLES}" --interval-sec "${GPU_INTERVAL_SEC}" --json > "${OUT_DIR}/ray_gpu_snapshot.json"

python3 - "${OUT_DIR}/training_perf.json" "${OUT_DIR}/ray_gpu_snapshot.json" > "${OUT_DIR}/summary.md" <<'PY'
import json
import pathlib
import sys

perf = json.loads(pathlib.Path(sys.argv[1]).read_text())
gpu = json.loads(pathlib.Path(sys.argv[2]).read_text())

def pick(report, section, key, stat):
    row = report.get(section, {}).get(key)
    if not row:
        return None
    return row.get(stat)

wait_ratio = pick(perf, "train_summary", "perf/wait_time_ratio", "mean")
rollout_tps = pick(perf, "rollout_summary", "perf/tokens_per_gpu_per_sec", "mean")
e2e_p95 = pick(perf, "teacher_summary", "teacher_e2e_latency", "p95")
trunc = pick(perf, "rollout_summary", "rollout/truncated_ratio", "mean")
local_gpu_mean = gpu.get("aggregate", {}).get("local_gpu_util_mean_across_samples")
cluster_gpu_used = gpu.get("aggregate", {}).get("cluster_gpu_used_mean_across_samples")
cluster_gpu_total = gpu.get("aggregate", {}).get("cluster_gpu_total_mean_across_samples")

print("# Privileged Training Perf Snapshot")
print("")
print("## Headline Metrics")
print(f"- train_wait_ratio_mean: {wait_ratio:.3f}" if wait_ratio is not None else "- train_wait_ratio_mean: n/a")
print(f"- rollout_tokens_per_gpu_per_sec_mean: {rollout_tps:.1f}" if rollout_tps is not None else "- rollout_tokens_per_gpu_per_sec_mean: n/a")
print(f"- teacher_e2e_latency_p95_s: {e2e_p95:.2f}" if e2e_p95 is not None else "- teacher_e2e_latency_p95_s: n/a")
print(f"- rollout_truncated_ratio_mean: {trunc:.3f}" if trunc is not None else "- rollout_truncated_ratio_mean: n/a")
print(
    f"- cluster_logical_gpu_used_mean: {cluster_gpu_used:.1f}/{cluster_gpu_total:.1f}"
    if cluster_gpu_used is not None and cluster_gpu_total is not None
    else "- cluster_logical_gpu_used_mean: n/a"
)
print(f"- local_gpu_util_mean_percent: {local_gpu_mean:.1f}" if local_gpu_mean is not None else "- local_gpu_util_mean_percent: n/a")
print("")
print("## Findings")
findings = perf.get("findings", [])
if not findings:
    print("- none")
else:
    for finding in findings:
        sev = finding.get("severity", "info")
        title = finding.get("title", "")
        evidence = finding.get("evidence", "")
        print(f"- [{sev}] {title}: {evidence}")
PY

echo "[perf] done"
echo "[perf] summary: ${OUT_DIR}/summary.md"
echo "[perf] full: ${OUT_DIR}/training_perf.txt, ${OUT_DIR}/ray_gpu_snapshot.txt"
