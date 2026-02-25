#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

RAY_GPU_RE = re.compile(
    r"([0-9.]+)/([0-9.]+)\s+GPU\s+\(([0-9.]+)\s+used of\s+([0-9.]+)\s+reserved in placement groups\)"
)
RAY_CPU_RE = re.compile(
    r"([0-9.]+)/([0-9.]+)\s+CPU\s+\(([0-9.]+)\s+used of\s+([0-9.]+)\s+reserved in placement groups\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect cluster/local GPU utilization snapshots.")
    parser.add_argument("--samples", type=int, default=1, help="Number of snapshots to collect.")
    parser.add_argument("--interval-sec", type=int, default=10, help="Seconds between snapshots.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file output path.")
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def parse_nvidia_smi() -> dict[str, Any]:
    query = "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "nvidia-smi failed", "gpus": []}

    rows: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parts = [p.strip() for p in raw_line.strip().split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "gpu_util": float(parts[1]),
                    "mem_util": float(parts[2]),
                    "mem_used_gb": float(parts[3]) / 1024.0,
                    "mem_total_gb": float(parts[4]) / 1024.0,
                    "power_w": float(parts[5]) if parts[5] not in {"N/A", "[N/A]"} else None,
                }
            )
        except ValueError:
            continue
    return {"error": "", "gpus": rows}


def parse_ray_status() -> dict[str, Any]:
    result = subprocess.run(["ray", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "ray status failed"}

    text = result.stdout
    gpu_match = RAY_GPU_RE.search(text)
    cpu_match = RAY_CPU_RE.search(text)
    data: dict[str, Any] = {"error": ""}

    if gpu_match:
        data.update(
            {
                "cluster_gpu_used": float(gpu_match.group(1)),
                "cluster_gpu_total": float(gpu_match.group(2)),
                "cluster_gpu_direct_used": float(gpu_match.group(3)),
                "cluster_gpu_reserved": float(gpu_match.group(4)),
            }
        )
    if cpu_match:
        data.update(
            {
                "cluster_cpu_used": float(cpu_match.group(1)),
                "cluster_cpu_total": float(cpu_match.group(2)),
                "cluster_cpu_direct_used": float(cpu_match.group(3)),
                "cluster_cpu_reserved": float(cpu_match.group(4)),
            }
        )
    return data


def summarize_local_gpus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "gpu_count": 0,
            "gpu_util_mean": 0.0,
            "gpu_util_p95": 0.0,
            "mem_used_gb_mean": 0.0,
            "mem_used_gb_p95": 0.0,
        }
    gpu_utils = sorted(float(row["gpu_util"]) for row in rows)
    mem_used = sorted(float(row["mem_used_gb"]) for row in rows)
    return {
        "gpu_count": len(rows),
        "gpu_util_mean": sum(gpu_utils) / len(gpu_utils),
        "gpu_util_p95": percentile(gpu_utils, 0.95),
        "mem_used_gb_mean": sum(mem_used) / len(mem_used),
        "mem_used_gb_p95": percentile(mem_used, 0.95),
    }


def collect_snapshot() -> dict[str, Any]:
    ts = time.time()
    local = parse_nvidia_smi()
    ray_status = parse_ray_status()
    local_summary = summarize_local_gpus(local["gpus"])
    return {
        "timestamp": ts,
        "local": {
            **local_summary,
            "error": local["error"],
            "gpus": local["gpus"],
        },
        "cluster": ray_status,
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")
    if args.interval_sec <= 0:
        raise SystemExit("--interval-sec must be > 0")

    shots: list[dict[str, Any]] = []
    for i in range(args.samples):
        shots.append(collect_snapshot())
        if i + 1 < args.samples:
            time.sleep(args.interval_sec)

    local_gpu_util_means = [s["local"]["gpu_util_mean"] for s in shots]
    local_mem_used_means = [s["local"]["mem_used_gb_mean"] for s in shots]
    cluster_gpu_used = [
        s["cluster"].get("cluster_gpu_used")
        for s in shots
        if s["cluster"].get("cluster_gpu_used") is not None
    ]
    cluster_gpu_total = [
        s["cluster"].get("cluster_gpu_total")
        for s in shots
        if s["cluster"].get("cluster_gpu_total") is not None
    ]

    report = {
        "samples": shots,
        "aggregate": {
            "sample_count": len(shots),
            "local_gpu_util_mean_across_samples": sum(local_gpu_util_means) / len(local_gpu_util_means),
            "local_mem_used_gb_mean_across_samples": sum(local_mem_used_means) / len(local_mem_used_means),
            "cluster_gpu_used_mean_across_samples": (
                sum(cluster_gpu_used) / len(cluster_gpu_used) if cluster_gpu_used else None
            ),
            "cluster_gpu_total_mean_across_samples": (
                sum(cluster_gpu_total) / len(cluster_gpu_total) if cluster_gpu_total else None
            ),
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=== GPU Snapshot ===")
    for idx, shot in enumerate(shots, start=1):
        local = shot["local"]
        cluster = shot["cluster"]
        cluster_gpu_used = cluster.get("cluster_gpu_used")
        cluster_gpu_total = cluster.get("cluster_gpu_total")
        cluster_line = "cluster_gpu=n/a"
        if cluster_gpu_used is not None and cluster_gpu_total is not None:
            cluster_line = f"cluster_gpu={cluster_gpu_used:.1f}/{cluster_gpu_total:.1f}"
        print(
            f"sample={idx} local_gpu_util_mean={local['gpu_util_mean']:.1f}% "
            f"local_mem_used_gb_mean={local['mem_used_gb_mean']:.1f} {cluster_line}"
        )

    agg = report["aggregate"]
    print("")
    print(
        "aggregate: "
        f"local_gpu_util_mean={agg['local_gpu_util_mean_across_samples']:.1f}% "
        f"local_mem_used_gb_mean={agg['local_mem_used_gb_mean_across_samples']:.1f} "
        f"cluster_gpu_used_mean={agg['cluster_gpu_used_mean_across_samples']}"
    )


if __name__ == "__main__":
    main()
