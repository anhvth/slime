#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TRAIN_PERF_RE = re.compile(r"train_metric_utils\.py:44 - perf \d+: (\{.*\})")
ROLLOUT_PERF_RE = re.compile(r"rollout\.py:732 - perf \d+: (\{.*\})")
TOKEN_USAGE_RE = re.compile(r"token usage:\s*([0-9]+(?:\.[0-9]+)?).*?#queue-req:\s*([0-9]+)")
E2E_LAT_RE = re.compile(r"'e2e_latency':\s*([0-9]+(?:\.[0-9]+)?)")
CLIENT_LAT_RE = re.compile(r"'client_http_latency':\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze privileged distillation training performance from log output.")
    parser.add_argument("--log", type=Path, required=True, help="Path to training log.")
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=0,
        help="Only analyze the last N lines (0 = analyze entire file).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable output.")
    return parser.parse_args()


def read_lines(path: Path, tail_lines: int) -> list[str]:
    if tail_lines > 0:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return list(deque(f, maxlen=tail_lines))
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_perf_dict(match: re.Match[str] | None) -> dict[str, Any] | None:
    if match is None:
        return None
    payload = match.group(1)
    try:
        value = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def build_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p50": percentile(sorted_values, 0.50),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
    }


def to_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def pick(summary: dict[str, dict[str, float | int] | None], key: str, stat: str) -> float | int | None:
    if key not in summary or summary[key] is None:
        return None
    return summary[key].get(stat)  # type: ignore[union-attr]


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def analyze(lines: list[str]) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    token_usage_values: list[float] = []
    queue_req_values: list[float] = []
    e2e_latency_values: list[float] = []
    client_latency_values: list[float] = []

    for raw_line in lines:
        line = strip_ansi(raw_line)

        train_perf = parse_perf_dict(TRAIN_PERF_RE.search(line))
        if train_perf is not None:
            train_rows.append(train_perf)

        rollout_perf = parse_perf_dict(ROLLOUT_PERF_RE.search(line))
        if rollout_perf is not None:
            rollout_rows.append(rollout_perf)

        token_match = TOKEN_USAGE_RE.search(line)
        if token_match:
            token_usage_values.append(float(token_match.group(1)))
            queue_req_values.append(float(token_match.group(2)))

        for lat in E2E_LAT_RE.findall(line):
            e2e_latency_values.append(float(lat))
        for lat in CLIENT_LAT_RE.findall(line):
            client_latency_values.append(float(lat))

    train_summary = {
        "perf/step_time": build_summary(to_float_list(train_rows, "perf/step_time")),
        "perf/train_wait_time": build_summary(to_float_list(train_rows, "perf/train_wait_time")),
        "perf/actor_train_time": build_summary(to_float_list(train_rows, "perf/actor_train_time")),
        "perf/wait_time_ratio": build_summary(to_float_list(train_rows, "perf/wait_time_ratio")),
        "perf/actor_train_tflops": build_summary(to_float_list(train_rows, "perf/actor_train_tflops")),
        "perf/actor_train_tok_per_s": build_summary(to_float_list(train_rows, "perf/actor_train_tok_per_s")),
    }

    rollout_summary = {
        "perf/rollout_time": build_summary(to_float_list(rollout_rows, "perf/rollout_time")),
        "perf/tokens_per_gpu_per_sec": build_summary(to_float_list(rollout_rows, "perf/tokens_per_gpu_per_sec")),
        "rollout/truncated_ratio": build_summary(to_float_list(rollout_rows, "rollout/truncated_ratio")),
        "rollout/response_len/mean": build_summary(to_float_list(rollout_rows, "rollout/response_len/mean")),
        "rollout/response_len/median": build_summary(to_float_list(rollout_rows, "rollout/response_len/median")),
    }

    sglang_summary = {
        "token_usage": build_summary(token_usage_values),
        "queue_req": build_summary(queue_req_values),
    }
    teacher_summary = {
        "teacher_e2e_latency": build_summary(e2e_latency_values),
        "client_http_latency": build_summary(client_latency_values),
    }

    findings: list[dict[str, str]] = []

    wait_ratio_mean = pick(train_summary, "perf/wait_time_ratio", "mean")
    if isinstance(wait_ratio_mean, (int, float)):
        if wait_ratio_mean >= 0.65:
            findings.append(
                {
                    "severity": "high",
                    "title": "Train is rollout-bound",
                    "evidence": f"mean wait_time_ratio={wait_ratio_mean:.3f}",
                }
            )
        elif wait_ratio_mean >= 0.50:
            findings.append(
                {
                    "severity": "medium",
                    "title": "Train has significant waiting",
                    "evidence": f"mean wait_time_ratio={wait_ratio_mean:.3f}",
                }
            )

    rollout_tps_mean = pick(rollout_summary, "perf/tokens_per_gpu_per_sec", "mean")
    if isinstance(rollout_tps_mean, (int, float)) and rollout_tps_mean < 180:
        findings.append(
            {
                "severity": "high",
                "title": "Rollout throughput is low",
                "evidence": f"mean tokens_per_gpu_per_sec={rollout_tps_mean:.1f}",
            }
        )

    token_usage_mean = pick(sglang_summary, "token_usage", "mean")
    queue_req_p95 = pick(sglang_summary, "queue_req", "p95")
    if isinstance(token_usage_mean, (int, float)) and isinstance(queue_req_p95, (int, float)):
        if token_usage_mean < 0.05 and queue_req_p95 <= 0.0:
            findings.append(
                {
                    "severity": "medium",
                    "title": "SGLang engines are under-filled",
                    "evidence": f"token_usage_mean={token_usage_mean:.3f}, queue_req_p95={queue_req_p95:.1f}",
                }
            )

    e2e_p95 = pick(teacher_summary, "teacher_e2e_latency", "p95")
    if isinstance(e2e_p95, (int, float)) and e2e_p95 > 10.0:
        findings.append(
            {
                "severity": "medium",
                "title": "Teacher latency has long tail",
                "evidence": f"teacher_e2e_latency_p95={e2e_p95:.2f}s",
            }
        )

    trunc_mean = pick(rollout_summary, "rollout/truncated_ratio", "mean")
    if isinstance(trunc_mean, (int, float)) and trunc_mean > 0.6:
        findings.append(
            {
                "severity": "medium",
                "title": "Responses are often max-length truncated",
                "evidence": f"truncated_ratio_mean={trunc_mean:.3f}",
            }
        )

    return {
        "line_count": len(lines),
        "train_samples": len(train_rows),
        "rollout_samples": len(rollout_rows),
        "sglang_samples": len(token_usage_values),
        "teacher_latency_samples": len(e2e_latency_values),
        "train_summary": train_summary,
        "rollout_summary": rollout_summary,
        "sglang_summary": sglang_summary,
        "teacher_summary": teacher_summary,
        "findings": findings,
    }


def print_human(report: dict[str, Any]) -> None:
    print("=== Training Perf Analysis ===")
    print(f"lines_analyzed: {report['line_count']}")
    print(
        "samples: "
        f"train={report['train_samples']}, rollout={report['rollout_samples']}, "
        f"sglang={report['sglang_samples']}, teacher_latency={report['teacher_latency_samples']}"
    )
    print("")

    def section(title: str, rows: list[tuple[str, str, int]]) -> None:
        print(title)
        for key, label, digits in rows:
            src = report["train_summary"] if key in report["train_summary"] else report["rollout_summary"]
            summary = src.get(key)
            if summary is None:
                print(f"  - {label}: n/a")
                continue
            print(
                f"  - {label}: mean={format_float(summary.get('mean'), digits)} "
                f"p95={format_float(summary.get('p95'), digits)} "
                f"min={format_float(summary.get('min'), digits)} "
                f"max={format_float(summary.get('max'), digits)}"
            )
        print("")

    section(
        "Train",
        [
            ("perf/step_time", "step_time_s", 2),
            ("perf/train_wait_time", "train_wait_time_s", 2),
            ("perf/actor_train_time", "actor_train_time_s", 2),
            ("perf/wait_time_ratio", "wait_time_ratio", 3),
            ("perf/actor_train_tflops", "actor_train_tflops", 1),
            ("perf/actor_train_tok_per_s", "actor_train_tok_per_s", 1),
        ],
    )

    section(
        "Rollout",
        [
            ("perf/rollout_time", "rollout_time_s", 2),
            ("perf/tokens_per_gpu_per_sec", "tokens_per_gpu_per_sec", 1),
            ("rollout/truncated_ratio", "truncated_ratio", 3),
            ("rollout/response_len/mean", "response_len_mean", 1),
            ("rollout/response_len/median", "response_len_median", 1),
        ],
    )

    sglang = report["sglang_summary"]
    print("SGLang")
    for key in ("token_usage", "queue_req"):
        summary = sglang.get(key)
        if summary is None:
            print(f"  - {key}: n/a")
            continue
        print(
            f"  - {key}: mean={format_float(summary.get('mean'), 3)} "
            f"p95={format_float(summary.get('p95'), 3)} "
            f"max={format_float(summary.get('max'), 3)}"
        )
    print("")

    teacher = report["teacher_summary"]
    print("Teacher")
    for key in ("teacher_e2e_latency", "client_http_latency"):
        summary = teacher.get(key)
        if summary is None:
            print(f"  - {key}: n/a")
            continue
        print(
            f"  - {key}_s: mean={format_float(summary.get('mean'), 2)} "
            f"p95={format_float(summary.get('p95'), 2)} "
            f"max={format_float(summary.get('max'), 2)}"
        )
    print("")

    findings: list[dict[str, str]] = report["findings"]
    print("Findings")
    if not findings:
        print("  - none")
        return
    for finding in findings:
        print(f"  - [{finding['severity']}] {finding['title']}: {finding['evidence']}")


def main() -> None:
    args = parse_args()
    if not args.log.is_file():
        raise SystemExit(f"Missing log file: {args.log}")
    if args.tail_lines < 0:
        raise SystemExit("--tail-lines must be >= 0")

    lines = read_lines(args.log, args.tail_lines)
    report = analyze(lines)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
