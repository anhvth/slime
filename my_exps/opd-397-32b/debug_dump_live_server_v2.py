from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer


ROLL_OUT_RE = re.compile(r"rollout_(\d+)")


@dataclass
class ServerState:
    runs_root: Path
    default_tokenizer_path: str


STATE = ServerState(
    runs_root=Path(__file__).resolve().parents[2] / "outputs",
    default_tokenizer_path="",
)

app = FastAPI(title="OPD Distill Debug Live Viewer v2", version="2.0")


def _tensor_to_list(value: Any, *, dtype: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if dtype == "int":
            return value.detach().cpu().to(dtype=torch.long).tolist()
        if dtype == "float":
            return value.detach().cpu().to(dtype=torch.float32).tolist()
        raise ValueError(f"Unsupported dtype request: {dtype}")
    if isinstance(value, (list, tuple)):
        if dtype == "int":
            return [int(v) for v in value]
        if dtype == "float":
            return [float(v) for v in value]
    raise TypeError(f"Unsupported value for tensor conversion: {type(value)}")


def _tensor_to_2d_float(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _tensor_to_2d_long(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def _discover_runs() -> list[Path]:
    root = STATE.runs_root
    if not root.exists():
        return []
    run_dirs = sorted({p.parent for p in root.rglob("distill_debug_dumps") if p.is_dir()})
    return run_dirs


def _run_id(run_dir: Path) -> str:
    return str(run_dir.relative_to(STATE.runs_root))


def _run_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for run_dir in _discover_runs():
        mapping[_run_id(run_dir)] = run_dir
    return mapping


def _resolve_run(run: str) -> Path:
    mapping = _run_map()
    if run not in mapping:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run}")
    return mapping[run]


def _list_dump_files(run_dir: Path) -> list[Path]:
    dump_dir = run_dir / "distill_debug_dumps"
    if not dump_dir.exists():
        return []
    files = [p for p in dump_dir.glob("distill_debug_*.pt") if p.is_file()]
    files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return files


def _extract_rollout_id(path: Path) -> int:
    m = ROLL_OUT_RE.search(path.name)
    if m:
        return int(m.group(1))
    return -1


@lru_cache(maxsize=256)
def _load_dump_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns  # included in cache key so cache invalidates on updates
    data = torch.load(path_str, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected dump payload type: {type(data)}")
    return data


def _load_dump(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return _load_dump_cached(str(path), int(stat.st_mtime_ns))


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _resolve_tokenizer_path(run_dir: Path, requested: str) -> tuple[str, str]:
    candidates: list[str] = []
    req = requested.strip()
    if req:
        candidates.append(req)
    if STATE.default_tokenizer_path:
        candidates.append(STATE.default_tokenizer_path)

    env_hint = run_dir / "tokenizer"
    if env_hint.exists():
        candidates.append(str(env_hint))

    unique_candidates: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    errors: list[str] = []
    for candidate in unique_candidates:
        try:
            _load_tokenizer(candidate)
            return candidate, ""
        except Exception as exc:  # pragma: no cover - best effort path probing
            errors.append(f"{candidate}: {exc}")
    return "", "; ".join(errors[-2:])


def _decode_ids(
    ids: list[int],
    tokenizer_path: str,
    *,
    max_tokens: int = 1024,
    max_chars: int = 8000,
) -> str:
    clipped = ids[:max_tokens]
    if not clipped:
        return ""
    if tokenizer_path:
        tokenizer = _load_tokenizer(tokenizer_path)
        text = tokenizer.decode(clipped, skip_special_tokens=False)
        if len(text) > max_chars:
            return text[: max_chars - 17] + "\n...[truncated]..."
        return text
    raw = " ".join(str(x) for x in clipped)
    if len(raw) > max_chars:
        return raw[: max_chars - 17] + "\n...[truncated]..."
    return raw


def _decode_token_map(tokenizer_path: str, token_ids: list[int]) -> dict[str, str]:
    if not tokenizer_path:
        return {}
    tokenizer = _load_tokenizer(tokenizer_path)
    decoded: dict[str, str] = {}
    for token_id in token_ids:
        try:
            token = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            token = "<decode-error>"
        decoded[str(int(token_id))] = token
    return decoded


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sample_summary(record: dict[str, Any], index: int) -> dict[str, Any]:
    response_length = _safe_int(record.get("response_length"), 0)
    response_start = _safe_int(record.get("response_start"), 0)
    position_indices = _tensor_to_list(record.get("position_indices", []), dtype="int")
    return {
        "idx": index,
        "sample_index": _safe_int(record.get("sample_index"), -1),
        "microbatch_sample_index": _safe_int(record.get("microbatch_sample_index"), index),
        "response_start": response_start,
        "response_length": response_length,
        "num_positions": len(position_indices),
    }


def _build_record_payload(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    record_idx: int,
    tokenizer_path: str,
) -> dict[str, Any]:
    records = payload.get("records") or []
    if not records:
        raise HTTPException(status_code=404, detail="No records in selected dump file")
    if record_idx < 0 or record_idx >= len(records):
        raise HTTPException(status_code=400, detail=f"record_idx out of range: {record_idx}")

    record = records[record_idx]
    student_ids = _tensor_to_list(record.get("student_input_ids", []), dtype="int")
    teacher_ids = _tensor_to_list(record.get("teacher_input_ids", []), dtype="int")
    response_start = _safe_int(record.get("response_start"), 0)
    response_length = _safe_int(record.get("response_length"), 0)
    response_end = max(response_start, response_start + response_length)
    position_indices = _tensor_to_list(record.get("position_indices", []), dtype="int")

    teacher_start = _safe_int(record.get("teacher_logprob_start_len"), response_start)
    recipe = payload.get("recipe") or {}
    mode = str(payload.get("mode") or recipe.get("distill_loss_mode") or "unknown").lower()

    context = {
        "student_prompt": _decode_ids(student_ids[:response_start], tokenizer_path),
        "student_response": _decode_ids(student_ids[response_start:response_end], tokenizer_path),
        "teacher_prompt": _decode_ids(teacher_ids[:teacher_start], tokenizer_path),
        "teacher_response": _decode_ids(teacher_ids[teacher_start : teacher_start + response_length], tokenizer_path),
        "student_input_length": len(student_ids),
        "teacher_input_length": len(teacher_ids),
        "response_start": response_start,
        "response_length": response_length,
        "teacher_logprob_start_len": teacher_start,
    }

    forward_kl: list[float] = []
    reverse_kl: list[float] = []
    jsd_vals: list[float] = []
    mass: dict[str, list[float]] = {}
    topk: dict[str, Any] = {}
    recompute: dict[str, float] = {}

    if mode in {"fkl", "mixed", "jsd"}:
        teacher_lp = _tensor_to_2d_float(record.get("teacher_topk_logprobs"))
        student_lp = _tensor_to_2d_float(record.get("student_topk_logprobs"))
        teacher_ids_2d = _tensor_to_2d_long(record.get("teacher_topk_token_ids"))

        teacher_prob = torch.softmax(teacher_lp, dim=-1)
        student_prob = torch.softmax(student_lp, dim=-1)
        teacher_top1_mass = teacher_prob.max(dim=-1).values
        student_top1_mass = student_prob.max(dim=-1).values
        tv_distance = 0.5 * torch.abs(teacher_prob - student_prob).sum(dim=-1)

        forward_tensor = _tensor_to_2d_float(record.get("forward_kl")).reshape(-1)
        reverse_tensor = _tensor_to_2d_float(record.get("reverse_kl")).reshape(-1)
        jsd_raw = record.get("jsd")
        jsd_tensor = None if jsd_raw is None else _tensor_to_2d_float(jsd_raw).reshape(-1)

        t_norm = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
        s_norm = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)
        recomputed_fkl = (teacher_prob * (t_norm - s_norm)).sum(dim=-1)
        recomputed_rkl = (student_prob * (s_norm - t_norm)).sum(dim=-1)
        recompute["forward_kl_max_abs_diff"] = float((recomputed_fkl - forward_tensor).abs().max().item())
        recompute["reverse_kl_max_abs_diff"] = float((recomputed_rkl - reverse_tensor).abs().max().item())

        if jsd_tensor is not None:
            beta = float(recipe.get("opd_jsd_beta", 0.5))
            mix = beta * teacher_prob + (1.0 - beta) * student_prob
            log_mix = mix.clamp_min(1e-12).log()
            recomputed_jsd = beta * (teacher_prob * (t_norm - log_mix)).sum(dim=-1)
            recomputed_jsd += (1.0 - beta) * (student_prob * (s_norm - log_mix)).sum(dim=-1)
            recompute["jsd_max_abs_diff"] = float((recomputed_jsd - jsd_tensor).abs().max().item())

        forward_kl = forward_tensor.tolist()
        reverse_kl = reverse_tensor.tolist()
        jsd_vals = [] if jsd_tensor is None else jsd_tensor.tolist()

        unique_token_ids = sorted({int(x) for x in teacher_ids_2d.reshape(-1).tolist()})
        token_map = _decode_token_map(tokenizer_path, unique_token_ids)

        topk = {
            "k": int(teacher_ids_2d.shape[1]),
            "token_ids": teacher_ids_2d.tolist(),
            "teacher_probs": teacher_prob.tolist(),
            "student_probs": student_prob.tolist(),
            "teacher_logprobs": teacher_lp.tolist(),
            "student_logprobs": student_lp.tolist(),
            "token_map": token_map,
        }
        mass = {
            "teacher_top1_mass": teacher_top1_mass.tolist(),
            "student_top1_mass": student_top1_mass.tolist(),
            "tv_distance": tv_distance.tolist(),
        }

    elif mode == "rkl":
        student_lp_1d = _tensor_to_2d_float(record.get("student_log_probs")).reshape(-1)
        teacher_lp_1d = _tensor_to_2d_float(record.get("teacher_log_probs")).reshape(-1)
        reverse_tensor = _tensor_to_2d_float(record.get("reverse_kl")).reshape(-1)
        recomputed = student_lp_1d - teacher_lp_1d
        recompute["reverse_kl_max_abs_diff"] = float((recomputed - reverse_tensor).abs().max().item())
        forward_kl = []
        reverse_kl = reverse_tensor.tolist()
        jsd_vals = []
        mass = {
            "student_prob": torch.exp(student_lp_1d).tolist(),
            "teacher_prob": torch.exp(teacher_lp_1d).tolist(),
            "prob_gap_abs": torch.abs(torch.exp(student_lp_1d) - torch.exp(teacher_lp_1d)).tolist(),
        }
        topk = {
            "student_log_probs": student_lp_1d.tolist(),
            "teacher_log_probs": teacher_lp_1d.tolist(),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode in file: {mode}")

    response_token_ids = student_ids[response_start:response_end]
    response_token_map = _decode_token_map(tokenizer_path, sorted({int(x) for x in response_token_ids}))
    response_tokens = [
        {
            "response_idx": i,
            "absolute_idx": response_start + i,
            "token_id": int(token_id),
            "token_text": response_token_map.get(str(int(token_id)), str(int(token_id))),
        }
        for i, token_id in enumerate(response_token_ids)
    ]

    logged_position_map: dict[str, int] = {str(i): -1 for i in range(max(0, response_length))}
    for logged_idx, pos in enumerate(position_indices):
        if 0 <= pos < response_length:
            logged_position_map[str(int(pos))] = int(logged_idx)

    warnings: list[str] = []
    training_effective: list[float]
    training_effective_label: str
    if mode == "fkl":
        training_effective = list(forward_kl)
        training_effective_label = "forward_kl"
    elif mode == "rkl":
        training_effective = list(reverse_kl)
        training_effective_label = "reverse_kl"
    elif mode == "jsd":
        if jsd_vals:
            training_effective = list(jsd_vals)
            training_effective_label = "jsd"
        else:
            training_effective = list(forward_kl)
            training_effective_label = "forward_kl_fallback_for_missing_jsd"
            warnings.append("mode=jsd but jsd tensor is missing; defaulting to forward_kl.")
    elif mode == "mixed":
        mixed_weight = float(recipe.get("opd_mixed_kl_weight", 0.5))
        training_effective = [
            float(mixed_weight * f + (1.0 - mixed_weight) * r) for f, r in zip(forward_kl, reverse_kl, strict=False)
        ]
        training_effective_label = f"mixed_kl(w={mixed_weight:.4f})"
    else:
        training_effective = []
        training_effective_label = "unknown"

    available_metrics = ["training_effective"]
    if forward_kl:
        available_metrics.append("forward_kl")
    if reverse_kl:
        available_metrics.append("reverse_kl")
    if jsd_vals:
        available_metrics.append("jsd")

    return {
        "mode": mode,
        "record_idx": record_idx,
        "sample_index": _safe_int(record.get("sample_index"), -1),
        "microbatch_sample_index": _safe_int(record.get("microbatch_sample_index"), -1),
        "position_indices": position_indices,
        "context": context,
        "loss": {
            "forward_kl": forward_kl,
            "reverse_kl": reverse_kl,
            "jsd": jsd_vals,
        },
        "loss_detail": {
            "available_metrics": available_metrics,
            "default_metric": "training_effective",
            "training_effective_label": training_effective_label,
            "warnings": warnings,
            "per_position": {
                "training_effective": training_effective,
                "forward_kl": forward_kl,
                "reverse_kl": reverse_kl,
                "jsd": jsd_vals,
            },
        },
        "response_tokens": response_tokens,
        "logged_position_map": logged_position_map,
        "inspector_mode_capabilities": {
            "has_topk_distribution": mode in {"fkl", "mixed", "jsd"},
        },
        "mass": mass,
        "topk": topk,
        "recompute": recompute,
        "run_dir": str(run_dir),
        "recipe": recipe,
        "writer_rank_info": payload.get("writer_rank_info") or {},
    }


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    runs = []
    for run_dir in _discover_runs():
        dump_files = _list_dump_files(run_dir)
        if not dump_files:
            continue
        last = dump_files[-1]
        runs.append(
            {
                "id": _run_id(run_dir),
                "path": str(run_dir),
                "n_files": len(dump_files),
                "last_file": last.name,
                "last_mtime": int(last.stat().st_mtime),
            }
        )
    return JSONResponse({"runs": runs})


@app.get("/api/steps")
def api_steps(run: str = Query(...)) -> JSONResponse:
    run_dir = _resolve_run(run)
    files = _list_dump_files(run_dir)
    steps: list[dict[str, Any]] = []
    for file in files:
        payload = _load_dump(file)
        rollout_id = _safe_int(payload.get("rollout_id"), _extract_rollout_id(file))
        mode = str(payload.get("mode") or payload.get("recipe", {}).get("distill_loss_mode") or "unknown")
        num_kept = _safe_int(payload.get("num_records_kept"), 0)
        created = float(payload.get("created_at_unix_s", file.stat().st_mtime))
        label = f"rollout={rollout_id} mode={mode} kept={num_kept} file={file.name}"
        steps.append(
            {
                "id": file.name,
                "label": label,
                "rollout_id": rollout_id,
                "mode": mode,
                "num_records_kept": num_kept,
                "created_at_unix_s": created,
                "size_mb": round(file.stat().st_size / 1024.0 / 1024.0, 3),
            }
        )
    steps.sort(key=lambda x: (x["rollout_id"], x["created_at_unix_s"]))
    return JSONResponse({"steps": steps})


@app.get("/api/samples")
def api_samples(
    run: str = Query(...),
    step: str = Query(...),
) -> JSONResponse:
    run_dir = _resolve_run(run)
    file = run_dir / "distill_debug_dumps" / step
    if file.name != step or not file.exists():
        raise HTTPException(status_code=404, detail=f"Unknown step file: {step}")
    payload = _load_dump(file)
    records = payload.get("records") or []
    summaries = [_sample_summary(record, index=i) for i, record in enumerate(records)]
    return JSONResponse(
        {
            "mode": payload.get("mode"),
            "rollout_id": payload.get("rollout_id"),
            "recipe": payload.get("recipe") or {},
            "samples": summaries,
        }
    )


@app.get("/api/record")
def api_record(
    run: str = Query(...),
    step: str = Query(...),
    sample: int = Query(0, ge=0),
    tokenizer_path: str = Query("", description="Optional tokenizer path override"),
) -> JSONResponse:
    run_dir = _resolve_run(run)
    file = run_dir / "distill_debug_dumps" / step
    if file.name != step or not file.exists():
        raise HTTPException(status_code=404, detail=f"Unknown step file: {step}")

    payload = _load_dump(file)
    tokenizer, tokenizer_error = _resolve_tokenizer_path(run_dir=run_dir, requested=tokenizer_path)
    record_payload = _build_record_payload(run_dir, payload, record_idx=sample, tokenizer_path=tokenizer)
    return JSONResponse(
        {
            "file": file.name,
            "rollout_id": payload.get("rollout_id"),
            "mode": payload.get("mode"),
            "tokenizer_path": tokenizer,
            "tokenizer_error": tokenizer_error,
            "payload": record_payload,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPD Distill Debug Live Viewer v2</title>
  <style>
    :root {
      --bg: #101217;
      --panel: #171a22;
      --muted: #8e95a3;
      --text: #ffffff;
      --accent: #26d7ae;
      --accent-2: #57a7ff;
      --warn: #f2b14d;
      --danger: #ff6e6e;
      --border: #2a2f3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 10%, rgba(38, 215, 174, 0.12), transparent 30%),
        radial-gradient(circle at 90% 0%, rgba(87, 167, 255, 0.12), transparent 35%),
        var(--bg);
    }
    h1 { margin: 0 0 14px; font-size: 20px; font-weight: 700; letter-spacing: 0.3px; }
    .controls, .panel {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 10px;
      align-items: end;
    }
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field label { font-size: 12px; color: var(--muted); }
    .field select, .field input, .field button {
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #111521;
      color: var(--text);
      font-size: 13px;
    }
    .field button {
      cursor: pointer;
      background: linear-gradient(90deg, #1c6f5c, #1f537f);
      border: none;
      font-weight: 700;
    }
    .field button:hover { filter: brightness(1.08); }
    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .token-stream {
      background: #0d1018;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      color: #ffffff;
      min-height: 120px;
      max-height: 320px;
      overflow: auto;
      white-space: normal;
      line-height: 1.45;
      font-size: 12px;
    }
    .context-box {
      background: #0d1018;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      color: #ffffff;
      min-height: 120px;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 12px;
    }
    .token {
      display: inline-block;
      margin: 2px;
      padding: 3px 6px;
      border-radius: 6px;
      border: 1px solid #2a3246;
      background: #121a2a;
      cursor: pointer;
      user-select: none;
      max-width: 100%;
      word-break: break-all;
    }
    .token:hover { border-color: #57a7ff; }
    .token.logged { background: #13211f; border-color: #245e56; }
    .token.unlogged { opacity: 0.75; }
    .token.selected {
      border-color: #f2b14d;
      outline: 1px solid #f2b14d;
      background: #2a2314;
    }
    .legend { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .inspector-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .box {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: #0f1320;
      min-height: 100px;
    }
    .kv { margin: 0; line-height: 1.5; font-size: 12px; }
    .warn {
      margin-top: 8px;
      color: #ffcf73;
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 8px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 700; }
    .bar-wrap {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .bar {
      height: 7px;
      border-radius: 5px;
      background: #243046;
      overflow: hidden;
      min-width: 140px;
    }
    .bar-inner-teacher {
      height: 100%;
      background: linear-gradient(90deg, #26d7ae, #19a783);
    }
    .bar-inner-student {
      height: 100%;
      background: linear-gradient(90deg, #57a7ff, #3a79d6);
    }
    @media (max-width: 1024px) {
      .inspector-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>OPD Distill Debug Live Viewer v2</h1>
  <div class="controls">
    <div class="grid">
      <div class="field" style="grid-column: span 4;">
        <label>Experiment Run Folder</label>
        <select id="runSelect"></select>
      </div>
      <div class="field" style="grid-column: span 4;">
        <label>Step / Rollout File</label>
        <select id="stepSelect"></select>
      </div>
      <div class="field" style="grid-column: span 2;">
        <label>Sample</label>
        <select id="sampleSelect"></select>
      </div>
      <div class="field" style="grid-column: span 6;">
        <label>Tokenizer Path (optional override)</label>
        <input id="tokenizerInput" placeholder="/path/to/tokenizer or leave empty" />
      </div>
      <div class="field" style="grid-column: span 2;">
        <label>Refresh</label>
        <button id="refreshBtn">Refresh Now</button>
      </div>
      <div class="field" style="grid-column: span 4;">
        <label>Status</label>
        <div id="status" class="meta">Loading...</div>
      </div>
    </div>
  </div>

  <div class="panel">
    <div id="summaryMeta" class="meta"></div>
  </div>

  <div class="panel">
    <div class="legend">Input Prompt (student, pre-response)</div>
    <div id="promptInput" class="context-box"></div>
  </div>

  <div class="panel">
    <div class="legend">Generated Rollout Tokens (student response only). Click a token to inspect next-token prediction and loss.</div>
    <div id="tokenStream" class="token-stream"></div>
  </div>

  <div class="panel">
    <div class="inspector-grid">
      <div class="box">
        <div class="legend">Token Inspector</div>
        <div class="field" style="margin-top: 8px;">
          <label>Loss Metric</label>
          <select id="lossMetricSelect"></select>
        </div>
        <div id="tokenMeta" class="kv"></div>
        <div id="tokenWarn" class="warn"></div>
      </div>
      <div class="box">
        <div class="legend">Teacher vs Student Mass (clicked position)</div>
        <div id="massMeta" class="kv"></div>
      </div>
    </div>
  </div>

  <div class="panel">
    <div id="topkTitle" class="legend">Top-k Next Token Distribution</div>
    <table id="topkTable">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Token ID</th>
          <th>Token Text</th>
          <th>Teacher p</th>
          <th>Student p</th>
          <th>Delta</th>
          <th>Mass View</th>
        </tr>
      </thead>
      <tbody id="topkBody"></tbody>
    </table>
    <div id="rklOnlyInfo" class="warn"></div>
  </div>

  <script>
    const els = {
      run: document.getElementById("runSelect"),
      step: document.getElementById("stepSelect"),
      sample: document.getElementById("sampleSelect"),
      tokenizer: document.getElementById("tokenizerInput"),
      refresh: document.getElementById("refreshBtn"),
      status: document.getElementById("status"),
      summary: document.getElementById("summaryMeta"),
      promptInput: document.getElementById("promptInput"),
      tokenStream: document.getElementById("tokenStream"),
      lossMetric: document.getElementById("lossMetricSelect"),
      tokenMeta: document.getElementById("tokenMeta"),
      tokenWarn: document.getElementById("tokenWarn"),
      massMeta: document.getElementById("massMeta"),
      topkTitle: document.getElementById("topkTitle"),
      topkBody: document.getElementById("topkBody"),
      rklOnlyInfo: document.getElementById("rklOnlyInfo"),
    };

    const state = {
      runs: [],
      steps: [],
      samples: [],
      record: null,
      selectedResponseIdx: -1,
      selectedLoggedIdx: -1,
      selectedMetric: "training_effective",
    };

    function setStatus(text) {
      els.status.textContent = text;
    }

    function fmt(n, digits = 6) {
      if (n === null || n === undefined || Number.isNaN(Number(n))) return "n/a";
      return Number(n).toFixed(digits);
    }

    function mean(arr) {
      if (!arr || arr.length === 0) return NaN;
      let s = 0;
      for (const v of arr) s += Number(v);
      return s / arr.length;
    }

    function safeText(s) {
      if (s === null || s === undefined) return "";
      return String(s);
    }

    function esc(s) {
      return safeText(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function populateSelect(selectEl, items, valueKey, labelKey) {
      selectEl.innerHTML = "";
      for (const item of items) {
        const opt = document.createElement("option");
        opt.value = item[valueKey];
        opt.textContent = item[labelKey];
        selectEl.appendChild(opt);
      }
    }

    async function loadRuns() {
      const res = await fetch("/api/runs");
      const data = await res.json();
      state.runs = data.runs || [];
      if (state.runs.length === 0) {
        setStatus("No run folders found");
        return;
      }
      populateSelect(els.run, state.runs, "id", "id");
      await loadSteps();
    }

    async function loadSteps() {
      const run = els.run.value;
      if (!run) return;
      const prevStep = els.step.value;
      const res = await fetch(`/api/steps?run=${encodeURIComponent(run)}`);
      const data = await res.json();
      state.steps = data.steps || [];
      if (state.steps.length === 0) {
        setStatus("No dump files in selected run");
        return;
      }
      populateSelect(els.step, state.steps, "id", "label");
      const stepIds = state.steps.map(x => x.id);
      if (prevStep && stepIds.includes(prevStep)) {
        els.step.value = prevStep;
      } else {
        els.step.value = state.steps[state.steps.length - 1].id;
      }
      await loadSamples();
    }

    async function loadSamples() {
      const run = els.run.value;
      const step = els.step.value;
      if (!run || !step) return;
      const prevSample = els.sample.value;
      const res = await fetch(`/api/samples?run=${encodeURIComponent(run)}&step=${encodeURIComponent(step)}`);
      const data = await res.json();
      state.samples = data.samples || [];
      const sampleItems = state.samples.map(s => ({
        id: String(s.idx),
        label: `idx=${s.idx} sample=${s.sample_index} resp_len=${s.response_length} pos=${s.num_positions}`,
      }));
      if (sampleItems.length === 0) {
        setStatus("No samples in step");
        return;
      }
      populateSelect(els.sample, sampleItems, "id", "label");
      const sampleIds = sampleItems.map(x => x.id);
      if (prevSample && sampleIds.includes(prevSample)) {
        els.sample.value = prevSample;
      }
      await loadRecord();
    }

    function selectInitialToken() {
      const p = state.record?.payload;
      if (!p) return;
      const tokens = p.response_tokens || [];
      if (tokens.length === 0) {
        state.selectedResponseIdx = -1;
        state.selectedLoggedIdx = -1;
        return;
      }
      const map = p.logged_position_map || {};
      let firstLogged = -1;
      for (const t of tokens) {
        const li = Number(map[String(t.response_idx)] ?? -1);
        if (li >= 0) {
          firstLogged = Number(t.response_idx);
          break;
        }
      }
      state.selectedResponseIdx = firstLogged >= 0 ? firstLogged : 0;
      state.selectedLoggedIdx = Number(map[String(state.selectedResponseIdx)] ?? -1);
    }

    function renderLossMetricSelect() {
      const p = state.record?.payload;
      if (!p) return;
      const options = (p.loss_detail?.available_metrics || []).map(x => ({ id: x, label: x }));
      if (options.length === 0) {
        els.lossMetric.innerHTML = "";
        return;
      }
      populateSelect(els.lossMetric, options, "id", "label");
      const ids = options.map(x => x.id);
      const def = p.loss_detail?.default_metric || "training_effective";
      if (!ids.includes(state.selectedMetric)) {
        state.selectedMetric = ids.includes(def) ? def : ids[0];
      }
      els.lossMetric.value = state.selectedMetric;
    }

    function renderTokenStream() {
      const p = state.record?.payload;
      els.tokenStream.innerHTML = "";
      if (!p) return;
      const tokens = p.response_tokens || [];
      const map = p.logged_position_map || {};
      if (tokens.length === 0) {
        els.tokenStream.textContent = "No response tokens in selected sample.";
        return;
      }
      for (const token of tokens) {
        const responseIdx = Number(token.response_idx);
        const loggedIdx = Number(map[String(responseIdx)] ?? -1);
        const span = document.createElement("span");
        span.className = `token ${loggedIdx >= 0 ? "logged" : "unlogged"} ${responseIdx === state.selectedResponseIdx ? "selected" : ""}`;
        span.innerHTML = esc(token.token_text);
        span.title = `response_idx=${responseIdx}, token_id=${token.token_id}, logged_idx=${loggedIdx}`;
        span.addEventListener("click", () => {
          state.selectedResponseIdx = responseIdx;
          state.selectedLoggedIdx = loggedIdx;
          renderTokenStream();
          renderInspector();
        });
        els.tokenStream.appendChild(span);
      }
    }

    function metricValue(metricKey, loggedIdx) {
      const per = state.record?.payload?.loss_detail?.per_position || {};
      const arr = per[metricKey] || [];
      if (loggedIdx < 0 || loggedIdx >= arr.length) return null;
      return Number(arr[loggedIdx]);
    }

    function renderInspector() {
      els.topkBody.innerHTML = "";
      els.rklOnlyInfo.textContent = "";
      els.tokenWarn.textContent = "";
      els.massMeta.textContent = "";

      const p = state.record?.payload;
      if (!p) return;
      const token = (p.response_tokens || []).find(x => Number(x.response_idx) === state.selectedResponseIdx);
      if (!token) {
        els.tokenMeta.textContent = "No token selected.";
        els.topkTitle.textContent = "Top-k Next Token Distribution";
        return;
      }

      const loggedIdx = state.selectedLoggedIdx;
      const mode = p.mode;
      const metric = state.selectedMetric;
      const metricValueSelected = metricValue(metric, loggedIdx);
      const fkl = metricValue("forward_kl", loggedIdx);
      const rkl = metricValue("reverse_kl", loggedIdx);
      const jsd = metricValue("jsd", loggedIdx);
      const trainingEffective = metricValue("training_effective", loggedIdx);

      const commonMeta = [
        `response_idx=${token.response_idx} absolute_idx=${token.absolute_idx}`,
        `token_id=${token.token_id}`,
        `token_text=${safeText(token.token_text)}`,
        `mode=${mode}`,
        `logged_position_idx=${loggedIdx}`,
        `selected_metric=${metric} value=${fmt(metricValueSelected, 6)}`,
        `training_effective(${p.loss_detail?.training_effective_label || "n/a"})=${fmt(trainingEffective, 6)}`,
        `forward_kl=${fmt(fkl, 6)} reverse_kl=${fmt(rkl, 6)} jsd=${fmt(jsd, 6)}`,
      ];
      els.tokenMeta.textContent = commonMeta.join("\\n");

      if (loggedIdx < 0) {
        els.tokenWarn.textContent = "This token is not logged in position_indices for this dump.";
        els.topkTitle.textContent = "Top-k Next Token Distribution";
        els.massMeta.textContent = "Mass view unavailable because selected token is not logged.";
        return;
      }
      const warnList = p.loss_detail?.warnings || [];
      if (warnList.length > 0) {
        els.tokenWarn.textContent = warnList.join(" | ");
      }

      const position = (p.position_indices || [])[loggedIdx];
      els.topkTitle.textContent = `Logged position i=${loggedIdx}, response_pos=${position}`;

      if (!p.inspector_mode_capabilities?.has_topk_distribution) {
        const stLogP = Number((p.topk?.student_log_probs || [])[loggedIdx]);
        const tcLogP = Number((p.topk?.teacher_log_probs || [])[loggedIdx]);
        const stP = Number((p.mass?.student_prob || [])[loggedIdx]);
        const tcP = Number((p.mass?.teacher_prob || [])[loggedIdx]);
        const gap = Number((p.mass?.prob_gap_abs || [])[loggedIdx]);
        els.rklOnlyInfo.textContent = "RKL dump: top-k candidates are unavailable. Showing scalar probabilities only.";
        els.massMeta.textContent =
          `student_prob=${fmt(stP, 6)} teacher_prob=${fmt(tcP, 6)} | abs_gap=${fmt(gap, 6)}\\n` +
          `student_log_prob=${fmt(stLogP, 6)} teacher_log_prob=${fmt(tcLogP, 6)} reverse_kl=${fmt(rkl, 6)}`;
        return;
      }

      const topk = p.topk || {};
      const tokenIds = (topk.token_ids || [])[loggedIdx] || [];
      const tProbs = (topk.teacher_probs || [])[loggedIdx] || [];
      const sProbs = (topk.student_probs || [])[loggedIdx] || [];

      const teacherTop1 = Math.max(...tProbs.map(Number));
      const studentTop1 = Math.max(...sProbs.map(Number));
      let tv = 0;
      for (let i = 0; i < tokenIds.length; i += 1) {
        tv += Math.abs(Number(tProbs[i] || 0) - Number(sProbs[i] || 0));
      }
      tv *= 0.5;
      els.massMeta.textContent =
        `teacher_top1_mass=${fmt(teacherTop1, 6)} student_top1_mass=${fmt(studentTop1, 6)} tv_distance=${fmt(tv, 6)}`;

      for (let i = 0; i < tokenIds.length; i += 1) {
        const tid = Number(tokenIds[i]);
        const tok = (topk.token_map && topk.token_map[String(tid)]) ? topk.token_map[String(tid)] : "";
        const tp = Number(tProbs[i] || 0);
        const sp = Number(sProbs[i] || 0);
        const delta = sp - tp;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td>${tid}</td>
          <td>${esc(tok)}</td>
          <td>${fmt(tp, 5)}</td>
          <td>${fmt(sp, 5)}</td>
          <td>${fmt(delta, 5)}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar"><div class="bar-inner-teacher" style="width:${Math.max(0, Math.min(100, tp * 100))}%"></div></div>
              <div class="bar"><div class="bar-inner-student" style="width:${Math.max(0, Math.min(100, sp * 100))}%"></div></div>
            </div>
          </td>
        `;
        els.topkBody.appendChild(tr);
      }
    }

    function renderRecord() {
      const data = state.record;
      if (!data) return;
      const p = data.payload;
      const sc = p.context || {};
      const prompt = safeText(sc.student_prompt);

      const summaryLines = [
        `run_dir: ${p.run_dir}`,
        `file: ${data.file}`,
        `mode: ${p.mode}`,
        `rollout_id: ${data.rollout_id}`,
        `sample_index: ${p.sample_index} microbatch_sample_index: ${p.microbatch_sample_index}`,
        `response_tokens: ${(p.response_tokens || []).length}`,
        `logged_positions: ${(p.position_indices || []).length}`,
        `tokenizer_path: ${data.tokenizer_path || "(none)"}`,
        `tokenizer_error: ${data.tokenizer_error || "(none)"}`,
        `recipe: ${JSON.stringify(p.recipe)}`,
        `writer_rank_info: ${JSON.stringify(p.writer_rank_info)}`,
        `recompute_diff: ${JSON.stringify(p.recompute)}`,
        `training_effective_label: ${p.loss_detail?.training_effective_label || "n/a"}`,
        `training_effective_mean=${fmt(mean((p.loss_detail?.per_position || {}).training_effective || []), 6)}`,
      ];
      els.summary.textContent = summaryLines.join("\\n");
      els.promptInput.textContent = prompt || "(empty prompt)";
      renderLossMetricSelect();
      selectInitialToken();
      renderTokenStream();
      renderInspector();
      setStatus(`Loaded rollout=${data.rollout_id} mode=${p.mode} sample=${p.record_idx}`);
    }

    async function loadRecord() {
      const run = els.run.value;
      const step = els.step.value;
      const sample = Number(els.sample.value || 0);
      if (!run || !step) return;
      setStatus("Loading record...");
      const tok = encodeURIComponent(els.tokenizer.value || "");
      const url = `/api/record?run=${encodeURIComponent(run)}&step=${encodeURIComponent(step)}&sample=${sample}&tokenizer_path=${tok}`;
      const res = await fetch(url);
      if (!res.ok) {
        setStatus(`Failed to load record: ${res.status}`);
        return;
      }
      state.record = await res.json();
      renderRecord();
    }

    els.run.addEventListener("change", async () => {
      await loadSteps();
    });
    els.step.addEventListener("change", async () => {
      await loadSamples();
    });
    els.sample.addEventListener("change", async () => {
      await loadRecord();
    });
    els.lossMetric.addEventListener("change", () => {
      state.selectedMetric = els.lossMetric.value || "training_effective";
      renderInspector();
    });
    els.refresh.addEventListener("click", async () => {
      await loadSteps();
    });

    async function start() {
      try {
        await loadRuns();
        setStatus("Ready");
      } catch (err) {
        setStatus(`Error: ${err}`);
      }
    }

    start();
  </script>
</body>
</html>
        """
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Live web viewer v2 for OPD debug dump .pt files.")
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(repo_root / "outputs"),
        help="Root folder containing experiment outputs. Run folders with distill_debug_dumps/ will be discovered recursively.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Optional default tokenizer path for decoding.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    STATE.runs_root = Path(args.runs_root).resolve()
    STATE.default_tokenizer_path = args.tokenizer_path.strip()
    print(f"[debug_dump_live_server_v2] runs_root={STATE.runs_root}")
    if STATE.default_tokenizer_path:
        print(f"[debug_dump_live_server_v2] default_tokenizer={STATE.default_tokenizer_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
