#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional for raw-id mode
    AutoTokenizer = None  # type: ignore[assignment]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_list_int(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.long).tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(f"Unsupported int-list value type: {type(value)}")


def _to_1d_float(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32).reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise TypeError(f"Unsupported float-list value type: {type(value)}")


def _to_2d_long(value: Any) -> torch.Tensor:
    if value is None:
        return torch.empty((0, 0), dtype=torch.long)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def _to_2d_float(value: Any) -> torch.Tensor:
    if value is None:
        return torch.empty((0, 0), dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _looks_like_hf_tokenizer_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "config.json").exists()
    )


def _qwen_student_sort_key(path: Path) -> tuple[int, int, int, str]:
    name = path.name.lower()
    preferred = ["qwen3-4b", "qwen3-8b", "qwen3-32b", "qwen3-14b", "qwen3-1.7b"]
    hint_rank = next((idx for idx, hint in enumerate(preferred) if hint in name), len(preferred))
    converted_penalty = 1 if "as-qwen35" in name else 0
    fp8_penalty = 1 if "fp8" in name else 0
    return (converted_penalty, hint_rank, fp8_penalty, name)


def _qwen_teacher_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    primary_rank = 0 if "qwen3.5-397b" in name else 1
    fp8_rank = 0 if "fp8" in name else 1
    return (primary_rank, fp8_rank, name)


@lru_cache(maxsize=1)
def _discover_qwen_tokenizer_candidates() -> dict[str, list[str]]:
    root = Path.home() / "ckpt" / "hf_models" / "Qwen"
    if not root.exists() or not root.is_dir():
        return {"student": [], "teacher": []}

    student_paths: list[Path] = []
    teacher_paths: list[Path] = []
    for child in root.iterdir():
        if not _looks_like_hf_tokenizer_dir(child):
            continue
        lower = child.name.lower()
        if "qwen3.5" in lower:
            teacher_paths.append(child)
        elif "qwen3" in lower:
            student_paths.append(child)

    return {
        "student": [str(p) for p in sorted(set(student_paths), key=_qwen_student_sort_key)],
        "teacher": [str(p) for p in sorted(set(teacher_paths), key=_qwen_teacher_sort_key)],
    }


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_path: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not available in this environment")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _resolve_tokenizer(
    *,
    requested: str,
    recipe_hint: str,
    auto_candidates: list[str],
) -> tuple[str, Any, str]:
    candidates: list[str] = []
    for item in [requested.strip(), recipe_hint.strip(), *auto_candidates]:
        if not item:
            continue
        if item not in candidates:
            candidates.append(item)

    if not candidates:
        return "", None, ""

    errors: list[str] = []
    for candidate in candidates:
        try:
            tok = _load_tokenizer(candidate)
            return candidate, tok, ""
        except Exception as exc:  # pragma: no cover - environment/path dependent
            errors.append(f"{candidate}: {exc}")

    return "", None, "; ".join(errors[-2:])


def _decode_ids(ids: list[int], tokenizer: Any, *, max_tokens: int = 160, max_chars: int = 1200) -> str:
    clipped = ids[:max_tokens]
    if not clipped:
        return ""
    if tokenizer is not None:
        text = tokenizer.decode(clipped, skip_special_tokens=False)
    else:
        text = " ".join(str(x) for x in clipped)
    if len(text) > max_chars:
        return text[: max_chars - 17] + "\n...[truncated]..."
    return text


def _decode_token_id(token_id: int, tokenizer: Any) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return "<decode-error>"
    compact = text.replace("\n", "\\n")
    if len(compact) > 60:
        compact = compact[:57] + "..."
    return compact


def _fmt(x: float, digits: int = 6) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}f}"


def _effective_metric(mode: str, fkl: float, rkl: float, jsd: float, mixed_weight: float) -> float:
    if mode == "fkl":
        return fkl
    if mode == "mixed":
        return mixed_weight * fkl + (1.0 - mixed_weight) * rkl
    if mode == "jsd":
        return jsd
    if mode == "rkl":
        return rkl
    return float("nan")


def _print_samples(records: list[dict[str, Any]], *, max_rows: int) -> None:
    print("\nSamples:")
    for idx, record in enumerate(records[:max_rows]):
        response_length = _safe_int(record.get("response_length"), 0)
        position_indices = _to_list_int(record.get("position_indices"))
        unit = str(record.get("position_unit") or "student_token")
        print(
            f"  [{idx}] sample={_safe_int(record.get('sample_index'), -1)} "
            f"micro={_safe_int(record.get('microbatch_sample_index'), idx)} "
            f"response_len={response_length} logged={len(position_indices)} unit={unit}"
        )
    if len(records) > max_rows:
        print(f"  ... ({len(records) - max_rows} more)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal inspector for OPD distill debug dumps (.pt)")
    parser.add_argument("dump", type=str, help="Path to distill_debug_*.pt file")
    parser.add_argument("--sample", type=int, default=0, help="Record index to inspect")
    parser.add_argument("--list-samples", action="store_true", help="List samples and exit")
    parser.add_argument("--max-samples", type=int, default=32, help="Max sample summaries to print")
    parser.add_argument("--max-units", type=int, default=8, help="Max logged units to print")
    parser.add_argument("--topk", type=int, default=5, help="Top-k entries per logged unit to print")
    parser.add_argument("--max-decode-tokens", type=int, default=160, help="Max tokens for prompt/response decoding")
    parser.add_argument("--tokenizer-path", type=str, default="", help="Legacy shared tokenizer override")
    parser.add_argument("--student-tokenizer-path", type=str, default="", help="Student tokenizer override")
    parser.add_argument("--teacher-tokenizer-path", type=str, default="", help="Teacher tokenizer override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_path = Path(args.dump).expanduser().resolve()
    if not dump_path.is_file():
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type: {type(payload)}")

    records = payload.get("records") or []
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Dump has no records")

    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
    mode = str(payload.get("mode") or recipe.get("distill_loss_mode") or "unknown").lower()
    rollout_id = _safe_int(payload.get("rollout_id"), -1)
    cross_enabled = _safe_int(recipe.get("opd_cross_tokenizer_enable"), 0)

    shared_override = str(args.tokenizer_path or "").strip()
    student_override = str(args.student_tokenizer_path or "").strip() or shared_override
    teacher_override = str(args.teacher_tokenizer_path or "").strip() or shared_override

    auto_qwen = _discover_qwen_tokenizer_candidates()
    student_path, student_tok, student_err = _resolve_tokenizer(
        requested=student_override,
        recipe_hint=str(recipe.get("opd_student_tokenizer_path") or ""),
        auto_candidates=auto_qwen.get("student", []),
    )
    teacher_path, teacher_tok, teacher_err = _resolve_tokenizer(
        requested=teacher_override,
        recipe_hint=str(recipe.get("opd_teacher_tokenizer_path") or ""),
        auto_candidates=auto_qwen.get("teacher", []),
    )

    print("=" * 88)
    print(f"dump: {dump_path}")
    print(f"mode: {mode} | rollout_id: {rollout_id} | records: {len(records)}")
    print(f"cross_tokenizer(recipe): {cross_enabled}")
    print(f"student_tokenizer: {student_path or '(raw ids)'}")
    if student_err:
        print(f"student_tokenizer_error: {student_err}")
    print(f"teacher_tokenizer: {teacher_path or '(raw ids)'}")
    if teacher_err:
        print(f"teacher_tokenizer_error: {teacher_err}")
    print(f"recipe: {json.dumps(recipe, ensure_ascii=True, sort_keys=True)}")

    _print_samples(records, max_rows=max(1, args.max_samples))
    if args.list_samples:
        return

    if args.sample < 0 or args.sample >= len(records):
        raise IndexError(f"sample index out of range: {args.sample} (records={len(records)})")

    record = records[args.sample]
    response_start = _safe_int(record.get("response_start"), 0)
    response_length = _safe_int(record.get("response_length"), 0)
    response_end = max(response_start, response_start + response_length)
    teacher_start = _safe_int(record.get("teacher_logprob_start_len"), response_start)

    student_ids = _to_list_int(record.get("student_input_ids"))
    teacher_ids = _to_list_int(record.get("teacher_input_ids"))

    position_indices = _to_list_int(record.get("position_indices"))
    position_unit = str(record.get("position_unit") or "student_token")
    group_lengths = _to_list_int(record.get("teacher_topk_group_lengths"))
    group_valid_mask = _to_list_int(record.get("teacher_topk_group_valid_mask"))

    student_prompt = _decode_ids(student_ids[:response_start], student_tok, max_tokens=max(1, args.max_decode_tokens))
    student_response = _decode_ids(
        student_ids[response_start:response_end],
        student_tok,
        max_tokens=max(1, args.max_decode_tokens),
    )
    teacher_prompt = _decode_ids(teacher_ids[:teacher_start], teacher_tok, max_tokens=max(1, args.max_decode_tokens))
    teacher_response = _decode_ids(teacher_ids[teacher_start:], teacher_tok, max_tokens=max(1, args.max_decode_tokens))

    print("\n" + "-" * 88)
    print(
        f"record[{args.sample}] sample={_safe_int(record.get('sample_index'), -1)} "
        f"micro={_safe_int(record.get('microbatch_sample_index'), args.sample)} "
        f"response_start={response_start} response_length={response_length}"
    )
    print(f"position_unit={position_unit} logged_units={len(position_indices)}")
    print(f"student_prompt: {student_prompt}")
    print(f"student_response: {student_response}")
    print(f"teacher_prompt: {teacher_prompt}")
    print(f"teacher_response: {teacher_response}")

    forward_kl = _to_1d_float(record.get("forward_kl"))
    reverse_kl = _to_1d_float(record.get("reverse_kl"))
    jsd_vals = _to_1d_float(record.get("jsd"))
    mixed_weight = _safe_float(recipe.get("opd_mixed_kl_weight"), 0.5)

    print("\nLogged Units:")
    for idx in range(min(len(position_indices), max(1, args.max_units))):
        anchor = int(position_indices[idx])
        fkl_val = float(forward_kl[idx]) if idx < len(forward_kl) else float("nan")
        rkl_val = float(reverse_kl[idx]) if idx < len(reverse_kl) else float("nan")
        jsd_val = float(jsd_vals[idx]) if idx < len(jsd_vals) else float("nan")
        train_val = _effective_metric(mode, fkl_val, rkl_val, jsd_val, mixed_weight)

        if position_unit == "teacher_group_anchor":
            span_len = int(group_lengths[idx]) if idx < len(group_lengths) else 0
            if span_len <= 0:
                span_len = 1
            span_end = min(response_length, anchor + span_len)
            valid = int(group_valid_mask[idx]) if idx < len(group_valid_mask) else 0
            unit_desc = f"anchor={anchor} span=[{anchor}:{span_end}) len={span_len} valid={valid}"
        else:
            unit_desc = f"response_idx={anchor}"

        print(
            f"  [{idx:03d}] {unit_desc} "
            f"fkl={_fmt(fkl_val)} rkl={_fmt(rkl_val)} jsd={_fmt(jsd_val)} train={_fmt(train_val)}"
        )

    teacher_topk_ids = _to_2d_long(record.get("teacher_topk_token_ids"))
    teacher_topk_lp = _to_2d_float(record.get("teacher_topk_logprobs"))
    student_topk_lp = _to_2d_float(record.get("student_topk_logprobs"))

    if teacher_topk_ids.numel() == 0 or teacher_topk_lp.numel() == 0 or student_topk_lp.numel() == 0:
        print("\nNo top-k tensor payload for selected record.")
        return

    rows = min(int(teacher_topk_ids.shape[0]), max(1, args.max_units))
    cols = min(int(teacher_topk_ids.shape[1]), max(1, args.topk))

    print("\nTop-k (decoded using student tokenizer):")
    for row in range(rows):
        anchor = int(position_indices[row]) if row < len(position_indices) else -1
        row_desc = f"unit[{row}]"
        if position_unit == "teacher_group_anchor":
            span_len = int(group_lengths[row]) if row < len(group_lengths) else 0
            if span_len <= 0:
                span_len = 1
            span_end = min(response_length, anchor + span_len)
            row_desc += f" anchor={anchor} span=[{anchor}:{span_end})"
        else:
            row_desc += f" response_idx={anchor}"
        print(f"  {row_desc}")

        for col in range(cols):
            token_id = int(teacher_topk_ids[row, col].item())
            t_lp = float(teacher_topk_lp[row, col].item())
            s_lp = float(student_topk_lp[row, col].item())
            token_text = _decode_token_id(token_id, student_tok)
            print(
                f"    k{col:02d} id={token_id:<8d} tok={token_text!r:<64} "
                f"t_lp={t_lp: .6f} s_lp={s_lp: .6f} t_p={math.exp(t_lp):.3e} s_p={math.exp(s_lp):.3e}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[debug_terminal] {exc}")
