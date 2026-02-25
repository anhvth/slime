from __future__ import annotations

import io
import logging
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    from megatron.core import mpu
except Exception:  # pragma: no cover - fallback for non-megatron unit tests
    mpu = None


logger = logging.getLogger(__name__)

DEBUG_DUMP_SCHEMA_VERSION = 1
_SEEN_UPDATE_KEYS: set[str] = set()


def clear_debug_dump_cache() -> None:
    """Reset in-process update dedupe cache (useful for tests)."""
    _SEEN_UPDATE_KEYS.clear()


def _normalize_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_long_cpu_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.long)
    if isinstance(value, (list, tuple)):
        return torch.tensor([int(v) for v in value], dtype=torch.long)
    raise TypeError(f"Unsupported token-like value type: {type(value)}")


def _to_float_cpu_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32)
    raise TypeError(f"Unsupported float-tensor-like value type: {type(value)}")


def _resolve_rollout_id(args: Namespace) -> int:
    rollout_id = getattr(args, "opd_debug_rollout_id", None)
    if rollout_id is None:
        rollout_id = getattr(args, "rollout_id", None)
    rid = _safe_int(rollout_id, -1)
    return rid if rid >= 0 else 0


def _is_global_writer_rank() -> bool:
    if mpu is not None:
        try:
            dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_world = mpu.get_pipeline_model_parallel_world_size()
            return dp_rank == 0 and tp_rank == 0 and pp_rank == pp_world - 1
        except Exception:
            pass

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _writer_rank_info() -> dict[str, int]:
    info = {
        "dist_rank": 0,
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "pp_world_size": 1,
    }
    if dist.is_available() and dist.is_initialized():
        info["dist_rank"] = int(dist.get_rank())
    if mpu is not None:
        try:
            info["dp_rank"] = int(mpu.get_data_parallel_rank(with_context_parallel=True))
            info["tp_rank"] = int(mpu.get_tensor_model_parallel_rank())
            info["pp_rank"] = int(mpu.get_pipeline_model_parallel_rank())
            info["pp_world_size"] = int(mpu.get_pipeline_model_parallel_world_size())
        except Exception:
            pass
    return info


def _debug_dump_enabled(args: Namespace) -> bool:
    return _normalize_bool(getattr(args, "opd_debug_dump_enable", False), default=False)


def _resolve_dump_dir(args: Namespace) -> Path:
    configured = str(getattr(args, "opd_debug_dump_dir", "") or "").strip()
    if configured:
        return Path(configured)
    base = str(getattr(args, "save", "") or "").strip()
    if not base:
        base = "."
    return Path(base) / "distill_debug_dumps"


def _retention_limits(args: Namespace) -> tuple[int, int]:
    max_total_mb = max(0, _safe_int(getattr(args, "opd_debug_dump_max_total_mb", 5120), 5120))
    max_files = max(1, _safe_int(getattr(args, "opd_debug_dump_max_files", 20000), 20000))
    return max_total_mb * 1024 * 1024, max_files


def _sampling_limits(args: Namespace) -> tuple[int, int]:
    max_samples = max(1, _safe_int(getattr(args, "opd_debug_dump_max_samples_per_update", 8), 8))
    # 0 means "log all response positions" for each kept sample.
    max_positions = max(0, _safe_int(getattr(args, "opd_debug_dump_max_positions_per_sample", 0), 0))
    return max_samples, max_positions


def _max_file_bytes(args: Namespace) -> int:
    max_file_mb = max(1, _safe_int(getattr(args, "opd_debug_dump_max_file_mb", 64), 64))
    return max_file_mb * 1024 * 1024


def _seed_for_update(args: Namespace, rollout_id: int, salt: int) -> int:
    base_seed = _safe_int(getattr(args, "opd_debug_dump_seed", getattr(args, "seed", 1234)), 1234)
    return base_seed + rollout_id * 1000003 + salt


def _reserve_update_key(mode: str, rollout_id: int) -> bool:
    key = f"{mode}:{rollout_id}"
    if key in _SEEN_UPDATE_KEYS:
        return False
    _SEEN_UPDATE_KEYS.add(key)
    return True


def _to_position_index_tensor(length: int, max_positions: int, rng: random.Random) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0,), dtype=torch.long)
    if max_positions <= 0:
        return torch.arange(length, dtype=torch.long)
    if length <= max_positions:
        return torch.arange(length, dtype=torch.long)
    selected = rng.sample(range(length), k=max_positions)
    selected.sort()
    return torch.tensor(selected, dtype=torch.long)


def _select_sample_indices(num_samples: int, max_samples: int, rng: random.Random) -> list[int]:
    if num_samples <= max_samples:
        return list(range(num_samples))
    selected = rng.sample(range(num_samples), k=max_samples)
    selected.sort()
    return selected


def _serialized_size_bytes(payload: dict[str, Any]) -> int:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.tell()


def _trim_for_soft_file_cap(payload: dict[str, Any], max_file_bytes: int) -> dict[str, Any]:
    size = _serialized_size_bytes(payload)
    if size <= max_file_bytes:
        payload["file_size_bytes"] = size
        return payload

    records = list(payload.get("records", []))
    while len(records) > 1 and size > max_file_bytes:
        records.pop()
        payload["records"] = records
        payload["num_records_kept"] = len(records)
        size = _serialized_size_bytes(payload)

    if size > max_file_bytes:
        logger.warning(
            "Distill debug dump exceeds soft file limit after trimming: size=%d bytes limit=%d bytes",
            size,
            max_file_bytes,
        )

    payload["file_size_bytes"] = size
    return payload


def _enforce_retention(dump_dir: Path, max_total_bytes: int, max_files: int) -> None:
    files = sorted(dump_dir.glob("distill_debug_*.pt"), key=lambda p: (p.stat().st_mtime, p.name))
    if not files:
        return

    total_bytes = sum(int(path.stat().st_size) for path in files)
    keep_all = len(files) <= max_files and (max_total_bytes <= 0 or total_bytes <= max_total_bytes)
    if keep_all:
        return

    while files:
        path = files[0]
        too_many_files = len(files) > max_files
        too_many_bytes = max_total_bytes > 0 and total_bytes > max_total_bytes
        if not too_many_files and not too_many_bytes:
            break
        try:
            file_size = int(path.stat().st_size)
        except FileNotFoundError:
            file_size = 0
        try:
            path.unlink(missing_ok=True)
        finally:
            total_bytes -= file_size
            files.pop(0)


def _build_recipe(args: Namespace, mode: str) -> dict[str, float | str]:
    return {
        "distill_loss_mode": mode,
        "opd_distill_coef": _safe_float(getattr(args, "opd_distill_coef", 1.0), 1.0),
        "opd_mixed_kl_weight": _safe_float(getattr(args, "opd_mixed_kl_weight", 0.5), 0.5),
        "opd_jsd_beta": _safe_float(getattr(args, "opd_jsd_beta", 0.5), 0.5),
        "opd_kl_coef": _safe_float(getattr(args, "opd_kl_coef", 1.0), 1.0),
    }


def _write_dump_file(
    args: Namespace,
    *,
    mode: str,
    rollout_id: int,
    payload: dict[str, Any],
) -> Path:
    dump_dir = _resolve_dump_dir(args)
    dump_dir.mkdir(parents=True, exist_ok=True)

    info = _writer_rank_info()
    timestamp_ms = int(time.time() * 1000)
    filename = (
        f"distill_debug_{mode}_rollout_{rollout_id:07d}_"
        f"rank_{info['dist_rank']:05d}_{timestamp_ms}.pt"
    )
    target = dump_dir / filename
    temp = target.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(target)

    max_total_bytes, max_files = _retention_limits(args)
    _enforce_retention(dump_dir, max_total_bytes=max_total_bytes, max_files=max_files)
    return target


def _teacher_view(
    student_input_ids: torch.Tensor,
    response_start: int,
    teacher_input_ids: object | None,
    teacher_logprob_start_len: object | None,
) -> tuple[torch.Tensor, int]:
    if teacher_input_ids is None:
        teacher_ids = student_input_ids.clone()
    else:
        teacher_ids = _to_long_cpu_tensor(teacher_input_ids)
    if teacher_logprob_start_len is None:
        teacher_start = response_start
    else:
        teacher_start = _safe_int(teacher_logprob_start_len, response_start)
    return teacher_ids, teacher_start


def dump_topk_debug_update(
    args: Namespace,
    *,
    mode: str,
    records: list[dict[str, Any]],
) -> Path | None:
    if not _debug_dump_enabled(args):
        return None
    if not _is_global_writer_rank():
        return None

    rollout_id = _resolve_rollout_id(args)
    if not _reserve_update_key(f"topk:{mode}", rollout_id):
        return None

    max_samples, max_positions = _sampling_limits(args)
    rng = random.Random(_seed_for_update(args, rollout_id, salt=17))
    kept_indices = _select_sample_indices(len(records), max_samples, rng)

    sampled_records: list[dict[str, Any]] = []
    for idx in kept_indices:
        raw = records[idx]
        student_input_ids = _to_long_cpu_tensor(raw["student_input_ids"])
        response_length = _safe_int(raw["response_length"], 0)
        response_start = _safe_int(raw["response_start"], 0)
        position_indices = _to_position_index_tensor(response_length, max_positions, rng)

        teacher_ids, teacher_start = _teacher_view(
            student_input_ids=student_input_ids,
            response_start=response_start,
            teacher_input_ids=raw.get("teacher_input_ids"),
            teacher_logprob_start_len=raw.get("teacher_logprob_start_len"),
        )

        teacher_topk_logprobs = _to_float_cpu_tensor(raw["teacher_topk_logprobs"]).index_select(0, position_indices)
        teacher_topk_token_ids = _to_long_cpu_tensor(raw["teacher_topk_token_ids"]).index_select(0, position_indices)
        student_topk_logprobs = _to_float_cpu_tensor(raw["student_topk_logprobs"]).index_select(0, position_indices)
        forward_kl = _to_float_cpu_tensor(raw["forward_kl"]).index_select(0, position_indices)
        reverse_kl = _to_float_cpu_tensor(raw["reverse_kl"]).index_select(0, position_indices)
        jsd = raw.get("jsd")
        jsd_tensor = None if jsd is None else _to_float_cpu_tensor(jsd).index_select(0, position_indices)

        sampled_records.append(
            {
                "sample_index": _safe_int(raw.get("sample_index"), -1),
                "microbatch_sample_index": _safe_int(raw.get("microbatch_sample_index"), idx),
                "student_input_ids": student_input_ids,
                "response_start": response_start,
                "response_length": response_length,
                "response_token_ids": student_input_ids[response_start : response_start + response_length],
                "position_indices": position_indices,
                "teacher_input_ids": teacher_ids,
                "teacher_logprob_start_len": teacher_start,
                "teacher_topk_logprobs": teacher_topk_logprobs,
                "teacher_topk_token_ids": teacher_topk_token_ids,
                "student_topk_logprobs": student_topk_logprobs,
                "forward_kl": forward_kl,
                "reverse_kl": reverse_kl,
                "jsd": jsd_tensor,
            }
        )

    payload = {
        "version": DEBUG_DUMP_SCHEMA_VERSION,
        "kind": "distill_debug_update",
        "mode": mode,
        "rollout_id": rollout_id,
        "created_at_unix_s": time.time(),
        "writer_rank_info": _writer_rank_info(),
        "recipe": _build_recipe(args, mode=mode),
        "limits": {
            "max_samples_per_update": max_samples,
            "max_positions_per_sample": max_positions,
            "max_file_mb": _safe_int(getattr(args, "opd_debug_dump_max_file_mb", 64), 64),
            "max_total_mb": _safe_int(getattr(args, "opd_debug_dump_max_total_mb", 5120), 5120),
            "max_files": _safe_int(getattr(args, "opd_debug_dump_max_files", 20000), 20000),
        },
        "num_records_total": len(records),
        "num_records_kept": len(sampled_records),
        "records": sampled_records,
    }

    payload = _trim_for_soft_file_cap(payload, max_file_bytes=_max_file_bytes(args))
    return _write_dump_file(args, mode=mode, rollout_id=rollout_id, payload=payload)


def dump_rkl_debug_update(
    args: Namespace,
    *,
    rollout_data: dict[str, Any],
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    reverse_kls: list[torch.Tensor],
) -> Path | None:
    if not _debug_dump_enabled(args):
        return None
    if not _is_global_writer_rank():
        return None

    rollout_id = _resolve_rollout_id(args)
    mode = "rkl"
    if not _reserve_update_key(mode, rollout_id):
        return None

    tokens = rollout_data.get("tokens")
    response_lengths = rollout_data.get("response_lengths")
    if tokens is None or response_lengths is None:
        logger.warning("Skip rkl debug dump: missing rollout_data tokens/response_lengths.")
        return None

    sample_indices = rollout_data.get("sample_indices")
    teacher_input_ids_list = rollout_data.get("teacher_input_ids")
    teacher_start_list = rollout_data.get("teacher_logprob_start_len")

    max_samples, max_positions = _sampling_limits(args)
    rng = random.Random(_seed_for_update(args, rollout_id, salt=31))
    num_records = min(len(tokens), len(student_log_probs), len(teacher_log_probs), len(reverse_kls))
    kept_indices = _select_sample_indices(num_records, max_samples, rng)

    sampled_records: list[dict[str, Any]] = []
    for idx in kept_indices:
        student_input_ids = _to_long_cpu_tensor(tokens[idx])
        response_length = _safe_int(response_lengths[idx], 0)
        response_start = max(0, student_input_ids.size(0) - response_length)
        position_indices = _to_position_index_tensor(response_length, max_positions, rng)

        teacher_ids_raw = teacher_input_ids_list[idx] if teacher_input_ids_list is not None else None
        teacher_start_raw = teacher_start_list[idx] if teacher_start_list is not None else None
        teacher_input_ids, teacher_start = _teacher_view(
            student_input_ids=student_input_ids,
            response_start=response_start,
            teacher_input_ids=teacher_ids_raw,
            teacher_logprob_start_len=teacher_start_raw,
        )

        sampled_records.append(
            {
                "sample_index": _safe_int(sample_indices[idx], -1) if sample_indices is not None else -1,
                "microbatch_sample_index": idx,
                "student_input_ids": student_input_ids,
                "response_start": response_start,
                "response_length": response_length,
                "response_token_ids": student_input_ids[response_start : response_start + response_length],
                "position_indices": position_indices,
                "teacher_input_ids": teacher_input_ids,
                "teacher_logprob_start_len": teacher_start,
                "student_log_probs": _to_float_cpu_tensor(student_log_probs[idx]).index_select(0, position_indices),
                "teacher_log_probs": _to_float_cpu_tensor(teacher_log_probs[idx]).index_select(0, position_indices),
                "reverse_kl": _to_float_cpu_tensor(reverse_kls[idx]).index_select(0, position_indices),
            }
        )

    payload = {
        "version": DEBUG_DUMP_SCHEMA_VERSION,
        "kind": "distill_debug_update",
        "mode": mode,
        "rollout_id": rollout_id,
        "created_at_unix_s": time.time(),
        "writer_rank_info": _writer_rank_info(),
        "recipe": _build_recipe(args, mode=mode),
        "limits": {
            "max_samples_per_update": max_samples,
            "max_positions_per_sample": max_positions,
            "max_file_mb": _safe_int(getattr(args, "opd_debug_dump_max_file_mb", 64), 64),
            "max_total_mb": _safe_int(getattr(args, "opd_debug_dump_max_total_mb", 5120), 5120),
            "max_files": _safe_int(getattr(args, "opd_debug_dump_max_files", 20000), 20000),
        },
        "num_records_total": num_records,
        "num_records_kept": len(sampled_records),
        "records": sampled_records,
    }

    payload = _trim_for_soft_file_cap(payload, max_file_bytes=_max_file_bytes(args))
    return _write_dump_file(args, mode=mode, rollout_id=rollout_id, payload=payload)
