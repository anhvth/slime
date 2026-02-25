from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from functools import lru_cache
from typing import Any

import aiohttp
from aiohttp import ClientError
from transformers import AutoTokenizer

from slime.utils.types import Sample

from opd_topk_parser import TOPK_PAD_LOGPROB, TOPK_PAD_TOKEN_ID, extract_topk_from_reward

_HTTP_SESSION: aiohttp.ClientSession | None = None
_HTTP_SESSION_LOOP: asyncio.AbstractEventLoop | None = None
_HTTP_SESSION_URL: str | None = None
logger = logging.getLogger(__name__)


def _get_topk(args) -> int:
    topk = int(getattr(args, "opd_top_logprobs_num", 16))
    if topk <= 0:
        raise ValueError(f"opd_top_logprobs_num must be > 0, got {topk}.")
    return topk


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


def _is_privileged_enabled(args) -> bool:
    return _normalize_bool(getattr(args, "opd_privileged_enable", False), default=False)


def _is_cross_tokenizer_enabled(args) -> bool:
    return _normalize_bool(getattr(args, "opd_cross_tokenizer_enable", False), default=False)


def _stringify_privileged_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)

    text = text.strip()
    return text if text else None


def _resolve_privileged_context(args, sample: Sample) -> str | None:
    metadata_key = str(getattr(args, "opd_privileged_metadata_key", "privileged_context") or "privileged_context")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    privileged_context = _stringify_privileged_value(metadata.get(metadata_key))
    if privileged_context is not None:
        return privileged_context

    fallback_label = _normalize_bool(getattr(args, "opd_privileged_fallback_label", True), default=True)
    if fallback_label:
        return _stringify_privileged_value(sample.label)

    return None


def _get_privileged_tokenizer_path(args) -> str:
    tokenizer_path = str(getattr(args, "opd_privileged_tokenizer_path", "") or "").strip()
    if tokenizer_path:
        return tokenizer_path

    hf_checkpoint = str(getattr(args, "hf_checkpoint", "") or "").strip()
    if hf_checkpoint:
        return hf_checkpoint

    raise ValueError(
        "Cannot resolve tokenizer path for privileged distillation. "
        "Set `opd_privileged_tokenizer_path` in custom config or provide --hf-checkpoint."
    )


def _get_teacher_tokenizer_path(args) -> str:
    path = str(getattr(args, "opd_teacher_tokenizer_path", "") or "").strip()
    if path:
        return path
    raise ValueError(
        "Cannot resolve teacher tokenizer path for cross-tokenizer distillation. "
        "Set `opd_teacher_tokenizer_path` in custom config."
    )


def _get_student_tokenizer_path(args) -> str:
    path = str(getattr(args, "opd_student_tokenizer_path", "") or "").strip()
    if path:
        return path

    hf_checkpoint = str(getattr(args, "hf_checkpoint", "") or "").strip()
    if hf_checkpoint:
        return hf_checkpoint

    raise ValueError(
        "Cannot resolve student tokenizer path for cross-tokenizer distillation. "
        "Set `opd_student_tokenizer_path` in custom config or provide --hf-checkpoint."
    )


@lru_cache(maxsize=8)
def _get_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


@lru_cache(maxsize=4)
def _build_teacher_to_student_token_map(
    teacher_tokenizer_path: str,
    student_tokenizer_path: str,
) -> dict[int, int]:
    """Pre-compute teacher_token_id -> student_token_id mapping (one-time cost).

    Only includes entries where a teacher token decodes to text that encodes
    to exactly one student token.  Cache miss at lookup time means "no valid
    1-to-1 mapping" — same semantics as the old per-call fallback.
    """
    teacher_tokenizer = _get_tokenizer(teacher_tokenizer_path)
    student_tokenizer = _get_tokenizer(student_tokenizer_path)

    mapping: dict[int, int] = {}
    vocab_size = len(teacher_tokenizer)  # includes added tokens

    for tid in range(vocab_size):
        try:
            surface = teacher_tokenizer.decode(
                [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            student_ids = student_tokenizer.encode(surface, add_special_tokens=False)
            if len(student_ids) == 1:
                mapping[tid] = int(student_ids[0])
        except Exception:
            pass

    return mapping


def _get_privileged_tag(args, name: str, default: str) -> str:
    value = str(getattr(args, name, "") or "").strip()
    return value if value else default


def _get_float_option(args, name: str, default: float) -> float:
    value = getattr(args, name, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _get_int_option(args, name: str, default: int) -> int:
    value = getattr(args, name, default)
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _summarize_seconds(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
    sorted_values = sorted(float(v) for v in values)
    value_count = len(sorted_values)
    mid = value_count // 2
    if value_count % 2 == 1:
        median = sorted_values[mid]
    else:
        median = 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    return {
        "mean": sum(sorted_values) / value_count,
        "median": median,
        "max": sorted_values[-1],
        "min": sorted_values[0],
    }


async def _close_http_session() -> None:
    global _HTTP_SESSION, _HTTP_SESSION_LOOP, _HTTP_SESSION_URL
    if _HTTP_SESSION is not None and not _HTTP_SESSION.closed:
        try:
            await _HTTP_SESSION.close()
        except RuntimeError:
            pass
    _HTTP_SESSION = None
    _HTTP_SESSION_LOOP = None
    _HTTP_SESSION_URL = None


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientOSError,
            asyncio.TimeoutError,
        ),
    ):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _is_retryable_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).strip().lower()
    return any(
        token in message
        for token in (
            "connection closed",
            "session is closed",
            "connector is closed",
            "cannot write to closing transport",
        )
    )


async def _get_http_session(args) -> aiohttp.ClientSession:
    global _HTTP_SESSION, _HTTP_SESSION_LOOP, _HTTP_SESSION_URL

    loop = asyncio.get_running_loop()
    rm_url = str(getattr(args, "rm_url", "") or "").strip()
    must_recreate = (
        _HTTP_SESSION is None
        or _HTTP_SESSION.closed
        or _HTTP_SESSION_LOOP is not loop
        or _HTTP_SESSION_URL != rm_url
    )
    if not must_recreate:
        return _HTTP_SESSION

    if _HTTP_SESSION is not None and not _HTTP_SESSION.closed:
        await _close_http_session()

    connect_timeout = _get_float_option(args, "opd_rm_connect_timeout_s", 2.0)
    sock_read_timeout = _get_float_option(args, "opd_rm_read_timeout_s", 120.0)
    total_timeout = _get_float_option(args, "opd_rm_total_timeout_s", 180.0)
    max_connections = _get_int_option(args, "opd_rm_max_connections", 512)
    max_connections_per_host = _get_int_option(args, "opd_rm_max_connections_per_host", 256)

    timeout = aiohttp.ClientTimeout(
        total=total_timeout,
        connect=connect_timeout,
        sock_connect=connect_timeout,
        sock_read=sock_read_timeout,
    )
    connector = aiohttp.TCPConnector(
        limit=max_connections,
        limit_per_host=max_connections_per_host,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    _HTTP_SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
    _HTTP_SESSION_LOOP = loop
    _HTTP_SESSION_URL = rm_url
    return _HTTP_SESSION


def _decode_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _encode_privileged_suffix_ids(args, tokenizer, privileged_context: str) -> list[int]:
    open_tag = _get_privileged_tag(args, "opd_privileged_open_tag", "[PRIVILEGED_CONTEXT]")
    close_tag = _get_privileged_tag(args, "opd_privileged_close_tag", "[/PRIVILEGED_CONTEXT]")
    suffix = f"\n{open_tag}\n{privileged_context}\n{close_tag}\n"
    return [int(tid) for tid in tokenizer.encode(suffix, add_special_tokens=False)]


def _build_teacher_input_ids_and_start_len_same_tokenizer(args, sample: Sample) -> tuple[list[int], int]:
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len < 0:
        raise ValueError(
            f"Invalid sample lengths: len(tokens)={len(sample.tokens)} < response_length={sample.response_length}."
        )

    prompt_ids = list(sample.tokens[:prompt_len])
    response_ids = list(sample.tokens[prompt_len:])
    if len(response_ids) != sample.response_length:
        raise ValueError(
            "Response slicing mismatch: "
            f"len(response_ids)={len(response_ids)} vs response_length={sample.response_length}."
        )

    if not _is_privileged_enabled(args):
        return prompt_ids + response_ids, len(prompt_ids)

    privileged_context = _resolve_privileged_context(args, sample=sample)
    if privileged_context is None:
        return prompt_ids + response_ids, len(prompt_ids)

    tokenizer = _get_tokenizer(_get_privileged_tokenizer_path(args))
    privileged_ids = _encode_privileged_suffix_ids(args, tokenizer, privileged_context)

    scored_prompt_ids = prompt_ids + privileged_ids
    return scored_prompt_ids + response_ids, len(scored_prompt_ids)


def _build_teacher_input_ids_and_start_len_cross_tokenizer(args, sample: Sample) -> tuple[list[int], int]:
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len < 0:
        raise ValueError(
            f"Invalid sample lengths: len(tokens)={len(sample.tokens)} < response_length={sample.response_length}."
        )

    student_prompt_ids = list(sample.tokens[:prompt_len])
    student_response_ids = list(sample.tokens[prompt_len:])
    if len(student_response_ids) != sample.response_length:
        raise ValueError(
            "Response slicing mismatch: "
            f"len(response_ids)={len(student_response_ids)} vs response_length={sample.response_length}."
        )

    student_tokenizer = _get_tokenizer(_get_student_tokenizer_path(args))
    teacher_tokenizer = _get_tokenizer(_get_teacher_tokenizer_path(args))

    prompt_text = _decode_ids(student_tokenizer, student_prompt_ids)
    response_text = _decode_ids(student_tokenizer, student_response_ids)

    teacher_prompt_ids = [int(tid) for tid in teacher_tokenizer.encode(prompt_text, add_special_tokens=False)]
    teacher_response_ids = [int(tid) for tid in teacher_tokenizer.encode(response_text, add_special_tokens=False)]

    privileged_ids: list[int] = []
    if _is_privileged_enabled(args):
        privileged_context = _resolve_privileged_context(args, sample=sample)
        if privileged_context is not None:
            privileged_ids = _encode_privileged_suffix_ids(args, teacher_tokenizer, privileged_context)

    scored_prompt_ids = teacher_prompt_ids + privileged_ids
    return scored_prompt_ids + teacher_response_ids, len(scored_prompt_ids)


def _build_teacher_input_ids_and_start_len(args, sample: Sample) -> tuple[list[int], int]:
    if _is_cross_tokenizer_enabled(args):
        return _build_teacher_input_ids_and_start_len_cross_tokenizer(args, sample)
    return _build_teacher_input_ids_and_start_len_same_tokenizer(args, sample)


def _resolve_teacher_score_logprob_start_len(logical_start_len: int) -> int:
    """Shift teacher scoring window by one token to recover first response-token supervision.

    Some teacher endpoints return an unusable first scored row (e.g. input_top_logprobs[0] is None).
    Requesting scores from one token earlier lets us recover supervision for logical response index 0
    while still slicing the final response span using the logical boundary.
    """
    logical_start_len = int(logical_start_len)
    return logical_start_len - 1 if logical_start_len > 0 else logical_start_len


def _build_teacher_payload(args, sample: Sample, topk: int) -> dict:
    input_ids, logical_logprob_start_len = _build_teacher_input_ids_and_start_len(args, sample)
    score_logprob_start_len = _resolve_teacher_score_logprob_start_len(logical_logprob_start_len)
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Request teacher scoring from one token earlier when possible.
        "logprob_start_len": score_logprob_start_len,
        "top_logprobs_num": topk,
        # Keep logical boundary for downstream response-span slicing/debugging.
        "_opd_teacher_logprob_start_len": logical_logprob_start_len,
        "_opd_teacher_score_logprob_start_len": score_logprob_start_len,
    }


async def reward_func_topk(args, sample: Sample, **kwargs):
    topk = _get_topk(args)
    payload = _build_teacher_payload(args, sample, topk)
    score_logprob_start_len = int(payload.get("_opd_teacher_score_logprob_start_len", payload.get("logprob_start_len", 0)))
    logical_logprob_start_len = int(payload.get("_opd_teacher_logprob_start_len", score_logprob_start_len))
    request_payload = {k: v for k, v in payload.items() if not str(k).startswith("_opd_")}
    attempts = _get_int_option(args, "opd_rm_retry_attempts", 5)
    base_sleep = _get_float_option(args, "opd_rm_retry_base_sleep_s", 0.15)
    max_sleep = _get_float_option(args, "opd_rm_retry_max_sleep_s", 2.0)

    start = time.perf_counter()
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            session = await _get_http_session(args)
            async with session.post(args.rm_url, json=request_payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
            latency = time.perf_counter() - start
            if isinstance(result, dict):
                meta_info = result.get("meta_info")
                if isinstance(meta_info, dict):
                    meta_info["client_http_latency"] = latency
                    meta_info["client_http_attempts"] = attempt
                result["_opd_teacher_input_ids"] = request_payload["input_ids"]
                result["_opd_teacher_logprob_start_len"] = logical_logprob_start_len
                result["_opd_teacher_score_logprob_start_len"] = score_logprob_start_len
            return result
        except ClientError as exc:
            last_exc = exc
            if not _is_retryable_http_error(exc) or attempt >= attempts:
                raise
            await _close_http_session()
        except asyncio.TimeoutError as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            await _close_http_session()
        except RuntimeError as exc:
            last_exc = exc
            if not _is_retryable_runtime_error(exc) or attempt >= attempts:
                raise
            await _close_http_session()

        sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
        await asyncio.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Teacher request failed without exception details.")


def _extract_required_meta_info(reward: dict[str, Any]) -> dict[str, Any]:
    meta_info = reward.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("reward payload missing dict field: meta_info")
    return meta_info


def _extract_input_top_logprobs_rows_with_text(
    reward: dict[str, Any],
    *,
    response_length: int,
    topk: int,
) -> list[list[tuple[float, int, str | None]]]:
    if response_length < 0:
        raise ValueError(f"response_length must be >= 0, got {response_length}.")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}.")

    meta_info = _extract_required_meta_info(reward)
    rows = meta_info.get("input_top_logprobs")
    if not isinstance(rows, list):
        raise ValueError(f"meta_info.input_top_logprobs must be a list, got {type(rows)}.")
    if len(rows) < response_length:
        raise ValueError(
            "input_top_logprobs shorter than response span: "
            f"rows={len(rows)}, response_length={response_length}."
        )

    selected_rows = rows[-response_length:] if response_length > 0 else []
    parsed: list[list[tuple[float, int, str | None]]] = []
    for row_idx, row in enumerate(selected_rows):
        if row is None:
            entries = []
        elif isinstance(row, list):
            entries = row
        else:
            raise ValueError(f"input_top_logprobs[{row_idx}] must be a list or None, got {type(row)}.")

        parsed_row: list[tuple[float, int, str | None]] = []
        for col_idx, entry in enumerate(entries[:topk]):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(
                    "input_top_logprobs row entry must be [logprob, token_id, ...], "
                    f"got {entry!r} at row={row_idx}, col={col_idx}."
                )

            logprob = entry[0]
            token_id = entry[1]
            token_text = entry[2] if len(entry) > 2 else None

            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                raise ValueError(
                    f"logprob must be numeric, got {type(logprob)} at row={row_idx}, col={col_idx}."
                )
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(
                    f"token_id must be int, got {type(token_id)} at row={row_idx}, col={col_idx}."
                )
            if token_text is not None and not isinstance(token_text, str):
                token_text = None

            parsed_row.append((float(logprob), int(token_id), token_text))
        parsed.append(parsed_row)

    return parsed


def _extract_input_token_logprob_rows(
    reward: dict[str, Any],
    *,
    response_length: int,
) -> list[tuple[float | None, int, str | None]]:
    if response_length < 0:
        raise ValueError(f"response_length must be >= 0, got {response_length}.")

    meta_info = _extract_required_meta_info(reward)
    rows = meta_info.get("input_token_logprobs")
    if not isinstance(rows, list):
        raise ValueError(f"meta_info.input_token_logprobs must be a list, got {type(rows)}.")
    if len(rows) < response_length:
        raise ValueError(
            "input_token_logprobs shorter than response span: "
            f"rows={len(rows)}, response_length={response_length}."
        )

    selected_rows = rows[-response_length:] if response_length > 0 else []
    parsed: list[tuple[float | None, int, str | None]] = []
    for row_idx, row in enumerate(selected_rows):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(
                f"input_token_logprobs[{row_idx}] must be [logprob, token_id, ...], got {row!r}."
            )

        logprob = row[0]
        token_id = row[1]
        token_text = row[2] if len(row) > 2 else None
        if logprob is not None and (isinstance(logprob, bool) or not isinstance(logprob, (int, float))):
            raise ValueError(
                "input_token_logprobs["
                f"{row_idx}] logprob must be numeric or None, got {type(logprob)}."
            )
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"input_token_logprobs[{row_idx}] token_id must be int, got {type(token_id)}.")
        if token_text is not None and not isinstance(token_text, str):
            token_text = None

        parsed.append((None if logprob is None else float(logprob), int(token_id), token_text))

    return parsed


def _to_canonical_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    pieces: list[str] = []
    prev = ""
    for idx in range(len(token_ids)):
        cur = tokenizer.decode(token_ids[: idx + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if cur.startswith(prev):
            piece = cur[len(prev) :]
        else:
            piece = tokenizer.decode([token_ids[idx]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pieces.append(piece)
        prev = cur
    return pieces


def _build_alignment_groups(
    student_tokenizer,
    teacher_tokenizer,
    student_token_ids: list[int],
    teacher_token_ids: list[int],
) -> tuple[list[list[int]], list[list[int]], bool]:
    s_pieces = _to_canonical_pieces(student_tokenizer, student_token_ids)
    t_pieces = _to_canonical_pieces(teacher_tokenizer, teacher_token_ids)

    i = 0
    j = 0
    s_buf = ""
    t_buf = ""
    s_group: list[int] = []
    t_group: list[int] = []
    s_groups: list[list[int]] = []
    t_groups: list[list[int]] = []
    clean = True

    def flush_groups() -> None:
        nonlocal s_group, t_group
        if s_group and t_group:
            s_groups.append(s_group.copy())
            t_groups.append(t_group.copy())
        else:
            nonlocal clean
            clean = False
        s_group = []
        t_group = []

    while i < len(s_pieces) or j < len(t_pieces):
        if s_buf == t_buf and s_buf != "":
            flush_groups()
            s_buf = ""
            t_buf = ""
            continue

        if s_buf == "" and i < len(s_pieces):
            s_buf += s_pieces[i]
            s_group.append(i)
            i += 1
            continue
        if t_buf == "" and j < len(t_pieces):
            t_buf += t_pieces[j]
            t_group.append(j)
            j += 1
            continue

        if len(s_buf) <= len(t_buf):
            if i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
            elif j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
            else:
                break
        else:
            if j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
            elif i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
            else:
                break

    if s_buf == t_buf and s_buf != "":
        flush_groups()
    elif s_group or t_group:
        clean = False

    if i != len(s_pieces) or j != len(t_pieces):
        clean = False

    return s_groups, t_groups, clean


def _build_fallback_alignment_groups(
    student_length: int,
    teacher_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    s_groups: list[list[int]] = []
    t_groups: list[list[int]] = []
    common = min(student_length, teacher_length)

    for idx in range(common):
        s_groups.append([idx])
        t_groups.append([idx])

    for idx in range(common, student_length):
        s_groups.append([idx])
        t_groups.append([])

    return s_groups, t_groups


def _logsumexp_pair(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _extract_teacher_response_ids_from_reward(reward: dict[str, Any]) -> list[int]:
    teacher_input_ids = reward.get("_opd_teacher_input_ids")
    teacher_logprob_start_len = reward.get("_opd_teacher_logprob_start_len")
    if not isinstance(teacher_input_ids, list):
        raise ValueError("reward payload missing list field: _opd_teacher_input_ids")
    if teacher_logprob_start_len is None:
        raise ValueError("reward payload missing field: _opd_teacher_logprob_start_len")

    start = int(teacher_logprob_start_len)
    if start < 0 or start > len(teacher_input_ids):
        raise ValueError(
            "Invalid _opd_teacher_logprob_start_len: "
            f"{start}, len(input_ids)={len(teacher_input_ids)}"
        )

    return [int(x) for x in teacher_input_ids[start:]]


def _build_cross_tokenizer_teacher_targets(
    args,
    sample: Sample,
    reward: dict[str, Any],
    *,
    topk: int,
    timing_stats: dict[str, float] | None = None,
) -> tuple[list[list[float]], list[list[int]], list[int], list[int]]:
    total_start_time = time.perf_counter()

    extract_rows_start_time = time.perf_counter()
    student_response_ids = [int(x) for x in sample.tokens[-sample.response_length :]] if sample.response_length > 0 else []
    student_length = len(student_response_ids)

    teacher_response_ids = _extract_teacher_response_ids_from_reward(reward)
    teacher_length = len(teacher_response_ids)

    teacher_top_rows = _extract_input_top_logprobs_rows_with_text(
        reward,
        response_length=teacher_length,
        topk=topk,
    )
    teacher_token_rows = _extract_input_token_logprob_rows(
        reward,
        response_length=teacher_length,
    )
    extract_rows_time_s = time.perf_counter() - extract_rows_start_time

    build_groups_start_time = time.perf_counter()
    student_tokenizer = _get_tokenizer(_get_student_tokenizer_path(args))
    teacher_tokenizer = _get_tokenizer(_get_teacher_tokenizer_path(args))

    s_groups, t_groups, clean = _build_alignment_groups(
        student_tokenizer,
        teacher_tokenizer,
        student_response_ids,
        teacher_response_ids,
    )
    fallback_used = 0
    if not clean:
        s_groups, t_groups = _build_fallback_alignment_groups(student_length=student_length, teacher_length=teacher_length)
        fallback_used = 1
    build_alignment_groups_time_s = time.perf_counter() - build_groups_start_time

    topk_logprobs = [[TOPK_PAD_LOGPROB] * topk for _ in range(student_length)]
    topk_token_ids = [[TOPK_PAD_TOKEN_ID] * topk for _ in range(student_length)]
    group_lengths = [0] * student_length
    group_valid_mask = [0] * student_length

    project_support_start_time = time.perf_counter()
    for s_group, t_group in zip(s_groups, t_groups, strict=True):
        if not s_group:
            continue

        s_anchor = int(s_group[0])
        if s_anchor < 0 or s_anchor >= student_length:
            continue

        s_group_len = int(len(s_group))
        group_lengths[s_anchor] = s_group_len

        if not t_group:
            continue

        t_anchor = int(t_group[0])
        if t_anchor < 0 or t_anchor >= teacher_length:
            continue

        continuation_logprob = 0.0
        valid_continuation = True
        for pos in t_group[1:]:
            if pos < 0 or pos >= teacher_length:
                valid_continuation = False
                break
            continuation_token_logprob = teacher_token_rows[pos][0]
            if continuation_token_logprob is None:
                valid_continuation = False
                break
            continuation_logprob += float(continuation_token_logprob)
        if not valid_continuation:
            continue

        row_entries = teacher_top_rows[t_anchor]
        merged_by_student_id: dict[int, float] = {}
        for logprob, token_id, token_text in row_entries:
            token_surface = token_text
            if token_surface is None:
                token_surface = teacher_tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

            student_ids = student_tokenizer.encode(token_surface, add_special_tokens=False)
            if len(student_ids) != 1:
                continue

            sid = int(student_ids[0])
            merged_logprob = float(logprob) + continuation_logprob
            prev = merged_by_student_id.get(sid)
            if prev is None:
                merged_by_student_id[sid] = merged_logprob
            else:
                merged_by_student_id[sid] = _logsumexp_pair(prev, merged_logprob)

        if not merged_by_student_id:
            continue

        sorted_candidates = sorted(merged_by_student_id.items(), key=lambda x: x[1], reverse=True)[:topk]
        for col, (sid, lp) in enumerate(sorted_candidates):
            topk_token_ids[s_anchor][col] = int(sid)
            topk_logprobs[s_anchor][col] = float(lp)

        group_valid_mask[s_anchor] = 1

    project_teacher_support_time_s = time.perf_counter() - project_support_start_time
    total_time_s = time.perf_counter() - total_start_time
    if timing_stats is not None:
        timing_stats.update(
            {
                "extract_rows_time_s": extract_rows_time_s,
                "build_alignment_groups_time_s": build_alignment_groups_time_s,
                "project_teacher_support_time_s": project_teacher_support_time_s,
                "total_time_s": total_time_s,
                "fallback_used": float(fallback_used),
            }
        )

    return topk_logprobs, topk_token_ids, group_lengths, group_valid_mask


def post_process_rewards_topk(args, samples: list[Sample], **kwargs):
    """Extract fixed-size teacher top-k tensors and attach to each sample."""
    stage_start_time = time.perf_counter()
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    topk = _get_topk(args)
    cross_tokenizer = _is_cross_tokenizer_enabled(args)
    sample_count = len(samples)
    progress_interval = int(getattr(args, "opd_postprocess_progress_interval", 0) or 0)
    if progress_interval <= 0:
        progress_interval = max(1, sample_count // 10)
    logger.info(
        (
            "[DEBUG][opd_topk] Begin reward post-process: sample_count=%s, topk=%s, "
            "cross_tokenizer=%s, progress_interval=%s"
        ),
        sample_count,
        topk,
        int(cross_tokenizer),
        progress_interval,
    )
    alignment_total_times: list[float] = []
    alignment_build_groups_times: list[float] = []
    alignment_project_support_times: list[float] = []
    alignment_fallback_flags: list[float] = []

    for i, (sample, reward, response_length) in enumerate(
        zip(samples, raw_rewards, response_lengths, strict=False),
        start=1,
    ):
        if cross_tokenizer:
            if not isinstance(reward, dict):
                raise ValueError(f"reward payload must be a dict in cross-tokenizer mode, got {type(reward)}.")
            timing_stats: dict[str, float] = {}
            topk_logprobs, topk_token_ids, group_lengths, group_valid_mask = _build_cross_tokenizer_teacher_targets(
                args,
                sample,
                reward,
                topk=topk,
                timing_stats=timing_stats,
            )
            if len(topk_logprobs) != response_length:
                raise ValueError(
                    "Cross-tokenizer top-k length mismatch: "
                    f"len(topk_logprobs)={len(topk_logprobs)} vs response_length={response_length}."
                )
            alignment_total_times.append(float(timing_stats.get("total_time_s", 0.0)))
            alignment_build_groups_times.append(float(timing_stats.get("build_alignment_groups_time_s", 0.0)))
            alignment_project_support_times.append(float(timing_stats.get("project_teacher_support_time_s", 0.0)))
            alignment_fallback_flags.append(float(timing_stats.get("fallback_used", 0.0)))
            sample.teacher_topk_group_lengths = group_lengths
            sample.teacher_topk_group_valid_mask = group_valid_mask
        else:
            topk_logprobs, topk_token_ids = extract_topk_from_reward(
                reward,
                response_length=response_length,
                topk=topk,
            )
            sample.teacher_topk_group_lengths = None
            sample.teacher_topk_group_valid_mask = None

        sample.teacher_topk_logprobs = topk_logprobs
        sample.teacher_topk_token_ids = topk_token_ids
        if isinstance(reward, dict):
            teacher_input_ids = reward.get("_opd_teacher_input_ids")
            if isinstance(teacher_input_ids, list):
                sample.teacher_input_ids = [int(x) for x in teacher_input_ids]
            teacher_logprob_start_len = reward.get("_opd_teacher_logprob_start_len")
            if teacher_logprob_start_len is not None:
                sample.teacher_logprob_start_len = int(teacher_logprob_start_len)
            teacher_score_logprob_start_len = reward.get("_opd_teacher_score_logprob_start_len")
            if teacher_score_logprob_start_len is not None:
                # Optional diagnostic field; Sample may not have a declared attribute in older versions.
                setattr(sample, "teacher_score_logprob_start_len", int(teacher_score_logprob_start_len))

        if i % progress_interval == 0 or i == sample_count:
            elapsed_s = time.perf_counter() - stage_start_time
            avg_s = elapsed_s / i if i > 0 else 0.0
            logger.info(
                "[DEBUG][opd_topk] reward post-process progress: %s/%s (%.1f%%), elapsed=%.2fs, avg=%.4fs/sample",
                i,
                sample_count,
                100.0 * i / max(sample_count, 1),
                elapsed_s,
                avg_s,
            )

    metrics_dict: dict[str, float] = {}
    if cross_tokenizer:
        total_stats = _summarize_seconds(alignment_total_times)
        build_stats = _summarize_seconds(alignment_build_groups_times)
        project_stats = _summarize_seconds(alignment_project_support_times)
        sample_count = len(alignment_total_times)
        fallback_ratio = (sum(alignment_fallback_flags) / sample_count) if sample_count > 0 else 0.0
        metrics_dict = {
            "perf/opd_alignment/total_time_s/mean": total_stats["mean"],
            "perf/opd_alignment/total_time_s/median": total_stats["median"],
            "perf/opd_alignment/total_time_s/max": total_stats["max"],
            "perf/opd_alignment/total_time_s/min": total_stats["min"],
            "perf/opd_alignment/build_groups_time_s/mean": build_stats["mean"],
            "perf/opd_alignment/build_groups_time_s/median": build_stats["median"],
            "perf/opd_alignment/build_groups_time_s/max": build_stats["max"],
            "perf/opd_alignment/build_groups_time_s/min": build_stats["min"],
            "perf/opd_alignment/project_support_time_s/mean": project_stats["mean"],
            "perf/opd_alignment/project_support_time_s/median": project_stats["median"],
            "perf/opd_alignment/project_support_time_s/max": project_stats["max"],
            "perf/opd_alignment/project_support_time_s/min": project_stats["min"],
            "perf/opd_alignment/fallback_ratio": fallback_ratio,
            "perf/opd_alignment/sample_count": float(sample_count),
        }

    scalar_rewards = [0.0] * len(samples)
    logger.info(
        "[DEBUG][opd_topk] Finished reward post-process in %.2fs",
        time.perf_counter() - stage_start_time,
    )
    return scalar_rewards, scalar_rewards, metrics_dict
