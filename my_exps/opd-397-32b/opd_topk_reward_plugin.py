from __future__ import annotations

import asyncio
import json
import time
from functools import lru_cache

import aiohttp
from aiohttp import ClientError
from transformers import AutoTokenizer

from slime.utils.types import Sample

from opd_topk_parser import extract_topk_from_reward

_HTTP_SESSION: aiohttp.ClientSession | None = None
_HTTP_SESSION_LOOP: asyncio.AbstractEventLoop | None = None
_HTTP_SESSION_URL: str | None = None


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


def _get_tokenizer_path(args) -> str:
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


@lru_cache(maxsize=4)
def _get_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


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


def _build_teacher_input_ids_and_start_len(args, sample: Sample) -> tuple[list[int], int]:
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

    privileged_context = _resolve_privileged_context(args, sample)
    if privileged_context is None:
        return prompt_ids + response_ids, len(prompt_ids)

    open_tag = _get_privileged_tag(args, "opd_privileged_open_tag", "[PRIVILEGED_CONTEXT]")
    close_tag = _get_privileged_tag(args, "opd_privileged_close_tag", "[/PRIVILEGED_CONTEXT]")
    suffix = f"\n{open_tag}\n{privileged_context}\n{close_tag}\n"

    tokenizer = _get_tokenizer(_get_tokenizer_path(args))
    privileged_ids = [int(tid) for tid in tokenizer.encode(suffix, add_special_tokens=False)]
    scored_prompt_ids = prompt_ids + privileged_ids
    return scored_prompt_ids + response_ids, len(scored_prompt_ids)


def _build_teacher_payload(args, sample: Sample, topk: int) -> dict:
    input_ids, logprob_start_len = _build_teacher_input_ids_and_start_len(args, sample)
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": topk,
    }


async def reward_func_topk(args, sample: Sample, **kwargs):
    topk = _get_topk(args)
    payload = _build_teacher_payload(args, sample, topk)
    attempts = _get_int_option(args, "opd_rm_retry_attempts", 5)
    base_sleep = _get_float_option(args, "opd_rm_retry_base_sleep_s", 0.15)
    max_sleep = _get_float_option(args, "opd_rm_retry_max_sleep_s", 2.0)

    start = time.perf_counter()
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            session = await _get_http_session(args)
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
            latency = time.perf_counter() - start
            if isinstance(result, dict):
                meta_info = result.get("meta_info")
                if isinstance(meta_info, dict):
                    meta_info["client_http_latency"] = latency
                    meta_info["client_http_attempts"] = attempt
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

        sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
        await asyncio.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Teacher request failed without exception details.")


def post_process_rewards_topk(args, samples: list[Sample], **kwargs):
    """Extract fixed-size teacher top-k tensors and attach to each sample."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    topk = _get_topk(args)

    for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        topk_logprobs, topk_token_ids = extract_topk_from_reward(
            reward,
            response_length=response_length,
            topk=topk,
        )
        sample.teacher_topk_logprobs = topk_logprobs
        sample.teacher_topk_token_ids = topk_token_ids

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
