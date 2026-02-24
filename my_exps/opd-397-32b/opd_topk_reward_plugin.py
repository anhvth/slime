from __future__ import annotations

import json
from functools import lru_cache

import aiohttp
from transformers import AutoTokenizer

from slime.utils.types import Sample

from opd_topk_parser import extract_topk_from_reward


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

    open_tag = str(getattr(args, "opd_privileged_open_tag", "[PRIVILEGED_CONTEXT]") or "[PRIVILEGED_CONTEXT]")
    close_tag = str(
        getattr(args, "opd_privileged_close_tag", "[/PRIVILEGED_CONTEXT]") or "[/PRIVILEGED_CONTEXT]"
    )
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

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


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
