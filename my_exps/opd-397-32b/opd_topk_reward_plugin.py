from __future__ import annotations

import aiohttp

from slime.utils.types import Sample

from opd_topk_parser import extract_topk_from_reward


def _get_topk(args) -> int:
    topk = int(getattr(args, "opd_top_logprobs_num", 16))
    if topk <= 0:
        raise ValueError(f"opd_top_logprobs_num must be > 0, got {topk}.")
    return topk


async def reward_func_topk(args, sample: Sample, **kwargs):
    topk = _get_topk(args)
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len < 0:
        raise ValueError(
            f"Invalid sample lengths: len(tokens)={len(sample.tokens)} < response_length={sample.response_length}."
        )

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": topk,
    }

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
