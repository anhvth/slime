from __future__ import annotations

from typing import Any

TOPK_PAD_LOGPROB = -1e9
TOPK_PAD_TOKEN_ID = 0


def parse_input_top_logprobs_rows(
    rows: Any,
    *,
    response_length: int,
    topk: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Parse SGLang ``input_top_logprobs`` rows into fixed-size top-k arrays.

    Returns:
        tuple(logprobs, token_ids), each with shape [response_length, topk].
    """
    if response_length < 0:
        raise ValueError(f"response_length must be >= 0, got {response_length}.")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}.")
    if not isinstance(rows, list):
        raise ValueError(f"input_top_logprobs must be a list, got {type(rows)}.")
    if len(rows) < response_length:
        raise ValueError(
            "input_top_logprobs shorter than response span: "
            f"rows={len(rows)}, response_length={response_length}."
        )

    selected_rows = rows[-response_length:] if response_length > 0 else []

    parsed_logprobs: list[list[float]] = []
    parsed_token_ids: list[list[int]] = []
    for row_idx, row in enumerate(selected_rows):
        if row is None:
            entries = []
        elif isinstance(row, list):
            entries = row
        else:
            raise ValueError(
                f"input_top_logprobs[{row_idx}] must be a list or None, got {type(row)}."
            )

        row_logprobs: list[float] = []
        row_token_ids: list[int] = []
        for col_idx, entry in enumerate(entries[:topk]):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(
                    "input_top_logprobs row entry must be [logprob, token_id, ...], "
                    f"got {entry!r} at row={row_idx}, col={col_idx}."
                )

            logprob = entry[0]
            token_id = entry[1]

            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                raise ValueError(
                    f"logprob must be numeric, got {type(logprob)} at row={row_idx}, col={col_idx}."
                )
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(
                    f"token_id must be int, got {type(token_id)} at row={row_idx}, col={col_idx}."
                )

            row_logprobs.append(float(logprob))
            row_token_ids.append(int(token_id))

        if len(row_logprobs) < topk:
            pad = topk - len(row_logprobs)
            row_logprobs.extend([TOPK_PAD_LOGPROB] * pad)
            row_token_ids.extend([TOPK_PAD_TOKEN_ID] * pad)

        parsed_logprobs.append(row_logprobs)
        parsed_token_ids.append(row_token_ids)

    return parsed_logprobs, parsed_token_ids


def extract_topk_from_reward(
    reward: dict[str, Any],
    *,
    response_length: int,
    topk: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Extract and parse teacher top-k rows from a reward payload."""
    if not isinstance(reward, dict):
        raise ValueError(f"reward payload must be a dict, got {type(reward)}.")

    meta_info = reward.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("reward payload missing dict field: meta_info")

    rows = meta_info.get("input_top_logprobs")
    if rows is None:
        raise ValueError("reward payload missing field: meta_info.input_top_logprobs")

    return parse_input_top_logprobs_rows(rows, response_length=response_length, topk=topk)
