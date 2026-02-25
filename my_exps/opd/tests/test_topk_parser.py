from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from opd_topk_parser import TOPK_PAD_LOGPROB, TOPK_PAD_TOKEN_ID, extract_topk_from_reward, parse_input_top_logprobs_rows


def test_parse_row_padding_and_truncation() -> None:
    rows = [
        [[-0.1, 10, None], [-0.2, 11, None]],
        [[-0.3, 20, None], [-0.4, 21, None], [-0.5, 22, None], [-0.6, 23, None]],
    ]

    logprobs, token_ids = parse_input_top_logprobs_rows(rows, response_length=2, topk=3)

    assert logprobs == [[-0.1, -0.2, TOPK_PAD_LOGPROB], [-0.3, -0.4, -0.5]]
    assert token_ids == [[10, 11, TOPK_PAD_TOKEN_ID], [20, 21, 22]]


def test_parse_uses_response_tail_slice() -> None:
    rows = [
        [[-9.9, 999, None]],
        [[-0.1, 1, None]],
        [[-0.2, 2, None]],
    ]

    logprobs, token_ids = parse_input_top_logprobs_rows(rows, response_length=2, topk=1)

    assert logprobs == [[-0.1], [-0.2]]
    assert token_ids == [[1], [2]]


@pytest.mark.parametrize(
    "bad_rows",
    [
        "not-a-list",
        ["bad-row"],
        [["bad-entry"]],
        [[["not-float", 1, None]]],
        [[[-0.1, "not-int", None]]],
    ],
)
def test_parse_rejects_malformed_rows(bad_rows) -> None:
    with pytest.raises(ValueError):
        parse_input_top_logprobs_rows(bad_rows, response_length=1, topk=1)


def test_parse_rejects_short_rows_for_response_span() -> None:
    rows = [[[-0.1, 1, None]]]
    with pytest.raises(ValueError, match="shorter than response span"):
        parse_input_top_logprobs_rows(rows, response_length=2, topk=1)


def test_extract_topk_from_reward_requires_meta_info_field() -> None:
    reward = {"meta_info": {}}
    with pytest.raises(ValueError, match="input_top_logprobs"):
        extract_topk_from_reward(reward, response_length=1, topk=1)
