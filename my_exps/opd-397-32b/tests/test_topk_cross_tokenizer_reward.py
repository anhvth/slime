from __future__ import annotations

import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import opd_topk_reward_plugin as reward_plugin
from opd_topk_parser import TOPK_PAD_LOGPROB, TOPK_PAD_TOKEN_ID


class _DummyTokenizer:
    def __init__(self, *, encode_map: dict[str, list[int]], decode_map: dict[int, str]):
        self._encode_map = {k: list(v) for k, v in encode_map.items()}
        self._decode_map = {int(k): v for k, v in decode_map.items()}

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return list(self._encode_map.get(text, []))

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False):
        del skip_special_tokens, clean_up_tokenization_spaces
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._decode_map.get(int(tid), "") for tid in token_ids)


def _make_args(**kwargs) -> Namespace:
    defaults = {
        "opd_cross_tokenizer_enable": 1,
        "opd_teacher_tokenizer_path": "teacher",
        "opd_student_tokenizer_path": "student",
        "opd_privileged_enable": 0,
        "opd_privileged_metadata_key": "privileged_context",
        "opd_privileged_fallback_label": 1,
        "opd_top_logprobs_num": 2,
        "hf_checkpoint": "/tmp/fake_hf",
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_cross_tokenizer_payload_retokenizes_teacher_input(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={
            "P": [10],
            "ABC": [101, 102],
        },
        decode_map={10: "P", 101: "AB", 102: "C"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={
            "P": [301],
            "ABC": [201, 202, 203],
        },
        decode_map={301: "P", 201: "A", 202: "B", 203: "C"},
    )

    def _fake_get_tokenizer(path: str):
        if path == "student":
            return student_tok
        if path == "teacher":
            return teacher_tok
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", _fake_get_tokenizer)

    args = _make_args()
    sample = SimpleNamespace(tokens=[10, 101, 102], response_length=2, metadata={}, label=None)

    payload = reward_plugin._build_teacher_payload(args, sample, topk=4)

    assert payload["input_ids"] == [301, 201, 202, 203]
    assert payload["logprob_start_len"] == 1
    assert payload["top_logprobs_num"] == 4


def test_cross_tokenizer_targets_alignment_and_duplicate_merge(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={
            "AB": [101],
            "C": [102],
            "ZZ": [401, 402],
            "CC": [501, 502],
        },
        decode_map={10: "P", 101: "AB", 102: "C"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={
            "P": [301],
            "ABC": [201, 202, 203],
        },
        decode_map={301: "P", 201: "A", 202: "B", 203: "C"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    args = _make_args(opd_top_logprobs_num=2)
    sample = SimpleNamespace(tokens=[10, 101, 102], response_length=2, metadata={}, label=None)
    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.1, 910, "AB"], [-0.3, 911, "AB"], [-0.2, 912, "ZZ"]],
                [[-0.5, 920, "B"]],
                [[-0.2, 930, "C"], [-1.5, 931, "CC"]],
            ],
            "input_token_logprobs": [
                [-0.4, 201, "A"],
                [-0.7, 202, "B"],
                [-0.3, 203, "C"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201, 202, 203],
        "_opd_teacher_logprob_start_len": 1,
    }

    topk_logprobs, topk_token_ids, group_lengths, group_valid_mask = reward_plugin._build_cross_tokenizer_teacher_targets(
        args,
        sample,
        reward,
        topk=2,
    )

    expected_merged = math.log(math.exp(-0.8) + math.exp(-1.0))
    assert group_lengths == [1, 1]
    assert group_valid_mask == [1, 1]

    assert topk_token_ids[0] == [101, TOPK_PAD_TOKEN_ID]
    assert topk_logprobs[0][0] == pytest.approx(expected_merged, abs=1e-6)
    assert topk_logprobs[0][1] == TOPK_PAD_LOGPROB

    assert topk_token_ids[1] == [102, TOPK_PAD_TOKEN_ID]
    assert topk_logprobs[1][0] == pytest.approx(-0.2, abs=1e-6)
    assert topk_logprobs[1][1] == TOPK_PAD_LOGPROB


def test_cross_tokenizer_fallback_marks_unmatched_student_groups_invalid(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={
            "A": [111],
            "B": [112],
        },
        decode_map={10: "P", 111: "A", 112: "B"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={
            "P": [301],
            "AB": [201],
        },
        decode_map={301: "P", 201: "A"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    args = _make_args(opd_top_logprobs_num=2)
    sample = SimpleNamespace(tokens=[10, 111, 112], response_length=2, metadata={}, label=None)
    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.2, 901, "A"]],
            ],
            "input_token_logprobs": [
                [-0.6, 201, "A"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201],
        "_opd_teacher_logprob_start_len": 1,
    }

    topk_logprobs, topk_token_ids, group_lengths, group_valid_mask = reward_plugin._build_cross_tokenizer_teacher_targets(
        args,
        sample,
        reward,
        topk=2,
    )

    assert group_lengths == [1, 1]
    assert group_valid_mask == [1, 0]
    assert topk_token_ids[0][0] == 111
    assert topk_logprobs[0][0] == pytest.approx(-0.2, abs=1e-6)
    assert topk_token_ids[1] == [TOPK_PAD_TOKEN_ID, TOPK_PAD_TOKEN_ID]
    assert topk_logprobs[1] == [TOPK_PAD_LOGPROB, TOPK_PAD_LOGPROB]


def test_post_process_rewards_topk_sets_group_fields_for_cross_mode(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={"A": [111]},
        decode_map={10: "P", 111: "A"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={"P": [301], "A": [201]},
        decode_map={301: "P", 201: "A"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.1, 901, "A"]],
            ],
            "input_token_logprobs": [
                [-0.3, 201, "A"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201],
        "_opd_teacher_logprob_start_len": 1,
    }

    sample = SimpleNamespace(
        tokens=[10, 111],
        response_length=1,
        metadata={},
        label=None,
        teacher_topk_logprobs=None,
        teacher_topk_token_ids=None,
        teacher_topk_group_lengths=None,
        teacher_topk_group_valid_mask=None,
        teacher_input_ids=None,
        teacher_logprob_start_len=None,
    )
    sample.get_reward_value = lambda _args: reward

    args = _make_args(opd_top_logprobs_num=1)
    reward_plugin.post_process_rewards_topk(args, [sample])

    assert sample.teacher_topk_group_lengths == [1]
    assert sample.teacher_topk_group_valid_mask == [1]
    assert sample.teacher_topk_token_ids == [[111]]


def test_cross_tokenizer_accepts_none_anchor_logprob_without_crashing(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={"P": [10], "A": [111]},
        decode_map={10: "P", 111: "A"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={"P": [301], "A": [201]},
        decode_map={301: "P", 201: "A"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    args = _make_args(opd_top_logprobs_num=1)
    sample = SimpleNamespace(tokens=[10, 111], response_length=1, metadata={}, label=None)
    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.2, 901, "A"]],
            ],
            "input_token_logprobs": [
                [None, 201, "A"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201],
        "_opd_teacher_logprob_start_len": 1,
    }

    topk_logprobs, topk_token_ids, group_lengths, group_valid_mask = reward_plugin._build_cross_tokenizer_teacher_targets(
        args,
        sample,
        reward,
        topk=1,
    )

    assert group_lengths == [1]
    assert group_valid_mask == [1]
    assert topk_token_ids == [[111]]
    assert topk_logprobs[0][0] == pytest.approx(-0.2, abs=1e-6)


def test_cross_tokenizer_skips_group_when_continuation_logprob_is_none(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={"P": [10], "AB": [111]},
        decode_map={10: "P", 111: "AB"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={"P": [301], "AB": [201, 202]},
        decode_map={301: "P", 201: "A", 202: "B"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    args = _make_args(opd_top_logprobs_num=2)
    sample = SimpleNamespace(tokens=[10, 111], response_length=1, metadata={}, label=None)
    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.2, 901, "AB"]],
                [[-0.8, 902, "B"]],
            ],
            "input_token_logprobs": [
                [-0.3, 201, "A"],
                [None, 202, "B"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201, 202],
        "_opd_teacher_logprob_start_len": 1,
    }

    topk_logprobs, topk_token_ids, group_lengths, group_valid_mask = reward_plugin._build_cross_tokenizer_teacher_targets(
        args,
        sample,
        reward,
        topk=2,
    )

    assert group_lengths == [1]
    assert group_valid_mask == [0]
    assert topk_token_ids == [[TOPK_PAD_TOKEN_ID, TOPK_PAD_TOKEN_ID]]
    assert topk_logprobs == [[TOPK_PAD_LOGPROB, TOPK_PAD_LOGPROB]]


def test_cross_tokenizer_build_targets_populates_timing_stats(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={"P": [10], "A": [111]},
        decode_map={10: "P", 111: "A"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={"P": [301], "A": [201]},
        decode_map={301: "P", 201: "A"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    args = _make_args(opd_top_logprobs_num=1)
    sample = SimpleNamespace(tokens=[10, 111], response_length=1, metadata={}, label=None)
    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.2, 901, "A"]],
            ],
            "input_token_logprobs": [
                [-0.3, 201, "A"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201],
        "_opd_teacher_logprob_start_len": 1,
    }

    timing_stats: dict[str, float] = {}
    reward_plugin._build_cross_tokenizer_teacher_targets(
        args,
        sample,
        reward,
        topk=1,
        timing_stats=timing_stats,
    )

    expected_keys = {
        "extract_rows_time_s",
        "build_alignment_groups_time_s",
        "project_teacher_support_time_s",
        "total_time_s",
        "fallback_used",
    }
    assert expected_keys.issubset(timing_stats.keys())
    for key in ("extract_rows_time_s", "build_alignment_groups_time_s", "project_teacher_support_time_s", "total_time_s"):
        assert math.isfinite(timing_stats[key])
        assert timing_stats[key] >= 0.0
    assert timing_stats["fallback_used"] in {0.0, 1.0}


def test_post_process_rewards_topk_returns_alignment_metrics_in_cross_mode(monkeypatch) -> None:
    student_tok = _DummyTokenizer(
        encode_map={"A": [111]},
        decode_map={10: "P", 111: "A"},
    )
    teacher_tok = _DummyTokenizer(
        encode_map={"P": [301], "A": [201]},
        decode_map={301: "P", 201: "A"},
    )

    monkeypatch.setattr(
        reward_plugin,
        "_get_tokenizer",
        lambda path: student_tok if path == "student" else teacher_tok,
    )

    reward = {
        "meta_info": {
            "input_top_logprobs": [
                [[-0.1, 901, "A"]],
            ],
            "input_token_logprobs": [
                [-0.3, 201, "A"],
            ],
        },
        "_opd_teacher_input_ids": [301, 201],
        "_opd_teacher_logprob_start_len": 1,
    }
    sample = SimpleNamespace(
        tokens=[10, 111],
        response_length=1,
        metadata={},
        label=None,
        teacher_topk_logprobs=None,
        teacher_topk_token_ids=None,
        teacher_topk_group_lengths=None,
        teacher_topk_group_valid_mask=None,
        teacher_input_ids=None,
        teacher_logprob_start_len=None,
    )
    sample.get_reward_value = lambda _args: reward

    args = _make_args(opd_top_logprobs_num=1)
    raw_rewards, rewards, metrics = reward_plugin.post_process_rewards_topk(args, [sample])

    assert raw_rewards == [0.0]
    assert rewards == [0.0]
    assert metrics["perf/opd_alignment/sample_count"] == pytest.approx(1.0, abs=1e-6)
    required_metric_keys = [
        "perf/opd_alignment/total_time_s/mean",
        "perf/opd_alignment/total_time_s/median",
        "perf/opd_alignment/total_time_s/max",
        "perf/opd_alignment/total_time_s/min",
        "perf/opd_alignment/build_groups_time_s/mean",
        "perf/opd_alignment/build_groups_time_s/median",
        "perf/opd_alignment/build_groups_time_s/max",
        "perf/opd_alignment/build_groups_time_s/min",
        "perf/opd_alignment/project_support_time_s/mean",
        "perf/opd_alignment/project_support_time_s/median",
        "perf/opd_alignment/project_support_time_s/max",
        "perf/opd_alignment/project_support_time_s/min",
        "perf/opd_alignment/fallback_ratio",
    ]
    for key in required_metric_keys:
        assert key in metrics
        assert math.isfinite(metrics[key])
        assert metrics[key] >= 0.0
    assert metrics["perf/opd_alignment/fallback_ratio"] <= 1.0
