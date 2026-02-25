from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.on_policy_distillation.on_policy_distillation as opd
from slime.utils.types import Sample


def test_rkl_post_process_sets_teacher_view_fields() -> None:
    args = Namespace(reward_key=None)
    sample = Sample(tokens=[10, 11, 12, 13], response_length=2)
    sample.reward = {
        "meta_info": {
            "input_token_logprobs": [
                [0.0, 10, None],
                [-0.2, 11, None],
                [-0.3, 12, None],
                [-0.4, 13, None],
            ]
        }
    }

    opd.post_process_rewards(args, [sample])

    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.3, -0.4], abs=1e-6)
    assert sample.teacher_input_ids == [10, 11, 12, 13]
    assert sample.teacher_logprob_start_len == 0


def test_sample_serialization_keeps_teacher_view_fields() -> None:
    sample = Sample(tokens=[1, 2, 3], response_length=1)
    sample.teacher_input_ids = [7, 8, 9]
    sample.teacher_logprob_start_len = 2

    loaded = Sample.from_dict(sample.to_dict())
    assert loaded.teacher_input_ids == [7, 8, 9]
    assert loaded.teacher_logprob_start_len == 2
