from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from opd_topk_loss_plugin import compute_cross_tokenizer_group_losses
import opd_topk_loss_plugin as loss_plugin


def test_cross_tokenizer_group_losses_applies_continuation_term() -> None:
    teacher_lp = torch.tensor([[-0.2, -1.2], [-3.0, -3.0]], dtype=torch.float32)
    student_anchor_lp = teacher_lp.clone()
    student_actual_lp = torch.tensor([-0.4, -0.7], dtype=torch.float32)
    group_lengths = torch.tensor([2, 0], dtype=torch.long)
    group_valid_mask = torch.tensor([1, 0], dtype=torch.long)

    res = compute_cross_tokenizer_group_losses(
        teacher_log_probs=teacher_lp,
        student_anchor_log_probs=student_anchor_lp,
        student_actual_log_probs=student_actual_lp,
        group_lengths=group_lengths,
        group_valid_mask=group_valid_mask,
        mode="fkl",
        mixed_weight=0.5,
        jsd_beta=0.5,
    )

    expected = 0.7 * torch.exp(torch.tensor(-0.2)).item() + 0.7 * torch.exp(torch.tensor(-1.2)).item()

    assert res["valid_groups"] == 1
    assert res["skipped_groups"] == 0
    assert res["mapped_support"] == 2
    assert res["total_support"] == 2
    assert res["forward"].numel() == 1
    assert res["forward"][0].item() == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_cross_tokenizer_group_losses_skips_invalid_groups() -> None:
    teacher_lp = torch.tensor(
        [
            [-0.2, -0.4],
            [-0.3, -0.5],
            [-0.1, -0.6],
        ],
        dtype=torch.float32,
    )
    student_anchor_lp = teacher_lp.clone()
    student_actual_lp = torch.tensor([-0.2, -0.2, -0.2], dtype=torch.float32)
    group_lengths = torch.tensor([1, 1, 1], dtype=torch.long)
    group_valid_mask = torch.tensor([1, 0, 1], dtype=torch.long)

    res = compute_cross_tokenizer_group_losses(
        teacher_log_probs=teacher_lp,
        student_anchor_log_probs=student_anchor_lp,
        student_actual_log_probs=student_actual_lp,
        group_lengths=group_lengths,
        group_valid_mask=group_valid_mask,
        mode="mixed",
        mixed_weight=0.5,
        jsd_beta=0.5,
    )

    assert res["valid_groups"] == 2
    assert res["skipped_groups"] == 1
    assert res["forward"].numel() == 2
    assert res["reverse"].numel() == 2
    assert res["mapped_support"] == 4
    assert res["total_support"] == 6


def test_cross_tokenizer_group_losses_rejects_oob_group_end() -> None:
    teacher_lp = torch.tensor([[-0.2, -0.4], [-0.3, -0.5], [-0.1, -0.6]], dtype=torch.float32)
    student_anchor_lp = teacher_lp.clone()
    student_actual_lp = torch.tensor([-0.2, -0.2, -0.2], dtype=torch.float32)
    group_lengths = torch.tensor([4, 0, 0], dtype=torch.long)
    group_valid_mask = torch.tensor([1, 0, 0], dtype=torch.long)

    res = compute_cross_tokenizer_group_losses(
        teacher_log_probs=teacher_lp,
        student_anchor_log_probs=student_anchor_lp,
        student_actual_log_probs=student_actual_lp,
        group_lengths=group_lengths,
        group_valid_mask=group_valid_mask,
        mode="jsd",
        mixed_weight=0.5,
        jsd_beta=0.5,
    )

    assert res["valid_groups"] == 0
    assert res["skipped_groups"] == 1
    assert res["forward"].numel() == 0
    assert res["reverse"].numel() == 0
    assert res["jsd"].numel() == 0


def test_distill_topk_cross_tokenizer_uses_per_sample_group_mean_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    args = Namespace(
        distill_loss_mode="fkl",
        opd_mixed_kl_weight=0.5,
        opd_jsd_beta=0.5,
        opd_distill_coef=1.0,
        opd_cross_tokenizer_enable=True,
    )

    batch = {
        "teacher_topk_logprobs": [
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((4, 2), dtype=torch.float32),
        ],
        "teacher_topk_token_ids": [
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros((4, 2), dtype=torch.long),
        ],
        "teacher_topk_group_lengths": [
            torch.tensor([1, 0], dtype=torch.long),
            torch.tensor([2, 0, 2, 0], dtype=torch.long),
        ],
        "teacher_topk_group_valid_mask": [
            torch.tensor([1, 0], dtype=torch.long),
            torch.tensor([1, 0, 1, 0], dtype=torch.long),
        ],
        "response_lengths": [2, 4],
        "total_lengths": [2, 4],
        "unconcat_tokens": [
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3, 4, 5, 6], dtype=torch.long),
        ],
    }
    logits = torch.zeros((6, 8), dtype=torch.float32)

    monkeypatch.setattr(loss_plugin.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_plugin,
        "get_responses",
        lambda *args, **kwargs: iter(
            [
                (torch.zeros((2, 8), dtype=torch.float32), None),
                (torch.zeros((4, 8), dtype=torch.float32), None),
            ]
        ),
    )
    monkeypatch.setattr(
        loss_plugin,
        "_gather_selected_logprobs_tp",
        lambda logits_chunk, token_ids: torch.zeros(token_ids.shape, dtype=torch.float32, device=logits_chunk.device),
    )

    fake_results = [
        {
            "forward": torch.tensor([1.0], dtype=torch.float32),
            "reverse": torch.tensor([10.0], dtype=torch.float32),
            "jsd": torch.tensor([100.0], dtype=torch.float32),
            "forward_debug": torch.zeros((2,), dtype=torch.float32),
            "reverse_debug": torch.zeros((2,), dtype=torch.float32),
            "jsd_debug": torch.zeros((2,), dtype=torch.float32),
            "valid_groups": 1,
            "skipped_groups": 0,
            "mapped_support": 2,
            "total_support": 2,
        },
        {
            "forward": torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
            "reverse": torch.tensor([20.0, 20.0, 20.0], dtype=torch.float32),
            "jsd": torch.tensor([200.0, 200.0, 200.0], dtype=torch.float32),
            "forward_debug": torch.zeros((4,), dtype=torch.float32),
            "reverse_debug": torch.zeros((4,), dtype=torch.float32),
            "jsd_debug": torch.zeros((4,), dtype=torch.float32),
            "valid_groups": 3,
            "skipped_groups": 0,
            "mapped_support": 6,
            "total_support": 6,
        },
    ]
    call_idx = {"value": 0}

    def fake_compute_cross_tokenizer_group_losses(**kwargs):
        idx = call_idx["value"]
        call_idx["value"] += 1
        return fake_results[idx]

    monkeypatch.setattr(
        loss_plugin,
        "compute_cross_tokenizer_group_losses",
        fake_compute_cross_tokenizer_group_losses,
    )
    monkeypatch.setattr(loss_plugin, "dump_topk_debug_update", None)

    loss, logs = loss_plugin.distill_topk_custom_loss(
        args=args,
        batch=batch,
        logits=logits,
        sum_of_sample_mean=lambda x: x.sum(),
    )

    # HF-style weighting for this setup:
    # sample 0 mean=1.0, sample 1 mean=2.0 => raw distill term should be 3.0 before global_batch scaling.
    assert loss.item() == pytest.approx(3.0, rel=1e-6, abs=1e-6)
    assert logs["distill_kl"].item() == pytest.approx(3.0, rel=1e-6, abs=1e-6)
    assert logs["distill_kl_forward"].item() == pytest.approx(3.0, rel=1e-6, abs=1e-6)
    assert logs["distill_kl_reverse"].item() == pytest.approx(30.0, rel=1e-6, abs=1e-6)
