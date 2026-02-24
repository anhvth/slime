from __future__ import annotations

import sys
from pathlib import Path

import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from opd_topk_loss_plugin import compute_topk_renormalized_jsd


def test_jsd_symmetry_at_beta_half() -> None:
    teacher = torch.tensor([[-0.2, -1.0, -2.3], [-3.0, -0.3, -1.2]], dtype=torch.float32)
    student = torch.tensor([[-0.6, -1.2, -1.1], [-1.1, -0.5, -2.4]], dtype=torch.float32)

    jsd_ts = compute_topk_renormalized_jsd(teacher, student, beta=0.5)
    jsd_st = compute_topk_renormalized_jsd(student, teacher, beta=0.5)

    assert torch.allclose(jsd_ts, jsd_st, atol=1e-6, rtol=1e-6)


def test_jsd_non_negative() -> None:
    teacher = torch.randn(8, 16, dtype=torch.float32)
    student = torch.randn(8, 16, dtype=torch.float32)

    jsd = compute_topk_renormalized_jsd(teacher, student, beta=0.5)
    assert torch.all(jsd >= -1e-6)


def test_jsd_zero_for_identical_distributions() -> None:
    teacher = torch.randn(6, 8, dtype=torch.float32)
    jsd = compute_topk_renormalized_jsd(teacher, teacher.clone(), beta=0.5)

    assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-6, rtol=1e-6)


def test_jsd_beta_endpoints_are_zero() -> None:
    teacher = torch.tensor([[-0.2, -1.0, -2.3], [-3.0, -0.3, -1.2]], dtype=torch.float32)
    student = torch.tensor([[-0.6, -1.2, -1.1], [-1.1, -0.5, -2.4]], dtype=torch.float32)

    jsd_beta0 = compute_topk_renormalized_jsd(teacher, student, beta=0.0)
    jsd_beta1 = compute_topk_renormalized_jsd(teacher, student, beta=1.0)

    assert torch.allclose(jsd_beta0, torch.zeros_like(jsd_beta0), atol=1e-6, rtol=1e-6)
    assert torch.allclose(jsd_beta1, torch.zeros_like(jsd_beta1), atol=1e-6, rtol=1e-6)


def test_jsd_finite_for_extreme_log_probs() -> None:
    teacher = torch.tensor(
        [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0]],
        dtype=torch.float32,
    )
    student = torch.tensor(
        [[-1000.0, 1000.0, -1000.0], [-1000.0, -1000.0, 1000.0]],
        dtype=torch.float32,
    )

    jsd = compute_topk_renormalized_jsd(teacher, student, beta=0.5)

    assert torch.all(torch.isfinite(jsd))
