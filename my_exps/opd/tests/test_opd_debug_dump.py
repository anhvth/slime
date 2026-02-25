from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import opd_debug_dump as debug_dump


def _make_args(tmp_path: Path, rollout_id: int) -> Namespace:
    return Namespace(
        save=str(tmp_path),
        seed=1234,
        opd_debug_rollout_id=rollout_id,
        opd_debug_dump_enable=1,
        opd_debug_dump_dir=str(tmp_path),
        opd_debug_dump_max_total_mb=5120,
        opd_debug_dump_max_files=20000,
        opd_debug_dump_max_file_mb=64,
        opd_debug_dump_max_samples_per_update=2,
        opd_debug_dump_max_positions_per_sample=3,
        opd_debug_dump_seed=777,
        opd_distill_coef=1.0,
        opd_mixed_kl_weight=0.5,
        opd_jsd_beta=0.5,
        opd_kl_coef=1.0,
        opd_cross_tokenizer_enable=0,
        opd_teacher_tokenizer_path="",
        opd_student_tokenizer_path="",
        hf_checkpoint="",
    )


def _make_topk_records(num_samples: int, response_len: int = 8, topk: int = 4) -> list[dict]:
    records = []
    for i in range(num_samples):
        student_ids = torch.arange(100 + i * 20, 100 + i * 20 + 4 + response_len, dtype=torch.long)
        teacher_ids = torch.arange(500 + i * 30, 500 + i * 30 + 4 + response_len, dtype=torch.long)
        token_ids = torch.arange(response_len * topk, dtype=torch.long).reshape(response_len, topk) + i * 100
        records.append(
            {
                "sample_index": i,
                "microbatch_sample_index": i,
                "student_input_ids": student_ids,
                "response_start": 4,
                "response_length": response_len,
                "teacher_input_ids": teacher_ids,
                "teacher_logprob_start_len": 4,
                "teacher_topk_logprobs": torch.randn(response_len, topk, dtype=torch.float32),
                "teacher_topk_token_ids": token_ids,
                "student_topk_logprobs": torch.randn(response_len, topk, dtype=torch.float32),
                "forward_kl": torch.randn(response_len, dtype=torch.float32),
                "reverse_kl": torch.randn(response_len, dtype=torch.float32),
                "jsd": torch.randn(response_len, dtype=torch.float32),
            }
        )
    return records


def test_writer_gate_suppresses_dump(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: False)
    args = _make_args(tmp_path, rollout_id=1)

    result = debug_dump.dump_topk_debug_update(args, mode="fkl", records=_make_topk_records(1))
    assert result is None
    assert list(tmp_path.glob("distill_debug_*.pt")) == []


def test_topk_subsample_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: True)

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    args_a = _make_args(dir_a, rollout_id=7)
    args_b = _make_args(dir_b, rollout_id=7)
    records = _make_topk_records(6, response_len=12, topk=5)

    path_a = debug_dump.dump_topk_debug_update(args_a, mode="jsd", records=records)
    assert path_a is not None

    debug_dump.clear_debug_dump_cache()
    path_b = debug_dump.dump_topk_debug_update(args_b, mode="jsd", records=records)
    assert path_b is not None

    payload_a = torch.load(path_a, weights_only=False)
    payload_b = torch.load(path_b, weights_only=False)
    kept_a = [(r["sample_index"], r["position_indices"].tolist()) for r in payload_a["records"]]
    kept_b = [(r["sample_index"], r["position_indices"].tolist()) for r in payload_b["records"]]
    assert kept_a == kept_b


def test_topk_logging_caps_samples_and_keeps_all_positions(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: True)

    args = _make_args(tmp_path, rollout_id=9)
    args.opd_debug_dump_max_samples_per_update = 99
    args.opd_debug_dump_max_positions_per_sample = 0
    records = _make_topk_records(20, response_len=12, topk=5)

    path = debug_dump.dump_topk_debug_update(args, mode="fkl", records=records)
    assert path is not None

    payload = torch.load(path, weights_only=False)
    assert payload["num_records_kept"] == 8
    assert payload["limits"]["max_samples_per_update"] == 8
    assert payload["limits"]["max_positions_per_sample"] == 0
    for record in payload["records"]:
        assert record["position_indices"].numel() == record["response_length"]


def test_retention_keeps_recent_files(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: True)

    args = _make_args(tmp_path, rollout_id=1)
    args.opd_debug_dump_max_files = 2
    args.opd_debug_dump_max_total_mb = 5120
    records = _make_topk_records(2)

    for rollout_id in (1, 2, 3):
        args.opd_debug_rollout_id = rollout_id
        path = debug_dump.dump_topk_debug_update(args, mode="mixed", records=records)
        assert path is not None

    files = sorted(tmp_path.glob("distill_debug_*.pt"))
    assert len(files) == 2
    names = [p.name for p in files]
    assert any("rollout_0000002" in n for n in names)
    assert any("rollout_0000003" in n for n in names)


def test_rkl_dump_contains_recomputable_reverse_kl(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: True)
    args = _make_args(tmp_path, rollout_id=42)
    args.opd_debug_dump_max_positions_per_sample = 10

    rollout_data = {
        "tokens": [torch.tensor([7, 8, 9, 10, 11], dtype=torch.long)],
        "response_lengths": [3],
        "sample_indices": [13],
        "teacher_input_ids": [[101, 102, 103, 10, 11]],
        "teacher_logprob_start_len": [2],
    }
    student = [torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)]
    teacher = [torch.tensor([0.1, -0.3, 0.0], dtype=torch.float32)]
    reverse = [student[0] - teacher[0]]

    path = debug_dump.dump_rkl_debug_update(
        args,
        rollout_data=rollout_data,
        student_log_probs=student,
        teacher_log_probs=teacher,
        reverse_kls=reverse,
    )
    assert path is not None

    payload = torch.load(path, weights_only=False)
    assert payload["mode"] == "rkl"
    record = payload["records"][0]
    assert record["sample_index"] == 13
    assert record["teacher_logprob_start_len"] == 2
    assert record["teacher_input_ids"].tolist() == [101, 102, 103, 10, 11]
    recomputed = record["student_log_probs"] - record["teacher_log_probs"]
    assert torch.allclose(recomputed, record["reverse_kl"], atol=1e-6, rtol=1e-6)


def test_cross_tokenizer_topk_dump_uses_teacher_anchor_positions(monkeypatch, tmp_path: Path) -> None:
    debug_dump.clear_debug_dump_cache()
    monkeypatch.setattr(debug_dump, "_is_global_writer_rank", lambda: True)
    args = _make_args(tmp_path, rollout_id=9)
    args.opd_debug_dump_max_positions_per_sample = 0
    args.opd_cross_tokenizer_enable = 1
    args.opd_teacher_tokenizer_path = "/teacher-tokenizer"
    args.opd_student_tokenizer_path = "/student-tokenizer"

    records = _make_topk_records(1, response_len=6, topk=3)
    records[0]["cross_tokenizer"] = 1
    records[0]["teacher_topk_group_lengths"] = torch.tensor([2, 0, 1, 0, 2, 0], dtype=torch.long)
    records[0]["teacher_topk_group_valid_mask"] = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.long)

    path = debug_dump.dump_topk_debug_update(args, mode="fkl", records=records)
    assert path is not None

    payload = torch.load(path, weights_only=False)
    assert payload["recipe"]["opd_cross_tokenizer_enable"] == 1
    assert payload["recipe"]["opd_teacher_tokenizer_path"] == "/teacher-tokenizer"
    assert payload["recipe"]["opd_student_tokenizer_path"] == "/student-tokenizer"
    assert payload["recipe"]["opd_teacher_tokenizer_path_effective"] == "/teacher-tokenizer"
    assert payload["recipe"]["opd_student_tokenizer_path_effective"] == "/student-tokenizer"

    record = payload["records"][0]
    assert record["cross_tokenizer"] == 1
    assert record["position_unit"] == "teacher_group_anchor"
    assert record["position_indices"].tolist() == [0, 2, 4]
    assert record["teacher_topk_group_lengths"].tolist() == [2, 1, 2]
    assert record["teacher_topk_group_valid_mask"].tolist() == [1, 1, 0]


def test_recipe_effective_tokenizer_paths_fallback_to_hf_checkpoint(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_id=1)
    args.hf_checkpoint = "/models/qwen3-native"
    args.opd_student_tokenizer_path = ""
    args.opd_teacher_tokenizer_path = ""

    recipe = debug_dump._build_recipe(args, mode="fkl")
    assert recipe["hf_checkpoint"] == "/models/qwen3-native"
    assert recipe["opd_student_tokenizer_path_effective"] == "/models/qwen3-native"
    assert recipe["opd_teacher_tokenizer_path_effective"] == "/models/qwen3-native"
