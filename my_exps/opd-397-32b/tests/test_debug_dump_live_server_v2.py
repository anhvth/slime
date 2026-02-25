from __future__ import annotations

import sys
from pathlib import Path

import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import debug_dump_live_server_v2 as viewer


def _make_topk_payload(record: dict) -> dict:
    return {
        "mode": "fkl",
        "recipe": {"distill_loss_mode": "fkl", "opd_jsd_beta": 0.5},
        "records": [record],
    }


def test_build_record_payload_cross_tokenizer_teacher_units() -> None:
    record = {
        "sample_index": 1,
        "microbatch_sample_index": 1,
        "student_input_ids": torch.tensor([11, 12, 21, 22, 23], dtype=torch.long),
        "teacher_input_ids": torch.tensor([101, 102, 103, 201, 202, 203, 204], dtype=torch.long),
        "response_start": 2,
        "response_length": 3,
        "teacher_logprob_start_len": 3,
        "position_indices": torch.tensor([0, 2], dtype=torch.long),
        "position_unit": "teacher_group_anchor",
        "cross_tokenizer": 1,
        "teacher_topk_group_lengths": torch.tensor([2, 1], dtype=torch.long),
        "teacher_topk_group_valid_mask": torch.tensor([1, 0], dtype=torch.long),
        "teacher_topk_logprobs": torch.tensor([[0.0, -1.0], [-0.2, -0.3]], dtype=torch.float32),
        "teacher_topk_token_ids": torch.tensor([[7, 8], [9, 10]], dtype=torch.long),
        "student_topk_logprobs": torch.tensor([[-0.1, -1.2], [-0.3, -0.4]], dtype=torch.float32),
        "forward_kl": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "reverse_kl": torch.tensor([0.3, 0.4], dtype=torch.float32),
        "jsd": torch.tensor([0.05, 0.06], dtype=torch.float32),
    }
    payload = _make_topk_payload(record)
    out = viewer._build_record_payload(
        run_dir=Path("/tmp"),
        payload=payload,
        record_idx=0,
        tokenizers={"student_path": "", "teacher_path": "", "student_error": "", "teacher_error": ""},
    )

    assert out["position_unit"] == "teacher_group_anchor"
    assert out["context"]["teacher_response"] == "201 202 203 204"
    assert out["teacher_units"] == [
        {"logged_idx": 0, "student_anchor_idx": 0, "student_span_len": 2, "student_span_end": 2, "group_valid": 1},
        {"logged_idx": 1, "student_anchor_idx": 2, "student_span_len": 1, "student_span_end": 3, "group_valid": 0},
    ]
    assert out["topk"]["token_id_space"] == "student"
    assert out["token_to_unit_map"] == {"0": 0, "1": 0, "2": 1}
    assert out["logged_position_map"] == {"0": 0, "1": 0, "2": 1}


def test_tokenizer_options_include_qwen_labels_and_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        viewer,
        "_discover_qwen_tokenizer_candidates",
        lambda: {"student": ["/models/qwen3"], "teacher": ["/models/qwen3.5"]},
    )
    payload = viewer._build_tokenizer_options_payload({"recipe": {}})

    student_opts = payload["student_options"]
    teacher_opts = payload["teacher_options"]
    assert student_opts[0]["id"] == viewer.STUDENT_QWEN3_PRESET_ID
    assert student_opts[0]["label"] == "Qwen3 (158k)"
    assert student_opts[0]["path"] == "/models/qwen3"
    assert teacher_opts[0]["id"] == viewer.TEACHER_QWEN35_PRESET_ID
    assert teacher_opts[0]["label"] == "Qwen3.5 (248k)"
    assert teacher_opts[0]["path"] == "/models/qwen3.5"
    assert student_opts[1]["id"] == viewer.TOKENIZER_CUSTOM_ID
    assert teacher_opts[1]["id"] == viewer.TOKENIZER_CUSTOM_ID


def test_tokenizer_options_defaults_recipe_path_matching_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        viewer,
        "_discover_qwen_tokenizer_candidates",
        lambda: {"student": ["/models/qwen3"], "teacher": ["/models/qwen3.5"]},
    )
    payload = viewer._build_tokenizer_options_payload(
        {
            "recipe": {
                "opd_student_tokenizer_path": "/models/qwen3",
                "opd_teacher_tokenizer_path": "/models/qwen3.5",
            }
        }
    )
    defaults = payload["defaults"]
    assert defaults["student_default_id"] == viewer.STUDENT_QWEN3_PRESET_ID
    assert defaults["teacher_default_id"] == viewer.TEACHER_QWEN35_PRESET_ID
    assert defaults["student_custom_path"] == ""
    assert defaults["teacher_custom_path"] == ""


def test_tokenizer_options_defaults_recipe_non_preset_uses_custom(monkeypatch) -> None:
    monkeypatch.setattr(
        viewer,
        "_discover_qwen_tokenizer_candidates",
        lambda: {"student": ["/models/qwen3"], "teacher": ["/models/qwen3.5"]},
    )
    payload = viewer._build_tokenizer_options_payload(
        {
            "recipe": {
                "opd_student_tokenizer_path": "/other/student/tokenizer",
                "opd_teacher_tokenizer_path": "/other/teacher/tokenizer",
            }
        }
    )
    defaults = payload["defaults"]
    assert defaults["student_default_id"] == viewer.TOKENIZER_CUSTOM_ID
    assert defaults["teacher_default_id"] == viewer.TOKENIZER_CUSTOM_ID
    assert defaults["student_custom_path"] == "/other/student/tokenizer"
    assert defaults["teacher_custom_path"] == "/other/teacher/tokenizer"


def test_tokenizer_options_unavailable_presets_still_listed(monkeypatch) -> None:
    monkeypatch.setattr(viewer, "_discover_qwen_tokenizer_candidates", lambda: {"student": [], "teacher": []})
    payload = viewer._build_tokenizer_options_payload({"recipe": {}})

    assert payload["student_options"][0]["id"] == viewer.STUDENT_QWEN3_PRESET_ID
    assert payload["teacher_options"][0]["id"] == viewer.TEACHER_QWEN35_PRESET_ID
    assert payload["student_options"][0]["available"] is False
    assert payload["teacher_options"][0]["available"] is False
    assert "unavailable" in payload["student_options"][0]["label"]
    assert "unavailable" in payload["teacher_options"][0]["label"]
    assert payload["defaults"]["student_default_id"] == viewer.STUDENT_QWEN3_PRESET_ID
    assert payload["defaults"]["teacher_default_id"] == viewer.TEACHER_QWEN35_PRESET_ID


def test_resolve_tokenizer_paths_prefers_override_then_defaults_then_recipe(monkeypatch, tmp_path: Path) -> None:
    accepted = {"student_override", "recipe_teacher"}

    def _fake_load(path: str):
        if path not in accepted:
            raise ValueError(f"bad tokenizer: {path}")
        return object()

    monkeypatch.setattr(viewer, "_load_tokenizer", _fake_load)
    monkeypatch.setattr(viewer, "_discover_qwen_tokenizer_candidates", lambda: {"student": [], "teacher": []})

    old_default = viewer.STATE.default_tokenizer_path
    old_student = viewer.STATE.default_student_tokenizer_path
    old_teacher = viewer.STATE.default_teacher_tokenizer_path
    try:
        viewer.STATE.default_tokenizer_path = "shared_default"
        viewer.STATE.default_student_tokenizer_path = "student_default"
        viewer.STATE.default_teacher_tokenizer_path = "teacher_default"

        tokenizers = viewer._resolve_tokenizer_paths(
            run_dir=tmp_path,
            payload={
                "recipe": {
                    "opd_student_tokenizer_path": "recipe_student",
                    "opd_teacher_tokenizer_path": "recipe_teacher",
                }
            },
            legacy_requested="",
            student_requested="student_override",
            teacher_requested="",
        )
    finally:
        viewer.STATE.default_tokenizer_path = old_default
        viewer.STATE.default_student_tokenizer_path = old_student
        viewer.STATE.default_teacher_tokenizer_path = old_teacher

    assert tokenizers["student_path"] == "student_override"
    assert tokenizers["teacher_path"] == "recipe_teacher"
    assert tokenizers["student_error"] == ""
    assert tokenizers["teacher_error"] == ""


def test_resolve_tokenizer_paths_uses_qwen_auto_defaults_when_unset(monkeypatch, tmp_path: Path) -> None:
    accepted = {"auto_student_qwen3", "auto_teacher_qwen35"}

    def _fake_load(path: str):
        if path not in accepted:
            raise ValueError(f"bad tokenizer: {path}")
        return object()

    monkeypatch.setattr(viewer, "_load_tokenizer", _fake_load)
    monkeypatch.setattr(
        viewer,
        "_discover_qwen_tokenizer_candidates",
        lambda: {"student": ["auto_student_qwen3"], "teacher": ["auto_teacher_qwen35"]},
    )

    old_default = viewer.STATE.default_tokenizer_path
    old_student = viewer.STATE.default_student_tokenizer_path
    old_teacher = viewer.STATE.default_teacher_tokenizer_path
    try:
        viewer.STATE.default_tokenizer_path = ""
        viewer.STATE.default_student_tokenizer_path = ""
        viewer.STATE.default_teacher_tokenizer_path = ""

        tokenizers = viewer._resolve_tokenizer_paths(
            run_dir=tmp_path,
            payload={"recipe": {}},
            legacy_requested="",
            student_requested="",
            teacher_requested="",
        )
    finally:
        viewer.STATE.default_tokenizer_path = old_default
        viewer.STATE.default_student_tokenizer_path = old_student
        viewer.STATE.default_teacher_tokenizer_path = old_teacher

    assert tokenizers["student_path"] == "auto_student_qwen3"
    assert tokenizers["teacher_path"] == "auto_teacher_qwen35"
    assert tokenizers["student_error"] == ""
    assert tokenizers["teacher_error"] == ""


def test_resolve_tokenizer_paths_prefers_effective_recipe_fields(monkeypatch, tmp_path: Path) -> None:
    accepted = {"effective_student", "effective_teacher"}

    def _fake_load(path: str):
        if path not in accepted:
            raise ValueError(f"bad tokenizer: {path}")
        return _FakeTokenizer(400000)

    monkeypatch.setattr(viewer, "_load_tokenizer", _fake_load)
    monkeypatch.setattr(viewer, "_discover_qwen_tokenizer_candidates", lambda: {"student": [], "teacher": []})

    old_default = viewer.STATE.default_tokenizer_path
    old_student = viewer.STATE.default_student_tokenizer_path
    old_teacher = viewer.STATE.default_teacher_tokenizer_path
    try:
        viewer.STATE.default_tokenizer_path = ""
        viewer.STATE.default_student_tokenizer_path = ""
        viewer.STATE.default_teacher_tokenizer_path = ""

        tokenizers = viewer._resolve_tokenizer_paths(
            run_dir=tmp_path,
            payload={
                "recipe": {
                    "opd_student_tokenizer_path": "legacy_student",
                    "opd_teacher_tokenizer_path": "legacy_teacher",
                    "opd_student_tokenizer_path_effective": "effective_student",
                    "opd_teacher_tokenizer_path_effective": "effective_teacher",
                },
                "records": [],
            },
            legacy_requested="",
            student_requested="",
            teacher_requested="",
        )
    finally:
        viewer.STATE.default_tokenizer_path = old_default
        viewer.STATE.default_student_tokenizer_path = old_student
        viewer.STATE.default_teacher_tokenizer_path = old_teacher

    assert tokenizers["student_path"] == "effective_student"
    assert tokenizers["teacher_path"] == "effective_teacher"


def test_build_record_payload_legacy_dump_defaults_to_student_token_mode() -> None:
    record = {
        "sample_index": 0,
        "microbatch_sample_index": 0,
        "student_input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "teacher_input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "response_start": 1,
        "response_length": 3,
        "teacher_logprob_start_len": 1,
        "position_indices": torch.tensor([1], dtype=torch.long),
        "teacher_topk_logprobs": torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        "teacher_topk_token_ids": torch.tensor([[3, 4]], dtype=torch.long),
        "student_topk_logprobs": torch.tensor([[-0.1, -1.2]], dtype=torch.float32),
        "forward_kl": torch.tensor([0.1], dtype=torch.float32),
        "reverse_kl": torch.tensor([0.2], dtype=torch.float32),
        "jsd": torch.tensor([0.05], dtype=torch.float32),
    }
    out = viewer._build_record_payload(
        run_dir=Path("/tmp"),
        payload=_make_topk_payload(record),
        record_idx=0,
        tokenizers={"student_path": "", "teacher_path": "", "student_error": "", "teacher_error": ""},
    )
    assert out["position_unit"] == "student_token"
    assert out["teacher_units"] == []
    assert out["topk"]["token_id_space"] == "shared"
    assert out["logged_position_map"]["1"] == 0


class _FakeTokenizer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def __len__(self) -> int:
        return self.capacity

    def decode(self, ids, skip_special_tokens=False):  # noqa: ANN001
        del skip_special_tokens
        return "|".join(str(int(x)) for x in ids)


def test_resolve_tokenizer_paths_legacy_dump_selects_capacity_compatible_student(monkeypatch, tmp_path: Path) -> None:
    tokenizers = {
        "small_student": _FakeTokenizer(151936),
        "large_student": _FakeTokenizer(248320),
        "teacher_tok": _FakeTokenizer(248320),
    }

    def _fake_load(path: str):
        tok = tokenizers.get(path)
        if tok is None:
            raise ValueError(f"unknown tokenizer: {path}")
        return tok

    monkeypatch.setattr(viewer, "_load_tokenizer", _fake_load)
    monkeypatch.setattr(
        viewer,
        "_discover_qwen_tokenizer_candidates",
        lambda: {"student": ["small_student", "large_student"], "teacher": ["teacher_tok"]},
    )

    old_default = viewer.STATE.default_tokenizer_path
    old_student = viewer.STATE.default_student_tokenizer_path
    old_teacher = viewer.STATE.default_teacher_tokenizer_path
    try:
        viewer.STATE.default_tokenizer_path = ""
        viewer.STATE.default_student_tokenizer_path = ""
        viewer.STATE.default_teacher_tokenizer_path = ""
        resolved = viewer._resolve_tokenizer_paths(
            run_dir=tmp_path,
            payload={
                "recipe": {"opd_student_tokenizer_path": "small_student"},
                "records": [
                    {
                        "student_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                        "teacher_input_ids": torch.tensor([10, 11], dtype=torch.long),
                        "teacher_topk_token_ids": torch.tensor([[248319, 12]], dtype=torch.long),
                    }
                ],
            },
            legacy_requested="",
            student_requested="",
            teacher_requested="",
        )
    finally:
        viewer.STATE.default_tokenizer_path = old_default
        viewer.STATE.default_student_tokenizer_path = old_student
        viewer.STATE.default_teacher_tokenizer_path = old_teacher

    assert resolved["student_path"] == "large_student"
    assert resolved["teacher_path"] == "teacher_tok"
    assert resolved["student_required_max_token_id"] == 248319
    assert resolved["student_capacity"] == 248320
    assert resolved["heuristic_backfill_used"] is True
    assert "Capacity fallback used" in resolved["student_selection_warning"]
