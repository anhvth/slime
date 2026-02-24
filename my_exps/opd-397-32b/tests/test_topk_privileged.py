from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import opd_topk_reward_plugin as reward_plugin


def _make_args(**kwargs) -> Namespace:
    defaults = {
        "opd_privileged_enable": 1,
        "opd_privileged_metadata_key": "privileged_context",
        "opd_privileged_fallback_label": 1,
        "opd_privileged_tokenizer_path": "",
        "hf_checkpoint": "/tmp/fake_hf_path",
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _make_sample(*, metadata=None, label=None) -> SimpleNamespace:
    return SimpleNamespace(tokens=[101, 102, 201, 202], response_length=2, metadata=metadata or {}, label=label)


def test_privileged_metadata_takes_precedence_over_label(monkeypatch) -> None:
    captured = {}

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            captured["messages"] = messages
            assert tokenize is False
            assert add_generation_prompt is False
            return "<|im_start|>system\nMETA_INFO<|im_end|>"

        def encode(self, text, add_special_tokens=False):
            captured["text"] = text
            assert add_special_tokens is False
            return [900, 901]

    def fake_get_tokenizer(path: str):
        captured["path"] = path
        return DummyTokenizer()

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", fake_get_tokenizer)

    args = _make_args()
    sample = _make_sample(metadata={"privileged_context": "META_INFO"}, label="LABEL_FALLBACK")

    input_ids, start_len = reward_plugin._build_teacher_input_ids_and_start_len(args, sample)

    assert captured["path"] == "/tmp/fake_hf_path"
    assert captured["messages"] == [{"role": "system", "content": "META_INFO"}]
    assert input_ids == [101, 102, 900, 901, 201, 202]
    assert start_len == 4


def test_fallback_to_label_when_metadata_missing(monkeypatch) -> None:
    captured = {}

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            captured["messages"] = messages
            return "<|im_start|>system\nLABEL_ONLY<|im_end|>"

        def encode(self, text, add_special_tokens=False):
            captured["text"] = text
            return [910]

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", lambda _: DummyTokenizer())

    args = _make_args()
    sample = _make_sample(metadata={}, label="LABEL_ONLY")

    input_ids, start_len = reward_plugin._build_teacher_input_ids_and_start_len(args, sample)

    assert captured["messages"] == [{"role": "system", "content": "LABEL_ONLY"}]
    assert input_ids == [101, 102, 910, 201, 202]
    assert start_len == 3


def test_prompt_only_fallback_when_no_privileged_context(monkeypatch) -> None:
    def fail_get_tokenizer(_path: str):
        raise AssertionError("Tokenizer should not be loaded when no privileged context is available.")

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", fail_get_tokenizer)

    args = _make_args()
    sample = _make_sample(metadata={}, label=None)

    input_ids, start_len = reward_plugin._build_teacher_input_ids_and_start_len(args, sample)

    assert input_ids == [101, 102, 201, 202]
    assert start_len == 2


def test_privileged_disabled_uses_prompt_only(monkeypatch) -> None:
    def fail_get_tokenizer(_path: str):
        raise AssertionError("Tokenizer should not be loaded when privileged mode is disabled.")

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", fail_get_tokenizer)

    args = _make_args(opd_privileged_enable=0)
    sample = _make_sample(metadata={"privileged_context": "META_INFO"}, label="LABEL")

    input_ids, start_len = reward_plugin._build_teacher_input_ids_and_start_len(args, sample)

    assert input_ids == [101, 102, 201, 202]
    assert start_len == 2


def test_payload_composition_with_privileged_context(monkeypatch) -> None:
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "<|im_start|>system\nCTX<|im_end|>"

        def encode(self, text, add_special_tokens=False):
            return [333, 334, 335]

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", lambda _: DummyTokenizer())

    args = _make_args()
    sample = _make_sample(metadata={"privileged_context": "CTX"}, label=None)

    payload = reward_plugin._build_teacher_payload(args, sample, topk=8)

    assert payload["top_logprobs_num"] == 8
    assert payload["logprob_start_len"] == 5
    assert payload["input_ids"] == [101, 102, 333, 334, 335, 201, 202]
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert payload["return_logprob"] is True


def test_payload_composition_without_privileged_context(monkeypatch) -> None:
    def fail_get_tokenizer(_path: str):
        raise AssertionError("Tokenizer should not be loaded for prompt-only fallback.")

    monkeypatch.setattr(reward_plugin, "_get_tokenizer", fail_get_tokenizer)

    args = _make_args(opd_privileged_fallback_label=0)
    sample = _make_sample(metadata={}, label=None)

    payload = reward_plugin._build_teacher_payload(args, sample, topk=16)

    assert payload["top_logprobs_num"] == 16
    assert payload["logprob_start_len"] == 2
    assert payload["input_ids"] == [101, 102, 201, 202]
