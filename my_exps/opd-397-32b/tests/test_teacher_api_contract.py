import json
import os
from functools import lru_cache
from typing import Any
from urllib import request

import pytest
from transformers import AutoTokenizer


DEFAULT_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://worker-30:13141")
DEFAULT_TOKENIZER_PATH = os.environ.get(
    "TEACHER_TOKENIZER_PATH",
    "/home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned",
)


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any] | None]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=30) as resp:  # noqa: S310
        code = int(resp.getcode())
        body = resp.read()
    if not body:
        return code, None
    return code, json.loads(body.decode("utf-8"))


def _post_generate(base_url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    code, body = _http_json("POST", f"{base_url}/generate", payload=payload)
    assert body is not None
    return code, body


def _first_non_null(rows: Any) -> Any:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if row:
            return row
    return None


@lru_cache(maxsize=1)
def _prompt_ids() -> list[int]:
    tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH, trust_remote_code=True)
    messages = [{"role": "user", "content": "Return exactly one short word: blue"}]
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


@pytest.fixture(scope="module")
def sequence_case() -> dict[str, Any]:
    prompt_ids = _prompt_ids()
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 4,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 5,
    }
    code, body = _post_generate(DEFAULT_BASE_URL, payload)
    assert code == 200
    out_ids = body.get("output_ids", []) or []
    assert len(out_ids) > 0, "Expected non-empty output_ids to build response-span contract tests."
    return {
        "prompt_ids": prompt_ids,
        "output_ids": out_ids,
        "full_ids": list(prompt_ids) + list(out_ids),
        "response_len": len(out_ids),
    }


def test_health_endpoint() -> None:
    code, _ = _http_json("GET", f"{DEFAULT_BASE_URL}/health")
    assert code == 200


def test_top_logprobs_num_supported(sequence_case: dict[str, Any]) -> None:
    full_ids = sequence_case["full_ids"]
    prompt_len = len(sequence_case["prompt_ids"])

    for k in [0, 1, 8, 16, 32, 64]:
        payload = {
            "input_ids": full_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": prompt_len,
            "top_logprobs_num": k,
        }
        code, body = _post_generate(DEFAULT_BASE_URL, payload)
        assert code == 200
        meta = body.get("meta_info", {})
        if k == 0:
            assert meta.get("input_top_logprobs") is None
        else:
            assert isinstance(meta.get("input_top_logprobs"), list)


def test_response_span_slicing_with_logprob_start_len(sequence_case: dict[str, Any]) -> None:
    full_ids = sequence_case["full_ids"]
    prompt_len = len(sequence_case["prompt_ids"])
    response_len = sequence_case["response_len"]

    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": 5,
    }
    code, body = _post_generate(DEFAULT_BASE_URL, payload)
    assert code == 200
    meta = body.get("meta_info", {})

    assert len(meta.get("input_top_logprobs") or []) == response_len
    assert len(meta.get("input_token_logprobs") or []) == response_len
    assert int(meta.get("completion_tokens", -1)) == 0


def test_input_top_logprobs_entry_types(sequence_case: dict[str, Any]) -> None:
    full_ids = sequence_case["full_ids"]
    prompt_len = len(sequence_case["prompt_ids"])

    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": 16,
    }
    code, body = _post_generate(DEFAULT_BASE_URL, payload)
    assert code == 200
    rows = body.get("meta_info", {}).get("input_top_logprobs")
    top_entry = _first_non_null(rows)
    assert isinstance(top_entry, list) and len(top_entry) > 0

    first_row = top_entry[0]
    assert isinstance(first_row, list)
    assert len(first_row) >= 3
    assert isinstance(first_row[0], (int, float))
    assert isinstance(first_row[1], int)
    assert first_row[2] is None or isinstance(first_row[2], str)


def test_input_token_logprobs_entry_types(sequence_case: dict[str, Any]) -> None:
    full_ids = sequence_case["full_ids"]
    prompt_len = len(sequence_case["prompt_ids"])

    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": 16,
    }
    code, body = _post_generate(DEFAULT_BASE_URL, payload)
    assert code == 200
    rows = body.get("meta_info", {}).get("input_token_logprobs")
    entry = _first_non_null(rows)
    assert isinstance(entry, list)
    assert len(entry) >= 3
    assert entry[0] is None or isinstance(entry[0], (int, float))
    assert isinstance(entry[1], int)
    assert entry[2] is None or isinstance(entry[2], str)


@pytest.mark.parametrize("k", [1, 8, 16, 32, 64])
def test_topk_count_matches_requested_k(sequence_case: dict[str, Any], k: int) -> None:
    full_ids = sequence_case["full_ids"]
    prompt_len = len(sequence_case["prompt_ids"])

    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len,
        "top_logprobs_num": k,
    }
    code, body = _post_generate(DEFAULT_BASE_URL, payload)
    assert code == 200
    rows = body.get("meta_info", {}).get("input_top_logprobs")
    top_entry = _first_non_null(rows)
    assert isinstance(top_entry, list)
    assert len(top_entry) == k
