from __future__ import annotations

import asyncio
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


def _make_args(**kwargs) -> Namespace:
    defaults = {
        "rm_url": "http://teacher.local/reward",
        "opd_rm_retry_attempts": 3,
        "opd_rm_retry_base_sleep_s": 0.001,
        "opd_rm_retry_max_sleep_s": 0.001,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class _DummyResponse:
    def __init__(self, *, payload: dict | None = None, json_exc: BaseException | None = None) -> None:
        self._payload = {} if payload is None else payload
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class _DummyRequestContext:
    def __init__(self, response: _DummyResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _DummyResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummySession:
    def __init__(self, responses: list[_DummyResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def post(self, _url, **_kwargs):
        if self.calls >= len(self._responses):
            raise AssertionError("Unexpected additional HTTP call.")
        response = self._responses[self.calls]
        self.calls += 1
        return _DummyRequestContext(response)


def test_reward_retries_on_connection_closed_runtime(monkeypatch) -> None:
    args = _make_args(opd_rm_retry_attempts=2)
    session = _DummySession(
        responses=[
            _DummyResponse(json_exc=RuntimeError("Connection closed.")),
            _DummyResponse(payload={"meta_info": {}}),
        ]
    )
    close_calls = 0

    async def fake_get_http_session(_args):
        return session

    async def fake_close_http_session():
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(reward_plugin, "_get_topk", lambda _args: 4)
    monkeypatch.setattr(
        reward_plugin,
        "_build_teacher_payload",
        lambda _args, _sample, _topk: {"input_ids": [11, 12], "logprob_start_len": 1},
    )
    monkeypatch.setattr(reward_plugin, "_get_http_session", fake_get_http_session)
    monkeypatch.setattr(reward_plugin, "_close_http_session", fake_close_http_session)

    result = asyncio.run(reward_plugin.reward_func_topk(args, SimpleNamespace()))

    assert session.calls == 2
    assert close_calls == 1
    assert result["meta_info"]["client_http_attempts"] == 2


def test_reward_does_not_retry_non_retryable_runtime(monkeypatch) -> None:
    args = _make_args(opd_rm_retry_attempts=3)
    session = _DummySession(responses=[_DummyResponse(json_exc=RuntimeError("boom"))])
    close_calls = 0

    async def fake_get_http_session(_args):
        return session

    async def fake_close_http_session():
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(reward_plugin, "_get_topk", lambda _args: 4)
    monkeypatch.setattr(
        reward_plugin,
        "_build_teacher_payload",
        lambda _args, _sample, _topk: {"input_ids": [11, 12], "logprob_start_len": 1},
    )
    monkeypatch.setattr(reward_plugin, "_get_http_session", fake_get_http_session)
    monkeypatch.setattr(reward_plugin, "_close_http_session", fake_close_http_session)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(reward_plugin.reward_func_topk(args, SimpleNamespace()))

    assert session.calls == 1
    assert close_calls == 0
