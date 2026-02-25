import concurrent.futures
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import ray

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
for _module_name in list(sys.modules):
    if _module_name == "slime" or _module_name.startswith("slime."):
        del sys.modules[_module_name]

_TRAIN_ASYNC_PATH = _REPO_ROOT / "train_async.py"
_TRAIN_ASYNC_SPEC = importlib.util.spec_from_file_location("slime_train_async_test_module", _TRAIN_ASYNC_PATH)
assert _TRAIN_ASYNC_SPEC is not None and _TRAIN_ASYNC_SPEC.loader is not None
train_async = importlib.util.module_from_spec(_TRAIN_ASYNC_SPEC)
_TRAIN_ASYNC_SPEC.loader.exec_module(train_async)


class _RemoteRecorder:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []

    def remote(self, *args):
        token = f"{self.prefix}-{len(self.calls)}"
        self.calls.append(args)
        return token


class _DummyRolloutManager:
    def __init__(self):
        self.generate = _RemoteRecorder("generate")
        self.recover_rollout_engines = _RemoteRecorder("recover")


def _cancelled_ray_task_error():
    return ray.exceptions.RayTaskError(
        "RolloutManager.generate",
        "traceback",
        concurrent.futures.CancelledError(),
    )


def test_get_rollout_data_with_retry_retries_cancelled_and_recovers(monkeypatch):
    args = SimpleNamespace(
        async_rollout_cancel_retry_times=3,
        async_rollout_cancel_retry_backoff_base_seconds=1.0,
        async_rollout_cancel_recover_engines=True,
    )
    rollout_manager = _DummyRolloutManager()
    first_future = rollout_manager.generate.remote(7)
    sleeps = []

    def fake_ray_get(ref):
        if ref == "generate-0":
            raise _cancelled_ray_task_error()
        if ref == "generate-1":
            return {"ok": True}
        if ref.startswith("recover-"):
            return None
        raise AssertionError(f"Unexpected ref: {ref}")

    monkeypatch.setattr(train_async.ray, "get", fake_ray_get)
    monkeypatch.setattr(train_async.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = train_async._get_rollout_data_with_retry(args, rollout_manager, first_future, rollout_id=7)

    assert result == {"ok": True}
    assert rollout_manager.generate.calls == [(7,), (7,)]
    assert len(rollout_manager.recover_rollout_engines.calls) == 1
    assert sleeps == [1.0]


def test_get_rollout_data_with_retry_does_not_retry_non_cancelled(monkeypatch):
    args = SimpleNamespace(
        async_rollout_cancel_retry_times=3,
        async_rollout_cancel_retry_backoff_base_seconds=1.0,
        async_rollout_cancel_recover_engines=True,
    )
    rollout_manager = _DummyRolloutManager()
    first_future = rollout_manager.generate.remote(3)

    monkeypatch.setattr(train_async.ray, "get", lambda _ref: (_ for _ in ()).throw(ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        train_async._get_rollout_data_with_retry(args, rollout_manager, first_future, rollout_id=3)

    # Only the initial generate call should exist; no retry generate call should be submitted.
    assert rollout_manager.generate.calls == [(3,)]
    assert len(rollout_manager.recover_rollout_engines.calls) == 0


def test_get_rollout_data_with_retry_raises_after_retry_exhaustion(monkeypatch):
    args = SimpleNamespace(
        async_rollout_cancel_retry_times=2,
        async_rollout_cancel_retry_backoff_base_seconds=0.5,
        async_rollout_cancel_recover_engines=False,
    )
    rollout_manager = _DummyRolloutManager()
    first_future = rollout_manager.generate.remote(11)
    sleeps = []

    def fake_ray_get(ref):
        if ref.startswith("generate-"):
            raise _cancelled_ray_task_error()
        raise AssertionError(f"Unexpected ref: {ref}")

    monkeypatch.setattr(train_async.ray, "get", fake_ray_get)
    monkeypatch.setattr(train_async.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RuntimeError, match="rollout_id=11 failed after 3 attempts"):
        train_async._get_rollout_data_with_retry(args, rollout_manager, first_future, rollout_id=11)

    assert rollout_manager.generate.calls == [(11,), (11,), (11,)]
    assert len(rollout_manager.recover_rollout_engines.calls) == 0
    assert sleeps == [0.5, 1.0]
