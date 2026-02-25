import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
for _module_name in list(sys.modules):
    if _module_name == "slime" or _module_name.startswith("slime."):
        del sys.modules[_module_name]

from slime.utils.arguments import get_slime_extra_args_provider


def _build_parser():
    parser = argparse.ArgumentParser()
    add_slime_arguments = get_slime_extra_args_provider()
    return add_slime_arguments(parser)


def test_async_rollout_cancel_args_defaults():
    parser = _build_parser()
    args, _ = parser.parse_known_args(["--rollout-batch-size", "1"])

    assert args.async_rollout_cancel_retry_times == 3
    assert args.async_rollout_cancel_retry_backoff_base_seconds == 1.0
    assert args.async_rollout_cancel_recover_engines is True


def test_async_rollout_cancel_args_can_override():
    parser = _build_parser()
    args, _ = parser.parse_known_args(
        [
            "--rollout-batch-size",
            "1",
            "--async-rollout-cancel-retry-times",
            "5",
            "--async-rollout-cancel-retry-backoff-base-seconds",
            "0.25",
            "--no-async-rollout-cancel-recover-engines",
        ]
    )

    assert args.async_rollout_cancel_retry_times == 5
    assert args.async_rollout_cancel_retry_backoff_base_seconds == 0.25
    assert args.async_rollout_cancel_recover_engines is False
