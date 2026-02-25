from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_post_process_rewards_contract_via_python_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        from slime.ray.rollout import RolloutManager

        cls = RolloutManager.__ray_metadata__.modified_class
        impl = cls._post_process_rewards.__wrapped__

        class DummySample:
            def __init__(self, reward):
                self._reward = reward

            def get_reward_value(self, args):
                return self._reward

        def make_manager(custom_func):
            manager = cls.__new__(cls)
            manager.args = SimpleNamespace(
                advantage_estimator="ppo",
                rewards_normalization=False,
                n_samples_per_prompt=1,
                rollout_batch_size=1,
                grpo_std_normalization=False,
            )
            manager.custom_reward_post_process_func = custom_func
            manager._latest_custom_reward_post_process_metrics = {}
            return manager

        manager_two = make_manager(lambda args, samples: ([1.0], [2.0]))
        raw, rewards = impl(manager_two, [DummySample(1.0)])
        assert raw == [1.0]
        assert rewards == [2.0]
        assert manager_two._latest_custom_reward_post_process_metrics == {}

        metrics = {"perf/opd_alignment/sample_count": 3.0}
        manager_three = make_manager(lambda args, samples: ([1.0], [2.0], metrics))
        raw, rewards = impl(manager_three, [DummySample(1.0)])
        assert raw == [1.0]
        assert rewards == [2.0]
        assert manager_three._latest_custom_reward_post_process_metrics == metrics

        manager_bad = make_manager(lambda args, samples: ([1.0], [2.0], ["bad"]))
        try:
            impl(manager_bad, [DummySample(1.0)])
        except ValueError as exc:
            assert "third return value must be a dict" in str(exc)
        else:
            raise AssertionError("Expected ValueError for non-dict metrics.")

        manager_default = make_manager(None)
        manager_default._latest_custom_reward_post_process_metrics = {"perf/x": 1.0}
        raw, rewards = impl(manager_default, [DummySample(0.5)])
        assert raw == [0.5]
        assert rewards == [0.5]
        assert manager_default._latest_custom_reward_post_process_metrics == {}
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
