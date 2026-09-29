"""Tests for `sandboxai.evaluation` (the `sandboxai evaluate` CLI backend).

Nothing here needed a real Godot binary previously either -- this module
was simply never exercised by any test file at all (verified by grepping
the whole `python/tests` tree for `evaluate_model`/`evaluation` imports).
Fake serial and vectorized environments stand in for
`GodotGymEnv`/`GodotVecEnv`, mirroring the pattern used by
`test_godot_env.py` and `test_pipeline.py`.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sandboxai.evaluation import _mean, evaluate_model, format_summary


class _FakeTensor:
    """Mimics a torch tensor closely enough for evaluate_model's duck typing."""

    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _FakeSerialEnv:
    """Stands in for GodotGymEnv: fixed-length episodes, deterministic metrics."""

    instances: list["_FakeSerialEnv"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.reset_seeds: list[int] = []
        _FakeSerialEnv.instances.append(self)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._step = 0
        return [0.0] * 84, {}

    def step(self, action):
        self._step += 1
        done = self._step >= 3
        info = {
            "metrics": {
                "kills": 1 if done else 0,
                "win": 1.0 if done else 0.0,
                "accuracy": 0.5,
            }
        } if done else {}
        return [0.0] * 84, 1.0, done, False, info

    def close(self):
        self.closed = True


class _FakeVecEnv:
    """Stands in for GodotVecEnv: N sub-envs, staggered episode lengths."""

    instances: list["_FakeVecEnv"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.environment_count = kwargs["environment_count"]
        self.closed = False
        self._steps = [0] * self.environment_count
        _FakeVecEnv.instances.append(self)

    def reset(self):
        return [[0.0] * 84 for _ in range(self.environment_count)]

    def step(self, actions):
        observations = []
        rewards = []
        dones = []
        infos = []
        for index in range(self.environment_count):
            self._steps[index] += 1
            # env 0 finishes every 2 steps, env 1 (if any) every 3 steps.
            period = 2 if index == 0 else 3
            done = self._steps[index] % period == 0
            observations.append([0.0] * 84)
            rewards.append(1.0)
            dones.append(done)
            infos.append({"metrics": {"kills": 1 if done else 0, "win": 1.0}} if done else {})
        return observations, rewards, dones, infos

    def close(self):
        self.closed = True


class _FakeModel:
    def __init__(self, action=None, as_tuple=True, as_tensor=False):
        self.action = action if action is not None else [1, 1, 1, 1, 1, 0]
        self.as_tuple = as_tuple
        self.as_tensor = as_tensor
        self.predict_calls = 0

    def predict(self, observation, deterministic=True):
        self.predict_calls += 1
        assert deterministic is True
        action = _FakeTensor(self.action) if self.as_tensor else self.action
        if self.as_tuple:
            return action, None
        return action


class MeanTests(unittest.TestCase):
    def test_mean_of_empty_list_is_zero(self):
        self.assertEqual(_mean([]), 0.0)

    def test_mean_of_values(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0)


class EvaluateModelValidationTests(unittest.TestCase):
    def test_rejects_non_positive_episode_count(self):
        with self.assertRaises(ValueError):
            evaluate_model(_FakeModel(), {}, episodes=0)


class EvaluateModelSerialTests(unittest.TestCase):
    def setUp(self):
        _FakeSerialEnv.instances.clear()
        patcher = patch("sandboxai.evaluation.GodotGymEnv", _FakeSerialEnv)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runs_requested_episode_count_and_closes_env(self):
        model = _FakeModel()
        summary = evaluate_model(model, {}, episodes=4, seed=100)
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["environment_count"], 1)
        self.assertEqual(len(summary["episodes_detail"]), 4)
        self.assertTrue(_FakeSerialEnv.instances[0].closed)
        # Each episode is 3 steps of reward 1.0.
        self.assertAlmostEqual(summary["mean_episode_reward"], 3.0)
        self.assertAlmostEqual(summary["win_rate"], 1.0)
        # Seeds are seed + episode_index, exactly once each.
        self.assertEqual(
            _FakeSerialEnv.instances[0].reset_seeds, [100, 101, 102, 103]
        )

    def test_accepts_tensor_like_and_tuple_actions(self):
        model = _FakeModel(as_tensor=True, as_tuple=True)
        summary = evaluate_model(model, {}, episodes=1, seed=1)
        self.assertEqual(summary["episodes"], 1)

    def test_accepts_bare_array_action_without_tuple(self):
        model = _FakeModel(as_tuple=False)
        summary = evaluate_model(model, {}, episodes=1, seed=1)
        self.assertEqual(summary["episodes"], 1)

    def test_closes_env_even_if_predict_raises(self):
        class _RaisingModel:
            def predict(self, observation, deterministic=True):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            evaluate_model(_RaisingModel(), {}, episodes=1)
        self.assertTrue(_FakeSerialEnv.instances[0].closed)

    def test_writes_summary_json_csv_and_txt_when_output_dir_given(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "eval_out"
            summary = evaluate_model(model, {}, episodes=2, seed=5, output_dir=out_dir)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "episodes.csv").exists())
            self.assertTrue((out_dir / "summary.txt").exists())

            on_disk = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["episodes"], 2)

            with (out_dir / "episodes.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)

            text = (out_dir / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("SandboxAI evaluation", text)
            self.assertEqual(text, format_summary(summary))


class EvaluateModelVectorizedTests(unittest.TestCase):
    def setUp(self):
        _FakeVecEnv.instances.clear()
        patcher = patch("sandboxai.godot_env.GodotVecEnv", _FakeVecEnv)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_environment_count_above_one_uses_vectorized_path(self):
        model = _FakeModel(action=[[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], as_tuple=False)
        summary = evaluate_model(
            model, {"environment_count": 2}, episodes=5, seed=7
        )
        self.assertEqual(summary["environment_count"], 2)
        # Exactly the requested number of episodes, never more.
        self.assertEqual(summary["episodes"], 5)
        self.assertEqual(len(summary["episodes_detail"]), 5)
        self.assertTrue(_FakeVecEnv.instances[0].closed)

    def test_stops_exactly_at_requested_episode_count(self):
        # With 2 sub-envs finishing at different cadences, more than the
        # requested count could complete on the same tick; the extra ones
        # must be discarded rather than over-counted.
        model = _FakeModel(action=[[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], as_tuple=False)
        summary = evaluate_model(
            model, {"environment_count": 2}, episodes=1, seed=0
        )
        self.assertEqual(summary["episodes"], 1)


class FormatSummaryTests(unittest.TestCase):
    def test_format_summary_renders_without_crashing_on_empty_summary(self):
        text = format_summary({})
        self.assertIn("episodes: 0", text)
        self.assertIn("mean reward: 0.0000", text)


if __name__ == "__main__":
    unittest.main()
