"""Unit tests for the pure helpers train_ppo was decomposed into.

These used to be unreachable: every one of them lived inside an 880-line
``train_ppo`` (or inside a callback class nested in it), so the only way
to execute them was to start a real PPO run against a Godot bridge. They
are now module-level functions and a plain ``_EvaluationDriver`` object,
which is exactly what makes this file possible — no stable-baselines3,
no torch, no engine.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from optional_deps import GYMNASIUM_REASON, HAS_GYMNASIUM

if HAS_GYMNASIUM:  # pragma: no branch - import guard
    from sandboxai import ppo
    from sandboxai.selection import CheckpointSelectionRule


class RecordingTelemetry:
    """Minimal stand-in for JsonlTelemetry that keeps the rows in memory."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, payload: dict[str, Any]) -> None:
        self.rows.append(payload)

    def events(self) -> list[str]:
        return [str(row.get("event")) for row in self.rows]


class SavingModel:
    """Stand-in for an SB3 model that only records where it was saved."""

    def __init__(self) -> None:
        self.saved_to: list[Path] = []

    def save(self, path: Any) -> None:
        self.saved_to.append(Path(path))
        Path(path).write_text("checkpoint", encoding="utf-8")


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class RunDirectoryTests(unittest.TestCase):
    """A resumed run must write back into the run it resumed."""

    def test_checkpoint_inside_checkpoints_maps_to_the_run_root(self) -> None:
        checkpoint = Path("/runs/run_a/checkpoints/ppo_1000_steps.zip")
        self.assertEqual(ppo._resolve_run_directory(_config(), checkpoint), Path("/runs/run_a"))

    def test_checkpoint_in_the_run_root_maps_to_that_root(self) -> None:
        # final.zip lives directly in the run directory; mapping to the
        # parent would scatter checkpoints and logs across sibling runs.
        checkpoint = Path("/runs/run_a/final.zip")
        self.assertEqual(ppo._resolve_run_directory(_config(), checkpoint), Path("/runs/run_a"))

    def test_without_a_checkpoint_the_config_decides(self) -> None:
        config = _config()
        self.assertEqual(ppo._resolve_run_directory(config, None), config.run_directory())

    def test_layout_creates_all_three_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw) / "run"
            checkpoints, logs, evaluations = ppo._prepare_run_layout(run_dir)
            self.assertEqual(checkpoints, run_dir / "checkpoints")
            self.assertEqual(logs, run_dir / "logs")
            self.assertEqual(evaluations, run_dir / "evaluations")
            for directory in (checkpoints, logs, evaluations):
                self.assertTrue(directory.is_dir(), directory)


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class ShootRequestCountingTests(unittest.TestCase):
    """The policy-side shooting diagnostic, counted before the bridge."""

    def test_counts_decisions_and_fire_bits(self) -> None:
        actions = [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 2]]
        self.assertEqual(ppo._count_shoot_requests(actions), (3, 2))

    def test_short_action_vectors_are_not_decisions(self) -> None:
        # Pre-fire-bit action vectors: counting them would invent
        # decisions the policy never made.
        self.assertEqual(ppo._count_shoot_requests([[1, 1, 1, 1]]), (0, 0))

    def test_accepts_array_like_batches(self) -> None:
        class Arrayish:
            def __init__(self, rows: list[list[int]]) -> None:
                self._rows = rows

            def tolist(self) -> list[list[int]]:
                return self._rows

        self.assertEqual(ppo._count_shoot_requests(Arrayish([[0, 0, 0, 0, 1, 0]])), (1, 1))

    def test_empty_batch_is_zero(self) -> None:
        self.assertEqual(ppo._count_shoot_requests([]), (0, 0))


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class EpisodeCollectionTests(unittest.TestCase):
    """Only finished episodes feed the interval averages."""

    def test_done_reason_and_terminal_observation_both_count(self) -> None:
        infos = [
            {"metrics": {"kills": 1.0}, "done_reason": "timeout"},
            {"metrics": {"kills": 2.0}, "terminal_observation": [0.0]},
            {"metrics": {"kills": 3.0}},
            "not a dict",
        ]
        collected = ppo._finished_episode_metrics(infos)
        self.assertEqual([item["kills"] for item in collected], [1.0, 2.0])

    def test_a_finished_episode_without_metrics_yields_an_empty_dict(self) -> None:
        self.assertEqual(ppo._finished_episode_metrics([{"done_reason": "x"}]), [{}])


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class EpisodeMeanTests(unittest.TestCase):
    """Averages are over the buffer, and missing keys read as zero."""

    def test_means_cover_every_documented_payload_key(self) -> None:
        buffer = [
            {"episode_reward": 2.0, "kills": 1.0, "win": 1.0},
            {"episode_reward": 4.0, "kills": 3.0, "win": 0.0},
        ]
        means = ppo._episode_means(buffer)
        self.assertEqual(means["mean_episode_reward"], 3.0)
        self.assertEqual(means["mean_kills"], 2.0)
        self.assertEqual(means["win_rate"], 0.5)
        # A key no episode reported is still present, as zero, so the
        # telemetry schema does not change shape mid-run.
        self.assertEqual(means["mean_accuracy"], 0.0)
        self.assertEqual(len(means), len(ppo._EPISODE_MEAN_KEYS))

    def test_reward_breakdown_reads_the_nested_dict(self) -> None:
        buffer = [
            {"reward_breakdown": {"reward_hits": 1.0, "penalty_death": -2.0}},
            {"reward_breakdown": {"reward_hits": 3.0}},
        ]
        breakdown = ppo._reward_breakdown_means(buffer)
        self.assertEqual(breakdown["hits"], 2.0)
        self.assertEqual(breakdown["death_penalty"], -1.0)
        self.assertEqual(len(breakdown), len(ppo._REWARD_BREAKDOWN_KEYS))

    def test_episodes_without_a_breakdown_count_as_zero(self) -> None:
        breakdown = ppo._reward_breakdown_means([{"reward_breakdown": {"reward_hits": 4.0}}, {}])
        self.assertEqual(breakdown["hits"], 2.0)


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class ProgressPayloadTests(unittest.TestCase):
    """The `progress` telemetry row."""

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "device": "cpu",
            "checkpoints": Path("/runs/run_a/checkpoints"),
            "elapsed": 2.0,
            "num_timesteps": 1200,
            "start_timesteps": 200,
            "target_timesteps": 2200,
            "episode_count": 7,
            "interval_decisions": 10,
            "interval_shoot_requests": 4,
            "episode_metrics": [],
            "resources": {"cpu_percent": 12.5},
        }
        arguments.update(overrides)
        return ppo._progress_payload(_config(total_training_steps=2000), **arguments)

    def test_throughput_and_eta_use_the_run_average(self) -> None:
        payload = self._payload()
        self.assertEqual(payload["steps_per_second"], 500.0)
        self.assertEqual(payload["eta_seconds"], 2.0)
        self.assertEqual(payload["progress"], 0.5)
        self.assertEqual(payload["policy_shoot_request_rate"], 0.4)
        self.assertEqual(payload["cpu_percent"], 12.5)

    def test_eta_is_omitted_rather_than_faked_before_the_first_step(self) -> None:
        payload = self._payload(num_timesteps=200)
        self.assertIsNone(payload["eta_seconds"])

    def test_progress_is_clamped_at_one(self) -> None:
        self.assertEqual(self._payload(num_timesteps=9000)["progress"], 1.0)

    def test_no_decisions_means_a_zero_rate_not_a_division_error(self) -> None:
        payload = self._payload(interval_decisions=0, interval_shoot_requests=0)
        self.assertEqual(payload["policy_shoot_request_rate"], 0.0)

    def test_episode_means_are_merged_only_when_episodes_finished(self) -> None:
        self.assertNotIn("mean_kills", self._payload())
        payload = self._payload(episode_metrics=[{"kills": 2.0}])
        self.assertEqual(payload["mean_kills"], 2.0)

    def test_reward_breakdown_follows_the_config_flag(self) -> None:
        metrics = [{"reward_breakdown": {"reward_hits": 1.0}}]

        def payload(enabled: bool) -> dict[str, Any]:
            return ppo._progress_payload(
                _config(reward_breakdown_logging=enabled),
                device="cpu",
                checkpoints=Path("/x"),
                elapsed=1.0,
                num_timesteps=100,
                start_timesteps=0,
                target_timesteps=100,
                episode_count=1,
                interval_decisions=1,
                interval_shoot_requests=0,
                resources={},
                episode_metrics=metrics,
            )

        self.assertEqual(payload(True)["reward_breakdown"]["hits"], 1.0)
        self.assertNotIn("reward_breakdown", payload(False))


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class BestScoreInheritanceTests(unittest.TestCase):
    """Resuming may only inherit a score the current rule can compare."""

    def setUp(self) -> None:
        self.rule = CheckpointSelectionRule()
        self.state = ppo._SelectionState(best_score=self.rule.initial_score())
        self.telemetry = RecordingTelemetry()

    def _record(self, body: dict[str, Any]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "best.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_missing_record_leaves_the_initial_score(self) -> None:
        ppo._inherit_best_score(Path("/does/not/exist.json"), self.rule, self.state, self.telemetry)
        self.assertEqual(self.state.best_score, self.rule.initial_score())
        self.assertEqual(self.telemetry.rows, [])

    def test_a_matching_rule_inherits_the_score(self) -> None:
        path = self._record({"selection_rule": self.rule.as_dict(), "score": 12.5})
        ppo._inherit_best_score(path, self.rule, self.state, self.telemetry)
        self.assertEqual(self.state.best_score, 12.5)

    def test_the_legacy_mean_reward_field_is_still_honoured(self) -> None:
        path = self._record({"selection_rule": self.rule.as_dict(), "mean_reward": 7.0})
        ppo._inherit_best_score(path, self.rule, self.state, self.telemetry)
        self.assertEqual(self.state.best_score, 7.0)

    def test_a_changed_rule_restarts_selection_and_says_so(self) -> None:
        path = self._record(
            {"selection_rule": {"metric": "win_rate", "direction": "max"}, "score": 99.0}
        )
        ppo._inherit_best_score(path, self.rule, self.state, self.telemetry)
        self.assertEqual(self.state.best_score, self.rule.initial_score())
        self.assertIn("checkpoint_selection_rule_changed", self.telemetry.events())

    def test_a_non_numeric_score_is_ignored(self) -> None:
        # `True` is an int in Python; inheriting it would set the bar to 1.0.
        path = self._record({"selection_rule": self.rule.as_dict(), "score": True})
        ppo._inherit_best_score(path, self.rule, self.state, self.telemetry)
        self.assertEqual(self.state.best_score, self.rule.initial_score())


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class EvaluationEnvKwargsTests(unittest.TestCase):
    """Bridge sizing for the two persistent evaluation environments."""

    def test_normal_bridge_is_capped_at_the_episode_count(self) -> None:
        config = _config(evaluation_episodes=3, evaluation_environment_count=8)
        self.assertEqual(ppo._evaluation_env_kwargs(config, None)["environment_count"], 3)

    def test_normal_bridge_never_drops_below_one(self) -> None:
        config = _config(evaluation_episodes=4, evaluation_environment_count=0)
        self.assertEqual(ppo._evaluation_env_kwargs(config, None)["environment_count"], 1)

    def test_the_profiler_view_is_only_attached_when_profiling(self) -> None:
        config = _config()
        self.assertNotIn("profiler", ppo._evaluation_env_kwargs(config, None))
        self.assertEqual(ppo._evaluation_env_kwargs(config, "view")["profiler"], "view")

    def test_battery_bridge_uses_the_eval_master_seed_and_compact_infos(self) -> None:
        from sandboxai.pipeline import EVAL_MASTER_SEED_SALT

        config = _config(seed=11, checkpoint_eval_environment_count=4)
        kwargs = ppo._battery_env_kwargs(config, None)
        self.assertEqual(kwargs["seed"], 11 + EVAL_MASTER_SEED_SALT)
        self.assertEqual(kwargs["environment_count"], 4)
        self.assertTrue(kwargs["compact_infos"])


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class SelectionTests(unittest.TestCase):
    """Best-checkpoint selection and early stopping, without a training run."""

    def _driver(self, **config_overrides: Any) -> tuple[Any, RecordingTelemetry, Path]:
        directory = Path(tempfile.mkdtemp())
        telemetry = RecordingTelemetry()
        rule = CheckpointSelectionRule()
        driver = ppo._EvaluationDriver(
            config=_config(**config_overrides),
            telemetry=telemetry,
            run_control=None,
            pipeline=None,
            profiler=None,
            device="cpu",
            checkpoints=directory / "checkpoints",
            evaluations=directory / "evaluations",
            best_path=directory / "best_eval.zip",
            best_record_path=directory / "best.json",
            selection_rule=rule,
            inference_scheduler=None,
            state=ppo._SelectionState(best_score=rule.initial_score()),
        )
        return driver, telemetry, directory

    def test_an_improvement_saves_the_checkpoint_and_writes_the_record(self) -> None:
        driver, _telemetry, directory = self._driver()
        model = SavingModel()
        driver._apply_selection(model, 1000, {"mean_episode_reward": 5.0}, 5.0)
        self.assertEqual(model.saved_to, [directory / "best_eval.zip"])
        record = json.loads((directory / "best.json").read_text(encoding="utf-8"))
        self.assertEqual(record["score"], 5.0)
        # Legacy field, kept so existing readers keep working.
        self.assertEqual(record["mean_reward"], 5.0)
        self.assertEqual(driver.state.best_score, 5.0)
        self.assertEqual(driver.state.eval_patience_counter, 0)

    def test_no_improvement_advances_patience_without_saving(self) -> None:
        driver, _telemetry, _directory = self._driver()
        driver.state.best_score = 10.0
        model = SavingModel()
        driver._apply_selection(model, 1000, {"mean_episode_reward": 1.0}, 1.0)
        self.assertEqual(model.saved_to, [])
        self.assertEqual(driver.state.eval_patience_counter, 1)
        self.assertFalse(driver.state.stop_training)

    def test_patience_exhaustion_stops_training(self) -> None:
        driver, _telemetry, _directory = self._driver(early_stopping_patience=2)
        driver.state.best_score = 10.0
        for _ in range(2):
            driver._apply_selection(SavingModel(), 1000, {"mean_episode_reward": 1.0}, 1.0)
        self.assertTrue(driver.state.stop_training)

    def test_patience_zero_never_stops(self) -> None:
        driver, _telemetry, _directory = self._driver(early_stopping_patience=0)
        driver.state.best_score = 10.0
        for _ in range(5):
            driver._apply_selection(SavingModel(), 1000, {"mean_episode_reward": 1.0}, 1.0)
        self.assertFalse(driver.state.stop_training)

    def test_reaching_the_target_reward_stops_training(self) -> None:
        driver, _telemetry, _directory = self._driver(min_eval_reward=4.0)
        driver._apply_selection(SavingModel(), 1000, {"mean_episode_reward": 4.0}, 4.0)
        self.assertTrue(driver.state.stop_training)

    def test_a_metric_this_run_does_not_produce_is_reported_once(self) -> None:
        driver, telemetry, _directory = self._driver()
        driver.selection_rule = CheckpointSelectionRule(metric="not_produced")
        model = SavingModel()
        driver._apply_selection(model, 1000, {"mean_episode_reward": 5.0}, 5.0)
        self.assertEqual(telemetry.events(), ["checkpoint_selection_metric_missing"])
        # Selecting on some other metric instead would be worse than
        # selecting on nothing, so nothing is saved.
        self.assertEqual(model.saved_to, [])
        self.assertEqual(driver.state.eval_patience_counter, 1)


@unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
class EvaluationScheduleTests(unittest.TestCase):
    """The evaluation boundary schedule, including the resume anchor."""

    def _driver(self, **config_overrides: Any) -> Any:
        directory = Path(tempfile.mkdtemp())
        rule = CheckpointSelectionRule()
        return ppo._EvaluationDriver(
            config=_config(**config_overrides),
            telemetry=RecordingTelemetry(),
            run_control=None,
            pipeline=None,
            profiler=None,
            device="cpu",
            checkpoints=directory / "checkpoints",
            evaluations=directory / "evaluations",
            best_path=directory / "best_eval.zip",
            best_record_path=directory / "best.json",
            selection_rule=rule,
            inference_scheduler=None,
            state=ppo._SelectionState(best_score=rule.initial_score()),
        )

    def test_the_first_boundary_is_relative_to_where_training_starts(self) -> None:
        # Regression: an absolute threshold was already in the past after
        # a resume, which made the callback evaluate on every single step.
        driver = self._driver(evaluation_frequency=5000)
        driver.on_training_start(120_000)
        self.assertEqual(driver.next_evaluation, 125_000)

    def test_a_step_before_the_boundary_does_no_work(self) -> None:
        driver = self._driver(evaluation_frequency=5000)
        driver.on_training_start(0)
        self.assertTrue(driver.on_step(SavingModel(), 4999))
        self.assertIsNone(driver.eval_env)

    def test_a_requested_stop_short_circuits_the_callback(self) -> None:
        driver = self._driver()
        driver.state.stop_training = True
        self.assertFalse(driver.on_step(SavingModel(), 10**9))

    def test_close_is_safe_before_any_bridge_was_started(self) -> None:
        driver = self._driver()
        driver.close()
        self.assertIsNone(driver.eval_env)
        self.assertIsNone(driver.battery_executor)

    def test_close_survives_a_bridge_that_already_died(self) -> None:
        class BrokenBridge:
            def close(self) -> None:
                raise RuntimeError("bridge already gone")

        driver = self._driver()
        driver.eval_env = BrokenBridge()
        # A dead bridge must not mask the real training result.
        driver.close()
        self.assertIsNone(driver.eval_env)


def _config(**overrides: Any) -> Any:
    from sandboxai.config import TrainingConfig

    defaults: dict[str, Any] = {
        "total_training_steps": 1000,
        "environment_count": 2,
        "evaluation_episodes": 2,
        "evaluation_frequency": 500,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
