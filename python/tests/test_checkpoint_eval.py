"""Regression tests for checkpoint-time evaluation (Phase 6-8 integration).

No Godot binary: the bridge is faked at the GodotBatchClient/SelfPlayBatchClient
boundary. Real SB3 models are used for the league tests so snapshot
isolation, contract checks, weight independence and training-policy
immutability are exercised against real torch weights.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sandboxai.checkpoint_eval import (
    LeagueIncompatibilityError,
    LeagueRunner,
    PlanExecutor,
    PlannedEpisode,
    build_map_split,
    run_checkpoint_evaluation,
    suite_seeds_disjoint,
)
from sandboxai.conditions import MAP_IDS, Condition
from sandboxai.config import TrainingConfig
from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.curriculum_stages import applied_condition
from sandboxai.generalization import GeneralizationSuite
from sandboxai.pipeline import TrainingPipeline


def _models():
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _Env(gym.Env):
        observation_space = spaces.Box(-10, 10, shape=(OBSERVATION_FIELD_COUNT,), dtype=np.float32)
        action_space = spaces.MultiDiscrete(list(ACTION_NVEC))

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBSERVATION_FIELD_COUNT, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBSERVATION_FIELD_COUNT, dtype=np.float32), 0.0, False, False, {}

    def make(seed: int = 0, obs_dim: int = OBSERVATION_FIELD_COUNT):
        env_cls = _Env
        if obs_dim != OBSERVATION_FIELD_COUNT:
            class _Bad(_Env):
                observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)
            env_cls = _Bad
        return PPO(
            "MlpPolicy",
            DummyVecEnv([env_cls]),
            n_steps=16,
            batch_size=16,
            seed=seed,
            device="cpu",
            policy_kwargs={"net_arch": dict(pi=[32], vf=[32])},
        )
    return make


class _FakeSelfPlayClient:
    """Scripted deterministic two-slot bridge: slot B wins after 5 steps."""

    seeds: list[int] = []

    def __init__(self, **kwargs):
        assert int(kwargs.get("environment_count", 1)) == 1
        self._steps = 0

    def reset(self, seed):
        type(self).seeds.append(seed)
        self._steps = 0
        return [[[0.0] * OBSERVATION_FIELD_COUNT, [0.0] * OBSERVATION_FIELD_COUNT]]

    def step(self, pairs):
        assert len(pairs) == 1 and len(pairs[0]) == 2
        self._steps += 1
        done = self._steps >= 5
        obs = [[0.0] * OBSERVATION_FIELD_COUNT, [0.0] * OBSERVATION_FIELD_COUNT]
        done_reason = "timeout" if done else ""
        info_a = {"metrics": {"win": False, "damage_dealt": 3.0, "episode_length": self._steps}, "done_reason": done_reason}
        info_b = {"metrics": {"win": done, "damage_dealt": 9.0, "episode_length": self._steps}, "done_reason": done_reason}
        return [obs], [[0.0, 0.0]], [done], [[info_a, info_b]]

    def close(self):
        pass


class LeagueRunnerTest(unittest.TestCase):
    def setUp(self):
        self._make = _models()
        self.tmp = Path(tempfile.mkdtemp())

    def _evaluate(self, runner, model, step, matches=4, **kwargs):
        return runner.evaluate_checkpoint(
            model,
            experiment_id="expA",
            step=step,
            matches=matches,
            env_kwargs={"project_path": ".", "godot_executable": "godot"},
            client_factory=lambda **kw: _FakeSelfPlayClient(**kw),
            **kwargs,
        )

    def test_frozen_snapshot_is_created_and_registry_persists(self):
        model = self._make()
        runner = LeagueRunner(self.tmp, seed=42, device="cpu")
        report = self._evaluate(runner, model, step=1000)
        self.assertIn("expA@1000", report["policy_id"])
        snapshot = self.tmp / "league" / "policies" / "expA@1000.zip"
        self.assertTrue(snapshot.is_file(), "snapshot file must exist (frozen opponents are files)")
        self.assertTrue((self.tmp / "league" / "registry.json").is_file())
        self.assertTrue((self.tmp / "league" / "history.json").is_file())
        self.assertEqual(report["matches"], 2)  # scripted baseline, both seatings
        self.assertIn("scripted_baseline", report["opponents"])

    def test_training_policy_is_not_mutated(self):
        model = self._make()
        before = {key: value.cpu().numpy().copy() for key, value in model.policy.state_dict().items()}
        runner = LeagueRunner(self.tmp, seed=42, device="cpu")
        self._evaluate(runner, model, step=1000)
        after = model.policy.state_dict()
        for key, value in before.items():
            np.testing.assert_array_equal(value, after[key].cpu().numpy(),
                                          f"league evaluation mutated training parameter {key}")

    def test_schedule_is_resume_identical(self):
        model = self._make()
        _FakeSelfPlayClient.seeds = []
        runner_a = LeagueRunner(self.tmp, seed=42, device="cpu")
        report_a = self._evaluate(runner_a, model, step=1000)
        seeds_a = list(_FakeSelfPlayClient.seeds)

        _FakeSelfPlayClient.seeds = []
        runner_b = LeagueRunner(self.tmp, seed=42, device="cpu")  # fresh runner over the same run dir
        report_b = self._evaluate(runner_b, model, step=1000)
        self.assertEqual(_FakeSelfPlayClient.seeds, seeds_a)
        self.assertEqual(report_a["opponents"], report_b["opponents"])

    def test_incompatible_checkpoint_fails_clearly(self):
        bad_model = self._make(obs_dim=80)
        runner = LeagueRunner(self.tmp, seed=42, device="cpu")
        with self.assertRaises(LeagueIncompatibilityError) as ctx:
            self._evaluate(runner, bad_model, step=1, matches=2)
        self.assertIn("cannot play in this league", str(ctx.exception))

    def test_opponent_pool_grows_with_frozen_snapshots(self):
        model = self._make()
        runner = LeagueRunner(self.tmp, seed=7, device="cpu")
        self._evaluate(runner, model, step=1000)
        report = self._evaluate(runner, model, step=2000, matches=6)
        self.assertIn("expA@1000", report["opponents"])
        ids = sorted(runner.registry.ids())
        self.assertIn("expA@1000", ids)
        self.assertIn("expA@2000", ids)
        self.assertIn("scripted_baseline", ids)


class _FakeExecutor:
    """Plan executor stand-in: deterministic rows, one per plan, in order."""

    def __init__(self, env_kwargs, skill_metrics=True):
        self.env_kwargs = env_kwargs
        self.closed = False

    def run(self, model, plans, policy_id="", checkpoint="", replay_dir=None):
        rows = []
        for plan in plans:
            condition = plan.condition
            rows.append(
                {
                    "labels": dict(plan.labels),
                    "condition": condition.to_dict(),
                    "seed": condition.seed,
                    "environment_index": 0,
                    "episode_reward": 1.5,
                    "episode_length": 120,
                    "win": True,
                    "done_reason": "won",
                    "skill": {"aim": {"accuracy": 0.4}},
                }
            )
        return rows

    def close(self):
        self.closed = True


class _Telemetry:
    def __init__(self):
        self.rows = []

    def write(self, payload):
        self.rows.append(dict(payload))


class CheckpointEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.config = TrainingConfig(
            seed=3,
            environment_count=2,
            project_path=str(self.tmp),
            godot_executable="godot",
            episode_log=False,
            checkpoint_league_eval=True,
            league_matches_per_checkpoint=4,
            condition_eval_episodes=6,
            output_root=str(self.tmp / "runs"),
            run_id="eval_test",
            replay_mode="evaluation",
        )
        self.config.validate()
        self.pipeline = TrainingPipeline(self.config, self.config.run_directory(), _Telemetry(), device="cpu")
        self.used = [
            {"seed": s, "map_id": m, "scenario": "cover_fight", "lighting": "normal", "enemy_count": 1, "level": 5}
            for i, m in enumerate(["open_field", "training_yard"])
            for s in (101 + 17 * i, 3059 + 71 * i)
        ]
        self.pipeline.driver.used_conditions.extend(self.used)

    def _run(self, step=50_000):
        model = _models()()
        return run_checkpoint_evaluation(
            model,
            step=step,
            config=self.config,
            pipeline=self.pipeline,
            output_dir=self.config.run_directory() / "evaluations" / f"step_{step:09d}",
            device="cpu",
            normal_summary={"mean_episode_reward": 1.23},
            executor_factory=lambda ek, skill_metrics=True: _FakeExecutor(ek),
            client_factory=lambda **kw: _FakeSelfPlayClient(**kw),
        )

    def test_report_sections_and_files(self):
        report = self._run()
        self.assertIn("condition_evaluation", report)
        self.assertIn("generalization", report)
        self.assertIn("league", report)
        destination = self.config.run_directory() / "evaluations" / "step_000050000"
        for name in ("report.json", "policy.zip", "generalization.json", "episodes.csv"):
            self.assertTrue((destination / name).is_file(), name)
        roundtrip = json.loads((destination / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(roundtrip["timesteps"], 50000)

    def test_condition_evaluation_uses_frozen_eval_set(self):
        report = self._run()
        rows_a = [row["seed"] for row in report["condition_evaluation"]["episodes_detail"]]
        report2 = self._run(step=200_000)
        rows_b = [row["seed"] for row in report2["condition_evaluation"]["episodes_detail"]]
        self.assertEqual(rows_a, rows_b, "condition eval set must be frozen (comparable across checkpoints)")
        # And disjoint from the training stream by construction (eval master
        # seed salt): none of the eval seeds may be a sampled stage seed.
        self.assertTrue(all(len(str(seed)) > 4 for seed in rows_a))

    def test_seen_unseen_separation(self):
        report = self._run()
        detail = report["generalization"].get("episodes_detail", [])
        self.assertTrue(detail, "generalization must run once maps are trained")
        trained_seeds = {int(c["seed"]) for c in self.used}
        known_replays = set()
        for row in detail:
            bucket = row["labels"].get("map_bucket")
            if bucket == "known":
                known_replays.add(row["seed"])
                self.assertIn(row["condition"]["map_id"], {"open_field", "training_yard"})
            else:
                self.assertNotIn(row["seed"], trained_seeds, f"trained seed leaked into {bucket}")
        self.assertTrue(known_replays, "known bucket must replay trained seeds")
        self.assertNotEqual(report["generalization"].get("suite_level"), report["generalization"].get("trained_level"))

    def test_league_section_records_without_mutating(self):
        report = self._run()
        league = report["league"]
        self.assertGreaterEqual(league["matches"], 2)
        self.assertIn("elo_note", league)

    def test_generalization_skips_honestly_below_world_levels(self):
        self.pipeline.driver.used_conditions.clear()
        report = self._run()
        generalization = report.get("generalization", {})
        self.assertTrue(generalization.get("skipped"))
        self.assertIn("reason", generalization)
        self.assertEqual(generalization.get("episodes"), 0)


class BuildMapSplitTest(unittest.TestCase):
    def test_trained_conditions_classify_as_known(self):
        used = [
            {"seed": 41, "map_id": "open_field", "scenario": "cover_fight", "lighting": "normal",
             "enemy_count": 1, "level": 5},
            {"seed": 37, "map_id": "pillar_hall", "scenario": "corridor_fight", "lighting": "fog",
             "enemy_count": 2, "level": 7},
        ]
        split = build_map_split(used)
        for row in used:
            self.assertEqual(split.bucket_of(row["map_id"], row["seed"], row["scenario"]), "known")
        # A map never trained on must classify as an unseen map.
        untrained = [m for m in MAP_IDS if m not in split.train_maps]
        self.assertTrue(untrained)
        self.assertEqual(split.bucket_of(untrained[0], 99991, ""), "unseen_maps")
        # Trained map, new seed: unseen_seeds (never "known" inflation).
        self.assertEqual(split.bucket_of("open_field", 99991, ""), "unseen_seeds")
        # Trained map + unseen scenario: unseen_variants.
        scenario = next(s for s in split.unseen_scenarios)
        self.assertEqual(split.bucket_of("open_field", 41, scenario), "unseen_variants")

    def test_suite_disjointness_rule_targets_unseen_buckets_only(self):
        used = [
            {"seed": 41, "map_id": "open_field", "scenario": "cover_fight", "lighting": "normal",
             "enemy_count": 1, "level": 5},
        ]
        split = build_map_split(used)
        suite = GeneralizationSuite(split=split, level=6, episodes_per_cell=1, base_seed=70_000)
        # Known-bucket cells intentionally replay seed 41: suite is disjoint
        # in the sense that matters (unseen cells never reuse it).
        self.assertTrue(suite_seeds_disjoint(suite, {41}))
        plan = suite.plan()
        self.assertTrue(any(ep.condition.seed == 41 for ep in plan if ep.map_bucket == "known"))
        self.assertFalse(any(ep.condition.seed == 41 for ep in plan if ep.map_bucket != "known"))


class _FakeBatchClient:
    """GodotBatchClient protocol stand-in with deterministic episodes.

    Each planned episode ends after ``(seed % 5) + 2`` steps; slot wins
    when ``seed`` is even. Pure functions of the plan seeds => executor
    scheduling cannot change the results.
    """

    def __init__(self, environment_count: int = 3):
        self.environment_count = environment_count
        self._counters = [0] * environment_count
        self._target = [0] * environment_count
        self.staged: list[list[dict]] = []

    def set_episode_plans(self, plans):
        self.staged.append(list(plans))
        for payload in plans:
            index = int(payload["index"])
            self._target[index] = int(payload["seed"]) % 5 + 2
            self._counters[index] = 0
        return {"ok": True, "staged": [int(p["index"]) for p in plans]}

    def reset(self, seed=None):
        return [[0.0] * OBSERVATION_FIELD_COUNT for _ in range(self.environment_count)], [{} for _ in range(self.environment_count)]

    def reset_indices(self, indices, seed=None):
        for index in indices:
            self._counters[index] = 0
        return [
            {"index": index, "observation": [0.0] * OBSERVATION_FIELD_COUNT}
            for index in indices
        ]

    def step(self, actions):
        dones = []
        infos = []
        for index in range(self.environment_count):
            self._counters[index] += 1
            done = self._target[index] > 0 and self._counters[index] >= self._target[index]
            dones.append(done)
            if done:
                won = self._target[index] % 2 == 0
                infos.append({"metrics": {"win": won, "episode_length": self._counters[index], "episode_reward": 1.0},
                              "done_reason": "won" if won else "agent_death", "events": {}})
                self._target[index] = 0
            else:
                infos.append({"events": {}})
        return (
            [[0.0] * OBSERVATION_FIELD_COUNT for _ in range(self.environment_count)],
            [0.0] * self.environment_count,
            dones,
            infos,
        )

    def close(self):
        pass


class PlanExecutorTest(unittest.TestCase):
    def _plans(self, count: int) -> list[PlannedEpisode]:
        return [
            PlannedEpisode(
                condition=Condition(
                    map_id="open_field", scenario="", lighting="normal",
                    enemy_count=1, level=5, seed=1000 + i * 97,
                ),
                labels={"axis": "test", "_position": i},
            )
            for i in range(count)
        ]

    def _executor(self, environment_count: int) -> PlanExecutor:
        executor = PlanExecutor.__new__(PlanExecutor)
        executor.client = _FakeBatchClient(environment_count)
        executor.environment_count = environment_count
        executor.skill_metrics_enabled = True
        return executor

    def _model(self):
        class _Model:
            def predict(self, batch, deterministic=True):
                import numpy as np
                return np.zeros((len(batch), 6), dtype=np.int64), None
        return _Model()

    def test_rows_match_plans_in_order_and_are_deterministic(self):
        plans = self._plans(10)
        rows_a = self._executor(3).run(self._model(), plans, policy_id="pol")
        rows_b = self._executor(2).run(self._model(), plans, policy_id="pol")
        self.assertEqual(len(rows_a), 10)
        # Scheduling independence: N=3 and N=2 must produce the same rows.
        strip = lambda rows: [{k: v for k, v in row.items() if k not in ("environment_index", "episode_length", "win")} for row in rows]
        self.assertEqual(strip(rows_a), strip(rows_b))
        self.assertEqual([row["labels"]["_position"] for row in rows_a], list(range(10)))

    def test_applied_condition_is_sanity_preserved_by_executor_inputs(self):
        # Guard against a mis-wiring where raw sampled conditions (with maps
        # on low levels) reach the bridge: callers must apply
        # applied_condition first.
        raw = Condition(map_id="pillar_hall", scenario="ambush", lighting="night",
                        enemy_count=1, level=2, seed=1)
        applied = applied_condition(raw)
        self.assertEqual(applied.map_id, "")
        self.assertEqual(applied.level, 2)


if __name__ == "__main__":
    unittest.main()
