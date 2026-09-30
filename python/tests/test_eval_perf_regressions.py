"""Performance-behaviour regression tests for the evaluation/training path.

The optimizations these tests guard (bridge-process reuse across evaluation
boundaries, batched checkpoint battery, optional inference-device split)
must be *behaviour-preserving*: identical observations, actions, rewards,
termination/truncation, evaluation semantics and seeded determinism. Every
test here compares optimized paths against the historical path on the same
deterministic fake bridge, so a change that silently alters any of those
contracts fails loudly.

No real Godot binary is required: the fakes speak the same JSON-lines
protocol as scripts/rl/rl_server.gd. Opt-in wall-clock benchmarks live at
the bottom (SANDBOXAI_PERF_BENCH=1) and are skipped in normal runs.
"""
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from optional_deps import HAS_SB3, HAS_TORCH, SB3_REASON, TORCH_REASON

from sandboxai.contract import OBSERVATION_FIELD_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_bridge(directory: Path, source: str) -> str:
    """Installs `source` as an executable fake Godot (argv-forwarding)."""
    bridge_py = directory / "fake_bridge.py"
    bridge_py.write_text(source, encoding="utf-8")
    wrapper = directory / "fake_godot"
    wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}' \"$@\"\n", encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(wrapper)


## A fake bridge whose responses are pure functions of (seed, step index,
## environment index) and honor compact_infos like the real server:
## episodes are 8 steps, rewards echo the shoot action, observations echo
## the per-environment step count, and terminal infos carry full metrics.
## Determinism of every wire value is what makes the reuse-equivalence
## assertions below meaningful.
DETERMINISTIC_FAKE_BRIDGE = r'''
import json, sys

OBS_DIM = __OBS_DIM__


def env_count_from_argv(default=2):
    for i, a in enumerate(sys.argv):
        if a == "--env-count" and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                return default
    return default


ENV_COUNT = env_count_from_argv()


def seed_from_argv(default=1234):
    for i, a in enumerate(sys.argv):
        if a == "--seed" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return default
    return default


BASE_SEED = seed_from_argv()
SPAWN_LOG = __SPAWN_LOG__


def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if SPAWN_LOG:
    with open(SPAWN_LOG, "a") as fh:
        fh.write(str(SPAWN_LOG) + "\n")

staged = []
pending = {}
step_counts = {i: 0 for i in range(ENV_COUNT)}
current_seeds = {i: BASE_SEED + i for i in range(ENV_COUNT)}


def obs(index):
    # Include the episode seed so serial-vs-vector equivalence tests prove
    # that plan scheduling preserves the exact seed flow, not merely length.
    value = ((step_counts.get(index, 0) + current_seeds.get(index, 0)) % 7) * 0.125 - 0.5
    return [round(value, 6)] * OBS_DIM


def consume_pending(index):
    if index in pending:
        current_seeds[index] = int(pending.pop(index))
    step_counts[index] = 0


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        out({"ok": True,
             "action_space": {"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2, 2], "dimension": 6},
             "observation_space": {"type": "structured_float_vector", "size": OBS_DIM,
                                   "shape": [OBS_DIM], "low": -1.0, "high": 1.0}})
    elif command == "reset":
        seed = int(request.get("seed", -1))
        for i in range(ENV_COUNT):
            if i in pending:
                consume_pending(i)
            else:
                if seed >= 0:
                    current_seeds[i] = seed + i
                step_counts[i] = 0
        out({"ok": True, "observations": [obs(i) for i in range(ENV_COUNT)],
             "infos": [{"seed": current_seeds[i]} for i in range(ENV_COUNT)]})
    elif command == "reset_indices":
        for item in request.get("indices", []):
            index = int(item)
            consume_pending(index)
        out({"ok": True, "results": [{"index": int(item), "observation": obs(int(item))}
                                     for item in request.get("indices", [])]})
    elif command == "set_episode_plans":
        staged = request.get("plans", [])
        for plan in staged:
            pending[int(plan.get("index", -1))] = int(plan.get("seed", -1))
        out({"ok": True, "staged": [int(p.get("index", -1)) for p in staged]})
    elif command == "episode_conditions":
        out({"ok": True, "conditions": staged})
    elif command == "step":
        actions = request.get("actions", [])
        observations, rewards, dones, infos = [], [], [], []
        for i in range(ENV_COUNT):
            step_counts[i] = step_counts.get(i, 0) + 1
            done = step_counts[i] >= 8
            reward = 1.0 if (i < len(actions) and actions[i][4] == 1) else 0.01
            info = {"done_reason": "timeout" if done else "", "events": {}}
            if done or not request.get("compact_infos", False):
                info["metrics"] = {"episode_reward": 1.0, "episode_length": step_counts[i],
                                   "evaluation_seed": current_seeds[i],
                                   "win": False, "loss": True, "truncated": True,
                                   "kills": 1, "deaths": 0, "damage_dealt": 2.0,
                                   "damage_received": 0.0, "survival_time": 1.0,
                                   "accuracy": 0.5, "shots_fired": 4, "shots_hit": 2}
            if done:
                info["terminal_observation"] = [0.5] * OBS_DIM
                info["TimeLimit.truncated"] = True
                consume_pending(i)
            observations.append(obs(i))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        out({"ok": True, "observations": observations, "rewards": rewards,
             "dones": dones, "infos": infos})
    elif command == "ping":
        out({"ok": True, "pong": True})
    elif command == "health_check":
        out({"ok": True, "health": [{"healthy": True} for _ in range(ENV_COUNT)]})
    elif command == "close":
        out({"ok": True, "close": True})
        break
    else:
        out({"ok": False, "error": "unknown command"})
'''.replace("__OBS_DIM__", str(OBSERVATION_FIELD_COUNT))


def deterministic_bridge_source(spawn_log: str = "") -> str:
    return DETERMINISTIC_FAKE_BRIDGE.replace("__SPAWN_LOG__", json.dumps(spawn_log))


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class EvaluationReuseEquivalenceTest(unittest.TestCase):
    """evaluate_model(env=...) reuse must reproduce the fresh-process path
    exactly: same observations, rewards, termination and evaluation rows."""

    def setUp(self):
        if not HAS_SB3:
            self.skipTest(SB3_REASON)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.executable = _write_fake_bridge(tmp, deterministic_bridge_source())
        self.env_kwargs = {
            "project_path": str(PROJECT_ROOT),
            "godot_executable": self.executable,
            "environment_count": 1,
            "enemy_count": 1,
            "seed": 4321,
            "curriculum_level": 3,
        }

    def _model(self):
        from stable_baselines3 import PPO
        from sandboxai.godot_env import GodotVecEnv

        env = GodotVecEnv(**{**self.env_kwargs, "environment_count": 2})
        try:
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=8,
                batch_size=16,
                n_epochs=1,
                seed=7,
                device="cpu",
                verbose=0,
                policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}},
            )
        finally:
            env.close()
        return model

    def test_serial_reuse_matches_fresh_process_rows_exactly(self):
        from sandboxai.evaluation import evaluate_model

        model = self._model()
        # Fresh process per call (the historical behavior)...
        fresh = evaluate_model(model, self.env_kwargs, episodes=3, seed=900)
        fresh_again = evaluate_model(model, self.env_kwargs, episodes=3, seed=900)
        # ...and one reused process serving both calls.
        from sandboxai.godot_env import GodotGymEnv

        env = GodotGymEnv(**self.env_kwargs)
        try:
            reused_first = evaluate_model(model, self.env_kwargs, episodes=3, seed=900, env=env)
            reused_again = evaluate_model(model, self.env_kwargs, episodes=3, seed=900, env=env)
            # A *seeding* difference must still be visible through the reused
            # env (guards against reuse accidentally freezing one seed).
            different_seed = evaluate_model(model, self.env_kwargs, episodes=3, seed=1234, env=env)
        finally:
            env.close()
        strip = lambda rows: [
            {k: v for k, v in row.items() if k != "episode_index"} for row in rows
        ]
        self.assertEqual(strip(fresh["episodes_detail"]), strip(reused_first["episodes_detail"]))
        self.assertEqual(strip(fresh_again["episodes_detail"]), strip(reused_again["episodes_detail"]))
        seeds = {row["seed"] for row in different_seed["episodes_detail"]}
        self.assertEqual(seeds, {1234, 1235, 1236})

    def test_vectorized_reuse_matches_fresh_process_rows_exactly(self):
        from sandboxai.evaluation import evaluate_model
        from sandboxai.godot_env import GodotVecEnv

        model = self._model()
        serial = evaluate_model(model, self.env_kwargs, episodes=5, seed=900)
        kwargs = {**self.env_kwargs, "environment_count": 3}
        fresh = evaluate_model(model, kwargs, episodes=5, seed=900)
        env = GodotVecEnv(**kwargs)
        try:
            reused = evaluate_model(model, kwargs, episodes=5, seed=900, env=env)
            # A second boundary on the same process must re-seed exactly.
            reused_again = evaluate_model(model, kwargs, episodes=5, seed=900, env=env)
        finally:
            env.close()
        strip = lambda rows: [
            {k: v for k, v in row.items() if k not in ("episode_index", "environment_index")}
            for row in rows
        ]
        self.assertEqual(strip(fresh["episodes_detail"]), strip(reused["episodes_detail"]))
        self.assertEqual(strip(fresh["episodes_detail"]), strip(reused_again["episodes_detail"]))
        self.assertEqual(
            strip(serial["episodes_detail"]),
            strip(fresh["episodes_detail"]),
            "batched evaluation must run the exact serial seed+episode set",
        )
        self.assertEqual(
            [row["evaluation_seed"] for row in fresh["episodes_detail"]],
            [900, 901, 902, 903, 904],
        )


class ParallelEvaluationCoordinatorTest(unittest.TestCase):
    def test_jobs_overlap_but_results_keep_semantic_order(self):
        from sandboxai.evaluation import run_parallel_evaluations

        barrier = threading.Barrier(2, timeout=2.0)

        def job(name):
            barrier.wait()
            time.sleep(0.04)
            return name

        normal, battery, timing = run_parallel_evaluations(
            lambda: job("normal"), lambda: job("battery")
        )
        self.assertEqual((normal, battery), ("normal", "battery"))
        self.assertGreater(timing["overlap_seconds"], 0.0)
        self.assertLess(timing["wall_seconds"], timing["normal_seconds"] + timing["battery_seconds"])

    def test_synchronized_model_never_enters_predict_concurrently(self):
        from sandboxai.evaluation import SynchronizedModel, run_parallel_evaluations

        class _Model:
            def __init__(self):
                self.active = 0
                self.max_active = 0

            def predict(self, value, deterministic=True):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                self.active -= 1
                return value

            def save(self, path):
                return path

        model = _Model()
        synchronized = SynchronizedModel(model)
        normal, battery, _timing = run_parallel_evaluations(
            lambda: synchronized.predict("normal"),
            lambda: synchronized.predict("battery"),
        )
        self.assertEqual((normal, battery), ("normal", "battery"))
        self.assertEqual(model.max_active, 1)


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
@unittest.skipUnless(HAS_SB3, SB3_REASON)
class TrainingEvaluationProcessReuseTest(unittest.TestCase):
    """The training loop must keep its evaluation bridge processes alive
    across evaluation boundaries instead of respawning Godot per boundary,
    and the reused processes must not change any evaluation result."""

    def setUp(self):
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            self.skipTest("tensorboard is not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.spawn_log = self.tmp / "spawns.log"
        self.executable = _write_fake_bridge(
            self.tmp, deterministic_bridge_source(str(self.spawn_log))
        )

    def _config(self, **overrides):
        from sandboxai.config import TrainingConfig

        values = dict(
            environment_count=2,
            enemy_count=1,
            rollout_length=16,
            batch_size=16,
            total_training_steps=96,
            checkpoint_frequency=10_000,
            evaluation_frequency=16,  # boundary every rollout iteration
            evaluation_episodes=3,
            condition_eval_episodes=6,
            seed=1234,
            device="cpu",
            godot_executable=self.executable,
            project_path=str(PROJECT_ROOT),
            output_root=str(self.tmp),
            run_id="perf_regression",
            net_arch=(32, 32),
            skill_metrics=False,
            episode_log=False,
            replay_mode="off",
            profile_training=True,
        )
        values.update(overrides)
        return TrainingConfig(**values).validate()

    def _boundaries(self, run_dir: Path) -> int:
        return len([p for p in (run_dir / "evaluations").glob("step_*") if p.is_dir()])

    def test_bridge_processes_spawn_once_for_the_whole_run(self):
        from sandboxai.ppo import train_ppo

        result = train_ppo(self._config())
        run_dir = Path(result["run_dir"])
        boundaries = self._boundaries(run_dir)
        self.assertGreaterEqual(boundaries, 3, "test needs several evaluation boundaries")
        spawns = self.spawn_log.read_text().splitlines()
        # 1 training bridge + 1 normal-eval bridge + 1 battery bridge, once,
        # plus one live Godot version probe for the run manifest (also
        # once: TrainingPipeline.manifest() caches the first probe and
        # reuses it for the final manifest rebuild instead of launching
        # Godot again just to re-read the same version string).
        # Before process reuse this was 1 + 2 * boundaries (+ 2 probes,
        # one per manifest rebuild, before the probe was cached).
        self.assertEqual(len(spawns), 4, f"expected 4 bridge spawns, got {len(spawns)}")
        profile = json.loads(Path(result["training_profile"]).read_text(encoding="utf-8"))
        self.assertEqual(profile["counters"]["eval.bridge_spawns"], 2)
        self.assertEqual(profile["timings"]["eval.env_startup"]["count"], 2)

    def test_reused_processes_preserve_evaluation_semantics(self):
        from sandboxai.ppo import train_ppo

        # Two identical runs must produce identical evaluation artifacts:
        # reuse must not leak state between boundaries (seeds, metrics,
        # termination) and must not break seeded reproducibility.
        first = train_ppo(self._config())
        second = train_ppo(self._config(run_id="perf_regression_2"))
        for relative in ("evaluations/latest.json",):
            a = json.loads((Path(first["run_dir"]) / relative).read_text(encoding="utf-8"))
            b = json.loads((Path(second["run_dir"]) / relative).read_text(encoding="utf-8"))
            for key in ("episodes", "mean_episode_reward", "mean_win", "mean_loss",
                        "mean_truncated", "mean_kills", "mean_episode_length", "win_rate"):
                self.assertEqual(a.get(key), b.get(key), f"latest.json[{key}] diverged between runs")
        # Identical trained weights: the training path itself is untouched.
        from stable_baselines3 import PPO

        model_a = PPO.load(str(Path(first["run_dir"]) / "final.zip"), device="cpu")
        model_b = PPO.load(str(Path(second["run_dir"]) / "final.zip"), device="cpu")
        for (name_a, tensor_a), (name_b, tensor_b) in zip(
            model_a.policy.state_dict().items(), model_b.policy.state_dict().items()
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue((tensor_a == tensor_b).all(), f"weights differ: {name_a}")

    def test_vectorized_normal_evaluation_also_reuses_and_reproduces(self):
        from sandboxai.ppo import train_ppo

        # evaluation_environment_count > 1 routes the normal evaluation
        # through the vectorised bridge; the reused process must re-seed
        # exactly and produce reproducible artifacts across two runs.
        first = train_ppo(self._config(evaluation_environment_count=2))
        second = train_ppo(self._config(evaluation_environment_count=2, run_id="perf_regression_vec2"))
        spawns = self.spawn_log.read_text().splitlines()
        # Two runs x (1 training + 1 version probe + 1 normal-eval + 1
        # battery) bridges; the vectorised normal evaluation must reuse its
        # process across the 6 boundaries of each run rather than
        # respawning per boundary. See the version-probe comment above.
        self.assertEqual(
            len(spawns), 8, f"expected 8 bridge spawns (2 runs x 4), got {len(spawns)}"
        )
        for key in ("episodes", "mean_episode_reward", "mean_win", "win_rate"):
            a = json.loads((Path(first["run_dir"]) / "evaluations/latest.json").read_text(encoding="utf-8"))
            b = json.loads((Path(second["run_dir"]) / "evaluations/latest.json").read_text(encoding="utf-8"))
            self.assertEqual(a.get(key), b.get(key), f"latest.json[{key}] diverged between identical runs")

    def test_parallel_path_preserves_best_selection_and_reward_stop(self):
        from sandboxai.ppo import train_ppo

        result = train_ppo(
            self._config(
                total_training_steps=160,
                min_eval_reward=-1_000_000.0,
                early_stopping_patience=5,
                run_id="parallel_selection_stop",
            )
        )
        run_dir = Path(result["run_dir"])
        step_dirs = sorted(path for path in (run_dir / "evaluations").glob("step_*") if path.is_dir())
        self.assertEqual(len(step_dirs), 1, "reward threshold must stop after the first joined boundary")
        best = json.loads((run_dir / "evaluations" / "best.json").read_text(encoding="utf-8"))
        latest = json.loads((run_dir / "evaluations" / "latest.json").read_text(encoding="utf-8"))
        report = json.loads((step_dirs[0] / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(best["mean_reward"], latest["mean_episode_reward"])
        self.assertEqual(best["timesteps"], latest["timesteps"])
        # The selection rule is recorded next to the score it produced, so
        # a later resume can tell whether the two are comparable.
        self.assertEqual(
            best["selection_rule"],
            {"metric": "mean_episode_reward", "goal": "max", "min_delta": 0.0},
        )
        self.assertEqual(best["score"], best["mean_reward"])
        self.assertEqual(
            report["normal_evaluation"]["mean_episode_reward"],
            latest["mean_episode_reward"],
        )
        self.assertTrue((run_dir / "checkpoints" / "best_eval.zip").is_file())
        self.assertTrue((step_dirs[0] / "normal_episodes.csv").is_file())
        self.assertLess(result["timesteps"], 160)

    def test_parallel_path_preserves_patience_and_first_best_checkpoint(self):
        from sandboxai.ppo import train_ppo

        result = train_ppo(
            self._config(
                total_training_steps=160,
                early_stopping_patience=1,
                min_eval_reward=None,
                run_id="parallel_patience_stop",
            )
        )
        run_dir = Path(result["run_dir"])
        step_dirs = sorted(path for path in (run_dir / "evaluations").glob("step_*") if path.is_dir())
        self.assertEqual(len(step_dirs), 2, "first tie must consume one patience check")
        first = json.loads((step_dirs[0] / "summary.json").read_text(encoding="utf-8"))
        second = json.loads((step_dirs[1] / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(first["mean_episode_reward"], second["mean_episode_reward"])
        best = json.loads((run_dir / "evaluations" / "best.json").read_text(encoding="utf-8"))
        self.assertEqual(best["timesteps"], first["timesteps"])
        self.assertLess(result["timesteps"], 160)

    def test_profile_reports_evaluation_breakdown(self):
        from sandboxai.ppo import train_ppo

        result = train_ppo(self._config())
        profile = json.loads(Path(result["training_profile"]).read_text(encoding="utf-8"))
        timings = profile["timings"]
        for bucket in (
            "callback.evaluation",
            "eval.env_startup",
            "eval.normal.total",
            "eval.normal.predict",
            "eval.normal.env_step",
            "eval.battery.total",
            "eval.battery.execution",
            "eval.battery.predict",
            "eval.battery.env_step",
            "eval.battery.bridge.step.total",
            "eval.normal.bridge.step.total",
            "eval.parallel.wall",
            "eval.parallel.overlap",
        ):
            self.assertIn(bucket, timings, f"missing profiling bucket {bucket}")
            self.assertGreater(timings[bucket]["total_seconds"], 0.0, bucket)
        # The training bridge's step bucket must not include evaluation steps:
        # 96 steps / 2 envs = 48 vector steps for the training bridge only.
        self.assertEqual(timings["bridge.step.total"]["count"], 48)
        # Three requested 8-step episodes run in one planned vector wave.
        # The configured maximum is capped to the requested episode count so
        # no idle slots are simulated.
        expected_normal_steps = 3 * 8 * self._boundaries(Path(result["run_dir"]))
        self.assertEqual(profile["counters"]["eval.normal.env_steps"], expected_normal_steps)
        latest = json.loads(
            (Path(result["run_dir"]) / "evaluations" / "latest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(latest["environment_count"], 3)


class CheckpointBatteryBatchingTest(unittest.TestCase):
    """The battery's environment count is a speed knob, not a semantics
    knob: planned episodes are result-invariant under scheduling."""

    @staticmethod
    def _model():
        class _Model:
            def save(self, path):
                Path(path).write_bytes(b"fake")

        return _Model()

    @staticmethod
    def _config(checkpoint_env_count: int):
        class _Config:
            seed = 5
            project = Path("/tmp/proj")
            godot_executable = "godot"
            enemy_count = 1
            curriculum_level = 3
            checkpoint_eval_environment_count = checkpoint_env_count
            checkpoint_condition_eval = True
            checkpoint_generalization_eval = False
            checkpoint_league_eval = False
            league_matches_per_checkpoint = 0
            condition_eval_episodes = 1
            generalization_episodes_per_cell = 1
            replay_mode = "off"

        return _Config()

    @staticmethod
    def _pipeline():
        class _Pipeline:
            policy_id = "p"
            run_dir = Path("/tmp/run")

            class skill_metrics:
                @staticmethod
                def flush_aggregate():
                    return {}

            class driver:
                used_conditions = []
                level = 5

                class tracker:
                    @staticmethod
                    def report():
                        return {}

                @staticmethod
                def curriculum_snapshot():
                    return {}

            class replays:
                saved = 0

        return _Pipeline()

    def test_battery_env_count_flows_to_the_executor(self):
        from sandboxai.checkpoint_eval import run_checkpoint_evaluation

        captured = {}

        class _FakeExecutor:
            def __init__(self, env_kwargs, skill_metrics=True):
                captured["env_kwargs"] = env_kwargs
                self.environment_count = int(env_kwargs["environment_count"])
                self.skill_metrics_enabled = bool(skill_metrics)
                self.client = None
                self.closed = False

            def run(self, *args, **kwargs):
                return []

            def close(self):
                self.closed = True

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run_checkpoint_evaluation(
                model=self._model(),
                step=1,
                config=self._config(5),
                pipeline=self._pipeline(),
                output_dir=tmp,
                executor_factory=_FakeExecutor,
            )
        self.assertEqual(captured["env_kwargs"]["environment_count"], 5)

    def test_condition_and_generalization_share_one_executor_run(self):
        from sandboxai.checkpoint_eval import run_checkpoint_evaluation

        calls = []

        class _CombinedExecutor:
            def run(self, model, plans, **kwargs):
                calls.append(list(plans))
                return [
                    {
                        "labels": dict(plan.labels),
                        "condition": plan.condition.to_dict(),
                        "seed": plan.condition.seed,
                        "environment_index": index % 3,
                        "episode_reward": float(index),
                        "episode_length": 2,
                        "win": index % 2 == 0,
                    }
                    for index, plan in enumerate(plans)
                ]

            def close(self):
                raise AssertionError("provided executor must not be closed")

        config = self._config(3)
        config.checkpoint_generalization_eval = True
        pipeline = self._pipeline()
        pipeline.driver.used_conditions = [
            {
                "seed": 41,
                "map_id": "open_field",
                "scenario": "cover_fight",
                "lighting": "normal",
                "enemy_count": 1,
                "level": 5,
            }
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report = run_checkpoint_evaluation(
                model=self._model(),
                step=1,
                config=config,
                pipeline=pipeline,
                output_dir=tmp,
                executor=_CombinedExecutor(),
            )
        self.assertEqual(len(calls), 1, "both planned sections must share one vector tail")
        self.assertEqual(len(report["condition_evaluation"]["episodes_detail"]), 1)
        self.assertGreater(report["generalization"]["episodes"], 0)
        # Both sections use local _position values starting at zero; scheduler
        # order must still keep the condition row first.
        self.assertEqual(report["condition_evaluation"]["episodes_detail"][0]["episode_reward"], 0.0)

    def test_provided_executor_is_reused_and_not_closed(self):
        from sandboxai.checkpoint_eval import run_checkpoint_evaluation

        calls = {"run": 0}

        class _FakeExecutor:
            def __init__(self):
                self.environment_count = 1
                self.skill_metrics_enabled = True
                self.client = None
                self.closed = False

            def run(self, *args, **kwargs):
                calls["run"] += 1
                return []

            def close(self):
                self.closed = True

        executor = _FakeExecutor()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                run_checkpoint_evaluation(
                    model=self._model(),
                    step=1,
                    config=self._config(1),
                    pipeline=self._pipeline(),
                    output_dir=tmp,
                    executor=executor,
                )
        self.assertEqual(calls["run"], 2)
        self.assertFalse(executor.closed, "a reused executor must not be closed by the callee")

    def test_config_validates_new_fields(self):
        from sandboxai.config import TrainingConfig

        TrainingConfig().validate()  # defaults are valid
        with self.assertRaises(ValueError):
            TrainingConfig(checkpoint_eval_environment_count=0).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(inference_device="tpu").validate()

    def test_resolved_inference_device(self):
        from sandboxai.config import TrainingConfig

        self.assertEqual(TrainingConfig(device="cpu").resolved_inference_device(), "cpu")
        self.assertEqual(
            TrainingConfig(device="cpu", inference_device="cpu").resolved_inference_device(),
            "cpu",
        )
        try:
            TrainingConfig(device="cpu", inference_device="cuda").resolved_inference_device()
        except RuntimeError:
            pass  # no CUDA in this environment: the request must fail loudly
        else:
            import torch

            if not torch.cuda.is_available():
                self.fail("cuda inference_device must raise without CUDA")


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class InferenceDeviceSchedulerTest(unittest.TestCase):
    """The opt-in inference-device split: rollout inference may run on a
    different device than the PPO update, switched exactly at rollout
    boundaries. Inactive configurations must be no-ops."""

    def _scheduler(self, training: str, inference: str, policy):
        from sandboxai.ppo import InferenceDeviceScheduler

        class _Model:
            def __init__(self):
                self.device = training
                self.policy = policy

        model = _Model()
        return InferenceDeviceScheduler(model, training, inference), model

    def test_inactive_when_devices_match(self):
        policy_moves = []

        class _Policy:
            def to(self, device):
                policy_moves.append(device)

        scheduler, model = self._scheduler("cpu", "cpu", _Policy())
        self.assertFalse(scheduler.active)
        scheduler.enter_rollout()
        scheduler.exit_rollout()
        self.assertEqual(policy_moves, [])
        self.assertEqual(model.device, "cpu")

    def test_switches_policy_and_model_device_at_boundaries(self):
        policy_moves = []

        class _Policy:
            def to(self, device):
                policy_moves.append(str(device))

        scheduler, model = self._scheduler("cuda", "cpu", _Policy())
        self.assertTrue(scheduler.active)
        scheduler.enter_rollout()
        self.assertEqual(policy_moves, ["cpu"])
        self.assertEqual(str(model.device), "cpu")  # predict/collect_rollouts follow
        scheduler.exit_rollout()
        self.assertEqual(policy_moves, ["cpu", "cuda"])
        self.assertEqual(str(model.device), "cuda")  # train() sees the training device
        # Idempotence: double-exit keeps the policy on the training device.
        scheduler.exit_rollout()
        self.assertEqual(policy_moves, ["cpu", "cuda"])

    def test_training_device_context_restores_rollout_placement(self):
        policy_moves = []

        class _Policy:
            def to(self, device):
                policy_moves.append(str(device))

        scheduler, model = self._scheduler("cuda", "cpu", _Policy())
        scheduler.enter_rollout()
        with scheduler.training_device_context():
            self.assertEqual(str(model.device), "cuda")  # e.g. mid-rollout save
        # rollout placement restored after the context: cpu -> cuda (save
        # placement) -> cpu (back to inference).
        self.assertEqual(policy_moves, ["cpu", "cuda", "cpu"])
        self.assertEqual(str(model.device), "cpu")
        scheduler.exit_rollout()
        self.assertEqual(policy_moves, ["cpu", "cuda", "cpu", "cuda"])

    def test_training_runs_unchanged_with_coupled_cpu_devices(self):
        # End-to-end smoke: inference_device="cpu" with device="cpu" must be
        # exactly the historical coupled behavior (no switching, no error).
        if not HAS_SB3 or os.name != "posix":
            self.skipTest("requires SB3 and a POSIX fake bridge")
        import tempfile

        from sandboxai.config import TrainingConfig
        from sandboxai.ppo import train_ppo

        with tempfile.TemporaryDirectory() as tmp:
            executable = _write_fake_bridge(Path(tmp), deterministic_bridge_source())
            config = TrainingConfig(
                environment_count=2,
                rollout_length=16,
                batch_size=16,
                total_training_steps=32,
                checkpoint_frequency=10_000,
                evaluation_frequency=10_000,
                seed=99,
                device="cpu",
                inference_device="cpu",
                godot_executable=executable,
                project_path=str(PROJECT_ROOT),
                output_root=tmp,
                run_id="inference_device_smoke",
                net_arch=(16, 16),
                skill_metrics=False,
                episode_log=False,
                replay_mode="off",
            ).validate()
            result = train_ppo(config)
            self.assertTrue(Path(result["run_dir"], "final.zip").is_file())


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
@unittest.skipUnless(HAS_SB3, SB3_REASON)
class CompactBatteryInfosTest(unittest.TestCase):
    """The battery bridge may request compact infos (payload reduction):
    events and terminal metrics - everything the battery consumes - must be
    identical, which this test proves by comparing rows from a bridge that
    honors compact_infos both ways."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.executable = _write_fake_bridge(self.tmp, deterministic_bridge_source())

    def test_battery_rows_identical_under_compact_and_full_infos(self):
        from sandboxai.checkpoint_eval import PlanExecutor
        from sandboxai.conditions import Condition
        from sandboxai.randomization import EpisodePlan  # noqa: F401

        def plans(count):
            from sandboxai.checkpoint_eval import PlannedEpisode

            return [
                PlannedEpisode(
                    condition=Condition(
                        map_id="", scenario="", lighting="",
                        enemy_count=1, level=3, seed=5000 + i * 13,
                    ),
                    labels={"axis": "test", "_position": i},
                )
                for i in range(count)
            ]

        from stable_baselines3 import PPO
        from sandboxai.godot_env import GodotVecEnv

        env = GodotVecEnv(
            project_path=str(PROJECT_ROOT), godot_executable=self.executable,
            environment_count=2, enemy_count=1, seed=1, curriculum_level=3,
        )
        try:
            model = PPO(
                "MlpPolicy", env, n_steps=8, batch_size=16, n_epochs=1, seed=3,
                device="cpu", verbose=0,
                policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}},
            )
        finally:
            env.close()

        def kwargs(compact):
            return {
                "project_path": str(PROJECT_ROOT),
                "godot_executable": self.executable,
                "environment_count": 2,
                "enemy_count": 1,
                "seed": 1,
                "curriculum_level": 3,
                "compact_infos": compact,
            }

        full = PlanExecutor(kwargs(False)).run(model, plans(4), policy_id="p")
        compact = PlanExecutor(kwargs(True)).run(model, plans(4), policy_id="p")
        strip = lambda rows: [
            {k: v for k, v in row.items() if k != "environment_index"} for row in rows
        ]
        self.assertEqual(strip(full), strip(compact))


@unittest.skipUnless(
    os.environ.get("SANDBOXAI_PERF_BENCH") == "1",
    "opt-in wall-clock benchmark (SANDBOXAI_PERF_BENCH=1)",
)
@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
@unittest.skipUnless(HAS_SB3, SB3_REASON)
class EvalPathBenchmark(unittest.TestCase):
    """Wall-clock before/after benchmarks against the fake bridge.

    The fake bridge's spawn cost (a Python interpreter start) is a LOWER
    BOUND for a real Godot process, which additionally loads the engine and
    the project; the spawn-count reduction it demonstrates is the same on
    the real engine, multiplied by its much higher startup cost.
    """

    def setUp(self):
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            self.skipTest("tensorboard is not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.spawn_log = self.tmp / "spawns.log"
        self.executable = _write_fake_bridge(
            self.tmp, deterministic_bridge_source(str(self.spawn_log))
        )

    def test_evaluation_boundary_cost(self):
        import time

        from sandboxai.config import TrainingConfig
        from sandboxai.ppo import train_ppo

        config = TrainingConfig(
            environment_count=2,
            rollout_length=16,
            batch_size=16,
            total_training_steps=96,
            checkpoint_frequency=10_000,
            evaluation_frequency=16,
            evaluation_episodes=4,
            condition_eval_episodes=6,
            seed=1234,
            device="cpu",
            godot_executable=self.executable,
            project_path=str(PROJECT_ROOT),
            output_root=str(self.tmp),
            run_id="bench",
            net_arch=(32, 32),
            skill_metrics=False,
            episode_log=False,
            replay_mode="off",
            profile_training=True,
        ).validate()
        started = time.perf_counter()
        result = train_ppo(config)
        total = time.perf_counter() - started
        profile = json.loads(Path(result["training_profile"]).read_text(encoding="utf-8"))
        boundaries = len([p for p in (Path(result["run_dir"]) / "evaluations").glob("step_*") if p.is_dir()])
        report = {
            "total_wall_seconds": round(total, 3),
            "boundaries": boundaries,
            "bridge_spawns": len(self.spawn_log.read_text().splitlines()),
            "callback_evaluation_seconds": round(profile["phase_totals_seconds"]["callback.evaluation"], 3),
            "eval_env_startup_seconds": round(profile["phase_totals_seconds"]["eval.env_startup"], 3),
            "eval_predict_seconds": round(
                profile["phase_totals_seconds"]["eval.normal.predict"]
                + profile["phase_totals_seconds"]["eval.battery.predict"], 3),
            "eval_env_step_seconds": round(
                profile["phase_totals_seconds"]["eval.normal.env_step"]
                + profile["phase_totals_seconds"]["eval.battery.env_step"], 3),
        }
        report["seconds_per_boundary"] = round(report["callback_evaluation_seconds"] / max(1, boundaries), 3)
        print(json.dumps({"eval_path_benchmark": report}, indent=2))


if __name__ == "__main__":
    unittest.main()
