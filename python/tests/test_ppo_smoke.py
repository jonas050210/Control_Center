"""End-to-end PPO smoke test against a scripted fake Godot bridge.

No real Godot binary is required: a tiny Python subprocess speaks the same
JSON-lines protocol as scripts/rl/rl_server.gd (see test_godot_env.py for the
same pattern). This validates that the Python PPO/Gymnasium wiring stays
correct for the current multi-enemy observation contract — the
dimension is read entirely from the bridge's "spaces" response, never
hardcoded on the Python side, so this test only needs to change OBS_DIM here
for the assertion to hold across future observation-contract changes.

This is the "small smoke training run" required by the milestone: it proves
the PPO pipeline still trains (loss/policy update runs without crashing) at
the new observation size, without spending real time on a full Godot-backed
500k-step run.
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.godot_env import GodotVecEnv
from sandboxai.manifest import MANIFEST_FORMAT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BRIDGE_SOURCE = r'''
import json, random, sys

OBS_DIM = {obs_dim}
random.seed(0)


def env_count_from_argv(default=2):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--env-count" and i + 1 < len(argv):
            try:
                return max(1, int(argv[i + 1]))
            except ValueError:
                return default
    return default


ENV_COUNT = env_count_from_argv()

def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

staged_plans = []

def make_obs():
    return [random.uniform(-1.0, 1.0) for _ in range(OBS_DIM)]

step_counts = {{}}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        out({{
            "ok": True,
            "action_space": {{"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2, 2], "dimension": 6}},
            "observation_space": {{"type": "structured_float_vector", "size": OBS_DIM,
                                   "shape": [OBS_DIM], "low": -1.0, "high": 1.0}},
        }})
    elif command == "reset":
        step_counts = {{i: 0 for i in range(ENV_COUNT)}}
        out({{"ok": True, "observations": [make_obs() for _ in range(ENV_COUNT)],
             "infos": [{{}} for _ in range(ENV_COUNT)]}})
    elif command == "reset_indices":
        indices = request.get("indices", [])
        for i in indices:
            step_counts[i] = 0
        out({{"ok": True, "results": [{{"index": i, "observation": make_obs()}} for i in indices]}})
    elif command == "set_episode_plans":
        plans = request.get("plans", [])
        staged_plans.clear()
        staged_plans.extend(plans)
        out({{"ok": True, "staged": [int(p.get("index", -1)) for p in plans]}})
    elif command == "episode_conditions":
        out({{"ok": True, "conditions": staged_plans}})
    elif command == "step":
        actions = request.get("actions", [])
        n = len(actions)
        observations, rewards, dones, infos = [], [], [], []
        for i in range(n):
            step_counts[i] = step_counts.get(i, 0) + 1
            done = step_counts[i] >= 8
            reward = 1.0 if actions[i][4] == 1 else 0.01
            info = {{"done_reason": "timeout" if done else "", "metrics": {{"episode_reward": reward}}}}
            if done:
                info["terminal_observation"] = make_obs()
                info["TimeLimit.truncated"] = True
                step_counts[i] = 0
            observations.append(make_obs())
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        out({{"ok": True, "observations": observations, "rewards": rewards,
             "dones": dones, "infos": infos}})
    elif command == "ping":
        out({{"ok": True, "pong": True}})
    elif command == "health_check":
        out({{"ok": True, "health": [{{"healthy": True}} for _ in range(len(step_counts))]}})
    elif command == "close":
        out({{"ok": True, "close": True}})
        break
    else:
        out({{"ok": False, "error": "unknown command"}})
'''.format(obs_dim=OBSERVATION_FIELD_COUNT)


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class PPOSmokeTest(unittest.TestCase):
    def setUp(self):
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            self.skipTest("stable-baselines3 is not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        bridge_py = tmp / "fake_bridge.py"
        bridge_py.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        wrapper = tmp / "fake_godot"
        wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}' \"$@\"\n", encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.executable = str(wrapper)

    def test_ppo_trains_a_few_steps_against_the_new_observation_contract(self):
        from stable_baselines3 import PPO  # type: ignore

        env = GodotVecEnv(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=2,
            enemy_count=2,
            seed=1234,
            curriculum_level=4,
        )
        try:
            self.assertEqual(env.observation_space.shape, (OBSERVATION_FIELD_COUNT,))
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=16,
                batch_size=16,
                n_epochs=1,
                policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
                device="cpu",
                verbose=0,
            )
            model.learn(total_timesteps=64)
            observation = env.reset()
            action, _ = model.predict(observation, deterministic=True)
            self.assertEqual(action.shape, (2, len(ACTION_NVEC)))
        finally:
            env.close()

    def test_vec_env_launches_the_configured_executable(self):
        # Regression (Windows WinError 2): the executable passed to
        # GodotVecEnv must be the binary GodotProcessTransport actually
        # launches — never a silent fallback to bare `godot` from PATH.
        env = GodotVecEnv(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=2,
            seed=1234,
            curriculum_level=4,
        )
        try:
            self.assertEqual(env.client.transport.executable, self.executable)
            self.assertEqual(env.num_envs, 2)
            self.assertTrue(env.client.ping().get("pong"))
        finally:
            env.close()


class EnvKwargsPropagationTests(unittest.TestCase):
    """TrainingConfig values must flow into the training env construction."""

    def test_env_kwargs_carry_the_configured_godot_executable(self):
        from sandboxai.config import TrainingConfig
        from sandboxai.ppo import _env_kwargs

        config = TrainingConfig(
            godot_executable="/opt/godot/Godot_v4.7.2-stable_win64_console.exe",
            environment_count=3,
            enemy_count=2,
            seed=77,
            curriculum_level=5,
        ).validate()
        kwargs = _env_kwargs(config)
        self.assertEqual(kwargs["godot_executable"], "/opt/godot/Godot_v4.7.2-stable_win64_console.exe")
        self.assertEqual(kwargs["environment_count"], 3)
        self.assertEqual(kwargs["enemy_count"], 2)
        self.assertEqual(kwargs["seed"], 77)
        self.assertEqual(kwargs["curriculum_level"], 5)


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class PPOTrainingWorkflowTests(unittest.TestCase):
    """End-to-end train_ppo() coverage against the fake bridge.

    Exercises the full orchestration that the plain PPO smoke test does not:
    run-directory layout, config/warm-start artifacts, periodic and latest
    checkpoints, the evaluation callback (which spawns its own bridge
    process), and checkpoint resume — including the run-directory detection
    for a checkpoint stored in the run root (final.zip).
    """

    def setUp(self):
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            self.skipTest("stable-baselines3 is not installed")
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            self.skipTest("tensorboard is not installed (train_ppo logs to TensorBoard by default)")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        bridge_py = tmp / "fake_bridge.py"
        bridge_py.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        wrapper = tmp / "fake_godot"
        wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}' \"$@\"\n", encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.executable = str(wrapper)

    def _config(self, **overrides):
        from sandboxai.config import TrainingConfig

        values = dict(
            environment_count=2,
            enemy_count=2,
            rollout_length=16,
            batch_size=16,
            total_training_steps=64,
            checkpoint_frequency=32,
            evaluation_frequency=32,
            evaluation_episodes=2,
            seed=1234,
            device="cpu",
            curriculum_level=4,
            godot_executable=self.executable,
            project_path=str(PROJECT_ROOT),
            output_root=str(Path(self._tmp.name)),
            run_id="workflow_test",
            net_arch=(32, 32),
        )
        values.update(overrides)
        return TrainingConfig(**values).validate()

    def test_train_ppo_produces_complete_run_layout(self):
        from sandboxai.ppo import train_ppo

        result = train_ppo(self._config())
        run_dir = Path(result["run_dir"])
        self.assertTrue(run_dir.is_dir())
        for relative in (
            "config.json",
            "warm_start.json",
            "run_summary.json",
            "final.zip",
            "checkpoints/latest.zip",
            "logs/training.jsonl",
        ):
            self.assertTrue((run_dir / relative).is_file(), f"missing run artifact: {relative}")
        # The evaluation callback must have run at least once and produced
        # its summaries plus a best-evaluation checkpoint.
        self.assertTrue((run_dir / "evaluations" / "latest.json").is_file())
        self.assertTrue((run_dir / "checkpoints" / "best_eval.zip").is_file())
        self.assertTrue(result["latest_checkpoint"].endswith("latest.zip"))
        # Integrated pipeline artifacts (curriculum_mode "auto", default):
        # manifest, per-episode log, resume-exact curriculum state, and the
        # checkpoint-time battery report.
        for relative in (
            "run_manifest.json",
            "checkpoints/curriculum_state.json",
            "logs/episodes.jsonl",
        ):
            self.assertTrue((run_dir / relative).is_file(), f"missing pipeline artifact: {relative}")
        import json

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("format"), MANIFEST_FORMAT)
        self.assertEqual(manifest.get("seed"), 1234)
        reports = list((run_dir / "evaluations").glob("step_*/report.json"))
        self.assertTrue(reports, "no checkpoint battery report was produced")
        battery = json.loads(reports[-1].read_text(encoding="utf-8"))
        self.assertIn("condition_evaluation", battery)
        self.assertIn("curriculum", battery)
        episodes = (run_dir / "logs" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(episodes, "episodes.jsonl must contain at least one episode row")
        first_row = json.loads(episodes[0])
        self.assertIn("plan", first_row)
        self.assertIn("engine_metrics", first_row)

    def test_profiled_training_writes_rollout_update_and_bridge_breakdown(self):
        import json
        from sandboxai.ppo import train_ppo

        result = train_ppo(
            self._config(
                total_training_steps=32,
                checkpoint_frequency=10_000,
                evaluation_frequency=10_000,
                curriculum_mode="fixed",
                profile_training=True,
            )
        )
        profile_path = Path(result["training_profile"])
        self.assertTrue(profile_path.is_file())
        report = json.loads(profile_path.read_text(encoding="utf-8"))
        self.assertEqual(report["format"], "sandboxai.training_profile/v1")
        self.assertEqual(report["metadata"]["observation_floats"], OBSERVATION_FIELD_COUNT)
        self.assertEqual(report["metadata"]["action_components"], len(ACTION_NVEC))
        self.assertGreater(report["timings"]["ppo.rollout_collection"]["total_seconds"], 0.0)
        self.assertGreater(report["timings"]["ppo.policy_update"]["total_seconds"], 0.0)
        self.assertEqual(report["timings"]["bridge.step.total"]["count"], 16)
        self.assertEqual(len(report["iterations"]), 1)

    def test_resume_from_latest_zip_stays_in_the_same_run_directory(self):
        from sandboxai.ppo import train_ppo

        first = train_ppo(self._config())
        run_dir = Path(first["run_dir"])
        latest = run_dir / "checkpoints" / "latest.zip"
        second = train_ppo(self._config(total_training_steps=96), resume_checkpoint=latest)
        self.assertEqual(Path(second["run_dir"]), run_dir)

    def test_resume_from_final_zip_does_not_escape_the_run_directory(self):
        # Regression: a checkpoint stored in the run ROOT (final.zip) used to
        # resolve run_dir to the run's PARENT, scattering checkpoints/logs
        # into unrelated directories.
        from sandboxai.ppo import train_ppo

        first = train_ppo(self._config())
        run_dir = Path(first["run_dir"])
        second = train_ppo(self._config(total_training_steps=96), resume_checkpoint=run_dir / "final.zip")
        self.assertEqual(Path(second["run_dir"]), run_dir)
        self.assertFalse((run_dir.parent / "checkpoints").exists())

    def test_resume_with_missing_checkpoint_fails_loudly(self):
        from sandboxai.ppo import train_ppo

        with self.assertRaises(FileNotFoundError):
            train_ppo(self._config(), resume_checkpoint=Path(self._tmp.name) / "does_not_exist.zip")


if __name__ == "__main__":
    unittest.main()
