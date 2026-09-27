"""End-to-end PPO smoke test against a scripted fake Godot bridge.

No real Godot binary is required: a tiny Python subprocess speaks the same
JSON-lines protocol as scripts/rl/rl_server.gd (see test_godot_env.py for the
same pattern). This validates that the Python PPO/Gymnasium wiring stays
correct for the NEW 33-field multi-enemy observation contract — the
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

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.godot_env import GodotVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BRIDGE_SOURCE = r'''
import json, random, sys

OBS_DIM = {obs_dim}
random.seed(0)

def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

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
            "action_space": {{"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2], "dimension": 5}},
            "observation_space": {{"type": "structured_float_vector", "size": OBS_DIM,
                                   "shape": [OBS_DIM], "low": -1.0, "high": 1.0}},
        }})
    elif command == "reset":
        n = len(request.get("indices", [])) if "indices" in request else 2
        step_counts = {{i: 0 for i in range(n)}}
        out({{"ok": True, "observations": [make_obs() for _ in range(n)],
             "infos": [{{}} for _ in range(n)]}})
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
        wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}'\n", encoding="utf-8")
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
            self.assertEqual(action.shape, (2, 5))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
