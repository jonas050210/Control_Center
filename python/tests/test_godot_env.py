"""Transport/adapter regression tests against a scripted fake Godot bridge.

These tests exercise the Python side of the JSON-lines protocol without a
real Godot binary: a small Python subprocess speaks the same protocol as
scripts/rl/rl_server.gd. POSIX-only because the fake executable relies on a
shebang wrapper.
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.godot_env import GodotProcessTransport, GodotGymEnv, GodotVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BRIDGE_SOURCE = r'''
import json, os, sys

OBS_DIM = __OBS_DIM__

def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

flood = int(os.environ.get("FAKE_BRIDGE_STDERR_FLOOD", "0"))
if flood:
    line = "warning spam\n"
    for _ in range(flood // len(line) + 1):
        sys.stderr.write(line)
    sys.stderr.flush()
silent_after_spaces = os.environ.get("FAKE_BRIDGE_SILENT", "") == "1"

step_count = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        out({
            "ok": True,
            "action_space": {"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2], "dimension": 5},
            "observation_space": {"type": "structured_float_vector", "size": OBS_DIM,
                                   "shape": [OBS_DIM], "low": -1.0, "high": 1.0},
        })
        continue
    if silent_after_spaces:
        continue
    if command == "reset":
        step_count = 0
        out({"ok": True, "observations": [[0.0] * OBS_DIM], "infos": [{"seed": request.get("seed")}]})
    elif command == "step":
        step_count += 1
        done = step_count >= 3
        info = {"done_reason": "timeout" if done else "", "metrics": {}}
        if done:
            info["terminal_observation"] = [0.5] * OBS_DIM
            info["TimeLimit.truncated"] = True
            step_count = 0
        out({"ok": True, "observations": [[0.25] * OBS_DIM], "rewards": [1.0],
             "dones": [done], "infos": [info]})
    elif command == "ping":
        out({"ok": True, "pong": True})
    elif command == "close":
        out({"ok": True, "close": True})
        break
    else:
        out({"ok": False, "error": "unknown command"})
'''.replace("__OBS_DIM__", str(OBSERVATION_FIELD_COUNT))


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class FakeBridgeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        bridge_py = tmp / "fake_bridge.py"
        bridge_py.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        wrapper = tmp / "fake_godot"
        wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}'\n", encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.executable = str(wrapper)

    def make_transport(self, timeout=10.0):
        return GodotProcessTransport(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=1,
            request_timeout=timeout,
        )

    def test_roundtrip_and_spaces(self):
        with self.make_transport() as transport:
            self.assertEqual(transport.spaces["observation_space"]["size"], OBSERVATION_FIELD_COUNT)
            self.assertTrue(transport.request({"cmd": "ping"}).get("pong"))

    def test_large_stderr_output_does_not_deadlock(self):
        # 512 KiB of stderr comfortably exceeds the OS pipe buffer; without a
        # dedicated drain thread this hangs the bridge child and times out.
        os.environ["FAKE_BRIDGE_STDERR_FLOOD"] = str(512 * 1024)
        try:
            with self.make_transport(timeout=30.0) as transport:
                self.assertTrue(transport.request({"cmd": "ping"}).get("pong"))
                self.assertIn("warning spam", transport.stderr_tail())
        finally:
            del os.environ["FAKE_BRIDGE_STDERR_FLOOD"]

    def test_request_timeout_is_enforced(self):
        os.environ["FAKE_BRIDGE_SILENT"] = "1"
        try:
            transport = self.make_transport(timeout=0.5)
            try:
                with self.assertRaises(TimeoutError):
                    transport.request({"cmd": "ping"})
            finally:
                transport.close()
        finally:
            del os.environ["FAKE_BRIDGE_SILENT"]

    def test_gym_env_returns_terminal_observation(self):
        env = GodotGymEnv(project_path=PROJECT_ROOT, godot_executable=self.executable)
        try:
            observation, _info = env.reset(seed=7)
            self.assertEqual(observation.shape, (OBSERVATION_FIELD_COUNT,))
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step([1, 1, 1, 1, 0])
                steps += 1
                self.assertLessEqual(steps, 10, "episode should end within the scripted 3 steps")
            self.assertTrue(truncated)
            # Gymnasium semantics: the final step must expose the terminal
            # observation, not the auto-reset observation of the next episode.
            self.assertAlmostEqual(float(observation[0]), 0.5)
        finally:
            env.close()

    def test_vec_env_consumes_sb3_seeds_on_reset(self):
        env = GodotVecEnv(project_path=PROJECT_ROOT, godot_executable=self.executable,
                          environment_count=1)
        try:
            env.seed(123)
            env.reset()
            self.assertEqual(env.reset_infos[0].get("seed"), 123)
            # Seeds are one-shot: the next reset must not silently re-seed.
            env.reset()
            self.assertEqual(env.reset_infos[0].get("seed"), -1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
