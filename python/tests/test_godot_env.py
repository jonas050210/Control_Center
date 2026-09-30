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

from optional_deps import GYMNASIUM_REASON, HAS_GYMNASIUM, HAS_SB3, SB3_REASON

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.godot_env import GodotGymEnv, GodotProcessTransport, GodotVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BRIDGE_SOURCE = r"""
import json, os, sys

OBS_DIM = __OBS_DIM__
SELF_PLAY = "--self-play" in sys.argv
SLOT_SEED_STRIDE = 1000003

def env_count_from_argv(default=1):
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

if "--version" in sys.argv:
    sys.stdout.write("4.7.2.fake\n")
    sys.stdout.flush()
    sys.exit(0)

flood = int(os.environ.get("FAKE_BRIDGE_STDERR_FLOOD", "0"))
if flood:
    line = "warning spam\n"
    for _ in range(flood // len(line) + 1):
        sys.stderr.write(line)
    sys.stderr.flush()
silent_after_spaces = os.environ.get("FAKE_BRIDGE_SILENT", "") == "1"

# Reproduces the exact real-server behavior of the self-play regression: a
# script that fails to compile makes SelfPlayEnvironmentCore.new() return
# null, SelfPlayAdapter._init aborts on the null reset() call before the
# append, `environments` stays empty, and every reset answers
# {"ok": true, "observations": []} while the engine's SCRIPT ERROR only ever
# appears on stderr.
EMPTY_SELF_PLAY_RESET = SELF_PLAY and os.environ.get("FAKE_BRIDGE_SELF_PLAY_EMPTY_RESET", "") == "1"
if EMPTY_SELF_PLAY_RESET:
    sys.stderr.write(
        'SCRIPT ERROR: Static function "mode()" not found in base "LightingProfile".\n'
        '   at: reset (res://scripts/self_play/self_play_environment.gd)\n'
    )
    sys.stderr.flush()

step_count = 0
staged = []
match_steps = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        payload = {
            "ok": True,
            "action_space": {"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2, 2], "dimension": 6},
            "observation_space": {"type": "structured_float_vector", "size": OBS_DIM,
                                   "shape": [OBS_DIM], "low": -1.0, "high": 1.0},
        }
        if SELF_PLAY:
            payload["policy_slots"] = 2
        out(payload)
        continue
    if silent_after_spaces:
        continue
    if command == "ping":
        out({"ok": True, "pong": True})
        continue
    if command == "profile_snapshot":
        out({"ok": True, "profile": {"available": True, "timings": {"command_step": {"count": step_count}}}})
        continue
    if command == "close":
        out({"ok": True, "close": True})
        break
    if SELF_PLAY:
        # --self-play channel: pair-shaped wire, explicit seeds, no auto-reset.
        if EMPTY_SELF_PLAY_RESET and command in ("reset", "step"):
            if command == "reset":
                out({"ok": True, "observations": [], "env_seeds": []})
            else:
                out({"ok": True, "observations": [], "rewards": [],
                     "dones": [], "infos": []})
        elif command == "reset":
            seed = int(request.get("seed", 0))
            match_steps = 0
            out({"ok": True,
                 "observations": [[[0.0] * OBS_DIM, [0.1] * OBS_DIM]],
                 "env_seeds": [[seed, seed + SLOT_SEED_STRIDE]]})
        elif command == "step":
            match_steps += 1
            done = match_steps >= 3
            done_reason = "timeout" if done else ""
            info_a = {"metrics": {"win": False, "damage_dealt": 4.0, "episode_length": match_steps},
                      "done_reason": done_reason}
            info_b = {"metrics": {"win": done, "damage_dealt": 12.0, "episode_length": match_steps},
                      "done_reason": done_reason}
            out({"ok": True,
                 "observations": [[[0.2] * OBS_DIM, [0.3] * OBS_DIM]],
                 "rewards": [[0.0, 1.0]],
                 "dones": [done],
                 "infos": [[info_a, info_b]]})
        elif command == "health_check":
            out({"ok": True, "health": [{"self_play": True}]})
        else:
            out({"ok": False, "error": "self-play bridge does not support command: " + str(command)})
        continue
    if command == "reset":
        step_count = 0
        out({"ok": True,
             "observations": [[0.0] * OBS_DIM for _ in range(ENV_COUNT)],
             "infos": [{"seed": request.get("seed")} for _ in range(ENV_COUNT)]})
    elif command == "health_check":
        out({"ok": True, "health": [{"healthy": True} for _ in range(ENV_COUNT)]})
    elif command == "set_episode_plans":
        plans = request.get("plans", [])
        indices = []
        broken = None
        for payload in plans:
            missing = [k for k in ("index", "seed", "map_id", "scenario", "lighting",
                                    "enemy_count", "curriculum_level") if k not in payload]
            if missing:
                broken = "plan missing fields: " + ",".join(missing)
                break
            indices.append(int(payload["index"]))
        if broken is not None:
            out({"error": broken})   # atomic failure: nothing staged
        else:
            staged = plans
            out({"ok": True, "staged": indices})
    elif command == "episode_conditions":
        conditions = []
        for payload in staged:
            condition = {k: payload[k] for k in ("seed", "map_id", "scenario", "lighting",
                                                  "enemy_count", "curriculum_level")}
            condition["environment_index"] = int(payload["index"])
            condition["resolved"] = True
            conditions.append(condition)
        out({"ok": True, "conditions": conditions})
    elif command == "step":
        step_count += 1
        done = step_count >= 3
        info = {"done_reason": "timeout" if done else "", "events": {}}
        if done or not request.get("compact_infos", False):
            info["metrics"] = {"episode_length": step_count}
        if not request.get("compact_infos", False):
            info["reward_components"] = {"reward_hit": 0.0}
        if done:
            info["terminal_observation"] = [0.5] * OBS_DIM
            info["TimeLimit.truncated"] = True
            step_count = 0
        out({"ok": True,
             "observations": [[0.25] * OBS_DIM for _ in range(ENV_COUNT)],
             "rewards": [1.0 for _ in range(ENV_COUNT)],
             "dones": [done for _ in range(ENV_COUNT)],
             "infos": [info for _ in range(ENV_COUNT)]})
    else:
        out({"ok": False, "error": "unknown command"})
""".replace("__OBS_DIM__", str(OBSERVATION_FIELD_COUNT))


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class FakeBridgeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        bridge_py = tmp / "fake_bridge.py"
        bridge_py.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        wrapper = tmp / "fake_godot"
        # Forward "$@" so launch flags (e.g. --self-play 1) reach the fake.
        wrapper.write_text(
            f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}' \"$@\"\n", encoding="utf-8"
        )
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

    @unittest.skipUnless(HAS_GYMNASIUM, GYMNASIUM_REASON)
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

    def test_compact_infos_keep_events_and_terminal_metrics_only(self):
        from sandboxai.godot_env import GodotBatchClient

        client = GodotBatchClient(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=1,
            compact_infos=True,
        )
        try:
            client.reset(42)
            _obs, _rewards, dones, infos = client.step([[1, 1, 1, 1, 0, 0]])
            self.assertFalse(bool(dones[0]))
            self.assertIn("events", infos[0])
            self.assertNotIn("metrics", infos[0])
            self.assertNotIn("reward_components", infos[0])
            client.step([[1, 1, 1, 1, 0, 0]])
            _obs, _rewards, dones, infos = client.step([[1, 1, 1, 1, 0, 0]])
            self.assertTrue(bool(dones[0]))
            self.assertIn("metrics", infos[0])
            self.assertNotIn("reward_components", infos[0])
        finally:
            client.close()

    def test_opt_in_profiler_records_transport_phase_and_byte_totals(self):
        from sandboxai.godot_env import GodotBatchClient
        from sandboxai.training_profile import TrainingProfiler

        profiler = TrainingProfiler()
        client = GodotBatchClient(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=1,
            profiler=profiler,
        )
        try:
            client.reset(42)
            client.step([[1, 1, 1, 1, 0, 0]])
            profiler.server = client.profile_snapshot()
        finally:
            client.close()
        report = profiler.report()
        self.assertEqual(report["timings"]["bridge.step.total"]["count"], 1)
        self.assertGreater(report["counters"]["bridge.step.response_bytes"], 0)
        self.assertTrue(report["godot_server"].get("available"))

    @unittest.skipUnless(HAS_SB3, SB3_REASON)
    def test_vec_env_consumes_sb3_seeds_on_reset(self):
        env = GodotVecEnv(
            project_path=PROJECT_ROOT, godot_executable=self.executable, environment_count=1
        )
        try:
            env.seed(123)
            env.reset()
            self.assertEqual(env.reset_infos[0].get("seed"), 123)
            # Seeds are one-shot: the next reset must not silently re-seed.
            env.reset()
            self.assertEqual(env.reset_infos[0].get("seed"), -1)
        finally:
            env.close()


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class EpisodePlanCommandTest(FakeBridgeTestCase):
    """Wire-level contract for the curriculum plan commands the pipeline
    issues at episode boundaries (command phase of rl_server.gd)."""

    def test_set_episode_plans_and_condition_readback(self):
        from sandboxai.godot_env import GodotBatchClient

        client = GodotBatchClient(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=2,
            seed=5,
        )
        try:
            plans = [
                {
                    "index": 0,
                    "seed": 4242,
                    "map_id": "open_field",
                    "scenario": "cover_fight",
                    "lighting": "normal",
                    "enemy_count": 1,
                    "curriculum_level": 5,
                },
                {
                    "index": 1,
                    "seed": 777,
                    "map_id": "pillar_hall",
                    "scenario": "",
                    "lighting": "fog",
                    "enemy_count": 3,
                    "curriculum_level": 7,
                },
            ]
            staged = client.set_episode_plans(plans)
            self.assertTrue(staged.get("ok"))
            self.assertEqual(staged.get("staged"), [0, 1])
            conditions = client.episode_conditions()
            by_index = {int(c["environment_index"]): c for c in conditions}
            self.assertEqual(by_index[0]["map_id"], "open_field")
            self.assertEqual(by_index[0]["curriculum_level"], 5)
            self.assertEqual(by_index[1]["lighting"], "fog")
            self.assertEqual(by_index[1]["enemy_count"], 3)
        finally:
            client.close()

    def test_invalid_plan_batch_fails_atomically(self):
        from sandboxai.godot_env import GodotBatchClient

        client = GodotBatchClient(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=2,
            seed=5,
        )
        try:
            with self.assertRaises(Exception) as ctx:
                client.set_episode_plans(
                    [{"index": 0, "seed": 1, "map_id": "open_field"}]  # missing fields
                )
            self.assertIn("plan missing fields", str(ctx.exception))
            # Nothing staged: the readback must still be empty.
            self.assertEqual(client.episode_conditions(), [])
        finally:
            client.close()


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class SelfPlayBridgeTest(FakeBridgeTestCase):
    """The --self-play channel: deterministic two-slot matches (league)."""

    def make_self_play_client(self):
        from sandboxai.self_play import SelfPlayBatchClient

        return SelfPlayBatchClient(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            environment_count=1,
            seed=42,
        )

    def test_self_play_channel_shapes_and_deterministic_match(self):
        from sandboxai.self_play import play_self_play_match

        with self.make_self_play_client() as client:
            self.assertEqual(client.transport.spaces.get("policy_slots"), 2)
            predictor = lambda obs: [0, 0, 0, 0, 0, 0]
            first = play_self_play_match(client, predictor, predictor, seed=1234)
            second = play_self_play_match(client, predictor, predictor, seed=1234)
        self.assertEqual(first, second, "self-play matches must be bit-for-bit deterministic")
        # The scripted fake makes slot B win on the timeout boundary.
        self.assertEqual(first["score_a"], 0.0)
        self.assertTrue(first["truncated"])
        self.assertEqual(first["done_reason"], "timeout")
        self.assertEqual(first["metrics_a"]["episode_length"], 3)
        self.assertEqual(first["seed"], 1234)

    def test_self_play_rejects_training_only_commands(self):
        client = self.make_self_play_client()
        try:
            with self.assertRaises(Exception) as ctx:
                client.transport.request({"cmd": "set_episode_plans", "plans": []})
            self.assertIn("self-play bridge does not support", str(ctx.exception))
        finally:
            client.close()

    def test_match_reports_malformed_bridge_clearly(self):
        from sandboxai.self_play import play_self_play_match

        class _Broken:
            def reset(self, seed):
                return []  # not a pair list

        # The public reader must raise a named, actionable error.
        class _Predictor:
            def __call__(self, obs):
                return [0] * 6

        with self.assertRaises(RuntimeError) as ctx:
            play_self_play_match(_Broken(), _Predictor(), _Predictor(), seed=1)
        self.assertIn("malformed observations", str(ctx.exception))


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class RuntimeValidatorSelfPlayRegressionTests(FakeBridgeTestCase):
    """End-to-end regression for the real-Godot self-play failure.

    The reported local failure (6/7 checks passing, only the self-play check
    failing with ``Self play reset observation shape invalid: []``) was
    caused INSIDE the engine process: ``self_play_environment.gd`` called
    ``LightingProfile.mode(...)``, an instance variable, through the script
    class — a Godot COMPILE error. The invalid script still preloads, so the
    single-agent path (which never instantiates it) kept passing every
    check; ``SelfPlayEnvironmentCore.new()`` returned null, the adapter was
    left with zero environments, and reset answered
    ``{"ok": true, "observations": []}``.

    These tests replay that exact wire behavior through the REAL
    RuntimeValidator with a fake bridge standing in for Godot.
    """

    def _validate(self):
        from sandboxai.runtime_validation import RuntimeValidator

        validator = RuntimeValidator(
            project_path=PROJECT_ROOT,
            godot_executable=self.executable,
            timeout=20.0,
        )
        return validator.validate(env_count=1, test_self_play=True)

    def _self_play_check(self, report):
        matches = [c for c in report.checks if c.check_id == "self_play_channel"]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_all_checks_pass_with_a_healthy_self_play_bridge(self):
        report = self._validate()
        self.assertEqual(report.status, "passed", report.to_dict())
        self.assertEqual(report.failed_checks, 0)
        check = self._self_play_check(report)
        self.assertTrue(check.passed, check.error)
        self.assertEqual(check.details.get("slot_0_obs_dim"), OBSERVATION_FIELD_COUNT)
        self.assertEqual(check.details.get("slot_1_obs_dim"), OBSERVATION_FIELD_COUNT)

    def test_empty_self_play_reset_fails_only_the_self_play_check_with_stderr_context(self):
        os.environ["FAKE_BRIDGE_SELF_PLAY_EMPTY_RESET"] = "1"
        try:
            report = self._validate()
        finally:
            del os.environ["FAKE_BRIDGE_SELF_PLAY_EMPTY_RESET"]

        # The broken script only executes in --self-play mode, so exactly the
        # self-play check fails — the reported 6/7 signature.
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.failed_checks, 1)
        check = self._self_play_check(report)
        self.assertFalse(check.passed)
        # The exact reported failure text:
        self.assertIn("Self play reset observation shape invalid: []", check.error)
        # The engine-side root cause must be surfaced with the stderr tail:
        self.assertIn("Godot stderr tail", check.error)
        self.assertIn('Static function "mode()" not found', check.error)


if __name__ == "__main__":
    unittest.main()
