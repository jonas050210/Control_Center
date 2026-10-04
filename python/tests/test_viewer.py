"""Tests for the 3D checkpoint viewer's Python side (python/sandboxai/viewer.py).

The rendered Godot half is covered by tests/test_viewer.gd. Here a tiny fake
"Godot" (a script speaking the viewer's TCP protocol) stands in for the
engine, so the whole Python path - checkpoint resolution, policy loading,
the socket server, process launch and the episode log - runs end to end.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from optional_deps import has_module

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.viewer import (
    VIEWER_ENTRY_SCRIPT,
    EpisodeLog,
    ViewerError,
    ViewerOptions,
    _gui_variant,
    build_viewer_command,
    checkpoint_in_directory,
    handle_message,
    infer_run_settings,
    newest_checkpoint,
    resolve_policy,
    run_viewer,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

FAKE_VIEWER_SOURCE = r"""
import json, os, random, socket, sys

args = sys.argv[sys.argv.index("--") + 1:]
options = dict(zip(args[0::2], args[1::2]))
log_path = os.environ["FAKE_VIEWER_LOG"]
mode = os.environ.get("FAKE_VIEWER_MODE", "ok")
record = {"argv": sys.argv[1:], "options": options, "actions": [], "replies": []}

sock = socket.create_connection(("127.0.0.1", int(options["--policy-port"])), timeout=20)
reader = sock.makefile("rb")

def request(message):
    sock.sendall((json.dumps(message) + "\n").encode())
    reply = json.loads(reader.readline())
    record["replies"].append(reply)
    return reply

size = 65 if mode == "bad-hello" else __OBS__
hello = request({"type": "hello", "observation_size": size, "action_nvec": __NVEC__})
if not hello.get("ok"):
    json.dump(record, open(log_path, "w"))
    sys.exit(4)
rng = random.Random(7)
for _ in range(25):
    obs = [rng.uniform(-1.0, 1.0) for _ in range(__OBS__)]
    record["actions"].append(request({"type": "act", "obs": obs})["action"])
request({"type": "act", "obs": [0.0, 1.0]})
request({"type": "episode", "summary": {"episode": 1, "outcome": "win", "reward": 3.5,
         "kills": 1, "enemy_count": 1, "survival_time": 4.2, "map_id": "compound"}})
sock.sendall(b'{"type": "bye"}\n')
sock.close()
json.dump(record, open(log_path, "w"))
""".replace("__OBS__", str(OBSERVATION_FIELD_COUNT)).replace("__NVEC__", str(list(ACTION_NVEC)))


class FakeViewerExecutable:
    """A POSIX executable that behaves like the Godot viewer on the wire."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="sandboxai-fake-viewer-")
        root = Path(self._tmp.name)
        source = root / "fake_viewer.py"
        source.write_text(FAKE_VIEWER_SOURCE, encoding="utf-8")
        wrapper = root / "fake_godot"
        wrapper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{source}' \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.path = str(wrapper)
        self.log = root / "record.json"

    def record(self) -> dict:
        return json.loads(self.log.read_text(encoding="utf-8"))

    def __enter__(self) -> FakeViewerExecutable:
        os.environ["FAKE_VIEWER_LOG"] = str(self.log)
        return self

    def __exit__(self, *_args: object) -> None:
        os.environ.pop("FAKE_VIEWER_LOG", None)
        os.environ.pop("FAKE_VIEWER_MODE", None)
        self._tmp.cleanup()


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _assert_valid_action(test: unittest.TestCase, action: list) -> None:
    test.assertEqual(len(action), len(ACTION_NVEC))
    for value, cardinality in zip(action, ACTION_NVEC, strict=True):
        test.assertTrue(0 <= int(value) < cardinality, f"{action} outside {ACTION_NVEC}")


class CheckpointResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_best_checkpoint_is_preferred_inside_a_run(self):
        run = self.root / "runs" / "a"
        _touch(run / "checkpoints" / "latest.zip")
        _touch(run / "checkpoints" / "best_eval.zip")
        self.assertEqual(checkpoint_in_directory(run), run / "checkpoints" / "best_eval.zip")

    def test_newest_periodic_checkpoint_when_no_named_one_exists(self):
        run = self.root / "runs" / "a"
        _touch(run / "checkpoints" / "ppo_900_steps.zip")
        _touch(run / "checkpoints" / "ppo_12000_steps.zip")
        self.assertEqual(checkpoint_in_directory(run).name, "ppo_12000_steps.zip")

    def test_behavior_cloning_directory_resolves_to_best_pt(self):
        run = self.root / "bc_runs" / "latest"
        _touch(run / "latest.pt")
        _touch(run / "best.pt")
        resolved = resolve_policy(str(run), None, self.root)
        self.assertEqual((resolved.kind, resolved.path), ("bc", run / "best.pt"))

    def test_without_checkpoint_the_most_recent_run_wins(self):
        _touch(self.root / "runs" / "old" / "final.zip", mtime=1_000_000)
        newest = _touch(self.root / "runs" / "new" / "checkpoints" / "latest.zip", 2_000_000)
        self.assertEqual(newest_checkpoint(self.root), newest)
        resolved = resolve_policy(None, None, self.root)
        self.assertEqual(resolved.kind, "ppo")
        self.assertEqual(resolved.label, "new/latest.zip")
        self.assertEqual(resolved.run_dir, self.root / "runs" / "new")

    def test_scripted_needs_no_checkpoint(self):
        resolved = resolve_policy(None, "scripted", self.root)
        self.assertEqual((resolved.kind, resolved.path), ("scripted", None))

    def test_clear_errors_for_missing_or_wrong_checkpoints(self):
        with self.assertRaisesRegex(ViewerError, "no trained checkpoint"):
            resolve_policy(None, None, self.root)
        with self.assertRaisesRegex(ViewerError, "does not exist"):
            resolve_policy(str(self.root / "nope.zip"), None, self.root)
        weird = _touch(self.root / "model.onnx")
        with self.assertRaisesRegex(ViewerError, "unsupported checkpoint type"):
            resolve_policy(str(weird), None, self.root)
        zip_path = _touch(self.root / "model.zip")
        with self.assertRaisesRegex(ViewerError, "does not match"):
            resolve_policy(str(zip_path), "bc", self.root)

    def test_run_settings_prefer_the_curriculum_state(self):
        run = self.root / "runs" / "a"
        (run / "checkpoints").mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"curriculum_level": 3, "enemy_count": 2}))
        self.assertEqual(infer_run_settings(run), {"level": 3, "enemies": 2})
        (run / "checkpoints" / "curriculum_state.json").write_text(
            json.dumps({"driver": {"auto": {"level": 11}}})
        )
        # Level 11 is the self-play environment; the viewer clamps to 10.
        self.assertEqual(infer_run_settings(run), {"level": 10, "enemies": 2})
        self.assertEqual(infer_run_settings(None), {})
        (run / "config.json").write_text("{broken")
        self.assertEqual(infer_run_settings(run), {"level": 10})


class ProtocolTests(unittest.TestCase):
    def _actor(self, obs):
        return [2, 1, 1, 1, 1, 0]

    def test_hello_checks_the_contract(self):
        log = EpisodeLog()
        good = {
            "type": "hello",
            "observation_size": OBSERVATION_FIELD_COUNT,
            "action_nvec": list(ACTION_NVEC),
        }
        self.assertEqual(
            handle_message(good, self._actor, "run/best.zip", log),
            ({"ok": True, "policy": "run/best.zip"}, True),
        )
        reply, keep = handle_message(dict(good, observation_size=65), self._actor, "x", log)
        self.assertFalse(reply["ok"])
        self.assertIn("contract mismatch", reply["error"])
        self.assertFalse(keep)

    def test_act_validates_the_observation_width(self):
        log = EpisodeLog()
        obs = [0.0] * OBSERVATION_FIELD_COUNT
        self.assertEqual(
            handle_message({"type": "act", "obs": obs}, self._actor, "x", log),
            ({"action": [2, 1, 1, 1, 1, 0]}, True),
        )
        reply, keep = handle_message({"type": "act", "obs": [1.0]}, self._actor, "x", log)
        self.assertIn("error", reply)
        self.assertEqual(log.decisions, 1, "only answered requests count")
        self.assertTrue(keep, "a bad request must not kill the session")

    def test_episodes_are_logged_and_bye_ends_the_session(self):
        log = EpisodeLog()
        summary = {"episode": 1, "outcome": "win", "reward": 2.0, "map_id": "compound"}
        self.assertEqual(
            handle_message({"type": "episode", "summary": summary}, self._actor, "x", log),
            ({"ok": True}, True),
        )
        handle_message(
            {"type": "episode", "summary": {"outcome": "loss", "reward": 0.0}},
            self._actor,
            "x",
            log,
        )
        self.assertEqual(log.totals(), {"episodes": 2, "wins": 1, "mean_reward": 1.0})
        self.assertEqual(handle_message({"type": "bye"}, self._actor, "x", log), (None, False))


class CommandTests(unittest.TestCase):
    def test_command_launches_the_viewer_entry_with_settings(self):
        command = build_viewer_command(
            "godot",
            REPOSITORY_ROOT,
            4242,
            {"map": "compound", "level": 7, "enemies": 2, "scenario": "", "max-steps": None},
        )
        self.assertEqual(command[0], "godot")
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--script") + 1], VIEWER_ENTRY_SCRIPT)
        user = command[command.index("--") + 1 :]
        self.assertEqual(user[user.index("--policy-port") + 1], "4242")
        self.assertEqual(user[user.index("--map") + 1], "compound")
        self.assertEqual(user[user.index("--level") + 1], "7")
        self.assertNotIn("--scenario", user, "empty settings are not passed")
        self.assertNotIn("--max-steps", user)
        self.assertIn(
            "--headless", build_viewer_command("godot", REPOSITORY_ROOT, 1, {}, headless=True)
        )

    def test_windows_console_build_is_swapped_for_the_windowed_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            console = _touch(Path(tmp) / "Godot_v4.7.2-stable_win64_console.exe")
            self.assertEqual(_gui_variant(str(console)), str(console))  # no sibling yet
            windowed = _touch(Path(tmp) / "Godot_v4.7.2-stable_win64.exe")
            self.assertEqual(_gui_variant(str(console)), str(windowed))
            self.assertEqual(_gui_variant("/usr/bin/godot"), "/usr/bin/godot")

    def test_unknown_map_is_rejected_before_launch(self):
        with self.assertRaisesRegex(ViewerError, "unknown map"):
            run_viewer(
                ViewerOptions(policy="scripted", map="atlantis", project_path=str(REPOSITORY_ROOT))
            )


@unittest.skipUnless(os.name == "posix", "the fake Godot wrapper relies on shebang execution")
class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _options(self, fake: FakeViewerExecutable, **overrides) -> ViewerOptions:
        values = {
            "godot_executable": fake.path,
            "project_path": str(REPOSITORY_ROOT),
            "output_root": str(self.output),
        }
        values.update(overrides)
        return ViewerOptions(**values)

    def test_scripted_policy_plays_through_the_socket(self):
        output = io.StringIO()
        with FakeViewerExecutable() as fake, contextlib.redirect_stdout(output):
            code = run_viewer(self._options(fake, policy="scripted", map="compound", seed=5))
            record = fake.record()
        self.assertEqual(code, 0)
        self.assertIn("Answered 25 policy decisions.", output.getvalue())
        self.assertIn("Watched 1 episode(s): 1 won", output.getvalue())
        self.assertEqual(record["replies"][0], {"ok": True, "policy": "scripted baseline"})
        self.assertEqual(len(record["actions"]), 25)
        for action in record["actions"]:
            _assert_valid_action(self, action)
        self.assertIn("error", record["replies"][-2], "the malformed act got an error reply")
        options = record["options"]
        self.assertEqual(options["--map"], "compound")
        self.assertEqual(options["--level"], "10", "default level without a run")
        self.assertEqual(options["--seed"], "5")

    def test_contract_mismatch_is_reported_as_a_failed_launch(self):
        with FakeViewerExecutable() as fake:
            os.environ["FAKE_VIEWER_MODE"] = "bad-hello"
            code = run_viewer(self._options(fake, policy="scripted"))
            record = fake.record()
        self.assertEqual(code, 4)
        self.assertFalse(record["replies"][0]["ok"])

    @unittest.skipUnless(
        has_module("stable_baselines3") and has_module("gymnasium"), "needs the training extras"
    )
    def test_real_ppo_checkpoint_drives_the_viewer(self):
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import PPO

        class ContractEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_FIELD_COUNT,), np.float32)
            action_space = gym.spaces.MultiDiscrete(list(ACTION_NVEC))

            def reset(self, *, seed=None, options=None):
                return np.zeros(OBSERVATION_FIELD_COUNT, np.float32), {}

            def step(self, action):
                return np.zeros(OBSERVATION_FIELD_COUNT, np.float32), 0.0, True, False, {}

        run = self.output / "runs" / "demo"
        (run / "checkpoints").mkdir(parents=True)
        PPO("MlpPolicy", ContractEnv(), n_steps=8, batch_size=8, device="cpu").save(
            run / "checkpoints" / "best_eval.zip"
        )
        (run / "config.json").write_text(json.dumps({"curriculum_level": 4}))
        with FakeViewerExecutable() as fake:
            code = run_viewer(self._options(fake))
            record = fake.record()
        self.assertEqual(code, 0)
        self.assertEqual(record["replies"][0]["policy"], "demo/best_eval.zip")
        self.assertEqual(record["options"]["--level"], "4", "the run's level is reused")
        for action in record["actions"]:
            _assert_valid_action(self, action)

    @unittest.skipUnless(has_module("torch"), "needs torch")
    def test_behavior_cloning_checkpoint_drives_the_viewer(self):
        import torch

        from sandboxai.bc import BehaviorCloningPolicy

        model = BehaviorCloningPolicy(OBSERVATION_FIELD_COUNT, (32, 32))
        run = self.output / "bc_runs" / "latest"
        run.mkdir(parents=True)
        torch.save(
            {
                "format": "sandboxai.bc.v1",
                "observation_dim": OBSERVATION_FIELD_COUNT,
                "hidden_sizes": [32, 32],
                "action_nvec": list(ACTION_NVEC),
                "model_state_dict": model.state_dict(),
            },
            run / "best.pt",
        )
        with FakeViewerExecutable() as fake:
            code = run_viewer(self._options(fake, checkpoint=str(run), stochastic=True))
            record = fake.record()
        self.assertEqual(code, 0)
        self.assertEqual(record["replies"][0]["policy"], "latest/best.pt")
        for action in record["actions"]:
            _assert_valid_action(self, action)


if __name__ == "__main__":  # pragma: no cover
    random.seed(0)
    unittest.main()
