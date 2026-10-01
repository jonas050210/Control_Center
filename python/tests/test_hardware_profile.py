"""Tests for the hardware device-comparison wizard.

The orchestration (candidate discovery, selection, fallback, cancellation
and persistence) is exercised with injected measurement functions and no
engine. One end-to-end test runs the real, engine-backed CPU measurement
against a scripted fake Godot bridge on POSIX — the same technique
``test_ppo_smoke.py`` uses — so the ``default_measure`` path is proven to
produce a genuine throughput number rather than only being type-checked.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from sandboxai import hardware_profile as hp
from sandboxai.contract import OBSERVATION_FIELD_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAKE_BRIDGE_SOURCE = rf"""
import json, random, sys

OBS_DIM = {OBSERVATION_FIELD_COUNT}
random.seed(0)


def env_count_from_argv(default=2):
    for i, arg in enumerate(sys.argv):
        if arg == "--env-count" and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                return default
    return default


ENV_COUNT = env_count_from_argv()


def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def make_obs():
    return [random.uniform(-1.0, 1.0) for _ in range(OBS_DIM)]


staged_plans = []
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
        for i in request.get("indices", []):
            step_counts[i] = 0
        out({{"ok": True, "results": [{{"index": i, "observation": make_obs()}}
                                     for i in request.get("indices", [])]}})
    elif command == "set_episode_plans":
        staged_plans.clear()
        staged_plans.extend(request.get("plans", []))
        out({{"ok": True, "staged": [int(p.get("index", -1)) for p in staged_plans]}})
    elif command == "episode_conditions":
        out({{"ok": True, "conditions": staged_plans}})
    elif command == "step":
        actions = request.get("actions", [])
        observations, rewards, dones, infos = [], [], [], []
        for i in range(len(actions)):
            step_counts[i] = step_counts.get(i, 0) + 1
            done = step_counts[i] >= 8
            reward = 1.0 if actions[i][4] == 1 else 0.01
            info = {{"done_reason": "timeout" if done else "",
                    "metrics": {{"episode_reward": reward}}}}
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
"""


def _fake_measurement(label: str, sps: float) -> hp.DeviceMeasurement:
    candidate = next(c for c in hp._ALL_CANDIDATES if c.label == label)
    return hp.DeviceMeasurement(
        label=candidate.label,
        device=candidate.device,
        inference_device=candidate.inference_device,
        status=hp.STATUS_MEASURED,
        steps_per_second=sps,
        steps_completed=5000,
        wall_seconds=5000.0 / sps,
    )


class CandidateDiscoveryTests(unittest.TestCase):
    def test_cpu_only_when_no_cuda(self):
        labels = [c.label for c in hp.available_candidates(has_cuda=False)]
        self.assertEqual(labels, [hp.DEVICE_CPU])

    def test_all_three_when_cuda_present(self):
        labels = [c.label for c in hp.available_candidates(has_cuda=True)]
        self.assertEqual(labels, [hp.DEVICE_CPU, hp.DEVICE_HYBRID, hp.DEVICE_CUDA])

    def test_hybrid_maps_to_cuda_updates_cpu_inference(self):
        hybrid = next(c for c in hp._ALL_CANDIDATES if c.label == hp.DEVICE_HYBRID)
        self.assertEqual((hybrid.device, hybrid.inference_device), ("cuda", "cpu"))


class SelectionAndFallbackTests(unittest.TestCase):
    def test_best_is_highest_throughput(self):
        measurements = [_fake_measurement("cpu", 90.0), _fake_measurement("hybrid", 120.0)]
        best = hp.select_best(measurements)
        assert best is not None
        self.assertEqual(best.label, hp.DEVICE_HYBRID)

    def test_only_measured_candidates_are_considered(self):
        failed = hp.DeviceMeasurement("cuda", "cuda", "cuda", hp.STATUS_FAILED, error="boom")
        best = hp.select_best([failed, _fake_measurement("cpu", 10.0)])
        assert best is not None
        self.assertEqual(best.label, hp.DEVICE_CPU)

    def test_profile_falls_back_to_cpu_when_nothing_measured(self):
        profile = hp.build_profile(
            [hp.DeviceMeasurement("cpu", "cpu", "cpu", hp.STATUS_UNAVAILABLE)],
            measurement_steps=5000,
            godot_executable=None,
            godot_available=False,
            host={},
        )
        self.assertTrue(profile.fallback)
        self.assertEqual(profile.config_overrides(), {"device": "cpu", "inference_device": "cpu"})

    def test_profile_records_only_measured_numbers(self):
        profile = hp.build_profile(
            [_fake_measurement("cpu", 42.0)],
            measurement_steps=5000,
            godot_executable="godot",
            godot_available=True,
            host={},
        )
        self.assertFalse(profile.fallback)
        self.assertEqual(profile.selected_device, hp.DEVICE_CPU)
        (measured,) = profile.measurements
        self.assertEqual(measured.steps_per_second, 42.0)


class CancellationTests(unittest.TestCase):
    def test_cancellation_stops_before_remaining_candidates(self):
        calls: list[str] = []

        def measure(candidate: hp.DeviceCandidate) -> hp.DeviceMeasurement:
            calls.append(candidate.label)
            return _fake_measurement(candidate.label, 10.0)

        # Cancel after the first candidate has been measured.
        state = {"count": 0}

        def cancel() -> bool:
            should = state["count"] >= 1
            state["count"] += 1
            return should

        results = hp.measure_devices(hp.available_candidates(has_cuda=True), measure, cancel=cancel)
        self.assertEqual(calls, [hp.DEVICE_CPU])
        self.assertEqual(results[0].status, hp.STATUS_MEASURED)
        self.assertEqual(results[1].status, hp.STATUS_CANCELLED)
        self.assertEqual(results[2].status, hp.STATUS_CANCELLED)

    def test_measurement_exception_becomes_failed_not_crash(self):
        def measure(candidate: hp.DeviceCandidate) -> hp.DeviceMeasurement:
            raise RuntimeError("bridge exploded")

        results = hp.measure_devices([hp._ALL_CANDIDATES[0]], measure)
        self.assertEqual(results[0].status, hp.STATUS_FAILED)
        self.assertEqual(results[0].error, "bridge exploded")


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware_profile.json"
            profile = hp.build_profile(
                [_fake_measurement("cpu", 55.0)],
                measurement_steps=5000,
                godot_executable="godot",
                godot_available=True,
                host={"machine": "x86_64"},
            )
            profile.save(path)
            loaded = hp.load_profile(path)
            assert loaded is not None
            self.assertEqual(loaded.selected_device, profile.selected_device)
            self.assertEqual(loaded.measurement_steps, 5000)
            self.assertEqual(loaded.host, {"machine": "x86_64"})

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(hp.load_profile(Path(tmp) / "nope.json"))

    def test_load_corrupt_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware_profile.json"
            path.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(hp.load_profile(path))

    def test_load_tolerates_bom(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware_profile.json"
            profile = hp.build_profile(
                [_fake_measurement("cpu", 1.0)],
                measurement_steps=10,
                godot_executable="godot",
                godot_available=True,
                host={},
            )
            path.write_text("\ufeff" + json.dumps(profile.to_dict()), encoding="utf-8")
            loaded = hp.load_profile(path)
            assert loaded is not None
            self.assertEqual(loaded.selected_device, hp.DEVICE_CPU)


class WizardWithoutEngineTests(unittest.TestCase):
    def test_missing_engine_yields_documented_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = hp.run_hardware_wizard(
                project_path=PROJECT_ROOT,
                godot_executable="definitely-not-a-real-godot-binary",
                steps=10,
                save=True,
                save_path=Path(tmp) / "profile.json",
                has_cuda=False,
            )
            self.assertTrue(profile.fallback)
            self.assertFalse(profile.godot_available)
            self.assertEqual(profile.selected_device, hp.DEVICE_CPU)
            self.assertEqual(profile.measurements[0].status, hp.STATUS_UNAVAILABLE)
            self.assertTrue((Path(tmp) / "profile.json").is_file())

    def test_injected_measure_is_used_even_without_engine(self):
        def measure(candidate: hp.DeviceCandidate) -> hp.DeviceMeasurement:
            return _fake_measurement(candidate.label, 33.0)

        with tempfile.TemporaryDirectory() as tmp:
            profile = hp.run_hardware_wizard(
                project_path=PROJECT_ROOT,
                godot_executable="definitely-not-a-real-godot-binary",
                steps=100,
                measure=measure,
                save=True,
                save_path=Path(tmp) / "profile.json",
                has_cuda=False,
            )
            self.assertFalse(profile.fallback)
            self.assertEqual(profile.selected_device, hp.DEVICE_CPU)
            self.assertEqual(profile.measurements[0].steps_per_second, 33.0)


@unittest.skipUnless(os.name == "posix", "fake bridge executable requires POSIX shebang support")
class DefaultMeasureEndToEndTests(unittest.TestCase):
    """Prove the engine-backed CPU measurement produces a real number."""

    def setUp(self) -> None:
        try:
            import stable_baselines3  # noqa: F401
            import tensorboard  # noqa: F401
        except ImportError:
            self.skipTest("training extras (stable-baselines3, tensorboard) are not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        bridge_py = tmp / "fake_bridge.py"
        bridge_py.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        wrapper = tmp / "fake_godot"
        wrapper.write_text(
            f"#!/bin/sh\nexec '{sys.executable}' '{bridge_py}' \"$@\"\n", encoding="utf-8"
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.executable = str(wrapper)

    def test_cpu_candidate_measures_positive_throughput(self):
        cpu = next(c for c in hp._ALL_CANDIDATES if c.label == hp.DEVICE_CPU)
        with tempfile.TemporaryDirectory() as out_root:
            measurement = hp.default_measure(
                cpu,
                project_path=PROJECT_ROOT,
                godot_executable=self.executable,
                steps=64,
                environment_count=2,
                output_root=out_root,
            )
        self.assertEqual(measurement.status, hp.STATUS_MEASURED)
        assert measurement.steps_per_second is not None
        self.assertGreater(measurement.steps_per_second, 0.0)
        assert measurement.steps_completed is not None
        self.assertGreater(measurement.steps_completed, 0)

    def test_wizard_selects_cpu_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = hp.run_hardware_wizard(
                project_path=PROJECT_ROOT,
                godot_executable=self.executable,
                steps=64,
                save=True,
                save_path=Path(tmp) / "profile.json",
                has_cuda=False,
            )
        self.assertFalse(profile.fallback)
        self.assertEqual(profile.selected_device, hp.DEVICE_CPU)
        self.assertTrue(profile.godot_available)


if __name__ == "__main__":
    unittest.main()
