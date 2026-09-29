import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sandboxai.config import (
    CURRICULUM_LEVEL_COUNT,
    BCConfig,
    TrainingConfig,
    find_godot_executable,
    load_godot_executable_setting,
    save_godot_executable_setting,
)


class ConfigTests(unittest.TestCase):
    def test_training_config_validates_and_detects_cpu(self):
        config = TrainingConfig(environment_count=2, rollout_length=32, batch_size=16, device="cpu").validate()
        self.assertEqual(config.resolved_device(), "cpu")
        self.assertEqual(config.to_dict()["environment_count"], 2)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(environment_count=2, rollout_length=4, batch_size=9).validate()

    def test_invalid_curriculum_level_rejected(self):
        # 1-10 are combat levels, 11 is the self-play hook; 0 and 12 are not.
        with self.assertRaises(ValueError):
            TrainingConfig(curriculum_level=CURRICULUM_LEVEL_COUNT + 1).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(curriculum_level=0).validate()

    def test_all_declared_curriculum_levels_are_accepted(self):
        for level in range(1, CURRICULUM_LEVEL_COUNT + 1):
            TrainingConfig(curriculum_level=level).validate()

    def test_negative_entropy_coefficient_rejected(self):
        # Negative entropy actively rewards determinism; it must fail at
        # config load instead of silently collapsing the policy.
        with self.assertRaises(ValueError):
            TrainingConfig(entropy_coefficient=-0.01).validate()

    def test_evaluation_defaults_use_exact_batched_plans(self):
        config = TrainingConfig()
        self.assertEqual(config.evaluation_environment_count, 8)
        self.assertEqual(config.checkpoint_eval_environment_count, 8)

    def test_invalid_evaluation_episodes_and_torch_threads_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(evaluation_episodes=0).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(torch_threads=-1).validate()

    def test_invalid_net_arch_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(net_arch=(128, 0)).validate()

    def test_valid_torch_threads_is_accepted(self):
        config = TrainingConfig(torch_threads=4).validate()
        self.assertEqual(config.torch_threads, 4)

    def test_entropy_coefficient_defaults_to_nonzero(self):
        # Regression: with MultiDiscrete actions, a 0 entropy coefficient lets
        # PPO collapse into a degenerate action distribution (e.g. never
        # shooting) in the first few updates. The default must keep gentle
        # exploration pressure so combat actions stay discoverable.
        config = TrainingConfig()
        self.assertGreaterEqual(config.entropy_coefficient, 0.005)

    def test_bc_config_validation(self):
        config = BCConfig(epochs=10, batch_size=64, early_stopping_patience=3).validate()
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.early_stopping_patience, 3)

    def test_find_godot_executable(self):
        exe = find_godot_executable("godot")
        self.assertIsInstance(exe, str)
        self.assertTrue(len(exe) > 0)


class GodotExecutableResolutionTests(unittest.TestCase):
    """Precedence and persistence of the Godot executable configuration.

    Regression: `validate-runtime --godot-executable X` worked while a later
    `train` (without the flag) still launched bare `godot` from PATH and
    failed with WinError 2 on Windows — nothing remembered the configured
    binary. `find_godot_executable` must resolve, in order: an explicit
    executable, GODOT_PATH/GODOT_EXECUTABLE, the remembered local setting,
    well-known candidates, and finally the `godot` default.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        # A settings location inside a fake checkout, so the tests never
        # touch (nor depend on) the developer's real .sandboxai/settings.json.
        self.checkout = tmp / "checkout"
        (self.checkout / ".sandboxai").mkdir(parents=True)
        (self.checkout / "project.godot").touch()
        self.settings_path = self.checkout / ".sandboxai" / "settings.json"
        self.executable = tmp / "Godot_v4.7.2-stable_win64_console.exe"
        self.executable.touch()
        patcher = mock.patch("sandboxai.config._settings_path", return_value=self.settings_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Neutralise ambient machine state (PATH lookups, GODOT_* variables)
        # so the assertions hold on any machine, with or without Godot.
        which_patcher = mock.patch("sandboxai.config.shutil.which", return_value=None)
        which_patcher.start()
        self.addCleanup(which_patcher.stop)
        env = {key: value for key, value in os.environ.items() if key not in ("GODOT_PATH", "GODOT_EXECUTABLE")}
        env_patcher = mock.patch.dict(os.environ, env, clear=True)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    def _write_setting(self, value: str) -> None:
        self.settings_path.write_text(json.dumps({"godot_executable": value}), encoding="utf-8")

    def test_explicit_executable_wins_over_env_and_remembered(self):
        other = self.executable.with_name("other_godot")
        other.touch()
        self._write_setting(str(other))
        with mock.patch.dict(os.environ, {"GODOT_PATH": str(other)}):
            resolved = find_godot_executable(str(self.executable))
        self.assertEqual(resolved, str(self.executable))

    def test_env_var_beats_remembered_setting(self):
        remembered = self.executable.with_name("remembered_godot")
        remembered.touch()
        self._write_setting(str(remembered))
        with mock.patch.dict(os.environ, {"GODOT_PATH": str(self.executable)}):
            resolved = find_godot_executable("godot")
        self.assertEqual(resolved, str(self.executable))

    def test_remembered_setting_is_used_when_nothing_else_resolves(self):
        # The reported bug: `train` without --godot-executable must pick up
        # the executable configured earlier, instead of bare `godot`.
        self._write_setting(str(self.executable))
        self.assertEqual(find_godot_executable("godot"), str(self.executable))

    def test_default_godot_when_nothing_is_configured(self):
        # No flag, no env var, no remembered setting: the documented default
        # (`godot` on PATH) must be preserved verbatim.
        with mock.patch("sandboxai.config._GODOT_CANDIDATES", ()):
            self.assertEqual(find_godot_executable("godot"), "godot")

    def test_remembered_windows_path_flows_through_unmangled(self):
        # On Windows the configured executable is an absolute drive-letter
        # path; the resolver must return it byte-for-byte, never rewrite it.
        windows_path = r"C:\Users\jonas\OneDrive\Desktop\Godot_v4.7.2-stable_win64_console.exe"
        self._write_setting(windows_path)
        with mock.patch(
            "sandboxai.config.shutil.which",
            side_effect=lambda candidate: windows_path if candidate == windows_path else None,
        ):
            self.assertEqual(find_godot_executable("godot"), windows_path)

    def test_tilde_relative_executable_is_expanded(self):
        home = self.executable.parent
        (home / "bin").mkdir()
        (home / "bin" / "godot").touch()
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            resolved = find_godot_executable("~/bin/godot")
        self.assertEqual(resolved, str(home / "bin" / "godot"))

    def test_save_and_load_roundtrip(self):
        saved_to = save_godot_executable_setting(str(self.executable))
        self.assertEqual(saved_to, self.settings_path)
        self.assertEqual(load_godot_executable_setting(), str(self.executable))
        # The stored value is absolute, so later runs work from any cwd.
        self.assertTrue(Path(load_godot_executable_setting()).is_absolute())

    def test_save_refuses_unresolvable_executables(self):
        self.assertIsNone(save_godot_executable_setting("definitely_not_a_real_godot_binary"))
        self.assertFalse(self.settings_path.exists())

    def test_save_refuses_locations_outside_a_checkout(self):
        # A pip-installed (site-packages) copy must never scatter settings
        # files outside a real SandboxAI checkout.
        outside = Path(self._tmp.name) / "elsewhere" / ".sandboxai" / "settings.json"
        with mock.patch("sandboxai.config._settings_path", return_value=outside):
            self.assertIsNone(save_godot_executable_setting(str(self.executable)))
        self.assertFalse(outside.exists())

    def test_load_ignores_missing_or_corrupt_settings(self):
        self.assertIsNone(load_godot_executable_setting())
        self.settings_path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(load_godot_executable_setting())
        self.settings_path.write_text(json.dumps({"godot_executable": 42}), encoding="utf-8")
        self.assertIsNone(load_godot_executable_setting())
        # A remembered path that no longer exists is ignored, never fatal.
        self._write_setting(str(self.executable.with_name("deleted.exe")))
        with mock.patch("sandboxai.config._GODOT_CANDIDATES", ()):
            self.assertEqual(find_godot_executable("godot"), "godot")


if __name__ == "__main__":
    unittest.main()
