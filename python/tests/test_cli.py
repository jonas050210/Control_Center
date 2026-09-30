import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from optional_deps import HAS_TKINTER, HAS_TORCH, TKINTER_REASON, TORCH_REASON

from sandboxai.cli import build_control_center_command, build_record_command, main

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict:
    """Environment that lets a child ``python -m sandboxai`` find the package.

    The package lives in ``python/`` and is not necessarily pip-installed in
    the environment running the tests (conftest.py only patches the in-process
    ``sys.path``). Without this the CLI subprocess tests only pass when the
    caller happened to export PYTHONPATH by hand.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    entries = [str(PACKAGE_ROOT)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


class CliTests(unittest.TestCase):
    def test_build_record_command_shape(self):
        command = build_record_command("godot", "", "training/datasets/demo.jsonl", 12.5, 2)
        self.assertIn("--script", command)
        self.assertIn("res://scripts/recording/record_demo.gd", command)
        # The recorder is a graphical session: it must NOT run headless.
        self.assertNotIn("--headless", command)
        # Output must be resolved to an absolute path so the Godot process
        # working directory cannot change where the dataset lands.
        output_value = command[command.index("--output") + 1]
        self.assertTrue(Path(output_value).is_absolute())
        self.assertEqual(command[command.index("--duration") + 1], "12.5")
        self.assertEqual(command[command.index("--enemy-count") + 1], "2")

    def test_record_command_is_dispatched(self):
        # Regression: `record` used to be advertised by the parser but had no
        # handler in main(), crashing with "unhandled command record".
        with mock.patch("sandboxai.cli.subprocess.call", return_value=0) as call:
            exit_code = main(["record", "--output", "demo.jsonl", "--duration", "1"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(call.call_count, 1)
        launched = call.call_args[0][0]
        self.assertIn("res://scripts/recording/record_demo.gd", launched)

    def test_build_control_center_command_is_graphical_and_passes_settings(self):
        command = build_control_center_command("godot", "", "human", 2, 3, 4, 77, "duel")
        # The Control Center is an operator tool: it opens a real window and
        # must never be launched headless (that is the training path).
        self.assertNotIn("--headless", command)
        self.assertIn("res://scenes/control_center.tscn", command)
        # Scene arguments are passed after the "--" separator so Godot does
        # not try to interpret them itself.
        separator = command.index("--")
        user_args = command[separator + 1 :]
        self.assertIn("--mode=human", user_args)
        self.assertIn("--env-count=2", user_args)
        self.assertIn("--enemy-count=3", user_args)
        self.assertIn("--curriculum-level=4", user_args)
        self.assertIn("--seed=77", user_args)
        self.assertIn("--scenario=duel", user_args)

    def test_control_center_command_omits_empty_scenario(self):
        command = build_control_center_command("godot", "", "watch", 1, 1, 3, 1234)
        self.assertFalse([arg for arg in command if arg.startswith("--scenario")])

    def test_control_center_command_is_dispatched(self):
        with mock.patch("sandboxai.cli.subprocess.call", return_value=0) as call:
            exit_code = main(["control-center", "--mode", "watch", "--env-count", "2"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(call.call_count, 1)
        launched = call.call_args[0][0]
        self.assertIn("res://scenes/control_center.tscn", launched)
        self.assertIn("--env-count=2", launched)

    def test_control_center_reports_missing_godot_instead_of_crashing(self):
        with mock.patch("sandboxai.cli.subprocess.call", side_effect=OSError("not found")):
            exit_code = main(["control-center"])
        self.assertEqual(exit_code, 1)

    @unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
    def test_control_center_desktop_is_dispatched_with_project_and_output_root(self):
        # Importing sandboxai.control_center_desktop (even just to patch its
        # main()) requires Tkinter; this never opens a real window because
        # main() itself is replaced before cli.main() reaches it.
        with mock.patch("sandboxai.control_center_desktop.main", return_value=0) as desktop_main:
            exit_code = main(
                [
                    "control-center-desktop",
                    "--project-path",
                    "/tmp/some-project",
                    "--output-root",
                    "runs-out",
                ]
            )
        self.assertEqual(exit_code, 0)
        desktop_main.assert_called_once_with(
            project_root="/tmp/some-project", output_root="runs-out"
        )

    @unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
    def test_control_center_desktop_defaults_to_no_explicit_project_path(self):
        with mock.patch("sandboxai.control_center_desktop.main", return_value=0) as desktop_main:
            main(["control-center-desktop"])
        desktop_main.assert_called_once_with(project_root=None, output_root="training")

    def test_help_lists_workflow_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "sandboxai", "--help"],
            capture_output=True,
            text=True,
            check=True,
            env=_subprocess_env(),
        )
        for command in (
            "train",
            "evaluate",
            "record",
            "control-center",
            "bc-train",
            "resume",
            "benchmark",
            "inspect-dataset",
            "smoke-test",
            "validate-runtime",
            "compare-experiments",
            "summarize-experiment",
        ):
            self.assertIn(command, result.stdout)

    def test_validate_runtime_cli_dispatched(self):
        exit_code = main(["validate-runtime", "--godot-executable", "missing_godot", "--json"])
        self.assertEqual(exit_code, 0)

    def test_compare_experiments_cli_dispatched(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_p = Path(tmp_dir) / "base.json"
            cand_p = Path(tmp_dir) / "cand.json"
            base_p.write_text(
                json.dumps({"metrics": {"win_rate": {"mean": 0.8, "std": 0.05, "count": 3}}}),
                encoding="utf-8",
            )
            cand_p.write_text(
                json.dumps({"metrics": {"win_rate": {"mean": 0.85, "std": 0.04, "count": 3}}}),
                encoding="utf-8",
            )
            exit_code = main(
                [
                    "compare-experiments",
                    "--baseline",
                    str(base_p),
                    "--candidate",
                    str(cand_p),
                    "--json",
                ]
            )
            self.assertEqual(exit_code, 0)

    def test_install_is_dependency_only_and_does_not_launch_godot(self):
        result = subprocess.run(
            [sys.executable, "-m", "sandboxai", "install"],
            capture_output=True,
            text=True,
            check=True,
            env=_subprocess_env(),
        )
        self.assertIn("pip install", result.stdout)

    @unittest.skipUnless(HAS_TORCH, TORCH_REASON)
    def test_smoke_test_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "sandboxai", "smoke-test", "--device", "cpu"],
            capture_output=True,
            text=True,
            check=True,
            env=_subprocess_env(),
        )
        self.assertIn("all_passed", result.stdout)
        # A failing smoke test must exit non-zero, not report success.
        self.assertIn('"all_passed": true', result.stdout)
        self.assertNotIn("sb3_weight_transfer_error", result.stdout)


class _StopTraining(Exception):
    """Raised by the fake transport to stop train_ppo right after capture."""


class _FakeCheckoutTestCase(unittest.TestCase):
    """Shared fixture: a hermetic settings location inside a fake checkout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        # The settings file lives under the fake checkout's .sandboxai/, so
        # these tests never touch (nor depend on) the developer's real
        # .sandboxai/settings.json or its remembered Godot executable.
        self.checkout = tmp / "checkout"
        (self.checkout / ".sandboxai").mkdir(parents=True)
        (self.checkout / "project.godot").touch()
        self.settings_path = self.checkout / ".sandboxai" / "settings.json"
        self.output_root = tmp / "runs"
        self.executable = tmp / "Godot_v4.7.2-stable_win64_console.exe"
        self.executable.touch()
        patcher = mock.patch("sandboxai.config._settings_path", return_value=self.settings_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def read_remembered_executable(self):
        if not self.settings_path.exists():
            return None
        return json.loads(self.settings_path.read_text(encoding="utf-8")).get("godot_executable")


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TrainGodotExecutablePropagationTests(_FakeCheckoutTestCase):
    """The configured Godot executable must reach the training transport.

    Regression (Windows WinError 2): `validate-runtime --godot-executable X`
    worked, but `python -m sandboxai train --steps ... --env-count ...
    --curriculum-level ...` still launched bare `godot` from PATH and died.
    These tests pin the whole CLI -> train_ppo -> GodotVecEnv ->
    GodotBatchClient -> GodotProcessTransport configuration chain.
    """

    def _run_train_capturing_transport(self, argv):
        captured = {}

        class FakeTransport:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                raise _StopTraining()

        with mock.patch("sandboxai.godot_env.GodotProcessTransport", FakeTransport):
            with contextlib.suppress(_StopTraining):
                main(argv)
        return captured

    def test_explicit_godot_executable_reaches_the_training_transport(self):
        captured = self._run_train_capturing_transport(
            [
                "train",
                "--steps",
                "2048",
                "--env-count",
                "2",
                "--curriculum-level",
                "3",
                "--godot-executable",
                str(self.executable),
                "--output-root",
                str(self.output_root),
            ]
        )
        self.assertEqual(captured.get("godot_executable"), str(self.executable))
        # The rest of the environment configuration keeps flowing too.
        self.assertEqual(captured.get("environment_count"), 2)
        self.assertEqual(captured.get("curriculum_level"), 3)
        # A verified explicit executable is remembered for later runs.
        self.assertEqual(self.read_remembered_executable(), str(self.executable))

    def test_train_defaults_to_godot_when_nothing_is_configured(self):
        # No flag, no remembered setting: the documented default (`godot` on
        # PATH) is passed to the transport unchanged.
        captured = self._run_train_capturing_transport(
            [
                "train",
                "--steps",
                "2048",
                "--env-count",
                "2",
                "--curriculum-level",
                "3",
                "--output-root",
                str(self.output_root),
            ]
        )
        self.assertEqual(captured.get("godot_executable"), "godot")

    def test_train_uses_remembered_godot_executable_without_the_flag(self):
        # The reported bug, verbatim: configure the executable once (here:
        # as if a previous validate-runtime had remembered it), then run
        # `train` without --godot-executable — the transport must receive
        # the configured binary, not the PATH default.
        self.settings_path.write_text(
            json.dumps({"godot_executable": str(self.executable)}), encoding="utf-8"
        )
        captured = self._run_train_capturing_transport(
            [
                "train",
                "--steps",
                "2048",
                "--env-count",
                "2",
                "--curriculum-level",
                "3",
                "--output-root",
                str(self.output_root),
            ]
        )
        self.assertEqual(captured.get("godot_executable"), str(self.executable))
        # Provenance: the run's saved config records the binary actually used.
        config_files = list(Path(self.output_root).glob("**/config.json"))
        self.assertTrue(config_files, "train_ppo must save config.json before building the env")
        saved = json.loads(config_files[0].read_text(encoding="utf-8"))
        self.assertEqual(saved["godot_executable"], str(self.executable))

    def test_config_file_godot_executable_wins_over_remembered(self):
        config_file = Path(self._tmp.name) / "training_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "environment_count": 2,
                    "rollout_length": 64,
                    "batch_size": 32,
                    "total_training_steps": 2048,
                    "godot_executable": "godot_from_config_file",
                }
            ),
            encoding="utf-8",
        )
        self.settings_path.write_text(
            json.dumps({"godot_executable": str(self.executable)}), encoding="utf-8"
        )
        captured = self._run_train_capturing_transport(
            [
                "train",
                "--config",
                str(config_file),
                "--output-root",
                str(self.output_root),
            ]
        )
        self.assertEqual(captured.get("godot_executable"), "godot_from_config_file")


class RememberedGodotExecutableTests(_FakeCheckoutTestCase):
    """How the CLI remembers (and refuses to remember) an executable."""

    def _run_validate_runtime(self, executable):
        # RuntimeValidator is mocked: the persistence hook must run for the
        # command exactly as it would before any real Godot launch.
        fake_report = mock.MagicMock()
        fake_report.status = "unavailable"
        fake_report.to_dict.return_value = {"status": "unavailable"}
        with mock.patch("sandboxai.runtime_validation.RuntimeValidator") as validator_cls:
            validator_cls.return_value.validate.return_value = fake_report
            exit_code = main(["validate-runtime", "--godot-executable", executable, "--json"])
        self.assertEqual(exit_code, 0)
        validator_cls.assert_called_once_with(
            project_path="", godot_executable=executable, timeout=15.0
        )

    def test_validate_runtime_remembers_explicit_executable(self):
        self._run_validate_runtime(str(self.executable))
        self.assertEqual(self.read_remembered_executable(), str(self.executable))

    def test_unresolvable_executable_is_not_remembered(self):
        self.settings_path.write_text(
            json.dumps({"godot_executable": str(self.executable)}), encoding="utf-8"
        )
        self._run_validate_runtime("definitely_not_a_real_godot_binary")
        # The typo must not poison the previously remembered setting.
        self.assertEqual(self.read_remembered_executable(), str(self.executable))

    def test_default_godot_flag_is_not_remembered(self):
        self.settings_path.write_text(
            json.dumps({"godot_executable": str(self.executable)}), encoding="utf-8"
        )
        self._run_validate_runtime("godot")
        self.assertEqual(self.read_remembered_executable(), str(self.executable))


if __name__ == "__main__":
    unittest.main()
