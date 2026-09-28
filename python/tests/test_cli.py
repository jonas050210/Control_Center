import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from optional_deps import HAS_TORCH, TORCH_REASON
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

    def test_help_lists_workflow_commands(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "--help"], capture_output=True, text=True, check=True, env=_subprocess_env())
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
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_p = Path(tmp_dir) / "base.json"
            cand_p = Path(tmp_dir) / "cand.json"
            base_p.write_text(json.dumps({"metrics": {"win_rate": {"mean": 0.8, "std": 0.05, "count": 3}}}), encoding="utf-8")
            cand_p.write_text(json.dumps({"metrics": {"win_rate": {"mean": 0.85, "std": 0.04, "count": 3}}}), encoding="utf-8")
            exit_code = main(["compare-experiments", "--baseline", str(base_p), "--candidate", str(cand_p), "--json"])
            self.assertEqual(exit_code, 0)

    def test_install_is_dependency_only_and_does_not_launch_godot(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "install"], capture_output=True, text=True, check=True, env=_subprocess_env())
        self.assertIn("pip install", result.stdout)

    @unittest.skipUnless(HAS_TORCH, TORCH_REASON)
    def test_smoke_test_command(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "smoke-test", "--device", "cpu"], capture_output=True, text=True, check=True, env=_subprocess_env())
        self.assertIn("all_passed", result.stdout)
        # A failing smoke test must exit non-zero, not report success.
        self.assertIn('"all_passed": true', result.stdout)
        self.assertNotIn("sb3_weight_transfer_error", result.stdout)


if __name__ == "__main__":
    unittest.main()
