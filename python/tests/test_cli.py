import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from sandboxai.cli import build_record_command, main


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
    def test_help_lists_workflow_commands(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "--help"], capture_output=True, text=True, check=True)
        for command in ("train", "evaluate", "record", "bc-train", "resume", "benchmark", "inspect-dataset", "smoke-test"):
            self.assertIn(command, result.stdout)

    def test_install_is_dependency_only_and_does_not_launch_godot(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "install"], capture_output=True, text=True, check=True)
        self.assertIn("pip install", result.stdout)

    def test_smoke_test_command(self):
        result = subprocess.run([sys.executable, "-m", "sandboxai", "smoke-test", "--device", "cpu"], capture_output=True, text=True, check=True)
        self.assertIn("all_passed", result.stdout)
        # A failing smoke test must exit non-zero, not report success.
        self.assertIn('"all_passed": true', result.stdout)
        self.assertNotIn("sb3_weight_transfer_error", result.stdout)


if __name__ == "__main__":
    unittest.main()
