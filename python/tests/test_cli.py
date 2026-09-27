import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
