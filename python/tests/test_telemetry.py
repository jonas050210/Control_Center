import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sandboxai.telemetry import JsonlTelemetry, _nvidia_smi_snapshot, resource_snapshot


class TelemetryTests(unittest.TestCase):
    def test_resource_snapshot(self):
        snapshot = resource_snapshot()
        self.assertIsInstance(snapshot, dict)

    def test_nvidia_smi_snapshot_parses_gpu_utilization(self):
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4060 Ti, 37, 1024, 8192, 61\n",
            stderr="",
        )
        with patch("sandboxai.telemetry.subprocess.run", return_value=completed):
            snapshot = _nvidia_smi_snapshot()
        self.assertEqual(snapshot["gpu_name"], "NVIDIA GeForce RTX 4060 Ti")
        self.assertEqual(snapshot["gpu_telemetry_source"], "nvidia-smi")
        self.assertAlmostEqual(snapshot["gpu_utilization_percent"], 37.0)
        self.assertAlmostEqual(snapshot["gpu_vram_used_mb"], 1024.0)
        self.assertAlmostEqual(snapshot["gpu_vram_total_mb"], 8192.0)
        self.assertAlmostEqual(snapshot["gpu_temperature_c"], 61.0)

    def test_nvidia_smi_snapshot_handles_missing_command(self):
        with patch("sandboxai.telemetry.subprocess.run", side_effect=OSError("missing")):
            self.assertEqual(_nvidia_smi_snapshot(), {})

    def test_jsonl_telemetry_write(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_telem.jsonl"
            with JsonlTelemetry(path) as telem:
                telem.write({"step": 100, "reward": 5.0})
                telem.write({"step": 200, "reward": 10.0})
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n")]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["step"], 100)
            self.assertEqual(lines[1]["reward"], 10.0)


if __name__ == "__main__":
    unittest.main()
