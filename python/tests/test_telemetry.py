import json
from pathlib import Path
import tempfile
import unittest

from sandboxai.telemetry import JsonlTelemetry, resource_snapshot


class TelemetryTests(unittest.TestCase):
    def test_resource_snapshot(self):
        snapshot = resource_snapshot()
        self.assertIsInstance(snapshot, dict)

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
