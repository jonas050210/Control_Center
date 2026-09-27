import json
from pathlib import Path
import tempfile
import unittest

from sandboxai.dataset import DemonstrationDataset, DemonstrationRecorder, action_to_multidiscrete


class DatasetTests(unittest.TestCase):
    def test_action_encoding_and_round_trip(self):
        self.assertEqual(action_to_multidiscrete([0, 0, 0, 0, 0, 0.2, -0.1]), [1, 1, 1, 1, 0])
        self.assertEqual(action_to_multidiscrete(9), [1, 1, 1, 1, 1])
        recorder = DemonstrationRecorder({"source": "test"})
        recorder.start()
        recorder.append([0.0] * 17, [0, 0, 0, 0, 0, 0, 0], [0.1] * 17, 1.0, True, episode_id=2)
        recorder.stop()
        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save(Path(directory) / "demo.jsonl")
            loaded = DemonstrationDataset.load(path)
            self.assertEqual(len(loaded.transitions), 1)
            self.assertEqual(loaded.summary()["terminal_transitions"], 1)

    def test_missing_fields_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"observation": [0.0]}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                DemonstrationDataset.load(path)


if __name__ == "__main__":
    unittest.main()
