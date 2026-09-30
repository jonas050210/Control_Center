import json
import tempfile
import unittest
from pathlib import Path

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.dataset import DemonstrationDataset, DemonstrationRecorder, action_to_multidiscrete


class DatasetTests(unittest.TestCase):
    def test_action_encoding_and_round_trip(self):
        self.assertEqual(action_to_multidiscrete([0, 0, 0, 0, 0, 0.2, -0.1]), [1, 1, 1, 1, 0, 0])
        # contract v2 log array: jump lives at index 5, look deltas shift right.
        self.assertEqual(action_to_multidiscrete([0, 0, 0, 0, 0, 1, 0.2, -0.1]), [1, 1, 1, 1, 0, 1])
        self.assertEqual(action_to_multidiscrete(9), [1, 1, 1, 1, 1, 0])
        self.assertEqual(action_to_multidiscrete(10), [1, 1, 1, 1, 0, 1])
        recorder = DemonstrationRecorder({"source": "test"})
        recorder.start()
        recorder.append(
            [0.0] * OBSERVATION_FIELD_COUNT,
            [0, 0, 0, 0, 0, 0, 0],
            [0.1] * OBSERVATION_FIELD_COUNT,
            1.0,
            True,
            episode_id=2,
        )
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

    def test_nan_observation_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.jsonl"
            bad_obs = [0.0] * (OBSERVATION_FIELD_COUNT - 1) + [float("nan")]
            record = {
                "observation": bad_obs,
                "action": [0, 0, 0, 0, 0, 0, 0],
                "next_observation": [0.0] * OBSERVATION_FIELD_COUNT,
                "reward": 0.0,
                "done": False,
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                DemonstrationDataset.load(path)

    def test_split_dataset(self):
        recorder = DemonstrationRecorder({"source": "split_test"})
        recorder.start()
        for i in range(10):
            recorder.append(
                [0.1 * i] * OBSERVATION_FIELD_COUNT,
                [0, 0, 0, 0, 0, 0, 0],
                [0.1 * (i + 1)] * OBSERVATION_FIELD_COUNT,
                0.5,
                i == 9,
            )
        recorder.stop()
        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save(Path(directory) / "split_demo.jsonl")
            dataset = DemonstrationDataset.load(path)
            train, val = dataset.split(0.2, seed=42)
            self.assertEqual(len(train.transitions) + len(val.transitions), 10)
            self.assertEqual(len(val.transitions), 2)


if __name__ == "__main__":
    unittest.main()
