import unittest

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from sandboxai.bc import create_bc_policy
from sandboxai.contract import OBSERVATION_FIELD_COUNT


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class BehaviorCloningTests(unittest.TestCase):
    def test_model_creation_and_action_shape(self):
        model = create_bc_policy(OBSERVATION_FIELD_COUNT)
        output = model.predict(torch.zeros((3, OBSERVATION_FIELD_COUNT)))
        self.assertEqual(tuple(output.shape), (3, 5))

    def test_resume_from_completed_run_returns_cleanly(self):
        # Regression: resuming with a checkpoint whose epoch >= config.epochs
        # used to crash with an UnboundLocalError on `epoch`.
        import tempfile
        from pathlib import Path

        from sandboxai.bc import train_behavior_cloning
        from sandboxai.config import BCConfig
        from sandboxai.dataset import DemonstrationRecorder

        recorder = DemonstrationRecorder({"source": "bc_resume_test"})
        recorder.start()
        for i in range(12):
            recorder.append([0.01 * (i + j) for j in range(OBSERVATION_FIELD_COUNT)],
                            [0, 0, 0, 0, i % 2, 0.0, 0.0],
                            [0.01 * (i + j + 1) for j in range(OBSERVATION_FIELD_COUNT)],
                            0.1, i == 11)
        recorder.stop()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = recorder.save(Path(tmp) / "demo.jsonl")
            config = BCConfig(epochs=1, batch_size=4, validation_fraction=0.25, output_root=tmp)
            first = train_behavior_cloning(dataset, config, output_dir=Path(tmp) / "run")
            self.assertEqual(first["epochs"], 1)
            resumed = train_behavior_cloning(
                dataset, config, output_dir=Path(tmp) / "run",
                resume_checkpoint=first["latest_checkpoint"],
            )
            self.assertEqual(resumed["epochs"], 1)


if __name__ == "__main__":
    unittest.main()
