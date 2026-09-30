import unittest

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from sandboxai.bc import create_bc_policy
from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT


def _touch(path: str) -> None:
    """Stand-in for a malicious payload: observable, harmless, picklable."""
    from pathlib import Path

    Path(path).touch()


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class BehaviorCloningTests(unittest.TestCase):
    def test_model_creation_and_action_shape(self):
        model = create_bc_policy(OBSERVATION_FIELD_COUNT)
        output = model.predict(torch.zeros((3, OBSERVATION_FIELD_COUNT)))
        self.assertEqual(tuple(output.shape), (3, len(ACTION_NVEC)))

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
            recorder.append(
                [0.01 * (i + j) for j in range(OBSERVATION_FIELD_COUNT)],
                [0, 0, 0, 0, i % 2, 0.0, 0.0],
                [0.01 * (i + j + 1) for j in range(OBSERVATION_FIELD_COUNT)],
                0.1,
                i == 11,
            )
        recorder.stop()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = recorder.save(Path(tmp) / "demo.jsonl")
            config = BCConfig(epochs=1, batch_size=4, validation_fraction=0.25, output_root=tmp)
            first = train_behavior_cloning(dataset, config, output_dir=Path(tmp) / "run")
            self.assertEqual(first["epochs"], 1)
            resumed = train_behavior_cloning(
                dataset,
                config,
                output_dir=Path(tmp) / "run",
                resume_checkpoint=first["latest_checkpoint"],
            )
            self.assertEqual(resumed["epochs"], 1)

    def _episode_dataset(self, directory, episodes=6, length=8, observation_dim=None):
        import json
        from pathlib import Path

        dimension = observation_dim or OBSERVATION_FIELD_COUNT
        path = Path(directory) / "episodes.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": "sandboxai.demonstrations"}) + "\n")
            for episode in range(episodes):
                for step in range(length):
                    value = 0.01 * (episode * length + step)
                    stream.write(
                        json.dumps(
                            {
                                "observation": [value + 0.001 * j for j in range(dimension)],
                                "action": [0, 0, 0, 0, step % 2, 0, 0.0, 0.0],
                                "next_observation": [value + 0.002 * j for j in range(dimension)],
                                "reward": 0.1,
                                "done": step == length - 1,
                                "episode_id": episode,
                                "environment_id": 0,
                                "step": step,
                            }
                        )
                        + "\n"
                    )
        return path

    def test_training_uses_an_episode_split_and_records_its_provenance(self):
        import json
        import tempfile
        from pathlib import Path

        from sandboxai.bc import train_behavior_cloning
        from sandboxai.config import BCConfig

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._episode_dataset(tmp)
            config = BCConfig(epochs=1, batch_size=8, validation_fraction=0.25, output_root=tmp)
            result = train_behavior_cloning(dataset, config, output_dir=Path(tmp) / "run")
            self.assertEqual(result["split"]["strategy"], "episode")
            self.assertTrue(result["split"]["leakage_free"])
            self.assertEqual(result["split"]["shared_groups"], 0)
            report = json.loads(
                (Path(tmp) / "run" / "dataset_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["format"], "sandboxai.bc_dataset_report/v1")
            self.assertEqual(report["statistics"]["episodes"], 6)
            self.assertTrue(report["fingerprint"].startswith("blake2b:"))
            self.assertEqual(result["dataset_fingerprint"], report["fingerprint"])

    def test_training_refuses_a_dataset_that_is_not_the_observation_contract(self):
        import tempfile
        from pathlib import Path

        from sandboxai.bc import train_behavior_cloning
        from sandboxai.config import BCConfig

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._episode_dataset(tmp, observation_dim=OBSERVATION_FIELD_COUNT - 1)
            config = BCConfig(epochs=1, batch_size=8, output_root=tmp)
            with self.assertRaises(ValueError):
                train_behavior_cloning(dataset, config, output_dir=Path(tmp) / "run")

    def test_training_refuses_a_mostly_duplicated_dataset(self):
        import json
        import tempfile
        from pathlib import Path

        from sandboxai.bc import train_behavior_cloning
        from sandboxai.config import BCConfig

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicated.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for episode in range(4):
                    for step in range(10):
                        stream.write(
                            json.dumps(
                                {
                                    "observation": [0.5] * OBSERVATION_FIELD_COUNT,
                                    "action": [0, 0, 0, 0, 0, 0, 0.0, 0.0],
                                    "next_observation": [0.5] * OBSERVATION_FIELD_COUNT,
                                    "reward": 0.0,
                                    "done": step == 9,
                                    "episode_id": episode,
                                    "environment_id": 0,
                                    "step": step,
                                }
                            )
                            + "\n"
                        )
            config = BCConfig(epochs=1, batch_size=8, output_root=tmp)
            with self.assertRaises(ValueError) as caught:
                train_behavior_cloning(path, config, output_dir=Path(tmp) / "run")
            self.assertIn("duplicate", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class CheckpointLoadingIsSandboxedTests(unittest.TestCase):
    """Loading a checkpoint must not be able to run code.

    bc.py used to pass ``weights_only=False`` to ``torch.load`` at three
    call sites, which unpickles arbitrary objects: opening a checkpoint
    executed whatever its author had put in it. Nothing in a SandboxAI
    checkpoint needs that - ``train_bc`` writes tensors, strings,
    numbers, lists and dicts - so the flag was removed.

    The first test is the one that matters: it builds an actual hostile
    checkpoint and asserts the loader refuses it instead of running it.

    These use ``mkdtemp`` rather than ``TemporaryDirectory`` on purpose.
    A ``torch.load`` that fails part-way leaves its file handle open, and
    Windows refuses to delete an open file (WinError 32), so the context
    manager's cleanup - not the assertion - would fail the test there.
    """

    def _workspace(self):
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())

    @staticmethod
    def _hostile_checkpoint(directory):
        """Write a checkpoint whose unpickling would create a marker file.

        The payload is a module-level function rather than a closure or a
        bound method so that it pickles identically on every platform.
        """
        marker = directory / "code_executed.marker"

        class Payload:
            def __reduce__(self):
                return (_touch, (str(marker),))

        torch.save({"model_state_dict": {}, "payload": Payload()}, directory / "hostile.pt")
        return directory / "hostile.pt", marker

    def test_a_checkpoint_that_wants_to_run_code_is_refused(self):
        from sandboxai.bc import _load_checkpoint

        checkpoint, marker = self._hostile_checkpoint(self._workspace())
        with self.assertRaises(ValueError) as caught:
            _load_checkpoint(checkpoint, "cpu")
        self.assertFalse(
            marker.exists(),
            "loading the checkpoint executed its payload - weights_only is not in effect",
        )
        self.assertIn("weights_only=True", str(caught.exception))

    def test_the_payload_really_would_have_run_without_the_guard(self):
        """Proves the fixture is hostile, not just inert."""
        checkpoint, marker = self._hostile_checkpoint(self._workspace())
        self.assertFalse(marker.exists())
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertTrue(
            marker.exists(),
            "the test fixture does not actually execute code, so the guard test proves nothing",
        )

    def test_load_bc_checkpoint_refuses_it_too(self):
        from sandboxai.bc import load_bc_checkpoint

        checkpoint, marker = self._hostile_checkpoint(self._workspace())
        with self.assertRaises(ValueError):
            load_bc_checkpoint(checkpoint)
        self.assertFalse(marker.exists())

    def test_a_real_checkpoint_still_round_trips(self):
        """The safe loader must not have broken the normal path."""
        from sandboxai.bc import _atomic_torch_save, load_bc_checkpoint
        from sandboxai.contract import OBSERVATION_FIELD_COUNT

        model = create_bc_policy(OBSERVATION_FIELD_COUNT)
        path = self._workspace() / "best.pt"
        _atomic_torch_save(
            {
                "format": "sandboxai.bc.v1",
                "observation_dim": model.observation_dim,
                "hidden_sizes": list(model.hidden_sizes),
                "action_nvec": list(model.action_nvec),
                "model_state_dict": model.state_dict(),
                "metrics": {"epoch": 1, "train_loss": 0.25},
            },
            path,
        )
        restored = load_bc_checkpoint(path)
        self.assertEqual(restored.observation_dim, model.observation_dim)
        self.assertEqual(tuple(restored.hidden_sizes), tuple(model.hidden_sizes))

    def test_a_non_checkpoint_file_is_reported_clearly(self):
        from sandboxai.bc import _load_checkpoint

        junk = self._workspace() / "junk.pt"
        junk.write_bytes(b"this is not a torch archive")
        with self.assertRaises(ValueError):
            _load_checkpoint(junk, "cpu")
