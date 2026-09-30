import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from sandboxai.bc import create_bc_policy, load_bc_checkpoint
from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class CheckpointTests(unittest.TestCase):
    def test_bc_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            model = create_bc_policy(OBSERVATION_FIELD_COUNT)
            path = Path(directory) / "bc.pt"
            torch.save(
                {
                    "format": "sandboxai.bc.v1",
                    "observation_dim": OBSERVATION_FIELD_COUNT,
                    "hidden_sizes": [128, 128],
                    "action_nvec": list(ACTION_NVEC),
                    "model_state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_bc_checkpoint(path)
            self.assertEqual(loaded.observation_dim, OBSERVATION_FIELD_COUNT)
            self.assertEqual(
                tuple(loaded.predict(torch.zeros((1, OBSERVATION_FIELD_COUNT))).shape),
                (1, len(ACTION_NVEC)),
            )


if __name__ == "__main__":
    unittest.main()
