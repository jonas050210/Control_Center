import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from sandboxai.bc import create_bc_policy, load_bc_checkpoint


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class CheckpointTests(unittest.TestCase):
    def test_bc_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            model = create_bc_policy(17)
            path = Path(directory) / "bc.pt"
            torch.save(
                {
                    "format": "sandboxai.bc.v1",
                    "observation_dim": 17,
                    "hidden_sizes": [128, 128],
                    "action_nvec": [3, 3, 3, 3, 2],
                    "model_state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_bc_checkpoint(path)
            self.assertEqual(loaded.observation_dim, 17)
            self.assertEqual(tuple(loaded.predict(torch.zeros((1, 17))).shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
