import unittest

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from sandboxai.bc import create_bc_policy


@unittest.skipUnless(torch is not None, "PyTorch is optional in the static test environment")
class BehaviorCloningTests(unittest.TestCase):
    def test_model_creation_and_action_shape(self):
        model = create_bc_policy(17)
        output = model.predict(torch.zeros((3, 17)))
        self.assertEqual(tuple(output.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
