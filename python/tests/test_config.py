import unittest

from sandboxai.config import TrainingConfig


class ConfigTests(unittest.TestCase):
    def test_training_config_validates_and_detects_cpu(self):
        config = TrainingConfig(environment_count=2, rollout_length=32, batch_size=16, device="cpu").validate()
        self.assertEqual(config.resolved_device(), "cpu")
        self.assertEqual(config.to_dict()["environment_count"], 2)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(environment_count=2, rollout_length=4, batch_size=9).validate()


if __name__ == "__main__":
    unittest.main()
