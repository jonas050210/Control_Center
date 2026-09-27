import unittest

from sandboxai.config import BCConfig, TrainingConfig, find_godot_executable


class ConfigTests(unittest.TestCase):
    def test_training_config_validates_and_detects_cpu(self):
        config = TrainingConfig(environment_count=2, rollout_length=32, batch_size=16, device="cpu").validate()
        self.assertEqual(config.resolved_device(), "cpu")
        self.assertEqual(config.to_dict()["environment_count"], 2)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(environment_count=2, rollout_length=4, batch_size=9).validate()

    def test_invalid_curriculum_level_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(curriculum_level=6).validate()

    def test_bc_config_validation(self):
        config = BCConfig(epochs=10, batch_size=64, early_stopping_patience=3).validate()
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.early_stopping_patience, 3)

    def test_find_godot_executable(self):
        exe = find_godot_executable("godot")
        self.assertIsInstance(exe, str)
        self.assertTrue(len(exe) > 0)


if __name__ == "__main__":
    unittest.main()
