import unittest

from sandboxai.config import CURRICULUM_LEVEL_COUNT, BCConfig, TrainingConfig, find_godot_executable


class ConfigTests(unittest.TestCase):
    def test_training_config_validates_and_detects_cpu(self):
        config = TrainingConfig(environment_count=2, rollout_length=32, batch_size=16, device="cpu").validate()
        self.assertEqual(config.resolved_device(), "cpu")
        self.assertEqual(config.to_dict()["environment_count"], 2)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(environment_count=2, rollout_length=4, batch_size=9).validate()

    def test_invalid_curriculum_level_rejected(self):
        # 1-10 are combat levels, 11 is the self-play hook; 0 and 12 are not.
        with self.assertRaises(ValueError):
            TrainingConfig(curriculum_level=CURRICULUM_LEVEL_COUNT + 1).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(curriculum_level=0).validate()

    def test_all_declared_curriculum_levels_are_accepted(self):
        for level in range(1, CURRICULUM_LEVEL_COUNT + 1):
            TrainingConfig(curriculum_level=level).validate()

    def test_negative_entropy_coefficient_rejected(self):
        # Negative entropy actively rewards determinism; it must fail at
        # config load instead of silently collapsing the policy.
        with self.assertRaises(ValueError):
            TrainingConfig(entropy_coefficient=-0.01).validate()

    def test_invalid_evaluation_episodes_and_torch_threads_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(evaluation_episodes=0).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(torch_threads=-1).validate()

    def test_invalid_net_arch_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(net_arch=(128, 0)).validate()

    def test_valid_torch_threads_is_accepted(self):
        config = TrainingConfig(torch_threads=4).validate()
        self.assertEqual(config.torch_threads, 4)

    def test_entropy_coefficient_defaults_to_nonzero(self):
        # Regression: with MultiDiscrete actions, a 0 entropy coefficient lets
        # PPO collapse into a degenerate action distribution (e.g. never
        # shooting) in the first few updates. The default must keep gentle
        # exploration pressure so combat actions stay discoverable.
        config = TrainingConfig()
        self.assertGreaterEqual(config.entropy_coefficient, 0.005)

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
