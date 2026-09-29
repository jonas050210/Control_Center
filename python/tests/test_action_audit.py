"""Policy-head and shoot-path diagnostics."""
from __future__ import annotations

import unittest

from optional_deps import HAS_SB3, SB3_REASON
from sandboxai.action_audit import (
    EpisodeActionAudit,
    component_probabilities,
    shoot_probability_at,
    summarize_action_pipeline,
)


class ActionAuditTests(unittest.TestCase):
    def test_counts_exact_multidiscrete_outputs(self):
        audit = EpisodeActionAudit()
        audit.record([1, 1, 2, 0, 1, 0], shoot_probability=0.4)
        audit.record([2, 1, 1, 1, 0, 1], shoot_probability=0.2)
        summary = audit.summary()
        self.assertEqual(summary["policy_action_decisions"], 2)
        self.assertEqual(summary["policy_shoot_requests"], 1)
        self.assertEqual(summary["policy_action_counts"]["shoot"], [1, 1])
        self.assertAlmostEqual(summary["policy_mean_shoot_probability"], 0.3)

    def test_pipeline_summary_localizes_a_policy_side_zero(self):
        row = {
            **EpisodeActionAudit().summary(),
            "trigger_pulls": 0,
            "shots_fired": 0,
        }
        result = summarize_action_pipeline([row])
        self.assertEqual(result["localization"], "policy_argmax_never_requested_shoot")

    def test_pipeline_summary_checks_request_delivery_and_discharge(self):
        audit = EpisodeActionAudit()
        audit.record([1, 1, 1, 1, 1, 0])
        audit.record([1, 1, 1, 1, 1, 0])
        result = summarize_action_pipeline(
            [{**audit.summary(), "trigger_pulls": 2, "shots_fired": 1}]
        )
        self.assertEqual(result["localization"], "weapon_discharged")
        self.assertEqual(result["request_trigger_difference"], 0)
        self.assertEqual(result["request_delivery_rate"], 1.0)
        self.assertEqual(result["fire_conversion_rate"], 0.5)


@unittest.skipUnless(HAS_SB3, SB3_REASON)
class SB3ActionHeadTests(unittest.TestCase):
    def test_initial_deterministic_no_shoot_can_coexist_with_stochastic_exploration(self):
        import gymnasium as gym
        import numpy as np
        import torch
        from stable_baselines3 import PPO

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(84,), dtype=np.float32)
            action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3, 2, 2])

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(84, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(84, dtype=np.float32), 0.0, False, False, {}

        model = PPO(
            "MlpPolicy",
            TinyEnv(),
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=11,
            device="cpu",
            verbose=0,
        )
        observation = np.zeros(84, dtype=np.float32)
        deterministic_action, _state = model.predict(observation, deterministic=True)
        self.assertEqual(int(deterministic_action[4]), 0)

        rng_before = torch.random.get_rng_state().clone()
        probabilities = component_probabilities(model, observation)
        rng_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(rng_before, rng_after), "diagnostics must not sample")
        shoot_probability = shoot_probability_at(probabilities)
        self.assertIsNotNone(shoot_probability)
        self.assertAlmostEqual(float(shoot_probability), 0.5, places=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
