import random
import unittest
from unittest.mock import MagicMock, patch

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.self_play import (
    PolicySlot,
    SelfPlayBatchClient,
    SelfPlayCoordinator,
    play_self_play_match,
)


class SelfPlayTests(unittest.TestCase):
    def test_coordinator_initialization_and_slots(self):
        slot_a = PolicySlot(name="learning_agent", checkpoint="")
        slot_b = PolicySlot(name="opponent_agent", checkpoint="training/checkpoints/opp.zip")
        coord = SelfPlayCoordinator(slot_a, slot_b, opponent_pool=["training/checkpoints/opp.zip", "training/checkpoints/opp2.zip"])
        self.assertTrue(coord.opponent_slot.frozen)
        self.assertEqual(len(coord.opponent_pool), 2)

    def test_record_match_and_summary(self):
        slot_a = PolicySlot(name="agent_a")
        slot_b = PolicySlot(name="agent_b")
        coord = SelfPlayCoordinator(slot_a, slot_b)
        coord.record_match({"win": True, "episode_reward": 10.0, "kills": 1, "accuracy": 0.8}, {"win": False, "episode_reward": -5.0, "kills": 0, "accuracy": 0.2})
        summary = coord.summary()
        self.assertEqual(summary["agent_a"]["matches"], 1)
        self.assertEqual(summary["agent_a"]["win_rate"], 1.0)
        self.assertEqual(summary["agent_b"]["win_rate"], 0.0)

    def test_play_self_play_match_end_to_end_flow(self):
        # Mock client simulating a 3-step match with agent A winning
        client = MagicMock(spec=SelfPlayBatchClient)
        obs_a = [0.1] * OBSERVATION_FIELD_COUNT
        obs_b = [-0.1] * OBSERVATION_FIELD_COUNT
        client.reset.return_value = [[obs_a, obs_b]]

        step_results = [
            ([[obs_a, obs_b]], [[0.1, -0.1]], [False], [[{"metrics": {"win": False}}, {"metrics": {"win": False}}]]),
            ([[obs_a, obs_b]], [[0.5, -0.5]], [False], [[{"metrics": {"win": False}}, {"metrics": {"win": False}}]]),
            ([[obs_a, obs_b]], [[10.0, -5.0]], [True], [[{"done_reason": "agent_a_win", "metrics": {"win": True, "kills": 1}}, {"done_reason": "agent_a_win", "metrics": {"win": False, "kills": 0}}]]),
        ]
        client.step.side_effect = step_results

        predict_a = MagicMock(return_value=[1, 1, 1, 1, 1, 0])
        predict_b = MagicMock(return_value=[1, 1, 1, 1, 0, 0])

        result = play_self_play_match(client, predict_a, predict_b, seed=42, max_steps=100)

        self.assertEqual(result["score_a"], 1.0)
        self.assertEqual(result["steps"], 3)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["done_reason"], "agent_a_win")
        self.assertTrue(result["metrics_a"]["win"])
        self.assertFalse(result["metrics_b"]["win"])
        self.assertEqual(predict_a.call_count, 3)
        self.assertEqual(predict_b.call_count, 3)

    def test_sample_opponent_without_explicit_rng_is_deterministic(self):
        # Regression: sample_opponent() used to fall back to the unseeded
        # global `random` module when no rng was supplied, so two
        # coordinators built identically could draw different opponents on
        # different runs. It must now fall back to an RNG seeded from the
        # opponent slot's own `seed` field.
        pool = ["ckpt_a.zip", "ckpt_b.zip", "ckpt_c.zip", "ckpt_d.zip"]

        def build():
            slot_a = PolicySlot(name="learning_agent", checkpoint="")
            slot_b = PolicySlot(name="opponent_agent", checkpoint="", seed=777)
            return SelfPlayCoordinator(slot_a, slot_b, opponent_pool=list(pool))

        with patch.object(PolicySlot, "load", lambda self, device="cpu": None):
            coord1 = build()
            coord2 = build()
            drawn1 = [coord1.sample_opponent().checkpoint for _ in range(10)]
            drawn2 = [coord2.sample_opponent().checkpoint for _ in range(10)]
            self.assertEqual(drawn1, drawn2)

            # Different opponent-slot seeds should (almost certainly) diverge.
            slot_a = PolicySlot(name="learning_agent", checkpoint="")
            slot_b = PolicySlot(name="opponent_agent", checkpoint="", seed=999)
            coord3 = SelfPlayCoordinator(slot_a, slot_b, opponent_pool=list(pool))
            drawn3 = [coord3.sample_opponent().checkpoint for _ in range(10)]
            self.assertNotEqual(drawn1, drawn3)

            # An explicit rng always takes priority over the fallback.
            coord4 = build()
            explicit_rng = random.Random(12345)
            drawn4 = [coord4.sample_opponent(rng=explicit_rng).checkpoint for _ in range(10)]
            replay_rng = random.Random(12345)
            replay = [replay_rng.choice(pool) for _ in range(10)]
            self.assertEqual(drawn4, replay)

    def test_play_self_play_match_timeout_draw(self):
        client = MagicMock(spec=SelfPlayBatchClient)
        obs_a = [0.0] * OBSERVATION_FIELD_COUNT
        obs_b = [0.0] * OBSERVATION_FIELD_COUNT
        client.reset.return_value = [[obs_a, obs_b]]
        client.step.return_value = ([[obs_a, obs_b]], [[0.0, 0.0]], [False], [[{}, {}]])

        predict_a = lambda obs: [1, 1, 1, 1, 0, 0]
        predict_b = lambda obs: [1, 1, 1, 1, 0, 0]

        result = play_self_play_match(client, predict_a, predict_b, seed=99, max_steps=5)
        self.assertEqual(result["score_a"], 0.5)
        self.assertEqual(result["steps"], 5)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["done_reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
