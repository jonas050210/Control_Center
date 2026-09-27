import unittest

from sandboxai.self_play import PolicySlot, SelfPlayCoordinator


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


if __name__ == "__main__":
    unittest.main()
