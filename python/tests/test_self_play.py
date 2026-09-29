import unittest
from unittest.mock import MagicMock

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.self_play import (
    OPPONENT_STRATEGIES,
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


class OpponentSamplingTests(unittest.TestCase):
    """Opponent selection must be reproducible from (seed, strategy, pool).

    The previous default fell back to the unseeded global ``random``
    module, so two runs of the same configuration silently trained
    against different opponent sequences.
    """

    POOL = [f"pool/c{index}.zip" for index in range(1, 6)]

    def _coordinator(self, strategy: str = "uniform", seed: int = 7) -> SelfPlayCoordinator:
        return SelfPlayCoordinator(
            PolicySlot(name="learner"),
            PolicySlot(name="frozen"),
            opponent_pool=list(self.POOL),
            strategy=strategy,
            seed=seed,
        )

    def test_default_sampling_is_seeded_and_reproducible(self):
        a = self._coordinator()
        b = self._coordinator()
        draws_a = [a.choose_opponent_checkpoint() for _ in range(20)]
        draws_b = [b.choose_opponent_checkpoint() for _ in range(20)]
        self.assertEqual(draws_a, draws_b)

    def test_different_seeds_produce_different_streams(self):
        stream_1 = [self._coordinator(seed=1).choose_opponent_checkpoint() for _ in range(30)]
        stream_2 = [self._coordinator(seed=2).choose_opponent_checkpoint() for _ in range(30)]
        self.assertNotEqual(stream_1, stream_2)

    def test_reset_sampling_rewinds_the_stream(self):
        coordinator = self._coordinator()
        first = [coordinator.choose_opponent_checkpoint() for _ in range(10)]
        coordinator.reset_sampling()
        self.assertEqual(coordinator.opponent_history, [])
        self.assertEqual(first, [coordinator.choose_opponent_checkpoint() for _ in range(10)])

    def test_uniform_sampling_reaches_the_whole_pool(self):
        coordinator = self._coordinator()
        drawn = {coordinator.choose_opponent_checkpoint() for _ in range(200)}
        self.assertEqual(drawn, set(self.POOL))

    def test_latest_always_returns_the_newest_checkpoint(self):
        coordinator = self._coordinator(strategy="latest")
        self.assertEqual(
            {coordinator.choose_opponent_checkpoint() for _ in range(10)}, {self.POOL[-1]}
        )
        coordinator.add_to_pool("pool/c6.zip")
        self.assertEqual(coordinator.choose_opponent_checkpoint(), "pool/c6.zip")

    def test_round_robin_gives_equal_exposure_without_rng(self):
        coordinator = self._coordinator(strategy="round_robin")
        drawn = [coordinator.choose_opponent_checkpoint() for _ in range(len(self.POOL) * 3)]
        self.assertEqual(drawn, self.POOL * 3)

    def test_recency_weighted_prefers_recent_but_keeps_pool_reachable(self):
        coordinator = self._coordinator(strategy="recency_weighted")
        drawn = [coordinator.choose_opponent_checkpoint() for _ in range(400)]
        self.assertEqual(set(drawn), set(self.POOL))
        self.assertGreater(drawn.count(self.POOL[-1]), drawn.count(self.POOL[0]))

    def test_explicit_rng_overrides_the_owned_generator(self):
        import random as random_module

        coordinator = self._coordinator()
        with_rng = [
            coordinator.choose_opponent_checkpoint(rng=random_module.Random(11))
            for _ in range(5)
        ]
        other = self._coordinator()
        self.assertEqual(
            with_rng,
            [
                other.choose_opponent_checkpoint(rng=random_module.Random(11))
                for _ in range(5)
            ],
        )

    def test_empty_pool_returns_the_configured_opponent(self):
        coordinator = SelfPlayCoordinator(
            PolicySlot(name="learner"), PolicySlot(name="frozen")
        )
        self.assertIsNone(coordinator.choose_opponent_checkpoint())
        self.assertIs(coordinator.sample_opponent(), coordinator.opponent_slot)

    def test_sample_opponent_can_select_without_loading_a_model(self):
        coordinator = self._coordinator(strategy="latest")
        slot = coordinator.sample_opponent(load=False)
        self.assertEqual(slot.checkpoint, self.POOL[-1])
        self.assertTrue(slot.frozen)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            SelfPlayCoordinator(
                PolicySlot(name="a"), PolicySlot(name="b"), strategy="whatever"
            )
        with self.assertRaises(ValueError):
            self._coordinator().choose_opponent_checkpoint(strategy="whatever")

    def test_sampling_snapshot_records_the_reproducibility_inputs(self):
        coordinator = self._coordinator(strategy="round_robin", seed=3)
        for _ in range(3):
            coordinator.choose_opponent_checkpoint()
        snapshot = coordinator.sampling_snapshot()
        self.assertEqual(snapshot["strategy"], "round_robin")
        self.assertEqual(snapshot["seed"], 3)
        self.assertEqual(snapshot["draws"], 3)
        self.assertEqual(snapshot["pool_size"], len(self.POOL))
        self.assertEqual(snapshot["recent_opponents"], self.POOL[:3])

    def test_from_config_carries_the_selection_rule(self):
        from sandboxai.config import SelfPlayConfig

        config = SelfPlayConfig(
            opponent_checkpoint="pool/c5.zip",
            opponent_pool=list(self.POOL),
            opponent_strategy="round_robin",
            opponent_seed=5,
        ).validate()
        coordinator = SelfPlayCoordinator.from_config(config)
        self.assertEqual(coordinator.opponent_strategy, "round_robin")
        self.assertEqual(coordinator.opponent_seed, 5)
        self.assertEqual(coordinator.opponent_pool, list(self.POOL))
        self.assertTrue(coordinator.opponent_slot.frozen)
        self.assertEqual(coordinator.choose_opponent_checkpoint(), self.POOL[0])

    def test_config_rejects_unknown_strategy(self):
        from sandboxai.config import SelfPlayConfig

        with self.assertRaises(ValueError):
            SelfPlayConfig(opponent_strategy="nope").validate()

    def test_every_documented_strategy_is_usable(self):
        for strategy in OPPONENT_STRATEGIES:
            coordinator = self._coordinator(strategy=strategy)
            self.assertIn(coordinator.choose_opponent_checkpoint(), self.POOL)


if __name__ == "__main__":
    unittest.main()
