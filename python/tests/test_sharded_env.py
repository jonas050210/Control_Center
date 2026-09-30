"""Multi-process (sharded) environment execution.

The contract under test is strong on purpose: sharding must be a pure
*wall-time* change. A batch produced by W workers has to be identical to the
batch the same environments produce inside one process, index for index.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from optional_deps import HAS_SB3, SB3_REASON
from simulated_bridge import SimulatedBridgeExecutable, supported
from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.godot_env import GodotBatchClient, make_batch_client
from sandboxai.sharded_env import (
    ShardFailure,
    ShardedBatchClient,
    plan_shards,
    recommended_worker_count,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ShardPlanningTests(unittest.TestCase):
    def test_even_split(self):
        shards = plan_shards(16, 4)
        self.assertEqual([(s.offset, s.count) for s in shards], [(0, 4), (4, 4), (8, 4), (12, 4)])

    def test_uneven_split_is_contiguous_and_complete(self):
        for environment_count in range(1, 33):
            for workers in range(1, 13):
                shards = plan_shards(environment_count, workers)
                self.assertEqual(sum(s.count for s in shards), environment_count)
                self.assertEqual(shards[0].offset, 0)
                for previous, current in zip(shards, shards[1:]):
                    self.assertEqual(previous.stop, current.offset)
                self.assertTrue(all(s.count >= 1 for s in shards))
                # Load imbalance never exceeds one environment.
                counts = [s.count for s in shards]
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_more_workers_than_environments_is_clamped(self):
        self.assertEqual(len(plan_shards(3, 12)), 3)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            plan_shards(0, 1)
        with self.assertRaises(ValueError):
            plan_shards(4, 0)

    def test_recommended_worker_count_leaves_cores_for_the_trainer(self):
        # 20 logical threads (i7-12700F) -> 10 physical estimate -> 8 after
        # reserving two cores for the trainer/OS.
        self.assertEqual(recommended_worker_count(16, cpu_count=20), 8)
        # Never more workers than environments.
        self.assertEqual(recommended_worker_count(4, cpu_count=20), 4)
        # Degenerate hosts still get a usable pool.
        self.assertEqual(recommended_worker_count(8, cpu_count=1), 1)
        self.assertEqual(recommended_worker_count(8, cpu_count=4), 2)


@unittest.skipUnless(supported(), "simulated bridge requires POSIX shebang support")
class ShardedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = SimulatedBridgeExecutable()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.cleanup()

    def make_client(self, environment_count: int, worker_count: int, **kwargs):
        client = make_batch_client(
            project_path=PROJECT_ROOT,
            godot_executable=self.bridge.path,
            environment_count=environment_count,
            enemy_count=1,
            seed=kwargs.pop("seed", 4242),
            curriculum_level=3,
            worker_count=worker_count,
            **kwargs,
        )
        self.addCleanup(client.close)
        return client

    def test_single_worker_returns_the_plain_client(self):
        client = self.make_client(4, 1)
        self.assertIsInstance(client, GodotBatchClient)

    def test_sharding_is_result_preserving(self):
        """8 environments in 1 process == 8 environments in 4 processes."""
        actions = [[1, 1, 2, 0, 1, 0] for _ in range(8)]
        baseline = self.make_client(8, 1)
        sharded = self.make_client(8, 4)
        self.assertIsInstance(sharded, ShardedBatchClient)
        self.assertEqual(sharded.worker_count, 4)

        base_obs, _ = baseline.reset(100)
        shard_obs, _ = sharded.reset(100)
        self.assertEqual(base_obs.shape, (8, OBSERVATION_FIELD_COUNT))
        self.assertEqual(shard_obs.shape, (8, OBSERVATION_FIELD_COUNT))
        self.assertTrue((base_obs == shard_obs).all())

        # Long enough to cross several auto-reset boundaries, which is
        # where an off-by-one in seed mapping would show up.
        for step in range(40):
            b_obs, b_rew, b_done, b_info = baseline.step(actions)
            s_obs, s_rew, s_done, s_info = sharded.step(actions)
            self.assertTrue((b_obs == s_obs).all(), f"observations diverged at step {step}")
            self.assertTrue((b_rew == s_rew).all(), f"rewards diverged at step {step}")
            self.assertTrue((b_done == s_done).all(), f"dones diverged at step {step}")
            self.assertEqual(
                [info["metrics"]["seed"] for info in b_info],
                [info["metrics"]["seed"] for info in s_info],
                f"per-environment seeds diverged at step {step}",
            )

    def test_uneven_shards_preserve_global_order(self):
        actions = [[1, 1, 1, 1, 0, 0] for _ in range(7)]
        baseline = self.make_client(7, 1)
        sharded = self.make_client(7, 3)  # 3/2/2
        baseline.reset(9)
        sharded.reset(9)
        for _ in range(5):
            b_obs, _, _, b_info = baseline.step(actions)
            s_obs, _, _, s_info = sharded.step(actions)
            self.assertTrue((b_obs == s_obs).all())
            self.assertEqual(len(s_info), 7)

    def test_episode_plans_are_routed_to_the_owning_worker(self):
        sharded = self.make_client(6, 3)
        plans = [
            {
                "index": index,
                "seed": 5000 + index,
                "map_id": "arena",
                "scenario": "duel",
                "lighting": "day",
                "enemy_count": 1,
                "curriculum_level": 3,
            }
            for index in (0, 3, 5)
        ]
        response = sharded.set_episode_plans(plans)
        self.assertEqual(response["staged"], [0, 3, 5])
        # The plan is consumed on reset; the resulting per-environment
        # seeds prove the routing used global indices.
        _obs, infos = sharded.reset(None)
        seeds = [info["seed"] for info in infos]
        self.assertEqual(seeds[0], 5000)
        self.assertEqual(seeds[3], 5003)
        self.assertEqual(seeds[5], 5005)

    def test_plan_index_outside_the_global_range_is_rejected(self):
        sharded = self.make_client(4, 2)
        with self.assertRaises(ValueError):
            sharded.set_episode_plans([{"index": 9, "seed": 1}])

    def test_gathered_commands_cover_every_environment_in_order(self):
        sharded = self.make_client(6, 3)
        sharded.reset(11)
        metrics = sharded.metrics()
        self.assertEqual(len(metrics), 6)
        self.assertEqual([row["seed"] for row in metrics], [11 + i for i in range(6)])
        self.assertEqual(len(sharded.health_check()), 6)
        self.assertEqual(len(sharded.reward_breakdown()), 6)
        self.assertEqual(len(sharded.episode_conditions()), 6)

    def test_action_count_mismatch_raises(self):
        sharded = self.make_client(4, 2)
        sharded.reset(1)
        with self.assertRaises(ValueError):
            sharded.step([[1, 1, 1, 1, 0, 0]])

    def test_worker_crash_names_the_worker_and_its_range(self):
        os.environ["SANDBOXAI_FAKE_CRASH_AFTER_STEPS"] = "2"
        # Only the second shard (base seed 4242 + 2) crashes.
        os.environ["SANDBOXAI_FAKE_CRASH_SEED"] = "4244"
        try:
            sharded = self.make_client(4, 2, request_timeout=5.0)
            sharded.reset(None)
            actions = [[1, 1, 1, 1, 0, 0] for _ in range(4)]
            with self.assertRaises(ShardFailure) as caught:
                for _ in range(10):
                    sharded.step(actions)
            self.assertEqual(caught.exception.worker, 1)
            self.assertEqual(caught.exception.offset, 2)
            self.assertIn("environments 2..3", str(caught.exception))
            self.assertEqual(
                sharded.clients,
                [],
                "one failed shard must close the whole facade; a partial batch may never continue",
            )
        finally:
            del os.environ["SANDBOXAI_FAKE_CRASH_AFTER_STEPS"]
            del os.environ["SANDBOXAI_FAKE_CRASH_SEED"]

    def test_close_is_idempotent(self):
        sharded = self.make_client(4, 2)
        sharded.close()
        sharded.close()

    @unittest.skipUnless(HAS_SB3, SB3_REASON)
    def test_vec_env_uses_workers_and_keeps_the_batch_shape(self):
        from sandboxai.godot_env import GodotVecEnv

        env = GodotVecEnv(
            project_path=PROJECT_ROOT,
            godot_executable=self.bridge.path,
            environment_count=8,
            enemy_count=1,
            seed=77,
            curriculum_level=3,
            worker_count=4,
        )
        try:
            self.assertEqual(env.num_envs, 8)
            self.assertEqual(env.worker_count, 4)
            observations = env.reset()
            self.assertEqual(observations.shape, (8, OBSERVATION_FIELD_COUNT))
            env.step_async([[1, 1, 1, 1, 0, 0] for _ in range(8)])
            observations, rewards, dones, infos = env.step_wait()
            self.assertEqual(observations.shape, (8, OBSERVATION_FIELD_COUNT))
            self.assertEqual(rewards.shape, (8,))
            self.assertEqual(len(infos), 8)
        finally:
            env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
