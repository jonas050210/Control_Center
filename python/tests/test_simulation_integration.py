import json
import math
import tempfile
import unittest
from pathlib import Path

try:  # Optional training extra; the suite must skip, never fail, without it.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None
import numpy as np

from optional_deps import HAS_TORCH, TORCH_REASON
from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.dataset import (
    ACTION_NVECS,
    DemonstrationDataset,
    DemonstrationRecorder,
    action_to_multidiscrete,
    discrete_to_multidiscrete,
)
from sandboxai.bc import (
    BehaviorCloningPolicy,
    create_bc_policy,
    load_bc_checkpoint,
    train_behavior_cloning,
    load_bc_into_sb3_policy,
)
from sandboxai.config import BCConfig, TrainingConfig
from sandboxai.self_play import PolicySlot, SelfPlayCoordinator


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class SimulationIntegrationTests(unittest.TestCase):
    """End-to-end integration and simulation physics / RL contract tests."""

    def test_multidiscrete_action_encodings_cover_full_space(self):
        for discrete_idx in range(10):
            encoded = discrete_to_multidiscrete(discrete_idx)
            self.assertEqual(len(encoded), len(ACTION_NVEC))
            self.assertTrue(all(0 <= val < nvec for val, nvec in zip(encoded, ACTION_NVECS)))

    def test_canonical_action_to_multidiscrete(self):
        # 7-field canonical array [move, strafe, yaw, pitch, shoot, dx, dy]
        idle_canonical = [0, 0, 0, 0, 0, 0.0, 0.0]
        self.assertEqual(action_to_multidiscrete(idle_canonical), [1, 1, 1, 1, 0, 0])

        forward_shoot = [1, 0, 0, 0, 1, 0.0, 0.0]
        self.assertEqual(action_to_multidiscrete(forward_shoot), [2, 1, 1, 1, 1, 0])

        backward_left = [-1, -1, -1, -1, 0, 0.0, 0.0]
        self.assertEqual(action_to_multidiscrete(backward_left), [0, 0, 0, 0, 0, 0])

    def test_demonstration_dataset_roundtrip_and_arrays(self):
        recorder = DemonstrationRecorder({"env": "test", "curriculum": 3})
        recorder.start()
        
        # 3 episodes of 10 steps
        for ep in range(3):
            for step in range(10):
                obs = [float(ep * 10 + step) / 100.0] * OBSERVATION_FIELD_COUNT
                next_obs = [float(ep * 10 + step + 1) / 100.0] * OBSERVATION_FIELD_COUNT
                action = [1, 0, 0, 0, 1 if step % 2 == 0 else 0, 0.0, 0.0]
                reward = 1.0 if step == 9 else 0.01
                done = (step == 9)
                recorder.append(obs, action, next_obs, reward, done, episode_id=ep)
        recorder.stop()

        with tempfile.TemporaryDirectory() as tmp:
            demo_file = Path(tmp) / "demo.jsonl"
            recorder.save(demo_file)
            dataset = DemonstrationDataset.load(demo_file)

            self.assertEqual(len(dataset.transitions), 30)
            summary = dataset.summary()
            self.assertEqual(summary["transitions"], 30)
            self.assertEqual(summary["episodes"], 3)
            self.assertEqual(summary["terminal_transitions"], 3)

            obs_arr, act_arr, next_obs_arr, rew_arr, done_arr = dataset.arrays()
            self.assertEqual(obs_arr.shape, (30, OBSERVATION_FIELD_COUNT))
            self.assertEqual(act_arr.shape, (30, len(ACTION_NVEC)))
            self.assertEqual(next_obs_arr.shape, (30, OBSERVATION_FIELD_COUNT))
            self.assertEqual(rew_arr.shape, (30,))
            self.assertEqual(done_arr.shape, (30,))

    def test_bc_training_and_weight_transfer_lifecycle(self):
        recorder = DemonstrationRecorder({"test": "bc_lifecycle"})
        recorder.start()
        for i in range(50):
            obs = [math.sin(i * 0.1 + j) for j in range(OBSERVATION_FIELD_COUNT)]
            next_obs = [math.sin((i + 1) * 0.1 + j) for j in range(OBSERVATION_FIELD_COUNT)]
            action = [0, 1, 0, 0, 1 if i % 5 == 0 else 0, 0.0, 0.0]
            recorder.append(obs, action, next_obs, 0.5, i == 49, episode_id=0)
        recorder.stop()

        with tempfile.TemporaryDirectory() as tmp:
            demo_path = Path(tmp) / "demo.jsonl"
            recorder.save(demo_path)

            bc_cfg = BCConfig(
                epochs=5,
                batch_size=8,
                learning_rate=1e-3,
                validation_fraction=0.2,
                early_stopping_patience=3,
                output_root=tmp,
            )
            out_dir = Path(tmp) / "bc_run"
            result = train_behavior_cloning(demo_path, bc_cfg, output_dir=out_dir)

            self.assertTrue(Path(result["latest_checkpoint"]).is_file())
            self.assertTrue(Path(result["best_checkpoint"]).is_file())

            # Load checkpoint
            loaded_model = load_bc_checkpoint(result["best_checkpoint"])
            pred = loaded_model.predict(torch.randn(4, OBSERVATION_FIELD_COUNT))
            self.assertEqual(pred.shape, (4, len(ACTION_NVEC)))

            # Test transfer into SB3 PPO policy
            import gymnasium as gym
            from stable_baselines3 import PPO

            class SimpleEnv(gym.Env):
                def __init__(self):
                    self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBSERVATION_FIELD_COUNT,))
                    self.action_space = gym.spaces.MultiDiscrete(list(ACTION_NVEC))
            
            ppo_model = PPO(
                "MlpPolicy",
                SimpleEnv(),
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                n_steps=16,
                batch_size=16,
                device="cpu",
            )
            transfer_info = load_bc_into_sb3_policy(ppo_model.policy, result["best_checkpoint"], device="cpu")
            self.assertTrue(transfer_info["transferred"])

    def test_self_play_coordinator_and_pool_sampling(self):
        slot_a = PolicySlot(name="agent_main")
        slot_b = PolicySlot(name="agent_frozen", checkpoint="path/to/c1.zip")
        coord = SelfPlayCoordinator(slot_a, slot_b, opponent_pool=["path/to/c1.zip", "path/to/c2.zip", "path/to/c3.zip"])
        self.assertEqual(len(coord.opponent_pool), 3)

        # Add new checkpoint
        coord.add_to_pool("path/to/c4.zip")
        self.assertEqual(len(coord.opponent_pool), 4)

        # Record match results
        coord.record_match(
            {"win": True, "episode_reward": 15.0, "kills": 2, "accuracy": 0.75},
            {"win": False, "episode_reward": -10.0, "kills": 0, "accuracy": 0.1},
        )
        coord.record_match(
            {"win": True, "episode_reward": 12.0, "kills": 1, "accuracy": 0.6},
            {"win": False, "episode_reward": -8.0, "kills": 0, "accuracy": 0.2},
        )

        summary = coord.summary()
        self.assertEqual(summary["agent_main"]["matches"], 2)
        self.assertEqual(summary["agent_main"]["win_rate"], 1.0)
        self.assertAlmostEqual(summary["agent_main"]["mean_reward"], 13.5)
        self.assertEqual(summary["agent_frozen"]["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
