"""PPO Reinforcement Learning Training Pipeline in SandboxAI (Phase 5).

Trains a PPO agent inside the Tactical FPS Sandbox using Stable-Baselines3.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from sandbox.env import MockTacticalArenaEnv, make_sandbox_env


def train_ppo_sandbox(
    timesteps: int = 10000,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    env_path: Optional[str] = None,
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    """Trains a PPO policy on visual observations inside the tactical sandbox."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    save_file = checkpoint_path / "ppo_sandbox.zip"

    print("============================================================")
    print("  SandboxAI: PPO Reinforcement Learning Training")
    print("============================================================")
    print(f"Total Timesteps: {timesteps}")
    print(f"Parallel Envs  : {n_envs}")
    print(f"Learning Rate  : {learning_rate}")
    print(f"Checkpoint     : {save_file}")
    print("============================================================")

    def _make_env_fn(env_idx: int):
        def _thunk():
            return make_sandbox_env(env_path=env_path, width=84, height=84, seed=seed + env_idx * 100)
        return _thunk

    vec_env = DummyVecEnv([_make_env_fn(i) for i in range(n_envs)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device=device,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0

    model.save(str(save_file))
    print(f"\n[INFO] PPO Training complete in {elapsed:.1f}s ({timesteps / max(1.0, elapsed):.1f} steps/s)")
    print(f"[INFO] Saved PPO model to: {save_file}")

    return {
        "timesteps": timesteps,
        "elapsed_sec": round(elapsed, 2),
        "steps_per_sec": round(timesteps / max(1.0, elapsed), 1),
        "checkpoint_path": str(save_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO in SandboxAI Tactical Arena")
    parser.add_argument("--timesteps", "-t", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--env_path", type=str, default=None, help="Exported Godot executable path (optional)")
    args = parser.parse_args()

    train_ppo_sandbox(
        timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        env_path=args.env_path,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
