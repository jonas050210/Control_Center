"""BC Policy Closed-Loop Runner inside the Tactical Sandbox (Phase 4).

Connects a trained Behavioral Cloning policy to the Sandbox environment,
translates visual observations to actions, and evaluates agent performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bc.policy import BCPolicy
from sandbox.env import make_sandbox_env


def run_bc_in_sandbox(
    checkpoint_path: str,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    device: str = "cpu",
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Runs closed-loop evaluation of a BC model inside the Tactical Sandbox."""
    print("============================================================")
    print("  SandboxAI: BC Policy -> Sandbox Closed-Loop Runner")
    print("============================================================")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Num Episodes : {num_episodes}")
    print(f"Max Steps/Ep : {max_steps_per_episode}")
    print(f"Device       : {device}")
    print("============================================================")

    policy = BCPolicy.load_from_checkpoint(checkpoint_path, device=device)
    env = make_sandbox_env(env_path=env_path, width=policy.target_w, height=policy.target_h)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_hits: List[int] = []

    fired_count = 0
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=ep * 100)
        policy.reset()

        ep_reward = 0.0
        ep_step = 0
        ep_hits = 0

        while ep_step < max_steps_per_episode:
            # Policy predicts ActionState from visual observation [3, H, W]
            action_state, raw_dict = policy.predict(obs, deterministic=deterministic)

            # Map to sandbox action format [move_x_idx, move_y_idx, fire, turn_yaw, turn_pitch]
            move_x_idx = int(raw_dict["move_x"] + 1)  # -1, 0, 1 -> 0, 1, 2
            move_y_idx = int(raw_dict["move_y"] + 1)
            fire = int(raw_dict["fire"])
            yaw_bin = int(raw_dict["mouse_dx_bin"])
            pitch_bin = int(raw_dict["mouse_dy_bin"])

            env_action = [move_x_idx, move_y_idx, fire, yaw_bin, pitch_bin]

            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += reward
            ep_step += 1
            total_steps += 1
            if fire == 1:
                fired_count += 1
            if info.get("last_shot_hit", False):
                ep_hits += 1

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_step)
        episode_hits.append(ep_hits)

        print(
            f"Episode {ep:02d}/{num_episodes:02d} | "
            f"Reward: {ep_reward:+.2f} | "
            f"Steps: {ep_step:3d} | "
            f"Targets Hit: {ep_hits:2d}"
        )

    mean_rew = float(np.mean(episode_rewards))
    std_rew = float(np.std(episode_rewards))
    mean_len = float(np.mean(episode_lengths))
    total_hits = sum(episode_hits)

    print("\n=== Closed-Loop Evaluation Summary ===")
    print(f"Mean Reward     : {mean_rew:+.2f} ± {std_rew:.2f}")
    print(f"Mean Length     : {mean_len:.1f} steps")
    print(f"Total Targets Hit: {total_hits}")
    print(f"Fire Rate       : {fired_count / max(1, total_steps) * 100.0:.1f}%")

    return {
        "mean_reward": round(mean_rew, 3),
        "std_reward": round(std_rew, 3),
        "mean_length": round(mean_len, 1),
        "total_hits": total_hits,
        "episode_rewards": episode_rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BC policy in Sandbox environment")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to BC checkpoint")
    parser.add_argument("--episodes", "-e", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--env_path", type=str, default=None, help="Path to exported Godot executable (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling")
    args = parser.parse_args()

    run_bc_in_sandbox(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        env_path=args.env_path,
        device=args.device,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
