"""Evaluation and Policy Benchmark Harness for SandboxAI (Phase 6).

Compares Random Policy vs Behavioral Cloning (BC) Policy vs PPO Policy
across standardized evaluation episodes and measurable tactical metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from bc.policy import BCPolicy
from sandbox.env import make_sandbox_env


def evaluate_policy_on_env(
    policy_type: str,  # "random", "bc", "ppo"
    policy_obj: Any,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    """Runs N evaluation episodes for a given policy in the Tactical Sandbox."""
    env = make_sandbox_env(env_path=env_path, width=84, height=84)

    rewards: List[float] = []
    lengths: List[int] = []
    hits: List[int] = []
    fired_steps = 0
    total_steps = 0

    for ep in range(num_episodes):
        ep_seed = base_seed + ep * 13
        obs, _ = env.reset(seed=ep_seed)
        if hasattr(policy_obj, "reset"):
            policy_obj.reset()

        ep_rew = 0.0
        ep_len = 0
        ep_hits = 0

        while ep_len < max_steps_per_episode:
            # Action selection
            if policy_type == "random":
                act = env.action_space.sample()
                fire_flag = int(act[2])
            elif policy_type == "bc":
                _, raw_dict = policy_obj.predict(obs, deterministic=True)
                act = [
                    int(raw_dict["move_x"] + 1),
                    int(raw_dict["move_y"] + 1),
                    int(raw_dict["fire"]),
                    int(raw_dict["mouse_dx_bin"]),
                    int(raw_dict["mouse_dy_bin"]),
                ]
                fire_flag = int(raw_dict["fire"])
            elif policy_type == "ppo":
                # PPO model predict
                act_pred, _ = policy_obj.predict(obs, deterministic=True)
                act = act_pred
                if isinstance(act, np.ndarray) and len(act.flatten()) > 2:
                    fire_flag = int(act.flatten()[2])
                else:
                    fire_flag = 0
            else:
                act = env.action_space.sample()
                fire_flag = 0

            obs, r, terminated, truncated, info = env.step(act)
            ep_rew += r
            ep_len += 1
            total_steps += 1
            if fire_flag == 1:
                fired_steps += 1
            if info.get("last_shot_hit", False):
                ep_hits += 1

            if terminated or truncated:
                break

        rewards.append(ep_rew)
        lengths.append(ep_len)
        hits.append(ep_hits)

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    std_r = float(np.std(rewards)) if rewards else 0.0
    mean_l = float(np.mean(lengths)) if lengths else 0.0
    total_h = sum(hits)
    hit_rate = (total_h / max(1, total_steps)) * 100.0
    fire_rate = (fired_steps / max(1, total_steps)) * 100.0

    return {
        "policy_name": policy_type.upper(),
        "mean_reward": round(mean_r, 2),
        "std_reward": round(std_r, 2),
        "min_reward": round(float(np.min(rewards)), 2) if rewards else 0.0,
        "max_reward": round(float(np.max(rewards)), 2) if rewards else 0.0,
        "mean_length": round(mean_l, 1),
        "total_hits": total_h,
        "hit_rate_pct": round(hit_rate, 2),
        "fire_rate_pct": round(fire_rate, 2),
        "num_episodes": num_episodes,
    }


def run_benchmark(
    bc_checkpoint: Optional[str] = None,
    ppo_checkpoint: Optional[str] = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    output_report: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a full benchmark comparing Random vs BC vs PPO policies."""
    print("============================================================")
    print("  SandboxAI: Policy Evaluation & Benchmark Harness")
    print("============================================================")
    print(f"Episodes per Policy: {num_episodes}")
    print(f"Max Steps / Episode: {max_steps_per_episode}")
    print("============================================================")

    results: Dict[str, Any] = {}

    # 1. Random Policy Baseline
    print("[1/3] Evaluating Random Policy Baseline...")
    res_random = evaluate_policy_on_env(
        policy_type="random",
        policy_obj=None,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        env_path=env_path,
    )
    results["random"] = res_random

    # 2. BC Policy
    if bc_checkpoint and Path(bc_checkpoint).exists():
        print(f"[2/3] Evaluating Behavioral Cloning (BC) Policy ({bc_checkpoint})...")
        bc_policy = BCPolicy.load_from_checkpoint(bc_checkpoint, device="cpu")
        res_bc = evaluate_policy_on_env(
            policy_type="bc",
            policy_obj=bc_policy,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            env_path=env_path,
        )
        results["bc"] = res_bc
    else:
        print("[2/3] Skipping BC Policy (no checkpoint provided)")

    # 3. PPO Policy
    if ppo_checkpoint and Path(ppo_checkpoint).exists():
        print(f"[3/3] Evaluating Reinforcement Learning (PPO) Policy ({ppo_checkpoint})...")
        from stable_baselines3 import PPO
        ppo_model = PPO.load(ppo_checkpoint, device="cpu")
        res_ppo = evaluate_policy_on_env(
            policy_type="ppo",
            policy_obj=ppo_model,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            env_path=env_path,
        )
        results["ppo"] = res_ppo
    else:
        print("[3/3] Skipping PPO Policy (no checkpoint provided)")

    # Print Formatted Comparison Table
    print("\n" + "=" * 76)
    print(f"{'Policy':<12} | {'Reward (Mean ± Std)':<22} | {'Hits':<6} | {'Hit Rate':<10} | {'Fire Rate':<10}")
    print("-" * 76)
    for p_key, p_res in results.items():
        rew_str = f"{p_res['mean_reward']:+.2f} ± {p_res['std_reward']:.2f}"
        print(
            f"{p_res['policy_name']:<12} | {rew_str:<22} | {p_res['total_hits']:<6} | "
            f"{p_res['hit_rate_pct']:>8.1f}% | {p_res['fire_rate_pct']:>8.1f}%"
        )
    print("=" * 76 + "\n")

    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Benchmark report saved to: {out_report_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Benchmark comparing Random vs BC vs PPO")
    parser.add_argument("--bc_checkpoint", "-b", type=str, default=None, help="Path to BC checkpoint")
    parser.add_argument("--ppo_checkpoint", "-p", type=str, default=None, help="Path to PPO checkpoint")
    parser.add_argument("--episodes", "-e", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--steps", "-s", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--env_path", type=str, default=None, help="Path to exported Godot executable")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save JSON report path")
    args = parser.parse_args()

    run_benchmark(
        bc_checkpoint=args.bc_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        env_path=args.env_path,
        output_report=args.output,
    )


if __name__ == "__main__":
    main()
