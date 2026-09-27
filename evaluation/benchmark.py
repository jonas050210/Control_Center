"""Deterministic comparison of random, scripted, BC, and PPO FPS policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bc.policy import BCPolicy
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry
from sandbox.actions import SandboxAction
from sandbox.env import make_sandbox_env


class ScriptedVisionPolicy:
    """Simple non-learning baseline that tracks red target pixels."""

    def reset(self) -> None:
        return None

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        frame = np.transpose(observation, (1, 2, 0)) if observation.shape[0] == 3 else observation
        mask = (frame[..., 0] > 150) & (frame[..., 0] > frame[..., 1] * 1.45)
        height, width = mask.shape
        if mask.any():
            ys, xs = np.nonzero(mask)
            offset_x = (float(xs.mean()) - width / 2) / max(1.0, width / 2)
            offset_y = (float(ys.mean()) - height / 2) / max(1.0, height / 2)
            yaw_bin = int(np.clip(round(10 + offset_x * 7), 0, 20))
            pitch_bin = int(np.clip(round(10 + offset_y * 5), 0, 20))
            centred = abs(offset_x) < 0.075 and abs(offset_y) < 0.11
            action = SandboxAction(
                move_x=-1 if offset_x < -0.35 else (1 if offset_x > 0.35 else 0),
                move_y=0 if centred else 1,
                fire=int(centred),
                ads=1,
                mouse_dx_bin=yaw_bin,
                mouse_dy_bin=pitch_bin,
            )
        else:
            action = SandboxAction(move_y=1, mouse_dx_bin=13)
        return action.to_array(), None


def evaluate_policy_on_env(
    policy_type: str,
    policy_obj: Any,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    base_seed: int = 1000,
    backend: str = "auto",
) -> Dict[str, Any]:
    if num_episodes < 1 or max_steps_per_episode < 1:
        raise ValueError("num_episodes and max_steps_per_episode must be positive")
    env = make_sandbox_env(
        env_path=env_path,
        backend=backend,
        width=84,
        height=84,
        max_steps=max_steps_per_episode,
    )
    rng = np.random.default_rng(base_seed + 99991)
    episodes: List[Dict[str, Any]] = []
    try:
        for episode_idx in range(num_episodes):
            seed = base_seed + episode_idx * 13
            observation, _ = env.reset(seed=seed)
            if hasattr(policy_obj, "reset"):
                policy_obj.reset()
            total_reward = 0.0
            fired_fallback = 0
            final_info: Dict[str, Any] = {}
            for step in range(max_steps_per_episode):
                if policy_type == "random":
                    # Seeded explicitly rather than relying on each backend's
                    # action-space sampler implementation.
                    action = np.asarray(
                        [rng.integers(0, n) for n in env.action_space.nvec], dtype=np.int64
                    )
                elif policy_type == "bc":
                    action, _ = policy_obj.predict_env_action(observation, deterministic=True)
                elif policy_type in {"ppo", "scripted"}:
                    action, _ = policy_obj.predict(observation, deterministic=True)
                else:
                    raise ValueError(f"Unknown policy_type: {policy_type}")
                action_array = np.asarray(action).reshape(-1)
                fire_idx = 6 if action_array.size >= 10 else 2
                fired_fallback += int(action_array[fire_idx]) if action_array.size > fire_idx else 0
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "episode": episode_idx,
                    "seed": seed,
                    "reward": round(total_reward, 6),
                    "length": step + 1,
                    "hits": int(final_info.get("targets_hit", 0)),
                    "kills": int(final_info.get("kills", 0)),
                    "shots": int(final_info.get("shots_fired", fired_fallback)),
                    "accuracy": float(final_info.get("accuracy", 0.0)),
                    "damage_taken": float(final_info.get("damage_taken", 0.0)),
                    "health": float(final_info.get("health", 0.0)),
                    "distance": float(final_info.get("distance_travelled", 0.0)),
                }
            )
    finally:
        env.close()

    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=np.float64)
    total_steps = sum(episode["length"] for episode in episodes)
    total_hits = sum(episode["hits"] for episode in episodes)
    total_shots = sum(episode["shots"] for episode in episodes)
    return {
        "policy_name": policy_type.upper(),
        "mean_reward": round(float(rewards.mean()), 4),
        "std_reward": round(float(rewards.std()), 4),
        "min_reward": round(float(rewards.min()), 4),
        "max_reward": round(float(rewards.max()), 4),
        "mean_length": round(float(np.mean([e["length"] for e in episodes])), 2),
        "total_hits": total_hits,
        "total_kills": sum(episode["kills"] for episode in episodes),
        "hit_rate_pct": round(total_hits / max(1, total_shots) * 100.0, 3),
        "fire_rate_pct": round(total_shots / max(1, total_steps) * 100.0, 3),
        "mean_damage_taken": round(float(np.mean([e["damage_taken"] for e in episodes])), 3),
        "mean_final_health": round(float(np.mean([e["health"] for e in episodes])), 3),
        "num_episodes": num_episodes,
        "episodes": episodes,
    }


def run_benchmark(
    bc_checkpoint: Optional[str] = None,
    ppo_checkpoint: Optional[str] = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    output_report: Optional[str] = None,
    backend: str = "auto",
    device: str = "cpu",
    experiment_root: Optional[Union[str, Path]] = None,
    state_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    tracker = (
        ExperimentTracker(
            "benchmark",
            {
                "bc_checkpoint": bc_checkpoint,
                "ppo_checkpoint": ppo_checkpoint,
                "episodes": num_episodes,
                "steps": max_steps_per_episode,
                "backend": backend,
            },
            experiment_root,
        )
        if experiment_root
        else None
    )
    telemetry = SystemTelemetry(state_file) if state_file else None
    if telemetry:
        telemetry.update_stage("EVALUATING")
    results: Dict[str, Any] = {}
    try:
        policies: List[tuple[str, Any]] = [
            ("random", None),
            ("scripted", ScriptedVisionPolicy()),
        ]
        if bc_checkpoint:
            if not Path(bc_checkpoint).is_file():
                raise FileNotFoundError(f"BC checkpoint not found: {bc_checkpoint}")
            policies.append(("bc", BCPolicy.load_from_checkpoint(bc_checkpoint, device=device)))
        if ppo_checkpoint:
            if not Path(ppo_checkpoint).is_file():
                raise FileNotFoundError(f"PPO checkpoint not found: {ppo_checkpoint}")
            from stable_baselines3 import PPO

            policies.append(("ppo", PPO.load(ppo_checkpoint, device=device)))

        for name, policy in policies:
            result = evaluate_policy_on_env(
                name,
                policy,
                num_episodes,
                max_steps_per_episode,
                env_path,
                base_seed=1000,
                backend=backend,
            )
            results[name] = result
            if tracker:
                tracker.log_metrics(0, result, phase=f"evaluate_{name}")

        if output_report:
            path = Path(output_report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if tracker:
            if output_report:
                tracker.add_artifact(output_report, "benchmark_report")
            tracker.finish(results)
            results["run_id"] = tracker.run_id
        if telemetry:
            telemetry.update_evaluation(results)
            telemetry.update_stage("IDLE")

        print("\n" + "=" * 84)
        print(f"{'Policy':<12} {'Reward mean±std':>20} {'Hits/Shots':>13} {'Kills':>7} {'Health':>9}")
        print("-" * 84)
        for name, result in results.items():
            if not isinstance(result, dict) or "policy_name" not in result:
                continue
            shots = sum(ep["shots"] for ep in result["episodes"])
            print(
                f"{result['policy_name']:<12} "
                f"{result['mean_reward']:>9.3f}±{result['std_reward']:<8.3f} "
                f"{result['total_hits']:>5}/{shots:<7} "
                f"{result['total_kills']:>7} {result['mean_final_health']:>9.1f}"
            )
        print("=" * 84)
        return results
    except Exception as exc:
        if tracker:
            tracker.fail(exc)
        if telemetry:
            telemetry.update_stage("FAILED")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SandboxAI policies")
    parser.add_argument("--bc_checkpoint", "-b", default=None)
    parser.add_argument("--ppo_checkpoint", "-p", default=None)
    parser.add_argument("--episodes", "-e", type=int, default=5)
    parser.add_argument("--steps", "-s", type=int, default=300)
    parser.add_argument("--env_path", default=None)
    parser.add_argument("--backend", choices=["auto", "python", "godot"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", "-o", "--report", "-r", default="logs/benchmark.json")
    parser.add_argument("--experiment_root", default="logs/experiments")
    parser.add_argument("--state_file", default="logs/system_state.json")
    args = parser.parse_args()
    run_benchmark(
        bc_checkpoint=args.bc_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        env_path=args.env_path,
        output_report=args.output,
        backend=args.backend,
        device=args.device,
        experiment_root=args.experiment_root,
        state_file=args.state_file,
    )


if __name__ == "__main__":
    main()
