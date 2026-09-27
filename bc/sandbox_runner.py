"""Run a BC policy in closed loop inside the controlled FPS sandbox."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bc.policy import BCPolicy
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry
from sandbox.env import make_sandbox_env


def run_bc_in_sandbox(
    checkpoint_path: str,
    num_episodes: int = 5,
    max_steps_per_episode: int = 300,
    env_path: Optional[str] = None,
    device: str = "cpu",
    deterministic: bool = True,
    backend: str = "auto",
    output_report: Optional[Union[str, Path]] = None,
    experiment_root: Optional[Union[str, Path]] = None,
    state_file: Optional[Union[str, Path]] = None,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    policy = BCPolicy.load_from_checkpoint(checkpoint_path, device=device)
    env = make_sandbox_env(
        env_path=env_path,
        backend=backend,
        width=policy.target_w,
        height=policy.target_h,
        max_steps=max_steps_per_episode,
    )
    tracker = (
        ExperimentTracker(
            "bc_closed_loop",
            {
                "checkpoint": checkpoint_path,
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
        telemetry.update_stage("BC_CLOSED_LOOP")
    episodes: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for episode_idx in range(num_episodes):
            observation, _ = env.reset(seed=base_seed + episode_idx * 13)
            policy.reset()
            reward_total = 0.0
            fired = 0
            final_info: Dict[str, Any] = {}
            for step in range(max_steps_per_episode):
                env_action, raw = policy.predict_env_action(
                    observation, deterministic=deterministic
                )
                observation, reward, terminated, truncated, final_info = env.step(env_action)
                reward_total += reward
                fired += int(raw["fire"])
                if telemetry and (step + 1) % 20 == 0:
                    elapsed = time.perf_counter() - started
                    telemetry.update_runtime(
                        fps=0.0,
                        steps_per_sec=(sum(item["length"] for item in episodes) + step + 1) / max(1e-9, elapsed),
                        episode_reward=reward_total,
                        action={key: raw[key] for key in ("move_x", "move_y", "fire", "ads", "mouse_dx_bin", "mouse_dy_bin")},
                        info={key: final_info.get(key) for key in ("step", "targets_hit", "kills", "health", "ammo")},
                    )
                if terminated or truncated:
                    break
            record = {
                "episode": episode_idx,
                "seed": base_seed + episode_idx * 13,
                "reward": round(reward_total, 6),
                "length": step + 1,
                "hits": int(final_info.get("targets_hit", 0)),
                "kills": int(final_info.get("kills", 0)),
                "shots": int(final_info.get("shots_fired", fired)),
                "accuracy": float(final_info.get("accuracy", 0.0)),
                "damage_taken": float(final_info.get("damage_taken", 0.0)),
                "health": float(final_info.get("health", 0.0)),
            }
            episodes.append(record)
            if tracker:
                tracker.log_metrics(episode_idx, record, phase="closed_loop")

        elapsed = time.perf_counter() - started
        rewards = [episode["reward"] for episode in episodes]
        total_steps = sum(episode["length"] for episode in episodes)
        total_hits = sum(episode["hits"] for episode in episodes)
        total_shots = sum(episode["shots"] for episode in episodes)
        result = {
            "policy_name": "BC",
            "checkpoint": checkpoint_path,
            "backend": (
                "godot" if backend == "godot" or (backend == "auto" and env_path) else "python"
            ),
            "episodes_completed": len(episodes),
            "mean_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "mean_length": round(float(np.mean([e["length"] for e in episodes])), 2),
            "total_hits": total_hits,
            "total_kills": sum(episode["kills"] for episode in episodes),
            "hit_rate_pct": round(total_hits / max(1, total_shots) * 100.0, 3),
            "fire_rate_pct": round(total_shots / max(1, total_steps) * 100.0, 3),
            "steps_per_sec": round(total_steps / max(1e-9, elapsed), 2),
            "episode_rewards": rewards,
            "episodes": episodes,
        }
        if output_report:
            report_path = Path(output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if tracker:
            if output_report:
                tracker.add_artifact(output_report, "closed_loop_report")
            tracker.finish(result)
            result["run_id"] = tracker.run_id
        if telemetry:
            telemetry.update_stage("IDLE")
        return result
    except Exception as exc:
        if tracker:
            tracker.fail(exc)
        if telemetry:
            telemetry.update_stage("FAILED")
        raise
    finally:
        env.close()


run_bc_sandbox = run_bc_in_sandbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BC in the controlled sandbox")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--episodes", "-e", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--env_path", default=None)
    parser.add_argument("--backend", choices=["auto", "python", "godot"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="logs/bc_closed_loop.json")
    parser.add_argument("--experiment_root", default="logs/experiments")
    parser.add_argument("--state_file", default="logs/system_state.json")
    args = parser.parse_args()
    result = run_bc_in_sandbox(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        env_path=args.env_path,
        device=args.device,
        deterministic=not args.stochastic,
        backend=args.backend,
        output_report=args.output,
        experiment_root=args.experiment_root,
        state_file=args.state_file,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
