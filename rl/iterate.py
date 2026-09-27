"""Repeated PPO train/evaluate/checkpoint cycles for progressive improvement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evaluation.benchmark import run_benchmark
from monitoring.experiments import ExperimentTracker
from rl.train_ppo import train_ppo_sandbox


def run_training_cycles(
    bc_checkpoint: Union[str, Path],
    cycles: int = 3,
    timesteps_per_cycle: int = 10000,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    logs_dir: Union[str, Path] = "logs",
    evaluation_episodes: int = 5,
    evaluation_steps: int = 300,
    env_path: Optional[str] = None,
    backend: str = "python",
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    checkpoints, logs = Path(checkpoint_dir), Path(logs_dir)
    experiments = logs / "experiments"
    tracker = ExperimentTracker(
        "rl_cycles",
        {"cycles": cycles, "timesteps_per_cycle": timesteps_per_cycle},
        experiments,
    )
    history: List[Dict[str, Any]] = []
    resume: Optional[str] = None
    try:
        for cycle in range(1, cycles + 1):
            cycle_dir = checkpoints / f"cycle_{cycle:02d}"
            training = train_ppo_sandbox(
                timesteps=timesteps_per_cycle,
                checkpoint_dir=cycle_dir,
                env_path=env_path,
                n_envs=1 if backend == "godot" else 2,
                seed=seed,
                device=device,
                bc_checkpoint=bc_checkpoint if cycle == 1 else None,
                resume_path=resume,
                backend=backend,
                checkpoint_freq=max(1000, timesteps_per_cycle // 2),
                experiment_root=experiments,
                state_file=logs / "system_state.json",
            )
            resume = training["checkpoint_path"]
            report_path = logs / f"benchmark_cycle_{cycle:02d}.json"
            benchmark = run_benchmark(
                bc_checkpoint=str(bc_checkpoint),
                ppo_checkpoint=resume,
                num_episodes=evaluation_episodes,
                max_steps_per_episode=evaluation_steps,
                env_path=env_path,
                output_report=str(report_path),
                backend=backend,
                experiment_root=experiments,
                state_file=logs / "system_state.json",
            )
            record = {"cycle": cycle, "training": training, "benchmark": benchmark}
            history.append(record)
            tracker.log_metrics(
                cycle,
                {
                    "ppo_mean_reward": benchmark["ppo"]["mean_reward"],
                    "ppo_hits": benchmark["ppo"]["total_hits"],
                    "ppo_kills": benchmark["ppo"]["total_kills"],
                },
                phase="cycle_evaluation",
            )
        result = {"cycles_completed": cycles, "final_checkpoint": resume, "history": history}
        output = logs / "rl_learning_curve.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        tracker.add_artifact(output, "learning_curve")
        tracker.finish(result)
        result["run_id"] = tracker.run_id
        return result
    except Exception as exc:
        tracker.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated PPO/evaluation cycles")
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=10000, help="Timesteps per cycle")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--env_path", default=None)
    parser.add_argument("--backend", choices=["python", "godot"], default="python")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_training_cycles(
        args.bc_checkpoint,
        args.cycles,
        args.timesteps,
        args.checkpoint_dir,
        args.logs_dir,
        args.episodes,
        args.steps,
        args.env_path,
        args.backend,
        args.device,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
