"""Stable-Baselines3 PPO orchestration for the Godot environment."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .config import TrainingConfig
from .evaluation import evaluate_model
from .godot_env import GodotVecEnv
from .telemetry import JsonlTelemetry, resource_snapshot


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PPO requires stable-baselines3, gymnasium, torch and numpy. "
            "Install with: python -m pip install -e '.[training]'"
        ) from exc
    return PPO, BaseCallback, CallbackList, CheckpointCallback


def _env_kwargs(config: TrainingConfig) -> dict[str, Any]:
    return {
        "project_path": config.project,
        "godot_executable": config.godot_executable,
        "environment_count": config.environment_count,
        "enemy_count": config.enemy_count,
        "seed": config.seed,
        "curriculum_level": config.curriculum_level,
    }


def train_ppo(config: TrainingConfig, resume_checkpoint: str | Path | None = None) -> dict[str, Any]:
    config.validate()
    PPO, BaseCallback, CallbackList, CheckpointCallback = _require_sb3()
    device = config.resolved_device()
    checkpoint_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    run_dir = checkpoint_path.parent.parent if checkpoint_path else config.run_directory()
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    evaluations = run_dir / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")

    env = GodotVecEnv(**_env_kwargs(config))
    telemetry = JsonlTelemetry(logs / "training.jsonl")
    best_score = float("-inf")
    best_path = checkpoints / "best_eval.zip"
    if (evaluations / "best.json").exists():
        best_score = float(json.loads((evaluations / "best.json").read_text(encoding="utf-8")).get("mean_reward", best_score))

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.perf_counter()
            self.last_telemetry_step = 0
            self.episode_count = 0

        def _on_training_start(self) -> None:
            telemetry.write(
                {
                    "event": "training_start",
                    "environment_count": config.environment_count,
                    "device": device,
                    "total_training_steps": config.total_training_steps,
                }
            )

        def _on_step(self) -> bool:
            nonlocal best_score
            infos = self.locals.get("infos", [])
            episode_metrics = []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                metrics = info.get("metrics", {})
                if info.get("done_reason") or info.get("terminal_observation") is not None:
                    self.episode_count += 1
                    episode_metrics.append(metrics)
            if self.num_timesteps - self.last_telemetry_step >= max(config.environment_count, 1) * 100:
                elapsed = max(time.perf_counter() - self.started, 1e-9)
                payload: dict[str, Any] = {
                    "event": "progress",
                    "timesteps": self.num_timesteps,
                    "progress": self.num_timesteps / config.total_training_steps,
                    "steps_per_second": self.num_timesteps / elapsed,
                    "episodes": self.episode_count,
                    "environment_count": config.environment_count,
                    "current_checkpoint": str(checkpoints),
                    **resource_snapshot(),
                }
                if episode_metrics:
                    payload["mean_episode_reward"] = sum(float(item.get("episode_reward", 0.0)) for item in episode_metrics) / len(episode_metrics)
                    payload["mean_kills"] = sum(float(item.get("kills", 0.0)) for item in episode_metrics) / len(episode_metrics)
                    payload["mean_accuracy"] = sum(float(item.get("accuracy", 0.0)) for item in episode_metrics) / len(episode_metrics)
                    payload["win_rate"] = sum(float(item.get("win", 0.0)) for item in episode_metrics) / len(episode_metrics)
                telemetry.write(payload)
                self.last_telemetry_step = self.num_timesteps
            return True

        def _on_training_end(self) -> None:
            telemetry.write({"event": "training_end", "timesteps": self.num_timesteps, "episodes": self.episode_count})

    class EvaluationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = config.evaluation_frequency

        def _on_step(self) -> bool:
            nonlocal best_score
            if self.num_timesteps < self.next_evaluation:
                return True
            eval_kwargs = _env_kwargs(config)
            eval_kwargs["environment_count"] = 1
            summary = evaluate_model(
                self.model,
                eval_kwargs,
                episodes=config.evaluation_episodes,
                seed=config.seed + self.num_timesteps,
                output_dir=evaluations / f"step_{self.num_timesteps:09d}",
            )
            reward = float(summary.get("mean_episode_reward", 0.0))
            summary["timesteps"] = self.num_timesteps
            (evaluations / "latest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            if reward > best_score:
                best_score = reward
                self.model.save(best_path)
                (evaluations / "best.json").write_text(
                    json.dumps({"mean_reward": reward, "timesteps": self.num_timesteps, "checkpoint": str(best_path)}, indent=2) + "\n",
                    encoding="utf-8",
                )
            self.next_evaluation += config.evaluation_frequency
            return True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config.checkpoint_frequency // config.environment_count),
        save_path=str(checkpoints),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([checkpoint_callback, MetricsCallback(), EvaluationCallback()])
    try:
        if checkpoint_path:
            model = PPO.load(str(checkpoint_path), env=env, device=device)
            model.set_env(env)
            warm_start = {"transferred": False, "source": "resume checkpoint"}
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config.learning_rate,
                n_steps=config.rollout_length,
                batch_size=config.batch_size,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                ent_coef=config.entropy_coefficient,
                clip_range=config.clip_range,
                seed=config.seed,
                device=device,
                policy_kwargs={"net_arch": {"pi": list(config.net_arch), "vf": list(config.net_arch)}},
                tensorboard_log=str(logs / "tensorboard"),
                verbose=1,
            )
            warm_start = {"transferred": False}
            if config.bc_checkpoint:
                from .bc import load_bc_into_sb3_policy
                warm_start = load_bc_into_sb3_policy(model.policy, config.bc_checkpoint, device=device)
                warm_start["source"] = config.bc_checkpoint
            (run_dir / "warm_start.json").write_text(json.dumps(warm_start, indent=2) + "\n", encoding="utf-8")
        model.learn(
            total_timesteps=config.total_training_steps,
            callback=callbacks,
            reset_num_timesteps=not bool(checkpoint_path),
            progress_bar=False,
        )
        latest = checkpoints / "latest.zip"
        model.save(latest)
        model.save(run_dir / "final.zip")
        result = {
            "run_dir": str(run_dir),
            "latest_checkpoint": str(latest),
            "final_checkpoint": str(run_dir / "final.zip"),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "device": device,
            "timesteps": config.total_training_steps,
            "warm_start": warm_start,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        telemetry.close()
        env.close()
