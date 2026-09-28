"""Stable-Baselines3 PPO orchestration for the Godot environment."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from .config import TrainingConfig
from .evaluation import evaluate_model
from .godot_env import GodotVecEnv
from .telemetry import JsonlTelemetry, resource_snapshot

if TYPE_CHECKING:
    from .run_control import RunControl


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


def train_ppo(
    config: TrainingConfig,
    resume_checkpoint: str | Path | None = None,
    run_control: "RunControl | None" = None,
) -> dict[str, Any]:
    config.validate()
    PPO, BaseCallback, CallbackList, CheckpointCallback = _require_sb3()
    device = config.resolved_device()
    checkpoint_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    if config.torch_threads > 0:
        import torch  # type: ignore

        torch.set_num_threads(config.torch_threads)
    # Resume artifacts stay inside the original run directory. A checkpoint
    # inside <run>/checkpoints/ maps to <run>; a checkpoint stored directly
    # in the run root (e.g. final.zip) maps to the run root itself, never to
    # the run's parent (which would scatter checkpoints/logs elsewhere).
    if checkpoint_path is not None:
        if checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parent.parent
        else:
            run_dir = checkpoint_path.parent
    else:
        run_dir = config.run_directory()
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    evaluations = run_dir / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")

    env = GodotVecEnv(**_env_kwargs(config))
    telemetry = JsonlTelemetry(logs / "training.jsonl")
    # Integrated research pipeline (curriculum plans at episode boundaries,
    # skill metrics, replays, per-condition tracking, checkpoint battery).
    # "fixed" mode keeps the historical single-level behavior exactly: no
    # hooks are installed and PPO drives the env as it always has.
    pipeline = None
    if config.curriculum_mode == "auto":
        from .pipeline import TrainingPipeline, write_manifest

        pipeline = TrainingPipeline(config, run_dir, telemetry, device=device)
        # Attach BEFORE any rollout starts: hook changes mid-training would
        # be a race (rollout collection runs in this thread, so sequencing
        # here is strict happens-before every step).
        pipeline.attach(env)
        if checkpoint_path is not None:
            # Resume continues the curriculum exactly where the saved
            # checkpoint left it: same stage, same staged plans, same
            # per-environment seed ordinals.
            if pipeline.load_state(checkpoints / "curriculum_state.json"):
                pipeline.reattach_after_load()
                telemetry.write(
                    {
                        "event": "curriculum_resumed",
                        "level": pipeline.driver.level,
                        "episodes_completed": pipeline.driver.episodes_completed,
                    }
                )
        write_manifest(run_dir, pipeline.manifest())
    best_score = float("-inf")
    best_path = checkpoints / "best_eval.zip"
    eval_patience_counter = 0
    stop_training = False

    if (evaluations / "best.json").exists():
        best_score = float(json.loads((evaluations / "best.json").read_text(encoding="utf-8")).get("mean_reward", best_score))

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.perf_counter()
            self.last_telemetry_step = 0
            self.episode_count = 0
            self.episode_metrics_buffer: list[dict[str, Any]] = []
            self.start_timesteps = 0
            self.target_timesteps = config.total_training_steps

        def _on_training_start(self) -> None:
            self.start_timesteps = self.num_timesteps
            self.target_timesteps = self.start_timesteps + config.total_training_steps
            self.last_telemetry_step = self.start_timesteps
            start_values = {
                "event": "training_start",
                "environment_count": config.environment_count,
                "device": device,
                "run_start_timesteps": self.start_timesteps,
                "total_training_steps": self.target_timesteps,
                "net_arch": list(config.net_arch),
                "learning_rate": config.learning_rate,
            }
            telemetry.write(start_values)
            if run_control is not None:
                run_control.running(**start_values)
                run_control.event("system", "PPO optimization started", start_values)

        def _on_step(self) -> bool:
            nonlocal stop_training
            if run_control is not None and not run_control.checkpoint():
                stop_training = True
                return False
            if pipeline is not None:
                # One attribute write per vector step: the pipeline stamps
                # episode rows / curriculum transitions with real PPO time.
                pipeline.timesteps = self.num_timesteps
            infos = self.locals.get("infos", [])
            for info in infos:
                if not isinstance(info, dict):
                    continue
                metrics = info.get("metrics", {})
                if info.get("done_reason") or info.get("terminal_observation") is not None:
                    self.episode_count += 1
                    self.episode_metrics_buffer.append(metrics)
            if self.num_timesteps - self.last_telemetry_step >= max(config.environment_count, 1) * 100:
                elapsed = max(time.perf_counter() - self.started, 1e-9)
                completed_steps = max(0, self.num_timesteps - self.start_timesteps)
                steps_per_second = completed_steps / elapsed
                remaining_steps = max(0, self.target_timesteps - self.num_timesteps)
                payload: dict[str, Any] = {
                    "event": "progress",
                    "timesteps": self.num_timesteps,
                    "run_start_timesteps": self.start_timesteps,
                    "total_training_steps": self.target_timesteps,
                    "progress": min(1.0, completed_steps / config.total_training_steps),
                    "steps_per_second": steps_per_second,
                    # Based on the measured run-average throughput. It is
                    # omitted until a positive rate exists rather than faked.
                    "eta_seconds": remaining_steps / steps_per_second if steps_per_second > 0 else None,
                    "episodes": self.episode_count,
                    "environment_count": config.environment_count,
                    "device": device,
                    "current_checkpoint": str(checkpoints),
                    **resource_snapshot(),
                }
                if self.episode_metrics_buffer:
                    buf = self.episode_metrics_buffer
                    payload["mean_episode_reward"] = sum(float(item.get("episode_reward", 0.0)) for item in buf) / len(buf)
                    payload["mean_kills"] = sum(float(item.get("kills", 0.0)) for item in buf) / len(buf)
                    payload["mean_deaths"] = sum(float(item.get("deaths", 0.0)) for item in buf) / len(buf)
                    payload["mean_damage_dealt"] = sum(float(item.get("damage_dealt", 0.0)) for item in buf) / len(buf)
                    payload["mean_damage_received"] = sum(float(item.get("damage_received", 0.0)) for item in buf) / len(buf)
                    payload["mean_shots_fired"] = sum(float(item.get("shots_fired", 0.0)) for item in buf) / len(buf)
                    payload["mean_shots_hit"] = sum(float(item.get("shots_hit", 0.0)) for item in buf) / len(buf)
                    payload["mean_survival_time"] = sum(float(item.get("survival_time", 0.0)) for item in buf) / len(buf)
                    payload["mean_accuracy"] = sum(float(item.get("accuracy", 0.0)) for item in buf) / len(buf)
                    payload["win_rate"] = sum(float(item.get("win", 0.0)) for item in buf) / len(buf)
                    payload["loss_rate"] = sum(float(item.get("loss", 0.0)) for item in buf) / len(buf)
                    if config.reward_breakdown_logging:
                        payload["reward_breakdown"] = {
                            "hits": sum(float(item.get("reward_breakdown", {}).get("reward_hits", 0.0)) for item in buf) / len(buf),
                            "kills": sum(float(item.get("reward_breakdown", {}).get("reward_kills", 0.0)) for item in buf) / len(buf),
                            "survive": sum(float(item.get("reward_breakdown", {}).get("reward_survive", 0.0)) for item in buf) / len(buf),
                            "damage_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_damage", 0.0)) for item in buf) / len(buf),
                            "death_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_death", 0.0)) for item in buf) / len(buf),
                            # Shot economy: how much reward was lost to real
                            # misses vs to trigger pulls that could not
                            # connect. Watching these two columns is the
                            # fastest way to tell whether shooting is being
                            # explored and becoming profitable.
                            "missed_shot_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_missed_shot", 0.0)) for item in buf) / len(buf),
                            "useless_shot_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_useless_shot", 0.0)) for item in buf) / len(buf),
                        }
                    self.episode_metrics_buffer.clear()
                telemetry.write(payload)
                if run_control is not None:
                    run_control.update(**payload)
                self.last_telemetry_step = self.num_timesteps
            return not stop_training

        def _on_training_end(self) -> None:
            values = {
                "event": "training_end",
                "timesteps": self.num_timesteps,
                "episodes": self.episode_count,
            }
            telemetry.write(values)
            if run_control is not None:
                # CLI publishes Finished only after final checkpoints are
                # safely written. Until then this is still stopping work.
                run_control.update(state="Stopping" if run_control.stop_requested else "Running", **values)

    class EvaluationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            # Anchored in _on_training_start, not here: `num_timesteps` is
            # restored from the checkpoint when resuming, so an absolute
            # threshold of `evaluation_frequency` was already in the past
            # and the callback evaluated on EVERY step for the rest of the
            # run. Anchoring to the current step makes the schedule
            # relative to wherever training actually starts.
            self.next_evaluation = config.evaluation_frequency

        def _on_training_start(self) -> None:
            self.next_evaluation = self.num_timesteps + config.evaluation_frequency

        def _on_step(self) -> bool:
            nonlocal best_score, eval_patience_counter, stop_training
            if stop_training:
                return False
            if self.num_timesteps < self.next_evaluation:
                return True
            eval_kwargs = _env_kwargs(config)
            eval_kwargs["environment_count"] = max(1, config.evaluation_environment_count)
            summary = evaluate_model(
                self.model,
                eval_kwargs,
                episodes=config.evaluation_episodes,
                seed=config.seed + self.num_timesteps,
                output_dir=evaluations / f"step_{self.num_timesteps:09d}",
            )
            reward = float(summary.get("mean_episode_reward", 0.0))
            summary["timesteps"] = self.num_timesteps
            if pipeline is not None:
                # Checkpoint-time battery: frozen conditions, generalization
                # split from what the run ACTUALLY trained on, optional
                # league, training metrics aggregation. Results go into the
                # same step_NNNNNNNNN directory as the normal evaluation so
                # one directory answers "how was the policy at step N".
                from .checkpoint_eval import run_checkpoint_evaluation

                pipeline.timesteps = self.num_timesteps
                pipeline.note_checkpoint(
                    best_path if best_path.exists() else checkpoints / "latest.zip"
                )
                pipeline.save_state()
                checkpoint_report = run_checkpoint_evaluation(
                    self.model,
                    step=self.num_timesteps,
                    config=config,
                    pipeline=pipeline,
                    output_dir=evaluations / f"step_{self.num_timesteps:09d}",
                    device=device,
                    normal_summary=dict(summary),
                )
                summary["curriculum"] = pipeline.driver.curriculum_snapshot()
                for section in ("condition_evaluation", "generalization", "league"):
                    body = checkpoint_report.get(section)
                    if body is None:
                        continue
                    # Terse mirror in latest.json; the full row-level
                    # evidence lives in step_NNNNNNNNN/report.json.
                    summary[section] = {
                        key: value for key, value in body.items() if not key.endswith("_detail")
                    }
                checkpoint_event = {
                    "event": "checkpoint_evaluation",
                    "timesteps": self.num_timesteps,
                    "curriculum_level": pipeline.driver.level,
                    "report": str(evaluations / f"step_{self.num_timesteps:09d}" / "report.json"),
                }
                telemetry.write(checkpoint_event)
                if run_control is not None:
                    run_control.event("system", "checkpoint evaluation completed", checkpoint_event)
            (evaluations / "latest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            if reward > best_score:
                best_score = reward
                eval_patience_counter = 0
                self.model.save(best_path)
                (evaluations / "best.json").write_text(
                    json.dumps({"mean_reward": reward, "timesteps": self.num_timesteps, "checkpoint": str(best_path)}, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                eval_patience_counter += 1
                if config.early_stopping_patience > 0 and eval_patience_counter >= config.early_stopping_patience:
                    stop_training = True
            if config.min_eval_reward is not None and reward >= config.min_eval_reward:
                stop_training = True
            # Advance past the current step rather than by a fixed stride:
            # with N parallel environments `num_timesteps` jumps by N per
            # rollout and can overshoot the threshold by more than one
            # frequency, which would otherwise queue up back-to-back
            # evaluations.
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += config.evaluation_frequency
            return not stop_training

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config.checkpoint_frequency // config.environment_count),
        save_path=str(checkpoints),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = MetricsCallback()
    callbacks = CallbackList([checkpoint_callback, metrics_callback, EvaluationCallback()])
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
            "timesteps": model.num_timesteps,
            "training_steps_completed": max(0, model.num_timesteps - metrics_callback.start_timesteps),
            "stopped": bool(run_control is not None and run_control.stop_requested),
            "warm_start": warm_start,
        }
        if pipeline is not None:
            from .pipeline import write_manifest

            pipeline.timesteps = model.num_timesteps
            pipeline.save_state()
            # The manifest reflects the final curriculum state so a resume
            # (or an outside analysis) starts from the true end state.
            write_manifest(run_dir, pipeline.manifest())
            result["curriculum"] = pipeline.driver.curriculum_snapshot()
            result["manifest"] = str(run_dir / "run_manifest.json")
        (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if pipeline is not None:
            pipeline.close()
        telemetry.close()
        env.close()
