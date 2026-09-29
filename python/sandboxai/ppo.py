"""Stable-Baselines3 PPO orchestration for the Godot environment."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from .config import TrainingConfig
from .evaluation import (
    SynchronizedModel,
    evaluate_model,
    run_parallel_evaluations,
    write_evaluation_artifacts,
)
from .godot_env import GodotVecEnv
from .telemetry import JsonlTelemetry, ResourceMonitor
from .training_profile import PrefixedProfiler, TrainingProfiler

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


def _env_kwargs(
    config: TrainingConfig,
    profiler: TrainingProfiler | None = None,
    worker_count: int = 1,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "project_path": config.project,
        "godot_executable": config.godot_executable,
        "environment_count": config.environment_count,
        "enemy_count": config.enemy_count,
        "seed": config.seed,
        "curriculum_level": config.curriculum_level,
        "compact_infos": config.compact_training_infos,
        # 1 = single Godot process (historical). Evaluation bridges pass
        # 1 explicitly: they already run concurrently with each other and
        # are not the throughput bottleneck the training bridge is.
        "worker_count": int(worker_count),
    }
    if profiler is not None:
        values["profiler"] = profiler
    return values


class InferenceDeviceScheduler:
    """Phase-device placement for PPO when inference and updates diverge.

    For this workload (an 84 -> 128 -> 128 MLP, batches of 1-8) a CUDA
    forward pass is slower than CPU: each step pays a host->device copy,
    ~20 kernel launches and a blocking device->host readback, which
    dominates the actual matmuls. Moving only the *inference* phases
    (rollout collection + frozen evaluation) to CPU keeps the PPO update
    on the configured device while removing those round trips.

    Mechanics: ``model.device`` drives both ``collect_rollouts``' tensor
    placement and ``model.predict``, so switching it (plus the policy
    module) at rollout boundaries relocates exactly the inference phases;
    ``train()`` always runs with the policy back on the training device.
    Parameter values are bit-preserved by the round trip (a plain copy);
    only freshly *computed* rollout quantities differ, in the same way any
    device choice already does.
    """

    def __init__(self, model: Any, training_device: str, inference_device: str) -> None:
        self.model = model
        self.training_device = str(training_device)
        self.inference_device = str(inference_device)
        self._in_rollout = False

    @property
    def active(self) -> bool:
        return self.inference_device != self.training_device

    def enter_rollout(self) -> None:
        if not self.active or self._in_rollout:
            return
        import torch  # type: ignore

        self.model.policy.to(torch.device(self.inference_device))
        self.model.device = torch.device(self.inference_device)
        self._in_rollout = True

    def exit_rollout(self) -> None:
        if not self._in_rollout:
            return
        import torch  # type: ignore

        self.model.policy.to(torch.device(self.training_device))
        self.model.device = torch.device(self.training_device)
        self._in_rollout = False

    def training_device_context(self):
        """Context manager: run a block with the policy on the training
        device, restoring the rollout placement afterwards.

        Used around mid-rollout checkpoint saves so the saved model's
        device metadata (SB3 stores ``model.device`` in the zip) keeps
        naming the training device, not the transient inference device.
        """

        class _Restore:
            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.was_in_rollout = False

            def __enter__(self):
                self.was_in_rollout = self.scheduler._in_rollout
                self.scheduler.exit_rollout()
                return self

            def __exit__(self, *_args):
                if self.was_in_rollout:
                    self.scheduler.enter_rollout()
                return False

        return _Restore(self)


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

    profiler = TrainingProfiler() if config.profile_training else None
    if profiler is not None:
        profiler.set_metadata(
            device=device,
            environment_count=config.environment_count,
            enemy_count=config.enemy_count,
            rollout_length=config.rollout_length,
            rollout_batch_size=config.rollout_length * config.environment_count,
            minibatch_size=config.batch_size,
            compact_training_infos=config.compact_training_infos,
        )
    env_workers = config.resolved_env_workers()
    env = GodotVecEnv(**_env_kwargs(config, profiler=profiler, worker_count=env_workers))
    if profiler is not None:
        profiler.set_metadata(
            observation_floats=env.client.observation_dim,
            action_components=len(env.action_space.nvec),
            env_workers=env_workers,
        )
    resource_monitor = ResourceMonitor()
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
                "env_workers": env_workers,
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
                resource_started = time.perf_counter() if profiler is not None else 0.0
                resources = resource_monitor.snapshot()
                if profiler is not None:
                    profiler.record(
                        "callback.resource_snapshot", time.perf_counter() - resource_started
                    )
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
                    **resources,
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
                    payload["mean_trigger_pulls"] = sum(float(item.get("trigger_pulls", 0.0)) for item in buf) / len(buf)
                    payload["mean_near_miss_shots"] = sum(float(item.get("near_miss_shots", 0.0)) for item in buf) / len(buf)
                    payload["mean_useless_shots"] = sum(float(item.get("useless_shots", 0.0)) for item in buf) / len(buf)
                    payload["mean_cooldown_shots"] = sum(float(item.get("cooldown_shots", 0.0)) for item in buf) / len(buf)
                    payload["mean_survival_time"] = sum(float(item.get("survival_time", 0.0)) for item in buf) / len(buf)
                    payload["mean_accuracy"] = sum(float(item.get("accuracy", 0.0)) for item in buf) / len(buf)
                    payload["win_rate"] = sum(float(item.get("win", 0.0)) for item in buf) / len(buf)
                    payload["loss_rate"] = sum(float(item.get("loss", 0.0)) for item in buf) / len(buf)
                    if config.reward_breakdown_logging:
                        payload["reward_breakdown"] = {
                            "hits": sum(float(item.get("reward_breakdown", {}).get("reward_hits", 0.0)) for item in buf) / len(buf),
                            "kills": sum(float(item.get("reward_breakdown", {}).get("reward_kills", 0.0)) for item in buf) / len(buf),
                            "damage_reward": sum(float(item.get("reward_breakdown", {}).get("reward_damage", 0.0)) for item in buf) / len(buf),
                            "survive": sum(float(item.get("reward_breakdown", {}).get("reward_survive", 0.0)) for item in buf) / len(buf),
                            "positioning": sum(float(item.get("reward_breakdown", {}).get("reward_positioning", 0.0)) for item in buf) / len(buf),
                            "aiming": sum(float(item.get("reward_breakdown", {}).get("reward_aiming", 0.0)) for item in buf) / len(buf),
                            "passivity_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_passivity", 0.0)) for item in buf) / len(buf),
                            "combat_time_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_combat_time", 0.0)) for item in buf) / len(buf),
                            "damage_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_damage", 0.0)) for item in buf) / len(buf),
                            "death_penalty": sum(float(item.get("reward_breakdown", {}).get("penalty_death", 0.0)) for item in buf) / len(buf),
                            # Shot economy: how much reward was lost to real
                            # near-misses vs to trigger pulls that could not
                            # plausibly connect. Watching these two columns
                            # is the fastest way to tell whether shooting is
                            # being explored and becoming profitable.
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
            # Persistent evaluation bridge processes. Each boundary used
            # to spawn a fresh Godot process for the normal evaluation AND
            # one for the checkpoint battery (three with the league
            # enabled) - engine + project startup paid per boundary, and
            # two of the three FPS cliffs in the profiled runs were
            # exactly these boundaries. Reuse is exact: every evaluation
            # episode is re-seeded over the bridge (reset(seed) / staged
            # plans), and the engine derives the world deterministically
            # from that seed, so a reused process produces bit-identical
            # episodes. Created lazily at the first boundary; closed by
            # train_ppo's finally block.
            self.eval_env: Any = None
            self.eval_env_kwargs: dict[str, Any] | None = None
            self.battery_env_kwargs: dict[str, Any] | None = None
            self.battery_executor: Any = None
            self.normal_profiler: Any = None
            self.battery_profiler: Any = None

        def _ensure_eval_env(self) -> Any:
            if self.eval_env is not None:
                return self.eval_env
            assert self.eval_env_kwargs is not None, "evaluation env kwargs missing (training start not fired?)"
            from .godot_env import GodotGymEnv, GodotVecEnv

            started = time.perf_counter() if profiler is not None else 0.0
            eval_kwargs = dict(self.eval_env_kwargs)
            if int(eval_kwargs.get("environment_count", 1) or 1) > 1:
                env: Any = GodotVecEnv(**eval_kwargs)
            else:
                eval_kwargs["environment_count"] = 1
                env = GodotGymEnv(**eval_kwargs)
            if profiler is not None:
                profiler.record("eval.env_startup", time.perf_counter() - started)
                profiler.add("eval.bridge_spawns", 1)
            self.eval_env = env
            return env

        def _ensure_battery_executor(self) -> Any:
            if self.battery_executor is not None:
                return self.battery_executor
            assert self.battery_env_kwargs is not None, "battery env kwargs missing (training start not fired?)"
            from .checkpoint_eval import PlanExecutor

            started = time.perf_counter() if profiler is not None else 0.0
            self.battery_executor = PlanExecutor(
                self.battery_env_kwargs, skill_metrics=True, profiler=self.battery_profiler
            )
            if profiler is not None:
                profiler.record("eval.env_startup", time.perf_counter() - started)
                profiler.add("eval.bridge_spawns", 1)
            return self.battery_executor

        def close(self) -> None:
            # Called from train_ppo's finally block: bridge processes are
            # precious (seconds of engine startup each), so they live for
            # the whole run and die exactly once.
            for resource in (self.eval_env, self.battery_executor):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:  # pragma: no cover - shutdown best effort
                        pass
            self.eval_env = None
            self.battery_executor = None

        def _on_training_start(self) -> None:
            self.next_evaluation = self.num_timesteps + config.evaluation_frequency
            # Profiling views keep the evaluation bridges' transport
            # timings out of the training bridge's buckets (see
            # training_profile.PrefixedProfiler).
            if profiler is not None:
                normal_view: Any = PrefixedProfiler(profiler, "eval.normal.")
                battery_view: Any = PrefixedProfiler(profiler, "eval.battery.")
            else:
                normal_view = None
                battery_view = None
            self.normal_profiler = normal_view
            self.battery_profiler = battery_view
            if profiler is not None:
                profiler.set_metadata(
                    evaluation_execution=("parallel_roles" if pipeline is not None else "normal_only")
                )
            eval_kwargs = _env_kwargs(config)
            # More slots than requested episodes can only simulate ignored
            # work. Cap the persistent normal bridge without changing the
            # configured maximum or any episode in the evaluation set.
            eval_kwargs["environment_count"] = min(
                config.evaluation_episodes,
                max(1, config.evaluation_environment_count),
            )
            if normal_view is not None:
                eval_kwargs["profiler"] = normal_view
            self.eval_env_kwargs = eval_kwargs
            # The battery's planned episodes are result-invariant under
            # parallelism (PlanExecutor contract), so its bridge runs
            # `checkpoint_eval_environment_count` environments to amortise
            # per-step inference and transport across the batch. The
            # transport seed matches run_checkpoint_evaluation's own
            # construction exactly (the staged plans carry the seeds that
            # matter); compact infos are safe here because the battery
            # reads only terminal metrics and per-step events, both of
            # which compact mode keeps.
            from .pipeline import EVAL_MASTER_SEED_SALT

            self.battery_env_kwargs = {
                "project_path": str(config.project),
                "godot_executable": config.godot_executable,
                "environment_count": int(config.checkpoint_eval_environment_count),
                "enemy_count": config.enemy_count,
                "seed": config.seed + EVAL_MASTER_SEED_SALT,
                "curriculum_level": config.curriculum_level,
                "compact_infos": True,
            }
            if battery_view is not None:
                self.battery_env_kwargs["profiler"] = battery_view

        def _on_step(self) -> bool:
            nonlocal best_score, eval_patience_counter, stop_training
            if stop_training:
                return False
            if self.num_timesteps < self.next_evaluation:
                return True
            evaluation_started = time.perf_counter() if profiler is not None else 0.0
            step_directory = evaluations / f"step_{self.num_timesteps:09d}"
            # Both bridges are persistent. In auto-curriculum mode their
            # independent simulations run concurrently, while
            # SynchronizedModel serializes the tiny shared-policy calls.
            # The callback joins both jobs before looking at reward, so best
            # checkpoint and early-stop ordering is exactly unchanged.
            env = self._ensure_eval_env()
            if pipeline is not None:
                from .checkpoint_eval import (
                    run_checkpoint_evaluation,
                    write_checkpoint_report,
                )

                battery_executor = self._ensure_battery_executor()
                pipeline.timesteps = self.num_timesteps
                pipeline.note_checkpoint(
                    best_path if best_path.exists() else checkpoints / "latest.zip"
                )
                pipeline.save_state()
                frozen_model = SynchronizedModel(self.model)
                summary, checkpoint_report, parallel_timing = run_parallel_evaluations(
                    lambda: evaluate_model(
                        frozen_model,
                        self.eval_env_kwargs,
                        episodes=config.evaluation_episodes,
                        seed=config.seed + self.num_timesteps,
                        # Written after both workers join. The generalization
                        # exporter owns episodes.csv in this shared directory;
                        # normal rows use normal_episodes.csv instead.
                        output_dir=None,
                        env=env,
                        profiler=profiler,
                    ),
                    lambda: run_checkpoint_evaluation(
                        frozen_model,
                        step=self.num_timesteps,
                        config=config,
                        pipeline=pipeline,
                        output_dir=step_directory,
                        device=device,
                        normal_summary=None,
                        executor=battery_executor,
                        profiler=profiler,
                        write_report=False,
                    ),
                )
                summary["timesteps"] = self.num_timesteps
                write_evaluation_artifacts(
                    summary, step_directory, episodes_name="normal_episodes.csv"
                )
                checkpoint_report["normal_evaluation"] = dict(summary)
                write_checkpoint_report(checkpoint_report, step_directory)
                if profiler is not None:
                    profiler.record("eval.parallel.wall", parallel_timing["wall_seconds"])
                    profiler.record(
                        "eval.parallel.overlap", parallel_timing["overlap_seconds"]
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
                    "report": str(step_directory / "report.json"),
                }
                telemetry.write(checkpoint_event)
                if run_control is not None:
                    run_control.event("system", "checkpoint evaluation completed", checkpoint_event)
            else:
                summary = evaluate_model(
                    self.model,
                    self.eval_env_kwargs,
                    episodes=config.evaluation_episodes,
                    seed=config.seed + self.num_timesteps,
                    output_dir=step_directory,
                    env=env,
                    # Raw profiler: evaluate_model records fully-qualified
                    # eval.normal.* buckets; the prefixed view above is only
                    # for the bridge transport's bridge.* names.
                    profiler=profiler,
                )
                summary["timesteps"] = self.num_timesteps
            reward = float(summary.get("mean_episode_reward", 0.0))
            (evaluations / "latest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            best_started = time.perf_counter() if profiler is not None else 0.0
            if reward > best_score:
                best_score = reward
                eval_patience_counter = 0
                if inference_scheduler is not None:
                    with inference_scheduler.training_device_context():
                        self.model.save(best_path)
                else:
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
            if profiler is not None:
                profiler.record("eval.best_checkpoint_save", time.perf_counter() - best_started)
            # Advance past the current step rather than by a fixed stride:
            # with N parallel environments `num_timesteps` jumps by N per
            # rollout and can overshoot the threshold by more than one
            # frequency, which would otherwise queue up back-to-back
            # evaluations.
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += config.evaluation_frequency
            if profiler is not None:
                profiler.record(
                    "callback.evaluation", time.perf_counter() - evaluation_started
                )
            return not stop_training

    class TimedCheckpointCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            should_save = self.n_calls % self.save_freq == 0
            started = time.perf_counter() if profiler is not None and should_save else 0.0
            # Mid-rollout saves record model.device in the zip metadata;
            # keep that at the training device under the split-device mode.
            if inference_scheduler is not None:
                with inference_scheduler.training_device_context():
                    keep_going = super()._on_step()
            else:
                keep_going = super()._on_step()
            if profiler is not None and should_save:
                profiler.record("callback.checkpoint_save", time.perf_counter() - started)
            return keep_going

    class ProfilingCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.rollout_started = 0.0
            self.update_started = 0.0
            self.pending_rollout: tuple[float, int] | None = None
            self.iteration = 0

        def _finish_update(self) -> None:
            if profiler is None or self.pending_rollout is None:
                return
            update_seconds = max(0.0, time.perf_counter() - self.update_started)
            rollout_seconds, timesteps = self.pending_rollout
            profiler.record("ppo.policy_update", update_seconds)
            self.iteration += 1
            profiler.add_iteration(
                self.iteration, rollout_seconds, update_seconds, timesteps
            )
            self.pending_rollout = None

        def _on_rollout_start(self) -> None:
            self._finish_update()
            self.rollout_started = time.perf_counter()

        def _on_rollout_end(self) -> None:
            if profiler is None:
                return
            rollout_seconds = max(0.0, time.perf_counter() - self.rollout_started)
            profiler.record("ppo.rollout_collection", rollout_seconds)
            self.pending_rollout = (rollout_seconds, self.num_timesteps)
            self.update_started = time.perf_counter()

        def _on_training_end(self) -> None:
            self._finish_update()

        def _on_step(self) -> bool:
            return True

    class InferenceDeviceCallback(BaseCallback):
        """Switches policy placement at rollout boundaries.

        See :class:`InferenceDeviceScheduler`. Inactive (a no-op callback)
        unless ``config.inference_device`` resolves differently from the
        training device. Placement is switched at ``_on_rollout_start``
        (before the first forward of the rollout) and restored at
        ``_on_rollout_end`` (which SB3 fires before ``train()``), so
        rollout collection, the mid-rollout evaluation callback and any
        terminal-value bootstraps all run on the inference device while
        the PPO update always runs on the training device.
        """

        def __init__(self, scheduler: InferenceDeviceScheduler) -> None:
            super().__init__()
            self.scheduler = scheduler

        def _on_rollout_start(self) -> None:
            self.scheduler.enter_rollout()

        def _on_rollout_end(self) -> None:
            self.scheduler.exit_rollout()

        def _on_training_end(self) -> None:
            # A stop during rollout collection must not leave the policy
            # stranded on the inference device for the final saves.
            self.scheduler.exit_rollout()

        def _on_step(self) -> bool:
            return True

    checkpoint_callback = TimedCheckpointCallback(
        save_freq=max(1, config.checkpoint_frequency // config.environment_count),
        save_path=str(checkpoints),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = MetricsCallback()
    evaluation_callback = EvaluationCallback()
    callback_items: list[Any] = [checkpoint_callback, metrics_callback, evaluation_callback]
    if profiler is not None:
        callback_items.insert(0, ProfilingCallback())
    inference_scheduler: InferenceDeviceScheduler | None = None
    if config.inference_device != "auto":
        inference_scheduler = InferenceDeviceScheduler(model=None, training_device=device, inference_device=config.resolved_inference_device())
        callback_items.insert(0, InferenceDeviceCallback(inference_scheduler))
    callbacks = CallbackList(callback_items)
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
        if profiler is not None:
            rollout_samples = int(model.n_steps) * int(model.n_envs)
            profiler.set_metadata(
                ppo_epochs=int(model.n_epochs),
                rollout_batch_size=rollout_samples,
                minibatch_size=int(model.batch_size),
                minibatches_per_update=(
                    int(model.n_epochs)
                    * ((rollout_samples + int(model.batch_size) - 1) // int(model.batch_size))
                ),
            )
        if inference_scheduler is not None:
            inference_scheduler.model = model
            if profiler is not None:
                profiler.set_metadata(
                    inference_device=inference_scheduler.inference_device,
                    training_device=inference_scheduler.training_device,
                )
        model.learn(
            total_timesteps=config.total_training_steps,
            callback=callbacks,
            reset_num_timesteps=not bool(checkpoint_path),
            progress_bar=False,
        )
        latest = checkpoints / "latest.zip"
        final_save_started = time.perf_counter() if profiler is not None else 0.0
        model.save(latest)
        model.save(run_dir / "final.zip")
        if profiler is not None:
            profiler.record("checkpoint.final_saves", time.perf_counter() - final_save_started)
            profiler.server = env.client.profile_snapshot()
            profiler.set_metadata(
                actual_timesteps=model.num_timesteps,
                training_steps_completed=max(
                    0, model.num_timesteps - metrics_callback.start_timesteps
                ),
            )
            profile_path = profiler.write(logs / "training_profile.json")
        else:
            profile_path = None
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
            "training_profile": str(profile_path) if profile_path is not None else None,
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
        if inference_scheduler is not None:
            # Defensive: never leave the policy on the inference device
            # after an exception (final saves would record it).
            inference_scheduler.exit_rollout()
        evaluation_callback.close()
        if pipeline is not None:
            pipeline.close()
        resource_monitor.close()
        telemetry.close()
        env.close()
