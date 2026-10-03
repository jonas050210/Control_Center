"""Stable-Baselines3 PPO orchestration for the Godot environment."""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        from stable_baselines3.common.callbacks import (  # type: ignore
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
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

    For this workload (a 106 -> 128 -> 128 MLP, batches of 1-8) a CUDA
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


@dataclass
class _SelectionState:
    """Checkpoint-selection state shared by the metrics and evaluation callbacks.

    `best_score` is in the units of `config.checkpoint_selection_rule()`,
    not necessarily mean reward; `stop_training` is the single early-stop
    flag that both callbacks honour so a stop request takes effect at the
    next step boundary no matter which callback observed it.
    """

    best_score: float
    eval_patience_counter: int = 0
    stop_training: bool = False
    # Why the run stopped early ("" = it did not). The first observer wins:
    # an operator stop, the wall-clock budget the trainer enforces itself, or
    # one of the evaluation-driven early stops below.
    stop_reason: str = ""


class _TimeBudget:
    """The trainer's own wall-clock budget, in seconds (0 = no budget).

    A tiny object rather than arithmetic inline in the callback: this is
    the whole policy ("has the budget been spent?") and it is testable
    without stable-baselines3, torch or an engine.
    """

    def __init__(self, minutes: float) -> None:
        self.minutes = max(0.0, float(minutes))
        self.seconds = self.minutes * 60.0
        self.started = time.monotonic()

    def restart(self) -> None:
        """A resumed run gets a fresh budget; the clock starts with the loop."""
        self.started = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def enabled(self) -> bool:
        return self.seconds > 0.0

    def reached(self) -> bool:
        return self.enabled and self.elapsed_seconds() >= self.seconds


def _stop_for_operator(run_control: RunControl | None, state: _SelectionState) -> bool:
    """Operator commands first: a stop request always wins."""
    if run_control is None or run_control.checkpoint():
        return False
    state.stop_training = True
    state.stop_reason = state.stop_reason or "operator"
    return True


def _should_continue_step(
    budget: _TimeBudget,
    state: _SelectionState,
    *,
    run_control: RunControl | None,
    telemetry: Any,
    timesteps: int,
) -> bool:
    """One place decides whether this PPO step may continue.

    Module-level so the whole stop policy (operator, then wall clock) is
    testable without stable-baselines3, torch or an engine.
    """
    if state.stop_training:
        # Something already ended this run (an operator stop on an earlier
        # step, an evaluation early stop, or this budget). Never re-announce
        # it: the reason that won is the one the summary reports.
        return False
    if _stop_for_operator(run_control, state):
        return False
    return not _stop_for_time_budget(
        budget, state, telemetry=telemetry, run_control=run_control, timesteps=timesteps
    )


def _stop_for_time_budget(
    budget: _TimeBudget,
    state: _SelectionState,
    *,
    telemetry: Any,
    run_control: RunControl | None,
    timesteps: int,
) -> bool:
    """Stop training once the wall-clock budget is spent; report the reason.

    Module-level so the policy is testable without stable-baselines3: it
    takes the clock, the shared stop state and the two publishers, and
    returns whether training must end at this step boundary. The final
    checkpoint is written by PPO's normal shutdown, exactly as for an
    operator stop.
    """
    if state.stop_training or not budget.reached():
        return False
    state.stop_training = True
    state.stop_reason = "time_budget"
    values = {
        "event": "time_budget_reached",
        "elapsed_seconds": round(budget.elapsed_seconds(), 3),
        "budget_minutes": budget.minutes,
        "timesteps": timesteps,
    }
    telemetry.write(values)
    if run_control is not None:
        run_control.update(state="Stopping", **values)
        run_control.event(
            "system",
            f"wall-clock budget of {budget.minutes:g} min reached; saving the final checkpoint",
            values,
        )
    return True


# --- callback factories --------------------------------------------------
#
# Every callback below used to be a class nested inside train_ppo, closing
# over a dozen of its locals (and mutating three of them through `nonlocal`).
# That made train_ppo an 880-line function with a cyclomatic complexity of
# 103 and meant no callback could be constructed - let alone tested -
# without starting a training run. The classes still capture their
# dependencies in a closure, but the closure is now an explicit, named
# parameter list per factory.
#
# The SB3 base classes are passed in rather than imported at module level:
# stable-baselines3 is an optional extra (`pip install -e ".[training]"`)
# and importing it here would make the numpy-only install fail.


def _make_profiled_ppo(ppo_class: Any, profiler: Any) -> Any:
    class ProfiledPPO(ppo_class):
        """PPO with an exact timer around SB3's optimizer update."""

        def train(self) -> None:
            started = time.perf_counter()
            try:
                super().train()
            finally:
                profiler.record("ppo.optimizer_update", time.perf_counter() - started)

    return ProfiledPPO


# Episode metrics averaged into every progress row: payload key -> info key.
_EPISODE_MEAN_KEYS: tuple[tuple[str, str], ...] = (
    ("mean_episode_reward", "episode_reward"),
    ("mean_kills", "kills"),
    ("mean_deaths", "deaths"),
    ("mean_damage_dealt", "damage_dealt"),
    ("mean_damage_received", "damage_received"),
    ("mean_shots_fired", "shots_fired"),
    ("mean_shots_hit", "shots_hit"),
    ("mean_trigger_pulls", "trigger_pulls"),
    ("mean_near_miss_shots", "near_miss_shots"),
    ("mean_useless_shots", "useless_shots"),
    ("mean_cooldown_shots", "cooldown_shots"),
    ("mean_survival_time", "survival_time"),
    ("mean_accuracy", "accuracy"),
    ("win_rate", "win"),
    ("loss_rate", "loss"),
)

# Reward-shaping terms, reported separately so a run can be read as "what
# is actually paying out". The last two are the shot economy: how much
# reward was lost to real near-misses vs to trigger pulls that could not
# plausibly connect - the fastest way to tell whether shooting is being
# explored and becoming profitable.
_REWARD_BREAKDOWN_KEYS: tuple[tuple[str, str], ...] = (
    ("hits", "reward_hits"),
    ("kills", "reward_kills"),
    ("damage_reward", "reward_damage"),
    ("survive", "reward_survive"),
    ("positioning", "reward_positioning"),
    ("aiming", "reward_aiming"),
    ("passivity_penalty", "penalty_passivity"),
    ("combat_time_penalty", "penalty_combat_time"),
    ("damage_penalty", "penalty_damage"),
    ("death_penalty", "penalty_death"),
    ("missed_shot_penalty", "penalty_missed_shot"),
    ("useless_shot_penalty", "penalty_useless_shot"),
)


def _count_shoot_requests(actions: Any) -> tuple[int, int]:
    """Returns (action decisions, shoot requests) in one batch of policy actions.

    Counted on the raw policy output, before it crosses JSON, the bridge
    and weapon handling, so an engine-side drop cannot be mistaken for the
    policy never asking to shoot. Component 4 is the fire bit; shorter
    action vectors predate it and are counted as no decision at all.
    """
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    decisions = 0
    shoot_requests = 0
    for action in actions:
        values = action.tolist() if hasattr(action, "tolist") else list(action)
        if len(values) > 4:
            decisions += 1
            shoot_requests += int(int(values[4]) == 1)
    return decisions, shoot_requests


def _finished_episode_metrics(infos: Any) -> list[dict[str, Any]]:
    """Metrics dicts of the episodes that ended in this vector step."""
    finished = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        if info.get("done_reason") or info.get("terminal_observation") is not None:
            finished.append(info.get("metrics", {}))
    return finished


def _mean(buffer: Sequence[dict[str, Any]], key: str, source: str | None = None) -> float:
    """Mean of `key` over `buffer`, reading it out of `source` when nested.

    An empty buffer means "no episodes finished in this interval", which
    is a normal state early in a rollout, not an error. It averages to
    0.0 rather than raising: the callers feed this straight into a
    progress payload, and a crash there would take down a training run
    over a missing log line.
    """
    if not buffer:
        return 0.0
    total = 0.0
    for item in buffer:
        container = item if source is None else item.get(source, {})
        total += float(container.get(key, 0.0))
    return total / len(buffer)


def _episode_means(buffer: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-episode averages for one telemetry interval."""
    return {payload_key: _mean(buffer, key) for payload_key, key in _EPISODE_MEAN_KEYS}


def _reward_breakdown_means(buffer: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-episode averages of the individual reward-shaping terms."""
    return {
        payload_key: _mean(buffer, key, "reward_breakdown")
        for payload_key, key in _REWARD_BREAKDOWN_KEYS
    }


def _progress_payload(
    config: TrainingConfig,
    *,
    device: str,
    checkpoints: Path,
    elapsed: float,
    num_timesteps: int,
    start_timesteps: int,
    target_timesteps: int,
    episode_count: int,
    interval_decisions: int,
    interval_shoot_requests: int,
    episode_metrics: Sequence[dict[str, Any]],
    resources: dict[str, Any],
) -> dict[str, Any]:
    """Builds one `progress` telemetry row. Pure: it does not touch the buffer."""
    completed_steps = max(0, num_timesteps - start_timesteps)
    steps_per_second = completed_steps / elapsed
    remaining_steps = max(0, target_timesteps - num_timesteps)
    payload: dict[str, Any] = {
        "event": "progress",
        "timesteps": num_timesteps,
        "run_start_timesteps": start_timesteps,
        "total_training_steps": target_timesteps,
        "progress": min(1.0, completed_steps / config.total_training_steps),
        "steps_per_second": steps_per_second,
        # Based on the measured run-average throughput. It is omitted
        # until a positive rate exists rather than faked.
        "eta_seconds": remaining_steps / steps_per_second if steps_per_second > 0 else None,
        "episodes": episode_count,
        "environment_count": config.environment_count,
        "device": device,
        "current_checkpoint": str(checkpoints),
        "policy_action_decisions": interval_decisions,
        "policy_shoot_requests": interval_shoot_requests,
        "policy_shoot_request_rate": (
            interval_shoot_requests / interval_decisions if interval_decisions else 0.0
        ),
        **resources,
    }
    if episode_metrics:
        payload.update(_episode_means(episode_metrics))
        if config.reward_breakdown_logging:
            payload["reward_breakdown"] = _reward_breakdown_means(episode_metrics)
    return payload


def _make_metrics_callback(
    BaseCallback: Any,
    *,
    config: TrainingConfig,
    telemetry: Any,
    run_control: RunControl | None,
    pipeline: Any,
    resource_monitor: Any,
    profiler: Any,
    device: str,
    env_workers: int,
    torch_threads: int,
    checkpoints: Path,
    selection_rule: Any,
    state: _SelectionState,
) -> Any:
    """Per-step telemetry, episode accounting and the run's progress record."""

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.perf_counter()
            # The trainer's own budget: a run resumed from a checkpoint gets
            # a fresh one, because "this run may take N minutes" is a
            # property of the run, not of the checkpoint lineage.
            self.budget = _TimeBudget(config.max_train_minutes)
            self.last_telemetry_step = 0
            self.episode_count = 0
            self.episode_metrics_buffer: list[dict[str, Any]] = []
            self.start_timesteps = 0
            self.target_timesteps = config.total_training_steps
            self.action_decisions = 0
            self.shoot_requests = 0
            self.interval_action_decisions = 0
            self.interval_shoot_requests = 0

        def _on_training_start(self) -> None:
            self.budget.restart()
            self.start_timesteps = self.num_timesteps
            self.target_timesteps = self.start_timesteps + config.total_training_steps
            self.last_telemetry_step = self.start_timesteps
            start_values = {
                "event": "training_start",
                "environment_count": config.environment_count,
                "env_workers": env_workers,
                "checkpoint_selection": selection_rule.as_dict(),
                "device": device,
                "run_start_timesteps": self.start_timesteps,
                "total_training_steps": self.target_timesteps,
                "requested_training_steps": config.total_training_steps,
                "rollout_schedule": config.rollout_schedule(),
                "ppo_epochs": config.ppo_epochs,
                "torch_threads": torch_threads,
                "net_arch": list(config.net_arch),
                "learning_rate": config.learning_rate,
            }
            telemetry.write(start_values)
            if run_control is not None:
                run_control.running(**start_values)
                run_control.event("system", "PPO optimization started", start_values)

        def _on_step(self) -> bool:
            if not _should_continue_step(
                self.budget,
                state,
                run_control=run_control,
                telemetry=telemetry,
                timesteps=self.num_timesteps,
            ):
                return False
            if pipeline is not None:
                # One attribute write per vector step: the pipeline stamps
                # episode rows / curriculum transitions with real PPO time.
                pipeline.timesteps = self.num_timesteps
            # Count the policy output before it crosses JSON/bridge/weapon
            # handling. This is the missing diagnostic for a zero-shot run:
            # stochastic PPO exploration can now be distinguished from a
            # deterministic eval argmax and from an engine-side drop.
            decisions, shoot_requests = _count_shoot_requests(self.locals.get("actions", []))
            self.action_decisions += decisions
            self.interval_action_decisions += decisions
            self.shoot_requests += shoot_requests
            self.interval_shoot_requests += shoot_requests
            for metrics in _finished_episode_metrics(self.locals.get("infos", [])):
                self.episode_count += 1
                self.episode_metrics_buffer.append(metrics)
            due = max(config.environment_count, 1) * 100
            if self.num_timesteps - self.last_telemetry_step >= due:
                self._emit_progress()
            return not state.stop_training

        def _emit_progress(self) -> None:
            """Writes one throughput/episode row and resets the interval counters."""
            resource_started = time.perf_counter() if profiler is not None else 0.0
            resources = resource_monitor.snapshot()
            if profiler is not None:
                profiler.record(
                    "callback.resource_snapshot", time.perf_counter() - resource_started
                )
            payload = _progress_payload(
                config,
                device=device,
                checkpoints=checkpoints,
                elapsed=max(time.perf_counter() - self.started, 1e-9),
                num_timesteps=self.num_timesteps,
                start_timesteps=self.start_timesteps,
                target_timesteps=self.target_timesteps,
                episode_count=self.episode_count,
                interval_decisions=self.interval_action_decisions,
                interval_shoot_requests=self.interval_shoot_requests,
                episode_metrics=self.episode_metrics_buffer,
                resources=resources,
            )
            self.episode_metrics_buffer.clear()
            telemetry.write(payload)
            if run_control is not None:
                run_control.update(**payload)
            self.interval_action_decisions = 0
            self.interval_shoot_requests = 0
            self.last_telemetry_step = self.num_timesteps

        def _on_training_end(self) -> None:
            values = {
                "event": "training_end",
                "timesteps": self.num_timesteps,
                "episodes": self.episode_count,
                "policy_action_decisions": self.action_decisions,
                "policy_shoot_requests": self.shoot_requests,
                "policy_shoot_request_rate": (
                    self.shoot_requests / self.action_decisions if self.action_decisions else 0.0
                ),
            }
            telemetry.write(values)
            if profiler is not None:
                profiler.add("policy.action_decisions", self.action_decisions)
                profiler.add("policy.shoot_requests", self.shoot_requests)
            if run_control is not None:
                # CLI publishes Finished only after final checkpoints are
                # safely written. Until then this is still stopping work.
                run_control.update(
                    state="Stopping" if run_control.stop_requested else "Running", **values
                )

    return MetricsCallback()


def _evaluation_env_kwargs(config: TrainingConfig, profiler_view: Any) -> dict[str, Any]:
    """Bridge kwargs for the persistent normal-evaluation environment.

    More slots than requested episodes can only simulate ignored work, so
    the persistent bridge is capped at the episode count without changing
    the configured maximum or any episode in the evaluation set.
    """
    kwargs = _env_kwargs(config)
    kwargs["environment_count"] = min(
        config.evaluation_episodes, max(1, config.evaluation_environment_count)
    )
    if profiler_view is not None:
        kwargs["profiler"] = profiler_view
    return kwargs


def _battery_env_kwargs(config: TrainingConfig, profiler_view: Any) -> dict[str, Any]:
    """Bridge kwargs for the persistent checkpoint-battery environment.

    The battery's planned episodes are result-invariant under parallelism
    (PlanExecutor contract), so its bridge runs
    ``checkpoint_eval_environment_count`` environments to amortise
    per-step inference and transport across the batch. The transport seed
    matches run_checkpoint_evaluation's own construction exactly (the
    staged plans carry the seeds that matter); compact infos are safe
    because the battery reads only terminal metrics and per-step events,
    both of which compact mode keeps.
    """
    from .pipeline import EVAL_MASTER_SEED_SALT

    kwargs: dict[str, Any] = {
        "project_path": str(config.project),
        "godot_executable": config.godot_executable,
        "environment_count": int(config.checkpoint_eval_environment_count),
        "enemy_count": config.enemy_count,
        "seed": config.seed + EVAL_MASTER_SEED_SALT,
        "curriculum_level": config.curriculum_level,
        "compact_infos": True,
    }
    if profiler_view is not None:
        kwargs["profiler"] = profiler_view
    return kwargs


class _EvaluationDriver:
    """Everything the evaluation callback does, with no stable-baselines3 in sight.

    The SB3 callback below is a five-line shim over this object. Keeping
    the logic here means periodic evaluation, the checkpoint battery and
    best-checkpoint selection can be driven - and tested - with a plain
    object that exposes `predict`, instead of a live PPO run.
    """

    def __init__(
        self,
        *,
        config: TrainingConfig,
        telemetry: Any,
        run_control: RunControl | None,
        pipeline: Any,
        profiler: Any,
        device: str,
        checkpoints: Path,
        evaluations: Path,
        best_path: Path,
        best_record_path: Path,
        selection_rule: Any,
        inference_scheduler: InferenceDeviceScheduler | None,
        state: _SelectionState,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.run_control = run_control
        self.pipeline = pipeline
        self.profiler = profiler
        self.device = device
        self.checkpoints = checkpoints
        self.evaluations = evaluations
        self.best_path = best_path
        self.best_record_path = best_record_path
        self.selection_rule = selection_rule
        self.inference_scheduler = inference_scheduler
        self.state = state

        # Anchored in _on_training_start, not here: `num_timesteps` is
        # restored from the checkpoint when resuming, so an absolute
        # threshold of `evaluation_frequency` was already in the past
        # and the callback evaluated on EVERY step for the rest of the
        # run. Anchoring to the current step makes the schedule
        # relative to wherever training actually starts.
        self.next_evaluation = self.config.evaluation_frequency
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

    def ensure_eval_env(self) -> Any:
        if self.eval_env is not None:
            return self.eval_env
        assert self.eval_env_kwargs is not None, (
            "evaluation env kwargs missing (training start not fired?)"
        )
        from .godot_env import GodotGymEnv, GodotVecEnv

        started = time.perf_counter() if self.profiler is not None else 0.0
        eval_kwargs = dict(self.eval_env_kwargs)
        if int(eval_kwargs.get("environment_count", 1) or 1) > 1:
            env: Any = GodotVecEnv(**eval_kwargs)
        else:
            eval_kwargs["environment_count"] = 1
            env = GodotGymEnv(**eval_kwargs)
        if self.profiler is not None:
            self.profiler.record("eval.env_startup", time.perf_counter() - started)
            self.profiler.add("eval.bridge_spawns", 1)
        self.eval_env = env
        return env

    def ensure_battery_executor(self) -> Any:
        if self.battery_executor is not None:
            return self.battery_executor
        assert self.battery_env_kwargs is not None, (
            "battery env kwargs missing (training start not fired?)"
        )
        from .checkpoint_eval import PlanExecutor

        started = time.perf_counter() if self.profiler is not None else 0.0
        self.battery_executor = PlanExecutor(
            self.battery_env_kwargs, skill_metrics=True, profiler=self.battery_profiler
        )
        if self.profiler is not None:
            self.profiler.record("eval.env_startup", time.perf_counter() - started)
            self.profiler.add("eval.bridge_spawns", 1)
        return self.battery_executor

    def close(self) -> None:
        # Called from train_ppo's finally block: bridge processes are
        # precious (seconds of engine startup each), so they live for
        # the whole run and die exactly once.
        for resource in (self.eval_env, self.battery_executor):
            if resource is not None:
                # Shutdown is best effort: a bridge that already died
                # must not mask the real training result.
                with contextlib.suppress(Exception):
                    resource.close()
        self.eval_env = None
        self.battery_executor = None

    def on_training_start(self, timesteps: int) -> None:
        self.next_evaluation = timesteps + self.config.evaluation_frequency
        # Profiling views keep the evaluation bridges' transport
        # timings out of the training bridge's buckets (see
        # training_profile.PrefixedProfiler).
        if self.profiler is not None:
            normal_view: Any = PrefixedProfiler(self.profiler, "eval.normal.")
            battery_view: Any = PrefixedProfiler(self.profiler, "eval.battery.")
        else:
            normal_view = None
            battery_view = None
        self.normal_profiler = normal_view
        self.battery_profiler = battery_view
        if self.profiler is not None:
            self.profiler.set_metadata(
                evaluation_execution=(
                    "parallel_roles" if self.pipeline is not None else "normal_only"
                )
            )
        self.eval_env_kwargs = _evaluation_env_kwargs(self.config, normal_view)
        self.battery_env_kwargs = _battery_env_kwargs(self.config, battery_view)

    def on_step(self, model: Any, timesteps: int) -> bool:
        if self.state.stop_training:
            return False
        if timesteps < self.next_evaluation:
            return True
        evaluation_started = time.perf_counter() if self.profiler is not None else 0.0
        step_directory = self.evaluations / f"step_{timesteps:09d}"
        # Both bridges are persistent. In auto-curriculum mode their
        # independent simulations run concurrently, while
        # SynchronizedModel serializes the tiny shared-policy calls.
        # The callback joins both jobs before looking at reward, so best
        # checkpoint and early-stop ordering is exactly unchanged.
        env = self.ensure_eval_env()
        if self.pipeline is not None:
            summary = self._pipeline_evaluation(model, timesteps, env, step_directory)
        else:
            summary = self._plain_evaluation(model, timesteps, env, step_directory)
        reward = float(summary.get("mean_episode_reward", 0.0))
        (self.evaluations / "latest.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        best_started = time.perf_counter() if self.profiler is not None else 0.0
        self._apply_selection(model, timesteps, summary, reward)
        if self.profiler is not None:
            self.profiler.record("eval.best_checkpoint_save", time.perf_counter() - best_started)
        # Advance past the current step rather than by a fixed stride:
        # with N parallel environments `num_timesteps` jumps by N per
        # rollout and can overshoot the threshold by more than one
        # frequency, which would otherwise queue up back-to-back
        # evaluations.
        while self.next_evaluation <= timesteps:
            self.next_evaluation += self.config.evaluation_frequency
        if self.profiler is not None:
            self.profiler.record("callback.evaluation", time.perf_counter() - evaluation_started)
        return not self.state.stop_training

    def _pipeline_evaluation(
        self, model: Any, timesteps: int, env: Any, step_directory: Path
    ) -> dict[str, Any]:
        """Normal evaluation and the checkpoint battery, run concurrently."""
        from .checkpoint_eval import (
            run_checkpoint_evaluation,
            write_checkpoint_report,
        )

        battery_executor = self.ensure_battery_executor()
        self.pipeline.timesteps = timesteps
        self.pipeline.note_checkpoint(
            self.best_path if self.best_path.exists() else self.checkpoints / "latest.zip"
        )
        self.pipeline.save_state()
        frozen_model = SynchronizedModel(model)
        eval_env_kwargs = self.eval_env_kwargs
        assert eval_env_kwargs is not None, "on_training_start has not run"
        summary, checkpoint_report, parallel_timing = run_parallel_evaluations(
            lambda: evaluate_model(
                frozen_model,
                eval_env_kwargs,
                episodes=self.config.evaluation_episodes,
                seed=self.config.seed + timesteps,
                # Written after both workers join. The generalization
                # exporter owns episodes.csv in this shared directory;
                # normal rows use normal_episodes.csv instead.
                output_dir=None,
                env=env,
                profiler=self.profiler,
            ),
            lambda: run_checkpoint_evaluation(
                frozen_model,
                step=timesteps,
                config=self.config,
                pipeline=self.pipeline,
                output_dir=step_directory,
                device=self.device,
                normal_summary=None,
                executor=battery_executor,
                profiler=self.profiler,
                write_report=False,
            ),
        )
        summary["timesteps"] = timesteps
        write_evaluation_artifacts(summary, step_directory, episodes_name="normal_episodes.csv")
        checkpoint_report["normal_evaluation"] = dict(summary)
        write_checkpoint_report(checkpoint_report, step_directory)
        if self.profiler is not None:
            self.profiler.record("eval.parallel.wall", parallel_timing["wall_seconds"])
            self.profiler.record("eval.parallel.overlap", parallel_timing["overlap_seconds"])
        summary["curriculum"] = self.pipeline.driver.curriculum_snapshot()
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
            "timesteps": timesteps,
            "curriculum_level": self.pipeline.driver.level,
            "report": str(step_directory / "report.json"),
        }
        self.telemetry.write(checkpoint_event)
        if self.run_control is not None:
            self.run_control.event("system", "checkpoint evaluation completed", checkpoint_event)
        return summary

    def _plain_evaluation(
        self, model: Any, timesteps: int, env: Any, step_directory: Path
    ) -> dict[str, Any]:
        """Single-bridge evaluation used when the research pipeline is off."""
        assert self.eval_env_kwargs is not None, "on_training_start has not run"
        summary = evaluate_model(
            model,
            self.eval_env_kwargs,
            episodes=self.config.evaluation_episodes,
            seed=self.config.seed + timesteps,
            output_dir=step_directory,
            env=env,
            # Raw profiler: evaluate_model records fully-qualified
            # eval.normal.* buckets; the prefixed view above is only
            # for the bridge transport's bridge.* names.
            profiler=self.profiler,
        )
        summary["timesteps"] = timesteps
        return summary

    def _apply_selection(
        self, model: Any, timesteps: int, summary: dict[str, Any], reward: float
    ) -> None:
        """Saves a new best checkpoint or advances the early-stop counter."""
        score = self.selection_rule.score(summary)
        if score is None:
            # A rule pointed at a metric this run does not produce must
            # not silently select on something else; it counts as "no
            # improvement" and is reported once per evaluation.
            self.telemetry.write(
                {
                    "event": "checkpoint_selection_metric_missing",
                    "metric": self.selection_rule.metric,
                    "timesteps": timesteps,
                }
            )
        if score is not None and self.selection_rule.is_improvement(score, self.state.best_score):
            self.state.best_score = score
            self.state.eval_patience_counter = 0
            if self.inference_scheduler is not None:
                with self.inference_scheduler.training_device_context():
                    model.save(self.best_path)
            else:
                model.save(self.best_path)
            self.best_record_path.write_text(
                json.dumps(
                    {
                        # Legacy field, kept so existing readers and
                        # resume paths keep working.
                        "mean_reward": reward,
                        "score": score,
                        "timesteps": timesteps,
                        "checkpoint": str(self.best_path),
                        "selection_rule": self.selection_rule.as_dict(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            self.state.eval_patience_counter += 1
            if (
                self.config.early_stopping_patience > 0
                and self.state.eval_patience_counter >= self.config.early_stopping_patience
            ):
                self.state.stop_training = True
                self.state.stop_reason = self.state.stop_reason or "early_stopping"
        if self.config.min_eval_reward is not None and reward >= self.config.min_eval_reward:
            self.state.stop_training = True
            self.state.stop_reason = self.state.stop_reason or "eval_reward_reached"


def _make_evaluation_callback(
    BaseCallback: Any,
    *,
    config: TrainingConfig,
    telemetry: Any,
    run_control: RunControl | None,
    pipeline: Any,
    profiler: Any,
    device: str,
    checkpoints: Path,
    evaluations: Path,
    best_path: Path,
    best_record_path: Path,
    selection_rule: Any,
    inference_scheduler: InferenceDeviceScheduler | None,
    state: _SelectionState,
) -> Any:
    """Periodic evaluation, the checkpoint battery and best-checkpoint selection."""
    driver = _EvaluationDriver(
        config=config,
        telemetry=telemetry,
        run_control=run_control,
        pipeline=pipeline,
        profiler=profiler,
        device=device,
        checkpoints=checkpoints,
        evaluations=evaluations,
        best_path=best_path,
        best_record_path=best_record_path,
        selection_rule=selection_rule,
        inference_scheduler=inference_scheduler,
        state=state,
    )

    class EvaluationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.driver = driver

        def close(self) -> None:
            driver.close()

        def _on_training_start(self) -> None:
            driver.on_training_start(self.num_timesteps)

        def _on_step(self) -> bool:
            return driver.on_step(self.model, self.num_timesteps)

    return EvaluationCallback()


def _make_checkpoint_callback(
    CheckpointCallback: Any,
    *,
    save_freq: int,
    save_path: str,
    profiler: Any,
    inference_scheduler: InferenceDeviceScheduler | None,
) -> Any:
    """SB3's periodic checkpointer, with save timing and device pinning."""

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

    return TimedCheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def kl_adaptive_learning_rate(
    current_lr: float,
    approx_kl: float | None,
    base_lr: float,
    *,
    target_kl: float = 0.015,
) -> float:
    """Compute a KL-bounded adaptive learning rate around ``base_lr``."""
    base = max(1e-6, float(base_lr))
    cur = max(1e-6, float(current_lr))
    if approx_kl is None or not math.isfinite(approx_kl) or approx_kl < 0.0:
        return cur
    if approx_kl > target_kl * 2.0:
        return max(base * 0.2, cur / 1.25)
    if 0.0 < approx_kl < target_kl * 0.5:
        return min(base * 1.5, cur * 1.10)
    return cur


def _ppo_stats_payload(
    values: dict[str, Any], keys: tuple[str, ...], timesteps: int
) -> dict[str, Any]:
    """Copy Stable-Baselines3's logged diagnostics into a telemetry row."""
    payload: dict[str, Any] = {"event": "ppo_update", "timesteps": timesteps}
    for key in keys:
        if key in values:
            payload[key.split("/", 1)[1]] = float(values[key])
    if "entropy_loss" in payload:
        # SB3 logs the negated mean policy entropy; the sign flip here is
        # traceable to the same measured optimizer value.
        payload["entropy"] = -payload["entropy_loss"]
    return payload


def _adapt_ppo_learning_rate(
    model: Any, payload: dict[str, Any], base_lr: float | None
) -> float | None:
    """Apply the measured-KL learning-rate guard and return its base rate."""
    current_lr = payload.get("learning_rate")
    if isinstance(current_lr, (int, float)) and base_lr is None:
        base_lr = float(current_lr)
    if base_lr is None or "approx_kl" not in payload:
        return base_lr

    next_lr = kl_adaptive_learning_rate(
        float(current_lr if isinstance(current_lr, (int, float)) else base_lr),
        float(payload["approx_kl"]),
        base_lr,
    )
    model.learning_rate = next_lr
    model.lr_schedule = lambda _progress, lr=next_lr: lr
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    for group in getattr(optimizer, "param_groups", ()) or ():
        if isinstance(group, dict):
            group["lr"] = next_lr
    payload["learning_rate"] = round(next_lr, 7)
    return base_lr


def _publish_ppo_stats(
    telemetry: Any, run_control: RunControl | None, payload: dict[str, Any]
) -> None:
    """Write one optimizer row to the JSONL stream and optional live status."""
    telemetry.write(payload)
    if run_control is not None:
        run_control.update(
            **{
                f"ppo_{key}": value
                for key, value in payload.items()
                if key not in ("event", "timesteps")
            }
        )


def _make_ppo_stats_callback(
    BaseCallback: Any, *, telemetry: Any, run_control: RunControl | None
) -> Any:
    class PPOStatsCallback(BaseCallback):
        """Relays SB3's own PPO optimizer diagnostics into real telemetry.

        Every ``train()`` call already computes explained variance,
        approximate KL, clip fraction and entropy loss and stores them in
        ``model.logger.name_to_value`` - but nothing outside a TensorBoard
        event file ever sees them. This callback reads that same
        dictionary (never recomputes it) at the first opportunity after a
        ``train()`` call has finished - ``_on_rollout_start`` of the next
        iteration, or ``_on_training_end`` for the final update - and
        writes it to the same bounded ``training.jsonl`` the rest of the
        Control Center already tails. ``train/n_updates`` is SB3's own
        optimizer step counter, so "PPO updates completed" is an exact
        count, never an estimate.
        """

        _KEYS: tuple[str, ...] = (
            "train/n_updates",
            "train/loss",
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/clip_range",
            "train/explained_variance",
            "train/learning_rate",
        )

        def __init__(self) -> None:
            super().__init__()
            self._last_n_updates: float | None = None
            self._base_lr: float | None = None

        def _flush(self) -> None:
            values = getattr(self.model.logger, "name_to_value", None)
            if not values:
                return
            n_updates = values.get("train/n_updates")
            if n_updates is not None and n_updates == self._last_n_updates:
                return  # train() has not run again since the last flush.
            payload = _ppo_stats_payload(values, self._KEYS, self.num_timesteps)
            self._base_lr = _adapt_ppo_learning_rate(self.model, payload, self._base_lr)
            _publish_ppo_stats(telemetry, run_control, payload)
            self._last_n_updates = n_updates

        def _on_rollout_start(self) -> None:
            self._flush()

        def _on_training_end(self) -> None:
            self._flush()

        def _on_step(self) -> bool:
            return True

    return PPOStatsCallback()


def _make_profiling_callback(BaseCallback: Any, *, profiler: Any) -> Any:
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
            profiler.add_iteration(self.iteration, rollout_seconds, update_seconds, timesteps)
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

    return ProfilingCallback()


def _make_inference_device_callback(BaseCallback: Any) -> Any:
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

    return InferenceDeviceCallback


def _resolve_run_directory(config: TrainingConfig, checkpoint_path: Path | None) -> Path:
    """Where this run writes its artifacts.

    Resume artifacts stay inside the original run directory: a checkpoint
    inside ``<run>/checkpoints/`` maps to ``<run>``; one stored directly in
    the run root (e.g. ``final.zip``) maps to the run root itself, never to
    the run's parent, which would scatter checkpoints and logs elsewhere.
    """
    if checkpoint_path is None:
        return config.run_directory()
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _prepare_run_layout(run_dir: Path) -> tuple[Path, Path, Path]:
    """Creates and returns the (checkpoints, logs, evaluations) directories."""
    directories = tuple(run_dir / name for name in ("checkpoints", "logs", "evaluations"))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories  # type: ignore[return-value]


def _record_plan_metadata(
    profiler: TrainingProfiler,
    config: TrainingConfig,
    *,
    device: str,
    rollout_length: int,
    torch_threads: int,
) -> None:
    """Profiler metadata known before SB3 exists: the run as it was planned."""
    schedule = config.rollout_schedule()
    profiler.set_metadata(
        device=device,
        environment_count=config.environment_count,
        enemy_count=config.enemy_count,
        rollout_length=rollout_length,
        rollout_length_requested=config.rollout_length,
        rollout_auto_sized=config.rollout_length == 0,
        rollout_batch_size=rollout_length * config.environment_count,
        minibatch_size=config.batch_size,
        ppo_epochs=config.ppo_epochs,
        requested_timesteps=config.total_training_steps,
        scheduled_timesteps=schedule["scheduled_timesteps"],
        expected_updates=schedule["expected_updates"],
        expected_timestep_overshoot=schedule["overshoot_timesteps"],
        torch_threads=torch_threads,
        compact_training_infos=config.compact_training_infos,
    )


def _record_model_metadata(profiler: TrainingProfiler, config: TrainingConfig, model: Any) -> None:
    """Profiler metadata read back off the constructed model: the run as it is.

    SB3 may round or override what the config asked for, so these values -
    not the planned ones - are what the profile report compares against.
    """
    rollout_samples = int(model.n_steps) * int(model.n_envs)
    actual_updates = (config.total_training_steps + rollout_samples - 1) // rollout_samples
    profiler.set_metadata(
        ppo_epochs=int(model.n_epochs),
        rollout_length=int(model.n_steps),
        rollout_batch_size=rollout_samples,
        expected_updates=actual_updates,
        scheduled_timesteps=actual_updates * rollout_samples,
        expected_timestep_overshoot=(
            actual_updates * rollout_samples - config.total_training_steps
        ),
        minibatch_size=int(model.batch_size),
        minibatches_per_update=(
            int(model.n_epochs)
            * ((rollout_samples + int(model.batch_size) - 1) // int(model.batch_size))
        ),
    )


def _inherit_best_score(
    best_record_path: Path, selection_rule: Any, state: _SelectionState, telemetry: Any
) -> None:
    """Carries the previous run's best score over, but only if it is comparable.

    A score produced by a different metric or a different direction would
    compare two unrelated quantities, so a changed rule restarts selection
    from the rule's initial score - and records that it did.
    """
    if not best_record_path.exists():
        return
    previous_best = json.loads(best_record_path.read_text(encoding="utf-8-sig"))
    if not selection_rule.matches(previous_best.get("selection_rule")):
        telemetry.write(
            {
                "event": "checkpoint_selection_rule_changed",
                "previous": previous_best.get("selection_rule"),
                "current": selection_rule.as_dict(),
            }
        )
        return
    # "score" is the rule's own quantity; "mean_reward" is the legacy
    # field, which under the default rule is the same number.
    inherited = previous_best.get("score", previous_best.get("mean_reward"))
    if isinstance(inherited, (int, float)) and not isinstance(inherited, bool):
        state.best_score = float(inherited)


def _build_model(
    algorithm_class: Any,
    config: TrainingConfig,
    *,
    env: Any,
    device: str,
    rollout_length: int,
    logs: Path,
    run_dir: Path,
    checkpoint_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    """Loads or constructs the PPO model. Returns (model, warm-start record).

    Only a fresh model can be warm-started from a behavior-cloning
    checkpoint; a resumed run already carries those weights and writing a
    new warm_start.json would overwrite the original run's provenance.
    """
    if checkpoint_path:
        model = algorithm_class.load(str(checkpoint_path), env=env, device=device)
        model.set_env(env)
        model.learning_rate = float(config.learning_rate)
        model.lr_schedule = lambda _progress: float(config.learning_rate)
        model.batch_size = int(config.batch_size)
        model.n_epochs = int(config.ppo_epochs)
        model.ent_coef = float(config.entropy_coefficient)
        optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        for group in getattr(optimizer, "param_groups", ()) or ():
            if isinstance(group, dict):
                group["lr"] = float(config.learning_rate)
        return model, {"transferred": False, "source": "resume checkpoint"}

    model = algorithm_class(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=rollout_length,
        batch_size=config.batch_size,
        n_epochs=config.ppo_epochs,
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
    warm_start: dict[str, Any] = {"transferred": False}
    if config.bc_checkpoint:
        from .bc import load_bc_into_sb3_policy

        warm_start = load_bc_into_sb3_policy(model.policy, config.bc_checkpoint, device=device)
        warm_start["source"] = config.bc_checkpoint
    (run_dir / "warm_start.json").write_text(
        json.dumps(warm_start, indent=2) + "\n", encoding="utf-8"
    )
    return model, warm_start


def _attach_pipeline(
    config: TrainingConfig,
    *,
    run_dir: Path,
    telemetry: Any,
    device: str,
    env: Any,
    checkpoints: Path,
    resuming: bool,
) -> Any:
    """Installs the research pipeline for `curriculum_mode == "auto"`.

    Returns None in "fixed" mode, which keeps the historical single-level
    behavior exactly: no hooks are installed and PPO drives the env as it
    always has.
    """
    if config.curriculum_mode != "auto":
        return None
    from .pipeline import TrainingPipeline, write_manifest

    pipeline = TrainingPipeline(config, run_dir, telemetry, device=device)
    # Attach BEFORE any rollout starts: hook changes mid-training would be
    # a race (rollout collection runs in this thread, so sequencing here is
    # strictly happens-before every step).
    pipeline.attach(env)
    # Resume continues the curriculum exactly where the saved checkpoint
    # left it: same stage, same staged plans, same per-environment seed
    # ordinals.
    if resuming and pipeline.load_state(checkpoints / "curriculum_state.json"):
        pipeline.reattach_after_load()
        telemetry.write(
            {
                "event": "curriculum_resumed",
                "level": pipeline.driver.level,
                "episodes_completed": pipeline.driver.episodes_completed,
            }
        )
    write_manifest(run_dir, pipeline.manifest())
    return pipeline


def train_ppo(
    config: TrainingConfig,
    resume_checkpoint: str | Path | None = None,
    run_control: RunControl | None = None,
) -> dict[str, Any]:
    config.validate()
    PPO, BaseCallback, CallbackList, CheckpointCallback = _require_sb3()
    device = config.resolved_device()
    env_workers = config.resolved_env_workers()
    rollout_length = config.resolved_rollout_length()
    torch_threads = config.resolved_torch_threads()
    checkpoint_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    # Never inherit an unbounded host-wide Torch pool for this small MLP.
    # The resolved value is persisted below, so resource scheduling is as
    # reproducible as the model/optimizer configuration.
    import torch  # type: ignore

    torch.set_num_threads(torch_threads)
    run_dir = _resolve_run_directory(config, checkpoint_path)
    checkpoints, logs, evaluations = _prepare_run_layout(run_dir)
    config.save(run_dir / "config.json")

    profiler = TrainingProfiler() if config.profile_training else None
    algorithm_class = PPO
    if profiler is not None:
        algorithm_class = _make_profiled_ppo(PPO, profiler)
        _record_plan_metadata(
            profiler,
            config,
            device=device,
            rollout_length=rollout_length,
            torch_threads=torch_threads,
        )
    env = GodotVecEnv(**_env_kwargs(config, profiler=profiler, worker_count=env_workers))
    if profiler is not None:
        profiler.set_metadata(
            observation_floats=env.client.observation_dim,
            action_components=len(env.action_space.nvec),  # type: ignore[attr-defined]
            env_workers=env_workers,
        )
    resource_monitor = ResourceMonitor()
    telemetry = JsonlTelemetry(logs / "training.jsonl")
    # Integrated research pipeline (curriculum plans at episode boundaries,
    # skill metrics, replays, per-condition tracking, checkpoint battery).
    # "fixed" mode keeps the historical single-level behavior exactly: no
    # hooks are installed and PPO drives the env as it always has.
    pipeline = _attach_pipeline(
        config,
        run_dir=run_dir,
        telemetry=telemetry,
        device=device,
        env=env,
        checkpoints=checkpoints,
        resuming=checkpoint_path is not None,
    )
    # Checkpoint selection is an explicit, recorded rule (see
    # sandboxai/selection.py). The default is identical to the historical
    # behavior: strictly higher mean shaped episode reward.
    selection_rule = config.checkpoint_selection_rule()
    best_path = checkpoints / "best_eval.zip"
    # Mutable state shared between the metrics and evaluation callbacks.
    # It used to be three `nonlocal` bindings into train_ppo, which is what
    # forced the callbacks to be defined inside it; an explicit object lets
    # them live at module scope and be constructed by the factories below.
    state = _SelectionState(best_score=selection_rule.initial_score())

    best_record_path = evaluations / "best.json"
    _inherit_best_score(best_record_path, selection_rule, state, telemetry)

    # Built before the callbacks because two of them hold on to it. The
    # scheduler is inert until a model is attached (`model=None` here), so
    # constructing it early has no effect on a single-device run.
    inference_scheduler: InferenceDeviceScheduler | None = None
    if config.inference_device != "auto":
        inference_scheduler = InferenceDeviceScheduler(
            model=None, training_device=device, inference_device=config.resolved_inference_device()
        )

    checkpoint_callback = _make_checkpoint_callback(
        CheckpointCallback,
        save_freq=max(1, config.checkpoint_frequency // config.environment_count),
        save_path=str(checkpoints),
        profiler=profiler,
        inference_scheduler=inference_scheduler,
    )
    metrics_callback = _make_metrics_callback(
        BaseCallback,
        config=config,
        telemetry=telemetry,
        run_control=run_control,
        pipeline=pipeline,
        resource_monitor=resource_monitor,
        profiler=profiler,
        device=device,
        env_workers=env_workers,
        torch_threads=torch_threads,
        checkpoints=checkpoints,
        selection_rule=selection_rule,
        state=state,
    )
    evaluation_callback = _make_evaluation_callback(
        BaseCallback,
        config=config,
        telemetry=telemetry,
        run_control=run_control,
        pipeline=pipeline,
        profiler=profiler,
        device=device,
        checkpoints=checkpoints,
        evaluations=evaluations,
        best_path=best_path,
        best_record_path=best_record_path,
        selection_rule=selection_rule,
        inference_scheduler=inference_scheduler,
        state=state,
    )
    callback_items: list[Any] = [
        checkpoint_callback,
        metrics_callback,
        evaluation_callback,
        _make_ppo_stats_callback(BaseCallback, telemetry=telemetry, run_control=run_control),
    ]
    if profiler is not None:
        callback_items.insert(0, _make_profiling_callback(BaseCallback, profiler=profiler))
    if inference_scheduler is not None:
        callback_items.insert(0, _make_inference_device_callback(BaseCallback)(inference_scheduler))
    callbacks = CallbackList(callback_items)
    try:
        model, warm_start = _build_model(
            algorithm_class,
            config,
            env=env,
            device=device,
            rollout_length=rollout_length,
            logs=logs,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
        )
        if profiler is not None:
            _record_model_metadata(profiler, config, model)
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
            "training_steps_completed": max(
                0, model.num_timesteps - metrics_callback.start_timesteps
            ),
            "stopped": bool(run_control is not None and run_control.stop_requested),
            "stop_reason": state.stop_reason or "completed",
            "max_train_minutes": config.max_train_minutes,
            "elapsed_minutes": round(metrics_callback.budget.elapsed_seconds() / 60.0, 3),
            "warm_start": warm_start,
            "training_profile": str(profile_path) if profile_path is not None else None,
        }
        if pipeline is not None:
            from .pipeline import write_manifest

            pipeline.timesteps = model.num_timesteps
            pipeline.save_state()
            # The manifest reflects the final curriculum state so a resume
            # (or an outside analysis) starts from the true end state.
            write_manifest(
                run_dir,
                pipeline.manifest(
                    "stopped" if state.stop_reason in {"operator", "time_budget"} else "completed"
                ),
            )
            result["curriculum"] = pipeline.driver.curriculum_snapshot()
            result["manifest"] = str(run_dir / "run_manifest.json")
        (run_dir / "run_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
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
