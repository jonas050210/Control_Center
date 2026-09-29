"""Weight-frozen evaluation and machine-readable summaries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from .action_audit import (
    EpisodeActionAudit,
    _raw_component_probabilities,
    component_probabilities,
    shoot_probability_at,
    summarize_action_pipeline,
)
from .godot_env import GodotGymEnv


METRIC_KEYS = (
    "episode_reward",
    "kills",
    "deaths",
    "damage_dealt",
    "damage_received",
    "survival_time",
    "accuracy",
    # Trigger pulls count shoot=1 requests that reached the engine, including
    # cooldown/reload-blocked pulls. Comparing them with the policy-side
    # counters below localizes a zero-shot result before changing rewards.
    "trigger_pulls",
    "shots_fired",
    "shots_hit",
    "win",
    "loss",
    # Timeouts are now reported separately instead of being folded into
    # `loss` (see EpisodeState.to_metrics), so win_rate + loss_rate +
    # timeout_rate == 1.0 and a policy that merely stops dying is no longer
    # indistinguishable from one that is actually losing.
    "truncated",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class SynchronizedModel:
    """Serializes access to a weight-frozen model shared by eval workers.

    Normal and checkpoint-battery simulations can run in separate Godot
    processes at the same time, but both call the same in-memory SB3 model.
    Read-only PyTorch inference is thread-safe; serializing the very short
    ``predict`` calls is stricter still and prevents an implementation detail
    such as ``set_training_mode(False)`` from racing. Expensive environment
    steps remain concurrent. ``save`` uses the same lock because checkpoint
    evaluation also serializes the frozen weights.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = threading.Lock()

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.predict(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.save(*args, **kwargs)

    def component_probabilities(self, observations: Any) -> list[Any] | None:
        with self._lock:
            return _raw_component_probabilities(self._model, observations)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def run_parallel_evaluations(
    normal_job: Callable[[], Any],
    battery_job: Callable[[], Any],
) -> tuple[Any, Any, dict[str, float]]:
    """Runs the two independent evaluation roles concurrently.

    Both jobs are joined before the caller performs checkpoint selection or
    early stopping, preserving the same strict boundary ordering as the
    former sequential path. Return order is fixed (normal, battery), never
    completion order. Timing metadata makes the overlap visible in training
    profiles and supports a deterministic regression benchmark.
    """

    def timed(job: Callable[[], Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        value = job()
        return value, time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sandboxai-eval") as pool:
        normal_future = pool.submit(timed, normal_job)
        battery_future = pool.submit(timed, battery_job)
        normal, normal_seconds = normal_future.result()
        battery, battery_seconds = battery_future.result()
    wall_seconds = time.perf_counter() - started
    return normal, battery, {
        "normal_seconds": normal_seconds,
        "battery_seconds": battery_seconds,
        "wall_seconds": wall_seconds,
        "overlap_seconds": max(0.0, normal_seconds + battery_seconds - wall_seconds),
    }


def evaluate_model(
    model: Any,
    env_kwargs: dict[str, Any],
    episodes: int = 20,
    seed: int = 9001,
    output_dir: str | Path | None = None,
    env: Any = None,
    profiler: Any = None,
) -> dict[str, Any]:
    """Runs `episodes` weight-frozen episodes and summarises them.

    When ``env_kwargs["environment_count"] > 1`` the episodes are collected
    from a vectorised Godot bridge instead of one at a time. Evaluation
    previously always constructed a single-environment ``GodotGymEnv``, so
    ``sandboxai evaluate --env-count N`` parsed the flag and then ignored
    it; a 200-episode evaluation took N times longer than it needed to.

    ``env`` (optional) reuses an already-running evaluation environment
    across calls instead of spawning a fresh Godot bridge process per
    call. This is exact, not approximate: every episode is (re)started
    through an explicit seeded reset (directly in serial mode, through a
    complete staged plan in vector mode), and the engine derives the whole
    world deterministically from that seed
    (``EnvironmentCore.reset`` re-seeds the RNG and rebuilds every
    subsystem), so a reused process produces bit-identical episodes to a
    fresh one. The caller owns the env lifecycle when it passes one: it is
    NOT closed here.

    ``profiler`` (optional) adds ``eval.normal.*`` buckets to the training
    profile: total wall time, per-step policy prediction and environment
    stepping, plus episode/step counters.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    requested_environment_count = int(env_kwargs.get("environment_count", 1) or 1)
    if env is None:
        environment_count = min(episodes, requested_environment_count)
    else:
        environment_count = int(
            getattr(env, "environment_count", getattr(env, "num_envs", requested_environment_count))
        )
    started = time.perf_counter()
    if environment_count > 1:
        rows = _evaluate_vectorized(model, env_kwargs, episodes, seed, environment_count, env, profiler)
    else:
        rows = _evaluate_serial(model, env_kwargs, episodes, seed, env, profiler)
    elapsed = time.perf_counter() - started
    if profiler is not None:
        profiler.record("eval.normal.total", elapsed)
        profiler.add("eval.normal.episodes", len(rows))
    return _summarize(rows, started, output_dir, environment_count)


def _evaluate_serial(
    model: Any,
    env_kwargs: dict[str, Any],
    episodes: int,
    seed: int,
    env: Any = None,
    profiler: Any = None,
) -> list[dict[str, Any]]:
    owned = env is None
    if owned:
        env = GodotGymEnv(**env_kwargs)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            observation, _reset_info = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            steps = 0
            last_info: dict[str, Any] = {}
            action_audit = EpisodeActionAudit()
            while not done:
                predict_started = time.perf_counter() if profiler is not None else 0.0
                prediction = model.predict(observation, deterministic=True)
                action = prediction[0] if isinstance(prediction, tuple) else prediction
                if hasattr(action, "cpu"):
                    action = action.cpu().numpy()
                # Sparse probability inspection is a second deterministic
                # forward pass. Sampling every 64 steps keeps its cost tiny,
                # and no RNG state changes because get_distribution does not
                # call sample(). Exact selected actions are counted each tick.
                probabilities = (
                    component_probabilities(model, observation) if steps % 64 == 0 else None
                )
                action_audit.record(action, shoot_probability_at(probabilities))
                if profiler is not None:
                    profiler.record("eval.normal.predict", time.perf_counter() - predict_started)
                    step_started = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                if profiler is not None:
                    profiler.record("eval.normal.env_step", time.perf_counter() - step_started)
                    profiler.add("eval.normal.env_steps", 1)
                total_reward += float(reward)
                steps += 1
                last_info = info
                done = bool(terminated or truncated)
            metrics = dict(last_info.get("metrics", {}))
            metrics.setdefault("episode_reward", total_reward)
            metrics.setdefault("episode_length", steps)
            metrics["episode_index"] = episode_index
            metrics["seed"] = seed + episode_index
            metrics.update(action_audit.summary())
            rows.append(metrics)
    finally:
        if owned:
            env.close()
    return rows


def _evaluate_vectorized(
    model: Any,
    env_kwargs: dict[str, Any],
    episodes: int,
    seed: int,
    environment_count: int,
    env: Any = None,
    profiler: Any = None,
) -> list[dict[str, Any]]:
    """Runs the serial evaluation episode set on N bridge environments.

    Every episode is explicitly plan-scheduled with the same ``seed + j``
    that :func:`_evaluate_serial` passes to ``reset``. Results are sorted by
    ``j`` after collection. Parallelism therefore changes neither coverage,
    seed flow nor evaluation ordering; it only batches deterministic policy
    inference and overlaps independent simulation slots.
    """
    from .checkpoint_eval import PlanExecutor, PlannedEpisode
    from .conditions import Condition
    from .godot_env import GodotVecEnv

    owned = env is None
    if owned:
        kwargs = dict(env_kwargs)
        kwargs["environment_count"] = environment_count
        kwargs.setdefault("seed", seed)
        env = GodotVecEnv(**kwargs)
    if profiler is not None:
        from .training_profile import PrefixedProfiler

        executor_profiler: Any = PrefixedProfiler(profiler, "eval.normal.")
    else:
        executor_profiler = None
    executor = PlanExecutor.from_client(
        env.client, skill_metrics=False, profiler=executor_profiler
    )
    plans = [
        PlannedEpisode(
            condition=Condition(
                map_id="",
                scenario="",
                lighting="",
                enemy_count=int(env_kwargs.get("enemy_count", 1)),
                level=int(env_kwargs.get("curriculum_level", 3)),
                seed=seed + episode_index,
            ),
            labels={"eval": "normal", "_position": episode_index},
        )
        for episode_index in range(episodes)
    ]
    try:
        planned_rows = executor.run(model, plans, policy_id="normal_evaluation")
    finally:
        if owned:
            env.close()

    rows: list[dict[str, Any]] = []
    for episode_index, planned in enumerate(planned_rows):
        row = {
            key: value
            for key, value in planned.items()
            if key not in ("labels", "condition", "skill", "seed")
        }
        row["episode_index"] = episode_index
        row["seed"] = seed + episode_index
        rows.append(row)
    return rows


def _summarize(
    rows: list[dict[str, Any]],
    started: float,
    output_dir: str | Path | None,
    environment_count: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "environment_count": environment_count,
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_per_second": len(rows) / max(time.perf_counter() - started, 1e-9),
    }
    for key in METRIC_KEYS:
        values = [float(row.get(key, 0.0)) for row in rows]
        summary[f"mean_{key}"] = _mean(values)
    summary["win_rate"] = summary.get("mean_win", 0.0)
    summary["loss_rate"] = summary.get("mean_loss", 0.0)
    summary["timeout_rate"] = summary.get("mean_truncated", 0.0)

    action_pipeline = summarize_action_pipeline(rows)
    summary["action_pipeline"] = action_pipeline
    summary["policy_shoot_request_rate"] = action_pipeline["policy_shoot_request_rate"]
    summary["mean_policy_shoot_probability"] = action_pipeline.get(
        "mean_stochastic_shoot_probability"
    )
    summary["episodes_detail"] = rows
    if output_dir is not None:
        write_evaluation_artifacts(summary, output_dir)
    return summary


def write_evaluation_artifacts(
    summary: dict[str, Any],
    output_dir: str | Path,
    episodes_name: str = "episodes.csv",
) -> None:
    """Writes a completed normal-evaluation summary without rerunning it.

    The training callback uses ``normal_episodes.csv`` because the checkpoint
    generalization exporter owns the historical ``episodes.csv`` in the same
    step directory. Keeping distinct files also removes a write race now that
    the two evaluations execute concurrently.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    rows = list(summary.get("episodes_detail", []))
    with (destination / episodes_name).open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (destination / "summary.txt").write_text(format_summary(summary), encoding="utf-8")


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "SandboxAI evaluation",
        "====================",
        f"episodes: {summary.get('episodes', 0)}",
        f"mean reward: {summary.get('mean_episode_reward', 0.0):.4f}",
        f"win/loss/timeout rate: {summary.get('win_rate', 0.0):.2%} / {summary.get('loss_rate', 0.0):.2%} / {summary.get('timeout_rate', 0.0):.2%}",
        f"mean kills/deaths: {summary.get('mean_kills', 0.0):.3f} / {summary.get('mean_deaths', 0.0):.3f}",
        f"mean damage dealt/received: {summary.get('mean_damage_dealt', 0.0):.3f} / {summary.get('mean_damage_received', 0.0):.3f}",
        f"mean survival time: {summary.get('mean_survival_time', 0.0):.3f}s",
        f"mean accuracy: {summary.get('mean_accuracy', 0.0):.2%}",
        f"mean trigger pulls / shots fired / hits: {summary.get('mean_trigger_pulls', 0.0):.3f} / {summary.get('mean_shots_fired', 0.0):.3f} / {summary.get('mean_shots_hit', 0.0):.3f}",
        f"policy shoot request rate: {summary.get('policy_shoot_request_rate', 0.0):.2%}",
        f"shoot path localization: {summary.get('action_pipeline', {}).get('localization', 'unavailable')}",
    ]
    mean_probability = summary.get("mean_policy_shoot_probability")
    if mean_probability is not None:
        lines.append(f"mean stochastic P(shoot=1): {float(mean_probability):.2%}")
    return "\n".join(lines) + "\n"
