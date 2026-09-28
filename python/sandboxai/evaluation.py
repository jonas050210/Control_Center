"""Weight-frozen evaluation and machine-readable summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from .godot_env import GodotGymEnv


METRIC_KEYS = (
    "episode_reward",
    "kills",
    "deaths",
    "damage_dealt",
    "damage_received",
    "survival_time",
    "accuracy",
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


def evaluate_model(
    model: Any,
    env_kwargs: dict[str, Any],
    episodes: int = 20,
    seed: int = 9001,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Runs `episodes` weight-frozen episodes and summarises them.

    When ``env_kwargs["environment_count"] > 1`` the episodes are collected
    from a vectorised Godot bridge instead of one at a time. Evaluation
    previously always constructed a single-environment ``GodotGymEnv``, so
    ``sandboxai evaluate --env-count N`` parsed the flag and then ignored
    it; a 200-episode evaluation took N times longer than it needed to.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    environment_count = int(env_kwargs.get("environment_count", 1) or 1)
    started = time.perf_counter()
    if environment_count > 1:
        rows = _evaluate_vectorized(model, env_kwargs, episodes, seed, environment_count)
    else:
        rows = _evaluate_serial(model, env_kwargs, episodes, seed)
    return _summarize(rows, started, output_dir, environment_count)


def _evaluate_serial(
    model: Any, env_kwargs: dict[str, Any], episodes: int, seed: int
) -> list[dict[str, Any]]:
    env = GodotGymEnv(**env_kwargs)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            observation, _reset_info = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            steps = 0
            last_info: dict[str, Any] = {}
            while not done:
                prediction = model.predict(observation, deterministic=True)
                action = prediction[0] if isinstance(prediction, tuple) else prediction
                if hasattr(action, "cpu"):
                    action = action.cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                last_info = info
                done = bool(terminated or truncated)
            metrics = dict(last_info.get("metrics", {}))
            metrics.setdefault("episode_reward", total_reward)
            metrics.setdefault("episode_length", steps)
            metrics["episode_index"] = episode_index
            metrics["seed"] = seed + episode_index
            rows.append(metrics)
    finally:
        env.close()
    return rows


def _evaluate_vectorized(
    model: Any,
    env_kwargs: dict[str, Any],
    episodes: int,
    seed: int,
    environment_count: int,
) -> list[dict[str, Any]]:
    """Collects `episodes` finished episodes from N parallel environments.

    The bridge auto-resets a finished sub-environment, so episodes are
    harvested from the `done` infos. Collection stops as soon as `episodes`
    episodes have completed; any extra episodes that finish on the same
    step are discarded so the requested count is exact.
    """
    from .godot_env import GodotVecEnv

    kwargs = dict(env_kwargs)
    kwargs["environment_count"] = environment_count
    kwargs.setdefault("seed", seed)
    env = GodotVecEnv(**kwargs)
    rows: list[dict[str, Any]] = []
    steps_per_env = [0] * environment_count
    rewards_per_env = [0.0] * environment_count
    try:
        observations = env.reset()
        while len(rows) < episodes:
            prediction = model.predict(observations, deterministic=True)
            actions = prediction[0] if isinstance(prediction, tuple) else prediction
            if hasattr(actions, "cpu"):
                actions = actions.cpu().numpy()
            observations, step_rewards, dones, infos = env.step(actions)
            for index in range(environment_count):
                steps_per_env[index] += 1
                rewards_per_env[index] += float(step_rewards[index])
                if not bool(dones[index]):
                    continue
                if len(rows) >= episodes:
                    break
                metrics = dict(infos[index].get("metrics", {}))
                metrics.setdefault("episode_reward", rewards_per_env[index])
                metrics.setdefault("episode_length", steps_per_env[index])
                metrics["episode_index"] = len(rows)
                metrics["environment_index"] = index
                metrics["seed"] = seed + index
                rows.append(metrics)
                steps_per_env[index] = 0
                rewards_per_env[index] = 0.0
    finally:
        env.close()
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
    summary["episodes_detail"] = rows
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        with (destination / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (destination / "summary.txt").write_text(format_summary(summary), encoding="utf-8")
    return summary


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
        f"mean shots fired/hit: {summary.get('mean_shots_fired', 0.0):.3f} / {summary.get('mean_shots_hit', 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"
