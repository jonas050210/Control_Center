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
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = GodotGymEnv(**env_kwargs)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for episode_index in range(episodes):
            observation, _reset_info = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            steps = 0
            last_info: dict[str, Any] = {}
            while not done:
                action, _state = model.predict(observation, deterministic=True)
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
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_per_second": len(rows) / max(time.perf_counter() - started, 1e-9),
    }
    for key in METRIC_KEYS:
        values = [float(row.get(key, 0.0)) for row in rows]
        summary[f"mean_{key}"] = _mean(values)
    summary["win_rate"] = summary.get("mean_win", 0.0)
    summary["loss_rate"] = summary.get("mean_loss", 0.0)
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
        f"win/loss rate: {summary.get('win_rate', 0.0):.2%} / {summary.get('loss_rate', 0.0):.2%}",
        f"mean kills/deaths: {summary.get('mean_kills', 0.0):.3f} / {summary.get('mean_deaths', 0.0):.3f}",
        f"mean damage dealt/received: {summary.get('mean_damage_dealt', 0.0):.3f} / {summary.get('mean_damage_received', 0.0):.3f}",
        f"mean survival time: {summary.get('mean_survival_time', 0.0):.3f}s",
        f"mean accuracy: {summary.get('mean_accuracy', 0.0):.2%}",
        f"mean shots fired/hit: {summary.get('mean_shots_fired', 0.0):.3f} / {summary.get('mean_shots_hit', 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"
