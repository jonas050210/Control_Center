"""Headless Godot simulation throughput benchmark.

Measures how steps/second scales with the number of parallel environments
inside a single Godot process, which is the practical scaling question for
this architecture (see docs/DEBUG_GUI_AND_BENCHMARKING.md for the full
bottleneck discussion). Each configuration is time-boxed by
`max_seconds_per_config` so sweeping many environment counts (as recommended:
1, 2, 4, 8, 16, 24, 32, 48, 64) does not take an unbounded amount of time —
higher environment counts naturally need fewer steps to produce a stable
steps/second estimate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from .godot_env import GodotBatchClient
from .telemetry import resource_snapshot

## Practical sweep recommended for a mid-range desktop (e.g. i7-12700F /
## RTX 4060 Ti 8GB / 32GB RAM). Kept as a module constant so the CLI and any
## script can share one canonical default instead of re-typing it.
DEFAULT_ENVIRONMENT_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def benchmark_simulation(
    project_path: str | Path,
    godot_executable: str = "godot",
    environment_counts: list[int] | tuple[int, ...] = DEFAULT_ENVIRONMENT_COUNTS,
    steps: int = 2_000,
    enemy_count: int = 1,
    seed: int = 1234,
    curriculum_level: int = 3,
    output_dir: str | Path | None = None,
    max_seconds_per_config: float = 20.0,
) -> list[dict[str, Any]]:
    if steps < 1 or not environment_counts:
        raise ValueError("benchmark needs positive steps and at least one environment count")
    if max_seconds_per_config <= 0.0:
        raise ValueError("max_seconds_per_config must be positive")
    results: list[dict[str, Any]] = []
    for environment_count in environment_counts:
        client = GodotBatchClient(
            project_path=project_path,
            godot_executable=godot_executable,
            environment_count=environment_count,
            enemy_count=enemy_count,
            seed=seed,
            curriculum_level=curriculum_level,
        )
        try:
            client.reset(seed)
            # MultiDiscrete idle action: all neutral axes, no shooting.
            actions = [[1, 1, 1, 1, 0] for _ in range(environment_count)]
            started = time.perf_counter()
            deadline = started + max_seconds_per_config
            episode_count = 0
            completed_steps = 0
            for _ in range(steps):
                _observations, _rewards, dones, _infos = client.step(actions)
                episode_count += int(dones.sum())
                completed_steps += 1
                if time.perf_counter() >= deadline:
                    break
            elapsed = max(time.perf_counter() - started, 1e-9)
            row = {
                "environments": environment_count,
                "steps_per_environment": completed_steps,
                "total_steps": environment_count * completed_steps,
                "elapsed_seconds": elapsed,
                "steps_per_second": environment_count * completed_steps / elapsed,
                "episodes": episode_count,
                "episodes_per_second": episode_count / elapsed,
                "time_boxed": completed_steps < steps,
                "resources": resource_snapshot(),
            }
            results.append(row)
        finally:
            client.close()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        with (destination / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "environments",
                "steps_per_environment",
                "total_steps",
                "elapsed_seconds",
                "steps_per_second",
                "episodes",
                "episodes_per_second",
                "time_boxed",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in results)
    return results


def summarize_scaling(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Finds the environment count with the best steps/second and flags
    diminishing returns (a cheap, deterministic heuristic, not a substitute
    for reading the raw numbers)."""
    if not results:
        return {}
    best = max(results, key=lambda row: row["steps_per_second"])
    sorted_rows = sorted(results, key=lambda row: row["environments"])
    diminishing_returns_at: int | None = None
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        if previous["steps_per_second"] <= 0:
            continue
        gain = (current["steps_per_second"] - previous["steps_per_second"]) / previous["steps_per_second"]
        env_growth = current["environments"] / max(previous["environments"], 1)
        # Scaling is "diminishing" once throughput grows much slower than
        # the environment count did (single-process/GIL/IPC overhead
        # dominating rather than more useful parallel simulation work).
        if gain < (env_growth - 1.0) * 0.5:
            diminishing_returns_at = current["environments"]
            break
    return {
        "best_environment_count": best["environments"],
        "best_steps_per_second": best["steps_per_second"],
        "diminishing_returns_at_environment_count": diminishing_returns_at,
    }
