"""Headless Godot simulation throughput benchmark."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from .godot_env import GodotBatchClient
from .telemetry import resource_snapshot


def benchmark_simulation(
    project_path: str | Path,
    godot_executable: str = "godot",
    environment_counts: list[int] | tuple[int, ...] = (1, 4, 8, 16),
    steps: int = 10_000,
    enemy_count: int = 1,
    seed: int = 1234,
    curriculum_level: int = 3,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if steps < 1 or not environment_counts:
        raise ValueError("benchmark needs positive steps and at least one environment count")
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
            episode_count = 0
            for _ in range(steps):
                _observations, _rewards, dones, _infos = client.step(actions)
                episode_count += int(dones.sum())
            elapsed = max(time.perf_counter() - started, 1e-9)
            row = {
                "environments": environment_count,
                "steps_per_environment": steps,
                "total_steps": environment_count * steps,
                "elapsed_seconds": elapsed,
                "steps_per_second": environment_count * steps / elapsed,
                "episodes": episode_count,
                "episodes_per_second": episode_count / elapsed,
                "cpu_gpu": resource_snapshot(),
            }
            results.append(row)
        finally:
            client.close()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        with (destination / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["environments", "steps_per_environment", "total_steps", "elapsed_seconds", "steps_per_second", "episodes", "episodes_per_second"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in results)
    return results
