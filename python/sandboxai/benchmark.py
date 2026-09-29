"""Headless Godot simulation throughput benchmark.

Measures how steps/second scales along the two axes that actually matter
for this architecture:

* **environments per process** — one ``rl_server.gd`` process steps its
  environments serially inside a single GDScript thread, so this axis
  amortises IPC and per-request overhead but never uses a second core.
* **worker processes** (``--env-workers`` / ``worker_counts``) — several
  bridge processes simulating concurrently, which is the only way to use
  more than one core (see :mod:`sandboxai.sharded_env`).

Both are swept here so the measured answer to "how many environments, in
how many processes?" comes from the target machine instead of a guess (see
docs/DEBUG_GUI_AND_BENCHMARKING.md for the bottleneck discussion). Each
configuration is time-boxed by `max_seconds_per_config` so sweeping many
configurations does not take an unbounded amount of time — higher
environment counts naturally need fewer steps to produce a stable
steps/second estimate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from .contract import ACTION_NVEC
from .godot_env import make_batch_client
from .telemetry import resource_snapshot

## Practical sweep recommended for a mid-range desktop (e.g. i7-12700F /
## RTX 4060 Ti 8GB / 32GB RAM). Kept as a module constant so the CLI and any
## script can share one canonical default instead of re-typing it.
DEFAULT_ENVIRONMENT_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64)

## Worker-process sweep. 1 is the historical single-process bridge and is
## always measured first so every multi-process number has a baseline in
## the same report.
DEFAULT_WORKER_COUNTS: tuple[int, ...] = (1,)


def percentile(values: list[float], quantile: float) -> float:
    """Deterministic linearly interpolated percentile (seconds)."""
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    worker_counts: list[int] | tuple[int, ...] = DEFAULT_WORKER_COUNTS,
    compact_infos: bool = True,
) -> list[dict[str, Any]]:
    if steps < 1 or not environment_counts:
        raise ValueError("benchmark needs positive steps and at least one environment count")
    if max_seconds_per_config <= 0.0:
        raise ValueError("max_seconds_per_config must be positive")
    if not worker_counts or any(int(value) < 1 for value in worker_counts):
        raise ValueError("worker_counts must contain at least one positive value")
    results: list[dict[str, Any]] = []
    for environment_count in environment_counts:
        # A worker per environment is the maximum useful shard count;
        # duplicates (e.g. 4 and 8 workers for 4 environments) collapse to
        # one measured configuration instead of two identical ones.
        planned_workers = sorted({min(int(value), int(environment_count)) for value in worker_counts})
        for worker_count in planned_workers:
            client = make_batch_client(
                project_path=project_path,
                godot_executable=godot_executable,
                environment_count=environment_count,
                enemy_count=enemy_count,
                seed=seed,
                curriculum_level=curriculum_level,
                worker_count=worker_count,
                # PPO uses compact non-terminal infos. Benchmark that real
                # wire path by default; full diagnostics remain available as
                # an explicit serialization-stress comparison.
                compact_infos=compact_infos,
            )
            try:
                client.reset(seed)
                # MultiDiscrete idle action: all neutral axes, no shooting, no
                # jump. Built from ACTION_NVEC rather than a literal so the
                # benchmark cannot drift away from the action contract.
                idle_action = [nvec // 2 if nvec == 3 else 0 for nvec in ACTION_NVEC]
                actions = [list(idle_action) for _ in range(environment_count)]
                started = time.perf_counter()
                deadline = started + max_seconds_per_config
                episode_count = 0
                completed_steps = 0
                step_latencies: list[float] = []
                for _ in range(steps):
                    step_started = time.perf_counter()
                    _observations, _rewards, dones, _infos = client.step(actions)
                    step_latencies.append(time.perf_counter() - step_started)
                    episode_count += int(dones.sum())
                    completed_steps += 1
                    if time.perf_counter() >= deadline:
                        break
                elapsed = max(time.perf_counter() - started, 1e-9)
                row = {
                    "environments": environment_count,
                    "workers": worker_count,
                    "environments_per_worker": environment_count / worker_count,
                    "steps_per_environment": completed_steps,
                    "total_steps": environment_count * completed_steps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": environment_count * completed_steps / elapsed,
                    "vector_step_latency_p50_ms": percentile(step_latencies, 0.50) * 1000.0,
                    "vector_step_latency_p95_ms": percentile(step_latencies, 0.95) * 1000.0,
                    "episodes": episode_count,
                    "episodes_per_second": episode_count / elapsed,
                    "time_boxed": completed_steps < steps,
                    "info_mode": "compact_training" if compact_infos else "full_diagnostics",
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
                "workers",
                "environments_per_worker",
                "steps_per_environment",
                "total_steps",
                "elapsed_seconds",
                "steps_per_second",
                "vector_step_latency_p50_ms",
                "vector_step_latency_p95_ms",
                "episodes",
                "episodes_per_second",
                "time_boxed",
                "info_mode",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in results)
    return results


def summarize_scaling(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Finds the environment count with the best steps/second and flags
    diminishing returns (a cheap, deterministic heuristic, not a substitute
    for reading the raw numbers).

    When the sweep contains several worker counts the environment-axis
    heuristic is computed on the single-process rows (the historical
    meaning of "diminishing returns at N environments") and the worker
    axis is summarised separately as measured speed-up over the
    single-process row with the same environment count."""
    if not results:
        return {}
    best = max(results, key=lambda row: row["steps_per_second"])
    single_worker = [row for row in results if int(row.get("workers", 1)) == 1]
    sorted_rows = sorted(single_worker or results, key=lambda row: row["environments"])
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
    summary: dict[str, Any] = {
        "best_environment_count": best["environments"],
        "best_steps_per_second": best["steps_per_second"],
        "diminishing_returns_at_environment_count": diminishing_returns_at,
    }
    worker_values = {int(row.get("workers", 1)) for row in results}
    if len(worker_values) > 1:
        baseline = {
            int(row["environments"]): float(row["steps_per_second"])
            for row in results
            if int(row.get("workers", 1)) == 1
        }
        speedups: list[dict[str, Any]] = []
        for row in sorted(
            (row for row in results if int(row.get("workers", 1)) > 1),
            key=lambda item: (item["environments"], item["workers"]),
        ):
            reference = baseline.get(int(row["environments"]))
            speedups.append(
                {
                    "environments": int(row["environments"]),
                    "workers": int(row.get("workers", 1)),
                    "steps_per_second": float(row["steps_per_second"]),
                    # None rather than 1.0 when the sweep has no
                    # single-process row to compare against: an invented
                    # baseline would be a fabricated measurement.
                    "speedup_vs_single_process": (
                        float(row["steps_per_second"]) / reference
                        if reference and reference > 0.0
                        else None
                    ),
                }
            )
        summary["best_worker_count"] = int(best.get("workers", 1))
        summary["worker_scaling"] = speedups
    return summary
