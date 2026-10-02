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
import time
from collections.abc import Callable
from pathlib import Path
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


def _emit_step_progress(
    on_step_progress: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    environment_count: int,
    worker_count: int,
    completed_steps: int,
    target_steps: int,
    elapsed_seconds: float,
    max_seconds_per_config: float,
    step_latencies: list[float],
    episode_count: int,
    startup_seconds: float,
    warmup_seconds: float,
    include_resources: bool = False,
) -> None:
    if on_step_progress is None:
        return
    total_steps = environment_count * completed_steps
    sps = (total_steps / elapsed_seconds) if elapsed_seconds > 0.0 and completed_steps > 0 else None
    eps = (
        (episode_count / elapsed_seconds) if elapsed_seconds > 0.0 and completed_steps > 0 else None
    )
    p50_ms = percentile(step_latencies, 0.50) * 1000.0 if step_latencies else None
    p95_ms = percentile(step_latencies, 0.95) * 1000.0 if step_latencies else None
    jitter = (p95_ms / p50_ms) if (p50_ms is not None and p95_ms is not None and p50_ms > 0.0) else None
    payload: dict[str, Any] = {
        "phase": phase,
        "environments": environment_count,
        "workers": worker_count,
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "total_steps": total_steps,
        "elapsed_seconds": elapsed_seconds,
        "max_seconds_per_config": max_seconds_per_config,
        "steps_per_second": sps,
        "episodes": episode_count,
        "episodes_per_second": eps,
        "vector_step_latency_p50_ms": p50_ms,
        "vector_step_latency_p95_ms": p95_ms,
        "latency_jitter": round(jitter, 3) if jitter is not None else None,
        "startup_seconds": startup_seconds,
        "warmup_seconds": warmup_seconds,
    }
    if include_resources:
        payload["resources"] = resource_snapshot()
    on_step_progress(payload)


def _measure_single_config(
    *,
    project_path: str | Path,
    godot_executable: str,
    environment_count: int,
    worker_count: int,
    steps: int,
    enemy_count: int,
    seed: int,
    curriculum_level: int,
    max_seconds_per_config: float,
    compact_infos: bool,
    warmup_steps: int,
    on_step_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    startup_started = time.perf_counter()
    _emit_step_progress(
        on_step_progress,
        phase="startup",
        environment_count=environment_count,
        worker_count=worker_count,
        completed_steps=0,
        target_steps=steps,
        elapsed_seconds=0.0,
        max_seconds_per_config=max_seconds_per_config,
        step_latencies=[],
        episode_count=0,
        startup_seconds=0.0,
        warmup_seconds=0.0,
    )
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
        startup_seconds = time.perf_counter() - startup_started
        # MultiDiscrete idle action: all neutral axes, no shooting, no
        # jump. Built from ACTION_NVEC rather than a literal so the
        # benchmark cannot drift away from the action contract.
        idle_action = [nvec // 2 if nvec == 3 else 0 for nvec in ACTION_NVEC]
        actions = [list(idle_action) for _ in range(environment_count)]
        warmup_seconds = 0.0
        if warmup_steps:
            _emit_step_progress(
                on_step_progress,
                phase="warmup",
                environment_count=environment_count,
                worker_count=worker_count,
                completed_steps=0,
                target_steps=steps,
                elapsed_seconds=0.0,
                max_seconds_per_config=max_seconds_per_config,
                step_latencies=[],
                episode_count=0,
                startup_seconds=startup_seconds,
                warmup_seconds=0.0,
            )
            warmup_started = time.perf_counter()
            for _ in range(warmup_steps):
                client.step(actions)
            warmup_seconds = time.perf_counter() - warmup_started
        started = time.perf_counter()
        deadline = started + max_seconds_per_config
        last_progress_emit = started - 1.0
        episode_count = 0
        completed_steps = 0
        step_latencies: list[float] = []
        for _ in range(steps):
            step_started = time.perf_counter()
            _observations, _rewards, dones, _infos = client.step(actions)
            now = time.perf_counter()
            step_latencies.append(now - step_started)
            episode_count += int(dones.sum())
            completed_steps += 1
            if on_step_progress is not None and (now - last_progress_emit >= 0.15 or now >= deadline):
                last_progress_emit = now
                _emit_step_progress(
                    on_step_progress,
                    phase="stepping",
                    environment_count=environment_count,
                    worker_count=worker_count,
                    completed_steps=completed_steps,
                    target_steps=steps,
                    elapsed_seconds=max(now - started, 1e-9),
                    max_seconds_per_config=max_seconds_per_config,
                    step_latencies=step_latencies,
                    episode_count=episode_count,
                    startup_seconds=startup_seconds,
                    warmup_seconds=warmup_seconds,
                    include_resources=True,
                )
            if now >= deadline:
                break
        elapsed = max(time.perf_counter() - started, 1e-9)
        resources = resource_snapshot()
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
            "resources": resources,
            "startup_seconds": startup_seconds,
            "warmup_steps": warmup_steps,
            "warmup_seconds": warmup_seconds,
        }
        return row
    finally:
        client.close()


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
    warmup_steps: int = 0,
    on_step_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Measure stepping throughput for one or more configurations.

    ``warmup_steps`` optionally runs that many vector steps before the
    timed measurement. Warmup covers one-off costs the trainer never sees
    on a hot loop (first-call code paths, allocator growth); its duration
    is reported as ``warmup_seconds`` and excluded from ``steps_per_second``.
    Startup is always reported separately as ``startup_seconds``: the time
    from starting to build the bridge client (process spawn, environment
    construction, protocol handshake) until the initial ``reset`` returns.
    """
    if steps < 1 or not environment_counts:
        raise ValueError("benchmark needs positive steps and at least one environment count")
    if max_seconds_per_config <= 0.0:
        raise ValueError("max_seconds_per_config must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not worker_counts or any(int(value) < 1 for value in worker_counts):
        raise ValueError("worker_counts must contain at least one positive value")
    results: list[dict[str, Any]] = []
    for environment_count in environment_counts:
        # A worker per environment is the maximum useful shard count;
        # duplicates (e.g. 4 and 8 workers for 4 environments) collapse to
        # one measured configuration instead of two identical ones.
        planned_workers = sorted(
            {min(int(value), int(environment_count)) for value in worker_counts}
        )
        for worker_count in planned_workers:
            results.append(
                _measure_single_config(
                    project_path=project_path,
                    godot_executable=godot_executable,
                    environment_count=environment_count,
                    worker_count=worker_count,
                    steps=steps,
                    enemy_count=enemy_count,
                    seed=seed,
                    curriculum_level=curriculum_level,
                    max_seconds_per_config=max_seconds_per_config,
                    compact_infos=compact_infos,
                    warmup_steps=warmup_steps,
                    on_step_progress=on_step_progress,
                )
            )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "benchmark.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
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
                "startup_seconds",
                "warmup_steps",
                "warmup_seconds",
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
        gain = (current["steps_per_second"] - previous["steps_per_second"]) / previous[
            "steps_per_second"
        ]
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
