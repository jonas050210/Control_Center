"""Staged, budget-aware runtime benchmark that recommends a configuration.

The Control Center needs an answer to "how many environments, in how many
worker processes, on which device should this machine train with?" — and
the answer must come from measurements on the host that will run the
training, never from a fixed guess such as ``48 environments / 4 workers``.

This module is an *orchestrator*, not a second measurement
implementation. Every number it reports was produced by one of the
existing, engine-backed measurers:

* the environment/worker screening stage times the real Godot bridge
  through :func:`sandboxai.benchmark.benchmark_simulation` (startup,
  warmup, throughput, latency percentiles, episodes, resources);
* the device stage reuses :mod:`sandboxai.hardware_profile` — the single
  device-comparison implementation — either by trusting an already
  persisted hardware profile or by measuring CPU/Hybrid/CUDA through its
  own ``measure_devices``;
* the finalist validation stage times a short real PPO training slice per
  surviving configuration through ``hardware_profile.default_measure``
  (which accepts ``env_workers`` for exactly this purpose).

Stages and budget
-----------------
The pipeline spends its budget in stages instead of running every
candidate for the whole time:

1. **discovery** — probe the runtime (Godot executable/version, torch,
   CUDA, CPU/RAM) without measuring anything.
2. **screening** — the cheap bridge benchmark across every planned
   ``(environments, workers)`` pair, each configuration time-capped so a
   large sweep stays bounded.
3. **devices** — only when more than one device candidate exists, no
   usable hardware profile is already persisted, and torch is available.
   Runs after screening so the slice sizes can be planned from measured
   throughput.
4. **validation** — a real training slice for the top few stable
   configurations, on the selected device.
5. **recommendation** — chosen from *validated* throughput when
   available, preferring stability (latency jitter, no errors) over an
   unstable peak, with the reasoning spelled out in ``rationale``.

Budget modes: **steps** (screen until a configured step count per
configuration) or **time** (approximately N minutes total, split across
the stages; the candidate grid is thinned to fit). Planning may *predict*
how long a slice will take, but only measured values are reported, and a
stage that could not run is recorded with a status and a reason instead
of a fabricated number.

Persistence
-----------
:func:`write_report` stores the full report (``pipeline.json``) plus the
screening rows in the ``benchmark.json`` shape the existing benchmark
history readers already understand. The recommendation itself is persisted
separately (``.sandboxai/recommended_config.json``, machine-local like the
hardware profile) so the Control Center can offer it again on the next
start.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import benchmark_simulation
from .sharded_env import recommended_worker_count, worker_compatibility

PIPELINE_FORMAT = "sandboxai.benchmark_pipeline/v1"
RECOMMENDATION_FORMAT = "sandboxai.recommended_config/v1"
PIPELINE_REPORT_NAME = "pipeline.json"

#: Default wall-clock budget for time mode: the middle of the 10-30 minute
#: range the Control Center offers.
DEFAULT_TIME_BUDGET_MINUTES = 15.0
MIN_TIME_BUDGET_MINUTES = 1.0
MAX_TIME_BUDGET_MINUTES = 60.0

#: Steps mode: per-configuration screening target (per environment).
DEFAULT_SCREEN_STEPS = 2_000
#: Steps mode: safety wall-clock cap per screening configuration so one
#: pathological configuration cannot wedge a sweep the user asked to bound
#: by steps.
DEFAULT_SCREEN_CAP_SECONDS = 120.0
#: Training-slice steps for the device comparison (cheap, small config).
DEFAULT_DEVICE_STEPS = 1_500
#: Steps mode: training-slice steps per validated finalist.
DEFAULT_VALIDATION_STEPS = 3_000

#: How many screening survivors get a real training slice.
DEFAULT_FINALISTS = 4

#: A configuration within this fraction of the best measured throughput is
#: "near the best"; among those, stability and resource use decide.
NEAR_BEST_FRACTION = 0.10
#: p95/p50 vector-step latency ratio above which a configuration counts
#: as unstable (intermittent stalls dominating the median step).
JITTER_THRESHOLD = 4.0

#: Time-mode share of the budget spent screening; the rest goes to the
#: device comparison and finalist validation.
SCREEN_TIME_SHARE = 0.55
#: Bounds for the per-configuration screening cap in time mode.
MIN_SCREEN_SECONDS_PER_CONFIG = 3.0
MAX_SCREEN_SECONDS_PER_CONFIG = 30.0
#: Estimated per-configuration process startup/warmup overhead that is not
#: part of the measurement itself (planning only).
ESTIMATED_STARTUP_SECONDS_PER_CONFIG = 2.0
#: Bounds for the per-finalist validation slice in time mode.
MIN_VALIDATION_SECONDS = 5.0
MAX_VALIDATION_SECONDS = 180.0
#: Planning factor converting measured bridge throughput into an expected
#: training-slice step count (training adds policy inference and PPO
#: update work per step). Sizes the slice only; never reported as a
#: measurement.
TRAINING_SLOWDOWN_PLANNING_FACTOR = 0.4

CancelFn = Callable[[], bool]
ProgressFn = Callable[[dict[str, Any]], None]


def _importable(name: str) -> bool:
    try:
        from importlib.util import find_spec

        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Runtime discovery
# ---------------------------------------------------------------------------


def available_runtime(
    project_path: str | Path, godot_executable: str | None = None
) -> dict[str, Any]:
    """Probe what this machine can actually run. Measures nothing but versions.

    A missing Godot binary is reported as ``godot_available: False`` (and
    the pipeline then records an honest "unavailable" report), never
    silently substituted. CPU affinity is honoured so a container-limited
    run does not plan a grid for cores it cannot use.
    """
    from .config import find_godot_executable
    from .runtime_validation import RuntimeValidator

    raw = (
        godot_executable
        or os.environ.get("GODOT_EXECUTABLE")
        or os.environ.get("GODOT_PATH")
        or "godot"
    )
    runtime: dict[str, Any] = {
        "godot_executable": raw,
        "platform": _platform.platform(),
        "cpu_count": max(1, os.cpu_count() or 1),
    }
    try:
        runtime["cpu_count_available_to_process"] = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # Windows / restricted runtimes
        runtime["cpu_count_available_to_process"] = runtime["cpu_count"]
    try:
        validator = RuntimeValidator(project_path=project_path, godot_executable=raw)
        available = validator.is_godot_available()
        runtime["godot_available"] = available
        if available:
            resolved = find_godot_executable(raw)
            runtime["godot_resolved_executable"] = resolved
            runtime["godot_version"] = validator.probe_version(resolved)
        else:
            runtime["godot_resolved_executable"] = None
            runtime["godot_version"] = None
    except Exception as exc:  # pragma: no cover - defensive, never fatal
        runtime["godot_available"] = False
        runtime["godot_resolved_executable"] = None
        runtime["godot_version"] = None
        runtime["godot_error"] = str(exc)
    runtime["torch_available"] = _importable("torch")
    runtime["psutil_available"] = _importable("psutil")
    if runtime["torch_available"]:
        from .hardware_profile import cuda_available

        runtime["cuda_available"] = cuda_available()
    else:
        runtime["cuda_available"] = False
    runtime["host_ram_gb"] = None
    if runtime["psutil_available"]:
        try:
            import psutil  # type: ignore

            runtime["host_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except (ImportError, OSError):  # pragma: no cover - defensive
            pass
    return runtime


# ---------------------------------------------------------------------------
# Candidate planning
# ---------------------------------------------------------------------------


def default_environment_counts(cpu_count: int | None = None) -> tuple[int, ...]:
    """Environment ladder scaled to the host (capped at four times the cores).

    Not a recommendation — it is the sweep the pipeline will *measure*.
    The cap keeps a small machine from burning its budget on environment
    counts it cannot feed, while a big machine still probes up to 64.
    """
    logical = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    cap = max(8, min(64, logical * 4))
    ladder = (1, 2, 4, 8, 16, 24, 32, 48, 64)
    counts = [value for value in ladder if value <= cap]
    if cap not in counts:
        counts.append(cap)
    return tuple(sorted(counts))


def default_worker_counts(environment_count: int, cpu_count: int | None = None) -> tuple[int, ...]:
    """Worker ladder for one environment count: 1 plus powers of two up to
    the sharding module's own recommendation for this host."""
    max_workers = recommended_worker_count(environment_count, cpu_count)
    counts = {1}
    for power in (2, 4, 8, 16, 32):
        if power <= max_workers:
            counts.add(power)
    if max_workers > 1 and len(counts) == 1:
        counts.add(max_workers)
    return tuple(sorted(counts))


def plan_candidates(
    environment_counts: list[int] | tuple[int, ...] | None,
    worker_counts: list[int] | tuple[int, ...] | None,
    *,
    cpu_count: int | None = None,
) -> list[dict[str, Any]]:
    """The ``(environments, workers)`` grid the pipeline will measure.

    Worker counts above the environment count and duplicates (a clamped
    worker count equal to an earlier one) collapse, exactly like
    :func:`sandboxai.benchmark.benchmark_simulation` clamps them, so the
    plan matches what will actually be measured. Only compatibility-valid
    pairs are planned; anything the launcher would reject is not
    benchmarked either.
    """
    environments = (
        list(environment_counts)
        if environment_counts
        else list(default_environment_counts(cpu_count))
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for environment_count in sorted({int(value) for value in environments}):
        if environment_count < 1:
            continue
        if worker_counts is not None:
            requested = sorted({int(value) for value in worker_counts if value >= 1})
            planned = tuple(sorted({min(value, environment_count) for value in requested}))
        else:
            planned = default_worker_counts(environment_count, cpu_count)
        for worker_count in planned:
            compatibility = worker_compatibility(
                environment_count, worker_count, cpu_count=cpu_count
            )
            if not compatibility.valid:
                continue
            key = (environment_count, compatibility.resolved_workers)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "environments": environment_count,
                    "workers": compatibility.resolved_workers,
                    "warnings": list(compatibility.warnings),
                }
            )
    return candidates


def _thin(values: list[int], target: int) -> list[int]:
    """Evenly-spaced subset of ``values`` (endpoints kept), at most ``target``."""
    if len(values) <= target:
        return list(values)
    if target <= 1:
        return [values[-1]]
    picked = [values[0]]
    stride = (len(values) - 1) / (target - 1)
    for index in range(1, target - 1):
        picked.append(values[round(index * stride)])
    picked.append(values[-1])
    return sorted(set(picked))


def fit_candidates_to_budget(
    candidates: list[dict[str, Any]],
    screen_budget_seconds: float,
    *,
    min_seconds_per_config: float = MIN_SCREEN_SECONDS_PER_CONFIG,
    max_seconds_per_config: float = MAX_SCREEN_SECONDS_PER_CONFIG,
) -> tuple[list[dict[str, Any]], float]:
    """Thin the candidate grid until it fits the screening time budget.

    Returns the (possibly reduced) candidate list and the per-configuration
    measurement cap. Thinning keeps the spread of the sweep rather than its
    density: worker variants are dropped first (the single-process baseline
    and the largest worker count survive longest), then the environment
    ladder is decimated evenly with its endpoints preserved.
    """
    current = list(candidates)

    def cap_for(count: int) -> float:
        measurable = screen_budget_seconds - count * ESTIMATED_STARTUP_SECONDS_PER_CONFIG
        return max(
            min_seconds_per_config,
            min(max_seconds_per_config, measurable / max(count, 1)),
        )

    max_configs = max(
        1,
        int(
            screen_budget_seconds // (min_seconds_per_config + ESTIMATED_STARTUP_SECONDS_PER_CONFIG)
        ),
    )
    while len(current) > max_configs:
        environments = sorted({row["environments"] for row in current})
        worker_counts = sorted({row["workers"] for row in current})
        if len(worker_counts) > 2:
            reduced = _thin(worker_counts, len(worker_counts) - 1)
            thinned = [row for row in current if row["workers"] in reduced]
        elif len(environments) > 2:
            reduced = _thin(environments, len(environments) - 1)
            thinned = [row for row in current if row["environments"] in reduced]
        else:
            # Two environments x two worker variants: drop the smaller
            # environment's multi-worker variant, the least informative row.
            thinned = [
                row
                for row in current
                if row["workers"] == 1 or row["environments"] == environments[-1]
            ]
        if len(thinned) == len(current):
            break
        current = thinned
    return current, cap_for(len(current))


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineBudget:
    """What the pipeline may spend, in one of two modes.

    ``steps`` mode screens every configuration until ``steps`` steps per
    environment (with a wall-clock safety cap). ``time`` mode targets
    approximately ``minutes`` minutes of total measurement and derives
    per-stage caps; the actual elapsed time of every stage is recorded in
    the report, because a plan is a plan, not a measurement.
    """

    mode: str
    steps: int = DEFAULT_SCREEN_STEPS
    minutes: float = DEFAULT_TIME_BUDGET_MINUTES
    screen_cap_seconds: float = DEFAULT_SCREEN_CAP_SECONDS
    device_steps: int = DEFAULT_DEVICE_STEPS
    validation_steps: int = DEFAULT_VALIDATION_STEPS

    def __post_init__(self) -> None:
        if self.mode not in ("steps", "time"):
            raise ValueError("budget mode must be 'steps' or 'time'")
        if self.mode == "steps" and self.steps < 1:
            raise ValueError("steps must be positive in steps mode")
        if self.mode == "time" and not (
            MIN_TIME_BUDGET_MINUTES <= self.minutes <= MAX_TIME_BUDGET_MINUTES
        ):
            raise ValueError(
                f"time budget must be between {MIN_TIME_BUDGET_MINUTES:g} and "
                f"{MAX_TIME_BUDGET_MINUTES:g} minutes"
            )

    @classmethod
    def for_steps(
        cls,
        steps: int = DEFAULT_SCREEN_STEPS,
        *,
        device_steps: int | None = None,
        validation_steps: int | None = None,
    ) -> PipelineBudget:
        return cls(
            mode="steps",
            steps=steps,
            device_steps=device_steps or min(DEFAULT_DEVICE_STEPS, steps),
            validation_steps=validation_steps or min(DEFAULT_VALIDATION_STEPS, max(steps, 500)),
        )

    @classmethod
    def for_time(cls, minutes: float = DEFAULT_TIME_BUDGET_MINUTES) -> PipelineBudget:
        return cls(mode="time", minutes=minutes)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "steps": self.steps if self.mode == "steps" else None,
            "minutes": round(self.minutes, 2) if self.mode == "time" else None,
            "device_steps": self.device_steps,
            "validation_steps": self.validation_steps,
        }


def allocate_time_budget(
    minutes: float,
    *,
    candidate_count: int,
    finalist_count: int,
    device_candidate_count: int,
    has_training_runtime: bool,
) -> dict[str, float]:
    """Split a time budget across the pipeline's stages (planning only).

    Screening gets the majority share; the device comparison only gets a
    share when there is more than one device to compare and a training
    runtime to compare it on; validation gets the remainder only when a
    training runtime exists, and otherwise folds back into screening.
    """
    total = max(0.0, float(minutes) * 60.0)
    validation_share = 0.30 if has_training_runtime else 0.0
    device_share = 0.15 if device_candidate_count > 1 and has_training_runtime else 0.0
    screen_share = max(0.05, 1.0 - validation_share - device_share)
    return {
        "total_seconds": total,
        "screen_seconds": total * screen_share,
        "device_seconds": total * device_share,
        "validation_seconds": total * validation_share,
        "validation_seconds_per_finalist": (
            total * validation_share / max(finalist_count, 1) if finalist_count else 0.0
        ),
    }


def estimate_validation_steps(screen_row: dict[str, Any], seconds: float) -> int:
    """Step count for a validation slice expected to take roughly ``seconds``.

    A *planning* estimate built from this configuration's own measured
    bridge throughput and a conservative training slowdown factor; it only
    sizes the slice. The reported result is whatever the slice actually
    measured.
    """
    bridge = float(screen_row.get("steps_per_second") or 0.0)
    if bridge <= 0.0 or seconds <= 0.0:
        return DEFAULT_VALIDATION_STEPS
    expected = bridge * seconds * TRAINING_SLOWDOWN_PLANNING_FACTOR
    return int(max(500.0, min(20_000.0, expected)))


# ---------------------------------------------------------------------------
# Configuration validation (shared by the benchmark and the launcher)
# ---------------------------------------------------------------------------


def validate_configuration(
    environment_count: int,
    env_workers: int,
    device: str,
    *,
    runtime: dict[str, Any] | None = None,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    """One compatibility verdict used by both the benchmark plan and the
    Control Center launcher, so the two can never disagree.

    ``errors`` make the configuration invalid; ``warnings`` describe a
    valid but suboptimal topology (uneven shards, worker
    oversubscription). A CUDA request on a host without CUDA is an error,
    not a silent fallback to CPU.
    """
    compatibility = worker_compatibility(environment_count, env_workers, cpu_count=cpu_count)
    errors = list(compatibility.errors)
    warnings = list(compatibility.warnings)
    if device not in ("auto", "cpu", "cuda"):
        errors.append("device must be one of auto, cpu, cuda")
    elif device == "cuda" and runtime is not None and not runtime.get("cuda_available"):
        errors.append("device 'cuda' requested but no CUDA runtime is available")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "environment_count": compatibility.environment_count,
        "env_workers": compatibility.resolved_workers,
        "shards": [
            {"worker": shard.worker, "offset": shard.offset, "count": shard.count}
            for shard in compatibility.shards
        ],
    }


# ---------------------------------------------------------------------------
# Stability and recommendation
# ---------------------------------------------------------------------------


def _row_jitter(row: dict[str, Any]) -> float | None:
    p50 = row.get("vector_step_latency_p50_ms")
    p95 = row.get("vector_step_latency_p95_ms")
    if not isinstance(p50, (int, float)) or not isinstance(p95, (int, float)) or p50 <= 0.0:
        return None
    return float(p95) / float(p50)


def _row_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row.get("environments", 0)), int(row.get("workers", 1)))


def _row_is_measured(row: dict[str, Any]) -> bool:
    # Screening rows use "ok", validation rows "measured"; both mean the
    # number in steps_per_second was actually measured.
    return row.get("status", "ok") in ("ok", "measured") and isinstance(
        row.get("steps_per_second"), (int, float)
    )


def select_finalists(
    screen_rows: list[dict[str, Any]], count: int = DEFAULT_FINALISTS
) -> list[dict[str, Any]]:
    """The best measured screening rows worth an expensive validation slice.

    Ranked by throughput, then by latency jitter, then by fewer workers
    (a cheaper topology at equal throughput). Failed and unmeasured rows
    are never selected.
    """
    measured = [row for row in screen_rows if _row_is_measured(row)]

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        jitter = _row_jitter(row)
        return (
            -float(row["steps_per_second"]),
            jitter if jitter is not None else float("inf"),
            int(row.get("workers", 1)),
        )

    return sorted(measured, key=sort_key)[: max(0, int(count))]


def recommend(
    screen_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]] | None = None,
    *,
    device: str = "cpu",
    inference_device: str = "cpu",
    near_best_fraction: float = NEAR_BEST_FRACTION,
    jitter_threshold: float = JITTER_THRESHOLD,
    cpu_count: int | None = None,
) -> dict[str, Any] | None:
    """Pick the recommended configuration from real measurements only.

    When validation rows exist, only validated configurations compete:
    validated training throughput and bridge stepping throughput are
    different quantities and must not be compared against each other.
    Among the eligible configurations the best throughput defines a
    near-best band (default: within 10 %); inside that band the winner is
    the most *stable* choice — lowest latency jitter, then the fewest
    workers — rather than the absolute peak. Every number quoted in the
    rationale comes from the chosen row.

    Returns ``None`` when nothing was measured; the caller then reports
    the reason instead of inventing a configuration.
    """
    validation_rows = validation_rows or []
    validated: dict[tuple[int, int], float] = {}
    for row in validation_rows:
        if (
            row.get("status", "ok") in ("ok", "measured")
            and isinstance(row.get("steps_per_second"), (int, float))
            and float(row["steps_per_second"]) > 0.0
        ):
            validated[_row_key(row)] = float(row["steps_per_second"])
    eligible = [
        row for row in screen_rows if _row_is_measured(row) and float(row["steps_per_second"]) > 0.0
    ]
    if validated:
        eligible = [row for row in eligible if _row_key(row) in validated]
        basis = "validated_training_slice"
    else:
        basis = "screening_only"
    if not eligible:
        return None

    def score_of(row: dict[str, Any]) -> float:
        if basis == "validated_training_slice":
            return validated[_row_key(row)]
        return float(row["steps_per_second"])

    scored = [(row, score_of(row)) for row in eligible]
    best_score = max(score for _, score in scored)
    if best_score <= 0.0:
        return None
    near_best = [
        (row, score) for row, score in scored if score >= best_score * (1.0 - near_best_fraction)
    ]

    def stability_key(item: tuple[dict[str, Any], float]) -> tuple[Any, ...]:
        row, score = item
        jitter = _row_jitter(row)
        unstable = 1 if (jitter is not None and jitter > jitter_threshold) else 0
        return (
            unstable,
            jitter if jitter is not None else float("inf"),
            int(row.get("workers", 1)),
            -score,
        )

    chosen_row, chosen_score = min(near_best, key=stability_key)
    jitter = _row_jitter(chosen_row)
    warnings = _recommendation_warnings(
        screen_rows, _row_key(chosen_row), jitter_threshold=jitter_threshold
    )
    environment_count = int(chosen_row["environments"])
    chosen_workers = int(chosen_row["workers"])
    compatibility = worker_compatibility(environment_count, chosen_workers, cpu_count=cpu_count)
    warnings.extend(compatibility.warnings)
    rationale = [
        f"measured {chosen_score:.1f} steps/s on the "
        + (
            "real training path (validated with a PPO training slice)"
            if basis == "validated_training_slice"
            else "bridge stepping path"
        ),
        f"best measured throughput in the sweep was {best_score:.1f} steps/s; "
        f"the recommendation stays within {near_best_fraction:.0%} of it",
    ]
    if jitter is not None:
        rationale.append(
            f"stable stepping: p95/p50 latency ratio {jitter:.2f} "
            f"(threshold {jitter_threshold:.1f})"
        )
    if basis == "screening_only":
        rationale.append(
            "no training-slice validation was available, so this is a "
            "bridge-throughput recommendation; run the pipeline again with the "
            "training extras installed for an end-to-end measurement"
        )
    shard_sizes = [shard.count for shard in compatibility.shards]
    if shard_sizes:
        rationale.append(
            f"topology: {environment_count} environments over {chosen_workers} "
            f"worker(s) ({'+'.join(str(size) for size in shard_sizes)})"
        )
    return {
        "format": RECOMMENDATION_FORMAT,
        "environment_count": environment_count,
        "env_workers": chosen_workers,
        "device": device,
        "inference_device": inference_device,
        "expected_steps_per_second": chosen_score,
        "screened_steps_per_second": float(chosen_row["steps_per_second"]),
        "validated_steps_per_second": (
            chosen_score if basis == "validated_training_slice" else None
        ),
        "latency_p50_ms": chosen_row.get("vector_step_latency_p50_ms"),
        "latency_p95_ms": chosen_row.get("vector_step_latency_p95_ms"),
        "startup_seconds": chosen_row.get("startup_seconds"),
        "basis": basis,
        "rationale": rationale,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def default_output_root(project_path: str | Path) -> Path:
    return Path(project_path) / "training" / "benchmarks" / "pipelines"


def recommendation_path(project_root: str | Path) -> Path:
    """Machine-local recommendation file (same convention as the hardware
    profile: a gitignored dot-directory inside the checkout)."""
    return Path(project_root) / ".sandboxai" / "recommended_config.json"


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / PIPELINE_REPORT_NAME
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    screen_rows = [
        row
        for stage in report.get("stages", [])
        if stage.get("name") == "screening"
        for row in stage.get("configurations", [])
        if _row_is_measured(row)
    ]
    if screen_rows:
        # The screening rows are benchmark.py-shaped, so the existing
        # benchmark history readers pick the sweep up unchanged.
        (destination / "benchmark.json").write_text(
            json.dumps(screen_rows, indent=2) + "\n", encoding="utf-8"
        )
    return path


def save_recommendation(
    recommendation: dict[str, Any],
    project_root: str | Path,
    *,
    source_report: Path | None = None,
) -> Path:
    path = recommendation_path(project_root)
    record = dict(recommendation)
    record["format"] = RECOMMENDATION_FORMAT
    record["created_utc"] = _utc_now()
    record["source_report"] = str(source_report) if source_report is not None else None
    record["applied_utc"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def load_recommendation(project_root: str | Path) -> dict[str, Any] | None:
    """The persisted recommendation, or ``None`` when absent or unreadable."""
    path = recommendation_path(project_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("format") != RECOMMENDATION_FORMAT:
        return None
    if not isinstance(value.get("environment_count"), int):
        return None
    return value


def mark_recommendation_applied(project_root: str | Path) -> dict[str, Any] | None:
    """Record that the operator applied the persisted recommendation."""
    recommendation = load_recommendation(project_root)
    if recommendation is None:
        return None
    recommendation["applied_utc"] = _utc_now()
    path = recommendation_path(project_root)
    path.write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")
    return recommendation


def discover_reports(root: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    """Every persisted pipeline report under ``root``, newest first."""
    base = Path(root)
    if not base.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in base.rglob(PIPELINE_REPORT_NAME):
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        entries.append(
            {
                "path": str(path),
                "directory": str(path.parent),
                "created_utc": report.get("created_utc"),
                "status": report.get("status"),
                "recommendation": report.get("recommendation"),
                "report": report,
            }
        )
    entries.sort(key=lambda item: item.get("created_utc") or "", reverse=True)
    return entries[:limit]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _emit(on_progress: ProgressFn | None, **event: Any) -> None:
    if on_progress is not None:
        on_progress(event)


def _stage_record(name: str, **values: Any) -> dict[str, Any]:
    return {"name": name, "status": "completed", **values}


def _recommendation_warnings(
    screen_rows: list[dict[str, Any]],
    chosen_key: tuple[int, int],
    *,
    jitter_threshold: float,
) -> list[str]:
    """Warnings about configurations the operator should know were rejected.

    Failed configurations and unstable near-best peaks are named explicitly
    so the recommendation is auditable against the raw sweep.
    """
    warnings: list[str] = []
    for row in screen_rows:
        key = _row_key(row)
        if key == chosen_key:
            continue
        if row.get("status", "ok") != "ok":
            warnings.append(
                f"{key[0]} environments / {key[1]} workers failed screening: "
                f"{row.get('error', 'unknown error')}"
            )
        elif _row_is_measured(row):
            row_jitter = _row_jitter(row)
            if row_jitter is not None and row_jitter > jitter_threshold:
                warnings.append(
                    f"{key[0]} environments / {key[1]} workers measured "
                    f"{float(row['steps_per_second']):.1f} steps/s but with unstable "
                    f"latency (p95/p50 = {row_jitter:.1f})"
                )
    return warnings


@dataclass
class _ScreeningOutcome:
    """Rows plus the stage record; ``complete`` False means no recommendation."""

    rows: list[dict[str, Any]]
    stage: dict[str, Any]
    complete: bool
    cancelled: bool


def _measure_screen_row(
    candidate: dict[str, Any],
    *,
    project: Path,
    executable: str,
    budget: PipelineBudget,
    screen_cap: float,
    enemy_count: int,
    seed: int,
    curriculum_level: int,
    compact_infos: bool,
    warmup_steps: int,
    screen: Callable[..., list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Measure one ``(environments, workers)`` configuration. Never raises."""
    environment_count = int(candidate["environments"])
    worker_count = int(candidate["workers"])
    try:
        if screen is None:
            measured = benchmark_simulation(
                project_path=project,
                godot_executable=executable,
                environment_counts=[environment_count],
                worker_counts=[worker_count],
                steps=budget.steps if budget.mode == "steps" else 10**9,
                enemy_count=enemy_count,
                seed=seed,
                curriculum_level=curriculum_level,
                max_seconds_per_config=(
                    budget.screen_cap_seconds if budget.mode == "steps" else screen_cap
                ),
                compact_infos=compact_infos,
                warmup_steps=warmup_steps,
            )
        else:
            measured = screen(environments=environment_count, workers=worker_count)
        row = dict(measured[0]) if measured else {}
        row["status"] = "ok" if measured else "failed"
        if not measured:
            row["error"] = "measurement returned no rows"
        row["candidate_warnings"] = list(candidate.get("warnings", []))
    except Exception as exc:  # one broken configuration must not sink the sweep
        row = {
            "environments": environment_count,
            "workers": worker_count,
            "status": "failed",
            "error": str(exc),
        }
    jitter = _row_jitter(row)
    if jitter is not None:
        row["latency_jitter"] = round(jitter, 3)
    return row


def _run_screening(
    candidates: list[dict[str, Any]],
    *,
    project: Path,
    executable: str,
    budget: PipelineBudget,
    screen_cap: float,
    enemy_count: int,
    seed: int,
    curriculum_level: int,
    compact_infos: bool,
    warmup_steps: int,
    screen: Callable[..., list[dict[str, Any]]] | None,
    cancel: CancelFn | None,
    on_progress: ProgressFn | None,
) -> _ScreeningOutcome:
    """Measure every planned configuration, honouring cancel and time budget."""
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    cancelled = False
    budget_exhausted = False
    _emit(on_progress, stage="screening", status="started", total=len(candidates), index=0)
    for index, candidate in enumerate(candidates):
        if cancel is not None and cancel():
            cancelled = True
            break
        _emit(
            on_progress,
            stage="screening",
            status="started",
            index=index,
            total=len(candidates),
            configuration=dict(candidate),
        )
        rows.append(
            _measure_screen_row(
                candidate,
                project=project,
                executable=executable,
                budget=budget,
                screen_cap=screen_cap,
                enemy_count=enemy_count,
                seed=seed,
                curriculum_level=curriculum_level,
                compact_infos=compact_infos,
                warmup_steps=warmup_steps,
                screen=screen,
            )
        )
        row = rows[-1]
        _emit(
            on_progress,
            stage="screening",
            status="completed" if row["status"] == "ok" else "failed",
            index=index,
            total=len(candidates),
            configuration=dict(candidate),
            steps_per_second=row.get("steps_per_second"),
        )
        if budget.mode == "time" and time.monotonic() - started > (
            screen_cap * len(candidates) * 1.5
            + ESTIMATED_STARTUP_SECONDS_PER_CONFIG * len(candidates)
            + 30.0
        ):
            budget_exhausted = True
            for remaining in candidates[index + 1 :]:
                rows.append(
                    {
                        "environments": remaining["environments"],
                        "workers": remaining["workers"],
                        "status": "skipped",
                        "error": "screening time budget exhausted",
                    }
                )
            break
    complete = not cancelled and not budget_exhausted
    stage = _stage_record(
        "screening",
        status="completed" if complete else ("cancelled" if cancelled else "failed"),
        reason=(
            None
            if complete
            else ("cancelled by operator" if cancelled else "time budget exhausted")
        ),
        configurations=rows,
        elapsed_seconds=time.monotonic() - started,
    )
    return _ScreeningOutcome(rows=rows, stage=stage, complete=complete, cancelled=cancelled)


def _plan_device_steps(
    *,
    budget: PipelineBudget,
    device_seconds: float,
    device_candidate_count: int,
    screen_rows: list[dict[str, Any]],
) -> int:
    """Slice size for the device comparison (planning estimate, never reported)."""
    steps = budget.device_steps
    if budget.mode != "time":
        return steps
    # Size each slice from the *measured* screening throughput so the stage
    # fits its share of the budget.
    measured_sps = sorted(
        float(row["steps_per_second"]) for row in screen_rows if _row_is_measured(row)
    )
    median_sps = measured_sps[len(measured_sps) // 2] if measured_sps else 0.0
    per_candidate_seconds = device_seconds / max(device_candidate_count, 1)
    if median_sps > 0.0:
        steps = int(
            max(
                200.0,
                min(
                    DEFAULT_DEVICE_STEPS,
                    median_sps * per_candidate_seconds * TRAINING_SLOWDOWN_PLANNING_FACTOR,
                ),
            )
        )
    return steps


def _run_devices(
    *,
    project: Path,
    executable: str,
    runtime: dict[str, Any],
    budget: PipelineBudget,
    screen_rows: list[dict[str, Any]],
    device_seconds: float,
    use_saved_profile: bool,
    measure_device: Callable[..., Any] | None,
    cancel: CancelFn | None,
    on_progress: ProgressFn | None,
) -> tuple[str, str, dict[str, Any]]:
    """Decide the training device. Returns (device, inference_device, stage)."""
    from .hardware_profile import (
        DeviceCandidate,
        available_candidates,
        load_profile,
        measure_devices,
        select_best,
    )

    device = "cpu"
    inference_device = "cpu"
    stage = _stage_record("devices", measurements=[])
    device_candidates = available_candidates(has_cuda=runtime.get("cuda_available"))
    saved_profile = (
        load_profile(project / ".sandboxai" / "hardware_profile.json")
        if use_saved_profile
        else None
    )
    if saved_profile is not None and not saved_profile.fallback:
        device = saved_profile.device
        inference_device = saved_profile.inference_device
        stage["status"] = "skipped"
        stage["reason"] = (
            f"reusing the persisted hardware profile (selected {saved_profile.selected_device})"
        )
    elif len(device_candidates) > 1 and runtime.get("torch_available"):
        _emit(
            on_progress,
            stage="devices",
            status="started",
            total=len(device_candidates),
        )

        def default_measure_bound(candidate: DeviceCandidate):
            from .hardware_profile import default_measure

            return default_measure(
                candidate,
                project_path=project,
                godot_executable=executable,
                steps=_plan_device_steps(
                    budget=budget,
                    device_seconds=device_seconds,
                    device_candidate_count=len(device_candidates),
                    screen_rows=screen_rows,
                ),
            )

        measurements = measure_devices(
            device_candidates,
            measure_device or default_measure_bound,
            cancel=cancel,
        )
        stage["measurements"] = [measurement.to_dict() for measurement in measurements]
        best = select_best(measurements)
        if best is None:
            stage["status"] = "failed"
            stage["reason"] = "no device could be measured; falling back to CPU for validation"
        else:
            device = best.device
            inference_device = best.inference_device
            stage["reason"] = f"selected {best.label} from live measurement"
    else:
        stage["status"] = "skipped"
        stage["reason"] = (
            "single device candidate (cpu); no comparison needed"
            if runtime.get("torch_available")
            else "training runtime unavailable (torch not installed); device axis not measurable"
        )
    _emit(
        on_progress,
        stage="devices",
        status=stage["status"],
        message=stage.get("reason", f"device: {device}"),
    )
    return device, inference_device, stage


def _run_validation(
    chosen_finalists: list[dict[str, Any]],
    *,
    project: Path,
    executable: str,
    runtime: dict[str, Any],
    budget: PipelineBudget,
    device: str,
    inference_device: str,
    validation_seconds_per_finalist: float,
    enemy_count: int,
    seed: int,
    validate_training: Callable[..., Any] | None,
    cancel: CancelFn | None,
    on_progress: ProgressFn | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """One real training slice per finalist. Returns (rows, stage, cancelled)."""
    rows: list[dict[str, Any]] = []
    stage = _stage_record("validation", configurations=[])
    cancelled = False
    _emit(
        on_progress,
        stage="validation",
        status="started",
        total=len(chosen_finalists),
        index=0,
    )
    started = time.monotonic()
    for index, row in enumerate(chosen_finalists):
        if cancel is not None and cancel():
            cancelled = True
            break
        environment_count = int(row["environments"])
        worker_count = int(row["workers"])
        steps = (
            estimate_validation_steps(row, validation_seconds_per_finalist)
            if budget.mode == "time"
            else budget.validation_steps
        )
        _emit(
            on_progress,
            stage="validation",
            status="started",
            index=index,
            total=len(chosen_finalists),
            configuration={
                "environments": environment_count,
                "workers": worker_count,
                "device": device,
                "steps": steps,
            },
        )
        validation_row: dict[str, Any] = {
            "environments": environment_count,
            "workers": worker_count,
            "device": device,
            "inference_device": inference_device,
            "steps": steps,
        }
        try:
            if validate_training is None:
                from .hardware_profile import DeviceCandidate, default_measure

                candidate_device = DeviceCandidate(
                    label=f"{device}/{worker_count}w",
                    device=device,
                    inference_device=inference_device,
                )
                measurement = default_measure(
                    candidate_device,
                    project_path=project,
                    godot_executable=executable,
                    steps=steps,
                    environment_count=environment_count,
                    env_workers=worker_count,
                    enemy_count=enemy_count,
                    seed=seed,
                )
            else:
                measurement = validate_training(
                    environments=environment_count,
                    workers=worker_count,
                    device=device,
                    steps=steps,
                )
            validation_row.update(measurement.to_dict())
            validation_row["status"] = "measured" if measurement.ok else measurement.status
            if measurement.error:
                validation_row["error"] = measurement.error
        except Exception as exc:  # a failed slice is data, not a crash
            validation_row["status"] = "failed"
            validation_row["error"] = str(exc)
        rows.append(validation_row)
        _emit(
            on_progress,
            stage="validation",
            status="completed" if validation_row["status"] == "measured" else "failed",
            index=index,
            total=len(chosen_finalists),
            configuration=dict(validation_row),
        )
    stage["configurations"] = rows
    stage["elapsed_seconds"] = time.monotonic() - started
    if cancelled:
        stage["status"] = "cancelled"
        stage["reason"] = "cancelled by operator"
    return rows, stage, cancelled


def run_benchmark_pipeline(
    *,
    project_path: str | Path,
    godot_executable: str | None = None,
    budget: PipelineBudget | None = None,
    environment_counts: list[int] | tuple[int, ...] | None = None,
    worker_counts: list[int] | tuple[int, ...] | None = None,
    finalists: int = DEFAULT_FINALISTS,
    enemy_count: int = 1,
    seed: int = 1234,
    curriculum_level: int = 3,
    compact_infos: bool = True,
    warmup_steps: int = 25,
    output_dir: str | Path | None = None,
    save: bool = True,
    use_saved_profile: bool = True,
    cancel: CancelFn | None = None,
    on_progress: ProgressFn | None = None,
    runtime: dict[str, Any] | None = None,
    screen: Callable[..., list[dict[str, Any]]] | None = None,
    measure_device: Callable[..., Any] | None = None,
    validate_training: Callable[..., Any] | None = None,
    recommendation_project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the staged benchmark and return the full pipeline report.

    The stage functions (``screen``, ``measure_device``, ``validate_training``)
    and the ``runtime`` facts are injectable so the orchestration is testable
    without a Godot binary; the defaults are the real measurers and a real
    runtime probe. Cancellation is honoured between configurations and
    between finalists — an in-flight slice always finishes, exactly like the
    hardware wizard.

    ``recommendation_project_root`` anchors the machine-local
    ``.sandboxai/recommended_config.json``; it defaults to ``project_path``
    and exists so an adapter with a separate output root still persists the
    recommendation next to its project.
    """
    started = time.monotonic()
    budget = budget or PipelineBudget.for_time()
    project = Path(project_path)
    report: dict[str, Any] = {
        "format": PIPELINE_FORMAT,
        "schema_version": 1,
        "created_utc": _utc_now(),
        "status": "running",
        "project_path": str(project),
        "budget": budget.describe(),
        "runtime": runtime if runtime is not None else {},
        "stages": [],
        "recommendation": None,
        "recommendation_reason": "",
        "notes": [],
    }

    def finish() -> dict[str, Any]:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if save:
            directory = (
                Path(output_dir)
                if output_dir is not None
                else default_output_root(project) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            )
            try:
                path = write_report(report, directory)
                report["report_path"] = str(path)
            except OSError as exc:
                report["notes"].append(f"could not persist the report: {exc}")
            recommendation = report.get("recommendation")
            if recommendation is not None:
                try:
                    root = (
                        Path(recommendation_project_root)
                        if recommendation_project_root is not None
                        else project
                    )
                    save_recommendation(
                        recommendation, root, source_report=Path(report.get("report_path", ""))
                    )
                except OSError as exc:
                    report["notes"].append(f"could not persist the recommendation: {exc}")
        return report

    # ---- Stage 1: runtime discovery ------------------------------------
    _emit(on_progress, stage="discovery", status="started")
    if runtime is None:
        runtime = available_runtime(project, godot_executable)
        report["runtime"] = runtime
    _emit(
        on_progress,
        stage="discovery",
        status="completed",
        message=("godot available" if runtime.get("godot_available") else "godot NOT available"),
    )
    if not runtime.get("godot_available"):
        report["status"] = "unavailable"
        report["recommendation_reason"] = (
            "no usable Godot runtime was found; configure the executable once with "
            "`sandboxai validate-runtime --godot-executable <path>` and re-run"
        )
        report["stages"].append(
            _stage_record("discovery", status="failed", reason=report["recommendation_reason"])
        )
        return finish()
    report["stages"].append(
        _stage_record(
            "discovery",
            message=(
                f"godot {runtime.get('godot_version') or '?'}"
                if runtime.get("godot_version")
                else "godot available"
            ),
        )
    )

    executable = str(runtime.get("godot_resolved_executable") or godot_executable or "godot")
    cpu_count = int(runtime.get("cpu_count_available_to_process") or runtime.get("cpu_count") or 1)
    candidates = plan_candidates(environment_counts, worker_counts, cpu_count=cpu_count)
    if not candidates:
        report["status"] = "unavailable"
        report["recommendation_reason"] = "no compatible environment/worker candidates to measure"
        return finish()
    report["plan"] = {
        "environment_counts": sorted({row["environments"] for row in candidates}),
        "worker_counts": sorted({row["workers"] for row in candidates}),
        "candidates": candidates,
        "finalists": int(finalists),
        "warmup_steps": int(warmup_steps),
        "compact_infos": bool(compact_infos),
    }

    # ---- Time-mode budget allocation ------------------------------------
    screen_cap = budget.screen_cap_seconds
    validation_seconds_per_finalist = 0.0
    device_seconds = 0.0
    if budget.mode == "time":
        from .hardware_profile import available_candidates

        allocation = allocate_time_budget(
            budget.minutes,
            candidate_count=len(candidates),
            finalist_count=max(1, finalists),
            device_candidate_count=len(
                available_candidates(has_cuda=runtime.get("cuda_available"))
            ),
            has_training_runtime=bool(runtime.get("torch_available")),
        )
        candidates, screen_cap = fit_candidates_to_budget(candidates, allocation["screen_seconds"])
        report["plan"]["candidates"] = candidates
        report["plan"]["environment_counts"] = sorted({row["environments"] for row in candidates})
        report["plan"]["worker_counts"] = sorted({row["workers"] for row in candidates})
        report["budget"]["screen_cap_seconds_per_config"] = screen_cap
        report["budget"]["validation_seconds_per_finalist"] = allocation[
            "validation_seconds_per_finalist"
        ]
        validation_seconds_per_finalist = allocation["validation_seconds_per_finalist"]
        device_seconds = allocation["device_seconds"]

    # ---- Stage 2: screening ----------------------------------------------
    screening = _run_screening(
        candidates,
        project=project,
        executable=executable,
        budget=budget,
        screen_cap=screen_cap,
        enemy_count=enemy_count,
        seed=seed,
        curriculum_level=curriculum_level,
        compact_infos=compact_infos,
        warmup_steps=warmup_steps,
        screen=screen,
        cancel=cancel,
        on_progress=on_progress,
    )
    report["stages"].append(screening.stage)
    if not screening.complete:
        report["status"] = "cancelled" if screening.cancelled else "incomplete"
        report["recommendation_reason"] = (
            "screening did not complete; the measured subset is not a fair basis "
            "for a recommendation"
        )
        return finish()

    # ---- Stage 3: device comparison -------------------------------------
    device, inference_device, device_stage = _run_devices(
        project=project,
        executable=executable,
        runtime=runtime,
        budget=budget,
        screen_rows=screening.rows,
        device_seconds=device_seconds,
        use_saved_profile=use_saved_profile,
        measure_device=measure_device,
        cancel=cancel,
        on_progress=on_progress,
    )
    report["stages"].append(device_stage)

    # ---- Stage 4: finalist validation ------------------------------------
    chosen_finalists = select_finalists(screening.rows, finalists)
    validation_rows: list[dict[str, Any]] = []
    validation_stage: dict[str, Any] = _stage_record("validation", configurations=[])
    if not chosen_finalists:
        validation_stage["status"] = "skipped"
        validation_stage["reason"] = "no successfully screened configuration to validate"
    elif not runtime.get("torch_available"):
        validation_stage["status"] = "skipped"
        validation_stage["reason"] = (
            "torch is not installed, so the real training path cannot be measured; "
            "the recommendation will be based on bridge throughput only"
        )
    else:
        validation_rows, validation_stage, validation_cancelled = _run_validation(
            chosen_finalists,
            project=project,
            executable=executable,
            runtime=runtime,
            budget=budget,
            device=device,
            inference_device=inference_device,
            validation_seconds_per_finalist=validation_seconds_per_finalist,
            enemy_count=enemy_count,
            seed=seed,
            validate_training=validate_training,
            cancel=cancel,
            on_progress=on_progress,
        )
        if validation_cancelled:
            report["stages"].append(validation_stage)
            report["status"] = "cancelled"
            report["recommendation_reason"] = "validation was cancelled"
            return finish()
    report["stages"].append(validation_stage)

    # ---- Stage 5: recommendation -----------------------------------------
    recommendation = recommend(
        screening.rows,
        validation_rows,
        device=device,
        inference_device=inference_device,
        cpu_count=cpu_count,
    )
    if recommendation is None:
        report["recommendation_reason"] = (
            "no configuration produced a usable measurement; inspect the stage "
            "errors in this report"
        )
    else:
        report["recommendation"] = recommendation
    report["status"] = "completed"
    return finish()
