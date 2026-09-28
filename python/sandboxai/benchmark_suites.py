"""Benchmark suites (Phase 13).

``benchmark.py`` answers one question — how does raw stepping throughput
scale with the number of environments in a single Godot process. This
module turns that into the four *comparable* suites the research plan
calls for, at the environment counts 1, 4, 8, 16, 32, 64:

* ``curriculum_1_4``  — analytic enemies, no world geometry, no perception
  gating. The cheapest configuration; the ceiling everything else is
  measured against.
* ``curriculum_5_10`` — obstacles, FOV/LOS, sound and memory enabled.
* ``perception_combat`` — the same, with several enemies actively fighting,
  which is the load an actual training run sees.
* ``map_analyzer`` — exploration sweeps dominating, combat minimal.

The difference between two suites at the same environment count is the
overhead attributable to what was switched on, which is the only honest
way to answer "how expensive is perception?" without instrumenting the
engine's internals from Python.

**This module measures; it never estimates.** Every runner here needs a
real Godot executable. If Godot is missing, :func:`run_suites` raises
rather than returning a plausible-looking table — a fabricated benchmark
is worse than no benchmark. :func:`describe_plan` exists so the plan can
be inspected, documented and tested without an engine present.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from .benchmark import benchmark_simulation, summarize_scaling
from .telemetry import resource_snapshot

## The environment counts the research plan asks for.
SUITE_ENVIRONMENT_COUNTS: tuple[int, ...] = (1, 4, 8, 16, 32, 64)


class BenchmarkUnavailableError(RuntimeError):
    """Raised when a benchmark cannot be *measured* on this machine.

    Deliberately an error rather than a fallback: the alternative is
    returning numbers nobody measured.
    """


@dataclass(frozen=True)
class BenchmarkSuite:
    """One comparable configuration to measure."""

    name: str
    description: str
    curriculum_level: int
    enemy_count: int
    ## What this suite is meant to stress, for the report.
    stresses: tuple[str, ...] = ()
    environment_counts: tuple[int, ...] = SUITE_ENVIRONMENT_COUNTS
    steps: int = 2_000
    max_seconds_per_config: float = 20.0
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUITES: tuple[BenchmarkSuite, ...] = (
    BenchmarkSuite(
        name="curriculum_1_4",
        description="analytic enemies, no geometry, no perception gating",
        curriculum_level=3,
        enemy_count=1,
        stresses=("bridge", "physics"),
    ),
    BenchmarkSuite(
        name="curriculum_5_10",
        description="obstacles, FOV/LOS, sound and memory enabled",
        curriculum_level=8,
        enemy_count=2,
        stresses=("bridge", "physics", "perception", "navigation"),
    ),
    BenchmarkSuite(
        name="perception_combat",
        description="perception-heavy combat with several active enemies",
        curriculum_level=8,
        enemy_count=5,
        stresses=("perception", "navigation", "combat"),
    ),
    BenchmarkSuite(
        name="map_analyzer",
        description="exploration sweeps dominating, combat minimal",
        curriculum_level=10,
        enemy_count=1,
        stresses=("perception", "exploration", "spatial_memory"),
    ),
)

SUITES_BY_NAME: dict[str, BenchmarkSuite] = {suite.name: suite for suite in SUITES}


def describe_plan(suites: Sequence[BenchmarkSuite] = SUITES) -> dict[str, Any]:
    """The plan as data. Contains no measurements and never will."""
    return {
        "environment_counts": list(SUITE_ENVIRONMENT_COUNTS),
        "suites": [suite.to_dict() for suite in suites],
        "measured": False,
        "note": (
            "Plan only. Numbers appear exclusively in the output of run_suites(), "
            "which requires a real Godot executable."
        ),
    }


def godot_available(godot_executable: str = "godot") -> bool:
    return shutil.which(godot_executable) is not None


def run_suite(
    suite: BenchmarkSuite,
    project_path: str | Path,
    godot_executable: str = "godot",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measures one suite. Requires a working Godot executable."""
    if not godot_available(godot_executable):
        raise BenchmarkUnavailableError(
            f"Godot executable {godot_executable!r} not found; benchmarks cannot be measured "
            "on this machine. No numbers will be reported."
        )
    destination = Path(output_dir) / suite.name if output_dir is not None else None
    rows = benchmark_simulation(
        project_path=project_path,
        godot_executable=godot_executable,
        environment_counts=suite.environment_counts,
        steps=suite.steps,
        enemy_count=suite.enemy_count,
        seed=suite.seed,
        curriculum_level=suite.curriculum_level,
        output_dir=destination,
        max_seconds_per_config=suite.max_seconds_per_config,
    )
    return {
        "suite": suite.name,
        "config": suite.to_dict(),
        "rows": rows,
        "scaling": summarize_scaling(rows),
        "resources": resource_snapshot(),
        "measured": True,
    }


def run_suites(
    project_path: str | Path,
    godot_executable: str = "godot",
    suites: Sequence[BenchmarkSuite] = SUITES,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measures every suite and derives the overhead comparison."""
    results = [run_suite(suite, project_path, godot_executable, output_dir) for suite in suites]
    report = {
        "environment_counts": list(SUITE_ENVIRONMENT_COUNTS),
        "suites": results,
        "overhead": overhead_table(results),
        "measured": True,
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "suites.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _throughput_by_environment(result: dict[str, Any]) -> dict[int, float]:
    return {
        int(row["environments"]): float(row["steps_per_second"]) for row in result.get("rows", [])
    }


def overhead_table(
    results: Sequence[dict[str, Any]], baseline: str = "curriculum_1_4"
) -> dict[str, Any]:
    """Relative cost of each suite against the cheapest configuration.

    ``slowdown`` is baseline throughput divided by suite throughput at the
    same environment count: 1.0 means free, 2.0 means it halved the step
    rate. Only environment counts present in *both* runs are compared.
    """
    indexed = {result["suite"]: _throughput_by_environment(result) for result in results}
    reference = indexed.get(baseline)
    if not reference:
        return {}
    table: dict[str, Any] = {"baseline": baseline, "by_suite": {}}
    for name, throughput in indexed.items():
        if name == baseline:
            continue
        slowdowns = {
            count: reference[count] / value
            for count, value in throughput.items()
            if count in reference and value > 0.0
        }
        table["by_suite"][name] = {
            "slowdown_by_environment_count": slowdowns,
            "mean_slowdown": (sum(slowdowns.values()) / len(slowdowns)) if slowdowns else None,
        }
    return table


def format_report(report: dict[str, Any]) -> str:
    """Human-readable rendering. Refuses to format an unmeasured plan."""
    if not report.get("measured"):
        return (
            "No benchmark measurements available. "
            "Run sandboxai benchmark with a real Godot executable."
        )
    lines = ["Benchmark suites", "=" * 72]
    for result in report.get("suites", []):
        lines.append(f"\n{result['suite']}: {result['config']['description']}")
        lines.append(f"{'envs':>6}{'steps/s':>14}{'episodes/s':>14}{'elapsed s':>12}")
        lines.append("-" * 72)
        for row in result.get("rows", []):
            lines.append(
                f"{int(row['environments']):>6}{float(row['steps_per_second']):>14.1f}"
                f"{float(row['episodes_per_second']):>14.2f}"
                f"{float(row['elapsed_seconds']):>12.2f}"
            )
        scaling = result.get("scaling", {})
        if scaling:
            lines.append(
                f"best: {scaling.get('best_environment_count')} envs at "
                f"{scaling.get('best_steps_per_second', 0.0):.1f} steps/s; "
                f"diminishing returns at {scaling.get('diminishing_returns_at_environment_count')}"
            )
    overhead = report.get("overhead", {})
    if overhead.get("by_suite"):
        lines.append(f"\nOverhead vs {overhead['baseline']}")
        lines.append("-" * 72)
        for name, entry in overhead["by_suite"].items():
            mean = entry.get("mean_slowdown")
            lines.append(f"{name:<24}mean slowdown {mean:.2f}x" if mean else f"{name:<24}n/a")
    return "\n".join(lines) + "\n"


@dataclass
class BenchmarkEnvironmentReport:
    """Whether this machine can produce real numbers at all.

    Written so a CI log or a final report can state the situation instead
    of quietly omitting the benchmark section.
    """

    godot_executable: str = "godot"
    notes: list[str] = field(default_factory=list)

    def build(self) -> dict[str, Any]:
        available = godot_available(self.godot_executable)
        return {
            "godot_executable": self.godot_executable,
            "godot_available": available,
            "can_measure": available,
            "resources": resource_snapshot(),
            "environment_counts": list(SUITE_ENVIRONMENT_COUNTS),
            "suites": [suite.name for suite in SUITES],
            "notes": list(self.notes)
            + (
                []
                if available
                else [
                    "Godot was not found on PATH, so no throughput, CPU or RAM numbers "
                    "were measured. Run the suites locally to obtain them."
                ]
            ),
        }
