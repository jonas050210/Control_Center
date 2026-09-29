"""Low-overhead wall-clock profiling for PPO and the Godot bridge.

The profiler is deliberately opt-in.  A normal training run pays no timing,
locking, or report-writing cost; ``TrainingConfig.profile_training`` (or the
CLI ``--profile-training`` flag) installs one profiler shared by the PPO
callback and environment transport.

Measurements are wall time rather than CPU time because the performance
question for this pipeline is mostly *waiting*: synchronous Godot IPC,
periodic evaluation/checkpoint work, and CUDA kernel launch/synchronization
for a very small policy.  Timings are aggregate counters, not a trace, so a
100k-step run does not create a large profiling artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any


@dataclass
class _Timing:
    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    def add(self, seconds: float) -> None:
        value = max(0.0, float(seconds))
        self.count += 1
        self.total_seconds += value
        self.min_seconds = min(self.min_seconds, value)
        self.max_seconds = max(self.max_seconds, value)

    def to_dict(self) -> dict[str, Any]:
        mean = self.total_seconds / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "mean_ms": mean * 1000.0,
            "min_ms": (self.min_seconds if self.count else 0.0) * 1000.0,
            "max_ms": self.max_seconds * 1000.0,
        }


class TrainingProfiler:
    """Aggregate phase timings for one training run.

    ``record()`` is intentionally just a dictionary lookup and a handful of
    arithmetic operations.  It is called only when profiling was explicitly
    enabled; callers keep the disabled path as a single ``is not None``
    branch.
    """

    FORMAT = "sandboxai.training_profile/v1"

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings: dict[str, _Timing] = {}
        self.counters: dict[str, float] = {}
        self.iterations: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.server: dict[str, Any] = {}

    def record(self, name: str, seconds: float) -> None:
        self.timings.setdefault(name, _Timing()).add(seconds)

    def add(self, name: str, value: float) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def set_metadata(self, **values: Any) -> None:
        self.metadata.update(values)

    def add_iteration(
        self,
        index: int,
        rollout_seconds: float,
        update_seconds: float,
        timesteps: int,
    ) -> None:
        self.iterations.append(
            {
                "iteration": int(index),
                "timesteps": int(timesteps),
                "rollout_seconds": max(0.0, float(rollout_seconds)),
                "ppo_update_seconds": max(0.0, float(update_seconds)),
            }
        )

    def report(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started, 0.0)
        timings = {name: value.to_dict() for name, value in sorted(self.timings.items())}
        phase_totals = {
            name: timings.get(name, {}).get("total_seconds", 0.0)
            for name in (
                "ppo.rollout_collection",
                "ppo.policy_update",
                "callback.evaluation",
                "callback.checkpoint_save",
                "callback.resource_snapshot",
                "pipeline.step_hook",
                "env.step_total",
                "bridge.step.wait_response",
                "bridge.step.json_encode",
                "bridge.step.json_decode",
                # Evaluation-path breakdown (see sandboxai.evaluation and
                # sandboxai.checkpoint_eval). `callback.evaluation` remains
                # the full synchronous boundary cost; these buckets split it
                # so a profile answers where the time actually went:
                # process startup, prediction, environment stepping,
                # combined battery execution and cross-role overlap.
                "eval.env_startup",
                "eval.normal.total",
                "eval.normal.predict",
                "eval.normal.env_step",
                "eval.normal.bridge.step.total",
                "eval.battery.total",
                "eval.battery.execution",
                "eval.battery.predict",
                "eval.battery.env_step",
                "eval.battery.bridge.step.total",
                "eval.parallel.wall",
                "eval.parallel.overlap",
            )
        }
        phase_percent = {
            name: (float(value) / elapsed * 100.0 if elapsed > 0.0 else 0.0)
            for name, value in phase_totals.items()
        }
        return {
            "format": self.FORMAT,
            "elapsed_seconds": elapsed,
            "metadata": dict(self.metadata),
            "timings": timings,
            "counters": dict(sorted(self.counters.items())),
            "phase_totals_seconds": phase_totals,
            "phase_percent_of_profile_wall_time": phase_percent,
            "iterations": list(self.iterations),
            "godot_server": dict(self.server),
            "notes": [
                "Nested phases intentionally overlap (for example env.step_total includes bridge timings).",
                "bridge.*.wait_response includes pipe transit plus Godot parsing, simulation and response encoding.",
                "godot_server separates those server-side phases when the running Godot bridge supports profiling.",
                "eval.* buckets belong to the evaluation bridges only; they never fold into the training bridge.* buckets.",
                "eval.env_startup counts bridge process spawns: with process reuse it should fire once per role per run, not once per evaluation boundary.",
                "eval.parallel.overlap is wall time hidden by running normal and battery bridges concurrently; both jobs still join before checkpoint decisions.",
            ],
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.report(), indent=2, default=str) + "\n", encoding="utf-8")
        return destination


class PrefixedProfiler:
    """Read-only *view* of a :class:`TrainingProfiler` under a name prefix.

    The evaluation bridges must not pollute the training bridge's
    ``bridge.*`` buckets: a reused evaluation environment stepping tens of
    thousands of times would silently fold its transport timings into
    ``bridge.step.wait_response`` and make the training-side numbers
    meaningless. Handing the (already instrumented) transports this view
    instead of the profiler itself keeps every bucket separable:

    * training bridge  -> ``bridge.step.*``
    * normal eval      -> ``eval.normal.bridge.step.*``
    * battery executor -> ``eval.battery.bridge.step.*``

    Implements exactly the profiler surface the transport/client code uses
    (``record``, ``add``); ``is not None`` checks elsewhere keep working.
    """

    def __init__(self, profiler: TrainingProfiler, prefix: str) -> None:
        if not prefix.endswith("."):
            prefix += "."
        self.profiler = profiler
        self.prefix = prefix

    def record(self, name: str, seconds: float) -> None:
        self.profiler.record(self.prefix + str(name), seconds)

    def add(self, name: str, value: float) -> None:
        self.profiler.add(self.prefix + str(name), value)

    def set_metadata(self, **values: Any) -> None:
        self.profiler.set_metadata(**values)
