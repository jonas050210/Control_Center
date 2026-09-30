"""Versioned data contract shared by Control Center producers and consumers.

The schema is intentionally dependency-free so trainers, adapters and desktop
views can use it without importing a GUI or an ML framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

STATUS_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1

RunState = Literal["Idle", "Starting", "Running", "Paused", "Stopping", "Finished", "Error"]
KNOWN_RUN_STATES: frozenset[str] = frozenset(
    {"Idle", "Starting", "Running", "Paused", "Stopping", "Finished", "Error"}
)


class RunStatus(TypedDict, total=False):
    schema_version: int
    state: RunState
    pid: int
    updated_at: float
    stop_requested: bool
    timesteps: int
    total_training_steps: int
    progress: float
    steps_per_second: float
    mean_episode_reward: float
    episodes: int
    error: str


class ProcessSnapshot(TypedDict, total=False):
    id: str
    kind: str
    state: str
    returncode: int | None
    started_at: float
    pid: int
    stdout: list[str]
    stderr: list[str]
    error: str
    run_dir: str
    meta: dict[str, Any]
    command: list[str]
    backend: RunStatus
    error_code: str
    backend_error_code: str
    backend_error: str


class DashboardSnapshot(TypedDict):
    output_root: str
    run_count: int
    latest_run: dict[str, Any] | None
    active_processes: list[ProcessSnapshot]


class RunEvent(TypedDict, total=False):
    schema_version: int
    wall_time: float
    category: str
    message: str
    values: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    unit: str
    decimals: int
    description: str
    higher_is_better: bool | None = None


METRICS: dict[str, MetricDefinition] = {
    metric.key: metric
    for metric in (
        MetricDefinition(
            "mean_episode_reward", "Mean reward", "", 3, "Mean reward of completed episodes", True
        ),
        MetricDefinition(
            "steps_per_second",
            "Throughput",
            "steps/s",
            1,
            "Environment steps processed per second",
            True,
        ),
        MetricDefinition("mean_accuracy", "Accuracy", "%", 1, "Share of shots that hit", True),
        MetricDefinition("mean_kills", "Kills", "", 2, "Mean kills per completed episode", True),
        MetricDefinition(
            "mean_deaths", "Deaths", "", 2, "Mean deaths per completed episode", False
        ),
        MetricDefinition("win_rate", "Win rate", "%", 1, "Share of evaluated episodes won", True),
        MetricDefinition(
            "loss_rate", "Loss rate", "%", 1, "Share of evaluated episodes lost", False
        ),
        MetricDefinition("approx_kl", "Approx. KL", "", 4, "Approximate PPO policy divergence"),
        MetricDefinition(
            "clip_fraction", "Clip fraction", "", 3, "Fraction of PPO updates clipped"
        ),
        MetricDefinition(
            "explained_variance", "Explained variance", "", 3, "Value-function fit quality", True
        ),
        MetricDefinition("cpu_percent", "CPU", "%", 1, "Host CPU utilization"),
        MetricDefinition("gpu_utilization_percent", "GPU", "%", 1, "GPU compute utilization"),
        MetricDefinition(
            "process_rss_mb", "Process memory", "MB", 1, "Resident memory used by the trainer"
        ),
    )
}


def metric_definition(key: str) -> MetricDefinition | None:
    return METRICS.get(key)


def validate_status(value: Any) -> list[str]:
    """Return human-readable contract violations without raising on untrusted files."""
    if not isinstance(value, dict):
        return ["status must be a JSON object"]
    problems: list[str] = []
    version = value.get("schema_version", STATUS_SCHEMA_VERSION)
    if version != STATUS_SCHEMA_VERSION:
        problems.append(f"unsupported status schema_version {version!r}")
    state = value.get("state")
    if not isinstance(state, str) or state not in KNOWN_RUN_STATES:
        problems.append(f"unknown run state {state!r}")
    for key in ("pid", "timesteps", "total_training_steps", "episodes"):
        if key in value and (isinstance(value[key], bool) or not isinstance(value[key], int)):
            problems.append(f"{key} must be an integer")
    for key in ("updated_at", "progress", "steps_per_second", "mean_episode_reward"):
        if key in value and (
            isinstance(value[key], bool) or not isinstance(value[key], (int, float))
        ):
            problems.append(f"{key} must be numeric")
    return problems


def validate_event(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["event must be a JSON object"]
    problems: list[str] = []
    if value.get("schema_version", EVENT_SCHEMA_VERSION) != EVENT_SCHEMA_VERSION:
        problems.append("unsupported event schema_version")
    for key in ("category", "message"):
        if not isinstance(value.get(key), str) or not value[key]:
            problems.append(f"{key} must be a non-empty string")
    return problems
