"""Pure presentation logic for the desktop Control Center.

Everything here is plain Python: no Tkinter, no adapter instantiation, no
process/file I/O. It takes the dictionaries :mod:`sandboxai.adapter` already
returns and turns them into display-ready structures (table rows, formatted
strings, a validated :class:`~sandboxai.config.TrainingConfig`) or does the
opposite (turning raw form strings into a validated config).

Splitting this out of ``control_center_desktop.py`` has two purposes:

* it is unit-testable without Tkinter, a display, or a mock adapter - useful
  in this repository (and CI images) where Tkinter may not even be
  installed, and required by the "injectable adapter at the unit-test
  boundary" testing guidance for GUI code;
* it keeps the Tk file a thin view: widgets call these functions instead of
  interpreting adapter dictionaries inline, so the same formatting/validation
  rules cannot silently drift between pages.

Nothing here invents a value. A field that is not present in the adapter's
data stays ``None`` (rendered as "n/a" by the formatters); no default,
estimate or placeholder is substituted for missing data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TrainingConfig

#: A live run's status.json is treated as stale (the process likely died
#: without a clean shutdown, or was started outside this Control Center and
#: is no longer being tracked) when it claims to still be active but has not
#: published an update in this long.
STALE_STATUS_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _finite_number(value: Any) -> float | None:
    """`value` as a float when it is a real, finite number, else ``None``.

    Every formatter below funnels through this. NaN and +/-inf are *not*
    displayable quantities: ``json`` round-trips them happily (a diverged
    run writes ``NaN``/``Infinity`` into its own summaries, and
    ``json.loads`` reads them straight back), so they do reach the GUI.
    Rendering them as "nan" or "inf" would be an estimate of a value that
    does not exist, and ``int(round(inf))`` raises OverflowError, which
    took a whole page down. Both become "n/a" instead.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def format_number(value: Any, decimals: int = 0) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    if decimals:
        return f"{number:,.{decimals}f}"
    return f"{int(round(number)):,}"


def format_fraction_as_percent(value: Any, decimals: int = 1) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.{decimals}f}%"


def format_duration(seconds: Any) -> str:
    number = _finite_number(seconds)
    if number is None or number < 0:
        return "n/a"
    total = int(number)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(value: Any) -> str:
    amount = _finite_number(value)
    if amount is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(amount) < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PB"


def format_timestamp(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(number))
    except (OverflowError, OSError, ValueError):
        return "n/a"


def format_ascii_bar(fraction: Any, width: int = 10) -> str:
    """Render a compact ASCII HUD bar ``[████░░░░░░]`` for a ``[0, 1]`` fraction."""
    number = _finite_number(fraction)
    if number is None or width <= 0:
        return "[" + "·" * max(width, 1) + "]"
    clamped = max(0.0, min(1.0, number))
    filled = int(round(clamped * width))
    return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"


def estimate_training_duration(total_steps: Any, steps_per_second: Any) -> str | None:
    """Return formatted ETA from real measured ``steps_per_second``, or ``None``."""
    steps = _finite_number(total_steps)
    sps = _finite_number(steps_per_second)
    if steps is None or sps is None or steps <= 0.0 or sps <= 0.0:
        return None
    return format_duration(steps / sps)


def ppo_health_view(ppo_diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize PPO optimization health from real diagnostics."""
    if not ppo_diagnostics or not isinstance(ppo_diagnostics, dict):
        return {"status": "N/A", "healthy": None, "summary": "PPO diagnostics: n/a"}
    kl = _finite_number(ppo_diagnostics.get("approx_kl"))
    clip = _finite_number(ppo_diagnostics.get("clip_fraction"))
    ev = _finite_number(ppo_diagnostics.get("explained_variance"))
    ent = _finite_number(ppo_diagnostics.get("entropy"))
    alerts: list[str] = []
    if kl is not None and kl > 0.04:
        alerts.append("KL SPIKE")
    if clip is not None and clip > 0.35:
        alerts.append("HIGH CLIP")
    if ev is not None and ev < 0.0:
        alerts.append("VALUE DRIFT")
    if ent is not None and abs(ent) < 0.01:
        alerts.append("LOW ENTROPY")
    status = " / ".join(alerts) if alerts else "OPTIMAL"
    summary = (
        f"PPO [{status}] — KL={format_number(kl, 4)}, "
        f"clip={format_number(clip, 3)}, "
        f"EV={format_number(ev, 3)}, "
        f"entropy={format_number(ent, 3)}"
    )
    return {"status": status, "healthy": not alerts, "summary": summary}


# ---------------------------------------------------------------------------
# Chart support
# ---------------------------------------------------------------------------


def downsample_series(
    points: list[tuple[float, float]], max_points: int = 400
) -> list[tuple[float, float]]:
    """Stride-decimates ``points`` to at most ``max_points``.

    Charts already receive a bounded series from the adapter
    (``deque(maxlen=...)``); this is the second, presentation-side bound so
    a canvas never has to draw more points than it has pixels for. Always
    keeps the most recent point.
    """
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    stride = math.ceil(len(points) / max_points)
    sampled = list(points[::stride])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def dashboard_view(
    snapshot: dict[str, Any], telemetry: dict[str, Any] | None = None, now: float | None = None
) -> dict[str, Any]:
    """Builds the Dashboard's display model from `adapter.dashboard_snapshot()`
    (and optionally `adapter.telemetry_series()` for the freshest reward/PPO
    numbers). Every field defaults to ``None`` ("not available"); nothing is
    estimated here that the backend did not already compute."""
    now = time.time() if now is None else now
    latest_run = snapshot.get("latest_run")
    active_processes = snapshot.get("active_processes") or []
    view: dict[str, Any] = {
        "has_run": latest_run is not None,
        "run_id": None,
        "run_dir": None,
        "state": None,
        "timesteps": None,
        "target_timesteps": None,
        "progress_percent": None,
        "fps": None,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "episodes": None,
        "environment_count": None,
        "env_workers": None,
        "rollout_length": None,
        "ppo_updates": None,
        "device": None,
        "torch_threads": None,
        "reward": None,
        "win_rate": None,
        "loss_rate": None,
        "ppo_diagnostics": {},
        "latest_evaluation": None,
        "current_checkpoint": None,
        "best_checkpoint": None,
        "final_checkpoint": None,
        "warnings": [],
        "problems": [],
        "is_active_process": False,
        "stale": False,
    }
    if latest_run is None:
        return view

    control = dict(latest_run.get("control") or {})
    if telemetry and telemetry.get("available"):
        # The telemetry.jsonl row behind `latest` is written by the exact
        # same callback that publishes status.json and is at least as
        # fresh; it also carries fields the status merge does not keep
        # (e.g. this event's own elapsed_seconds).
        control.update(
            {key: value for key, value in telemetry.get("latest", {}).items() if value is not None}
        )

    config = latest_run.get("config") or {}
    checkpoints = latest_run.get("checkpoints") or {}
    evaluation = latest_run.get("evaluation") or {}
    progress = latest_run.get("progress") or {}

    view["run_id"] = (
        latest_run.get("run_id") or Path(str(latest_run.get("run_dir", ""))).name or None
    )
    view["run_dir"] = latest_run.get("run_dir")
    view["state"] = control.get("state") or (latest_run.get("status") or {}).get("state")
    view["timesteps"] = control.get("timesteps", progress.get("timesteps"))
    view["target_timesteps"] = control.get("total_training_steps", progress.get("target_timesteps"))
    timesteps, target = view["timesteps"], view["target_timesteps"]
    if isinstance(timesteps, (int, float)) and isinstance(target, (int, float)) and target:
        view["progress_percent"] = max(0.0, min(100.0, 100.0 * timesteps / target))
    view["fps"] = control.get("steps_per_second")
    view["elapsed_seconds"] = control.get("elapsed_seconds")
    view["eta_seconds"] = control.get("eta_seconds")
    view["episodes"] = control.get("episodes")
    view["environment_count"] = control.get("environment_count", config.get("environment_count"))
    view["env_workers"] = config.get("env_workers")
    view["rollout_length"] = config.get("resolved_rollout_length", config.get("rollout_length"))
    view["ppo_updates"] = control.get("ppo_n_updates")
    view["device"] = control.get("device", config.get("device"))
    view["torch_threads"] = config.get("resolved_torch_threads", config.get("torch_threads"))
    view["reward"] = control.get("mean_episode_reward")
    view["win_rate"] = control.get("win_rate")
    view["loss_rate"] = control.get("loss_rate")
    view["ppo_diagnostics"] = {
        key: control.get(f"ppo_{key}")
        for key in (
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "entropy",
            "value_loss",
            "loss",
        )
        if control.get(f"ppo_{key}") is not None
    }
    view["latest_evaluation"] = evaluation.get("latest")
    view["current_checkpoint"] = "checkpoints/latest.zip" if checkpoints.get("has_latest") else None
    view["best_checkpoint"] = evaluation.get("best")
    view["final_checkpoint"] = "final.zip" if checkpoints.get("has_final") else None
    view["warnings"] = list(latest_run.get("warnings") or [])
    view["problems"] = list(latest_run.get("problems") or [])
    view["is_active_process"] = any(
        process.get("run_dir") == latest_run.get("run_dir") for process in active_processes
    )

    updated_at = control.get("updated_at")
    if (
        view["state"] in {"Running", "Starting", "Stopping", "Paused"}
        and not view["is_active_process"]
        and isinstance(updated_at, (int, float))
        and (now - updated_at) > STALE_STATUS_SECONDS
    ):
        view["stale"] = True
        view["warnings"] = view["warnings"] + [
            f"status.json still says '{view['state']}' but no managed process is running and it has not "
            f"updated in {int(now - updated_at)}s; the process may have crashed or was started outside "
            "this Control Center session"
        ]
    return view


# ---------------------------------------------------------------------------
# Runs / checkpoints
# ---------------------------------------------------------------------------


def runs_table_rows(list_runs_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in list_runs_result.get("runs", []):
        progress = report.get("progress", {}) or {}
        checkpoints = report.get("checkpoints", {}) or {}
        evaluation = report.get("evaluation", {}) or {}
        config = report.get("config", {}) or {}
        latest_eval = evaluation.get("latest") or {}
        fraction = progress.get("fraction")
        rows.append(
            {
                "run_id": report.get("run_id"),
                "state": (report.get("status") or {}).get("state"),
                "timesteps": progress.get("timesteps"),
                "target_timesteps": progress.get("target_timesteps"),
                "progress_percent": fraction * 100.0
                if isinstance(fraction, (int, float))
                else None,
                "device": config.get("device"),
                "environment_count": config.get("environment_count"),
                "env_workers": config.get("env_workers"),
                "checkpoints": checkpoints.get("count"),
                "has_best": checkpoints.get("has_best"),
                "reward": latest_eval.get("mean_episode_reward"),
                "win_rate": latest_eval.get("win_rate"),
                "modified_utc": report.get("modified_utc"),
                "warning_count": len(report.get("warnings") or []),
                "run_dir": report.get("run_dir"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Agent process registry
# ---------------------------------------------------------------------------


def _process_progress_percent(backend: dict[str, Any]) -> float | None:
    timesteps = backend.get("timesteps")
    target = backend.get("total_training_steps")
    if isinstance(timesteps, (int, float)) and isinstance(target, (int, float)) and target:
        return max(0.0, min(100.0, 100.0 * timesteps / target))
    return None


def process_table_rows(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for process in processes:
        backend = process.get("backend") or {}
        meta = process.get("meta") or {}
        rows.append(
            {
                "id": process.get("id"),
                "kind": process.get("kind"),
                "run_id": meta.get("run_id") or Path(str(process.get("run_dir", ""))).name,
                "status": backend.get("state") or process.get("state"),
                "pid": process.get("pid"),
                "environment_count": backend.get(
                    "environment_count", meta.get("environment_count")
                ),
                "env_workers": meta.get("env_workers"),
                "progress_percent": _process_progress_percent(backend),
                "started_at": process.get("started_at"),
                "updated_at": backend.get("updated_at"),
                "error": process.get("error") or backend.get("error"),
                "run_dir": process.get("run_dir"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluation_view(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"available": False, "error": "no evaluation selected"}
    if summary.get("available") is False:
        return {"available": False, "error": summary.get("error", "evaluation summary unavailable")}
    pipeline = summary.get("action_pipeline") or {}
    return {
        "available": True,
        "path": summary.get("path"),
        "episodes": summary.get("episodes"),
        "timesteps": summary.get("timesteps"),
        "environment_count": summary.get("environment_count"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "episodes_per_second": summary.get("episodes_per_second"),
        "reward": summary.get("mean_episode_reward"),
        "outcomes": {
            "win_rate": summary.get("win_rate"),
            "loss_rate": summary.get("loss_rate"),
            "timeout_rate": summary.get("timeout_rate"),
        },
        "combat": {
            "mean_kills": summary.get("mean_kills"),
            "mean_deaths": summary.get("mean_deaths"),
            "mean_damage_dealt": summary.get("mean_damage_dealt"),
            "mean_damage_received": summary.get("mean_damage_received"),
            "mean_survival_time": summary.get("mean_survival_time"),
        },
        "accuracy": {
            "mean_accuracy": summary.get("mean_accuracy"),
            "mean_shots_fired": summary.get("mean_shots_fired"),
            "mean_shots_hit": summary.get("mean_shots_hit"),
            "mean_trigger_pulls": summary.get("mean_trigger_pulls"),
        },
        "action_head_diagnostics": {
            "policy_shoot_request_rate": summary.get("policy_shoot_request_rate"),
            "mean_policy_shoot_probability": summary.get("mean_policy_shoot_probability"),
            "localization": pipeline.get("localization"),
            "engine_trigger_pulls_reported": pipeline.get("engine_trigger_pulls_reported"),
            "engine_trigger_pulls": pipeline.get("engine_trigger_pulls"),
            "engine_shots_fired": pipeline.get("engine_shots_fired"),
            "request_delivery_rate": pipeline.get("request_delivery_rate"),
            # "discharge rate": of the requests that reached the engine,
            # how many actually fired a shot (vs. cooldown/reload/ammo
            # blocking the weapon). Same measured value as
            # action_audit.summarize_action_pipeline's fire_conversion_rate.
            "discharge_rate": pipeline.get("fire_conversion_rate"),
            "policy_action_counts": pipeline.get("policy_action_counts"),
        },
    }


def evaluation_comparison_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        view = evaluation_view(summary)
        if not view.get("available"):
            continue
        rows.append(
            {
                "path": view["path"],
                "timesteps": view["timesteps"],
                "episodes": view["episodes"],
                "reward": view["reward"],
                "win_rate": view["outcomes"]["win_rate"],
                "loss_rate": view["outcomes"]["loss_rate"],
                "timeout_rate": view["outcomes"]["timeout_rate"],
                "accuracy": view["accuracy"]["mean_accuracy"],
                "shoot_request_rate": view["action_head_diagnostics"]["policy_shoot_request_rate"],
                "discharge_rate": view["action_head_diagnostics"]["discharge_rate"],
                "localization": view["action_head_diagnostics"]["localization"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark_result_rows(
    results: list[dict[str, Any]], source: str | None = None
) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        resources = row.get("resources") or {}
        rows.append(
            {
                "source": source,
                "environments": row.get("environments"),
                "workers": row.get("workers"),
                "total_steps": row.get("total_steps"),
                "steps_per_second": row.get("steps_per_second"),
                "episodes_per_second": row.get("episodes_per_second"),
                "p50_ms": row.get("vector_step_latency_p50_ms"),
                "p95_ms": row.get("vector_step_latency_p95_ms"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "info_mode": row.get("info_mode"),
                "cpu_percent": resources.get("cpu_percent"),
                "gpu_utilization_percent": resources.get("gpu_utilization_percent"),
                "gpu_vram_used_mb": resources.get("gpu_vram_used_mb"),
            }
        )
    return rows


def benchmark_history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flattens every measured configuration across benchmark result sets
    into one comparison table, each row labelled by its source directory."""
    rows: list[dict[str, Any]] = []
    for entry in history:
        label = entry.get("directory")
        rows.extend(benchmark_result_rows(entry.get("results") or [], source=label))
    return rows


# ---------------------------------------------------------------------------
# Agent lifecycle registry
# ---------------------------------------------------------------------------

#: Operator-facing lifecycle states and how they are coloured. Derived
#: states only (see sandboxai.agents.derive_lifecycle); nothing here is a
#: measurement, so a missing state renders muted, never guessed.
LIFECYCLE_COLORS: dict[str, str] = {
    "AVAILABLE": "#4ee6a1",
    "LAUNCHING": "#ffc861",
    "RUNNING": "#4ee6a1",
    "PAUSED": "#ffc861",
    "STOPPING": "#ffc861",
    "STOPPED": "#91a7bf",
    "FINISHED": "#91a7bf",
    "FAILED": "#ff718d",
    "RESTARTING": "#ffc861",
}


def agent_table_rows(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One table row per agent view, straight from published facts.

    Environment/worker/device context comes from the launch metadata or the
    backend's own status — never re-derived here — and stays ``None``
    ("n/a") when the kind does not publish it.
    """
    rows: list[dict[str, Any]] = []
    for view in views:
        backend = view.get("backend") or {}
        meta = view.get("meta") or {}
        spec = view.get("spec") or {}
        config = spec.get("config") or {}
        rows.append(
            {
                "agent_id": view.get("agent_id"),
                "name": view.get("name") or view.get("agent_id"),
                "kind": view.get("kind"),
                "lifecycle": str(view.get("lifecycle", "")),
                "pid": view.get("pid"),
                "environment_count": backend.get("environment_count")
                or meta.get("environment_count")
                or config.get("environment_count"),
                "env_workers": meta.get("env_workers") or config.get("env_workers"),
                "device": backend.get("device") or meta.get("device") or config.get("device"),
                "timesteps": backend.get("timesteps"),
                "total_steps": backend.get("total_training_steps"),
                "steps_per_second": backend.get("steps_per_second"),
                "mean_episode_reward": backend.get("mean_episode_reward"),
                "episodes": backend.get("episodes"),
                "started_at": view.get("started_at") or view.get("created_at"),
                "error": view.get("error"),
                "run_dir": view.get("run_dir"),
            }
        )
    return rows


def agent_progress_percent(row: dict[str, Any]) -> float | None:
    timesteps = row.get("timesteps")
    total = row.get("total_steps")
    if isinstance(timesteps, (int, float)) and isinstance(total, (int, float)) and total:
        return max(0.0, min(100.0, 100.0 * timesteps / total))
    return None


def agent_action_availability(lifecycle: str, kind: str) -> dict[str, Any]:
    """Which lifecycle actions make sense for one agent right now.

    Pause/Resume exist only on the training backend's cooperative command
    protocol; for other kinds the reason says so instead of a dead button.
    """
    alive = lifecycle in ("LAUNCHING", "RUNNING", "PAUSED", "STOPPING", "RESTARTING")
    pausable = kind == "training"
    reason = None if pausable else f"the '{kind}' backend has no pause protocol"
    return {
        "pause": alive and lifecycle == "RUNNING" and pausable,
        "resume": alive and lifecycle == "PAUSED" and pausable,
        "stop": alive and lifecycle not in ("STOPPING", "RESTARTING"),
        "restart": lifecycle in ("RUNNING", "PAUSED", "STOPPED", "FINISHED", "FAILED"),
        "force_stop": alive,
        "remove": not alive,
        "pause_unsupported_reason": reason,
    }


def launch_slot_view(
    values: dict[str, str],
    compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The launch slot's state before any agent exists.

    ``AVAILABLE`` means the current form values parse, validate and are
    compatible with the runtime; ``INVALID`` carries every reason found.
    The summary is what a launch would use — the resolved worker count
    included, so "0 = auto" is shown as the concrete number of workers.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        config = parse_training_form(values)
    except ValueError as exc:
        errors.extend(part.strip() for part in str(exc).split(";") if part.strip())
        config = None
    if compatibility is not None:
        errors.extend(str(reason) for reason in compatibility.get("errors", []))
        warnings.extend(str(reason) for reason in compatibility.get("warnings", []))
    if config is None:
        return {"state": "INVALID", "errors": errors, "warnings": warnings, "summary": None}
    summary = {
        "environment_count": config.environment_count,
        "env_workers": config.resolved_env_workers(),
        "device": config.device,
        "total_training_steps": config.total_training_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "ppo_epochs": config.ppo_epochs,
        "resolved_rollout_length": config.resolved_rollout_length(),
        "entropy_coefficient": config.entropy_coefficient,
        "run_id": config.run_id or "",
    }
    return {
        "state": "INVALID" if errors else "AVAILABLE",
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def topology_rows(environment_count: Any, env_workers: Any) -> list[dict[str, Any]]:
    """The Environment -> Worker mapping for one launch or agent.

    Recomputes the same contiguous shard plan the sharded bridge will
    build (``sharded_env.plan_shards``), so what the operator sees is what
    the runtime does. Invalid inputs produce an empty list.
    """
    from .sharded_env import plan_shards

    if not isinstance(environment_count, int) or not isinstance(env_workers, int):
        return []
    if environment_count < 1 or env_workers < 1:
        return []
    return [
        {
            "worker": shard.worker,
            "first_environment": shard.offset,
            "last_environment": shard.stop - 1,
            "environments": shard.count,
        }
        for shard in plan_shards(environment_count, env_workers)
    ]


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


#: The phases the one-button Benchmark tab reports, in execution order.
#: The first five mirror the pipeline's own stages; "apply" is the tab's
#: final automatic step (persisting + activating the winning configuration).
BENCHMARK_PHASES: tuple[tuple[str, str], ...] = (
    ("discovery", "Discover runtime"),
    ("screening", "Screen env/worker grid"),
    ("devices", "Compare devices"),
    ("validation", "Validate finalists (real PPO)"),
    ("recommendation", "Pick best configuration"),
    ("apply", "Apply best configuration"),
)

_PHASE_MARKS = {
    "pending": "·",
    "active": "▶",
    "done": "✓",
    "skipped": "–",
    "failed": "✗",
}


def benchmark_workflow_view(
    *,
    running: bool,
    event: dict[str, Any] | None,
    report: dict[str, Any] | None,
    applied: bool | None = None,
) -> dict[str, Any]:
    """Display model for the zero-configuration benchmark workflow.

    Phase states are derived from what actually happened: the persisted
    stage records of ``report`` and, while ``running``, the latest progress
    ``event``. ``applied`` is the tab's own final step (None = not reached,
    True = the recommendation was activated, False = there was nothing to
    apply). Nothing is predicted; a phase that has not reported anything
    is simply "pending".
    """
    stage_status: dict[str, str] = {}
    for stage in (report or {}).get("stages", []):
        name = str(stage.get("name", ""))
        status = str(stage.get("status", "completed"))
        if status in ("completed", "ok"):
            stage_status[name] = "done"
        elif status == "skipped":
            stage_status[name] = "skipped"
        else:
            stage_status[name] = "failed"
    if report is not None and report.get("recommendation") is not None:
        stage_status.setdefault("recommendation", "done")
    elif report is not None and not running:
        stage_status.setdefault("recommendation", "failed")
    if applied is True:
        stage_status["apply"] = "done"
    elif applied is False:
        stage_status["apply"] = "skipped"
    active_stage = str(event.get("stage", "")) if (running and event) else ""
    phases: list[dict[str, str]] = []
    for key, label in BENCHMARK_PHASES:
        if key == active_stage and key not in stage_status:
            status = "active"
        else:
            status = stage_status.get(key, "pending")
        phases.append({"key": key, "label": label, "status": status})
    phase_line = "   ".join(f"{_PHASE_MARKS[phase['status']]} {phase['label']}" for phase in phases)
    if running:
        detail = pipeline_progress_view(event)["text"]
    elif report is not None:
        detail = (
            f"finished ({report.get('status', 'unknown')}, "
            f"{format_duration(report.get('elapsed_seconds'))})"
        )
    else:
        detail = "idle - press Start to measure this machine"
    return {"phases": phases, "phase_line": phase_line, "detail": detail}


def _format_pipeline_row(stage_name: str | None, row: dict[str, Any]) -> dict[str, Any]:
    resources = row.get("resources") or {}
    p50 = row.get("vector_step_latency_p50_ms")
    p95 = row.get("vector_step_latency_p95_ms")
    jitter = row.get("latency_jitter")
    p50_val = _finite_number(p50)
    p95_val = _finite_number(p95)
    if jitter is None and p50_val is not None and p95_val is not None and p50_val > 0.0:
        jitter = round(p95_val / p50_val, 3)
    return {
        "stage": stage_name or row.get("stage"),
        "status": row.get("status", "ok"),
        "environments": row.get("environments"),
        "workers": row.get("workers"),
        "device": row.get("device"),
        "steps": row.get("steps") or row.get("total_steps"),
        "steps_per_second": row.get("steps_per_second"),
        "speedup": row.get("speedup_vs_baseline"),
        "episodes_per_second": row.get("episodes_per_second"),
        "p50_ms": p50,
        "p95_ms": p95,
        "jitter": jitter,
        "startup_seconds": row.get("startup_seconds"),
        "elapsed_seconds": row.get("elapsed_seconds") or row.get("wall_seconds"),
        "cpu_percent": resources.get("cpu_percent") or row.get("host_cpu_percent_mean"),
        "bottleneck": row.get("bottleneck"),
        "error": row.get("error"),
    }


def _enrich_pipeline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Populate speedup and bottleneck for any rows that do not yet carry them."""
    measured = [
        r
        for r in rows
        if r.get("status", "ok") in ("ok", "measured")
        and _finite_number(r.get("steps_per_second")) is not None
        and float(r["steps_per_second"]) > 0.0
    ]
    if not measured:
        return rows
    smallest = min(
        measured,
        key=lambda r: (
            int(_finite_number(r.get("environments")) or 1),
            int(_finite_number(r.get("workers")) or 1),
        ),
    )
    baseline_sps = float(smallest["steps_per_second"])
    peak_sps = max(float(r["steps_per_second"]) for r in measured)
    for row in rows:
        sps = _finite_number(row.get("steps_per_second"))
        if row.get("speedup") is None and sps is not None and sps > 0.0 and baseline_sps > 0.0:
            row["speedup"] = round(sps / baseline_sps, 2)
        if not row.get("bottleneck"):
            jitter = _finite_number(row.get("jitter"))
            if row.get("status", "ok") not in ("ok", "measured"):
                row["bottleneck"] = str(row.get("status", "failed"))
            elif jitter is not None and jitter > 4.0:
                row["bottleneck"] = "jitter-bound"
            elif sps is not None and peak_sps > 0.0 and sps >= peak_sps * 0.90:
                row["bottleneck"] = "optimal"
            else:
                row["bottleneck"] = "scaling"
    return rows


def benchmark_pipeline_rows(
    report: dict[str, Any] | None,
    live_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten one pipeline report's measured configurations into table rows.

    Screening and validation rows are labelled by stage; a failed or
    skipped configuration keeps its status and error so the table shows
    what actually happened, not a green wash. While a benchmark is running,
    ``live_rows`` supplies the already-completed measurements in real time.
    """
    if report:
        rows: list[dict[str, Any]] = []
        for stage in report.get("stages", []):
            for row in stage.get("configurations", []):
                rows.append(_format_pipeline_row(stage.get("name"), row))
        return _enrich_pipeline_rows(rows)
    if live_rows:
        return _enrich_pipeline_rows(
            [_format_pipeline_row(row.get("stage"), row) for row in live_rows]
        )
    return []


def benchmark_recommendation_view(recommendation: dict[str, Any] | None) -> dict[str, Any]:
    """Display model for the Recommended Configuration card.

    ``available`` is False when no recommendation exists (never run, or
    nothing could be measured); the UI then shows the reason instead of a
    fabricated configuration.
    """
    if not recommendation or not isinstance(recommendation.get("environment_count"), int):
        return {"available": False, "reason": "no benchmark recommendation yet"}
    return {
        "available": True,
        "environment_count": recommendation.get("environment_count"),
        "env_workers": recommendation.get("env_workers"),
        "device": recommendation.get("device"),
        "inference_device": recommendation.get("inference_device"),
        "expected_steps_per_second": recommendation.get("expected_steps_per_second"),
        "basis": recommendation.get("basis"),
        "created_utc": recommendation.get("created_utc"),
        "applied_utc": recommendation.get("applied_utc"),
        "source_report": recommendation.get("source_report"),
        "rationale": list(recommendation.get("rationale") or []),
        "warnings": list(recommendation.get("warnings") or []),
        "summary": (
            f"{recommendation.get('environment_count')} environments / "
            f"{recommendation.get('env_workers')} workers on "
            f"{recommendation.get('device')} — "
            f"{format_number(recommendation.get('expected_steps_per_second'), 1)} steps/s"
        ),
    }


def pipeline_progress_view(event: dict[str, Any] | None) -> dict[str, Any]:
    """Display model for the live progress line of a running pipeline."""
    if not event:
        return {"text": "idle", "stage": None, "fraction": None}
    stage = event.get("stage")
    index = event.get("index")
    total = event.get("total")
    configuration = event.get("configuration") or {}
    live = event.get("live") or {}
    label = ""
    if configuration:
        label = (
            f" — {configuration.get('environments', '?')} envs / "
            f"{configuration.get('workers', '?')} workers"
        )
    if isinstance(index, int) and isinstance(total, int) and total:
        sub_fraction = 0.0
        if event.get("status") == "completed":
            sub_fraction = 1.0
        elif live:
            elapsed = _finite_number(live.get("elapsed_seconds")) or 0.0
            cap = _finite_number(live.get("max_seconds_per_config")) or 0.0
            done_steps = _finite_number(live.get("completed_steps")) or 0.0
            target_steps = _finite_number(live.get("target_steps")) or 0.0
            time_frac = (elapsed / cap) if cap > 0.0 else 0.0
            step_frac = (done_steps / target_steps) if (0.0 < target_steps < 10**8) else 0.0
            sub_fraction = min(0.99, max(time_frac, step_frac))
        fraction = (
            ((index + sub_fraction) / total)
            if event.get("status") == "running"
            else ((index + 1) / total)
        )
        text = f"{stage}: {index + 1}/{total}{label} ({event.get('status', '')})"
        if live and live.get("steps_per_second") is not None:
            text += (
                f" • {format_number(live.get('steps_per_second'), 1)} steps/s "
                f"({format_number(live.get('total_steps'))} steps)"
            )
    else:
        fraction = None
        message = event.get("message") or event.get("status", "")
        text = f"{stage}: {message}"
    return {"text": text, "stage": stage, "fraction": fraction}


def _extract_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    measured = [
        row
        for row in rows
        if row.get("status", "ok") in ("ok", "measured")
        and _finite_number(row.get("steps_per_second")) is not None
        and float(row["steps_per_second"]) > 0.0
    ]
    if not measured:
        return None
    return max(measured, key=lambda item: float(item["steps_per_second"]))


def _format_live_benchmark_cards(
    *,
    running: bool,
    event: dict[str, Any] | None,
    report: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    best_row: dict[str, Any] | None,
) -> dict[str, Any]:
    live = (event or {}).get("live") or {}
    config = (event or {}).get("configuration") or {}
    rec = (report or {}).get("recommendation") if report else None

    # Active topology
    if running and config:
        envs = config.get("environments", "?")
        workers = config.get("workers", "?")
        dev = config.get("device") or "bridge"
        active_config = f"{envs} envs / {workers} w ({dev})"
    elif rec and isinstance(rec.get("environment_count"), int):
        active_config = (
            f"{rec['environment_count']} envs / {rec.get('env_workers', 1)} w "
            f"({rec.get('device', 'cpu')})"
        )
    elif best_row:
        active_config = f"{best_row.get('environments')} envs / {best_row.get('workers')} w"
    else:
        active_config = "n/a"

    # One clearly labelled throughput value: bridge steps per second.
    steps_per_second = _finite_number(
        live.get("steps_per_second")
        or (event or {}).get("steps_per_second")
        or (rec or {}).get("expected_steps_per_second")
        or (best_row or {}).get("steps_per_second")
    )

    # Live Steps
    if running and live.get("total_steps") is not None:
        live_steps_text = (
            f"{format_number(live.get('total_steps'))} "
            f"({format_number(live.get('completed_steps'))}/env)"
        )
    elif running and config.get("steps") is not None:
        live_steps_text = f"target {format_number(config.get('steps'))}"
    elif rows:
        total_measured = sum(
            int(r["steps"]) for r in rows if _finite_number(r.get("steps")) is not None
        )
        live_steps_text = format_number(total_measured) if total_measured > 0 else "n/a"
    else:
        live_steps_text = "n/a"

    # Latency & Jitter
    p50 = _finite_number(
        live.get("vector_step_latency_p50_ms")
        or (rec or {}).get("latency_p50_ms")
        or (best_row or {}).get("p50_ms")
    )
    p95 = _finite_number(
        live.get("vector_step_latency_p95_ms")
        or (rec or {}).get("latency_p95_ms")
        or (best_row or {}).get("p95_ms")
    )
    latency_text = (
        f"{format_number(p50, 2)} / {format_number(p95, 2)} ms"
        if (p50 is not None and p95 is not None)
        else "n/a"
    )
    jitter = _finite_number(live.get("latency_jitter") or (best_row or {}).get("jitter"))
    if jitter is None and p50 is not None and p95 is not None and p50 > 0.0:
        jitter = p95 / p50
    jitter_stable = (jitter <= 4.0) if jitter is not None else None
    jitter_text = (
        f"{format_number(jitter, 2)}x ({'STABLE' if jitter_stable else 'HIGH JITTER'})"
        if jitter is not None
        else "n/a"
    )

    # Elapsed & Host CPU
    elapsed_val = _finite_number(
        live.get("elapsed_seconds")
        or (event or {}).get("stage_elapsed_seconds")
        or (report or {}).get("elapsed_seconds")
    )
    resources = live.get("resources") or {}
    cpu_pct = _finite_number(resources.get("cpu_percent") or (best_row or {}).get("cpu_percent"))
    elapsed_str = format_duration(elapsed_val)
    if cpu_pct is not None and elapsed_str != "n/a":
        elapsed_host_text = f"{elapsed_str} • CPU {format_number(cpu_pct, 0)}%"
    elif cpu_pct is not None:
        elapsed_host_text = f"CPU {format_number(cpu_pct, 0)}%"
    else:
        elapsed_host_text = elapsed_str

    return {
        "active_config": active_config,
        "steps_per_second": steps_per_second,
        "live_phase": live.get("phase"),
        "steps_per_second_text": (
            f"{format_number(steps_per_second, 1)} steps/s"
            if steps_per_second is not None
            else "n/a"
        ),
        "live_steps": _finite_number(live.get("total_steps"))
        if running and live.get("total_steps") is not None
        else (
            sum(int(r["steps"]) for r in rows if _finite_number(r.get("steps")) is not None)
            if rows
            else None
        ),
        "steps_per_env": _finite_number(live.get("completed_steps")) if running else None,
        "live_steps_text": live_steps_text,
        "p50_ms": p50,
        "p95_ms": p95,
        "latency_text": latency_text,
        "jitter": jitter,
        "jitter_text": jitter_text,
        "jitter_stable": jitter_stable,
        "elapsed_seconds": elapsed_val,
        "cpu_percent": cpu_pct,
        "elapsed_host_text": elapsed_host_text,
    }


def benchmark_live_telemetry_view(
    *,
    running: bool,
    event: dict[str, Any] | None,
    report: dict[str, Any] | None,
    applied: bool | None = None,
) -> dict[str, Any]:
    """Full live HUD model for the Benchmark tab.

    Combines stage progression, real-time steps/s, latency and jitter readouts,
    the current leading configuration, streaming table rows and throughput
    chart coordinates without inventing a single number.
    """
    workflow = benchmark_workflow_view(running=running, event=event, report=report, applied=applied)
    live_rows = (event or {}).get("completed_rows") if running else None
    rows = benchmark_pipeline_rows(report, live_rows=live_rows)
    best_row = _extract_best_row(rows)
    cards = _format_live_benchmark_cards(
        running=running, event=event, report=report, rows=rows, best_row=best_row
    )
    prog = pipeline_progress_view(event)
    raw_idx = (event or {}).get("index")
    raw_tot = (event or {}).get("total")
    if running:
        stage_label = str((event or {}).get("stage") or "starting").upper()
        stage_status_text = (
            f"{stage_label} ({int(raw_idx) + 1}/{int(raw_tot)})"
            if isinstance(raw_idx, int) and isinstance(raw_tot, int) and raw_tot
            else stage_label
        )
        fraction = float(prog["fraction"]) if prog.get("fraction") is not None else 0.05
    elif report is not None:
        stage_label = str(report.get("status", "completed")).upper()
        stage_status_text = stage_label
        fraction = 1.0 if report.get("status") == "completed" else 0.0
    else:
        stage_label = "IDLE"
        stage_status_text = "IDLE"
        fraction = 0.0

    if running and best_row is not None:
        speedup_val = _finite_number(best_row.get("speedup"))
        speedup_suffix = f", {format_number(speedup_val, 2)}x speedup" if speedup_val else ""
        leader_summary = (
            f"{best_row.get('environments')} environments / "
            f"{best_row.get('workers')} workers — "
            f"{format_number(best_row.get('steps_per_second'), 1)} steps/s "
            f"(p50 {format_number(best_row.get('p50_ms'), 2)} ms, "
            f"jitter {format_number(best_row.get('jitter'), 2)}x{speedup_suffix})"
        )
        leader_text = f"LEADING SO FAR: {leader_summary}"
    elif report and report.get("recommendation"):
        rec = report.get("recommendation") or {}
        rec_view = benchmark_recommendation_view(rec)
        speedup_val = _finite_number(
            rec.get("speedup_vs_baseline") or (best_row or {}).get("speedup")
        )
        speedup_suffix = (
            f" ({format_number(speedup_val, 2)}x speedup vs baseline)" if speedup_val else ""
        )
        leader_summary = f"{rec_view['summary']}{speedup_suffix}"
        leader_text = f"WINNER SELECTED: {leader_summary}"
    else:
        leader_summary = "awaiting benchmark telemetry"
        leader_text = ""

    chart_points: list[tuple[float, float]] = []
    for idx_row, row in enumerate(rows, start=1):
        sps = _finite_number(row.get("steps_per_second"))
        if sps is not None and sps > 0.0:
            chart_points.append((float(idx_row), sps))
    if (
        running
        and cards["steps_per_second"] is not None
        and (event or {}).get("status") == "running"
    ):
        chart_points.append((float(len(chart_points) + 1), float(cards["steps_per_second"])))

    return {
        **workflow,
        **cards,
        "stage_label": stage_label,
        "index": (int(raw_idx) + 1) if isinstance(raw_idx, int) else None,
        "total": int(raw_tot) if isinstance(raw_tot, int) else None,
        "stage_status_text": stage_status_text,
        "progress_fraction": max(0.0, min(1.0, fraction)),
        "leader_summary": leader_summary,
        "leader_text": leader_text,
        "rows": rows,
        "best_row_key": (
            (best_row.get("environments"), best_row.get("workers")) if best_row else None
        ),
        "chart_points": chart_points,
    }


# ---------------------------------------------------------------------------
# Hardware wizard
# ---------------------------------------------------------------------------


def hardware_measurement_rows(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One display row per measured device candidate.

    Throughput is shown only for candidates that were actually measured;
    everything else shows its status (unavailable / failed / cancelled) and
    never a fabricated number, matching the measurement layer's own honesty.
    """
    rows: list[dict[str, Any]] = []
    for measurement in measurements:
        status = measurement.get("status")
        measured = status == "measured" and measurement.get("steps_per_second") is not None
        rows.append(
            {
                "label": measurement.get("label"),
                "status": status,
                "steps_per_second": (
                    format_number(measurement.get("steps_per_second"), 1) if measured else "n/a"
                ),
                "wall": (format_duration(measurement.get("wall_seconds")) if measured else "n/a"),
                "detail": measurement.get("error") or ("measured" if measured else status),
            }
        )
    return rows


def hardware_profile_view(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Presentation view of a persisted hardware profile.

    Returns ``available: False`` when the wizard has not run yet, so the
    Settings page can offer to start it. Permanently-unavailable fields are
    left out entirely rather than rendered as ``n/a`` (Phase 2: hide fields
    this machine can never fill).
    """
    if not profile:
        return {"available": False, "note": "No hardware profile yet — run the wizard."}
    view: dict[str, Any] = {
        "available": True,
        "selected_device": profile.get("selected_device"),
        "device": profile.get("device"),
        "inference_device": profile.get("inference_device"),
        "measurement_steps": profile.get("measurement_steps"),
        "fallback": bool(profile.get("fallback")),
        "note": profile.get("note", ""),
        "godot_available": bool(profile.get("godot_available")),
        "measurements": hardware_measurement_rows(profile.get("measurements") or []),
    }
    host = profile.get("host") or {}
    # Only surface accelerator facts that exist on this host: a CPU-only
    # machine should not show a blank CUDA line.
    if host.get("cuda_available"):
        view["cuda_device"] = host.get("cuda_device")
    return view


# ---------------------------------------------------------------------------
# Training form: basic vs. advanced, validated before launch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingFieldSpec:
    name: str
    label: str
    kind: str  # "int" | "float" | "str" | "bool" | "choice"
    group: str  # "basic" | "advanced"
    choices: tuple[str, ...] = ()
    help: str = ""


TRAINING_FIELDS: tuple[TrainingFieldSpec, ...] = (
    # --- Basic: the five core controls shown in the Training launch deck. ---
    TrainingFieldSpec(
        "environment_count",
        "Environments",
        "int",
        "basic",
        help="Parallel Godot environments simulated.",
    ),
    TrainingFieldSpec(
        "env_workers",
        "Godot workers",
        "int",
        "basic",
        help="Bridge processes hosting the environments. 0 = auto.",
    ),
    TrainingFieldSpec(
        "total_training_steps",
        "Total timesteps",
        "int",
        "basic",
        help="Total environment transitions collected during training.",
    ),
    TrainingFieldSpec(
        "device",
        "Device",
        "choice",
        "basic",
        choices=("auto", "cpu", "cuda"),
        help="Training compute target.",
    ),
    TrainingFieldSpec(
        "curriculum_mode",
        "Curriculum mode",
        "choice",
        "basic",
        choices=("auto", "fixed"),
        help="Automatic progression vs. fixed stage.",
    ),
    # --- Internal / programmatic defaults (not shown in the GUI deck). ----
    TrainingFieldSpec(
        "max_train_minutes",
        "Max train minutes",
        "float",
        "advanced",
        help=(
            "Trainer-enforced wall-clock budget; the Budget row owns this value. "
            "0 = run to the step count only."
        ),
    ),
    TrainingFieldSpec(
        "run_id", "Run ID", "str", "advanced", help="Blank = timestamped automatically."
    ),
    TrainingFieldSpec(
        "experiment_id",
        "Experiment ID",
        "str",
        "advanced",
        help="Optional grouping prefix for the run directory.",
    ),
    TrainingFieldSpec(
        "curriculum_level",
        "Fixed curriculum level",
        "int",
        "advanced",
        help="Used only when curriculum mode is 'fixed'.",
    ),
    TrainingFieldSpec("curriculum_start_level", "Auto curriculum start level", "int", "advanced"),
    TrainingFieldSpec("adaptive_curriculum", "Adaptive curriculum", "bool", "advanced"),
    TrainingFieldSpec(
        "rollout_length",
        "Rollout length",
        "int",
        "advanced",
        help="Per-environment horizon. 0 = auto-sized.",
    ),
    TrainingFieldSpec("batch_size", "Batch size", "int", "advanced"),
    TrainingFieldSpec("ppo_epochs", "PPO epochs", "int", "advanced"),
    TrainingFieldSpec("learning_rate", "Learning rate", "float", "advanced"),
    TrainingFieldSpec("gamma", "Discount (gamma)", "float", "advanced"),
    TrainingFieldSpec("gae_lambda", "GAE lambda", "float", "advanced"),
    TrainingFieldSpec("entropy_coefficient", "Entropy coefficient", "float", "advanced"),
    TrainingFieldSpec("clip_range", "PPO clip range", "float", "advanced"),
    TrainingFieldSpec(
        "torch_threads", "Torch CPU threads", "int", "advanced", help="0 = bounded auto."
    ),
    TrainingFieldSpec(
        "inference_device",
        "Inference device",
        "choice",
        "advanced",
        choices=("auto", "cpu", "cuda"),
    ),
    TrainingFieldSpec("checkpoint_frequency", "Checkpoint every N steps", "int", "advanced"),
    TrainingFieldSpec("evaluation_frequency", "Evaluate every N steps", "int", "advanced"),
    TrainingFieldSpec("evaluation_episodes", "Evaluation episodes", "int", "advanced"),
    TrainingFieldSpec(
        "checkpoint_selection_metric", "Checkpoint selection metric", "str", "advanced"
    ),
    TrainingFieldSpec(
        "checkpoint_selection_goal",
        "Checkpoint selection goal",
        "choice",
        "advanced",
        choices=("max", "min"),
    ),
    TrainingFieldSpec("enemy_count", "Enemy count", "int", "advanced"),
    TrainingFieldSpec("seed", "Seed", "int", "advanced"),
    TrainingFieldSpec(
        "bc_checkpoint",
        "BC warm-start checkpoint",
        "str",
        "advanced",
        help="Optional .pt behavior-cloning checkpoint.",
    ),
    TrainingFieldSpec(
        "godot_executable",
        "Godot executable",
        "str",
        "advanced",
        help="Blank uses the configured/remembered default.",
    ),
    TrainingFieldSpec("profile_training", "Enable wall-clock profiling", "bool", "advanced"),
)

_TRAINING_DEFAULTS = TrainingConfig()


def default_training_values() -> dict[str, str]:
    """String defaults for every field in :data:`TRAINING_FIELDS`, read
    straight from ``TrainingConfig()`` so the form can never drift from the
    dataclass's own defaults."""
    values: dict[str, str] = {}
    for spec in TRAINING_FIELDS:
        default = getattr(_TRAINING_DEFAULTS, spec.name, "")
        if spec.kind == "bool":
            values[spec.name] = "true" if default else "false"
        else:
            values[spec.name] = "" if default is None else str(default)
    return values


def training_values_from_profile(profile: dict[str, Any] | None) -> dict[str, str]:
    """Training-form defaults with the hardware profile's device applied.

    Starts from :func:`default_training_values` and overlays the persisted
    profile's ``device``/``inference_device`` (the concrete pair its
    ``config_overrides`` recommends) so the Training page opens with the
    wizard's measured choice already selected. A fallback profile — one
    where nothing could be measured — is ignored, because it carries no
    measured preference worth imposing on the form.
    """
    values = default_training_values()
    if not profile or profile.get("fallback"):
        return values
    device = profile.get("device")
    inference = profile.get("inference_device")
    if isinstance(device, str) and device:
        values["device"] = device
    if isinstance(inference, str) and inference:
        values["inference_device"] = inference
    return values


def parse_training_form(values: dict[str, str]) -> TrainingConfig:
    """Turns raw Training-page form strings into a validated TrainingConfig.

    Type coercion is the only thing done here; every domain rule (rollout
    vs. batch-size compatibility, PPO epoch bounds, curriculum ranges, ...)
    stays owned by :meth:`TrainingConfig.validate`, so the GUI can never
    accept a combination the CLI/trainer would reject - or the reverse.
    Raises ``ValueError`` with every problem found, so the caller can show
    one clear message instead of the first exception encountered.
    """
    known_names = {spec.name for spec in TRAINING_FIELDS}
    parsed: dict[str, Any] = {}
    errors: list[str] = []
    for spec in TRAINING_FIELDS:
        raw = values.get(spec.name)
        if raw is None:
            continue
        text = str(raw).strip()
        if text == "":
            # Let the dataclass default apply. This matters even for
            # string fields: e.g. a blank "Godot executable" must fall
            # back to TrainingConfig's own "godot" default/resolution
            # chain, not literally become an empty executable path.
            continue
        try:
            if spec.kind == "int":
                parsed[spec.name] = int(text)
            elif spec.kind == "float":
                parsed[spec.name] = float(text)
            elif spec.kind == "bool":
                parsed[spec.name] = text.lower() in ("1", "true", "yes", "on")
            elif spec.kind == "choice" and spec.choices and text not in spec.choices:
                errors.append(f"{spec.label}: '{text}' must be one of {', '.join(spec.choices)}")
                continue
            else:
                parsed[spec.name] = text
        except ValueError:
            errors.append(f"{spec.label}: '{raw}' is not a valid {spec.kind}")
    # Anything the form passed that is not a known TrainingConfig field is a
    # programming error in the GUI, not a user mistake - fail loudly here
    # instead of TrainingConfig.from_dict() ignoring it silently.
    unknown = set(values) - known_names
    if unknown:
        errors.append(f"unknown training fields: {', '.join(sorted(unknown))}")
    if errors:
        raise ValueError("; ".join(errors))
    config = TrainingConfig.from_dict(parsed)
    config.validate()
    return config


def training_field_groups() -> dict[str, list[TrainingFieldSpec]]:
    groups: dict[str, list[TrainingFieldSpec]] = {"basic": [], "advanced": []}
    for spec in TRAINING_FIELDS:
        groups.setdefault(spec.group, []).append(spec)
    return groups


def launch_field_specs() -> list[TrainingFieldSpec]:
    """The streamlined fields exposed on the Training launch deck."""
    return training_field_groups()["basic"]


#: Curated PPO hyperparameter profiles for maximum training efficiency:
#: - standard: balanced SB3 defaults (LR 3e-4, 256 batch, 10 epochs)
#: - fast_convergence: linear-scaled LR (5e-4) + 512 minibatch + 6 epochs for
#:   ~40% faster PPO update wall-time on multi-worker rollouts
#: - max_turbo: aggressive early exploration (LR 8e-4, 512 batch, 4 epochs,
#:   entropy 0.02) for rapid initial policy climb
#: - fine_tune: low-KL late-stage refinement (LR 1e-4, 256 batch, 8 epochs,
#:   entropy 0.005) when resuming best_eval.zip or latest.zip
PPO_EFFICIENCY_PRESETS: dict[str, dict[str, str]] = {
    "standard": {
        "label": "Standard (LR 3e-4)",
        "learning_rate": "0.0003",
        "batch_size": "256",
        "ppo_epochs": "10",
        "entropy_coefficient": "0.01",
        "rollout_length": "0",
    },
    "fast_convergence": {
        "label": "Fast Climb (LR 5e-4 · 6 Ep)",
        "learning_rate": "0.0005",
        "batch_size": "512",
        "ppo_epochs": "6",
        "entropy_coefficient": "0.015",
        "rollout_length": "0",
    },
    "max_turbo": {
        "label": "Max Efficiency (LR 8e-4 · 4 Ep)",
        "learning_rate": "0.0008",
        "batch_size": "512",
        "ppo_epochs": "4",
        "entropy_coefficient": "0.02",
        "rollout_length": "0",
    },
    "fine_tune": {
        "label": "Fine-Tune Peak (LR 1e-4)",
        "learning_rate": "0.0001",
        "batch_size": "256",
        "ppo_epochs": "8",
        "entropy_coefficient": "0.005",
        "rollout_length": "0",
    },
}


def optimizer_field_specs() -> list[TrainingFieldSpec]:
    """PPO optimizer & rollout efficiency fields exposed in the Training deck."""
    wanted = (
        "learning_rate",
        "batch_size",
        "ppo_epochs",
        "entropy_coefficient",
        "rollout_length",
    )
    by_name = {spec.name: spec for spec in TRAINING_FIELDS}
    return [by_name[name] for name in wanted if name in by_name]


def recommend_ppo_hyperparameters(
    environment_count: int = 8,
    env_workers: int = 2,
    device: str = "auto",
    total_steps: int = 100_000,
) -> dict[str, Any]:
    """Compute topology-aware PPO hyperparameters using square-root batch scaling.

    Wider environment topologies collect lower-variance gradient estimates per
    rollout, allowing a proportionally higher learning rate (`3e-4 * sqrt(scale)`)
    and larger minibatches with fewer optimizer epochs per rollout — maximizing
    both sample efficiency and wall-clock steps/second.
    """
    envs = max(1, int(environment_count))
    workers = max(1, min(envs, int(env_workers or 1)))
    steps = max(1_000, int(total_steps or 100_000))
    if envs >= 32:
        batch_size = 1024 if device == "cuda" else 512
        ppo_epochs = 4 if device == "cpu" else 6
        lr = 0.00075 if steps <= 150_000 else 0.0006
        entropy = 0.015
    elif envs >= 12:
        batch_size = 512
        ppo_epochs = 6
        lr = 0.00055 if steps <= 150_000 else 0.00045
        entropy = 0.015
    elif envs >= 4:
        batch_size = 256
        ppo_epochs = 8
        lr = 0.0004
        entropy = 0.012
    else:
        batch_size = 256
        ppo_epochs = 10
        lr = 0.0003
        entropy = 0.01
    config = TrainingConfig(
        environment_count=envs,
        env_workers=workers,
        batch_size=batch_size,
        rollout_length=0,
    )
    resolved_rollout = config.resolved_rollout_length()
    rollout_transitions = envs * resolved_rollout
    return {
        "learning_rate": f"{lr:g}",
        "batch_size": str(batch_size),
        "ppo_epochs": str(ppo_epochs),
        "entropy_coefficient": f"{entropy:g}",
        "rollout_length": "0",
        "resolved_rollout_length": resolved_rollout,
        "rollout_transitions": rollout_transitions,
        "rationale": (
            f"Auto-tuned for {envs}e/{workers}w ({rollout_transitions} transitions/rollout): "
            f"LR={lr:g}, batch={batch_size}, epochs={ppo_epochs}, entropy={entropy:g}"
        ),
    }


# ---------------------------------------------------------------------------
# Roblox TTK Testing Live Bridge & Calibration viewmodel
# ---------------------------------------------------------------------------


def ttk_testing_view(status: dict[str, Any] | None) -> dict[str, Any]:
    """Display model for the Roblox TTK Testing live bridge & calibration view."""
    if not status or not isinstance(status, dict):
        return {
            "available": False,
            "connected": False,
            "roblox_running": False,
            "status_badge": "ROBLOX // OFFLINE",
            "launcher_text": "n/a",
            "window_text": "no window",
            "place_text": "n/a",
            "rows": [],
            "verified_count": 0,
            "calibrated_count": 0,
            "pending_count": 0,
        }
    live = status.get("live_session") or {}
    calibration = status.get("calibration") or {}
    mechanics_groups = status.get("mechanics") or {}

    running = bool(live.get("running"))
    in_ttk = bool(live.get("in_ttk_testing"))
    connected = bool(live.get("ttk_session_active"))

    if connected:
        badge = f"Connected to TTK Testing (PID {live.get('pid') or '?'})"
    elif running:
        badge = f"Roblox running (PID {live.get('pid') or '?'}) — open TTK Testing"
    else:
        badge = "Roblox offline (click Launch)"

    launcher_text = (
        str(live.get("resolved_launcher"))
        if live.get("launcher_found")
        else f"shortcut not found ({live.get('configured_shortcut') or 'default'})"
    )
    if live.get("window_found"):
        focus_str = "focused" if live.get("window_focused") else "background"
        window_text = f"{live.get('window_width')}x{live.get('window_height')} ({focus_str})"
    elif running:
        window_text = "process active (headless / minimized)"
    else:
        window_text = "closed"

    detected_place = live.get("detected_place_id")
    if in_ttk:
        place_text = f"{detected_place} (TTK Testing ✓)"
    elif detected_place:
        place_text = f"{detected_place} (other experience)"
    else:
        place_text = f"target {live.get('place_id', '120189115846709')}"

    rows: list[dict[str, Any]] = []
    verified_count = 0
    calibrated_count = 0
    pending_count = 0
    for group_key in ("verified", "calibration_required", "excluded"):
        for item in mechanics_groups.get(group_key, []):
            mech = str(item.get("mechanic") or "")
            cal_entry = calibration.get(mech) or {}
            measured_val = str(cal_entry.get("measured_value") or "").strip()
            if group_key == "verified":
                verified_count += 1
                effective_status = "VERIFIED (CALIBRATED)" if measured_val else "VERIFIED"
            elif group_key == "calibration_required":
                if measured_val:
                    calibrated_count += 1
                    effective_status = "CALIBRATED"
                else:
                    pending_count += 1
                    effective_status = "NEEDS CALIBRATION"
            else:
                effective_status = "EXCLUDED"
            rows.append(
                {
                    "mechanic": mech,
                    "base_status": group_key,
                    "status": effective_status,
                    # An empty cell beats a bare dash: a wall of "—" reads
                    # like a broken table, and this column is the one the
                    # operator fills in. The page keeps dashes out of the
                    # form field by testing the raw value.
                    "measured_value": measured_val,
                    "rule": str(item.get("implementation_rule") or ""),
                    "source": str(item.get("source_label") or ""),
                    "notes": str(cal_entry.get("notes") or item.get("notes") or ""),
                    "evidence_path": str(cal_entry.get("evidence_path") or ""),
                }
            )
    total_calibratable = max(1, calibrated_count + pending_count)
    cal_pct = round(100.0 * calibrated_count / total_calibratable, 1)
    cal_bar = format_ascii_bar(calibrated_count / total_calibratable, 8)
    progress_text = f"{cal_bar} {calibrated_count}/{total_calibratable} ({cal_pct:.0f}%)"
    return {
        "available": True,
        "connected": connected,
        "roblox_running": running,
        "status_badge": badge,
        "launcher_text": launcher_text,
        "window_text": window_text,
        "place_text": place_text,
        "log_file": live.get("log_file"),
        "rss_mb": live.get("rss_mb"),
        "rows": rows,
        "verified_count": verified_count,
        "calibrated_count": calibrated_count,
        "pending_count": pending_count,
        "calibration_progress_pct": cal_pct,
        "calibration_progress_text": progress_text,
        "studio_summary": "Sable Digital (PoptartNoahh & CanyonJack) | Universe 10090256806 | 8P FFA [MAP VOTING]",
        "recent_screenshots": list(status.get("recent_screenshots") or []),
        "presets": dict(status.get("presets") or {}),
    }


# ---------------------------------------------------------------------------
# Ubuntu CPU Performance Turbo, Training Convergence & Tactical Combat Lab
# ---------------------------------------------------------------------------


def ubuntu_cpu_turbo_view(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Presentation model for the Ubuntu CPU Performance Turbo engine."""
    if not profile or not isinstance(profile, dict):
        return {
            "available": False,
            "badge": "CPU TURBO // STANDBY",
            "summary": "n/a",
            "recommended_envs": 16,
            "recommended_workers": 4,
            "anti_thrash_active": False,
        }
    active = bool(profile.get("anti_thrash_active") or profile.get("turbo_enabled"))
    rec_envs = int(profile.get("recommended_envs") or 16)
    rec_workers = int(profile.get("recommended_workers") or 4)
    logical = int(profile.get("logical_cores") or 1)
    physical = int(profile.get("physical_cores_est") or 1)
    gov = str(profile.get("cpu_governor") or "unknown")
    os_name = str(profile.get("os_name") or "Linux")
    badge = (
        f"Ubuntu CPU Turbo Active ({rec_envs}e/{rec_workers}w, OMP=1)"
        if active
        else f"CPU Ready ({physical}C/{logical}T {os_name})"
    )
    summary = (
        f"{os_name}  ·  {physical} phys / {logical} threads  ·  gov={gov}  ·  "
        f"Optimal: {rec_envs} Envs x {rec_workers} Workers  ·  "
        f"BLAS Anti-Thrash: {'ON (OMP/MKL=1)' if active else 'OFF'}"
    )
    return {
        "available": True,
        "badge": badge,
        "summary": summary,
        "os_name": os_name,
        "cpu_model": str(profile.get("cpu_model") or "CPU"),
        "logical_cores": logical,
        "physical_cores_est": physical,
        "cpu_governor": gov,
        "recommended_envs": rec_envs,
        "recommended_workers": rec_workers,
        "recommended_trainer_threads": int(profile.get("recommended_trainer_threads") or 2),
        "anti_thrash_active": active,
    }


def training_convergence_view(
    reward_points: list[tuple[float, float]] | list[float] | None,
) -> dict[str, Any]:
    """Detect whether a training run is improving, plateauing, or regressing."""
    values: list[float] = []
    for item in reward_points or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            val = _finite_number(item[1])
        else:
            val = _finite_number(item)
        if val is not None:
            values.append(val)

    if len(values) < 6:
        return {
            "state": "WARMING_UP",
            "badge": "WARMING UP",
            "delta_pct": 0.0,
            "recommendation": "Collecting initial PPO rollouts (need >= 6 updates for trend radar).",
        }

    half = max(3, min(10, len(values) // 2))
    recent = values[-half:]
    prev = values[-2 * half : -half] if len(values) >= 2 * half else values[:half]
    recent_mean = sum(recent) / len(recent)
    prev_mean = sum(prev) / len(prev)
    peak = max(values)
    denom = max(abs(prev_mean), 1.0)
    delta_pct = ((recent_mean - prev_mean) / denom) * 100.0
    drop_from_peak_pct = ((peak - recent_mean) / max(abs(peak), 1.0)) * 100.0

    if drop_from_peak_pct > 8.0 and recent_mean < prev_mean:
        return {
            "state": "REGRESSING",
            "badge": f"REGRESSING (-{drop_from_peak_pct:.1f}% vs peak)",
            "delta_pct": round(delta_pct, 1),
            "recommendation": "Reward dropped from peak — evaluate best_eval.zip checkpoint or lower LR.",
        }
    if delta_pct >= 3.0:
        return {
            "state": "IMPROVING",
            "badge": f"IMPROVING (+{delta_pct:.1f}%)",
            "delta_pct": round(delta_pct, 1),
            "recommendation": "Policy is actively climbing — keep training on current curriculum.",
        }
    if len(values) >= 10 and abs(delta_pct) < 3.0:
        return {
            "state": "PLATEAU",
            "badge": f"PLATEAU ({delta_pct:+.1f}%)",
            "delta_pct": round(delta_pct, 1),
            "recommendation": "Reward stabilized — ready for Evaluation or next Curriculum stage.",
        }
    return {
        "state": "STEADY",
        "badge": f"STEADY ({delta_pct:+.1f}%)",
        "delta_pct": round(delta_pct, 1),
        "recommendation": "Steady progression across recent rollouts.",
    }


def _episode_metric_mean(detail: dict[str, Any], summary_key: str, ep_key: str) -> float | None:
    direct = _finite_number(detail.get(summary_key))
    if direct is not None:
        return direct
    eps = detail.get("per_episode") or detail.get("episode_metrics") or []
    if not isinstance(eps, list):
        return None
    vals = [_finite_number(e.get(ep_key)) for e in eps if isinstance(e, dict)]
    clean = [v for v in vals if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def tactical_combat_profile_view(detail: dict[str, Any] | None) -> dict[str, Any]:
    """Compute tactical FPS combat profile (K/D, Damage Trade, Archetype) from evaluation data."""
    if not detail or not detail.get("available"):
        return {
            "available": False,
            "kd_ratio": "n/a",
            "damage_trade": "n/a",
            "lethality": "n/a",
            "survival_rate": "n/a",
            "archetype": "NO EVALUATION DATA",
            "archetype_summary": "Select or run an evaluation to profile tactical combat behavior.",
        }

    win_rate = _finite_number(detail.get("win_rate")) or 0.0
    loss_rate = _finite_number(detail.get("loss_rate")) or 0.0
    kills = _episode_metric_mean(detail, "mean_kills", "kills")
    deaths = _episode_metric_mean(detail, "mean_deaths", "deaths")
    dmg_dealt = _episode_metric_mean(detail, "mean_damage_dealt", "damage_dealt")
    dmg_taken = _episode_metric_mean(detail, "mean_damage_taken", "damage_taken")
    mean_len = _finite_number(detail.get("mean_episode_length")) or 0.0

    eff_kills = kills if kills is not None else win_rate
    eff_deaths = deaths if deaths is not None else max(loss_rate, 0.1)
    kd_val = eff_kills / max(eff_deaths, 0.1)

    if dmg_dealt is not None and dmg_taken is not None:
        trade_val = dmg_dealt / max(dmg_taken, 1.0)
        trade_str = f"{trade_val:.2f}x ({dmg_dealt:.0f} / {dmg_taken:.0f} HP)"
    else:
        trade_val = (win_rate + 0.05) / max(loss_rate + 0.05, 0.1)
        trade_str = f"{trade_val:.2f}x (win/loss proxy)"

    survival_pct = max(0.0, min(100.0, (1.0 - loss_rate) * 100.0))
    lethality_pct = max(0.0, min(100.0, win_rate * 100.0))

    if win_rate >= 0.70 and (mean_len > 0 and mean_len <= 180):
        archetype = "AGGRESSIVE ENTRY FRAGGER"
        summary = "Fast-closing high-lethality policy that finishes engagements quickly."
    elif win_rate >= 0.65 and trade_val >= 1.5:
        archetype = "TACTICAL MARKSMAN"
        summary = "High damage-trade efficiency with strong cover discipline and low HP loss."
    elif survival_pct >= 75.0 and win_rate < 0.65:
        archetype = "EVASIVE ANCHOR"
        summary = (
            "Prioritizes survival and positioning; increase offensive reward weight for faster TTK."
        )
    elif win_rate >= 0.45:
        archetype = "BALANCED COMBATANT"
        summary = "Solid mid-tier duelist trading evenly; continue curriculum training."
    else:
        archetype = "DEVELOPING RECRUIT"
        summary = "Early-stage policy still learning aim acquisition and engagement spacing."

    return {
        "available": True,
        "kd_ratio": f"{kd_val:.2f}",
        "damage_trade": trade_str,
        "lethality": f"{lethality_pct:.1f}%",
        "survival_rate": f"{survival_pct:.1f}%",
        "archetype": archetype,
        "archetype_summary": summary,
    }


# ---------------------------------------------------------------------------
# Training budget (Steps vs. Time) and benchmark mode planning
# ---------------------------------------------------------------------------

#: Budget selectors the Training page renders as a segmented control.
BUDGET_MODES: tuple[tuple[str, str], ...] = (("steps", "Steps"), ("time", "Time"))

#: Benchmark modes: the automatic host-scaled sweep, its deliberately
#: oversubscribing "push" variant, and the fully manual candidate lists.
BENCHMARK_MODES: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("push", "Push"),
    ("custom", "Custom"),
)

#: Environment ladders. ``auto`` is the host-scaled sweep (capped at 128
#: environments, which is where the sharded bridge stops scaling on a
#: 16-32 thread desktop), ``push`` continues to 256 to find the plateau.
#: Push mode keeps climbing past the Auto ceiling. The point of Push is to
#: measure where throughput stops improving, so it is allowed to oversubscribe
#: the host on purpose.
PUSH_ENVIRONMENT_LADDER: tuple[int, ...] = (16, 32, 64, 96, 128, 192, 256)
PUSH_WORKER_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48)


def _physical_cores(cpu_count: int | None) -> int:
    """Physical-core estimate - the sharding module's, with a GUI default.

    The Control Center passes ``os.cpu_count()``; a bare ``None`` only
    happens in tests and previews, where eight logical cores is the least
    surprising machine to plan for.
    """
    from .sharded_env import physical_core_estimate

    return physical_core_estimate(cpu_count if cpu_count else 8)


def budget_view(
    mode: str,
    *,
    steps_raw: str,
    minutes_raw: str,
    steps_per_second: float | None = None,
    checkpoint_frequency: int | None = None,
) -> dict[str, Any]:
    """Validate and describe the *Steps* or *Time* training budget.

    The Time budget is honest about what it is: the window asks the trainer
    to stop at its next safe boundary, which is up to one checkpoint
    interval later, and the trainer then saves its final checkpoint. Nothing
    here predicts a step count it did not measure; the estimate is only
    shown when a measured ``steps_per_second`` exists.
    """
    chosen = "time" if str(mode) == "time" else "steps"
    view: dict[str, Any] = {
        "mode": chosen,
        "steps": None,
        "minutes": None,
        "errors": [],
        "warnings": [],
        "estimate": None,
        "budget_line": "",
    }
    if chosen == "steps":
        try:
            steps = int(str(steps_raw).strip())
        except (TypeError, ValueError):
            view["errors"].append("Total timesteps must be a whole number")
            return view
        if steps <= 0:
            view["errors"].append("Total timesteps must be positive")
            return view
        view["steps"] = steps
        view["budget_line"] = f"{format_number(steps)} steps"
        if steps_per_second and steps_per_second > 0:
            eta = estimate_training_duration(steps, steps_per_second)
            if eta:
                view["estimate"] = eta
        return view

    try:
        minutes = float(str(minutes_raw).strip())
    except (TypeError, ValueError):
        view["errors"].append("Time budget must be a number of minutes")
        return view
    if minutes <= 0:
        view["errors"].append("Time budget must be positive")
        return view
    if minutes > 24 * 60:
        view["errors"].append("Time budget above 1440 minutes (24 h) is refused")
        return view
    view["minutes"] = minutes
    view["budget_line"] = (
        f"{format_number(minutes, 1)} min (cooperative stop, final checkpoint saved)"
    )
    if steps_per_second and steps_per_second > 0:
        view["estimate"] = (
            f"~{format_number(minutes * 60 * steps_per_second)} steps at the measured rate"
        )
    if (
        checkpoint_frequency
        and checkpoint_frequency > 0
        and steps_per_second
        and steps_per_second > 0
    ):
        interval_minutes = (checkpoint_frequency / steps_per_second) / 60.0
        if interval_minutes > minutes / 4:
            view["warnings"].append(
                "Checkpoint interval is about "
                f"{format_number(interval_minutes, 1)} min — the stop can land up to that "
                "much after the budget"
            )
    return view


def parse_int_list(text: str) -> list[int]:
    """Parse ``"16, 32,64"`` into a de-duplicated, ascending candidate list."""
    values: list[int] = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise ValueError(f"'{chunk}' is not a whole number") from exc
        if value < 1:
            raise ValueError(f"'{chunk}' must be at least 1")
        values.append(value)
    return sorted(set(values))


def _positive_number(
    raw: str, errors: list[str], label: str, *, allow_float: bool = False
) -> float | None:
    """Parse a positive number for a form field, appending one clear error."""
    text = str(raw).strip()
    if not text:
        errors.append(f"{label} must be given")
        return None
    try:
        value = float(text) if allow_float else int(text)
    except ValueError:
        errors.append(f"{label} must be a {'number' if allow_float else 'whole number'}")
        return None
    if value <= 0:
        errors.append(f"{label} must be positive")
        return None
    return float(value) if allow_float else int(value)


def _custom_benchmark_plan(
    view: dict[str, Any],
    *,
    environment_text: str,
    worker_text: str,
    steps_raw: str,
    minutes_raw: str,
) -> dict[str, Any]:
    """Validate the manual candidate lists and budget for Custom mode."""
    try:
        environments = parse_int_list(environment_text)
    except ValueError as exc:
        view["errors"].append(f"Environments: {exc}")
        environments = []
    try:
        workers = parse_int_list(worker_text)
    except ValueError as exc:
        view["errors"].append(f"Workers: {exc}")
        workers = []
    if not environments:
        view["errors"].append("Environments: give at least one value (e.g. 16,32,64)")
    if not workers:
        view["errors"].append("Workers: give at least one value (e.g. 4,8)")
    if any(environment > 512 for environment in environments):
        view["errors"].append("Environments above 512 are refused")
    if any(worker > 128 for worker in workers):
        view["errors"].append("Workers above 128 are refused")
    steps_text = str(steps_raw).strip()
    minutes_text = str(minutes_raw).strip()
    if steps_text:
        view["budget_mode"] = "steps"
        steps = _positive_number(steps_text, view["errors"], "Steps per configuration")
        view["steps"] = steps
    else:
        view["budget_mode"] = "time"
        view["minutes"] = _positive_number(
            minutes_text, view["errors"], "Minutes", allow_float=True
        )
    view["environments"] = environments
    view["workers"] = workers
    configurations = sum(
        1 for environment in environments for worker in workers if worker <= environment
    )
    view["expected_configurations"] = configurations
    wanted = len(environments) * len(workers)
    if 0 < configurations < wanted:
        view["warnings"].append(
            f"{wanted - configurations} of {wanted} requested configurations are skipped: "
            "a worker count above the environment count cannot run"
        )
    if configurations == 0 and not view["errors"]:
        view["errors"].append(
            "No valid pair: a worker count above the environment count is skipped"
        )
    view["summary"] = (
        f"custom sweep: {len(environments)} environment(s) x {len(workers)} worker count(s)"
        f" -> {configurations} configuration(s)"
    )
    return view


def _automatic_benchmark_plan(
    view: dict[str, Any],
    *,
    mode: str,
    physical: int,
    logical: int,
    minutes_raw: str,
    measured_steps_per_second: float | None,
) -> dict[str, Any]:
    """Host-scaled candidate ladders for the Auto and Push modes.

    Auto uses the pipeline's own ladder (:mod:`sandboxai.benchmark_pipeline`),
    so the plan shown in the window and the sweep the pipeline runs are the
    same numbers - two hand-maintained ladders drifted apart before, which is
    how a run could stop at four workers while the window advertised more.
    """
    from .benchmark_pipeline import default_environment_counts, default_worker_counts

    if mode == "push":
        ceiling = max(32, min(256, physical * 16))
        environments = [value for value in PUSH_ENVIRONMENT_LADDER if value <= ceiling]
        worker_ceiling = min(48, max(4, physical * 2))
        workers = [value for value in PUSH_WORKER_LADDER if value <= worker_ceiling]
        if workers[-1:] != [worker_ceiling]:
            workers.append(worker_ceiling)
        view["warnings"].append(
            "Push mode deliberately oversubscribes: it keeps measuring past the knee of the "
            "scaling curve to find where throughput stops improving"
        )
    else:
        environments = list(default_environment_counts(logical))
        widest = max(environments)
        workers = list(default_worker_counts(widest, logical))
        view["environments_note"] = (
            "Auto probes past the conservative --env-workers auto recommendation, so a wide "
            "environment count cannot look slow only because it was starved of workers"
        )
    view["environments"] = environments
    view["workers"] = workers
    view["budget_mode"] = "time"
    try:
        view["minutes"] = float(str(minutes_raw).strip()) if str(minutes_raw).strip() else 15.0
    except ValueError:
        view["minutes"] = 15.0
    view["expected_configurations"] = sum(
        1 for environment in environments for worker in workers if worker <= environment
    )
    view["summary"] = (
        f"{mode} sweep scaled to {logical} logical / {physical} physical cores: "
        f"up to {max(environments)} environments, up to {max(workers)} workers"
    )
    if measured_steps_per_second:
        view["summary"] += f" (last measured {format_number(measured_steps_per_second, 1)} steps/s)"
    return view


def benchmark_mode_view(
    mode: str,
    *,
    environment_text: str = "",
    worker_text: str = "",
    steps_raw: str = "",
    minutes_raw: str = "",
    cpu_count: int | None = None,
    measured_steps_per_second: float | None = None,
) -> dict[str, Any]:
    """Build the host-scaled plan for a benchmark strategy.

    The GUI uses the automatic strategy; the lower-level planner also keeps
    its push/custom variants for scripted callers and tests. This is a *plan*:
    which environment/worker topologies will be measured and with which
    budget. Every number the pipeline later reports comes from the real
    bridge benchmark; the plan only decides what to try, which is why it is
    safe to scale it from the host's core count.
    """
    chosen = str(mode) if str(mode) in {key for key, _ in BENCHMARK_MODES} else "auto"
    physical = _physical_cores(cpu_count)
    logical = int(cpu_count) if cpu_count else 8
    view: dict[str, Any] = {
        "mode": chosen,
        "errors": [],
        "warnings": [],
        "environments": [],
        "workers": [],
        "budget_mode": "time",
        "minutes": None,
        "steps": None,
        "summary": "",
        "expected_configurations": 0,
    }
    if chosen == "custom":
        return _custom_benchmark_plan(
            view,
            environment_text=environment_text,
            worker_text=worker_text,
            steps_raw=steps_raw,
            minutes_raw=minutes_raw,
        )
    return _automatic_benchmark_plan(
        view,
        mode=chosen,
        physical=physical,
        logical=logical,
        minutes_raw=minutes_raw,
        measured_steps_per_second=measured_steps_per_second,
    )


# ---------------------------------------------------------------------------
# The Stats page: what the policy actually receives
# ---------------------------------------------------------------------------

#: Contact slots the observation vector tracks individually.
_CONTACT_SLOTS: tuple[tuple[str, str], ...] = (
    ("primary", "Contacts 1 (closest)"),
    ("secondary", "Contacts 2"),
    ("tertiary", "Contacts 3"),
)


def _field_section(name: str) -> str:
    """Which block of the vector a field belongs to, for the grouped table.

    Derived from the field name (never from a second hand-kept list), so a
    field added to ``contract.OBSERVATION_SPEC`` lands in the right block
    automatically.
    """
    if name.startswith("agent_"):
        return "Agent"
    if name.startswith("primary_enemy_") or name.startswith("enemy_"):
        return "Contact 1 (closest)"
    if name.startswith("secondary_enemy_"):
        return "Contact 2"
    if name.startswith("tertiary_enemy_"):
        return "Contact 3"
    if name.startswith("overflow_"):
        return "Contacts beyond slot 3"
    if name.startswith("target_"):
        return "Target selection"
    if (
        name.startswith(("last_sound_", "second_sound_", "sound_"))
        or name.startswith("audible_")
        or name.startswith("distinct_sound_")
    ):
        return "Hearing"
    if name.startswith(("remembered_", "explored_")) or name in {
        "current_area_known",
        "time_since_area_visited_norm",
        "contact_uncertainty_norm",
    }:
        return "Memory and exploration"
    if name.startswith(("nearest_obstacle", "visible_enemy_count", "corpse_count")):
        return "World and objects"
    return "Environment"


def _format_observation_value(values: Any) -> str:
    """One observation field's value(s) as the operator reads them."""
    if values is None:
        return "n/a"
    if isinstance(values, (list, tuple)):
        parts = [format_number(_finite_number(item), 3) for item in values]
        return ", ".join("n/a" if part is None else part for part in parts)
    number = _finite_number(values)
    return "n/a" if number is None else format_number(number, 3)


def observation_field_rows(observation: Any = None) -> list[dict[str, Any]]:
    """Every field of the observation contract, with this tick's value.

    This is the literal answer to "what does the trained policy see": the
    rows come from ``contract.OBSERVATION_SPEC`` (the same table the bridge
    and the tests are checked against), and only the value column depends on
    a recorded observation. Without one the table is still complete - it is
    the contract, not a recording.
    """
    from .contract import OBSERVATION_SPEC, observation_slice

    values: list[float] = []
    if observation is not None:
        values = [float(value) for value in observation]
    rows: list[dict[str, Any]] = []
    for field in OBSERVATION_SPEC:
        index = (
            str(field.index)
            if field.width == 1
            else f"{field.index}\u2013{field.index + field.width - 1}"
        )
        raw: Any = None
        if values and len(values) >= field.index + field.width:
            raw = observation_slice(values, field.name) if field.width > 1 else values[field.index]
        rows.append(
            {
                "index": index,
                "field": field.name,
                "section": _field_section(field.name),
                "meaning": field.description,
                "normalization": field.normalization,
                "value": _format_observation_value(raw),
            }
        )
    return rows


def observation_section_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group field rows by their section, in contract order."""
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    for row in rows:
        section = str(row.get("section") or "Other")
        if grouped and grouped[-1][0] == section:
            grouped[-1][1].append(row)
        else:
            grouped.append((section, [row]))
    return grouped


def _obs_scalar(observation: Any, name: str) -> float | None:
    """One scalar observation field by name, or ``None``."""
    if observation is None:
        return None
    from .contract import OBSERVATION_INDEX

    position = OBSERVATION_INDEX.get(name)
    if position is None:
        return None
    index, width = position
    values = list(observation)
    if width != 1 or len(values) <= index:
        return None
    return _finite_number(values[index])


def _obs_vector(observation: Any, name: str) -> list[float]:
    """One vector observation field by name, or an empty list."""
    if observation is None:
        return []
    from .contract import OBSERVATION_INDEX

    position = OBSERVATION_INDEX.get(name)
    if position is None:
        return []
    index, width = position
    values = list(observation)
    if len(values) < index + width:
        return []
    return [float(value) for value in values[index : index + width]]


def _triple(values: list[float]) -> str:
    """A relative-position triple or a metre distance, as text."""
    if not values:
        return "n/a"
    return "(" + ", ".join(format_number(value, 3) for value in values) + ")"


def contact_rows(observation: Any) -> list[dict[str, Any]]:
    """The three individually tracked contacts, decoded from the vector.

    The policy never receives a world-space enemy position: every slot holds
    a *relative* position, a distance, a bearing and the perception flags
    (visible / in FOV / line of sight / information age / confidence /
    source). That is exactly what this table shows, so an operator can see
    when the policy is aiming at a memory instead of a sighting.
    """
    rows: list[dict[str, Any]] = []
    for slot, label in _CONTACT_SLOTS:
        prefix = f"{slot}_enemy_"
        # Only slots 2 and 3 carry an explicit alive flag; the primary slot's
        # "is anyone alive" answer is the alive-enemy count, exactly as the
        # contract documents the dead-enemy fallback.
        alive_name = "alive_enemy_count_norm" if slot == "primary" else f"{slot}_enemy_alive"
        alive = _obs_scalar(observation, alive_name) if observation is not None else None
        health = _obs_scalar(observation, f"{prefix}health_norm")
        distance = _obs_scalar(observation, f"{prefix}distance_norm")
        bearing = _obs_scalar(observation, f"{prefix}bearing_norm")
        elevation = _obs_scalar(observation, f"{prefix}elevation_norm")
        position = _obs_vector(observation, f"{prefix}relative_position_norm")
        visible = _obs_scalar(observation, f"{prefix}visible")
        in_fov = _obs_scalar(observation, "primary_enemy_in_fov") if slot == "primary" else None
        los = _obs_scalar(observation, "primary_enemy_los_clear") if slot == "primary" else None
        age = _obs_scalar(observation, f"{prefix}info_age_norm")
        confidence = (
            _obs_scalar(observation, "primary_enemy_confidence") if slot == "primary" else None
        )
        from_vision = (
            _obs_scalar(observation, "primary_enemy_source_visual") if slot == "primary" else None
        )
        from_sound = (
            _obs_scalar(observation, "primary_enemy_source_sound") if slot == "primary" else None
        )
        if alive is None and not position and distance is None:
            state = "not tracked"
        elif alive is not None and alive < 0.5:
            state = "dead / absent"
        elif visible is not None and visible < 0.5:
            state = "remembered (no sighting)"
        else:
            state = "visible"
        source = "n/a"
        if from_vision is not None or from_sound is not None:
            source = (
                "vision"
                if (from_vision or 0.0) >= 0.5
                else ("sound" if (from_sound or 0.0) >= 0.5 else "none")
            )
        rows.append(
            {
                "slot": label,
                "state": state,
                "relative_position": _triple(position),
                "distance": _format_fraction(distance),
                "bearing": _format_fraction(bearing),
                "elevation": _format_fraction(elevation),
                "health": _format_fraction(health),
                "visible": _format_flag(visible),
                "in_fov": _format_flag(in_fov),
                "los": _format_flag(los),
                "info_age": _format_fraction(age),
                "confidence": _format_fraction(confidence),
                "source": source,
            }
        )
    return rows


def _format_fraction(value: float | None, decimals: int = 3) -> str:
    """A normalised value as text (``n/a`` when absent, never 0 by default)."""
    return "n/a" if value is None else format_number(value, decimals)


def _format_flag(value: float | None) -> str:
    """A 0/1 observation flag as yes/no/unknown."""
    if value is None:
        return "n/a"
    return "yes" if value >= 0.5 else "no"


def _count_text(count_norm: float | None, normalizer: int | None = None) -> str:
    """A normalized count field as a human count ("2", "8+" or "n/a").

    The vector stores ``count / N`` clamped to 1.0, so the last bucket means
    "N or more" and must not be displayed as an exact number. ``None`` stays
    ``n/a`` instead of becoming a zero that looks measured.
    """
    from .contract import OBSERVATION_COUNT_NORMALIZER

    if count_norm is None:
        return "n/a"
    limit = int(normalizer or OBSERVATION_COUNT_NORMALIZER)
    count = int(round(max(0.0, min(1.0, float(count_norm))) * limit))
    return f"{limit}+" if count >= limit else str(count)


def _object_kind_label(kind_norm: float | None) -> str:
    """The object kind behind ``object_k_kind_norm``, in operator language.

    The contract stores the ordinal divided by ``len(OBJECT_KIND_NAMES) - 1``
    (never the raw enum), so the label is recovered by rounding back. An
    unknown value reads "unknown kind" instead of being silently snapped to
    the nearest entry.
    """
    from .contract import OBJECT_KIND_NAMES

    if kind_norm is None:
        return "unknown kind"
    ordinal = round(kind_norm * (len(OBJECT_KIND_NAMES) - 1))
    if not 0 <= ordinal < len(OBJECT_KIND_NAMES):
        return "unknown kind"
    return OBJECT_KIND_NAMES[ordinal].replace("_", " ")


def _object_slot_row(observation: Any, rank: int) -> dict[str, Any]:
    """One contract-v4 object slot: a live sighting, or honestly empty."""
    prefix = f"object_{rank}"
    visible = _obs_scalar(observation, f"{prefix}_visible")
    if observation is None:
        return {
            "field": f"visible object {rank}",
            "distance": "n/a",
            "bearing": "n/a",
            "note": "nearest visible object (contract v4); no recording loaded",
        }
    if visible is None or visible < 0.5:
        return {
            "field": f"visible object {rank}",
            "distance": "n/a",
            "bearing": "n/a",
            "note": "no object in this slot: nothing visible there this tick",
        }
    kind = _object_kind_label(_obs_scalar(observation, f"{prefix}_kind_norm"))
    return {
        "field": f"visible object {rank}",
        "distance": _format_fraction(_obs_scalar(observation, f"{prefix}_distance_norm")),
        "bearing": _format_fraction(_obs_scalar(observation, f"{prefix}_bearing_norm")),
        "note": f"{kind} - in the FOV cone and not occluded",
    }


def world_rows(observation: Any) -> list[dict[str, Any]]:
    """World, object and memory fields in operator language.

    "Objects" are the cover boxes, walls and corpses the simulation actually
    has. Each row says whether that knowledge is a live sighting or a memory,
    because that distinction is what the policy is trained on. Without a
    recording the rows keep their shape and read ``n/a`` - the set of fields
    is part of the contract, the values are not.
    """
    return [
        {
            "field": "nearest object (cover / wall)",
            "distance": _format_fraction(
                _obs_scalar(observation, "nearest_obstacle_distance_norm")
            ),
            "bearing": _format_fraction(_obs_scalar(observation, "nearest_obstacle_bearing_norm")),
            "note": "live geometry in front of the agent",
        },
        {
            "field": "forward clearance",
            "distance": _format_fraction(_obs_scalar(observation, "agent_forward_clearance_norm")),
            "bearing": "n/a",
            "note": "first sight-blocking surface straight ahead",
        },
        {
            "field": "corpses in the arena",
            "distance": _format_fraction(_obs_scalar(observation, "corpse_count_norm")),
            "bearing": "n/a",
            "note": "count of dead bodies (environmental information)",
        },
        {
            "field": "remembered cover",
            "distance": _format_fraction(
                _obs_scalar(observation, "remembered_cover_distance_norm")
            ),
            "bearing": _format_fraction(_obs_scalar(observation, "remembered_cover_bearing_norm")),
            "note": "from the agent's own map memory, not a live sighting",
        },
        {
            "field": "remembered danger",
            "distance": _format_fraction(
                _obs_scalar(observation, "remembered_danger_distance_norm")
            ),
            "bearing": _format_fraction(_obs_scalar(observation, "remembered_danger_bearing_norm")),
            "note": "place the agent was last hurt",
        },
        {
            "field": "map explored",
            "distance": _format_fraction(_obs_scalar(observation, "explored_fraction")),
            "bearing": "n/a",
            "note": "share of the map the agent has actually looked at",
        },
        _object_slot_row(observation, 1),
        _object_slot_row(observation, 2),
        _object_slot_row(observation, 3),
        {
            "field": "objects visible now",
            "distance": _format_fraction(_obs_scalar(observation, "visible_object_count_norm")),
            "bearing": "n/a",
            "note": (
                "how many objects the query returned this tick "
                f"({_count_text(_obs_scalar(observation, 'visible_object_count_norm'))})"
            ),
        },
    ]


def audio_rows(observation: Any) -> list[dict[str, Any]]:
    """The hearing channel: what the policy perceives, not what is audible."""
    return [
        {
            "field": "loudest event direction",
            "value": _triple(_obs_vector(observation, "last_sound_direction")),
            "note": "unit vector with the documented directional error",
        },
        {
            "field": "loudest event distance",
            "value": _format_fraction(_obs_scalar(observation, "last_sound_distance_norm")),
            "note": "normalised by the arena diagonal",
        },
        {
            "field": "loudest event loudness",
            "value": _format_fraction(_obs_scalar(observation, "last_sound_loudness")),
            "note": "after distance and wall occlusion",
        },
        {
            "field": "loudest event category",
            "value": _format_fraction(_obs_scalar(observation, "last_sound_category_norm")),
            "note": "footstep/jump/land/shot/impact/death/environment",
        },
        {
            "field": "loudest event age",
            "value": _format_fraction(_obs_scalar(observation, "last_sound_age_norm")),
            "note": "seconds since the event, normalised",
        },
        {
            "field": "audible events this tick",
            "value": _format_fraction(_obs_scalar(observation, "audible_event_count_norm")),
            "note": "how many events the agent can hear right now",
        },
        {
            "field": "direction precision",
            "value": _format_fraction(_obs_scalar(observation, "sound_direction_error_norm")),
            "note": "how imprecisely the loudest event can be placed",
        },
    ]


def action_rows(action: Any) -> list[dict[str, Any]]:
    """The action vector the policy emitted, per component."""
    from .contract import ACTION_SPEC

    values = [int(value) for value in action] if action else []
    rows: list[dict[str, Any]] = []
    for field in ACTION_SPEC:
        value = values[field.index] if len(values) > field.index else None
        rows.append(
            {
                "component": field.name,
                "value": "n/a" if value is None else str(value),
                "meaning": field.description,
            }
        )
    return rows


def _nearest_object_suffix(observation: Any) -> str:
    """``" (crate)"`` when slot 1 holds a sighting, otherwise nothing.

    Reads the same slot the world table shows, so the headline and the table
    can never disagree about what the nearest visible object is.
    """
    if observation is None:
        return ""
    visible = _obs_scalar(observation, "object_1_visible")
    if visible is None or visible < 0.5:
        return ""
    return f" ({_object_kind_label(_obs_scalar(observation, 'object_1_kind_norm'))})"


def observation_summary_view(observation: Any) -> dict[str, Any]:
    """The one-line answer to "what is the policy looking at right now"."""
    if observation is None:
        return {
            "available": False,
            "summary": "No observation loaded.",
            "headline": "n/a",
        }
    in_combat = _obs_scalar(observation, "in_combat")
    ready = _obs_scalar(observation, "weapon_ready")
    health = _obs_scalar(observation, "agent_health_norm")
    alive = _obs_scalar(observation, "alive_enemy_count_norm")
    visible = _obs_scalar(observation, "visible_enemy_count_norm")
    remembered = _obs_scalar(observation, "remembered_enemy_count_norm")
    distance = _obs_scalar(observation, "enemy_distance_norm")
    illumination = _obs_scalar(observation, "local_illumination")
    in_cover = _obs_scalar(observation, "agent_in_cover")
    parts = [
        f"health {format_fraction_as_percent(health) if health is not None else 'n/a'}",
        f"objects {_count_text(_obs_scalar(observation, 'visible_object_count_norm'))}"
        + _nearest_object_suffix(observation),
        f"weapon {'ready' if (ready or 0.0) >= 0.5 else 'not ready'}",
        f"contacts alive {format_fraction_as_percent(alive) if alive is not None else 'n/a'}",
        f"visible {format_fraction_as_percent(visible) if visible is not None else 'n/a'}",
        f"remembered {format_fraction_as_percent(remembered) if remembered is not None else 'n/a'}",
        f"closest {format_number(distance, 3) if distance is not None else 'n/a'}",
        f"cover {'yes' if (in_cover or 0.0) >= 0.5 else 'no'}",
        f"light {format_number(illumination, 2) if illumination is not None else 'n/a'}",
    ]
    return {
        "available": True,
        "in_combat": bool((in_combat or 0.0) >= 0.5),
        "headline": ("in combat" if (in_combat or 0.0) >= 0.5 else "not engaging"),
        "summary": "   ·   ".join(parts),
    }


def replay_contract_view(result: dict[str, Any]) -> dict[str, Any]:
    """Does this recording match the contract the policy trains on?

    A replay is evidence and keeps loading after the observation vector grows
    (the adapter reads it with ``strict_contract=False``), so the page has to
    say which contract the numbers came from. ``recorded_observation_dim`` is
    the width in the file; a mismatch means those values cannot be compared
    with a current policy's input, even though they are still readable.
    """
    recorded = result.get("recorded_observation_dim")
    current = int(result.get("current_observation_dim") or 0)
    matches = result.get("contract_match")
    if matches is None:
        return {"matches": True, "severity": "text_dim", "text": "contract n/a"}
    if matches:
        return {
            "matches": True,
            "severity": "text_dim",
            "text": f"recorded under the current contract ({recorded} floats)",
        }
    return {
        "matches": False,
        "severity": "warn",
        "text": (
            f"recorded under an older contract ({recorded} floats, current {current}) - "
            "readable, but not comparable with a current policy's input"
        ),
    }


def table_signature(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> tuple[tuple[Any, ...], ...]:
    """A cheap identity for a table's contents.

    Pages poll on a timer and rebuild their Treeviews from the result. A
    rebuild re-inserts every row and drops the operator's selection, so a page
    that receives the same rows as last time should leave the table alone.
    Comparing this signature decides that. Only the keys that are rendered are
    part of it, so a change that would not be visible does not force a
    rebuild, and vice versa.
    """
    return tuple(tuple(row.get(key) for key in keys) for row in rows)


def replay_table_rows(replays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per recorded replay, as the Stats page's source picker."""
    rows: list[dict[str, Any]] = []
    for replay in replays:
        header = replay.get("header") or {}
        detail = str(header.get("detail") or replay.get("detail") or "light")
        rows.append(
            {
                "name": str(replay.get("name") or replay.get("path") or ""),
                "run": str(replay.get("run") or ""),
                "detail": detail,
                "ticks": format_number(replay.get("ticks"), 0),
                "seed": str(header.get("seed") if header.get("seed") is not None else "n/a"),
                "map": str(header.get("map_id") or "n/a"),
                "scenario": str(header.get("scenario") or "n/a"),
                "curriculum": str(
                    header.get("curriculum_level")
                    if header.get("curriculum_level") is not None
                    else "n/a"
                ),
                "enemies": str(
                    header.get("enemy_count") if header.get("enemy_count") is not None else "n/a"
                ),
                "observations": "yes" if detail == "detailed" else "no",
                "size": format_bytes(replay.get("size_bytes")),
            }
        )
    return rows


def stats_source_view(info: dict[str, Any] | None) -> dict[str, Any]:
    """Headline + guidance for the replay the Stats page is showing."""
    if not info:
        return {
            "available": False,
            "headline": "No replay selected",
            "detail": (
                "Replays are written to <output root>/<run>/replays/. A light replay "
                "has no observation vector; record with replay_detail=detailed to see "
                "exactly what the policy received."
            ),
            "tick_count": 0,
            "detailed": False,
        }
    header = info.get("header") or {}
    detailed = str(header.get("detail") or "") == "detailed"
    detail = (
        "Observation vector recorded per tick — this is the policy's real input."
        if detailed
        else (
            "This is a light replay: it stores actions and rewards, not the "
            "observation vector. Re-run with replay_detail=detailed to inspect "
            "what the policy saw."
        )
    )
    return {
        "available": True,
        "detailed": detailed,
        "tick_count": int(info.get("tick_count") or 0),
        "headline": (
            f"{info.get('name', '')}   ·   seed {header.get('seed')}   ·   "
            f"{header.get('map_id') or 'generated map'}   ·   "
            f"level {header.get('curriculum_level')}"
        ),
        "detail": detail,
        "policy": str(header.get("policy_id") or header.get("checkpoint") or "n/a"),
    }


def ttk_evidence_view(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Counts + rows for the TTK Testing calibration evidence table."""
    if not summary:
        return {"available": False, "headline": "n/a", "rows": [], "checked_on": ""}
    mechanics = summary.get("mechanics") or {}

    def count(status: str) -> int:
        return len(mechanics.get(status) or [])

    verified = count("verified")
    pending = count("calibration_required")
    excluded = count("excluded")
    return {
        "available": True,
        "checked_on": str(summary.get("evidence_checked_on") or ""),
        "verified": verified,
        "calibration_required": pending,
        "excluded": excluded,
        "headline": (
            f"{verified} verified   ·   {pending} need calibration   ·   {excluded} excluded"
        ),
    }


def ttk_evidence_rows(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The evidence matrix as table rows (mechanic, status, rule, source)."""
    if not summary:
        return []
    rows: list[dict[str, Any]] = []
    for status, items in (summary.get("mechanics") or {}).items():
        for item in items or []:
            rows.append(
                {
                    "mechanic": str(item.get("mechanic") or ""),
                    "status": status.replace("_", " "),
                    "rule": str(item.get("implementation_rule") or ""),
                    "source": str(item.get("source_label") or item.get("source_url") or ""),
                    "measured_value": str(item.get("notes") or ""),
                }
            )
    return rows


def resolve_run_checkpoint(run_dir: str | Path, *, prefer_best: bool = False) -> Path | None:
    """Resolve the best or latest checkpoint inside ``run_dir`` across all trainer naming conventions."""
    import re

    base = Path(run_dir)
    ckpt_dir = base / "checkpoints"
    best_eval = ckpt_dir / "best_eval.zip"
    best_plain = ckpt_dir / "best.zip"
    latest = ckpt_dir / "latest.zip"
    final = base / "final.zip"

    periodic: list[tuple[int, Path]] = []
    if ckpt_dir.is_dir():
        for candidate in ckpt_dir.glob("ppo_*_steps.zip"):
            match = re.search(r"ppo_(\d+)_steps\.zip$", candidate.name)
            if match:
                periodic.append((int(match.group(1)), candidate))
        periodic.sort()
    newest_periodic = periodic[-1][1] if periodic else None

    order = (
        (best_eval, best_plain, latest, newest_periodic, final)
        if prefer_best
        else (latest, newest_periodic, best_eval, best_plain, final)
    )
    for item in order:
        if item is not None and item.is_file():
            return item
    return None

