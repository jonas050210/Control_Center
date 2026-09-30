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


def format_number(value: Any, decimals: int = 0) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if decimals:
        return f"{value:,.{decimals}f}"
    return f"{int(round(value)):,}"


def format_fraction_as_percent(value: Any, decimals: int = 1) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def format_duration(seconds: Any) -> str:
    if seconds is None or isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "n/a"
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "n/a"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(amount) < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PB"


def format_timestamp(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))
    except (OverflowError, OSError, ValueError):
        return "n/a"


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
# Agents (process registry)
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
    # --- Basic: what almost every run changes. ---------------------------
    TrainingFieldSpec(
        "run_id", "Run ID", "str", "basic", help="Blank = timestamped automatically."
    ),
    TrainingFieldSpec(
        "experiment_id",
        "Experiment ID",
        "str",
        "basic",
        help="Optional grouping prefix for the run directory.",
    ),
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
    TrainingFieldSpec("total_training_steps", "Total timesteps", "int", "basic"),
    TrainingFieldSpec("device", "Device", "choice", "basic", choices=("auto", "cpu", "cuda")),
    TrainingFieldSpec(
        "curriculum_mode", "Curriculum mode", "choice", "basic", choices=("auto", "fixed")
    ),
    # --- Advanced: PPO/optimizer/curriculum internals. --------------------
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
