"""Read-only inspection of training runs on disk.

The Control Center needs to answer "what runs exist, what state are they
in, and what did they produce?" without re-implementing the run layout in
GDScript, and without the risk that an inspector mutates a live run. This
module is that backend interface - and deliberately nothing else:

* **read-only.** Nothing here opens a file for writing, creates a
  directory or deletes anything. Inspecting a run that is currently
  training is safe.
* **tolerant.** A run directory that is half-written, was killed
  mid-evaluation, or contains corrupt JSON must still produce a report.
  Every parse failure lands in ``problems`` instead of raising.
* **honest.** Nothing is inferred that cannot be read. A run with no
  ``run_summary.json`` is reported as ``incomplete``, never as "probably
  finished"; a value that is absent is absent, not zero.

Consumers: ``sandboxai inspect-runs`` (text or JSON) and, through it, the
Control Center HISTORY page. The JSON is a stable contract - documents
carry a ``format`` field and new fields are additive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

RUN_REPORT_FORMAT = "sandboxai.run_report/v1"
RUN_INDEX_FORMAT = "sandboxai.run_index/v1"

#: Files a completed integrated-pipeline run is expected to leave behind.
#: Absence is reported, not fatal: `curriculum_mode="fixed"` runs skip the
#: manifest, and a killed run skips the summary.
EXPECTED_ARTIFACTS = (
    "config.json",
    "run_manifest.json",
    "run_summary.json",
    "checkpoints/latest.zip",
    "evaluations/latest.json",
    "logs/training.jsonl",
)

#: Log files worth reporting a size/line count for.
_LOG_FILES = (
    "logs/training.jsonl",
    "logs/episodes.jsonl",
    "logs/curriculum.jsonl",
    "logs/training_profile.json",
    "events.jsonl",
)

_MAX_TAIL_BYTES = 1 << 20


def _utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def read_json(path: Path) -> tuple[Any, str | None]:
    """Reads a JSON document, returning ``(value, problem)``.

    A missing file is ``(None, None)``: absence is a normal state, not a
    problem. Unreadable or malformed content is ``(None, description)``.
    """
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name}: unreadable ({exc.__class__.__name__})"
    except json.JSONDecodeError as exc:
        return None, f"{path.name}: invalid JSON at line {exc.lineno}"


def tail_jsonl(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Last ``limit`` parsable JSON objects of a JSONL file.

    Reads at most the trailing megabyte, so tailing a multi-gigabyte
    telemetry log costs the same as tailing a small one. Unparsable lines
    are skipped rather than failing the tail: a live run's last line is
    routinely half-written.
    """
    if limit <= 0 or not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _MAX_TAIL_BYTES:
                handle.seek(size - _MAX_TAIL_BYTES)
                handle.readline()
            blob = handle.read()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-limit:]


def _file_stat(path: Path) -> dict[str, Any] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return {"bytes": info.st_size, "modified_utc": _utc(info.st_mtime)}


def _count_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def is_run_directory(path: Path) -> bool:
    """A run is a directory that carries a config or a manifest."""
    return path.is_dir() and (
        (path / "config.json").is_file() or (path / "run_manifest.json").is_file()
    )


def discover_run_directories(root: str | Path) -> list[Path]:
    """Run directories under ``root``, newest last.

    ``root`` may be the runs directory itself, a training output root
    (``<root>/runs``), or a single run directory.
    """
    base = Path(root).expanduser()
    if is_run_directory(base):
        return [base]
    if (base / "runs").is_dir():
        base = base / "runs"
    if not base.is_dir():
        return []
    found = [child for child in base.iterdir() if is_run_directory(child)]
    # Name-then-mtime: run ids are timestamps, so the name order is the
    # chronological order for normal runs and mtime settles the rest.
    found.sort(key=lambda item: (item.name, item.stat().st_mtime if item.exists() else 0.0))
    return found


def _checkpoint_inventory(run_dir: Path) -> dict[str, Any]:
    directory = run_dir / "checkpoints"
    entries: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.zip")):
            stat = _file_stat(path)
            if stat is None:
                continue
            entries.append({"name": path.name, **stat})
    return {
        "directory": str(directory),
        "count": len(entries),
        "entries": entries,
        "has_latest": (directory / "latest.zip").is_file(),
        "has_best": (directory / "best_eval.zip").is_file(),
        "has_final": (run_dir / "final.zip").is_file()
        or (directory / "final.zip").is_file(),
    }


def _evaluation_inventory(run_dir: Path, problems: list[str]) -> dict[str, Any]:
    directory = run_dir / "evaluations"
    steps: list[str] = []
    if directory.is_dir():
        steps = sorted(path.name for path in directory.glob("step_*") if path.is_dir())
    latest, problem = read_json(directory / "latest.json")
    if problem:
        problems.append(problem)
    best, problem = read_json(directory / "best.json")
    if problem:
        problems.append(problem)
    summary: dict[str, Any] = {"evaluation_count": len(steps), "steps": steps}
    if isinstance(latest, dict):
        summary["latest"] = {
            key: latest.get(key)
            for key in ("timesteps", "episodes", "mean_episode_reward", "win_rate", "mean_kills")
            if key in latest
        }
    if isinstance(best, dict):
        summary["best"] = dict(best)
    return summary


def _progress(
    config: Any, summary: Any, evaluation: dict[str, Any], control: Any
) -> dict[str, Any]:
    timesteps: int | None = None
    source = ""
    for candidate, label in (
        (summary, "run_summary.json"),
        (control, "status.json"),
        (evaluation.get("latest"), "evaluations/latest.json"),
    ):
        if isinstance(candidate, dict):
            value = candidate.get("timesteps") or candidate.get("steps")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timesteps = int(value)
                source = label
                break
    target = None
    if isinstance(config, dict):
        value = config.get("total_training_steps")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target = int(value)
    progress: dict[str, Any] = {
        "timesteps": timesteps,
        "target_timesteps": target,
        "source": source,
    }
    if timesteps is not None and target:
        progress["fraction"] = min(1.0, timesteps / float(target))
    return progress


def _status(summary: Any, control: Any) -> dict[str, Any]:
    """Run state, with the evidence it was derived from.

    Never guesses: a run without a summary and without a control status
    is ``incomplete``, which is exactly what is known about it.
    """
    if isinstance(summary, dict):
        state = "stopped" if summary.get("stopped") else "finished"
        return {"state": state, "source": "run_summary.json"}
    if isinstance(control, dict) and control.get("state"):
        return {"state": str(control["state"]).lower(), "source": "status.json"}
    return {"state": "incomplete", "source": "no run_summary.json"}


def _warnings(
    manifest: Any, config: Any, checkpoints: dict[str, Any], status: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    if isinstance(manifest, dict):
        code = manifest.get("code")
        if isinstance(code, dict) and code.get("dirty") is True:
            warnings.append(
                "code tree was dirty when the manifest was written; the commit "
                "hash does not fully describe this run"
            )
        contract = manifest.get("contract")
        if isinstance(contract, dict):
            from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

            if contract.get("observation_dim") not in (None, OBSERVATION_FIELD_COUNT):
                warnings.append(
                    "run used a %s-float observation; the current contract is %d"
                    % (contract.get("observation_dim"), OBSERVATION_FIELD_COUNT)
                )
            if contract.get("action_nvec") not in (None, list(ACTION_NVEC)):
                warnings.append("run used a different action space than the current contract")
    elif config is not None:
        warnings.append("no run_manifest.json: provenance for this run is unknown")
    if checkpoints["count"] == 0:
        warnings.append("no checkpoints were written")
    elif not checkpoints["has_best"]:
        warnings.append("no best_eval.zip: no evaluation ever improved on the initial score")
    if status["state"] == "incomplete":
        warnings.append("run has no run_summary.json; it did not reach a clean end")
    return warnings


def inspect_run(run_dir: str | Path, event_limit: int = 0) -> dict[str, Any]:
    """Read-only report for one run directory."""
    path = Path(run_dir).expanduser()
    problems: list[str] = []

    manifest, problem = read_json(path / "run_manifest.json")
    if problem:
        problems.append(problem)
    config, problem = read_json(path / "config.json")
    if problem:
        problems.append(problem)
    summary, problem = read_json(path / "run_summary.json")
    if problem:
        problems.append(problem)
    control, problem = read_json(path / "status.json")
    if problem:
        problems.append(problem)

    checkpoints = _checkpoint_inventory(path)
    evaluation = _evaluation_inventory(path, problems)
    status = _status(summary, control)

    logs: dict[str, Any] = {}
    for relative in _LOG_FILES:
        stat = _file_stat(path / relative)
        if stat is None:
            continue
        if relative.endswith(".jsonl"):
            stat["lines"] = _count_lines(path / relative)
        logs[relative] = stat

    report: dict[str, Any] = {
        "format": RUN_REPORT_FORMAT,
        "run_dir": str(path),
        "exists": path.is_dir(),
        "run_id": "",
        "experiment_id": "",
        "status": status,
        "progress": _progress(config, summary, evaluation, control),
        "checkpoints": checkpoints,
        "evaluation": evaluation,
        "logs": logs,
        "artifacts": {name: (path / name).is_file() for name in EXPECTED_ARTIFACTS},
        "problems": problems,
    }
    if not path.is_dir():
        report["problems"] = problems + ["run directory does not exist"]
        report["warnings"] = []
        return report

    if isinstance(manifest, dict):
        report["run_id"] = str(manifest.get("run_id", "") or "")
        report["experiment_id"] = str(manifest.get("experiment_id", "") or "")
        report["manifest"] = {
            key: manifest.get(key)
            for key in (
                "format",
                "created_utc",
                "seed",
                "device",
                "contract",
                "code",
                "code_revision",
                "host",
                "godot",
                "curriculum",
                "checkpoint_selection",
                "parallelism",
                "package_versions",
            )
            if key in manifest
        }
    if isinstance(config, dict):
        report["run_id"] = report["run_id"] or str(config.get("run_id", "") or "")
        report["experiment_id"] = report["experiment_id"] or str(
            config.get("experiment_id", "") or ""
        )
        report["config"] = {
            key: config.get(key)
            for key in (
                "seed",
                "device",
                "environment_count",
                "env_workers",
                "enemy_count",
                "total_training_steps",
                "learning_rate",
                "batch_size",
                "rollout_length",
                "resolved_rollout_length",
                "ppo_epochs",
                "torch_threads",
                "resolved_torch_threads",
                "curriculum_mode",
                "curriculum_level",
                "curriculum_start_level",
                "checkpoint_selection_metric",
                "checkpoint_selection_goal",
            )
            if key in config
        }
    if not report["run_id"]:
        report["run_id"] = path.name
    if isinstance(summary, dict):
        report["summary"] = {
            key: summary.get(key)
            for key in (
                "timesteps",
                "training_steps_completed",
                "device",
                "stopped",
                "best_checkpoint",
                "latest_checkpoint",
                "final_checkpoint",
                "training_profile",
            )
            if key in summary
        }
    if isinstance(control, dict):
        # status.json is a small, atomically-written flat document (see
        # run_control.RunControl): every key the running process publishes
        # is relayed verbatim rather than whitelisted, so a live Dashboard
        # can show training FPS, ETA, device, resource gauges and PPO
        # optimizer diagnostics without this module growing a parallel
        # whitelist every time the trainer starts publishing one more
        # field. It is still exactly what was on disk - nothing computed
        # or guessed is added here.
        report["control"] = dict(control)
    stat = _file_stat(path)
    if stat is not None:
        report["modified_utc"] = stat["modified_utc"]
    if event_limit > 0:
        report["recent_events"] = tail_jsonl(path / "events.jsonl", event_limit) or tail_jsonl(
            path / "logs" / "training.jsonl", event_limit
        )
    report["warnings"] = _warnings(manifest, config, checkpoints, status)
    return report


def inspect_runs(
    root: str | Path, limit: int = 0, event_limit: int = 0
) -> dict[str, Any]:
    """Read-only index of every run under ``root`` (newest last)."""
    directories = discover_run_directories(root)
    if limit > 0:
        directories = directories[-limit:]
    return {
        "format": RUN_INDEX_FORMAT,
        "root": str(Path(root).expanduser()),
        "run_count": len(directories),
        "runs": [inspect_run(directory, event_limit=event_limit) for directory in directories],
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _reward(report: dict[str, Any]) -> str:
    latest = report.get("evaluation", {}).get("latest")
    if isinstance(latest, dict) and isinstance(latest.get("mean_episode_reward"), (int, float)):
        return f"{float(latest['mean_episode_reward']):.2f}"
    return "n/a"


def format_run_index(index: dict[str, Any]) -> str:
    lines = [f"Runs under {index['root']} ({index['run_count']})", ""]
    if not index["runs"]:
        lines.append("  (none)")
        return "\n".join(lines)
    header = f"{'run':<34} {'state':<11} {'steps':>12} {'evals':>6} {'ckpt':>5} {'reward':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for report in index["runs"]:
        progress = report.get("progress", {})
        steps = progress.get("timesteps")
        lines.append(
            "%-34s %-11s %12s %6d %5d %9s"
            % (
                report["run_id"][:34],
                report["status"]["state"][:11],
                "n/a" if steps is None else f"{steps:,}",
                report.get("evaluation", {}).get("evaluation_count", 0),
                report.get("checkpoints", {}).get("count", 0),
                _reward(report),
            )
        )
    warned = [report for report in index["runs"] if report.get("warnings")]
    if warned:
        lines.append("")
        lines.append("Warnings:")
        for report in warned:
            for warning in report["warnings"]:
                lines.append(f"  {report['run_id']}: {warning}")
    return "\n".join(lines)


def format_run_report(report: dict[str, Any]) -> str:
    lines = [
        f"Run {report['run_id']}",
        f"  directory      {report['run_dir']}",
        f"  state          {report['status']['state']} (via {report['status']['source']})",
    ]
    if report.get("experiment_id"):
        lines.append(f"  experiment     {report['experiment_id']}")
    progress = report.get("progress", {})
    steps = progress.get("timesteps")
    target = progress.get("target_timesteps")
    if steps is not None:
        text = f"{steps:,}"
        if target:
            text += f" / {target:,} ({progress.get('fraction', 0.0) * 100:.1f}%)"
        lines.append(f"  timesteps      {text}")
    manifest = report.get("manifest", {})
    if manifest:
        code = manifest.get("code") or {}
        revision = str(code.get("commit", manifest.get("code_revision", "")) or "")[:12]
        dirty = code.get("dirty")
        marker = "" if dirty is None else ("+dirty" if dirty else "+clean")
        lines.append(f"  code           {revision or 'unknown'} {marker}".rstrip())
        host = manifest.get("host") or {}
        if host:
            lines.append(
                "  host           python %s, %s, %s CPUs"
                % (host.get("python", "?"), host.get("system", "?"), host.get("logical_cpus", "?"))
            )
        godot = manifest.get("godot") or {}
        if godot.get("version"):
            lines.append(f"  godot          {godot['version']}")
        selection = manifest.get("checkpoint_selection") or {}
        if selection:
            lines.append(
                "  selection      %s %s (min_delta %s)"
                % (selection.get("goal"), selection.get("metric"), selection.get("min_delta"))
            )
    checkpoints = report.get("checkpoints", {})
    lines.append(
        "  checkpoints    %d (latest=%s best=%s final=%s)"
        % (
            checkpoints.get("count", 0),
            checkpoints.get("has_latest"),
            checkpoints.get("has_best"),
            checkpoints.get("has_final"),
        )
    )
    evaluation = report.get("evaluation", {})
    lines.append(
        "  evaluations    %d (latest mean reward %s)"
        % (evaluation.get("evaluation_count", 0), _reward(report))
    )
    best = evaluation.get("best")
    if isinstance(best, dict):
        lines.append(
            "  best           score %s at %s timesteps"
            % (best.get("score", best.get("mean_reward")), best.get("timesteps"))
        )
    missing = [name for name, present in report.get("artifacts", {}).items() if not present]
    if missing:
        lines.append(f"  missing        {', '.join(missing)}")
    for warning in report.get("warnings", []):
        lines.append(f"  WARNING        {warning}")
    for problem in report.get("problems", []):
        lines.append(f"  PROBLEM        {problem}")
    events = report.get("recent_events")
    if events:
        lines.append("  recent events")
        for event in events:
            label = event.get("event") or event.get("message") or event.get("kind") or "event"
            lines.append(f"    - {label}")
    return "\n".join(lines)
