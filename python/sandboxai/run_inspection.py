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

import copy
import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .control_center_schema import validate_status

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
        return json.loads(path.read_text(encoding="utf-8-sig")), None
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

    The window is walked from its end and parsing stops once ``limit`` rows
    are in hand. A megabyte of a training log holds thousands of lines, and
    decoding and parsing all of them to return the last five made this the
    most expensive thing the Dashboard did on every poll tick.
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
    for line in reversed(blob.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
            if len(rows) >= limit:
                break
    rows.reverse()
    return rows


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


#: Remembered log line counts, ``path -> (bytes, lines)``. The Control Center
#: asks for a run report on a poll timer, and a run's log is the largest file
#: it owns: without this, a finished run's 100 MB ``training.jsonl`` was read
#: from the first byte to the last, twice a second, forever. Bounded, and only
#: ever used for files that grew - see ``_count_lines_cached``.
_LINE_COUNTS: dict[str, tuple[int, int]] = {}
_LINE_COUNTS_LIMIT = 512
_line_counts_lock = threading.Lock()

#: Stamp-keyed memo cache for ``inspect_run`` so 600ms GUI poll ticks do not
#: re-open and re-parse unchanged JSON manifests/configs/summaries across runs.
_INSPECT_RUN_CACHE: dict[tuple[str, int], tuple[tuple[Any, ...], str]] = {}
_INSPECT_RUN_CACHE_LIMIT = 256
_inspect_run_cache_lock = threading.Lock()

_INSPECT_CKPT_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_INSPECT_CKPT_CACHE_LIMIT = 256


def _stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _scandir_zip_sigs(directory: Path) -> tuple[tuple[str, int, int], ...]:
    import os

    if not directory.is_dir():
        return ()
    items: list[tuple[str, int, int]] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.endswith(".zip"):
                    st = entry.stat()
                    items.append((entry.name, st.st_mtime_ns, st.st_size))
    except OSError:
        return ()
    items.sort()
    return tuple(items)


def _scandir_eval_steps(directory: Path) -> tuple[str, ...]:
    import os

    if not directory.is_dir():
        return ()
    items: list[str] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.startswith("step_") and entry.is_dir():
                    items.append(entry.name)
    except OSError:
        return ()
    items.sort()
    return tuple(items)


def _run_dir_stamp(path: Path, event_limit: int) -> tuple[Any, ...] | None:
    root_sig = _stat_sig(path)
    if root_sig is None:
        return None
    tracked_files = (
        "run_manifest.json",
        "config.json",
        "run_summary.json",
        "status.json",
        "final.zip",
        "evaluations/latest.json",
        "evaluations/best.json",
        "logs/training.jsonl",
        "events.jsonl",
    )
    file_sigs = tuple((rel, _stat_sig(path / rel)) for rel in tracked_files)
    ckpt_sig = _stat_sig(path / "checkpoints")
    eval_sig = _stat_sig(path / "evaluations")
    return (root_sig, int(event_limit), file_sigs, ckpt_sig, eval_sig)


def _count_lines_cached(path: Path) -> int | None:
    """Line count of a log, counting only what was appended since last time.

    ``sum(1 for _ in handle)`` reads every byte, which is the wrong cost model
    for a value a GUI polls: a live run's log grows by a few hundred kilobytes
    per second and a finished run's log never changes at all. The count is
    remembered as ``(bytes, lines)`` and extended from the previous offset -
    the same append-only assumption :class:`~sandboxai.telemetry.
    IncrementalJsonlTailer` already makes for these files, and a file that
    shrank is recounted from zero rather than resumed from a stale offset.

    The previous count may have ended mid-line (a live writer's last line is
    routinely half-written). That fragment was already counted once, so it is
    subtracted before the appended bytes are counted, keeping the result
    identical to a full count.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    key = str(path)
    with _line_counts_lock:
        cached = _LINE_COUNTS.get(key)
    if cached is not None and cached[0] == size:
        return cached[1]
    lines: int | None = None
    if cached is not None and size > cached[0] > 0:
        known_size, known_lines = cached
        appended = _count_lines_from(path, known_size)
        if appended is not None:
            # Was the known prefix cut off mid-line? Then that fragment is
            # part of ``known_lines`` already and must not be counted twice.
            ends_on_newline = _byte_at(path, known_size - 1) == b"\n"
            lines = known_lines - (0 if ends_on_newline else 1) + appended
    if lines is None:
        lines = _count_lines(path)
    if lines is not None:
        with _line_counts_lock:
            _LINE_COUNTS[key] = (size, lines)
            while len(_LINE_COUNTS) > _LINE_COUNTS_LIMIT:
                _LINE_COUNTS.pop(next(iter(_LINE_COUNTS)))
    return lines


def _count_lines_from(path: Path, offset: int) -> int | None:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return sum(1 for _ in handle)
    except OSError:
        return None


def _byte_at(path: Path, offset: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(1)
    except OSError:
        return b""


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
        "has_best": (directory / "best_eval.zip").is_file() or (directory / "best.zip").is_file(),
        "has_final": (run_dir / "final.zip").is_file() or (directory / "final.zip").is_file(),
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
                    f"run used a {contract.get('observation_dim')}-float observation; "
                    f"the current contract is {OBSERVATION_FIELD_COUNT}"
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


# Keys copied verbatim out of each run document, in report order. Kept as
# module constants so the report's shape is one readable list per source
# instead of three inline tuples buried in a 140-line function.
_MANIFEST_KEYS = (
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
_CONFIG_KEYS = (
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
_SUMMARY_KEYS = (
    "timesteps",
    "training_steps_completed",
    "device",
    "stopped",
    "best_checkpoint",
    "latest_checkpoint",
    "final_checkpoint",
    "training_profile",
)


def _read_run_documents(path: Path, problems: list[str]) -> dict[str, Any]:
    """Loads the four JSON documents a run directory is expected to contain.

    Missing files are not an error (a run in progress has no summary yet);
    unreadable ones append to `problems` so the report stays complete
    instead of raising halfway through.
    """
    documents: dict[str, Any] = {}
    for key, filename in (
        ("manifest", "run_manifest.json"),
        ("config", "config.json"),
        ("summary", "run_summary.json"),
        ("control", "status.json"),
    ):
        document, problem = read_json(path / filename)
        if problem:
            problems.append(problem)
        documents[key] = document
    control = documents["control"]
    if isinstance(control, dict):
        problems.extend(f"status.json: {entry}" for entry in validate_status(control))
    return documents


def _log_inventory(path: Path) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    for relative in _LOG_FILES:
        stat = _file_stat(path / relative)
        if stat is None:
            continue
        if relative.endswith(".jsonl"):
            stat["lines"] = _count_lines_cached(path / relative)
        logs[relative] = stat
    return logs


def _picked(document: Any, keys: Sequence[str]) -> dict[str, Any]:
    """The subset of `keys` present in `document`, or {} if it is not a dict."""
    if not isinstance(document, dict):
        return {}
    return {key: document.get(key) for key in keys if key in document}


def inspect_run_checkpoints(run_dir: str | Path) -> dict[str, Any]:
    """Fast checkpoint-only inventory for ``run_dir`` without scanning JSONL logs."""
    path = Path(run_dir).expanduser()
    key = str(path)
    stamp = (
        _stat_sig(path / "checkpoints"),
        _stat_sig(path / "final.zip"),
        _stat_sig(path / "run_manifest.json"),
        _stat_sig(path / "config.json"),
    )
    with _inspect_run_cache_lock:
        cached = _INSPECT_CKPT_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    checkpoints = _checkpoint_inventory(path)
    run_id = path.name
    for filename in ("run_manifest.json", "config.json"):
        doc, _ = read_json(path / filename)
        if isinstance(doc, dict) and doc.get("run_id"):
            run_id = str(doc["run_id"])
            break
    result = {
        "run_id": run_id,
        "run_dir": str(path),
        "checkpoints": checkpoints,
    }
    with _inspect_run_cache_lock:
        _INSPECT_CKPT_CACHE[key] = (stamp, result)
        while len(_INSPECT_CKPT_CACHE) > _INSPECT_CKPT_CACHE_LIMIT:
            _INSPECT_CKPT_CACHE.pop(next(iter(_INSPECT_CKPT_CACHE)))
    return result


def inspect_run(run_dir: str | Path, event_limit: int = 0) -> dict[str, Any]:
    """Read-only report for one run directory."""
    path = Path(run_dir).expanduser()
    cache_key = (str(path), int(event_limit))
    stamp = _run_dir_stamp(path, event_limit)
    if stamp is not None:
        with _inspect_run_cache_lock:
            cached = _INSPECT_RUN_CACHE.get(cache_key)
        if cached is not None and cached[0] == stamp:
            return json.loads(cached[1])
    problems: list[str] = []

    documents = _read_run_documents(path, problems)
    manifest = documents["manifest"]
    config = documents["config"]
    summary = documents["summary"]
    control = documents["control"]

    checkpoints = _checkpoint_inventory(path)
    evaluation = _evaluation_inventory(path, problems)
    status = _status(summary, control)
    logs = _log_inventory(path)

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
        report["manifest"] = _picked(manifest, _MANIFEST_KEYS)
    if isinstance(config, dict):
        report["run_id"] = report["run_id"] or str(config.get("run_id", "") or "")
        report["experiment_id"] = report["experiment_id"] or str(
            config.get("experiment_id", "") or ""
        )
        report["config"] = _picked(config, _CONFIG_KEYS)
    if not report["run_id"]:
        report["run_id"] = path.name
    if isinstance(summary, dict):
        report["summary"] = _picked(summary, _SUMMARY_KEYS)
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
    if stamp is not None:
        with _inspect_run_cache_lock:
            _INSPECT_RUN_CACHE[cache_key] = (stamp, json.dumps(report))
            while len(_INSPECT_RUN_CACHE) > _INSPECT_RUN_CACHE_LIMIT:
                _INSPECT_RUN_CACHE.pop(next(iter(_INSPECT_RUN_CACHE)))
    return report


def inspect_runs(root: str | Path, limit: int = 0, event_limit: int = 0) -> dict[str, Any]:
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
            "{:<34} {:<11} {:>12} {:>6} {:>5} {:>9}".format(
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
                "  host           python {}, {}, {} CPUs".format(
                    host.get("python", "?"), host.get("system", "?"), host.get("logical_cpus", "?")
                )
            )
        godot = manifest.get("godot") or {}
        if godot.get("version"):
            lines.append(f"  godot          {godot['version']}")
        selection = manifest.get("checkpoint_selection") or {}
        if selection:
            lines.append(
                "  selection      {} {} (min_delta {})".format(
                    selection.get("goal"), selection.get("metric"), selection.get("min_delta")
                )
            )
    checkpoints = report.get("checkpoints", {})
    lines.append(
        "  checkpoints    {} (latest={} best={} final={})".format(
            checkpoints.get("count", 0),
            checkpoints.get("has_latest"),
            checkpoints.get("has_best"),
            checkpoints.get("has_final"),
        )
    )
    evaluation = report.get("evaluation", {})
    lines.append(
        "  evaluations    {} (latest mean reward {})".format(
            evaluation.get("evaluation_count", 0), _reward(report)
        )
    )
    best = evaluation.get("best")
    if isinstance(best, dict):
        lines.append(
            "  best           score {} at {} timesteps".format(
                best.get("score", best.get("mean_reward")), best.get("timesteps")
            )
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
