"""Stable application boundary for the SandboxAI Control Center.

This module is intentionally GUI-free.  It translates validated domain
configuration into the existing CLI commands and reads the existing run
artifacts; it does not reimplement PPO, evaluation, benchmarks, or the Godot
bridge.  Long operations are owned by :class:`ProcessManager` and are never
run on a caller's UI thread.

Everything here is read-only or process-management plumbing over data other
modules already own:

* ``run_inspection`` for historical/live run state (reused, not duplicated).
* ``telemetry`` for machine resource probes and bounded incremental JSONL
  tailing (charts poll ``telemetry_series`` instead of re-reading whole
  files).
* ``config.TrainingConfig`` for validated training configuration.
* ``runtime_validation``/``config.find_godot_executable`` for Godot
  discovery (the same logic ``sandboxai validate-runtime`` uses).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

from .artifact_repository import ArtifactRepository
from .benchmark import summarize_scaling
from .config import TrainingConfig, find_godot_executable
from .control_center_schema import DashboardSnapshot, ProcessSnapshot, RunStatus
from .run_inspection import read_json, tail_jsonl
from .telemetry import IncrementalJsonlTailer, resource_snapshot

#: Numeric telemetry.jsonl fields the Control Center can chart. Anything not
#: in this list is still visible via `telemetry()`/`inspect_run`, just not
#: accumulated into a bounded time series. Kept as one list so adding a new
#: chartable metric never requires touching the GUI layer.
SERIES_KEYS: tuple[str, ...] = (
    "timesteps",
    "steps_per_second",
    "eta_seconds",
    "episodes",
    "mean_episode_reward",
    "mean_kills",
    "mean_deaths",
    "mean_damage_dealt",
    "mean_damage_received",
    "mean_accuracy",
    "mean_survival_time",
    "win_rate",
    "loss_rate",
    "policy_shoot_request_rate",
    "cpu_percent",
    "process_rss_mb",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "gpu_utilization_percent",
    "gpu_vram_used_mb",
    "gpu_temperature_c",
    # PPO optimizer diagnostics (see ppo.PPOStatsCallback): real values SB3
    # already computes every update, never estimated here.
    "n_updates",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "entropy",
    "value_loss",
    "policy_gradient_loss",
    "loss",
)


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _modified_utc(path: Path) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    except OSError:
        return None


def _pid_tree(pid: int) -> list[int]:
    """``pid`` plus any discoverable children (best effort, needs psutil)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return [pid]
    try:
        process = psutil.Process(pid)
        return [pid] + [child.pid for child in process.children(recursive=True)]
    except (psutil.Error, OSError):  # pragma: no cover - process may have exited
        return [pid]


def _stop_process_tree(process: subprocess.Popen[str], *, hard: bool) -> None:
    """Best-effort graceful/hard stop of ``process`` and its children.

    ``Popen.terminate()``/``kill()`` only ever signal the direct child. A
    CLI subprocess (``sandboxai benchmark``/``evaluate``) itself spawns a
    Godot bridge process; without discovering that child explicitly, a
    single-process signal leaves the simulator running after the Control
    Center believes the job was cancelled. When ``psutil`` (an existing
    optional training dependency) is installed this reaches the whole
    tree; without it, only the direct child is signalled and that
    limitation is surfaced rather than silently assumed away.
    """
    for pid in _pid_tree(process.pid):
        try:
            if hard and hasattr(signal, "SIGKILL"):
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        if hard:
            process.kill()
        else:
            process.terminate()
    except OSError:
        pass


@dataclass
class ProcessRecord:
    id: str
    kind: str
    command: list[str]
    run_dir: Path
    process: subprocess.Popen[str]
    started_at: float = field(default_factory=time.time)
    returncode: int | None = None
    error: str | None = None
    #: Small, kind-specific launch summary (environment/worker counts,
    #: checkpoint path, ...) so the Agents/Dashboard pages do not need to
    #: re-parse the launched command line.
    meta: dict[str, Any] = field(default_factory=dict)
    #: (sequence, line) pairs, bounded to ProcessManager.max_lines. The
    #: sequence number is monotonic even after old lines are evicted, so a
    #: poller can ask for "everything after N" without re-reading what it
    #: already has.
    stdout: list[tuple[int, str]] = field(default_factory=list)
    stderr: list[tuple[int, str]] = field(default_factory=list)
    stdout_total: int = 0
    stderr_total: int = 0

    @property
    def state(self) -> str:
        if self.returncode is not None:
            return "finished" if self.returncode == 0 else "failed"
        return "running"


class ProcessManager:
    """Non-blocking subprocess lifecycle with bounded, thread-safe output."""

    def __init__(self, max_lines: int = 2000, finished_retention: int = 50) -> None:
        self.max_lines = max(100, int(max_lines))
        self.finished_retention = max(1, int(finished_retention))
        self._records: dict[str, ProcessRecord] = {}
        self._lock = threading.RLock()

    def start(
        self,
        kind: str,
        command: list[str],
        run_dir: Path,
        cwd: Path,
        meta: dict[str, Any] | None = None,
    ) -> ProcessRecord:
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
                env={
                    **os.environ,
                    "PYTHONPATH": str(cwd / "python")
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                },
            )
        except OSError as exc:
            raise RuntimeError(f"could not start {kind}: {exc}") from exc
        record = ProcessRecord(
            uuid.uuid4().hex, kind, list(command), run_dir, process, meta=dict(meta or {})
        )
        with self._lock:
            self._records[record.id] = record
            self._prune_finished_locked()
        for stream, is_stdout in ((process.stdout, True), (process.stderr, False)):
            threading.Thread(
                target=self._drain, args=(record, stream, is_stdout), daemon=True
            ).start()
        threading.Thread(target=self._wait, args=(record,), daemon=True).start()
        return record

    def _prune_finished_locked(self) -> None:
        """Bounds memory for a long GUI session: keep the newest N finished
        records and every still-running one, never an unbounded history."""
        finished = sorted(
            (r for r in self._records.values() if r.returncode is not None),
            key=lambda r: r.started_at,
        )
        excess = len(finished) - self.finished_retention
        for record in finished[: max(0, excess)]:
            del self._records[record.id]

    def _drain(self, record: ProcessRecord, stream: Any, is_stdout: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                with self._lock:
                    target = record.stdout if is_stdout else record.stderr
                    seq = record.stdout_total if is_stdout else record.stderr_total
                    target.append((seq, line.rstrip("\n")))
                    if is_stdout:
                        record.stdout_total += 1
                    else:
                        record.stderr_total += 1
                    if len(target) > self.max_lines:
                        del target[: len(target) - self.max_lines]
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    def _wait(self, record: ProcessRecord) -> None:
        code = record.process.wait()
        with self._lock:
            record.returncode = code
            if code and not record.error:
                record.error = f"process exited with code {code}"

    def get(self, process_id: str) -> ProcessRecord | None:
        with self._lock:
            return self._records.get(process_id)

    def list(self) -> list[ProcessRecord]:
        with self._lock:
            return list(self._records.values())

    def snapshot(self, process_id: str) -> ProcessSnapshot:
        record = self.get(process_id)
        if record is None:
            return {
                "state": "unknown",
                "error_code": "process_not_found",
                "error": "process not found",
            }
        with self._lock:
            result: ProcessSnapshot = {
                "id": record.id,
                "kind": record.kind,
                "state": record.state,
                "returncode": record.returncode,
                "started_at": record.started_at,
                # The OS pid of the process this Control Center itself
                # launched. Always safe to show (it is not a handle to
                # anything the GUI did not start) and is what the Agents
                # page needs to answer "which real process is this".
                "pid": record.process.pid,
                "stdout": [text for _, text in record.stdout[-100:]],
                "stderr": [text for _, text in record.stderr[-100:]],
                "error": record.error or "",
                "run_dir": str(record.run_dir),
                "meta": dict(record.meta),
                "command": list(record.command),
            }
        # The trainer's atomically published status is authoritative for
        # training state and progress; the OS process state is supplementary.
        status_path = record.run_dir / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8-sig"))
            if isinstance(status, dict):
                result["backend"] = cast("RunStatus", status)
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            result["backend_error_code"] = "status_unreadable"
            result["backend_error"] = f"status.json unreadable: {exc.__class__.__name__}"
        return result

    def log_since(
        self, process_id: str, stdout_after: int = -1, stderr_after: int = -1, limit: int = 1000
    ) -> dict[str, Any]:
        """Incremental stdout/stderr fetch for a live log panel.

        Only lines newer than ``stdout_after``/``stderr_after`` are
        returned, so a GUI polling at a fixed interval never re-renders
        lines it already has. If old lines were evicted (the process has
        produced more than ``max_lines`` output) the corresponding
        ``*_truncated`` flag is set so the panel can say so instead of
        silently skipping output.
        """
        record = self.get(process_id)
        if record is None:
            return {
                "error_code": "process_not_found",
                "error": "process not found",
                "stdout": [],
                "stderr": [],
            }
        with self._lock:
            new_stdout = [(seq, text) for seq, text in record.stdout if seq > stdout_after][-limit:]
            new_stderr = [(seq, text) for seq, text in record.stderr if seq > stderr_after][-limit:]
            stdout_cursor = record.stdout_total - 1
            stderr_cursor = record.stderr_total - 1
            stdout_gap = bool(record.stdout) and record.stdout[0][0] > stdout_after + 1
            stderr_gap = bool(record.stderr) and record.stderr[0][0] > stderr_after + 1
        return {
            "stdout": [text for _, text in new_stdout],
            "stderr": [text for _, text in new_stderr],
            "stdout_cursor": max(stdout_cursor, stdout_after),
            "stderr_cursor": max(stderr_cursor, stderr_after),
            "stdout_truncated": stdout_gap,
            "stderr_truncated": stderr_gap,
        }

    def cancel(self, process_id: str) -> ProcessSnapshot:
        """Requests a safe stop without blocking the caller.

        Training uses the existing cooperative command-file protocol (see
        ``run_control.RunControl``): the trainer keeps running until its
        next safe callback boundary, saves final checkpoints, and exits on
        its own. Anything else receives a graceful terminate. Neither
        branch waits for exit — the background reaper thread already
        updates process state asynchronously — so this is safe to call
        directly from a GUI event handler.
        """
        record = self.get(process_id)
        if record is None:
            return {
                "state": "unknown",
                "error_code": "process_not_found",
                "error": "process not found",
            }
        if record.returncode is None:
            if record.kind == "training":
                command_file = record.run_dir / "command.json"
                try:
                    command_file.write_text(
                        json.dumps({"command": "stop", "sequence": time.time_ns()}) + "\n",
                        encoding="utf-8",
                    )
                except OSError as exc:
                    record.error = f"could not write stop command: {exc.__class__.__name__}"
            else:
                _stop_process_tree(record.process, hard=False)
        return self.snapshot(process_id)

    def force_stop(self, process_id: str) -> ProcessSnapshot:
        """Immediately kills a process and its discoverable children.

        Skips cooperative shutdown: for a training process this means no
        final checkpoint is saved. Intended as an escalation when
        ``cancel()`` makes no progress, never the default action.
        """
        record = self.get(process_id)
        if record is None:
            return {
                "state": "unknown",
                "error_code": "process_not_found",
                "error": "process not found",
            }
        if record.returncode is None:
            _stop_process_tree(record.process, hard=True)
        return self.snapshot(process_id)

    # Backward-compatible blocking variant, used only at application
    # shutdown where a bounded wait is acceptable.
    def terminate(self, process_id: str, timeout: float = 5.0) -> ProcessSnapshot:
        record = self.get(process_id)
        if record is None:
            return {
                "state": "unknown",
                "error_code": "process_not_found",
                "error": "process not found",
            }
        if record.returncode is None:
            self.cancel(process_id)
            try:
                record.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.force_stop(process_id)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    record.process.wait(timeout=timeout)
        return self.snapshot(process_id)

    def close(self, timeout: float = 5.0) -> None:
        for record in list(self._records.values()):
            if record.returncode is None:
                self.terminate(record.id, timeout=timeout)


class _RunSeries:
    """Bounded, incrementally-updated numeric time series for one run.

    Backs :meth:`SandboxAIAdapter.telemetry_series`. Each poll only reads
    the bytes appended to ``logs/training.jsonl`` since the previous poll
    (:class:`~sandboxai.telemetry.IncrementalJsonlTailer`); every per-metric
    series is a ``deque(maxlen=...)`` so charts stay bounded no matter how
    long the run has been going.
    """

    def __init__(self, path: Path, max_points: int) -> None:
        self.path = path
        self.max_points = max_points
        self.tailer = IncrementalJsonlTailer(path)
        self.series: dict[str, deque[tuple[float, float]]] = {
            key: deque(maxlen=max_points) for key in SERIES_KEYS
        }
        self.latest: dict[str, Any] = {}
        self.warnings: deque[str] = deque(maxlen=50)

    def poll(self) -> int:
        rows = self.tailer.read_new()
        for row in rows:
            x = row.get("timesteps")
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                continue
            for key in SERIES_KEYS:
                if key == "timesteps":
                    continue
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.series[key].append((float(x), float(value)))
            self.series["timesteps"].append((float(x), float(x)))
            self.latest = row
            if row.get("event") == "checkpoint_selection_metric_missing":
                self.warnings.append(
                    f"checkpoint-selection metric '{row.get('metric')}' missing at {x} timesteps"
                )
        return len(rows)


class SandboxAIAdapter:
    """Application-facing operations over the real SandboxAI infrastructure."""

    def __init__(
        self, project_root: str | Path | None = None, output_root: str | Path = "training"
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        # Two real bugs fixed here: (1) an already-absolute output_root used
        # to skip .resolve() entirely, so it could disagree (as a plain
        # string) with anything downstream that does resolve it - e.g.
        # TrainingConfig.run_directory(); (2) "~/..." is not Path.is_absolute()
        # in pathlib, so it used to fall into the *relative* branch and get
        # literally joined onto project_root as a folder named "~" instead of
        # expanding to the home directory.
        expanded_output_root = Path(output_root).expanduser()
        self.output_root = (
            expanded_output_root.resolve()
            if expanded_output_root.is_absolute()
            else (self.project_root / expanded_output_root).resolve()
        )
        self.processes = ProcessManager()
        self.artifacts = ArtifactRepository(self.output_root)
        self._series_cache: OrderedDict[str, _RunSeries] = OrderedDict()
        self._series_cache_limit = 6

    def _python_command(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "sandboxai", *args]

    # ------------------------------------------------------------------
    # System / dependency status
    # ------------------------------------------------------------------

    def system_status(self) -> dict[str, Any]:
        status = resource_snapshot()
        status.update(
            {
                "python": sys.executable,
                "python_version": sys.version.split()[0],
                "project_root": str(self.project_root),
            }
        )
        godot_raw = os.environ.get("GODOT_EXECUTABLE") or os.environ.get("GODOT_PATH") or "godot"
        try:
            from .runtime_validation import RuntimeValidator

            validator = RuntimeValidator(project_path=self.project_root, godot_executable=godot_raw)
            available = validator.is_godot_available()
            resolved = find_godot_executable(godot_raw)
            status["godot_executable"] = godot_raw
            status["godot_resolved_executable"] = resolved
            status["godot_available"] = available
            status["godot_version"] = validator.probe_version(resolved) if available else None
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            status["godot_executable"] = godot_raw
            status["godot_available"] = False
            status["godot_version"] = None
            status["godot_error"] = str(exc)
        status["dependencies"] = {
            name: _module_available(name)
            for name in (
                "numpy",
                "torch",
                "gymnasium",
                "stable_baselines3",
                "tensorboard",
                "psutil",
            )
        }
        return status

    # ------------------------------------------------------------------
    # Runs, checkpoints, evaluations, benchmarks (read-only)
    # ------------------------------------------------------------------

    def project_status(self) -> dict[str, Any]:
        index = self.artifacts.list_runs(limit=1)
        return {
            "output_root": str(self.output_root),
            "runs": index,
            "active_processes": self.list_processes(active_only=True),
            "processes": self.list_processes(active_only=False),
        }

    def dashboard_snapshot(self) -> DashboardSnapshot:
        """Everything the Dashboard page needs, assembled from existing
        read-only inspection and the process registry. Computes nothing of
        its own beyond picking the newest run.

        Only the latest run gets a full ``inspect_run`` report (config,
        checkpoints, evaluation, control, warnings); ``run_count`` is a
        cheap directory listing so this stays fast no matter how many runs
        exist, unlike calling ``inspect_runs`` with a small ``limit``
        (which reports the *sliced* count, not the true total).
        """
        return self.artifacts.dashboard(lambda: self.list_processes(active_only=True))

    def list_runs(self, limit: int = 100) -> dict[str, Any]:
        return self.artifacts.list_runs(limit)

    def inspect_run(self, run: str | Path, event_limit: int = 50) -> dict[str, Any]:
        return self.artifacts.inspect_run(run, event_limit)

    def list_checkpoints(self, run: str | Path) -> dict[str, Any]:
        report = self.inspect_run(run)
        return report.get("checkpoints", {}) if isinstance(report, dict) else {}

    def discover_checkpoints(self, limit: int = 300) -> list[dict[str, Any]]:
        """Newest-first checkpoint inventory from the central artifact repository."""
        return self.artifacts.checkpoints(limit)

    def run_metrics(self, run: str | Path) -> dict[str, Any]:
        report = self.inspect_run(run, event_limit=100)
        return {
            "progress": report.get("progress", {}),
            "evaluation": report.get("evaluation", {}),
            "control": report.get("control", {}),
            "events": report.get("recent_events", []),
        }

    def telemetry(self, run: str | Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
        path = self._resolve_training_log(run)
        return tail_jsonl(path, limit) if path is not None else []

    def telemetry_series(
        self, run: str | Path | None = None, max_points: int = 1500
    ) -> dict[str, Any]:
        """Bounded, incrementally-updated per-metric time series for charts.

        Repeated polling only costs what changed since the previous call
        (:class:`~sandboxai.telemetry.IncrementalJsonlTailer`), and every
        series is capped at ``max_points`` regardless of run length.
        """
        path = self._resolve_training_log(run)
        if path is None:
            return {
                "available": False,
                "reason": "no training run with logs/training.jsonl was found",
                "series": {},
            }
        key = f"{path}::{max_points}"
        cached = self._series_cache.get(key)
        if cached is None:
            cached = _RunSeries(path, max_points)
            self._series_cache[key] = cached
        self._series_cache.move_to_end(key)
        while len(self._series_cache) > self._series_cache_limit:
            self._series_cache.popitem(last=False)
        cached.poll()
        return {
            "available": True,
            "path": str(path),
            "series": {name: list(points) for name, points in cached.series.items() if points},
            "latest": dict(cached.latest),
            "warnings": list(cached.warnings),
        }

    def _resolve_training_log(self, run: str | Path | None) -> Path | None:
        return self.artifacts.resolve_training_log(run)

    def profiling(self, run: str | Path) -> dict[str, Any]:
        path = Path(run) / "logs" / "training_profile.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {"error": "profile is not an object"}
        except FileNotFoundError:
            return {"available": False, "reason": "profiling not enabled or run incomplete"}
        except (OSError, json.JSONDecodeError) as exc:
            return {"available": False, "error": str(exc)}

    def discover_evaluations(
        self, run: str | Path | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Lightweight, newest-first index of evaluation summaries.

        Looks at a run's ``evaluations/`` tree (``latest.json`` plus every
        checkpoint-battery ``step_*/summary.json``) or, with no run given,
        every discovered run's evaluations plus the adapter's own default
        evaluation output root. Each entry is cheap; call
        :meth:`evaluation_detail` for the full structured summary.
        """
        roots = self.artifacts.evaluation_roots(run)
        entries: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.is_dir():
                continue
            candidates = [
                root / "latest.json",
                *sorted(root.glob("step_*/summary.json")),
                *sorted(root.glob("*/summary.json")),
            ]
            for path in candidates:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                value, problem = read_json(path)
                entry: dict[str, Any] = {
                    "path": str(path),
                    "run_directory": str(root.parent if root.name == "evaluations" else root),
                    "modified_utc": _modified_utc(path),
                    "problem": problem,
                }
                if isinstance(value, dict):
                    for key in (
                        "timesteps",
                        "episodes",
                        "mean_episode_reward",
                        "win_rate",
                        "loss_rate",
                    ):
                        entry[key] = value.get(key)
                entries.append(entry)
        entries.sort(key=lambda item: item.get("modified_utc") or "", reverse=True)
        return entries[:limit]

    def evaluation_detail(self, path: str | Path) -> dict[str, Any]:
        """Full structured evaluation summary read straight from disk."""
        target = Path(path)
        value, problem = read_json(target)
        if problem:
            return {"available": False, "error": problem, "path": str(target)}
        if not isinstance(value, dict):
            return {
                "available": False,
                "error": "evaluation summary is not a JSON object",
                "path": str(target),
            }
        result = dict(value)
        result["available"] = True
        result["path"] = str(target)
        return result

    def benchmark_results(
        self, directory: str | Path = "training/benchmarks/latest"
    ) -> dict[str, Any]:
        path = Path(directory)
        if not path.is_absolute():
            path = self.project_root / path
        try:
            rows = json.loads((path / "benchmark.json").read_text(encoding="utf-8-sig"))
            return {
                "results": rows,
                "scaling": summarize_scaling(rows) if isinstance(rows, list) else {},
                "directory": str(path),
            }
        except (OSError, json.JSONDecodeError) as exc:
            return {"results": [], "error": str(exc), "directory": str(path)}

    def benchmark_history(
        self, root: str | Path = "training/benchmarks", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Every benchmark result set under ``root``, newest first."""
        base = Path(root)
        if not base.is_absolute():
            base = self.project_root / base
        if not base.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for path in base.rglob("benchmark.json"):
            directory = path.parent
            result = self.benchmark_results(directory)
            entries.append(
                {"directory": str(directory), "modified_utc": _modified_utc(path), **result}
            )
        entries.sort(key=lambda item: item.get("modified_utc") or "", reverse=True)
        return entries[:limit]

    # ------------------------------------------------------------------
    # Process registry
    # ------------------------------------------------------------------

    def list_processes(self, active_only: bool = False) -> list[ProcessSnapshot]:
        records = self.processes.list()
        if active_only:
            records = [r for r in records if r.returncode is None]
        records.sort(key=lambda r: r.started_at, reverse=True)
        return [self.processes.snapshot(r.id) for r in records]

    def process_log(
        self, process_id: str, stdout_after: int = -1, stderr_after: int = -1
    ) -> dict[str, Any]:
        return self.processes.log_since(process_id, stdout_after, stderr_after)

    # ------------------------------------------------------------------
    # Launching real work
    # ------------------------------------------------------------------

    def _managed_training(self, config: TrainingConfig) -> tuple[Path, list[str]]:
        config.validate()
        run_dir = config.run_directory()
        if not run_dir.is_absolute():
            run_dir = self.project_root / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        # Persist the exact domain config (the trainer also writes its own copy).
        config.save(run_dir / "config.json")
        command = self._python_command(
            "train",
            "--config",
            str(run_dir / "config.json"),
            "--control-file",
            str(run_dir / "command.json"),
            "--status-file",
            str(run_dir / "status.json"),
            "--event-log-file",
            str(run_dir / "events.jsonl"),
        )
        return run_dir, command

    def start_training(self, config: TrainingConfig | dict[str, Any]) -> dict[str, Any]:
        cfg = config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config)
        run_dir, command = self._managed_training(cfg)
        meta = {
            "run_id": cfg.run_id or run_dir.name,
            "environment_count": cfg.environment_count,
            "env_workers": cfg.resolved_env_workers(),
            "total_training_steps": cfg.total_training_steps,
            "rollout_length": cfg.resolved_rollout_length(),
            "device": cfg.device,
            "curriculum_mode": cfg.curriculum_mode,
        }
        record = self.processes.start("training", command, run_dir, self.project_root, meta=meta)
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def start_benchmark(
        self,
        *,
        environment_counts: Iterable[int],
        worker_counts: Iterable[int] = (1,),
        steps: int = 2000,
        enemy_count: int = 1,
        compact_infos: bool = True,
        output_dir: str | Path = "training/benchmarks/latest",
    ) -> dict[str, Any]:
        environment_counts = [int(value) for value in environment_counts]
        worker_counts = [int(value) for value in worker_counts]
        if not environment_counts or any(value < 1 for value in environment_counts):
            raise ValueError("environment_counts must contain at least one positive integer")
        if not worker_counts or any(value < 1 for value in worker_counts):
            raise ValueError("worker_counts must contain at least one positive integer")
        if steps < 1:
            raise ValueError("steps must be positive")
        if enemy_count < 1:
            raise ValueError("enemy_count must be >= 1")
        directory = Path(output_dir)
        directory = directory if directory.is_absolute() else self.project_root / directory
        command = self._python_command(
            "benchmark",
            "--env-counts",
            ",".join(map(str, environment_counts)),
            "--worker-counts",
            ",".join(map(str, worker_counts)),
            "--steps",
            str(steps),
            "--enemy-count",
            str(enemy_count),
            "--output-dir",
            str(directory),
            *([] if compact_infos else ["--full-infos"]),
        )
        meta = {
            "environment_counts": environment_counts,
            "worker_counts": worker_counts,
            "steps": steps,
            "enemy_count": enemy_count,
            "compact_infos": compact_infos,
        }
        record = self.processes.start("benchmark", command, directory, self.project_root, meta=meta)
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def start_evaluation(
        self,
        checkpoint: str | Path,
        *,
        episodes: int = 20,
        environment_count: int = 1,
        device: str = "auto",
        output_dir: str | Path = "training/evaluations/control_center",
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        if environment_count < 1:
            raise ValueError("environment_count must be >= 1")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be one of auto, cpu, cuda")
        directory = Path(output_dir)
        directory = directory if directory.is_absolute() else self.project_root / directory
        command = self._python_command(
            "evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(episodes),
            "--env-count",
            str(environment_count),
            "--device",
            device,
            "--output-dir",
            str(directory),
        )
        meta = {
            "checkpoint": str(checkpoint_path),
            "episodes": episodes,
            "environment_count": environment_count,
            "device": device,
        }
        record = self.processes.start(
            "evaluation", command, directory, self.project_root, meta=meta
        )
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def process_status(self, process_id: str) -> ProcessSnapshot:
        return self.processes.snapshot(process_id)

    def cancel(self, process_id: str) -> ProcessSnapshot:
        return self.processes.cancel(process_id)

    def force_stop(self, process_id: str) -> ProcessSnapshot:
        return self.processes.force_stop(process_id)

    def close(self) -> None:
        self.processes.close()
