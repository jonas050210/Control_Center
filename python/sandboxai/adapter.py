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
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
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
    "learning_rate",
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
    #: checkpoint path, ...) so the Training/Dashboard pages do not need to
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
            error_text = record.error or ""
            if error_text.startswith("process exited with code"):
                # A bare exit code is useless in an error column. The last
                # non-empty stderr line is the process's own account of why
                # it died; appending it here (instead of in _wait) avoids
                # racing the stderr drain thread, which may still be
                # delivering that line when wait() returns.
                detail = next(
                    (text.strip() for _, text in reversed(record.stderr) if text.strip()), ""
                )
                if detail:
                    error_text = f"{error_text}: {detail}"
            result: ProcessSnapshot = {
                "id": record.id,
                "kind": record.kind,
                "state": record.state,
                "returncode": record.returncode,
                "started_at": record.started_at,
                # The OS pid of the process this Control Center itself
                # launched. Always safe to show (it is not a handle to
                # anything the GUI did not start) and is what the Training
                # page needs to answer "which real process is this".
                "pid": record.process.pid,
                "stdout": [text for _, text in record.stdout[-100:]],
                "stderr": [text for _, text in record.stderr[-100:]],
                "error": error_text,
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

    def send_command(self, process_id: str, command: str) -> ProcessSnapshot:
        """Write one cooperative command (``pause``/``resume``/``stop``).

        Only training processes speak the command-file protocol, so this
        refuses every other kind instead of writing a file nothing reads.
        Like :meth:`cancel` it never blocks: the trainer acknowledges at
        its next safe callback boundary.
        """
        if command not in ("pause", "resume", "stop"):
            raise ValueError(f"unsupported control command: {command!r}")
        record = self.get(process_id)
        if record is None:
            return {
                "state": "unknown",
                "error_code": "process_not_found",
                "error": "process not found",
            }
        if record.kind != "training":
            return self.snapshot(process_id) | {
                "error_code": "unsupported_command",
                "error": f"process kind '{record.kind}' does not support the "
                f"cooperative '{command}' command",
            }
        if record.returncode is not None:
            return self.snapshot(process_id) | {
                "error_code": "process_not_running",
                "error": "the process has already exited",
            }
        command_file = record.run_dir / "command.json"
        try:
            command_file.write_text(
                json.dumps({"command": command, "sequence": time.time_ns()}) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            return self.snapshot(process_id) | {
                "error_code": "command_unwritable",
                "error": f"could not write the '{command}' command: {exc.__class__.__name__}",
            }
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
        # Agent lifecycle layer over the process registry. Constructed after
        # the methods it calls exist; the manager only ever calls the three
        # launch methods through the AgentLauncher protocol.
        from .agents import AgentManager

        self.agents = AgentManager(self.processes, self)
        self._series_cache: OrderedDict[str, _RunSeries] = OrderedDict()
        self._series_cache_limit = 6
        #: Stats page: replay header + tick count (one cheap pass, see
        #: ``_replay_meta``) and one parsed episode, both keyed by
        #: (path, size, mtime) so a poll never re-reads what has not changed.
        #: The metadata limit is large on purpose: an entry is a small tuple,
        #: and the page rescans the folder on a timer, so the whole corpus
        #: stays warm instead of only the most recently read files.
        self._replay_meta_cache: OrderedDict[
            str, tuple[tuple[int, int], dict[str, Any] | None, str, int]
        ] = OrderedDict()
        self._replay_meta_cache_limit = 2048
        self._replay_cache: OrderedDict[str, tuple[tuple[int, int], dict[str, Any], Any]] = (
            OrderedDict()
        )
        self._replay_cache_limit = 2
        self._eval_file_cache: OrderedDict[str, tuple[tuple[int, int], dict[str, Any]]] = (
            OrderedDict()
        )
        self._eval_file_cache_limit = 1024
        #: executable -> (monotonic time, runtime facts) for compatibility
        #: checks; probing the Godot version is a subprocess call, too slow
        #: for every poll. Keyed by the requested executable (None = the
        #: default resolution chain) so a form override is probed honestly.
        self._runtime_status: dict[str | None, tuple[float, dict[str, Any]]] = {}

    def _python_command(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "sandboxai", *args]

    def ensure_simulated_godot_executable(self) -> str:
        """Materialize a local cross-platform stand-in speaking the headless rl_server.gd JSONL protocol.

        Used as an automatic fallback by GUI launches and benchmark sweeps when no
        external Godot 4 binary is installed or found on the host machine, ensuring
        training, agent quick-start, and benchmark workflows always run out of the box.
        """
        import stat as _stat

        from .contract import ACTION_NVEC, GODOT_VERSION, OBSERVATION_FIELD_COUNT

        runtime_dir = self.output_root / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        script_path = runtime_dir / "simulated_godot_bridge.py"
        script_source = (
            "from __future__ import annotations\n"
            "import json, math, sys\n\n"
            f"OBS_DIM = {int(OBSERVATION_FIELD_COUNT)}\n"
            f"ACTION_NVEC = {list(ACTION_NVEC)!r}\n"
            f"VERSION_STR = {f'{GODOT_VERSION}.stable.official.simulated'!r}\n\n"
            "if '--version' in sys.argv:\n"
            "    print(VERSION_STR, flush=True)\n"
            "    raise SystemExit(0)\n\n"
            "args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []\n"
            "def _arg(flag: str, default: str) -> str:\n"
            "    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default\n\n"
            "env_count = max(1, int(_arg('--env-count', '1')))\n"
            "enemy_count = max(1, int(_arg('--enemy-count', '1')))\n"
            "seed = int(_arg('--seed', '1234'))\n"
            "curriculum_level = int(_arg('--curriculum-level', '1'))\n"
            "self_play = _arg('--self-play', '0') in ('1', 'true', 'True')\n"
            "map_id = 'arena_01'\n"
            "layout_id = 'standard'\n"
            "lighting_mode = 'day'\n"
            "step_count = 0\n"
            "plans = []\n\n"
            "def _obs(step: int, idx: int) -> list[float]:\n"
            "    phase = (step + idx) * 0.05\n"
            "    vec = [0.0] * OBS_DIM\n"
            "    vec[0] = round(math.sin(phase) * 0.5, 4)\n"
            "    vec[1] = round(math.cos(phase) * 0.5, 4)\n"
            "    if OBS_DIM > 9:\n"
            "        vec[9] = 1.0\n"
            "    if OBS_DIM > 16:\n"
            "        vec[16] = 1.0\n"
            "    return vec\n\n"
            "def _info(step: int, done: bool, reward: float, shot: bool) -> dict:\n"
            "    won = bool(done and reward >= 0.25)\n"
            "    return {\n"
            "        'step': step,\n"
            "        'episode_steps': step % 32,\n"
            "        'episode_reward': round(reward * 4.0, 4),\n"
            "        'kills': 1 if won else 0,\n"
            "        'deaths': 0 if won else (1 if done else 0),\n"
            "        'won': won,\n"
            "        'lost': bool(done and not won),\n"
            "        'damage_dealt': 100.0 if won else (34.0 if shot else 10.0),\n"
            "        'damage_taken': 25.0,\n"
            "        'shots_fired': 4 if shot else 1,\n"
            "        'shots_hit': 3 if won else 1,\n"
            "        'shot_requested': 1 if shot else 0,\n"
            "        'shot_discharged': 1 if shot else 0,\n"
            "        'engagement_occurred': True,\n"
            "        'First Engagement Time': 1.2,\n"
            "        'curriculum_level': curriculum_level,\n"
            "    }\n\n"
            "for raw in sys.stdin:\n"
            "    raw = raw.strip()\n"
            "    if not raw:\n"
            "        continue\n"
            "    msg = json.loads(raw)\n"
            "    cmd = msg.get('cmd')\n"
            "    if cmd == 'spaces':\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'observation_size': OBS_DIM,\n"
            "            'observation_space': {'type': 'Box', 'size': OBS_DIM, 'low': -1.0, 'high': 1.0},\n"
            "            'action_space': {'type': 'MultiDiscrete', 'nvec': ACTION_NVEC},\n"
            "            'environment_count': env_count,\n"
            "            **({'policy_slots': 2} if self_play else {}),\n"
            "        }\n"
            "    elif cmd == 'set_map':\n"
            "        map_id = str(msg.get('map_id', map_id))\n"
            "        out = {'ok': True, 'map_id': map_id}\n"
            "    elif cmd == 'set_layout':\n"
            "        layout_id = str(msg.get('layout_id', layout_id))\n"
            "        out = {'ok': True, 'layout_id': layout_id}\n"
            "    elif cmd == 'set_lighting':\n"
            "        lighting_mode = str(msg.get('lighting', lighting_mode))\n"
            "        out = {'ok': True, 'lighting': lighting_mode}\n"
            "    elif cmd == 'ping':\n"
            "        out = {'ok': True, 'pong': True, 'step_count': step_count, 'environment_count': env_count}\n"
            "    elif cmd in ('configure_episodes', 'set_episode_plan', 'set_episode_plans'):\n"
            "        plans = list(msg.get('plans') or msg.get('episodes') or [])\n"
            "        staged = [int(p.get('index', idx)) if isinstance(p, dict) else idx for idx, p in enumerate(plans)]\n"
            "        out = {'ok': True, 'configured': len(plans), 'staged': staged}\n"
            "    elif cmd == 'episode_conditions':\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'conditions': [\n"
            "                {\n"
            "                    'index': i,\n"
            "                    'seed': seed + i,\n"
            "                    'map_id': 'arena_01',\n"
            "                    'scenario': 'duel',\n"
            "                    'lighting': 'day',\n"
            "                    'enemy_count': enemy_count,\n"
            "                    'curriculum_level': curriculum_level,\n"
            "                }\n"
            "                for i in range(env_count)\n"
            "            ],\n"
            "        }\n"
            "    elif cmd == 'reset_indices':\n"
            "        indices = [int(i) for i in (msg.get('indices') or [])]\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'results': [\n"
            "                {'index': idx, 'observation': _obs(step_count, idx), 'info': _info(step_count, False, 0.0, False)}\n"
            "                for idx in indices\n"
            "            ],\n"
            "        }\n"
            "    elif cmd == 'metrics':\n"
            "        out = {'ok': True, 'metrics': [_info(step_count, False, 0.2, True) for _ in range(env_count)]}\n"
            "    elif cmd == 'reward_breakdown':\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'breakdowns': [\n"
            "                {'damage_dealt': 0.4, 'kill_bonus': 0.3, 'survival': 0.1, 'accuracy': 0.2}\n"
            "                for _ in range(env_count)\n"
            "            ],\n"
            "        }\n"
            "    elif cmd == 'set_curriculum':\n"
            "        curriculum_level = int(msg.get('level', curriculum_level))\n"
            "        out = {'ok': True, 'curriculum_level': curriculum_level}\n"
            "    elif cmd == 'health_check':\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'health': [\n"
            "                {'index': i, 'ok': True, 'agent_alive': True, 'enemies_alive': enemy_count}\n"
            "                for i in range(env_count)\n"
            "            ],\n"
            "        }\n"
            "    elif cmd == 'profile_snapshot':\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'profile': {\n"
            "                'available': True,\n"
            "                'timings': {\n"
            "                    'command_step': {'count': max(1, step_count), 'total_seconds': round(step_count * 0.0004, 6), 'mean_ms': 0.4, 'min_ms': 0.2, 'max_ms': 0.9},\n"
            "                    'response_encode': {'count': max(1, step_count), 'total_seconds': round(step_count * 0.0001, 6), 'mean_ms': 0.1, 'min_ms': 0.05, 'max_ms': 0.25},\n"
            "                },\n"
            "                'counters': {'request_bytes': step_count * 96, 'response_bytes': step_count * 512},\n"
            "            },\n"
            "        }\n"
            "    elif cmd == 'reset':\n"
            "        step_count = 0\n"
            "        out = {'ok': True, 'observations': [_obs(0, i) for i in range(env_count)], 'infos': [_info(0, False, 0.0, False) for _ in range(env_count)]}\n"
            "    elif cmd == 'step':\n"
            "        step_count += 1\n"
            "        actions = msg.get('actions') or [[0] * len(ACTION_NVEC) for _ in range(env_count)]\n"
            "        done = (step_count % 32 == 0)\n"
            "        rewards = []\n"
            "        infos = []\n"
            "        for i in range(env_count):\n"
            "            act = actions[i] if i < len(actions) and isinstance(actions[i], list) else [0] * len(ACTION_NVEC)\n"
            "            shot = bool(len(act) > 4 and int(act[4]) == 1)\n"
            "            r = round(0.15 + (0.2 if shot else 0.0) + (0.02 * ((i % 3) - 1)), 4)\n"
            "            rewards.append(r)\n"
            "            infos.append(_info(step_count, done, r, shot))\n"
            "        out = {\n"
            "            'ok': True,\n"
            "            'observations': [_obs(step_count, i) for i in range(env_count)],\n"
            "            'rewards': rewards,\n"
            "            'terminated': [done] * env_count,\n"
            "            'truncated': [False] * env_count,\n"
            "            'dones': [done] * env_count,\n"
            "            'infos': infos,\n"
            "        }\n"
            "    elif cmd == 'close':\n"
            "        print(json.dumps({'ok': True, 'close': True, 'closed': True}), flush=True)\n"
            "        break\n"
            "    else:\n"
            "        out = {'ok': False, 'error': f'unknown command: {cmd}'}\n"
            "    print(json.dumps(out), flush=True)\n"
        )
        script_path.write_text(script_source, encoding="utf-8")
        if os.name == "nt":
            wrapper = runtime_dir / "simulated_godot_bridge.cmd"
            wrapper.write_text(
                f'@echo off\r\n"{sys.executable}" -u "{script_path}" %*\r\n',
                encoding="utf-8",
            )
            return str(wrapper)
        wrapper = runtime_dir / "simulated_godot_bridge.sh"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -u "{script_path}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
        return str(wrapper)

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
        status.update(self.training_dependency_status())
        return status

    def training_dependency_status(self) -> dict[str, Any]:
        """Whether training can start on this interpreter, and the fix if not.

        A missing optional extra used to surface as a traceback from inside
        ``train_ppo``. The window needs the answer *before* an operator presses
        Launch, and it needs the exact command, because "install the training
        extras" is not a string anybody should have to reconstruct.
        """
        from .ppo import TRAINING_INSTALL_HINT, missing_training_dependencies

        missing = missing_training_dependencies()
        return {
            "training_ready": not missing,
            "training_missing": list(missing),
            "training_install_hint": TRAINING_INSTALL_HINT,
        }

    def install_training_extras(self, *, timeout_seconds: float = 1800.0) -> dict[str, Any]:
        """Run ``pip install -e '.[training]'`` in this checkout.

        Deliberately not a silent repair: it is the operator's own button, it
        runs in their interpreter, and its whole output comes back so the
        window can show what actually happened instead of a success flag.
        """
        command = [sys.executable, "-m", "pip", "install", "-e", ".[training]"]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ok": False,
                "command": " ".join(command),
                "returncode": None,
                "error": str(exc),
                "output": "",
            }
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        return {
            "ok": completed.returncode == 0,
            "command": " ".join(command),
            "returncode": completed.returncode,
            "error": None
            if completed.returncode == 0
            else f"pip exited with {completed.returncode}",
            "output": output[-4000:],
        }

    # ------------------------------------------------------------------
    # Hardware profile / first-start wizard
    # ------------------------------------------------------------------

    def _hardware_profile_path(self) -> Path:
        """Where this adapter's project keeps its hardware profile.

        Anchored to ``project_root`` rather than the installed package, so an
        adapter pointed at a specific checkout reads and writes that
        checkout's profile — and tests are hermetic under a temp root.
        """
        return self.project_root / ".sandboxai" / "hardware_profile.json"

    def hardware_profile(self) -> dict[str, Any] | None:
        """The persisted hardware profile, or ``None`` if the wizard has not run.

        A thin pass-through to :mod:`sandboxai.hardware_profile` so the GUI
        never grows its own profile-reading logic.
        """
        from .hardware_profile import load_profile

        profile = load_profile(self._hardware_profile_path())
        return profile.to_dict() if profile is not None else None

    def hardware_candidates(self) -> list[dict[str, str]]:
        """The device candidates measurable on this host (CPU-only or all three)."""
        from .hardware_profile import available_candidates

        return [
            {
                "label": candidate.label,
                "device": candidate.device,
                "inference_device": candidate.inference_device,
                "description": candidate.description,
            }
            for candidate in available_candidates()
        ]

    def run_hardware_wizard(
        self,
        *,
        godot_executable: str | None = None,
        steps: int | None = None,
        cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Measure the available devices and persist a hardware profile.

        The single measurement implementation lives in
        :mod:`sandboxai.hardware_profile`; this only forwards the GUI's
        cancel/progress hooks and the project root. Never raises for a
        missing engine — that becomes an honest ``fallback`` profile.
        """
        from .hardware_profile import (
            DEFAULT_MEASUREMENT_STEPS,
            DeviceMeasurement,
            run_hardware_wizard,
        )

        progress_adapter = None
        if on_progress is not None:

            def progress_adapter(measurement: DeviceMeasurement) -> None:
                on_progress(measurement.to_dict())

        profile = run_hardware_wizard(
            project_path=self.project_root,
            godot_executable=godot_executable,
            steps=steps if steps is not None else DEFAULT_MEASUREMENT_STEPS,
            cancel=cancel,
            on_progress=progress_adapter,
            save=save,
            save_path=self._hardware_profile_path() if save else None,
        )
        return profile.to_dict()

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

    def export_run_report(
        self,
        run: str | Path,
        destination: str | Path | None = None,
    ) -> dict[str, Any]:
        """Export a human-readable Markdown + JSON provenance report for ``run``."""
        from .run_inspection import format_run_report

        report = self.inspect_run(run, event_limit=25)
        run_dir = Path(report.get("run_dir") or run).expanduser()
        target = (
            Path(destination).expanduser()
            if destination is not None
            else (run_dir / "run_report.md")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        run_id = str(report.get("run_id") or run_dir.name)
        status = report.get("status") or {}
        progress = report.get("progress") or {}
        evaluation = report.get("evaluation") or {}
        checkpoints = report.get("checkpoints") or {}
        config = report.get("config") or {}
        lines = [
            f"# SandboxAI Run Report — `{run_id}`",
            "",
            f"- **Directory:** `{run_dir}`",
            f"- **State:** `{status.get('state', 'unknown')}` (source: `{status.get('source', 'n/a')}`)",
            f"- **Timesteps:** `{progress.get('timesteps', 'n/a')}` / `{progress.get('target_timesteps', 'n/a')}`",
            f"- **Checkpoints:** `{checkpoints.get('count', 0)}` (`latest={checkpoints.get('has_latest')}`, `best={checkpoints.get('has_best')}`, `final={checkpoints.get('has_final')}`)",
            f"- **Evaluations:** `{evaluation.get('evaluation_count', 0)}`",
            "",
            "## Configuration",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
            "```",
            "",
            "## Inspection Summary",
            "```text",
            format_run_report(report),
            "```",
            "",
        ]
        markdown = "\n".join(lines)
        target.write_text(markdown, encoding="utf-8")
        return {
            "ok": True,
            "run_id": run_id,
            "path": str(target),
            "markdown": markdown,
        }

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
                *sorted(root.glob("*/summary.json")),
            ]
            for path in candidates:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                cache_key = str(path)
                try:
                    st = path.stat()
                    stamp = (st.st_mtime_ns, st.st_size)
                except OSError:
                    continue
                cached = self._eval_file_cache.get(cache_key)
                if cached is not None and cached[0] == stamp:
                    entries.append(dict(cached[1]))
                    continue
                value, problem = read_json(path)
                entry: dict[str, Any] = {
                    "path": cache_key,
                    "run_directory": str(root.parent if root.name == "evaluations" else root),
                    "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
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
                self._eval_file_cache[cache_key] = (stamp, dict(entry))
                while len(self._eval_file_cache) > self._eval_file_cache_limit:
                    self._eval_file_cache.popitem(last=False)
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
    # Benchmark pipeline (staged sizing benchmark + recommendation)
    # ------------------------------------------------------------------

    def run_benchmark_pipeline(
        self,
        *,
        budget_mode: str = "time",
        steps: int | None = None,
        minutes: float | None = None,
        environment_counts: list[int] | None = None,
        worker_counts: list[int] | None = None,
        finalists: int | None = None,
        godot_executable: str | None = None,
        cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the staged benchmark pipeline on this machine.

        All measurement lives in
        :mod:`sandboxai.benchmark_pipeline` (which itself reuses
        ``benchmark.benchmark_simulation`` and ``hardware_profile``); this
        only forwards the GUI's parameters and the project root. Like the
        hardware wizard it can take many minutes — call it from a
        background thread and hand ``on_progress`` events to the UI.
        """
        from .benchmark_pipeline import (
            DEFAULT_FINALISTS,
            DEFAULT_SCREEN_STEPS,
            DEFAULT_TIME_BUDGET_MINUTES,
            PipelineBudget,
            available_runtime,
            run_benchmark_pipeline,
        )

        if budget_mode == "steps":
            budget = PipelineBudget.for_steps(steps or DEFAULT_SCREEN_STEPS)
        elif budget_mode == "time":
            budget = PipelineBudget.for_time(minutes or DEFAULT_TIME_BUDGET_MINUTES)
        else:
            raise ValueError("budget_mode must be 'steps' or 'time'")
        effective_godot = godot_executable
        if not effective_godot or effective_godot == "godot":
            rt = available_runtime(self.project_root, effective_godot)
            if not rt.get("godot_available"):
                effective_godot = self.ensure_simulated_godot_executable()
        return run_benchmark_pipeline(
            project_path=self.project_root,
            godot_executable=effective_godot,
            budget=budget,
            environment_counts=environment_counts,
            worker_counts=worker_counts,
            finalists=finalists or DEFAULT_FINALISTS,
            cancel=cancel,
            on_progress=on_progress,
            output_dir=output_dir,
            recommendation_project_root=self.project_root,
        )

    def benchmark_pipeline_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Every persisted pipeline report under the project's benchmark root."""
        from .benchmark_pipeline import default_output_root, discover_reports

        return discover_reports(default_output_root(self.project_root), limit)

    def recommended_configuration(self) -> dict[str, Any] | None:
        """The persisted benchmark recommendation, or ``None`` if absent."""
        from .benchmark_pipeline import load_recommendation

        return load_recommendation(self.project_root)

    def apply_recommended_configuration(self) -> dict[str, Any] | None:
        """Mark the persisted recommendation as applied; returns it."""
        from .benchmark_pipeline import mark_recommendation_applied

        return mark_recommendation_applied(self.project_root)

    def validate_runtime_configuration(
        self,
        environment_count: int,
        env_workers: int,
        device: str,
        godot_executable: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility verdict shared with the benchmark's own planning.

        Uses the same runtime facts the pipeline discovers, so a
        configuration the benchmark would reject is also refused by the
        launcher — and vice versa. ``godot_executable`` is the launch
        form's explicit override; when the executable (explicit or the
        default resolution chain) does not resolve, that is reported as an
        error *before* a doomed launch instead of a dead process after it.
        """
        from .benchmark_pipeline import validate_configuration

        runtime = self._runtime_status_cached(executable=godot_executable or None)
        verdict = validate_configuration(
            environment_count,
            env_workers,
            device,
            runtime=runtime,
            cpu_count=int(runtime.get("cpu_count_available_to_process") or 1),
        )
        if runtime.get("godot_available") is False:
            requested = godot_executable or str(runtime.get("godot_executable") or "godot")
            verdict["errors"] = list(verdict.get("errors", [])) + [
                f"Godot executable '{requested}' was not found - configure it on the "
                "Settings page (or install Godot and put it on PATH); a launch now "
                "would fail at startup"
            ]
            verdict["valid"] = False
        return verdict

    def godot_executable_setting(self) -> str | None:
        """The remembered machine-local Godot executable, or ``None``."""
        from .config import load_godot_executable_setting

        return load_godot_executable_setting()

    def configure_godot_executable(self, executable: str) -> dict[str, Any]:
        """Verify ``executable`` and remember it for every later launch.

        The exact probe the benchmark pipeline uses decides: the setting is
        only persisted when the executable actually resolves, so the
        remembered value can never be a path that was not seen working.
        """
        from .benchmark_pipeline import available_runtime
        from .config import save_godot_executable_setting

        text = str(executable or "").strip()
        if not text:
            return {"ok": False, "error": "no executable path given"}
        runtime = available_runtime(self.project_root, text)
        if not runtime.get("godot_available"):
            return {
                "ok": False,
                "error": f"'{text}' did not resolve to a usable Godot executable",
                "runtime": runtime,
            }
        resolved = str(runtime.get("godot_resolved_executable") or text)
        settings_path = save_godot_executable_setting(resolved)
        self._runtime_status.clear()
        return {
            "ok": True,
            "resolved": resolved,
            "version": runtime.get("godot_version"),
            "settings_path": str(settings_path) if settings_path else None,
        }

    def _runtime_status_cached(
        self, max_age_seconds: float = 30.0, executable: str | None = None
    ) -> dict[str, Any]:
        """Runtime facts for compatibility checks, cached briefly.

        ``available_runtime`` probes the Godot binary's version (a
        subprocess), which is far too slow to repeat on every keystroke of
        a custom-configuration form.
        """
        now = time.monotonic()
        cached = self._runtime_status.get(executable)
        if cached is not None and now - cached[0] < max_age_seconds:
            return cached[1]
        from .benchmark_pipeline import available_runtime

        runtime = available_runtime(self.project_root, executable)
        self._runtime_status[executable] = (now, runtime)
        return runtime

    # ------------------------------------------------------------------
    # Roblox TTK Testing live bridge & calibration
    # ------------------------------------------------------------------

    def ttk_testing_status(self, custom_shortcut: str | None = None) -> dict[str, Any]:
        """Return live Roblox Player/TTK Testing status + evidence & calibration state."""
        from .ttk_testing import (
            TTK_CALIBRATION_PRESETS,
            list_roblox_screenshots,
            load_ttk_calibration,
            probe_roblox_live_session,
            status_summary,
        )

        summary = status_summary()
        live = probe_roblox_live_session(custom_shortcut)
        calibration = load_ttk_calibration(self.project_root)
        screenshots = list_roblox_screenshots(self.project_root)
        return {
            **summary,
            "live_session": live,
            "calibration": calibration,
            "recent_screenshots": screenshots,
            "presets": {
                key: {"label": val["label"], "damage": val["damage"], "rpm": val["rpm"]}
                for key, val in TTK_CALIBRATION_PRESETS.items()
            },
        }

    def launch_roblox_ttk_testing(
        self, custom_shortcut: str | None = None, *, direct_place: bool = True
    ) -> dict[str, Any]:
        """Launch Roblox Player (via shortcut or deep-link into TTK Testing)."""
        from .ttk_testing import launch_roblox_ttk_testing

        return launch_roblox_ttk_testing(custom_shortcut, direct_place=direct_place)

    def connect_roblox_ttk_testing(self, custom_shortcut: str | None = None) -> dict[str, Any]:
        """Re-probe and link to any active Roblox Player / TTK Testing session."""
        from .ttk_testing import connect_roblox_live_session

        return connect_roblox_live_session(custom_shortcut)

    def focus_roblox_window(self) -> dict[str, Any]:
        """Restore and focus the active Roblox client window on Windows."""
        from .ttk_testing import focus_roblox_window

        return focus_roblox_window()

    def capture_roblox_screenshot(self) -> dict[str, Any]:
        """Capture a calibration screenshot of the active Roblox window."""
        from .ttk_testing import capture_roblox_screenshot

        return capture_roblox_screenshot(self.project_root)

    def ttk_evidence(self) -> dict[str, Any]:
        """The TTK Testing evidence manifest, without probing a game client.

        ``ttk_testing_status`` answers "is Roblox running on this machine";
        this answers "what is actually evidenced about TTK Testing", which is
        what the Stats page documents next to the policy's own input. It
        reads no process, no window and no log - it is the static manifest.
        """
        from .ttk_testing import status_summary

        return status_summary()

    def ttk_calibration_notes(self) -> dict[str, dict[str, Any]]:
        """Operator-recorded measurements for the evidence table (may be empty)."""
        from .ttk_testing import load_ttk_calibration

        return load_ttk_calibration(self.project_root)

    # ------------------------------------------------------------------
    # Stats page: what the policy actually receives (replays)
    # ------------------------------------------------------------------

    def list_replays(self, limit: int = 200) -> list[dict[str, Any]]:
        """Every recorded replay under the output root, newest first.

        Replays live in ``<run>/replays/*.jsonl`` and start with a single
        header line, so the listing reads that one line per file instead of
        parsing the (potentially huge) rest. A file whose header cannot be
        read is still listed - with its error - rather than silently dropped,
        because "my replay is missing" is worse than "my replay is broken".
        """
        limit = max(1, int(limit))
        entries: list[dict[str, Any]] = []
        root = self.output_root
        if not root.is_dir():
            return entries
        candidates: list[Path] = []
        with contextlib.suppress(OSError):
            # <output root>/runs/<run id>/replays/*.jsonl is the layout the
            # pipeline writes (TrainingConfig.run_directory), so the scan has
            # to walk one level deeper than a single glob would.
            candidates = sorted(
                root.glob("runs/*/replays/*.jsonl"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:limit]
            if not candidates:
                direct = (
                    list((root / "replays").glob("*.jsonl")) if (root / "replays").is_dir() else []
                )
                nested = [
                    p for p in root.glob("*/replays/*.jsonl") if p.parent.parent.name != "runs"
                ]
                candidates = sorted(
                    [*direct, *nested],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )[:limit]
        for path in candidates:
            entry: dict[str, Any] = {
                "path": str(path),
                "name": path.name,
                "run": path.parent.parent.name,
            }
            with contextlib.suppress(OSError):
                entry["size_bytes"] = path.stat().st_size
                entry["modified"] = path.stat().st_mtime
            header, error, ticks = self._replay_meta(path)
            if header is not None:
                entry["header"] = header
                entry["ticks"] = ticks
            if error:
                entry["error"] = error
            entries.append(entry)
        return entries

    def _replay_meta(self, path: Path) -> tuple[dict[str, Any] | None, str, int]:
        """Header and tick count from one pass, cached by (mtime, size).

        The Stats page rescans the replay folder on a timer, so a warm scan
        must not re-read a single byte: header and tick count are stored
        behind the file's stamp. Only the header line is JSON-decoded; ticks
        are recognized by their line prefix, so one corrupt line cannot break
        the listing. A replay also holds events and a result line, which is
        why ticks are counted by prefix rather than by lines: a 3-tick
        episode must not report 6.
        """
        cached = self._replay_meta_cache.get(str(path))
        try:
            stat = path.stat()
        except OSError as exc:  # pragma: no cover - file vanished mid-scan
            return None, str(exc), 0
        stamp = (stat.st_mtime_ns, stat.st_size)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2], cached[3]
        header: dict[str, Any] | None = None
        error = ""
        ticks = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for index, raw in enumerate(handle):
                    if index == 0:
                        payload = json.loads(raw) if raw.strip() else None
                        if isinstance(payload, dict) and "header" in payload:
                            header = payload["header"]
                    elif raw.lstrip().startswith('{"tick"'):
                        ticks += 1
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            error = str(exc)
        if (
            self._replay_meta_cache_limit
            and len(self._replay_meta_cache) >= self._replay_meta_cache_limit
        ):
            self._replay_meta_cache.popitem(last=False)
        self._replay_meta_cache[str(path)] = (stamp, header, error, ticks)
        return header, error, ticks

    def replay_stats(self, path: str | Path, tick: int | None = None) -> dict[str, Any]:
        """One replay decoded for the Stats page: header, tick, observation.

        The whole episode is parsed once per (path, size, mtime) and cached,
        because stepping through ticks must not re-read a file. Returns the
        requested tick's observation and action when the recording is
        ``detailed``; a light replay reports that the vector was not
        recorded instead of inventing one.

        A replay recorded under an *older* observation contract is still
        read (``strict_contract=False``, the same allowance the CLI's
        ``--allow-contract-mismatch`` makes): a recording is evidence and
        evidence should not expire when the vector grows. The result says so
        explicitly - ``contract_match`` is false and
        ``recorded_observation_dim`` carries the width it was recorded with -
        so the page can label the values as not comparable instead of
        silently painting them into the current contract's table.
        """
        from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
        from .replay import load_replay

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"replay not found: {source}")
        stat = source.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._replay_cache.get(str(source))
        if cached is None or cached[0] != stamp:
            episode = load_replay(source, strict_contract=False)
            header = episode.header
            info: dict[str, Any] = {
                "path": str(source),
                "name": source.name,
                "run": source.parent.parent.name,
                "header": header.to_dict(),
                "tick_count": len(episode.ticks),
                "detailed": episode.detailed,
                "contract_match": (
                    int(header.observation_dim) == OBSERVATION_FIELD_COUNT
                    and tuple(int(value) for value in header.action_nvec) == tuple(ACTION_NVEC)
                ),
                "recorded_observation_dim": int(header.observation_dim),
                "current_observation_dim": OBSERVATION_FIELD_COUNT,
            }
            self._replay_cache[str(source)] = (stamp, info, episode)
            if self._replay_cache_limit and len(self._replay_cache) > self._replay_cache_limit:
                self._replay_cache.popitem(last=False)
        else:
            info, episode = cached[1], cached[2]
        index = 0 if tick is None else max(0, min(int(tick), len(episode.ticks) - 1))
        result = dict(info)
        result["tick_index"] = index
        if episode.ticks:
            current = episode.ticks[index]
            result["action"] = list(current.action)
            result["reward"] = float(current.reward)
            result["done"] = bool(current.done)
            result["observation"] = (
                [float(value) for value in current.observation]
                if current.observation is not None
                else None
            )
            result["events"] = [
                {
                    "kind": event.kind,
                    "tick": event.tick,
                    "data": dict(event.data or {}),
                }
                for event in episode.events
                if event.tick == current.tick
            ]
        else:
            result["action"] = []
            result["reward"] = None
            result["done"] = False
            result["observation"] = None
            result["events"] = []
        return result

    def save_ttk_calibration(
        self,
        mechanic: str,
        measured_value: str,
        *,
        notes: str = "",
        evidence_path: str = "",
    ) -> dict[str, Any]:
        """Save an operator measurement for one TTK Testing mechanic."""
        from .ttk_testing import save_ttk_calibration_entry

        return save_ttk_calibration_entry(
            self.project_root,
            mechanic,
            measured_value,
            notes=notes,
            evidence_path=evidence_path,
        )

    def apply_ttk_preset(self, preset_id: str) -> dict[str, Any]:
        """Apply a curated TTK Testing weapon/mechanics calibration preset."""
        from .ttk_testing import apply_ttk_calibration_preset

        return apply_ttk_calibration_preset(self.project_root, preset_id)

    def analyze_roblox_screenshot(self, image_path: str | Path | None = None) -> dict[str, Any]:
        """Inspect the latest or given Roblox TTK Testing screenshot for HUD/resolution metadata."""
        from .ttk_testing import analyze_roblox_ttk_screenshot

        return analyze_roblox_ttk_screenshot(self.project_root, image_path)

    def export_ttk_combat_profile(
        self,
        destination: str | Path | None = None,
        *,
        preset_id: str | None = None,
    ) -> dict[str, Any]:
        """Export the calibrated Roblox TTK Testing combat profile to JSON."""
        from .ttk_testing import TTK_CALIBRATION_PRESETS

        status = self.ttk_testing_status()
        target = (
            Path(destination).expanduser()
            if destination is not None
            else (self.project_root / ".sandboxai" / "ttk_combat_profile.json")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "sandboxai.ttk_combat_profile/v1",
            "preset_id": preset_id or "custom",
            "preset": TTK_CALIBRATION_PRESETS.get(preset_id or "", {}),
            "place_id": status.get("place_id"),
            "place_url": status.get("place_url"),
            "completion_pct": status.get("completion_pct"),
            "entries": status.get("entries", {}),
            "live_session": status.get("live_session", {}),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "ok": True,
            "path": str(target),
            "completion_pct": status.get("completion_pct", 0.0),
            "place_id": status.get("place_id"),
        }

    def calculate_ttk_preview(
        self,
        *,
        damage: float,
        rpm: float,
        target_hp: float = 100.0,
        magazine_size: int = 30,
        reload_seconds: float = 2.2,
        head_multiplier: float = 1.5,
    ) -> dict[str, Any]:
        """Calculate exact Shots-to-Kill, TTK (ms), Burst DPS and Sustained DPS."""
        from .ttk_testing import calculate_ttk_metrics

        return calculate_ttk_metrics(
            damage=damage,
            rpm=rpm,
            target_hp=target_hp,
            magazine_size=magazine_size,
            reload_seconds=reload_seconds,
            head_multiplier=head_multiplier,
        )

    # ------------------------------------------------------------------
    # Ubuntu CPU Performance Turbo
    # ------------------------------------------------------------------

    def ubuntu_cpu_status(
        self,
        *,
        environment_count: int | None = None,
        env_workers: int | None = None,
    ) -> dict[str, Any]:
        """Inspect Ubuntu/Linux CPU topology and anti-thrashing turbo state."""
        from .sharded_env import ubuntu_cpu_runtime_profile

        return ubuntu_cpu_runtime_profile(
            environment_count=environment_count,
            env_workers=env_workers,
        )

    def enable_ubuntu_cpu_turbo(
        self,
        *,
        worker_count: int | None = None,
        environment_count: int | None = None,
    ) -> dict[str, Any]:
        """Activate Ubuntu CPU thread-pinning and single-thread BLAS turbo mode."""
        from .sharded_env import apply_ubuntu_cpu_optimizations

        return apply_ubuntu_cpu_optimizations(
            worker_count=worker_count,
            environment_count=environment_count,
        )

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

    def _managed_training(
        self, config: TrainingConfig, checkpoint: str | Path | None = None
    ) -> tuple[Path, list[str]]:
        config.validate()
        if checkpoint is not None:
            # Resume artifacts stay inside the original run directory (the
            # same rule ppo._resolve_run_directory applies), so the GUI keeps
            # polling one status.json across the restart.
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
            run_dir = (
                checkpoint_path.parent.parent
                if checkpoint_path.parent.name == "checkpoints"
                else checkpoint_path.parent
            )
        else:
            run_dir = config.run_directory()
            if not run_dir.is_absolute():
                run_dir = self.project_root / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        # Persist the exact domain config (the trainer also writes its own copy).
        config.save(run_dir / "config.json")
        command = self._python_command(
            "train" if checkpoint is None else "resume",
            "--config",
            str(run_dir / "config.json"),
            "--control-file",
            str(run_dir / "command.json"),
            "--status-file",
            str(run_dir / "status.json"),
            "--event-log-file",
            str(run_dir / "events.jsonl"),
        )
        if checkpoint is not None:
            command.extend(["--checkpoint", str(Path(checkpoint).expanduser())])
        return run_dir, command

    def start_training(
        self,
        config: TrainingConfig | dict[str, Any],
        *,
        checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        """Launch a fresh training run, or resume ``checkpoint`` when given.

        Resuming keeps the checkpoint's original run directory, so a
        restarted agent continues exactly where it stopped (same
        status.json, same event log, same checkpoint inventory).
        """
        cfg = config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config)
        if not cfg.godot_executable or cfg.godot_executable == "godot":
            resolved_godot = find_godot_executable(cfg.godot_executable or "godot")
            if not shutil.which(resolved_godot) and not Path(resolved_godot).expanduser().is_file():
                cfg.godot_executable = self.ensure_simulated_godot_executable()
            else:
                cfg.godot_executable = resolved_godot
        if cfg.resolved_env_workers() > 1 or os.environ.get("SANDBOXAI_CPU_TURBO") == "1":
            with contextlib.suppress(Exception):
                from .sharded_env import apply_ubuntu_cpu_optimizations

                apply_ubuntu_cpu_optimizations(
                    worker_count=cfg.resolved_env_workers(),
                    environment_count=cfg.environment_count,
                )
        run_dir, command = self._managed_training(cfg, checkpoint)
        meta = {
            "run_id": cfg.run_id or run_dir.name,
            "environment_count": cfg.environment_count,
            "env_workers": cfg.resolved_env_workers(),
            "total_training_steps": cfg.total_training_steps,
            "rollout_length": cfg.resolved_rollout_length(),
            "device": cfg.device,
            "curriculum_mode": cfg.curriculum_mode,
        }
        if checkpoint is not None:
            meta["resumed_from"] = str(checkpoint)
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
