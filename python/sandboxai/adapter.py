"""Stable application boundary for the SandboxAI Control Center.

This module is intentionally GUI-free.  It translates validated domain
configuration into the existing CLI commands and reads the existing run
artifacts; it does not reimplement PPO, evaluation, benchmarks, or the Godot
bridge.  Long operations are owned by :class:`ProcessManager` and are never
run on a caller's UI thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterable

from .benchmark import summarize_scaling
from .config import EvaluationConfig, TrainingConfig
from .run_inspection import discover_run_directories, inspect_run, inspect_runs, tail_jsonl
from .telemetry import resource_snapshot


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
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        if self.returncode is not None:
            return "finished" if self.returncode == 0 else "failed"
        return "running"


class ProcessManager:
    """Non-blocking subprocess lifecycle with bounded, thread-safe output."""
    def __init__(self, max_lines: int = 2000) -> None:
        self.max_lines = max(100, int(max_lines))
        self._records: dict[str, ProcessRecord] = {}
        self._lock = threading.RLock()

    def start(self, kind: str, command: list[str], run_dir: Path, cwd: Path) -> ProcessRecord:
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                bufsize=1, start_new_session=(os.name != "nt"),
                env={**os.environ, "PYTHONPATH": str(cwd / "python") + os.pathsep + os.environ.get("PYTHONPATH", "")},
            )
        except OSError as exc:
            raise RuntimeError(f"could not start {kind}: {exc}") from exc
        record = ProcessRecord(uuid.uuid4().hex, kind, list(command), run_dir, process)
        with self._lock:
            self._records[record.id] = record
        for stream, target in ((process.stdout, record.stdout), (process.stderr, record.stderr)):
            threading.Thread(target=self._drain, args=(record, stream, target), daemon=True).start()
        threading.Thread(target=self._wait, args=(record,), daemon=True).start()
        return record

    def _drain(self, record: ProcessRecord, stream: Any, target: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                with self._lock:
                    target.append(line.rstrip("\n"))
                    del target[:-self.max_lines]
        finally:
            stream.close()

    def _wait(self, record: ProcessRecord) -> None:
        code = record.process.wait()
        with self._lock:
            record.returncode = code
            if code and not record.error:
                record.error = "process exited with code %d" % code

    def get(self, process_id: str) -> ProcessRecord | None:
        with self._lock:
            return self._records.get(process_id)

    def snapshot(self, process_id: str) -> dict[str, Any]:
        record = self.get(process_id)
        if record is None:
            return {"state": "unknown", "error": "process not found"}
        with self._lock:
            result = {"id": record.id, "kind": record.kind, "state": record.state,
                      "returncode": record.returncode, "started_at": record.started_at,
                      "stdout": list(record.stdout[-100:]), "stderr": list(record.stderr[-100:]),
                      "error": record.error, "run_dir": str(record.run_dir)}
        # The trainer's atomically published status is authoritative for
        # training state and progress; the OS process state is supplementary.
        status_path = record.run_dir / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict): result["backend"] = status
        except (OSError, json.JSONDecodeError):
            pass
        return result

    def terminate(self, process_id: str, timeout: float = 5.0) -> dict[str, Any]:
        record = self.get(process_id)
        if record is None:
            return {"state": "unknown", "error": "process not found"}
        if record.returncode is None:
            # Training owns a cooperative command file. Other commands have
            # no safe callback boundary and receive a graceful terminate.
            command_file = record.run_dir / "command.json"
            if record.kind == "training":
                command_file.write_text(json.dumps({"command": "stop", "sequence": time.time_ns()}) + "\n", encoding="utf-8")
            else:
                record.process.terminate()
            try:
                record.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record.process.kill()
                record.process.wait(timeout=timeout)
        return self.snapshot(process_id)

    def close(self) -> None:
        for record in list(self._records.values()):
            if record.returncode is None:
                self.terminate(record.id)


class SandboxAIAdapter:
    """Application-facing operations over the real SandboxAI infrastructure."""
    def __init__(self, project_root: str | Path | None = None, output_root: str | Path = "training") -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.output_root = (self.project_root / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root)
        self.processes = ProcessManager()

    def _python_command(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "sandboxai", *args]

    def system_status(self) -> dict[str, Any]:
        status = resource_snapshot()
        status.update({"python": sys.executable, "python_version": sys.version,
                       "project_root": str(self.project_root),
                       "godot_executable": os.environ.get("GODOT_EXECUTABLE", "godot")})
        return status

    def project_status(self) -> dict[str, Any]:
        index = inspect_runs(self.output_root, limit=1, event_limit=10)
        return {"output_root": str(self.output_root), "runs": index,
                "active_processes": [self.processes.snapshot(r.id) for r in self.processes._records.values() if r.returncode is None]}

    def list_runs(self, limit: int = 100) -> dict[str, Any]:
        return inspect_runs(self.output_root, limit=limit, event_limit=0)

    def inspect_run(self, run: str | Path, event_limit: int = 50) -> dict[str, Any]:
        return inspect_run(run, event_limit=event_limit)

    def list_checkpoints(self, run: str | Path) -> dict[str, Any]:
        report = self.inspect_run(run)
        return report.get("checkpoints", {}) if isinstance(report, dict) else {}

    def run_metrics(self, run: str | Path) -> dict[str, Any]:
        report = self.inspect_run(run, event_limit=100)
        return {"progress": report.get("progress", {}), "evaluation": report.get("evaluation", {}),
                "telemetry": report.get("telemetry", {}), "events": report.get("events", [])}

    def telemetry(self, run: str | Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
        path = Path(run) / "logs" / "training.jsonl" if run else None
        if path is None:
            dirs = discover_run_directories(self.output_root)
            path = dirs[-1] / "logs" / "training.jsonl" if dirs else Path()
        return tail_jsonl(path, limit) if str(path) else []

    def profiling(self, run: str | Path) -> dict[str, Any]:
        path = Path(run) / "logs" / "training_profile.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {"error": "profile is not an object"}
        except FileNotFoundError:
            return {"available": False, "reason": "profiling not enabled or run incomplete"}
        except (OSError, json.JSONDecodeError) as exc:
            return {"available": False, "error": str(exc)}

    def benchmark_results(self, directory: str | Path = "training/benchmarks/latest") -> dict[str, Any]:
        path = Path(directory)
        if not path.is_absolute(): path = self.project_root / path
        try:
            rows = json.loads((path / "benchmark.json").read_text(encoding="utf-8"))
            return {"results": rows, "scaling": summarize_scaling(rows) if isinstance(rows, list) else {}, "directory": str(path)}
        except (OSError, json.JSONDecodeError) as exc:
            return {"results": [], "error": str(exc), "directory": str(path)}

    def _managed_training(self, config: TrainingConfig) -> tuple[Path, list[str]]:
        config.validate()
        run_dir = config.run_directory()
        if not run_dir.is_absolute():
            run_dir = self.project_root / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        # Persist the exact domain config (the trainer also writes its own copy).
        config.save(run_dir / "config.json")
        command = self._python_command("train", "--config", str(run_dir / "config.json"),
                                      "--control-file", str(run_dir / "command.json"),
                                      "--status-file", str(run_dir / "status.json"),
                                      "--event-log-file", str(run_dir / "events.jsonl"))
        return run_dir, command

    def start_training(self, config: TrainingConfig | dict[str, Any]) -> dict[str, Any]:
        cfg = config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config)
        run_dir, command = self._managed_training(cfg)
        record = self.processes.start("training", command, run_dir, self.project_root)
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def start_benchmark(self, *, environment_counts: Iterable[int], worker_counts: Iterable[int] = (1,), steps: int = 2000,
                        enemy_count: int = 1, compact_infos: bool = True, output_dir: str | Path = "training/benchmarks/latest") -> dict[str, Any]:
        directory = Path(output_dir); directory = directory if directory.is_absolute() else self.project_root / directory
        command = self._python_command("benchmark", "--env-counts", ",".join(map(str, environment_counts)),
            "--worker-counts", ",".join(map(str, worker_counts)), "--steps", str(steps), "--enemy-count", str(enemy_count),
            "--output-dir", str(directory), *( [] if compact_infos else ["--full-infos"]))
        record = self.processes.start("benchmark", command, directory, self.project_root)
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def start_evaluation(self, checkpoint: str | Path, *, episodes: int = 20, environment_count: int = 1,
                         device: str = "auto", output_dir: str | Path = "training/evaluations/control_center") -> dict[str, Any]:
        directory = Path(output_dir); directory = directory if directory.is_absolute() else self.project_root / directory
        command = self._python_command("evaluate", "--checkpoint", str(checkpoint), "--episodes", str(episodes),
            "--env-count", str(environment_count), "--device", device, "--output-dir", str(directory))
        record = self.processes.start("evaluation", command, directory, self.project_root)
        return self.processes.snapshot(record.id) | {"process_id": record.id}

    def process_status(self, process_id: str) -> dict[str, Any]: return self.processes.snapshot(process_id)
    def cancel(self, process_id: str) -> dict[str, Any]: return self.processes.terminate(process_id)
    def close(self) -> None: self.processes.close()
