"""Agent lifecycle management for the headless Control Center.

An *agent* is one unit of launched work this Control Center operates: a
PPO training run, a benchmark sweep or an evaluation. The layer below it
(:class:`sandboxai.adapter.ProcessManager`) already owns process plumbing
— spawn, bounded output buffers, cooperative stop. What it does not own
is *lifecycle*: it reports ``running``/``finished``/``failed`` OS states,
while an operator thinks in the states of the work:

    AVAILABLE -> LAUNCHING -> RUNNING -> PAUSED -> STOPPING -> STOPPED
                                    |-> RESTARTING -> LAUNCHING
                                    |-> FINISHED
                                    |-> FAILED

This module derives those states honestly instead of inventing them:

* ``LAUNCHING`` — the process exists but the backend has not published a
  state yet (training publishes ``Starting``/``Running`` through its
  ``status.json``; benchmark/evaluation processes have no cooperative
  protocol, so a live process means the work is running).
* ``PAUSED``/``STOPPING`` — the backend's own published state, plus the
  operator's requested action while the trainer is on its way to the next
  safe boundary.
* ``STOPPED`` vs ``FINISHED`` — the process exited after a stop request
  (cooperative or forced) versus on its own.
* ``FAILED`` — non-zero exit, a backend ``Error`` state, or a launch that
  never got off the ground (the error is kept on the record).

Pause/Resume exists only where the backend supports it: the training
command-file protocol (``sandboxai.run_control``) honours ``pause``/
``resume``/``stop`` at safe callback boundaries. Benchmark and evaluation
processes have no such protocol, so those actions are refused with a
reason instead of faked.

Restart = stop (cooperative, bounded wait, then escalate) + relaunch. A
training agent restarts from the run's latest checkpoint when one exists
(pure wall-clock continuation of the same run directory) and from scratch
otherwise.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    # Only for annotations (deferred by `from __future__ import annotations`);
    # importing at runtime would create an import cycle, because the adapter
    # constructs this manager.
    from .adapter import ProcessManager, ProcessRecord
    from .control_center_schema import ProcessSnapshot

#: Lifecycle states an agent view can report.
LIFECYCLE_AVAILABLE = "AVAILABLE"
LIFECYCLE_LAUNCHING = "LAUNCHING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_PAUSED = "PAUSED"
LIFECYCLE_STOPPING = "STOPPING"
LIFECYCLE_STOPPED = "STOPPED"
LIFECYCLE_FINISHED = "FINISHED"
LIFECYCLE_FAILED = "FAILED"
LIFECYCLE_RESTARTING = "RESTARTING"

#: Kinds whose backend speaks the cooperative command protocol.
PAUSABLE_KINDS = frozenset({"training"})

#: How long a cooperative restart waits for the trainer to save its final
#: checkpoint before escalating to a force stop.
RESTART_STOP_TIMEOUT_SECONDS = 90.0


@dataclass
class AgentRecord:
    """One launched (or launch-attempted) agent and its requested action."""

    agent_id: str
    kind: str
    spec: dict[str, Any]
    name: str = ""
    process_id: str | None = None
    run_dir: str | None = None
    created_at: float = field(default_factory=time.time)
    #: The operator's most recent requested action ("", "pause", "resume",
    #: "stop", "restart"). Cleared when the action completes.
    desired: str = ""
    stop_requested_at: float | None = None
    error: str | None = None


class AgentLauncher(Protocol):
    """The launch surface :class:`SandboxAIAdapter` already provides.

    The signatures mirror the adapter's methods exactly (keyword-only
    parameters, not ``**spec``), so the adapter satisfies the protocol
    structurally without a runtime cast.
    """

    def start_training(
        self, config: Any, *, checkpoint: str | Path | None = None
    ) -> dict[str, Any]: ...

    def start_benchmark(
        self,
        *,
        environment_counts: Iterable[int],
        worker_counts: Iterable[int] = ...,
        steps: int = ...,
        enemy_count: int = ...,
        compact_infos: bool = ...,
        output_dir: str | Path = ...,
    ) -> dict[str, Any]: ...

    def start_evaluation(
        self,
        checkpoint: str | Path,
        *,
        episodes: int = ...,
        environment_count: int = ...,
        device: str = ...,
        output_dir: str | Path = ...,
    ) -> dict[str, Any]: ...


def derive_lifecycle(record: AgentRecord, snapshot: Mapping[str, Any] | None) -> str:
    """Map (record, process snapshot) onto the operator-facing lifecycle.

    Pure function: it only reads published state (OS process state, the
    backend's ``status.json``, the record's requested action). It never
    guesses — an unknown process state is reported as ``FAILED`` when the
    launch itself failed, otherwise the process state decides.
    """
    if record.process_id is None:
        return LIFECYCLE_FAILED if record.error else LIFECYCLE_AVAILABLE
    if snapshot is None:
        return LIFECYCLE_AVAILABLE if record.error is None else LIFECYCLE_FAILED
    if record.desired == "restart":
        # Spanning both the stopping and the relaunching phase.
        return LIFECYCLE_RESTARTING
    state = str(snapshot.get("state", "unknown"))
    backend = snapshot.get("backend") or {}
    backend_state = str(backend.get("state", "") or "")
    if state == "running":
        if backend_state == "Paused":
            return LIFECYCLE_PAUSED
        if record.desired == "stop" or backend_state == "Stopping":
            return LIFECYCLE_STOPPING
        if backend_state in ("", "Idle", "Starting"):
            # No published backend state yet. Training publishes as soon as
            # the trainer process starts; a benchmark/evaluation process
            # simply is its work, so "alive" already means running.
            return LIFECYCLE_LAUNCHING if record.kind in PAUSABLE_KINDS else LIFECYCLE_RUNNING
        return LIFECYCLE_RUNNING
    if state == "failed":
        return LIFECYCLE_FAILED
    if state in ("finished", "unknown"):
        if record.stop_requested_at is not None or backend_state == "Stopping":
            return LIFECYCLE_STOPPED
        if backend_state == "Error":
            return LIFECYCLE_FAILED
        return LIFECYCLE_FINISHED
    return LIFECYCLE_FAILED


class AgentManager:
    """Registry of agents with lifecycle actions, on top of ProcessManager."""

    def __init__(
        self,
        processes: ProcessManager,
        launcher: AgentLauncher,
        *,
        restart_stop_timeout: float = RESTART_STOP_TIMEOUT_SECONDS,
    ) -> None:
        self._processes = processes
        self._launcher = launcher
        self._records: dict[str, AgentRecord] = {}
        self._lock = threading.RLock()
        self._restart_stop_timeout = restart_stop_timeout

    # ------------------------------------------------------------------
    # Registry reads
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> AgentRecord | None:
        with self._lock:
            return self._records.get(agent_id)

    def views(self) -> list[dict[str, Any]]:
        """One display/API view per agent, newest first."""
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        views = []
        for record in records:
            snapshot = self._processes.snapshot(record.process_id) if record.process_id else None
            views.append(self._view(record, snapshot))
        return views

    def _view(self, record: AgentRecord, snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
        lifecycle = derive_lifecycle(record, snapshot)
        backend = (snapshot or {}).get("backend") or {}
        # Error precedence: the launch failure first, then the backend's own
        # published root cause (status.json "error"), then the process
        # layer's account. The backend error outranks the process error
        # because "process exited with code 1" describes the symptom of the
        # very failure the backend already named.
        error = (
            record.error or str(backend.get("error") or "") or (snapshot or {}).get("error") or ""
        )
        spec = dict(record.spec)
        view = {
            "agent_id": record.agent_id,
            "kind": record.kind,
            "name": record.name,
            "lifecycle": lifecycle,
            "process_id": record.process_id,
            "run_dir": record.run_dir,
            "desired": record.desired,
            "created_at": record.created_at,
            "error": error,
            "spec": spec,
            # Straight pass-through of measured backend facts; nothing here
            # is derived or estimated.
            "pid": (snapshot or {}).get("pid"),
            "started_at": (snapshot or {}).get("started_at"),
            "returncode": (snapshot or {}).get("returncode"),
            "backend": backend,
            "meta": (snapshot or {}).get("meta") or {},
        }
        return view

    def summary(self) -> dict[str, int]:
        """Lifecycle counts for dashboard chips (measured states only)."""
        counts: dict[str, int] = {}
        for view in self.views():
            counts[view["lifecycle"]] = counts.get(view["lifecycle"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------

    def launch_training(
        self, config: Any, *, checkpoint: str | Path | None = None, name: str = ""
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "kind": "training",
            "checkpoint": str(checkpoint) if checkpoint else None,
        }
        if hasattr(config, "to_dict"):
            spec["config"] = dict(config.to_dict())
        else:
            spec["config"] = dict(config)
        record = self._new_record("training", spec, name)
        self._launch(record, lambda: self._launcher.start_training(config, checkpoint=checkpoint))
        return self._view(record, self._snapshot(record))

    def launch_benchmark(self, **spec: Any) -> dict[str, Any]:
        record = self._new_record("benchmark", {"kind": "benchmark", "spec": dict(spec)})
        self._launch(record, lambda: self._launcher.start_benchmark(**spec))
        return self._view(record, self._snapshot(record))

    def launch_evaluation(self, checkpoint: str | Path, **spec: Any) -> dict[str, Any]:
        full = {"kind": "evaluation", "checkpoint": str(checkpoint), "spec": dict(spec)}
        record = self._new_record("evaluation", full)
        self._launch(record, lambda: self._launcher.start_evaluation(checkpoint, **spec))
        return self._view(record, self._snapshot(record))

    def _new_record(self, kind: str, spec: dict[str, Any], name: str = "") -> AgentRecord:
        record = AgentRecord(agent_id=uuid.uuid4().hex[:12], kind=kind, spec=spec, name=name)
        with self._lock:
            self._records[record.agent_id] = record
        return record

    def _launch(self, record: AgentRecord, launch: Any) -> None:
        record.desired = "launch"
        try:
            result = launch()
        except Exception as exc:  # invalid config, missing engine, spawn error
            record.desired = ""
            record.error = str(exc)
            return
        record.process_id = result.get("process_id")
        record.run_dir = result.get("run_dir")
        if not record.name:
            record.name = Path(str(record.run_dir or "")).name or record.kind
        record.desired = ""

    # ------------------------------------------------------------------
    # Lifecycle actions
    # ------------------------------------------------------------------

    def _action_result(
        self, record: AgentRecord, ok: bool, error: str | None = None
    ) -> dict[str, Any]:
        return {
            "ok": ok,
            "error": error,
            "agent_id": record.agent_id,
            "lifecycle": derive_lifecycle(record, self._snapshot(record)),
        }

    def _snapshot(self, record: AgentRecord) -> ProcessSnapshot | None:
        if record.process_id is None:
            return None
        return self._processes.snapshot(record.process_id)

    def _live_process(self, record: AgentRecord) -> ProcessRecord | None:
        if record.process_id is None:
            return None
        process = self._processes.get(record.process_id)
        if process is None or process.returncode is not None:
            return None
        return process

    def pause(self, agent_id: str) -> dict[str, Any]:
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        if record.kind not in PAUSABLE_KINDS:
            return self._action_result(
                record,
                False,
                f"pause is not supported by '{record.kind}' processes; only the training "
                "backend implements the cooperative pause protocol",
            )
        if record.desired in ("stop", "restart"):
            return self._action_result(record, False, "a stop is already in progress")
        if record.process_id is None or self._live_process(record) is None:
            return self._action_result(record, False, "the agent process is not running")
        self._processes.send_command(record.process_id, "pause")
        record.desired = "pause"
        return self._action_result(record, True)

    def resume(self, agent_id: str) -> dict[str, Any]:
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        if record.kind not in PAUSABLE_KINDS:
            return self._action_result(
                record,
                False,
                f"resume is not supported by '{record.kind}' processes; only the training "
                "backend implements the cooperative pause protocol",
            )
        if record.process_id is None or self._live_process(record) is None:
            return self._action_result(record, False, "the agent process is not running")
        self._processes.send_command(record.process_id, "resume")
        record.desired = ""
        return self._action_result(record, True)

    def stop(self, agent_id: str) -> dict[str, Any]:
        """Request a safe stop: cooperative for training, terminate otherwise."""
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        process_id = record.process_id
        if process_id is None or self._live_process(record) is None:
            return self._action_result(record, False, "the agent process is not running")
        if record.kind in PAUSABLE_KINDS:
            self._processes.send_command(process_id, "stop")
        else:
            self._processes.cancel(process_id)
        record.desired = "stop"
        record.stop_requested_at = time.time()
        return self._action_result(record, True)

    def stop_all(self) -> list[dict[str, Any]]:
        """Stop every agent with a live process; returns one result per agent."""
        with self._lock:
            records = list(self._records.values())
        results = []
        for record in records:
            if self._live_process(record) is None:
                continue
            results.append(self.stop(record.agent_id))
        return results

    def force_stop(self, agent_id: str) -> dict[str, Any]:
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        process_id = record.process_id
        if process_id is None or self._live_process(record) is None:
            return self._action_result(record, False, "the agent process is not running")
        self._processes.force_stop(process_id)
        record.desired = ""
        record.stop_requested_at = record.stop_requested_at or time.time()
        return self._action_result(record, True)

    def restart(self, agent_id: str) -> dict[str, Any]:
        """Stop the agent, then launch it again from the same specification.

        Blocks until the old process has exited (cooperatively where
        possible, escalated after ``restart_stop_timeout``) and the new
        process has been spawned — run it on a background thread, which is
        where the Control Center calls it from. A training agent with an
        existing checkpoint resumes from it (same run directory, pure
        wall-clock continuation); without one it starts from scratch.
        """
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        record.desired = "restart"
        record.stop_requested_at = None
        if record.process_id is not None and self._live_process(record) is not None:
            if record.kind in PAUSABLE_KINDS:
                self._processes.send_command(record.process_id, "stop")
            else:
                self._processes.cancel(record.process_id)
            self._wait_for_exit(record, self._restart_stop_timeout)
        try:
            self._relaunch(record)
        except Exception as exc:
            record.desired = ""
            record.error = str(exc)
            return self._action_result(record, False, str(exc))
        record.desired = ""
        return self._action_result(record, True)

    def _wait_for_exit(self, record: AgentRecord, timeout: float) -> None:
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self._live_process(record) is None:
                return
            time.sleep(0.1)
        # A trainer that will not finish its checkpoint in time is wedged;
        # escalation loses the final checkpoint but must not lose the
        # operator's restart.
        if record.process_id is not None:
            self._processes.force_stop(record.process_id)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._live_process(record) is None:
                return
            time.sleep(0.1)

    def _relaunch(self, record: AgentRecord) -> None:
        record.error = None
        if record.kind == "training":
            config = record.spec.get("config") or {}
            from .config import TrainingConfig

            training_config = (
                config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config)
            )
            checkpoint = self._latest_checkpoint(record)
            result = self._launcher.start_training(training_config, checkpoint=checkpoint)
        elif record.kind == "benchmark":
            result = self._launcher.start_benchmark(**record.spec.get("spec", {}))
        else:
            checkpoint = record.spec.get("checkpoint")
            if not checkpoint:
                raise ValueError("the evaluation agent record has no checkpoint to re-run")
            result = self._launcher.start_evaluation(checkpoint, **record.spec.get("spec", {}))
        record.process_id = result.get("process_id")
        record.run_dir = result.get("run_dir")
        if not record.name:
            record.name = Path(str(record.run_dir or "")).name or record.kind

    @staticmethod
    def _latest_checkpoint(record: AgentRecord) -> str | None:
        """The newest usable checkpoint of a run, wherever it was written.

        ``latest.zip`` only appears when a run finishes naturally; a run
        stopped mid-training leaves the periodic ``ppo_<steps>_steps.zip``
        checkpoints instead, and those are exactly what a restart should
        resume from.
        """
        if not record.run_dir:
            return None
        run_dir = Path(record.run_dir)
        latest = run_dir / "checkpoints" / "latest.zip"
        if latest.is_file():
            return str(latest)
        periodic = []
        for candidate in (run_dir / "checkpoints").glob("ppo_*_steps.zip"):
            match = re.search(r"ppo_(\d+)_steps\.zip$", candidate.name)
            if match:
                periodic.append((int(match.group(1)), candidate))
        if periodic:
            periodic.sort()
            return str(periodic[-1][1])
        best = run_dir / "checkpoints" / "best_eval.zip"
        if best.is_file():
            return str(best)
        final = run_dir / "final.zip"
        return str(final) if final.is_file() else None

    def remove(self, agent_id: str) -> dict[str, Any]:
        """Forget a terminal agent. Refused while its process is still live."""
        record = self.get(agent_id)
        if record is None:
            return {"ok": False, "error": "agent not found", "agent_id": agent_id}
        if self._live_process(record) is not None:
            return self._action_result(
                record, False, "stop the agent before removing it from the list"
            )
        with self._lock:
            del self._records[agent_id]
        return {"ok": True, "error": None, "agent_id": agent_id}

    def clear_finished(self) -> int:
        """Forget every agent whose process has exited. Returns the count."""
        removed = 0
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.process_id is not None and self._live_process(record) is None:
                with self._lock:
                    self._records.pop(record.agent_id, None)
                removed += 1
        return removed
