"""Cooperative process control for training launched by the Control Center.

The Godot Control Center and the Python trainer are deliberately separate
processes.  This module provides their small, dependency-free boundary:
Godot writes one JSON command file and Python atomically publishes one JSON
status file plus a bounded-rate JSONL event stream.  The trainer remains the
owner of training; the GUI only requests pause/resume/stop at safe callback
boundaries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from .control_center_schema import EVENT_SCHEMA_VERSION, STATUS_SCHEMA_VERSION


TERMINAL_STATES = frozenset({"Finished", "Error"})


class RunControl:
    """Polls operator commands and publishes an authoritative run status.

    All paths are optional so normal CLI training has zero extra file I/O.
    ``checkpoint`` is called from existing PPO/BC loops.  A pause waits in
    the Python process (the Godot simulation is already blocked on its pipe),
    and a stop returns ``False`` so the backend can save its normal final
    checkpoint and close cleanly.
    """

    def __init__(
        self,
        command_path: str | Path | None = None,
        status_path: str | Path | None = None,
        event_path: str | Path | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        self.command_path = Path(command_path).expanduser().resolve() if command_path else None
        self.status_path = Path(status_path).expanduser().resolve() if status_path else None
        self.event_path = Path(event_path).expanduser().resolve() if event_path else None
        self.poll_interval = max(0.02, float(poll_interval))
        self.state = "Idle"
        self.stop_requested = False
        self._last_sequence: int | str | None = None
        self._status: dict[str, Any] = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "state": self.state,
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
        for path in (self.command_path, self.status_path, self.event_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.command_path is not None or self.status_path is not None

    def start(self, **values: Any) -> None:
        self.stop_requested = False
        self.update(state="Starting", **values)
        self.event("system", "training process started", values)

    def running(self, **values: Any) -> None:
        self.update(state="Running", **values)

    def update(self, state: str | None = None, **values: Any) -> None:
        if state is not None:
            self.state = str(state)
        self._status.update(values)
        self._status.update(
            {
                "state": self.state,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "stop_requested": self.stop_requested,
            }
        )
        if self.status_path is not None:
            _atomic_json_write(self.status_path, self._status)

    def event(self, category: str, message: str, values: dict[str, Any] | None = None) -> None:
        if self.event_path is None:
            return
        row = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "wall_time": time.time(),
            "category": str(category),
            "message": str(message),
            "values": dict(values or {}),
        }
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()

    def checkpoint(self) -> bool:
        """Honours the latest command; ``False`` requests a graceful stop."""
        command = self._read_command()
        name = str(command.get("command", "")).strip().lower()
        if name == "stop":
            self.stop_requested = True
            if self.state != "Stopping":
                self.update(state="Stopping")
                self.event("system", "stop requested; saving at the next safe boundary")
            return False
        if name != "pause":
            if self.state == "Paused":
                self.update(state="Running")
                self.event("system", "training resumed")
            return True

        if self.state != "Paused":
            self.update(state="Paused")
            self.event("system", "training paused")
        while True:
            time.sleep(self.poll_interval)
            command = self._read_command(force=True)
            name = str(command.get("command", "")).strip().lower()
            if name == "stop":
                self.stop_requested = True
                self.update(state="Stopping")
                self.event("system", "stop requested while paused")
                return False
            if name in {"resume", "run", ""}:
                self.update(state="Running")
                self.event("system", "training resumed")
                return True

    def finish(self, **values: Any) -> None:
        values.setdefault("stopped", self.stop_requested)
        self.event("system", "training stopped" if self.stop_requested else "training finished", values)
        # Publish the terminal status last, after its final event is durable.
        self.update(state="Finished", **values)

    def fail(self, error: BaseException | str) -> None:
        message = str(error)
        self.event("error", message)
        self.update(state="Error", error=message)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._status)

    def _read_command(self, force: bool = False) -> dict[str, Any]:
        if self.command_path is None or not self.command_path.is_file():
            return {}
        try:
            value = json.loads(self.command_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A malformed/partially replaced command is ignored.  Godot will
            # retry on the next UI action; training must never crash because
            # a control file was momentarily unavailable.
            return {}
        if not isinstance(value, dict):
            return {}
        sequence = value.get("sequence", value.get("issued_at"))
        if not force and sequence is not None and sequence == self._last_sequence:
            # Pause and stop remain level-triggered even after acknowledgement.
            name = str(value.get("command", "")).lower()
            return value if name in {"pause", "stop"} else {}
        self._last_sequence = sequence
        return value


def from_cli_paths(
    command_path: str | None,
    status_path: str | None,
    event_path: str | None,
) -> RunControl | None:
    """Builds a controller only for managed runs (normal CLI stays untouched)."""
    if not command_path and not status_path and not event_path:
        return None
    return RunControl(command_path, status_path, event_path)


def _atomic_json_write(path: Path, values: dict[str, Any]) -> None:
    """Publishes complete JSON so a polling GUI never sees a partial object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(values, stream, default=str, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
