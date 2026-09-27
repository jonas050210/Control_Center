"""Atomic persistent state for CLI/web monitoring and future control UI."""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import psutil


def _strict_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _strict_json(value.item())
        except Exception:
            pass
    return value


class SystemTelemetry:
    """Small durable state store shared by recording, training, and evaluation."""

    _process_lock = threading.RLock()

    def __init__(self, state_file: Union[str, Path] = "logs/system_state.json") -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = self._load_or_create()

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _load_or_create(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                loaded = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except (OSError, json.JSONDecodeError):
                corrupt = self.state_file.with_suffix(".corrupt.json")
                try:
                    os.replace(self.state_file, corrupt)
                except OSError:
                    pass
        return {
            "schema_version": 2,
            "pipeline_stage": "IDLE",
            "last_updated": self._now(),
            "hardware": self.get_hardware_info(),
            "datasets": {"total_sessions": 0, "total_steps": 0, "total_mb": 0.0},
            "bc": {"active_checkpoint": None, "best_val_loss": None, "metrics": {}},
            "rl": {"active_checkpoint": None, "timesteps": 0, "steps_per_sec": 0.0},
            "evaluation": {"latest_benchmark": {}},
            "runtime": {
                "fps": 0.0,
                "steps_per_sec": 0.0,
                "episode_reward": 0.0,
                "last_action": {},
                "last_info": {},
            },
            "events": [],
        }

    def _refresh(self) -> None:
        """Merge updates written by other pipeline components/processes."""
        if not self.state_file.exists():
            return
        try:
            latest = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(latest, dict):
                self.state = latest
        except (OSError, json.JSONDecodeError):
            # Keep the in-memory state; save() will atomically replace a torn file.
            pass

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Serialize read-modify-write transactions across local processes."""
        lock_path = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        events = self.state.setdefault("events", [])
        events.append({"timestamp": self._now(), "type": event_type, "payload": payload})
        del events[:-100]

    def _save_unlocked(self, refresh_hardware: bool = True) -> None:
        if refresh_hardware:
            self.state["hardware"] = self.get_hardware_info()
        self.state["last_updated"] = self._now()
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_strict_json(self.state), indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, self.state_file)

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        info: Dict[str, Any] = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_gb": round(vm.total / (1024**3), 1),
            "ram_used_gb": round(vm.used / (1024**3), 1),
            "ram_percent": vm.percent,
            "gpu": "Unavailable",
        }
        try:
            import torch

            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**2)
                )
                info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2))
        except Exception:
            pass
        return info

    def update_stage(self, stage: str) -> None:
        with self._process_lock, self._file_lock():
            self._refresh()
            previous = self.state.get("pipeline_stage")
            self.state["pipeline_stage"] = stage
            self._append_event("stage", {"from": previous, "to": stage})
            self._save_unlocked()

    def update_dataset_stats(self, datasets_dir: Union[str, Path] = "datasets") -> None:
        root = Path(datasets_dir)
        total_sessions = total_steps = total_bytes = 0
        valid_sessions = 0
        if root.exists():
            for metadata_path in root.rglob("metadata.json"):
                total_sessions += 1
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    total_steps += int(metadata.get("summary_stats", {}).get("total_steps", 0))
                    if metadata.get("summary_stats", {}).get("status", "complete") == "complete":
                        valid_sessions += 1
                except Exception:
                    continue
            total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        dataset_state = {
            "total_sessions": total_sessions,
            "complete_sessions": valid_sessions,
            "total_steps": total_steps,
            "total_mb": round(total_bytes / (1024**2), 2),
        }
        with self._process_lock, self._file_lock():
            self._refresh()
            self.state["datasets"] = dataset_state
            self._save_unlocked()

    def update_bc_status(
        self, checkpoint: str, best_val_loss: float, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._process_lock, self._file_lock():
            self._refresh()
            self.state["bc"] = {
                "active_checkpoint": str(checkpoint),
                "best_val_loss": round(float(best_val_loss), 6),
                "metrics": metrics or {},
            }
            self._save_unlocked()

    def update_rl_status(self, checkpoint: str, timesteps: int, steps_per_sec: float) -> None:
        with self._process_lock, self._file_lock():
            self._refresh()
            self.state["rl"] = {
                "active_checkpoint": str(checkpoint),
                "timesteps": int(timesteps),
                "steps_per_sec": round(float(steps_per_sec), 3),
            }
            self._save_unlocked()

    def update_evaluation(self, benchmark_results: Dict[str, Any]) -> None:
        with self._process_lock, self._file_lock():
            self._refresh()
            self.state["evaluation"] = {"latest_benchmark": benchmark_results}
            self._save_unlocked()

    def update_runtime(
        self,
        fps: float,
        steps_per_sec: float,
        episode_reward: float,
        action: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._process_lock, self._file_lock():
            self._refresh()
            self.state["runtime"] = {
                "fps": round(float(fps), 2),
                "steps_per_sec": round(float(steps_per_sec), 2),
                "episode_reward": round(float(episode_reward), 5),
                "last_action": action or {},
                "last_info": info or {},
            }
            self._save_unlocked(refresh_hardware=False)

    def record_event(self, event_type: str, payload: Dict[str, Any], save: bool = True) -> None:
        if not save:
            self._append_event(event_type, payload)
            return
        with self._process_lock, self._file_lock():
            self._refresh()
            self._append_event(event_type, payload)
            self._save_unlocked(refresh_hardware=False)

    def save(self, refresh_hardware: bool = True) -> None:
        with self._process_lock, self._file_lock():
            self._save_unlocked(refresh_hardware)
