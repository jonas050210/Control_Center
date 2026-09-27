"""System Telemetry and State Tracking for SandboxAI (Phase 7).

Maintains persistent structured state of recording sessions, BC training progress,
RL checkpoints, evaluation metrics, and system hardware telemetry.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


class SystemTelemetry:
    """Manages persistent structured telemetry for the SandboxAI control layer."""

    def __init__(self, state_file: Union[str, Path] = "logs/system_state.json") -> None:
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = self._load_or_create()

    def _load_or_create(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "pipeline_stage": "IDLE",
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hardware": self.get_hardware_info(),
            "datasets": {"total_sessions": 0, "total_steps": 0, "total_mb": 0.0},
            "bc": {"active_checkpoint": None, "best_val_loss": None, "metrics": {}},
            "rl": {"active_checkpoint": None, "timesteps": 0, "steps_per_sec": 0.0},
            "evaluation": {"latest_benchmark": {}},
        }

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Collects current CPU, RAM, and GPU statistics."""
        vm = psutil.virtual_memory()
        info: Dict[str, Any] = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_gb": round(vm.total / (1024**3), 1),
            "ram_used_gb": round(vm.used / (1024**3), 1),
            "ram_percent": vm.percent,
            "gpu": "CPU Only",
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
        except Exception:
            pass
        return info

    def update_stage(self, stage: str) -> None:
        self.state["pipeline_stage"] = stage
        self.state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.save()

    def update_dataset_stats(self, datasets_dir: Union[str, Path] = "datasets") -> None:
        d_path = Path(datasets_dir)
        total_sessions = 0
        total_steps = 0
        total_bytes = 0

        if d_path.exists():
            for s_dir in d_path.iterdir():
                if s_dir.is_dir() and (s_dir / "metadata.json").exists():
                    total_sessions += 1
                    try:
                        with open(s_dir / "metadata.json", "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            total_steps += meta.get("summary_stats", {}).get("total_steps", 0)
                    except Exception:
                        pass
            for p in d_path.rglob("*"):
                if p.is_file():
                    total_bytes += p.stat().st_size

        self.state["datasets"] = {
            "total_sessions": total_sessions,
            "total_steps": total_steps,
            "total_mb": round(total_bytes / (1024**2), 2),
        }
        self.save()

    def update_bc_status(
        self,
        checkpoint: str,
        best_val_loss: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state["bc"] = {
            "active_checkpoint": str(checkpoint),
            "best_val_loss": round(best_val_loss, 4),
            "metrics": metrics or {},
        }
        self.save()

    def update_rl_status(
        self,
        checkpoint: str,
        timesteps: int,
        steps_per_sec: float,
    ) -> None:
        self.state["rl"] = {
            "active_checkpoint": str(checkpoint),
            "timesteps": timesteps,
            "steps_per_sec": steps_per_sec,
        }
        self.save()

    def update_evaluation(self, benchmark_results: Dict[str, Any]) -> None:
        self.state["evaluation"] = {"latest_benchmark": benchmark_results}
        self.save()

    def save(self) -> None:
        self.state["hardware"] = self.get_hardware_info()
        self.state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
