"""Structured training telemetry with optional CPU/GPU gauges."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any


def _nvidia_smi_snapshot(timeout: float = 0.75) -> dict[str, Any]:
    """Best-effort NVIDIA utilization/VRAM sample.

    PyTorch exposes allocator memory but not whole-device utilization.  The
    Control Center and training JSONL therefore use ``nvidia-smi`` when it is
    available (Linux, WSL and Windows NVIDIA drivers all ship it) and simply
    omit these keys otherwise. No metric is guessed.
    """
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    line = next((row.strip() for row in completed.stdout.splitlines() if row.strip()), "")
    if not line:
        return {}
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return {}
    name = ", ".join(parts[: len(parts) - 4])
    util, used, total, temp = parts[-4:]
    snapshot: dict[str, Any] = {"gpu_name": name, "gpu_telemetry_source": "nvidia-smi"}
    for key, value in (
        ("gpu_utilization_percent", util),
        ("gpu_vram_used_mb", used),
        ("gpu_vram_total_mb", total),
        ("gpu_temperature_c", temp),
    ):
        try:
            snapshot[key] = float(value)
        except ValueError:
            pass
    return snapshot


def resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        import psutil  # type: ignore
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=None)
        process = psutil.Process()
        snapshot["process_rss_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    snapshot.update(_nvidia_smi_snapshot())
    return snapshot


class JsonlTelemetry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._stream = self.path.open("a", encoding="utf-8")

    def write(self, values: dict[str, Any]) -> None:
        payload = {"wall_time": time.time(), "elapsed_seconds": time.perf_counter() - self.started, **values}
        self._stream.write(json.dumps(payload, default=str) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "JsonlTelemetry":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
