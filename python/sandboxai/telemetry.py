"""Structured training telemetry with optional CPU/GPU gauges."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


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
