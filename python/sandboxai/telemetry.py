"""Structured training telemetry with optional CPU/GPU gauges."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
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
    """Collect one resource sample synchronously.

    This remains the right API for one-shot commands such as ``benchmark``.
    Training loops must use :class:`ResourceMonitor` instead: ``nvidia-smi``
    is an external process and can take tens or hundreds of milliseconds on
    Windows/WSL, so launching it synchronously from a PPO callback stalls
    rollout collection.
    """
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


class ResourceMonitor:
    """Periodically samples resources without blocking the training thread.

    The worker owns every potentially slow probe, including ``nvidia-smi``.
    ``snapshot()`` only copies the most recent dictionary under a lock and is
    therefore bounded independently of driver/OS latency.  Resource readings
    are diagnostics and never feed the policy, reward, curriculum, or RNG.
    """

    def __init__(self, interval_seconds: float = 5.0, autostart: bool = True) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("resource monitor interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: dict[str, Any] = {}
        self._sampled_at: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="sandboxai-resource-monitor",
            daemon=True,
        )
        self._started = False
        if autostart:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = resource_snapshot()
            sampled_at = time.monotonic()
            with self._lock:
                self._latest = sample
                self._sampled_at = sampled_at
            if self._stop.wait(self.interval_seconds):
                break

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._latest)
            sampled_at = self._sampled_at
        if sampled_at is not None:
            result["resource_sample_age_seconds"] = max(0.0, time.monotonic() - sampled_at)
        else:
            # Startup can legitimately reach the first progress callback
            # before a slow first nvidia-smi probe has completed. Reporting
            # that state is more honest than synchronously waiting for it.
            result["resource_sample_pending"] = True
        return result

    def close(self) -> None:
        self._stop.set()
        if self._started:
            # Do not turn shutdown into another telemetry stall. A stuck
            # external probe has its own timeout and the daemon may finish
            # after this bounded join.
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


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
