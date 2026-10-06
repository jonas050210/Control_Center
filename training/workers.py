"""CPU-aware environment worker creation and background benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing
import os
import threading
import time
from typing import Any, Callable

import numpy as np


# Only one heavyweight CPU job should saturate the arena workers at a time.
CPU_JOB_LOCK = threading.Lock()


@dataclass(frozen=True)
class Parallelism:
    requested_workers: int
    envs_per_worker: int
    effective_envs: int
    max_envs: int | None


def cpu_core_counts() -> tuple[int, int]:
    """Return (physical, logical) cores using psutil when it is available."""
    logical = max(1, os.cpu_count() or 1)
    physical = logical
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        pass
    return max(1, int(physical)), max(1, int(logical))


def auto_worker_count(max_workers: int = 20) -> int:
    """Leave one physical core for the web server and the operating system."""
    physical, _ = cpu_core_counts()
    return max(1, min(int(max_workers), physical - 1 if physical > 1 else 1))


def resolve_parallelism(
    n_workers: int,
    envs_per_worker: int = 1,
    max_envs: int | None = 24,
) -> Parallelism:
    workers = max(1, int(n_workers))
    per_worker = max(1, int(envs_per_worker))
    requested = workers * per_worker
    effective = min(requested, max_envs) if max_envs is not None else requested
    return Parallelism(workers, per_worker, max(1, int(effective)), max_envs)


def _environment_factory(
    map_name: Any,
    seed: int,
    curriculum: bool,
    initial_phase: int,
    max_episode_seconds: float,
    vision_mode: str = "coarse_los",
    frame_skip: int = 4,
) -> Callable[[], Any]:
    """Create one Monitor-wrapped environment with a process-safe factory."""
    def make_one() -> Any:
        from env.shooter_env import ShooterEnv
        from stable_baselines3.common.monitor import Monitor

        env = ShooterEnv(
            map_name=map_name,
            frame_skip=frame_skip,
            max_episode_seconds=max_episode_seconds,
            curriculum=curriculum,
            curriculum_phase=initial_phase,
            seed=seed,
            vision_mode=vision_mode,
        )
        return Monitor(env)
    return make_one


def make_vector_env(
    n_workers: int,
    envs_per_worker: int,
    map_name: Any = "Dust",
    seed: int = 0,
    curriculum: bool = True,
    initial_phase: int = 1,
    max_episode_seconds: float = 120.0,
    max_envs: int | None = 24,
    start_method: str | None = None,
    vision_mode: str = "coarse_los",
    frame_skip: int = 4,
) -> tuple[Any, Parallelism]:
    """Construct DummyVecEnv for one env or SubprocVecEnv for parallel rollouts.

    SB3's SubprocVecEnv starts one child per Gymnasium environment. In the UI,
    the requested worker count times environments-per-worker is therefore the
    number of isolated simulation processes; ``max_envs`` protects a desktop
    from accidental oversubscription.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from env.shooter_env import VISION_MODES

    if vision_mode not in VISION_MODES:
        raise ValueError(f"Unknown vision_mode {vision_mode!r}. Choose one of: {', '.join(VISION_MODES)}")
    parallelism = resolve_parallelism(n_workers, envs_per_worker, max_envs)
    factories = [
        _environment_factory(
            map_name=map_name,
            seed=seed + rank,
            curriculum=curriculum,
            initial_phase=initial_phase,
            max_episode_seconds=max_episode_seconds,
            vision_mode=vision_mode,
            frame_skip=frame_skip,
        )
        for rank in range(parallelism.effective_envs)
    ]
    if parallelism.effective_envs == 1:
        return DummyVecEnv(factories), parallelism

    methods = multiprocessing.get_all_start_methods()
    preferred = start_method or ("forkserver" if "forkserver" in methods else "spawn")
    if preferred not in methods:
        preferred = "spawn" if "spawn" in methods else methods[0]
    # ``forkserver``/``spawn`` re-import the caller's main module in every child.
    # A script without an ``if __name__ == "__main__":`` guard then aborts while
    # bootstrapping; ``fork`` inherits the loaded process and keeps training
    # working, so it is the documented fallback instead of a hard failure.
    candidates = [preferred, *(m for m in ("forkserver", "spawn", "fork") if m != preferred)]
    for method in candidates:
        if method not in methods:
            continue
        try:
            return SubprocVecEnv(factories, start_method=method), parallelism
        except RuntimeError as exc:
            if "bootstrapping" not in str(exc):
                raise
            logging.getLogger(__name__).warning(
                "Start method %r failed (%s); falling back to another one.", method, exc
            )
    raise RuntimeError("No usable multiprocessing start method for the training workers.")


def valid_benchmark_configurations() -> list[tuple[int, int]]:
    workers = (1, 2, 4, 8, 12, 16)
    envs_per_worker = (1, 2, 4, 8)
    return [(worker, env_count) for worker in workers for env_count in envs_per_worker
            if worker * env_count <= 24]


def _process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        try:
            rss += sum(child.memory_info().rss for child in process.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        return rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def run_benchmark_combo(
    n_workers: int,
    envs_per_worker: int,
    seconds: float = 20.0,
    map_name: str = "Dust",
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Measure aggregate random-policy simulation throughput for a fixed window."""
    from env.shooter_env import ACTION_NVECS

    vec_env, parallelism = make_vector_env(
        n_workers=n_workers,
        envs_per_worker=envs_per_worker,
        map_name=map_name,
        seed=700 + n_workers * 11 + envs_per_worker,
        curriculum=False,
        initial_phase=4,
        max_episode_seconds=120.0,
        max_envs=24,
    )
    num_envs = int(vec_env.num_envs)
    rng = np.random.default_rng(2026 + num_envs)
    nvec = np.asarray(ACTION_NVECS, dtype=np.int64)
    steps = 0
    active_seconds = 0.0
    ram_mb = 0.0
    status = "complete"
    try:
        vec_env.reset()
        try:
            import psutil
            psutil.cpu_percent(interval=None)  # prime the rolling CPU sampler
        except Exception:
            pass
        started = time.perf_counter()
        deadline = started + max(0.01, float(seconds))
        while time.perf_counter() < deadline:
            if stop_event is not None and stop_event.is_set():
                status = "stopped"
                break
            actions = np.column_stack([
                rng.integers(0, nvec[index], size=num_envs, endpoint=False)
                for index in range(len(nvec))
            ]).astype(np.int64)
            vec_env.step(actions)
            steps += num_envs
        active_seconds = max(1e-9, time.perf_counter() - started)
        ram_mb = _process_memory_mb()
    finally:
        vec_env.close()
    try:
        import psutil
        cpu_percent = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu_percent = 0.0
    return {
        "workers": int(n_workers),
        "envs_per_worker": int(envs_per_worker),
        "total_envs": parallelism.effective_envs,
        "steps_per_second": steps / max(active_seconds, 1e-9),
        "steps": steps,
        "cpu_percent": cpu_percent,
        "ram_mb": _process_memory_mb(),
        "seconds": active_seconds,
        "status": status,
    }


class BenchmarkRunner:
    """Thread-safe full benchmark runner driven by the control center API."""

    def __init__(self, seconds_per_combo: float = 20.0, map_name: str = "Dust") -> None:
        self.seconds_per_combo = float(seconds_per_combo)
        self.map_name = map_name
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._results: list[dict[str, Any]] = []
        self._status = "stopped"
        self._current: tuple[int, int] | None = None
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._results = []
            self._error = None
            self._status = "running"
            self._thread = threading.Thread(target=self._run, name="neural-arena-benchmark", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._status == "running":
                self._status = "stopping"

    def _run(self) -> None:
        if not CPU_JOB_LOCK.acquire(blocking=False):
            with self._lock:
                self._error = "Another CPU-heavy training or benchmark job is already active. Stop it before starting this run."
                self._status = "error"
            return
        try:
            for workers, envs in valid_benchmark_configurations():
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._current = (workers, envs)
                    self._results.append({
                        "workers": workers,
                        "envs_per_worker": envs,
                        "total_envs": workers * envs,
                        "steps_per_second": 0.0,
                        "cpu_percent": 0.0,
                        "ram_mb": 0.0,
                        "seconds": 0.0,
                        "status": "running",
                    })
                    row_index = len(self._results) - 1
                result = run_benchmark_combo(
                    workers,
                    envs,
                    seconds=self.seconds_per_combo,
                    map_name=self.map_name,
                    stop_event=self._stop_event,
                )
                with self._lock:
                    self._results[row_index] = result
                    self._current = None
                if self._stop_event.is_set():
                    break
            with self._lock:
                self._status = "stopped" if self._stop_event.is_set() else "complete"
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._status = "error"
                self._current = None
        finally:
            CPU_JOB_LOCK.release()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "results": [dict(row) for row in self._results],
                "current": self._current,
                "error": self._error,
                "total": len(valid_benchmark_configurations()),
            }
