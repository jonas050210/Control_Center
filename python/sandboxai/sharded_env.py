"""Multi-process (sharded) Godot environment execution.

Why this exists
---------------
One ``rl_server.gd`` process steps its environments **serially** inside a
single GDScript thread, so a single Godot bridge saturates exactly one CPU
core no matter how many environments it hosts. On a 12-physical-core host
that caps end-to-end PPO throughput at roughly one core's worth of
simulation while eleven cores idle.

This module splits the same total environment count across *W* Godot
processes and keeps them busy simultaneously. The critical detail is the
request pipeline: every shard is **sent** its step request before any
response is read, so the shards simulate concurrently and Python only pays
the slowest shard's latency per step instead of the sum.

Determinism
-----------
Sharding is result-preserving, not merely "close enough":

* Environments never interact. ``EnvironmentCore`` owns its own RNG stream
  and ``SimulationManager`` seeds environment *i* with ``base_seed + i``
  (``simulation_manager.gd:build/reset_all``). Shard *s* covering global
  indices ``[offset, offset + count)`` is therefore launched with
  ``--seed base_seed + offset``, which reproduces exactly the same per
  environment seeds as the single-process layout.
* Shards are contiguous and ordered, and responses are concatenated in
  shard order, so the global batch index of every environment is
  unchanged. Observations, rewards, dones and infos keep their positions,
  which keeps SB3's rollout buffer layout and every seed/plan mapping
  identical.
* Episode plans (``set_episode_plans``) are routed by global index to the
  owning shard and rewritten to that shard's local index, so the
  curriculum director's per-environment streams are unaffected.

The practical consequence: a run with ``environment_count=16,
env_workers=4`` produces the same trajectories as ``environment_count=16,
env_workers=1`` given the same policy; only wall time changes. The Python
test-suite pins this with a scripted bridge
(``python/tests/test_sharded_env.py``).

Failure handling
----------------
A crashed or hung shard raises :class:`ShardFailure` naming the worker, its
global index range and the tail of that process' stderr, and every other
shard is closed before the error propagates. Partial, silently truncated
batches are never returned.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .contract import OBSERVATION_FIELD_COUNT
from .godot_env import GodotBatchClient, PendingRequest
from .training_profile import PrefixedProfiler

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is a hard dependency of the adapter
    np = None  # type: ignore[assignment]


class ShardFailure(RuntimeError):
    """A worker process failed; carries the worker identity and its range."""

    def __init__(self, worker: int, offset: int, count: int, cause: BaseException) -> None:
        super().__init__(
            f"Godot worker {worker} (environments {offset}..{offset + count - 1}) failed: {cause}"
        )
        self.worker = worker
        self.offset = offset
        self.count = count
        self.cause = cause


@dataclass(frozen=True)
class ShardSpec:
    """One worker process: which global environment indices it owns."""

    worker: int
    offset: int
    count: int

    @property
    def stop(self) -> int:
        return self.offset + self.count

    def contains(self, global_index: int) -> bool:
        return self.offset <= global_index < self.stop

    def local_index(self, global_index: int) -> int:
        return global_index - self.offset


def plan_shards(environment_count: int, worker_count: int) -> list[ShardSpec]:
    """Contiguous, deterministic, as-even-as-possible index ranges.

    The first ``environment_count % worker_count`` workers take one extra
    environment. Requesting more workers than environments is clamped
    rather than producing empty processes (an empty Godot bridge would
    still cost a process and a handshake while simulating nothing).
    """
    if environment_count < 1:
        raise ValueError("environment_count must be >= 1")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    workers = min(int(worker_count), int(environment_count))
    base, remainder = divmod(int(environment_count), workers)
    shards: list[ShardSpec] = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        shards.append(ShardSpec(worker=worker, offset=offset, count=count))
        offset += count
    return shards


def physical_core_estimate(cpu_count: int | None = None) -> int:
    """Estimate physical cores from ``os.cpu_count()``.

    ``os.cpu_count()`` reports logical threads; simulation is compute-bound
    scalar GDScript, and the Godot bridge saturates about one physical core
    per worker, so hyper-threads are counted at half value. This is the one
    place that decision is made: the worker recommendation, the benchmark
    pipeline's host-scaled sweep and the Control Center's plan all call it,
    so a 32-thread machine cannot be described as 32 cores by one of them
    and 16 by another.
    """
    logical = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    logical = max(1, logical)
    if logical <= 4:
        return logical
    return max(1, (logical + 1) // 2)


def recommended_worker_count(
    environment_count: int,
    cpu_count: int | None = None,
    reserved_cores: int = 2,
) -> int:
    """A conservative default for ``--env-workers auto``.

    One Godot bridge saturates about one core, so the useful worker count
    is bounded by both the environment count and the number of physical
    cores left after reserving some for the trainer process (PPO update,
    evaluation, telemetry) and the OS. The *sweep* may probe above this
    recommendation on purpose - measuring where throughput stops improving
    is the point of the benchmark - but a default launch uses it.
    """
    if environment_count < 1:
        raise ValueError("environment_count must be >= 1")
    budget = max(1, physical_core_estimate(cpu_count) - max(0, int(reserved_cores)))
    return max(1, min(int(environment_count), budget))


@dataclass(frozen=True)
class WorkerCompatibility:
    """Compatibility verdict for one ``environment_count × env_workers`` pair.

    This is the single place where the Environment -> Worker -> Agent
    relationship is judged, so the benchmark pipeline and the Control
    Center launcher can never disagree about what a valid topology is.
    ``errors`` make a configuration invalid (it must not be launched);
    ``warnings`` describe a valid but suboptimal topology. The shard plan
    is included so callers can display the exact environment -> worker
    mapping instead of re-deriving it.
    """

    environment_count: int
    requested_workers: int
    resolved_workers: int
    shards: tuple[ShardSpec, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def worker_compatibility(
    environment_count: int,
    env_workers: int,
    *,
    cpu_count: int | None = None,
) -> WorkerCompatibility:
    """Validate an environment/worker request against the sharding rules.

    Hard errors (mirroring what ``plan_shards`` and
    ``TrainingConfig.resolved_env_workers`` would do anyway, but reported
    instead of silently clamped, so a UI can prevent the invalid launch):

    * ``environment_count < 1``
    * ``env_workers < 1`` (0/"auto" is resolved by the caller before this
      point; an explicit worker request must be positive)
    * ``env_workers > environment_count`` — an empty shard would still cost
      a Godot process and a handshake while simulating nothing.

    Warnings (the configuration runs, but the operator should know):

    * the shard split is uneven (``environment_count % env_workers != 0``),
      listing the actual per-worker environment counts;
    * the worker count exceeds the host's physical-core estimate, which
      oversubscribes the CPU without adding simulation throughput.
    """
    errors: list[str] = []
    if environment_count < 1:
        errors.append("environment_count must be >= 1")
    if env_workers < 1:
        errors.append("env_workers must be >= 1 (0 = auto)")
    if errors:
        return WorkerCompatibility(
            environment_count=max(int(environment_count), 0),
            requested_workers=int(env_workers),
            resolved_workers=0,
            shards=(),
            errors=tuple(errors),
            warnings=(),
        )
    if env_workers > environment_count:
        errors.append(
            f"env_workers ({env_workers}) must not exceed environment_count "
            f"({environment_count}): every worker needs at least one environment"
        )
        return WorkerCompatibility(
            environment_count=int(environment_count),
            requested_workers=int(env_workers),
            resolved_workers=0,
            shards=(),
            errors=tuple(errors),
            warnings=(),
        )
    shards = tuple(plan_shards(int(environment_count), int(env_workers)))
    warnings: list[str] = []
    per_worker = [shard.count for shard in shards]
    if len(set(per_worker)) > 1:
        warnings.append(
            f"uneven shard split: {environment_count} environments over "
            f"{env_workers} workers gives {per_worker} per worker; the "
            "slowest worker caps every vector step"
        )
    logical = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    physical_estimate = max(1, (logical + 1) // 2) if logical > 4 else logical
    if env_workers > physical_estimate:
        warnings.append(
            f"{env_workers} workers exceed the estimated {physical_estimate} physical "
            f"cores; workers beyond that contend for CPU instead of adding throughput"
        )
    return WorkerCompatibility(
        environment_count=int(environment_count),
        requested_workers=int(env_workers),
        resolved_workers=len(shards),
        shards=shards,
        errors=(),
        warnings=tuple(warnings),
    )


_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def ubuntu_cpu_runtime_profile(
    *,
    cpu_count: int | None = None,
    environment_count: int | None = None,
    env_workers: int | None = None,
) -> dict[str, Any]:
    """Inspect host CPU topology and return Ubuntu-CPU optimization guidance.

    Designed for Linux/Ubuntu CPU training setups where multi-process Godot
    sharding + compact bridge payloads + controlled BLAS/PyTorch thread counts
    deliver the highest end-to-end steps/s.
    """
    import platform
    from pathlib import Path

    logical = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    affinity_cores: list[int] = []
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            affinity_cores = sorted(os.sched_getaffinity(0))
    usable_logical = len(affinity_cores) if affinity_cores else logical
    physical_estimate = max(1, (usable_logical + 1) // 2) if usable_logical > 4 else usable_logical

    cpu_model = platform.processor() or "x86_64 CPU"
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        with contextlib.suppress(OSError):
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break

    governor = "unknown"
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_path.is_file():
        with contextlib.suppress(OSError):
            governor = gov_path.read_text(encoding="utf-8", errors="ignore").strip() or "unknown"

    os_release = platform.system()
    distro_name = os_release
    os_release_path = Path("/etc/os-release")
    if os_release_path.is_file():
        with contextlib.suppress(OSError):
            for line in os_release_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("PRETTY_NAME="):
                    distro_name = line.split("=", 1)[1].strip().strip('"')
                    break

    is_linux = os_release.lower() == "linux"
    is_ubuntu = "ubuntu" in distro_name.lower()

    rec_workers = max(1, min(physical_estimate, max(1, physical_estimate - 1)))
    if env_workers is not None and int(env_workers) > 0:
        active_workers = int(env_workers)
    else:
        active_workers = rec_workers

    rec_envs = max(8, active_workers * 4)
    if environment_count is not None and int(environment_count) > 0:
        active_envs = int(environment_count)
    else:
        active_envs = rec_envs

    trainer_threads = max(1, min(4, usable_logical - active_workers))
    thread_env = {key: os.environ.get(key, "") for key in _THREAD_ENV_VARS}
    anti_thrash_active = all(os.environ.get(key) == "1" for key in _THREAD_ENV_VARS[:3])

    return {
        "os_name": distro_name,
        "is_linux": is_linux,
        "is_ubuntu": is_ubuntu,
        "cpu_model": cpu_model,
        "logical_cores": usable_logical,
        "physical_cores_est": physical_estimate,
        "affinity_count": len(affinity_cores) if affinity_cores else usable_logical,
        "cpu_governor": governor,
        "recommended_workers": rec_workers,
        "recommended_envs": rec_envs,
        "active_workers": active_workers,
        "active_envs": active_envs,
        "recommended_trainer_threads": trainer_threads,
        "anti_thrash_active": anti_thrash_active,
        "thread_env": thread_env,
        "compact_infos_recommended": True,
        "fast_bridge_active": True,
    }


def apply_ubuntu_cpu_optimizations(
    *,
    worker_count: int | None = None,
    environment_count: int | None = None,
    pin_Single_thread_blas: bool = True,
) -> dict[str, Any]:
    """Apply Ubuntu-CPU anti-thrashing and thread-budget optimizations in-process.

    When multiple Godot worker processes run concurrently on Ubuntu CPU, leaving
    OpenMP/MKL/OpenBLAS unrestricted causes every worker and NumPy/PyTorch call
    to spawn ``logical_cores`` threads, creating severe context-switch thrashing.
    Pinning BLAS env vars to ``1`` and sizing PyTorch intra-op threads to the
    remaining core budget maximizes multi-shard CPU throughput.
    """
    profile = ubuntu_cpu_runtime_profile(
        environment_count=environment_count,
        env_workers=worker_count,
    )
    applied_env: dict[str, str] = {}
    if pin_Single_thread_blas:
        for key in _THREAD_ENV_VARS:
            os.environ[key] = "1"
            applied_env[key] = "1"
        os.environ["SANDBOXAI_CPU_TURBO"] = "1"
        applied_env["SANDBOXAI_CPU_TURBO"] = "1"

    trainer_threads = int(profile["recommended_trainer_threads"])
    torch_threads_set: int | None = None
    with contextlib.suppress(Exception):
        import torch  # type: ignore

        torch.set_num_threads(trainer_threads)
        torch_threads_set = int(torch.get_num_threads())

    updated = ubuntu_cpu_runtime_profile(
        environment_count=environment_count,
        env_workers=worker_count,
    )
    updated["applied_env"] = applied_env
    updated["torch_threads"] = torch_threads_set
    updated["turbo_enabled"] = True
    return updated


class ShardedBatchClient:
    """``GodotBatchClient``-compatible facade over several bridge processes.

    Implements the subset of the client surface that the trainer, the
    pipeline, evaluation and the benchmark actually use, so callers can
    treat it as a drop-in replacement (``GodotVecEnv`` picks it up
    automatically when ``worker_count > 1``).
    """

    def __init__(
        self,
        worker_count: int = 2,
        compact_infos: bool = False,
        profiler: Any | None = None,
        startup_parallelism: bool = True,
        **kwargs: Any,
    ) -> None:
        if np is None:  # pragma: no cover - adapter requires numpy anyway
            raise RuntimeError("numpy is required by the Python environment adapter")
        environment_count = int(kwargs.pop("environment_count", 1))
        base_seed = int(kwargs.pop("seed", 1234))
        self.profiler = profiler
        self.compact_infos = bool(compact_infos)
        self.environment_count = environment_count
        self.base_seed = base_seed
        self.shards: list[ShardSpec] = plan_shards(environment_count, worker_count)
        self.worker_count = len(self.shards)
        if self.worker_count > 1:
            for env_key in _THREAD_ENV_VARS:
                os.environ.setdefault(env_key, "1")
        self.clients: list[GodotBatchClient] = []
        self._pending: list[PendingRequest | None] = [None] * self.worker_count
        # A list assignment such as ``self.clients = list(pool.map(...))``
        # happens only after *every* constructor returned. If shard two
        # failed, shard one had a live Godot process but was still invisible
        # to close(), leaking it. Register each completed bridge immediately
        # under a lock so the shared cleanup path owns every partial startup.
        started_clients: list[GodotBatchClient] = []
        started_lock = threading.Lock()

        def build(shard: ShardSpec) -> GodotBatchClient:
            shard_profiler = (
                PrefixedProfiler(profiler, f"worker{shard.worker}.")
                if profiler is not None
                else None
            )
            client = GodotBatchClient(
                environment_count=shard.count,
                # Identical per-environment seeds to the single-process
                # layout: the engine seeds env j with base_seed + j.
                seed=base_seed + shard.offset,
                compact_infos=self.compact_infos,
                profiler=shard_profiler,
                **kwargs,
            )
            with started_lock:
                started_clients.append(client)
            return client

        started = time.perf_counter()
        try:
            if startup_parallelism and self.worker_count > 1:
                # Engine + project startup is seconds per process and is
                # independent per shard; overlapping it keeps a 12-worker
                # launch close to a single-worker launch.
                with ThreadPoolExecutor(max_workers=self.worker_count) as pool:
                    self.clients = list(pool.map(build, self.shards))
            else:
                self.clients = [build(shard) for shard in self.shards]
        except BaseException:
            # See the registration above. The executor waits for submitted
            # constructors on exit, so this owns every client that completed
            # before (or alongside) the failing constructor.
            self.clients = started_clients
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - started
        dims = {client.observation_dim for client in self.clients}
        if len(dims) != 1 or dims.pop() != OBSERVATION_FIELD_COUNT:
            self.close()
            raise RuntimeError(
                "Godot workers disagree about the observation contract "
                f"(expected {OBSERVATION_FIELD_COUNT} floats from every shard)"
            )
        self.observation_dim = OBSERVATION_FIELD_COUNT
        if profiler is not None:
            profiler.set_metadata(
                env_workers=self.worker_count,
                env_worker_layout=[[s.offset, s.count] for s in self.shards],
            )
            profiler.record("env.worker_startup", self.startup_seconds)

    # -- fan-out helpers ------------------------------------------------

    def _fail(self, worker: int, cause: BaseException) -> ShardFailure:
        shard = self.shards[worker]
        client = self.clients[worker] if worker < len(self.clients) else None
        tail = ""
        if client is not None:
            try:
                tail = client.transport.stderr_tail()
            except (OSError, AttributeError, ValueError):  # pragma: no cover - diagnostics
                tail = ""
        failure = ShardFailure(shard.worker, shard.offset, shard.count, cause)
        if tail:
            failure.args = (f"{failure.args[0]}\n--- worker stderr ---\n{tail}",)
        return failure

    def _broadcast(self, payload_for: Any) -> list[dict[str, Any]]:
        """Sends one request per shard, then collects every response."""
        pending: list[PendingRequest | None] = [None] * self.worker_count
        for worker, client in enumerate(self.clients):
            try:
                pending[worker] = client.transport.send(payload_for(self.shards[worker]))
            except BaseException as exc:  # noqa: BLE001 - re-raised as ShardFailure
                self._drain(pending)
                failure = self._fail(worker, exc)
                self.close()
                raise failure from exc
        responses: list[dict[str, Any]] = []
        first_error: ShardFailure | None = None
        for worker, client in enumerate(self.clients):
            request = pending[worker]
            assert request is not None
            try:
                responses.append(client.transport.receive(request))
            except BaseException as exc:  # noqa: BLE001
                responses.append({})
                if first_error is None:
                    first_error = self._fail(worker, exc)
        if first_error is not None:
            self.close()
            raise first_error
        return responses

    def _drain(self, pending: Sequence[PendingRequest | None]) -> None:
        """Best-effort collection of already-sent requests after a failure."""
        for worker, request in enumerate(pending):
            if request is None:
                continue
            # Draining in-flight requests is best effort; a worker that
            # already exited simply has nothing left to drain.
            with contextlib.suppress(Exception):
                self.clients[worker].transport.receive(request)

    # -- GodotBatchClient surface --------------------------------------

    def reset(self, seed: int | None = None):
        def payload(shard: ShardSpec) -> dict[str, Any]:
            # A global seed base maps to shard-local `seed + offset` so
            # environment i keeps receiving seed + i exactly as it would
            # in a single process.
            shard_seed = -1 if seed is None else int(seed) + shard.offset
            return {"cmd": "reset", "seed": shard_seed}

        responses = self._broadcast(payload)
        observations = np.empty(
            (self.environment_count, self.observation_dim),
            dtype=np.float32,
        )
        infos: list[dict[str, Any]] = []
        for worker, response in enumerate(responses):
            shard = self.shards[worker]
            observations[shard.offset : shard.stop] = response["observations"]
            infos.extend(response.get("infos", []))
        return observations, infos

    def step(self, actions):
        profiler = self.profiler
        started = time.perf_counter() if profiler is not None else 0.0
        actions = actions.tolist() if hasattr(actions, "tolist") else list(actions)
        if len(actions) != self.environment_count:
            raise ValueError(f"expected {self.environment_count} actions, got {len(actions)}")

        def payload(shard: ShardSpec) -> dict[str, Any]:
            return {
                "cmd": "step",
                "actions": actions[shard.offset : shard.stop],
                "compact_infos": self.compact_infos,
            }

        responses = self._broadcast(payload)
        convert_started = time.perf_counter() if profiler is not None else 0.0
        observations = np.empty(
            (self.environment_count, self.observation_dim),
            dtype=np.float32,
        )
        rewards = np.empty(self.environment_count, dtype=np.float32)
        dones = np.empty(self.environment_count, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        for worker, response in enumerate(responses):
            shard = self.shards[worker]
            observations[shard.offset : shard.stop] = response["observations"]
            rewards[shard.offset : shard.stop] = response["rewards"]
            dones[shard.offset : shard.stop] = response["dones"]
            shard_infos = response.get("infos")
            if shard_infos is None:
                shard_infos = [{} for _ in range(shard.count)]
            infos.extend(shard_infos)
        if profiler is not None:
            profiler.record("env.numpy_conversion", time.perf_counter() - convert_started)
            profiler.record("env.step_total", time.perf_counter() - started)
        return observations, rewards, dones, infos

    def set_episode_plans(self, plans: list[dict[str, Any]]):
        """Routes globally indexed plans to their owning worker.

        Validation stays atomic *per worker* (the Godot side rejects the
        whole batch on one bad plan); an out-of-range global index is
        rejected here before anything is staged. Requests are dispatched as
        one fan-out operation, and any remote rejection closes every shard,
        so callers cannot continue on a partially staged distribution.
        """
        if not plans:
            return {"ok": True, "staged": []}
        per_worker: list[list[dict[str, Any]]] = [[] for _ in self.shards]
        for plan in plans:
            index = int(plan.get("index", -1))
            if index < 0 or index >= self.environment_count:
                raise ValueError(
                    f"episode plan index {index} is outside 0..{self.environment_count - 1}"
                )
            shard = self.shards[self._worker_of(index)]
            local = dict(plan)
            local["index"] = shard.local_index(index)
            per_worker[shard.worker].append(local)
        # Dispatch every shard before collecting any response, just like a
        # simulation step. If one shard rejects/fails, _broadcast closes the
        # entire facade: callers can never continue with a partially staged
        # worker set.
        responses = self._broadcast(
            lambda shard: {
                "cmd": "set_episode_plans",
                "plans": per_worker[shard.worker],
            }
        )
        staged: list[int] = []
        for worker, response in enumerate(responses):
            if not response.get("ok", True):
                failure = self._fail(
                    worker, RuntimeError(str(response.get("error", "plan staging failed")))
                )
                self.close()
                raise failure
            offset = self.shards[worker].offset
            staged.extend(offset + int(value) for value in response.get("staged", []))
        return {"ok": True, "staged": sorted(staged)}

    def _worker_of(self, global_index: int) -> int:
        for shard in self.shards:
            if shard.contains(global_index):
                return shard.worker
        raise ValueError(f"environment index {global_index} is not owned by any worker")

    def _gather(self, method: str) -> list[Any]:
        values: list[Any] = []
        for worker, client in enumerate(self.clients):
            try:
                values.extend(getattr(client, method)())
            except BaseException as exc:  # noqa: BLE001
                failure = self._fail(worker, exc)
                self.close()
                raise failure from exc
        return values

    def metrics(self):
        return self._gather("metrics")

    def health_check(self):
        return self._gather("health_check")

    def reward_breakdown(self):
        return self._gather("reward_breakdown")

    def episode_conditions(self):
        return self._gather("episode_conditions")

    def set_curriculum(self, level: int):
        responses = self._broadcast(lambda _shard: {"cmd": "set_curriculum", "level": int(level)})
        return {"ok": True, "curriculum_level": int(level), "workers": len(responses)}

    def ping(self):
        self._broadcast(lambda _shard: {"cmd": "ping"})
        return {"ok": True, "pong": True, "workers": self.worker_count}

    def profile_snapshot(self) -> dict[str, Any]:
        """Per-worker Godot server profiles under a worker-indexed key."""
        if self.profiler is None:
            return {"available": False, "reason": "profiling disabled"}
        snapshots: dict[str, Any] = {}
        for worker, client in enumerate(self.clients):
            snapshots[f"worker{worker}"] = client.profile_snapshot()
        available = any(bool(value.get("available")) for value in snapshots.values())
        return {
            "available": available,
            "workers": self.worker_count,
            "worker_profiles": snapshots,
        }

    def close(self) -> None:
        for client in getattr(self, "clients", []):
            # Every client gets a close attempt even if an earlier one
            # raised, so one dead bridge cannot leak the others.
            with contextlib.suppress(Exception):
                client.close()
        self.clients = []

    # -- introspection --------------------------------------------------

    def worker_layout(self) -> list[dict[str, int]]:
        return [
            {"worker": shard.worker, "offset": shard.offset, "count": shard.count}
            for shard in self.shards
        ]

    def __enter__(self) -> ShardedBatchClient:
        return self

    def __exit__(self, *_args: Iterable[Any]) -> None:
        self.close()
