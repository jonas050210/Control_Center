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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time
from typing import Any, Iterable, Sequence

from .contract import OBSERVATION_FIELD_COUNT
from .godot_env import GodotBatchClient, PendingRequest
from .training_profile import PrefixedProfiler

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is a hard dependency of the adapter
    np = None


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


def recommended_worker_count(
    environment_count: int,
    cpu_count: int | None = None,
    reserved_cores: int = 2,
) -> int:
    """A conservative default for ``--env-workers auto``.

    One Godot bridge saturates about one core, so the useful worker count
    is bounded by both the environment count and the number of cores left
    after reserving some for the trainer process (PPO update, evaluation,
    telemetry) and the OS. ``os.cpu_count()`` reports logical threads;
    simulation is compute-bound scalar GDScript, so hyper-threads are
    counted at half value.
    """
    if environment_count < 1:
        raise ValueError("environment_count must be >= 1")
    logical = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    physical_estimate = max(1, (logical + 1) // 2) if logical > 4 else logical
    budget = max(1, physical_estimate - max(0, int(reserved_cores)))
    return max(1, min(int(environment_count), budget))


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
        self.clients: list[GodotBatchClient] = []
        self._pending: list[PendingRequest | None] = [None] * self.worker_count

        def build(shard: ShardSpec) -> GodotBatchClient:
            shard_profiler = (
                PrefixedProfiler(profiler, f"worker{shard.worker}.")
                if profiler is not None
                else None
            )
            return GodotBatchClient(
                environment_count=shard.count,
                # Identical per-environment seeds to the single-process
                # layout: the engine seeds env j with base_seed + j.
                seed=base_seed + shard.offset,
                compact_infos=self.compact_infos,
                profiler=shard_profiler,
                **kwargs,
            )

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
            except Exception:  # pragma: no cover - diagnostics only
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
            try:
                self.clients[worker].transport.receive(request)
            except Exception:  # pragma: no cover - shutdown best effort
                pass

    # -- GodotBatchClient surface --------------------------------------

    def reset(self, seed: int | None = None):
        def payload(shard: ShardSpec) -> dict[str, Any]:
            # A global seed base maps to shard-local `seed + offset` so
            # environment i keeps receiving seed + i exactly as it would
            # in a single process.
            shard_seed = -1 if seed is None else int(seed) + shard.offset
            return {"cmd": "reset", "seed": shard_seed}

        responses = self._broadcast(payload)
        observations = np.concatenate(
            [np.asarray(response["observations"], dtype=np.float32) for response in responses],
            axis=0,
        )
        infos: list[dict[str, Any]] = []
        for response in responses:
            infos.extend(response.get("infos", []))
        return observations, infos

    def step(self, actions):
        profiler = self.profiler
        started = time.perf_counter() if profiler is not None else 0.0
        if hasattr(actions, "tolist"):
            actions = actions.tolist()
        else:
            actions = list(actions)
        if len(actions) != self.environment_count:
            raise ValueError(
                f"expected {self.environment_count} actions, got {len(actions)}"
            )

        def payload(shard: ShardSpec) -> dict[str, Any]:
            return {
                "cmd": "step",
                "actions": actions[shard.offset : shard.stop],
                "compact_infos": self.compact_infos,
            }

        responses = self._broadcast(payload)
        convert_started = time.perf_counter() if profiler is not None else 0.0
        observations = np.concatenate(
            [np.asarray(response["observations"], dtype=np.float32) for response in responses],
            axis=0,
        )
        rewards = np.concatenate(
            [np.asarray(response["rewards"], dtype=np.float32) for response in responses],
            axis=0,
        )
        dones = np.concatenate(
            [np.asarray(response["dones"], dtype=np.bool_) for response in responses],
            axis=0,
        )
        infos: list[dict[str, Any]] = []
        for worker, response in enumerate(responses):
            shard_infos = response.get("infos")
            if shard_infos is None:
                shard_infos = [{} for _ in range(self.shards[worker].count)]
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
            try:
                client.close()
            except Exception:  # pragma: no cover - shutdown best effort
                pass
        self.clients = []

    # -- introspection --------------------------------------------------

    def worker_layout(self) -> list[dict[str, int]]:
        return [
            {"worker": shard.worker, "offset": shard.offset, "count": shard.count}
            for shard in self.shards
        ]

    def __enter__(self) -> "ShardedBatchClient":
        return self

    def __exit__(self, *_args: Iterable[Any]) -> None:
        self.close()
