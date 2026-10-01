"""Gymnasium/SB3 adapters for the headless Godot JSON-lines bridge."""

from __future__ import annotations

import contextlib
import json
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .training_profile import PrefixedProfiler, TrainingProfiler

    ## Either the run-wide profiler or one of its prefixed views: the
    ## sharded client hands each worker its own view so worker transport
    ## timings do not collapse into one bucket.
    ProfilerLike = TrainingProfiler | PrefixedProfiler

from .config import find_godot_executable
from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from .wsl import WindowsInterop, normalize_host_path

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class PendingRequest:
    """Bookkeeping for one in-flight bridge request.

    Deliberately tiny and allocation-cheap: one of these exists per
    outstanding request per shard on every environment step.
    """

    __slots__ = ("command", "total_started", "wait_started", "deadline")

    def __init__(
        self,
        command: str,
        total_started: float,
        wait_started: float,
        deadline: float,
    ) -> None:
        self.command = command
        self.total_started = total_started
        self.wait_started = wait_started
        self.deadline = deadline


class GodotProcessTransport:
    def __init__(
        self,
        project_path: str | Path,
        godot_executable: str = "godot",
        environment_count: int = 1,
        enemy_count: int = 1,
        seed: int = 1234,
        curriculum_level: int = 3,
        request_timeout: float = 30.0,
        self_play: bool = False,
        profiler: ProfilerLike | None = None,
    ) -> None:
        project = Path(normalize_host_path(project_path)).expanduser().resolve()
        if not project.exists():
            raise FileNotFoundError(f"Godot project path does not exist: {project}")
        executable = find_godot_executable(godot_executable)
        self.executable = executable
        self.request_timeout = request_timeout
        self.profiler = profiler
        # WSL driving the Windows Godot build: the project path must reach the
        # engine in Windows form (C:\...) and the spawn must survive a refused
        # direct .exe launch (cmd.exe fallback). Everywhere else this adapter
        # is an exact pass-through.
        interop = WindowsInterop(executable)
        command = [
            executable,
            "--headless",
            "--path",
            interop.windows_path(project),
            "--script",
            "res://scripts/rl/rl_server.gd",
            "--",
            "--stdio",
            "--env-count",
            str(environment_count),
            "--enemy-count",
            str(enemy_count),
            "--seed",
            str(seed),
            "--curriculum-level",
            str(curriculum_level),
        ]
        if self_play:
            # Two-agent match batch (league evaluation); see
            # scripts/rl/self_play_adapter.gd for the wire shapes.
            command += ["--self-play", "1"]
        if profiler is not None:
            # Godot's aggregate server timings split the Python-side
            # wait_response bucket into parse/simulation/encode/write phases.
            command += ["--profile", "1"]
        self.process = interop.popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._closed = False
        # Both output pipes are drained by daemon threads. Draining stderr is
        # not optional: a long training run in which Godot logs warnings would
        # otherwise fill the OS pipe buffer and deadlock the whole bridge.
        # Reading stdout through a queue makes request_timeout enforceable
        # portably (select() does not work on pipes on Windows).
        #
        # The queue is unbounded on purpose, unlike the stderr tail below.
        # A maxsize would apply backpressure to the pump thread, which
        # would let the OS pipe buffer fill, which would block Godot's
        # next write - reintroducing exactly the deadlock the previous
        # paragraph exists to prevent. Growth is self-limiting anyway:
        # stdout carries the request/response protocol, so a shard that
        # produced faster than the trainer consumed would be one that had
        # already stopped answering requests.
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.spaces = self.request({"cmd": "spaces"})
        except BaseException:
            # BaseException, not Exception: a Ctrl-C during the handshake
            # would otherwise leak the Godot process and its pump threads.
            self.close()
            raise

    def _pump_stdout(self) -> None:
        try:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self._stdout_lines.put(line)
        except ValueError:  # stream closed while shutting down
            pass
        finally:
            self._stdout_lines.put(None)  # EOF sentinel

    def _pump_stderr(self) -> None:
        try:
            assert self.process.stderr is not None
            for line in self.process.stderr:
                self._stderr_tail.append(line)
        except ValueError:  # stream closed while shutting down
            pass

    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail)[-2000:]

    def send(self, payload: dict[str, Any]) -> PendingRequest:
        """Writes one request and returns without waiting for the answer.

        Split from :meth:`request` so several bridge processes can be kept
        busy at once (see :mod:`sandboxai.sharded_env`): the caller sends to
        every shard first and only then collects the responses, which is
        what turns N serial Godot processes into N concurrently simulating
        ones. Single-process callers keep using :meth:`request`, which is
        exactly ``receive(send(payload))``.
        """
        if self._closed or self.process.poll() is not None:
            raise RuntimeError("Godot bridge process is not running")
        assert self.process.stdin is not None
        profiler = self.profiler
        command = str(payload.get("cmd", "unknown"))
        total_started = time.perf_counter() if profiler is not None else 0.0
        encode_started = total_started
        encoded = json.dumps(payload, separators=(",", ":")) + "\n"
        if profiler is not None:
            profiler.record(f"bridge.{command}.json_encode", time.perf_counter() - encode_started)
            profiler.add(f"bridge.{command}.request_bytes", len(encoded.encode("utf-8")))
        try:
            write_started = time.perf_counter() if profiler is not None else 0.0
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
            if profiler is not None:
                profiler.record(
                    f"bridge.{command}.write_flush", time.perf_counter() - write_started
                )
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Godot bridge pipe is broken. {self.stderr_tail()}") from exc
        return PendingRequest(
            command=command,
            total_started=total_started,
            wait_started=time.perf_counter() if profiler is not None else 0.0,
            deadline=time.monotonic() + self.request_timeout,
        )

    def receive(self, pending: PendingRequest) -> dict[str, Any]:
        """Blocks until the response to `pending` arrives (or times out)."""
        profiler = self.profiler
        command = pending.command
        # Godot can emit startup informational lines. Only protocol JSON is
        # accepted; a malformed protocol response is a hard error.
        while True:
            remaining = pending.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Godot bridge request timed out after {self.request_timeout}s. "
                    f"{self.stderr_tail()}"
                )
            try:
                line = self._stdout_lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"Godot bridge exited without a response. {self.stderr_tail()}")
            decode_started = time.perf_counter() if profiler is not None else 0.0
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if profiler is not None:
                now = time.perf_counter()
                profiler.record(
                    f"bridge.{command}.wait_response", decode_started - pending.wait_started
                )
                profiler.record(f"bridge.{command}.json_decode", now - decode_started)
                profiler.record(f"bridge.{command}.total", now - pending.total_started)
                profiler.add(f"bridge.{command}.response_bytes", len(line.encode("utf-8")))
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Godot bridge returned an error"))
            return response

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.receive(self.send(payload))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                if self.process.stdin:
                    self.process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    self.process.stdin.flush()
                self.process.wait(timeout=3.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=3.0)
        # The process has exited (or been killed), so the pump threads see
        # EOF and finish; join briefly before closing their streams.
        for worker in (
            getattr(self, "_stdout_thread", None),
            getattr(self, "_stderr_thread", None),
        ):
            if worker is not None:
                worker.join(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                with contextlib.suppress(OSError):
                    stream.close()

    def __enter__(self) -> GodotProcessTransport:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class GodotBatchClient:
    def __init__(
        self,
        compact_infos: bool = False,
        profiler: ProfilerLike | None = None,
        **kwargs: Any,
    ) -> None:
        self.profiler = profiler
        self.compact_infos = bool(compact_infos)
        self.transport = GodotProcessTransport(profiler=profiler, **kwargs)
        if np is None:
            self.close()
            raise RuntimeError("numpy is required by the Python environment adapter")
        self.environment_count = int(kwargs.get("environment_count", 1))
        self.observation_dim = int(self.transport.spaces["observation_space"]["size"])
        # The observation dimension is read from the bridge so the Python
        # side never hardcodes it — but it must still match the documented
        # contract (contract.OBSERVATION_FIELD_COUNT), otherwise the Godot
        # and Python halves of the contract have drifted apart and any
        # trained policy / BC checkpoint would be silently incompatible.
        if self.observation_dim != OBSERVATION_FIELD_COUNT:
            self.close()
            raise RuntimeError(
                f"Godot bridge reports a {self.observation_dim}-float observation "
                f"space, but the Python contract (python/sandboxai/contract.py) "
                f"defines {OBSERVATION_FIELD_COUNT} floats. The two halves of the "
                "observation contract are out of sync; update contract.py (and the "
                "docs) to match scripts/core/observation.gd."
            )
        # A matching observation width alone is insufficient: a reordered or
        # resized MultiDiscrete action component silently changes every
        # trained policy's meaning. Validate the full wire declaration during
        # the handshake before the first reset or training update.
        try:
            reported_nvec = tuple(
                int(value) for value in self.transport.spaces["action_space"]["nvec"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.close()
            raise RuntimeError(
                "Godot bridge did not provide a valid MultiDiscrete action nvec "
                "in its spaces handshake"
            ) from exc
        if reported_nvec != ACTION_NVEC:
            self.close()
            raise RuntimeError(
                f"Godot bridge reports action nvec {reported_nvec}, but the Python contract "
                f"defines {ACTION_NVEC}. The two halves of the action contract are out of sync; "
                "update contract.py, scripts/core/action.gd and the docs together."
            )

    def reset_send(self, seed: int | None = None) -> PendingRequest:
        return self.transport.send({"cmd": "reset", "seed": -1 if seed is None else int(seed)})

    def reset_receive(self, pending: PendingRequest):
        response = self.transport.receive(pending)
        return np.asarray(response["observations"], dtype=np.float32), response.get("infos", [])

    def reset(self, seed: int | None = None):
        return self.reset_receive(self.reset_send(seed))

    def reset_indices(self, indices: list[int], seed: int | None = None):
        response = self.transport.request(
            {
                "cmd": "reset_indices",
                "indices": indices,
                "seed": -1 if seed is None else int(seed),
            }
        )
        return response.get("results", [])

    def step_send(self, actions) -> PendingRequest:
        if hasattr(actions, "tolist"):
            actions = actions.tolist()
        return self.transport.send(
            {
                "cmd": "step",
                "actions": actions,
                "compact_infos": self.compact_infos,
            }
        )

    def step_receive(self, pending: PendingRequest):
        profiler = self.profiler
        response = self.transport.receive(pending)
        convert_started = time.perf_counter() if profiler is not None else 0.0
        observations = np.asarray(response["observations"], dtype=np.float32)
        rewards = np.asarray(response["rewards"], dtype=np.float32)
        dones = np.asarray(response["dones"], dtype=np.bool_)
        infos = response.get("infos", [{} for _ in range(self.environment_count)])
        if profiler is not None:
            profiler.record("env.numpy_conversion", time.perf_counter() - convert_started)
        return observations, rewards, dones, infos

    def step(self, actions):
        profiler = self.profiler
        started = time.perf_counter() if profiler is not None else 0.0
        result = self.step_receive(self.step_send(actions))
        if profiler is not None:
            profiler.record("env.step_total", time.perf_counter() - started)
        return result

    def health_check(self):
        return self.transport.request({"cmd": "health_check"}).get("health", [])

    def reward_breakdown(self):
        return self.transport.request({"cmd": "reward_breakdown"}).get("breakdowns", [])

    def set_curriculum(self, level: int):
        return self.transport.request({"cmd": "set_curriculum", "level": int(level)})

    def metrics(self):
        """Per-environment episode metrics (rl_server.gd `metrics` command)."""
        return self.transport.request({"cmd": "metrics"}).get("metrics", [])

    def set_episode_plans(self, plans: list[dict[str, Any]]):
        """Stages the next episode for each listed environment.

        `plans` entries carry an `index` plus the
        ``EpisodePlan.replay_header_fields()`` fields (seed, map_id,
        scenario, lighting, enemy_count, curriculum_level). Validation is
        atomic on the Godot side: one bad plan fails the whole batch, and
        the staging semantics (pending/current/consumed-on-auto-reset) are
        documented on ``SimulationManager.pending_plans``.
        """
        if not plans:
            return {"ok": True, "staged": []}
        return self.transport.request({"cmd": "set_episode_plans", "plans": plans})

    def episode_conditions(self):
        """The resolved per-environment episode configuration (ground truth)."""
        return self.transport.request({"cmd": "episode_conditions"}).get("conditions", [])

    def ping(self):
        return self.transport.request({"cmd": "ping"})

    def profile_snapshot(self) -> dict[str, Any]:
        """Returns optional aggregate timings from a profiling Godot server."""
        if self.profiler is None:
            return {"available": False, "reason": "profiling disabled"}
        try:
            return self.transport.request({"cmd": "profile_snapshot"}).get("profile", {})
        except RuntimeError as exc:
            # Older/fake bridges need not implement diagnostics. Training and
            # its Python-side profile remain valid when this optional command
            # is unavailable.
            return {"available": False, "reason": str(exc)}

    def close(self) -> None:
        self.transport.close()


def make_batch_client(**kwargs: Any):
    """Builds the single-process or multi-process batch client.

    ``worker_count`` (default 1) selects how many Godot processes host the
    requested ``environment_count``. One process keeps the historical
    behaviour bit-for-bit; more than one shards the environments across
    concurrently simulating processes with identical per-environment seeds
    (see :mod:`sandboxai.sharded_env`).
    """
    worker_count = int(kwargs.pop("worker_count", 1) or 1)
    environment_count = int(kwargs.get("environment_count", 1) or 1)
    if worker_count <= 1 or environment_count <= 1:
        return GodotBatchClient(**kwargs)
    from .sharded_env import ShardedBatchClient

    return ShardedBatchClient(worker_count=worker_count, **kwargs)


if gym is not None:

    class GodotGymEnv(gym.Env):  # type: ignore[misc]
        """Single-environment Gymnasium wrapper, useful for evaluation."""

        metadata = {"render_modes": []}

        def __init__(self, **kwargs: Any) -> None:
            kwargs["environment_count"] = 1
            # One environment can only live in one process.
            kwargs.pop("worker_count", None)
            self.client = GodotBatchClient(**kwargs)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.client.observation_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_NVEC, dtype=np.int64))

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            observations, infos = self.client.reset(seed)
            return observations[0], infos[0] if infos else {}

        def step(self, action):
            observations, rewards, dones, infos = self.client.step(
                np.asarray(action).reshape(1, -1)
            )
            info = infos[0] if infos else {}
            reason = str(info.get("done_reason", info.get("metrics", {}).get("done_reason", "")))
            done = bool(dones[0])
            terminated = done and reason != "timeout"
            truncated = done and reason == "timeout"
            info["TimeLimit.truncated"] = truncated
            observation = observations[0]
            if done and "terminal_observation" in info:
                # The bridge auto-resets on episode end and returns the new
                # episode's first observation; Gymnasium semantics require
                # step() to return the TERMINAL observation instead, with the
                # caller invoking reset() to start the next episode.
                observation = np.asarray(info["terminal_observation"], dtype=np.float32)
            return observation, float(rewards[0]), terminated, truncated, info

        def close(self):
            self.client.close()

else:

    class GodotGymEnv:  # type: ignore[no-redef] # pragma: no cover
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("gymnasium is required; install the training dependencies")


try:
    from stable_baselines3.common.vec_env import VecEnv  # type: ignore
except ImportError:  # pragma: no cover
    VecEnv = None  # type: ignore[assignment,misc]


if VecEnv is not None:

    class GodotVecEnv(VecEnv):  # type: ignore[misc]
        """One Godot process containing N independent vector environments.

        Optional instrumentation hooks (used by the integrated training
        pipeline; both default to None and cost nothing when unset):

        ``reset_hook(observations)``
            Called at the end of every ``reset()``.
        ``step_hook(actions, observations, rewards, dones, infos)``
            Called at the end of every ``step_wait()`` with the exact data
            about to be returned, before SB3 consumes it. The hook runs in
            the trainer thread, so it is race-free to issue further bridge
            requests (e.g. staging the next episode plan) from inside it.
        """

        def __init__(self, **kwargs: Any) -> None:
            if np is None or spaces is None:
                raise RuntimeError("numpy and gymnasium are required for PPO")
            # `worker_count > 1` spreads the environments over several
            # Godot processes; the client surface is identical either way.
            self.client = make_batch_client(**kwargs)
            self.profiler = self.client.profiler
            self.environment_count = self.client.environment_count
            self.worker_count = int(getattr(self.client, "worker_count", 1))
            observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.client.observation_dim,),
                dtype=np.float32,
            )
            action_space = spaces.MultiDiscrete(np.asarray(ACTION_NVEC, dtype=np.int64))
            super().__init__(self.environment_count, observation_space, action_space)
            self.actions = None
            self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
            self.reset_hook = None
            self.step_hook = None

        def reset(self):
            # Honor seeds requested through VecEnv.seed() (SB3 calls it when a
            # training seed is configured); otherwise keep the bridge's own
            # deterministic seeded stream by not re-seeding.
            seed = None
            pending = getattr(self, "_seeds", None)
            if pending and pending[0] is not None:
                seed = int(pending[0])
            observations, self.reset_infos = self.client.reset(seed)
            self._reset_seeds()
            if self.reset_hook is not None:
                hook_started = time.perf_counter() if self.profiler is not None else 0.0
                self.reset_hook(observations)
                if self.profiler is not None:
                    self.profiler.record("pipeline.reset_hook", time.perf_counter() - hook_started)
            return observations

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            observations, rewards, dones, infos = self.client.step(self.actions)
            for index, done in enumerate(dones):
                if done:
                    info = infos[index]
                    if "terminal_observation" in info:
                        info["terminal_observation"] = np.asarray(
                            info["terminal_observation"], dtype=np.float32
                        )
                    reason = str(
                        info.get("done_reason", info.get("metrics", {}).get("done_reason", ""))
                    )
                    info["TimeLimit.truncated"] = reason == "timeout"
            if self.step_hook is not None:
                hook_started = time.perf_counter() if self.profiler is not None else 0.0
                self.step_hook(self.actions, observations, rewards, dones, infos)
                if self.profiler is not None:
                    self.profiler.record("pipeline.step_hook", time.perf_counter() - hook_started)
            return observations, rewards, dones, infos

        def close(self):
            self.client.close()

        # Attributes SB3 and user code may read through the VecEnv API.
        # Everything here is a property of the *bridge*, not of a Python
        # sub-environment, so they are computed on demand.
        _READABLE_ATTRS = (
            "environment_count",
            "num_envs",
            "observation_dim",
            "observation_space",
            "action_space",
            "render_mode",
            "spec",
        )

        def _attr_value(self, attr_name: str):
            if attr_name == "environment_count" or attr_name == "num_envs":
                return self.environment_count
            if attr_name == "observation_dim":
                return self.client.observation_dim
            if attr_name == "observation_space":
                return self.observation_space
            if attr_name == "action_space":
                return self.action_space
            if attr_name == "render_mode":
                return getattr(self, "render_mode", None)
            if attr_name == "spec":
                return getattr(self, "spec", None)
            raise AttributeError(
                f"Godot bridge exposes no attribute {attr_name!r}; "
                f"readable attributes are {sorted(self._READABLE_ATTRS)}"
            )

        def get_attr(self, attr_name: str, indices=None):
            """Reads a bridge attribute for the selected sub-environments.

            This used to be ``values.get(attr_name)``, which returned
            ``None`` for anything it did not know about. SB3 and callbacks
            probe VecEnvs with ``get_attr`` all the time, so an unsupported
            name produced a silent ``None`` that surfaced much later as an
            unrelated ``TypeError``. Unknown names now raise immediately.
            """
            indices = self._get_indices(indices)
            value = self._attr_value(attr_name)
            return [value for _ in indices]

        def set_attr(self, attr_name: str, value, indices=None):
            """Rejects every write.

            No attribute of the Godot bridge is remotely mutable: the
            environment count is fixed by the engine process and the spaces
            are derived from the contract. The previous implementation only
            raised for *unknown* names and silently no-opped for the known
            ones, so ``set_attr("environment_count", 8)`` appeared to
            succeed and changed nothing.
            """
            raise AttributeError(
                f"Godot environment attributes are read-only over the bridge: {attr_name}"
            )

        def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
            """Forwards a supported command to the Godot process.

            The result is replicated per selected index so the return value
            matches the VecEnv contract (one entry per sub-environment)
            instead of a single bare value.
            """
            indices = self._get_indices(indices)
            if method_name == "health_check":
                result = self.client.health_check()
            elif method_name == "reward_breakdown":
                result = self.client.reward_breakdown()
            elif method_name == "set_curriculum":
                result = self.client.set_curriculum(*method_args)
            elif method_name == "metrics":
                result = self.client.metrics()
            else:
                raise AttributeError(f"Godot bridge has no remote env method {method_name}")
            return [result for _ in indices]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

else:

    class GodotVecEnv:  # type: ignore[no-redef] # pragma: no cover
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("stable-baselines3 is required for PPO")
