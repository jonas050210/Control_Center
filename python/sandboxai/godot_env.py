"""Gymnasium/SB3 adapters for the headless Godot JSON-lines bridge."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any

from .config import find_godot_executable
from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


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
    ) -> None:
        project = Path(project_path).expanduser().resolve()
        if not project.exists():
            raise FileNotFoundError(f"Godot project path does not exist: {project}")
        executable = find_godot_executable(godot_executable)
        self.executable = executable
        self.request_timeout = request_timeout
        command = [
            executable,
            "--headless",
            "--path",
            str(project),
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
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not launch Godot executable {executable!r}. "
                "Install Godot 4.7.2 and put it on PATH or pass --godot-executable."
            ) from exc
        self._closed = False
        # Both output pipes are drained by daemon threads. Draining stderr is
        # not optional: a long training run in which Godot logs warnings would
        # otherwise fill the OS pipe buffer and deadlock the whole bridge.
        # Reading stdout through a queue makes request_timeout enforceable
        # portably (select() does not work on pipes on Windows).
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.spaces = self.request({"cmd": "spaces"})
        except Exception:
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

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed or self.process.poll() is not None:
            raise RuntimeError("Godot bridge process is not running")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Godot bridge pipe is broken. {self.stderr_tail()}") from exc
        # Godot can emit startup informational lines. Only protocol JSON is
        # accepted; a malformed protocol response is a hard error.
        deadline = time.monotonic() + self.request_timeout
        while True:
            remaining = deadline - time.monotonic()
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
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Godot bridge returned an error"))
            return response

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
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        # The process has exited (or been killed), so the pump threads see
        # EOF and finish; join briefly before closing their streams.
        for worker in (getattr(self, "_stdout_thread", None), getattr(self, "_stderr_thread", None)):
            if worker is not None:
                worker.join(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "GodotProcessTransport":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class GodotBatchClient:
    def __init__(self, **kwargs: Any) -> None:
        self.transport = GodotProcessTransport(**kwargs)
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
                "Godot bridge reports a %d-float observation space, but the Python "
                "contract (python/sandboxai/contract.py) defines %d floats. The two "
                "halves of the observation contract are out of sync; update "
                "contract.py (and the docs) to match scripts/core/observation.gd."
                % (self.observation_dim, OBSERVATION_FIELD_COUNT)
            )

    def reset(self, seed: int | None = None):
        response = self.transport.request({"cmd": "reset", "seed": -1 if seed is None else int(seed)})
        return np.asarray(response["observations"], dtype=np.float32), response.get("infos", [])

    def reset_indices(self, indices: list[int], seed: int | None = None):
        response = self.transport.request({
            "cmd": "reset_indices",
            "indices": indices,
            "seed": -1 if seed is None else int(seed),
        })
        return response.get("results", [])

    def step(self, actions):
        if hasattr(actions, "tolist"):
            actions = actions.tolist()
        response = self.transport.request({"cmd": "step", "actions": actions})
        observations = np.asarray(response["observations"], dtype=np.float32)
        rewards = np.asarray(response["rewards"], dtype=np.float32)
        dones = np.asarray(response["dones"], dtype=np.bool_)
        infos = response.get("infos", [{} for _ in range(self.environment_count)])
        return observations, rewards, dones, infos

    def health_check(self):
        return self.transport.request({"cmd": "health_check"}).get("health", [])

    def reward_breakdown(self):
        return self.transport.request({"cmd": "reward_breakdown"}).get("breakdowns", [])

    def set_curriculum(self, level: int):
        return self.transport.request({"cmd": "set_curriculum", "level": int(level)})

    def ping(self):
        return self.transport.request({"cmd": "ping"})

    def close(self) -> None:
        self.transport.close()


if gym is not None:

    class GodotGymEnv(gym.Env):  # type: ignore[misc]
        """Single-environment Gymnasium wrapper, useful for evaluation."""

        metadata = {"render_modes": []}

        def __init__(self, **kwargs: Any) -> None:
            kwargs["environment_count"] = 1
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
            observations, rewards, dones, infos = self.client.step(np.asarray(action).reshape(1, -1))
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

    class GodotGymEnv:  # pragma: no cover
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("gymnasium is required; install the training dependencies")


try:
    from stable_baselines3.common.vec_env import VecEnv  # type: ignore
except ImportError:  # pragma: no cover
    VecEnv = None


if VecEnv is not None:

    class GodotVecEnv(VecEnv):  # type: ignore[misc]
        """One Godot process containing N independent vector environments."""

        def __init__(self, **kwargs: Any) -> None:
            if np is None or spaces is None:
                raise RuntimeError("numpy and gymnasium are required for PPO")
            self.client = GodotBatchClient(**kwargs)
            self.environment_count = self.client.environment_count
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
            return observations

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            observations, rewards, dones, infos = self.client.step(self.actions)
            for index, done in enumerate(dones):
                if done:
                    info = infos[index]
                    if "terminal_observation" in info:
                        info["terminal_observation"] = np.asarray(info["terminal_observation"], dtype=np.float32)
                    reason = str(info.get("done_reason", info.get("metrics", {}).get("done_reason", "")))
                    info["TimeLimit.truncated"] = (reason == "timeout")
            return observations, rewards, dones, infos

        def close(self):
            self.client.close()

        def get_attr(self, attr_name: str, indices=None):
            indices = self._get_indices(indices)
            values = {"environment_count": self.environment_count, "observation_dim": self.client.observation_dim}
            return [values.get(attr_name) for _ in indices]

        def set_attr(self, attr_name: str, value, indices=None):
            if attr_name not in {"environment_count", "observation_dim"}:
                raise AttributeError(f"Godot environment attribute is not mutable: {attr_name}")

        def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
            if method_name == "health_check":
                return self.client.health_check()
            if method_name == "reward_breakdown":
                return self.client.reward_breakdown()
            if method_name == "set_curriculum":
                return self.client.set_curriculum(*method_args)
            raise AttributeError(f"Godot bridge has no remote env method {method_name}")

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

else:

    class GodotVecEnv:  # pragma: no cover
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("stable-baselines3 is required for PPO")
