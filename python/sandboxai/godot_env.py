"""Gymnasium/SB3 adapters for the headless Godot JSON-lines bridge."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - dependency error is reported at construction
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
    ) -> None:
        project = Path(project_path).expanduser().resolve()
        if not project.exists():
            raise FileNotFoundError(f"Godot project path does not exist: {project}")
        command = [
            godot_executable,
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
                f"Could not launch Godot executable {godot_executable!r}. "
                "Install Godot 4.7.2 and put it on PATH or pass --godot-executable."
            ) from exc
        self._closed = False
        try:
            self.spaces = self.request({"cmd": "spaces"})
        except Exception:
            self.close()
            raise

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed or self.process.poll() is not None:
            raise RuntimeError("Godot bridge process is not running")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        # Godot can emit a startup informational line. Only protocol JSON is
        # accepted; a malformed protocol response is a hard error.
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"Godot bridge exited without a response. {stderr[-2000:]}")
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
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()

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

    def reset(self, seed: int | None = None):
        response = self.transport.request({"cmd": "reset", "seed": -1 if seed is None else int(seed)})
        return np.asarray(response["observations"], dtype=np.float32), response.get("infos", [])

    def step(self, actions):
        if hasattr(actions, "tolist"):
            actions = actions.tolist()
        response = self.transport.request({"cmd": "step", "actions": actions})
        observations = np.asarray(response["observations"], dtype=np.float32)
        rewards = np.asarray(response["rewards"], dtype=np.float32)
        dones = np.asarray(response["dones"], dtype=np.bool_)
        infos = response.get("infos", [{} for _ in range(self.environment_count)])
        return observations, rewards, dones, infos

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
            self.action_space = spaces.MultiDiscrete(np.asarray([3, 3, 3, 3, 2], dtype=np.int64))

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
            return observations[0], float(rewards[0]), terminated, truncated, info

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
            action_space = spaces.MultiDiscrete(np.asarray([3, 3, 3, 3, 2], dtype=np.int64))
            super().__init__(self.environment_count, observation_space, action_space)
            self.actions = None
            self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        def reset(self):
            observations, self.reset_infos = self.client.reset()
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
            raise AttributeError(f"Godot bridge has no remote env method {method_name}")

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

else:

    class GodotVecEnv:  # pragma: no cover
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("stable-baselines3 is required for PPO")
