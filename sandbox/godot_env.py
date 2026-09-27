"""Gymnasium client for SandboxAI's exported Godot TCP bridge.

The protocol is intentionally tiny (newline-delimited JSON over localhost) so
an exported game remains self-contained and no editor plugin is required.
Only the local machine is accepted by the Godot server.  RGB observations are
PNG-compressed in transit and all privileged state remains in ``info``.
"""

from __future__ import annotations

import base64
import binascii
import collections
import io
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from sandbox.actions import (
    ACTION_CONTRACT_VERSION,
    ACTION_DIMS,
    SandboxAction,
    coerce_sandbox_action,
)


class GodotBridgeError(RuntimeError):
    """Raised when the exported Godot process or bridge is unavailable."""


class GodotSandboxEnv(gym.Env[np.ndarray, np.ndarray]):
    """Control one exported Godot tactical arena through its local bridge."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    PROTOCOL_VERSION = 1
    MAX_LINE_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        env_path: Union[str, Path],
        width: int = 84,
        height: int = 84,
        seed: int = 42,
        max_steps: int = 500,
        port: Optional[int] = None,
        startup_timeout: float = 20.0,
        headless: bool = True,
    ) -> None:
        super().__init__()
        self.env_path = Path(env_path).resolve()
        if not self.env_path.is_file():
            raise FileNotFoundError(f"Godot export not found: {self.env_path}")
        if width < 1 or height < 1:
            raise ValueError("Godot observation dimensions must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.width, self.height = int(width), int(height)
        self.max_steps = int(max_steps)
        self._seed = int(seed)
        self.observation_space = spaces.Box(
            0, 255, shape=(3, self.height, self.width), dtype=np.uint8
        )
        self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_DIMS, dtype=np.int64))
        self.action_space.seed(seed)
        self._socket: Optional[socket.socket] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._recv_buffer = b""
        self._request_id = 0
        self._last_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)
        self._output_lines: collections.deque[str] = collections.deque(maxlen=100)
        self._output_thread: Optional[threading.Thread] = None
        self._port = int(port or self._find_free_port())
        if not 1 <= self._port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self._port}")
        self._start(startup_timeout=startup_timeout, headless=headless)

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _drain_process_output(self) -> None:
        """Prevent a verbose Godot export from blocking on a full stdout pipe."""
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for raw_line in process.stdout:
                self._output_lines.append(raw_line.decode(errors="replace").rstrip())
        except (OSError, ValueError):
            return

    def _process_output_tail(self) -> str:
        return "\n".join(self._output_lines)[-4000:]

    def _start(self, startup_timeout: float, headless: bool) -> None:
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        args = [str(self.env_path)]
        if headless:
            args.append("--headless")
        args.extend(
            [
                "--",
                "--rl-server",
                f"--rl-port={self._port}",
                f"--obs-width={self.width}",
                f"--obs-height={self.height}",
                f"--max-steps={self.max_steps}",
            ]
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self._output_thread = threading.Thread(
            target=self._drain_process_output, name="sandboxai-godot-output", daemon=True
        )
        self._output_thread.start()

        deadline = time.monotonic() + startup_timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise GodotBridgeError(
                    f"Godot sandbox exited during startup ({self._process.returncode}): "
                    f"{self._process_output_tail()}"
                )
            try:
                sock = socket.create_connection(("127.0.0.1", self._port), timeout=0.4)
                sock.settimeout(10.0)
                self._socket = sock
                hello = self._request({"command": "hello"})
                if hello.get("protocol") != self.PROTOCOL_VERSION:
                    raise GodotBridgeError(f"Unsupported Godot bridge response: {hello}")
                if hello.get("action_contract") != ACTION_CONTRACT_VERSION:
                    raise GodotBridgeError(
                        "Godot action contract mismatch: "
                        f"expected {ACTION_CONTRACT_VERSION}, got {hello.get('action_contract')}"
                    )
                if hello.get("action_dims") != list(ACTION_DIMS):
                    raise GodotBridgeError(
                        f"Godot action dimensions mismatch: expected {list(ACTION_DIMS)}, "
                        f"got {hello.get('action_dims')}"
                    )
                return
            except (OSError, GodotBridgeError) as exc:
                last_error = exc
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None
                self._recv_buffer = b""
                time.sleep(0.1)
        self.close()
        raise GodotBridgeError(
            f"Timed out connecting to Godot sandbox on local port {self._port}: {last_error}\n"
            f"Process output:\n{self._process_output_tail()}"
        )

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._socket is None:
            raise GodotBridgeError("Godot bridge is closed")
        if not isinstance(payload, dict) or not payload.get("command"):
            raise ValueError("bridge payload must contain a command")
        self._request_id += 1
        request = dict(payload)
        request["request_id"] = self._request_id
        wire = (json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
        if len(wire) > self.MAX_LINE_BYTES:
            raise GodotBridgeError("Godot bridge request exceeds the maximum line size")
        try:
            self._socket.sendall(wire)
            while b"\n" not in self._recv_buffer:
                chunk = self._socket.recv(65536)
                if not chunk:
                    raise GodotBridgeError("Godot bridge disconnected")
                self._recv_buffer += chunk
                if len(self._recv_buffer) > self.MAX_LINE_BYTES:
                    raise GodotBridgeError("Godot bridge response exceeds the maximum line size")
            line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            response = json.loads(line.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GodotBridgeError(f"Godot bridge request failed: {exc}") from exc
        if not isinstance(response, dict):
            raise GodotBridgeError("Godot bridge response must be a JSON object")
        if response.get("request_id") != self._request_id:
            raise GodotBridgeError(
                f"Godot bridge response mismatch: expected {self._request_id}, "
                f"got {response.get('request_id')}"
            )
        if not response.get("ok", False):
            raise GodotBridgeError(str(response.get("error", "unknown Godot bridge error")))
        return response

    def _decode_observation(self, response: Dict[str, Any]) -> np.ndarray:
        encoded = response.get("observation_png")
        if not isinstance(encoded, str) or not encoded:
            raise GodotBridgeError("Godot response did not contain observation_png")
        try:
            png = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(png)) as image:
                image = image.convert("RGB")
                if image.size != (self.width, self.height):
                    image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
                obs = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1)).copy()
        except (binascii.Error, OSError, ValueError) as exc:
            raise GodotBridgeError(f"Invalid observation from Godot: {exc}") from exc
        self._last_observation = obs
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.action_space.seed(seed)
        request: Dict[str, Any] = {"command": "reset", "seed": self._seed}
        if options:
            if "map" in options:
                request["map"] = str(options["map"])
            if "scenario" in options:
                request["scenario"] = str(options["scenario"])
        response = self._request(request)
        return self._decode_observation(response), dict(response.get("info", {}))

    def step(
        self, action: Union[np.ndarray, Sequence[int], Dict[str, Any], SandboxAction]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        command = coerce_sandbox_action(action)
        response = self._request(
            {"command": "step", "action": command.to_array().astype(int).tolist()}
        )
        return (
            self._decode_observation(response),
            float(response.get("reward", 0.0)),
            bool(response.get("terminated", False)),
            bool(response.get("truncated", False)),
            dict(response.get("info", {})),
        )

    def replay(self) -> list[Dict[str, Any]]:
        """Return bounded deterministic episode events from the Godot controller."""
        response = self._request({"command": "replay"})
        events = response.get("events", [])
        return list(events) if isinstance(events, list) else []

    def save_replay(self, path: str = "user://replays") -> str:
        """Persist the current episode replay inside the Godot process."""
        response = self._request({"command": "save_replay", "path": path})
        return str(response.get("path", ""))

    def render(self) -> np.ndarray:
        return np.transpose(self._last_observation, (1, 2, 0))

    def close(self) -> None:
        if self._socket is not None:
            try:
                previous_timeout = self._socket.gettimeout()
                self._socket.settimeout(0.5)
                self._request({"command": "close"})
                self._socket.settimeout(previous_timeout)
            except Exception:
                pass
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
