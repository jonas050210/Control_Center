"""3D checkpoint viewer: watch a trained policy play in a rendered Godot window.

    python start.py view                              # newest run, best checkpoint
    python start.py view --checkpoint training/runs/<run>  --map compound
    python start.py view --checkpoint path/to/best.pt # behavior-cloning checkpoint
    python start.py view --policy scripted            # no checkpoint needed

How it works: this process loads the checkpoint (CPU inference - the policy
is a tiny MLP), opens a TCP server on ``127.0.0.1`` and launches Godot with
``scripts/viewer/viewer_entry.gd``. Every simulation tick the viewer sends
the agent's 126-value observation and receives one MultiDiscrete action -
the same contract, the same observation code and the same action decoder as
headless training, so what you see is what the policy actually does.
Protocol details: ``docs/CHECKPOINT_VIEWER.md``.

Nothing here is a training path: the viewer never writes checkpoints, and
torch / stable-baselines3 are imported only when such a checkpoint is
actually loaded (``--policy scripted`` needs neither).
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

VIEWER_ENTRY_SCRIPT = "res://scripts/viewer/viewer_entry.gd"
DEFAULT_LEVEL = 10  # CurriculumConfig.Level.MIXED_RANDOMIZED
MAX_VIEWER_LEVEL = 10  # level 11 is the two-agent self-play environment
ACCEPT_TIMEOUT_SECONDS = 90.0
POLICY_KINDS = ("ppo", "bc", "scripted")

#: Checkpoint preference inside a PPO run directory (best first).
_PPO_RUN_ORDER = (
    "checkpoints/best_eval.zip",
    "checkpoints/best.zip",
    "final.zip",
    "checkpoints/latest.zip",
)
#: Checkpoint preference inside a behavior-cloning output directory.
_BC_RUN_ORDER = ("best.pt", "latest.pt")


class ViewerError(RuntimeError):
    """A user-facing problem (bad checkpoint, Godot missing, ...)."""


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPolicy:
    kind: str  # "ppo" | "bc" | "scripted"
    path: Path | None
    label: str
    run_dir: Path | None = None


def _newest_periodic(checkpoints: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for candidate in checkpoints.glob("ppo_*_steps.zip"):
        digits = candidate.stem.removeprefix("ppo_").removesuffix("_steps")
        if digits.isdigit() and (best is None or int(digits) > best[0]):
            best = (int(digits), candidate)
    return best[1] if best else None


def checkpoint_in_directory(directory: Path) -> Path | None:
    """The preferred checkpoint inside a PPO run or BC output directory."""
    for relative in _PPO_RUN_ORDER:
        if (directory / relative).is_file():
            return directory / relative
    periodic = _newest_periodic(directory / "checkpoints")
    if periodic is not None:
        return periodic
    for relative in _BC_RUN_ORDER:
        for base in (directory, directory / "checkpoints"):
            if (base / relative).is_file():
                return base / relative
    return None


def newest_checkpoint(output_root: Path) -> Path | None:
    """The preferred checkpoint of the most recently modified run."""
    candidates: list[tuple[float, Path]] = []
    for group in ("runs", "bc_runs"):
        base = output_root / group
        if not base.is_dir():
            continue
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            checkpoint = checkpoint_in_directory(run_dir)
            if checkpoint is not None:
                candidates.append((checkpoint.stat().st_mtime, checkpoint))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _run_dir_of(checkpoint: Path) -> Path:
    return (
        checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    )


def resolve_policy(checkpoint: str | None, policy: str | None, output_root: Path) -> ResolvedPolicy:
    """Turns the CLI's ``--checkpoint`` / ``--policy`` into something loadable."""
    if policy == "scripted":
        return ResolvedPolicy("scripted", None, "scripted baseline")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            found = checkpoint_in_directory(path)
            if found is None:
                raise ViewerError(f"no checkpoint (.zip or .pt) found in {path}")
            path = found
        elif not path.is_file():
            raise ViewerError(f"checkpoint does not exist: {path}")
    else:
        found = newest_checkpoint(output_root)
        if found is None:
            raise ViewerError(
                f"no trained checkpoint found under {output_root}. Train one first "
                "(python start.py train ...), pass --checkpoint PATH, or use "
                "--policy scripted to watch the scripted baseline."
            )
        path = found
    suffix = path.suffix.lower()
    if suffix == ".zip":
        kind = "ppo"
    elif suffix in (".pt", ".pth"):
        kind = "bc"
    else:
        raise ViewerError(f"unsupported checkpoint type {suffix!r} (expected .zip or .pt)")
    if policy and policy != kind:
        raise ViewerError(f"--policy {policy} does not match the checkpoint type ({kind})")
    run_dir = _run_dir_of(path)
    return ResolvedPolicy(kind, path, f"{run_dir.name}/{path.name}", run_dir)


def infer_run_settings(run_dir: Path | None) -> dict[str, Any]:
    """Curriculum level and enemy count the run was last training on.

    The automatic curriculum's saved state wins (it is where training actually
    ended up); the run's config.json is the fallback. Missing or unreadable
    files simply yield fewer keys - the viewer then uses its defaults.
    """
    settings: dict[str, Any] = {}
    if run_dir is None:
        return settings
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
        if isinstance(config, dict):
            if config.get("curriculum_level"):
                settings["level"] = int(config["curriculum_level"])
            if config.get("enemy_count"):
                settings["enemies"] = int(config["enemy_count"])
    except (OSError, ValueError, TypeError):
        pass
    try:
        state = json.loads(
            (run_dir / "checkpoints" / "curriculum_state.json").read_text(encoding="utf-8-sig")
        )
        level = state.get("driver", {}).get("auto", {}).get("level")
        if level:
            settings["level"] = int(level)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    if "level" in settings:
        settings["level"] = max(1, min(MAX_VIEWER_LEVEL, settings["level"]))
    return settings


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


Actor = Callable[[Sequence[float]], list[int]]


def _as_action_list(values: Any) -> list[int]:
    flat = values.reshape(-1).tolist() if hasattr(values, "reshape") else list(values)
    if len(flat) != len(ACTION_NVEC):
        raise ViewerError(
            f"the policy returned {len(flat)} action values, expected {len(ACTION_NVEC)}"
        )
    return [max(0, min(int(n) - 1, int(v))) for v, n in zip(flat, ACTION_NVEC, strict=True)]


def load_actor(resolved: ResolvedPolicy, *, deterministic: bool = True, seed: int = 0) -> Actor:
    """A callable observation -> MultiDiscrete action list for ``resolved``."""
    if resolved.kind == "scripted":
        from .policies import ScriptedBaseline

        baseline = ScriptedBaseline(seed=seed)
        return lambda obs: _as_action_list(baseline.predict(list(obs))[0])

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a core dependency
        raise ViewerError("numpy is required to run a checkpoint") from exc

    if resolved.kind == "ppo":
        try:
            from stable_baselines3 import PPO  # type: ignore
        except ImportError as exc:
            raise ViewerError(
                "stable-baselines3/torch are not installed; run `python install.py`"
            ) from exc
        model = PPO.load(str(resolved.path), device="cpu")
        shape = tuple(getattr(model.observation_space, "shape", ()) or ())
        if shape != (OBSERVATION_FIELD_COUNT,):
            raise ViewerError(
                f"{resolved.path} expects observations of shape {shape}, "
                f"the simulator produces ({OBSERVATION_FIELD_COUNT},)"
            )

        def act_ppo(obs: Sequence[float]) -> list[int]:
            action, _ = model.predict(
                np.asarray(obs, dtype=np.float32), deterministic=deterministic
            )
            return _as_action_list(action)

        return act_ppo

    try:
        import torch  # type: ignore

        from .bc import load_bc_checkpoint
    except ImportError as exc:
        raise ViewerError("torch is not installed; run `python install.py`") from exc
    bc_model = load_bc_checkpoint(str(resolved.path), device="cpu")
    if int(bc_model.observation_dim) != OBSERVATION_FIELD_COUNT:
        raise ViewerError(
            f"{resolved.path} expects {bc_model.observation_dim} observation values, "
            f"the simulator produces {OBSERVATION_FIELD_COUNT}"
        )

    def act_bc(obs: Sequence[float]) -> list[int]:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            return _as_action_list(bc_model.predict(tensor, deterministic=deterministic).numpy())

    return act_bc


# ---------------------------------------------------------------------------
# Wire protocol (one JSON object per line, strict request/response)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLog:
    summaries: list[dict[str, Any]] = field(default_factory=list)
    decisions: int = 0

    def add(self, summary: dict[str, Any]) -> str:
        self.summaries.append(summary)
        return (
            f"episode {summary.get('episode', len(self.summaries))}: "
            f"{str(summary.get('outcome', '?')).upper():<7} "
            f"reward {float(summary.get('reward', 0.0)):8.2f}  "
            f"kills {summary.get('kills', 0)}/{summary.get('enemy_count', 0)}  "
            f"{float(summary.get('survival_time', 0.0)):5.1f} s"
            + (f"  [{summary['map_id']}]" if summary.get("map_id") else "")
        )

    def totals(self) -> dict[str, Any]:
        count = len(self.summaries)
        wins = sum(1 for item in self.summaries if item.get("outcome") == "win")
        mean_reward = (
            sum(float(item.get("reward", 0.0)) for item in self.summaries) / count if count else 0.0
        )
        return {"episodes": count, "wins": wins, "mean_reward": mean_reward}


def handle_message(
    message: dict[str, Any], actor: Actor, label: str, log: EpisodeLog
) -> tuple[dict[str, Any] | None, bool]:
    """Answers one viewer message. Returns ``(reply or None, keep_running)``."""
    kind = message.get("type")
    if kind == "act":
        obs = message.get("obs")
        if not isinstance(obs, list) or len(obs) != OBSERVATION_FIELD_COUNT:
            got = len(obs) if isinstance(obs, list) else "no"
            return {
                "error": f"expected {OBSERVATION_FIELD_COUNT} observation values, got {got}"
            }, True
        log.decisions += 1
        return {"action": actor(obs)}, True
    if kind == "hello":
        size: Any = message.get("observation_size")
        nvec = list(message.get("action_nvec") or [])
        if size != OBSERVATION_FIELD_COUNT or nvec != list(ACTION_NVEC):
            return {
                "ok": False,
                "error": (
                    f"contract mismatch: viewer has {size} observations / actions {nvec}, "
                    f"this checkout expects {OBSERVATION_FIELD_COUNT} / {list(ACTION_NVEC)}"
                ),
            }, False
        return {"ok": True, "policy": label}, True
    if kind == "episode":
        summary = message.get("summary") or {}
        print("  " + log.add(summary if isinstance(summary, dict) else {}), flush=True)
        return {"ok": True}, True
    if kind == "bye":
        return None, False
    return {"error": f"unknown message type {kind!r}"}, True


def serve_connection(conn: socket.socket, actor: Actor, label: str, log: EpisodeLog) -> None:
    """Serves one viewer connection until it says bye or disconnects."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with conn, conn.makefile("rb") as reader, conn.makefile("wb") as writer:
        for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                reply: dict[str, Any] | None = {"error": "invalid JSON"}
                keep_running = True
            else:
                reply, keep_running = handle_message(
                    message if isinstance(message, dict) else {}, actor, label, log
                )
            if reply is not None:
                writer.write(json.dumps(reply).encode("utf-8") + b"\n")
                writer.flush()
            if not keep_running:
                return


# ---------------------------------------------------------------------------
# Godot launch
# ---------------------------------------------------------------------------


def _gui_variant(executable: str) -> str:
    """On Windows prefer Godot.exe over Godot_console.exe for a window."""
    path = Path(executable)
    if path.name.lower().endswith("_console.exe"):
        sibling = path.with_name(path.name[: -len("_console.exe")] + ".exe")
        if sibling.is_file():
            return str(sibling)
    return executable


def build_viewer_command(
    godot_executable: str,
    project: Path,
    port: int,
    settings: dict[str, Any],
    *,
    headless: bool = False,
) -> list[str]:
    from .wsl import WindowsInterop

    executable = godot_executable if headless else _gui_variant(godot_executable)
    interop = WindowsInterop(executable)
    command = [executable]
    if headless:
        command.append("--headless")
    command += ["--path", interop.windows_path(project), "--script", VIEWER_ENTRY_SCRIPT, "--"]
    command += ["--policy-host", "127.0.0.1", "--policy-port", str(port)]
    for key in (
        "map",
        "level",
        "enemies",
        "seed",
        "scenario",
        "lighting",
        "weapon",
        "speed",
        "camera",
        "max-steps",
        "max-episodes",
    ):
        value = settings.get(key)
        if value not in (None, ""):
            command += [f"--{key}", str(value)]
    return command


@dataclass
class ViewerOptions:
    checkpoint: str | None = None
    policy: str | None = None
    map: str = ""
    level: int | None = None
    enemies: int | None = None
    seed: int | None = None
    scenario: str = ""
    lighting: str = ""
    weapon: str = ""
    speed: float = 1.0
    camera: str = "chase"
    stochastic: bool = False
    headless: bool = False
    max_steps: int = 0
    max_episodes: int = 0
    godot_executable: str | None = None
    project_path: str = ""
    output_root: str = "training"


def validate_ids(options: ViewerOptions) -> None:
    """Rejects unknown map/lighting ids before Godot is even started."""
    from .conditions import LIGHTING_IDS, MAP_IDS

    if options.map and options.map not in MAP_IDS:
        raise ViewerError(f"unknown map {options.map!r}; known maps: {', '.join(MAP_IDS)}")
    if options.lighting and options.lighting not in LIGHTING_IDS:
        raise ViewerError(
            f"unknown lighting {options.lighting!r}; known: {', '.join(LIGHTING_IDS)}"
        )


def run_viewer(options: ViewerOptions) -> int:
    """Loads the policy, launches the Godot viewer and serves it until it closes."""
    from .config import find_godot_executable

    project = Path(options.project_path or Path(__file__).resolve().parents[2]).resolve()
    if not (project / "project.godot").is_file():
        raise ViewerError(f"{project} is not a SandboxAI checkout (no project.godot)")
    output_root = Path(options.output_root)
    if not output_root.is_absolute():
        output_root = project / output_root
    validate_ids(options)

    resolved = resolve_policy(options.checkpoint, options.policy, output_root)
    settings: dict[str, Any] = infer_run_settings(resolved.run_dir)
    settings.setdefault("level", DEFAULT_LEVEL)
    for key, value in (
        ("level", options.level),
        ("enemies", options.enemies),
        ("seed", options.seed),
    ):
        if value is not None:
            settings[key] = value
    settings["level"] = max(1, min(MAX_VIEWER_LEVEL, int(settings["level"])))
    settings.update(
        {
            "map": options.map,
            "scenario": options.scenario,
            "lighting": options.lighting,
            "weapon": options.weapon,
            "speed": options.speed,
            "camera": options.camera,
            "max-steps": options.max_steps or None,
            "max-episodes": options.max_episodes or None,
        }
    )

    print(f"Loading {resolved.kind.upper()} policy: {resolved.path or resolved.label}", flush=True)
    actor = load_actor(resolved, deterministic=not options.stochastic, seed=options.seed or 0)
    godot = find_godot_executable(options.godot_executable or "godot")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        command = build_viewer_command(godot, project, port, settings, headless=options.headless)
        print("Launching viewer:", " ".join(command), flush=True)
        try:
            process = subprocess.Popen(command, cwd=str(project))
        except OSError as exc:
            raise ViewerError(
                f"could not start Godot ({godot!r}): {exc}. Run `python install.py` "
                "or pass --godot-executable."
            ) from exc
        conn = _accept_while_alive(server, process)
        if conn is None:
            code = process.wait()
            raise ViewerError(f"Godot exited (code {code}) before connecting to the policy")
        log = EpisodeLog()
        print(
            f"Viewer connected - level {settings['level']}, "
            f"{'stochastic' if options.stochastic else 'deterministic'} actions. "
            "Close the window or press Esc to stop.",
            flush=True,
        )
        # ConnectionError/OSError: the window was closed mid-request.
        with contextlib.suppress(ConnectionError, OSError):
            serve_connection(conn, actor, resolved.label, log)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            code = process.wait()
    totals = log.totals()
    if totals["episodes"]:
        print(
            f"Watched {totals['episodes']} episode(s): {totals['wins']} won, "
            f"mean reward {totals['mean_reward']:.2f}"
        )
    print(f"Answered {log.decisions} policy decisions.", flush=True)
    return int(code or 0)


def _accept_while_alive(server: socket.socket, process: subprocess.Popen) -> socket.socket | None:
    """Waits for the viewer to connect, giving up if Godot exits first."""
    server.settimeout(0.5)
    deadline = time.monotonic() + ACCEPT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            conn, _ = server.accept()
        except TimeoutError:
            if process.poll() is not None:
                return None
            continue
        conn.settimeout(None)
        return conn
    process.terminate()
    return None


def main(options: ViewerOptions) -> int:
    """CLI entry: user-facing errors become one clean line and exit code 1."""
    try:
        return run_viewer(options)
    except ViewerError as exc:
        print(f"viewer: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


__all__ = [
    "POLICY_KINDS",
    "ResolvedPolicy",
    "ViewerError",
    "ViewerOptions",
    "build_viewer_command",
    "checkpoint_in_directory",
    "handle_message",
    "infer_run_settings",
    "load_actor",
    "main",
    "newest_checkpoint",
    "resolve_policy",
    "run_viewer",
    "serve_connection",
]
