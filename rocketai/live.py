"""Live matches: watch the AI play in real time, including what it is "thinking".

A background thread runs a RocketSim match paced to real time (or faster),
and keeps recent frames in a ring buffer. The browser polls
``/api/live?since=<seq>`` — plain HTTP, so it works through any proxy.

A player spec ``run:<name>`` follows a training run: whenever the run saves
a newer ``latest.pt``, the live match switches to it, so you can watch the
AI improve while it trains.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import RUN_NAME_PATTERN, TICK_SKIP, TICKS_PER_SECOND, run_paths
from .env import make_env
from .match import frame_of
from .model import load_checkpoint, model_from_checkpoint
from .opponents import SCRIPTED_BOTS, PolicyPlayer, make_player

STEP_SECONDS = TICK_SKIP / TICKS_PER_SECOND
BUFFER = 15 * 30  # 30 s of frames
IDLE_TIMEOUT = 45.0  # stop when nobody has watched for this long
FOLLOW_CHECK_SECONDS = 5.0


class SpecError(ValueError):
    pass


def resolve_spec(spec: str) -> tuple[str, Path | None]:
    """Return (kind, path): kind is 'scripted', 'checkpoint' or 'follow'."""
    if spec in SCRIPTED_BOTS:
        return "scripted", None
    if spec.startswith("run:"):
        run = spec[4:]
        if not RUN_NAME_PATTERN.match(run):
            raise SpecError(f"Ungültiger Run {run!r}")
        return "follow", run_paths(run).checkpoints / "latest.pt"
    run, _, file = spec.partition("/")
    if not RUN_NAME_PATTERN.match(run) or not file.endswith(".pt") or "/" in file or "\\" in file:
        raise SpecError(f"Ungültiger Spieler {spec!r}")
    path = run_paths(run).checkpoints / file
    if not path.is_file():
        raise SpecError(f"Checkpoint {spec} nicht gefunden")
    return "checkpoint", path


@dataclass
class Side:
    spec: str
    kind: str
    path: Path | None
    player: Any = None
    label: str = ""
    steps: int | None = None
    mtime: float = 0.0
    last_check: float = 0.0

    def load(self) -> None:
        if self.kind == "scripted":
            self.player = make_player(self.spec)
            self.label = SCRIPTED_BOTS[self.spec][0]
            return
        assert self.path is not None
        if not self.path.is_file():
            raise SpecError(f"{self.path.parent.parent.name} hat noch keinen Checkpoint")
        payload = load_checkpoint(self.path)
        player = PolicyPlayer(model_from_checkpoint(payload))
        player.explain = True
        self.player = player
        self.steps = int(payload["steps"])
        self.mtime = self.path.stat().st_mtime
        run = self.path.parent.parent.name
        self.label = f"{run} · {self.steps:,}".replace(",", ".")
        player.name = self.label

    def maybe_reload(self, now: float) -> bool:
        """For 'follow' sides: switch to a newer checkpoint. True if reloaded."""
        if self.kind != "follow" or now - self.last_check < FOLLOW_CHECK_SECONDS:
            return False
        self.last_check = now
        try:
            mtime = self.path.stat().st_mtime if self.path else 0.0
        except OSError:
            return False
        if mtime <= self.mtime:
            return False
        try:
            self.load()
        except Exception:  # a checkpoint being written; try again later
            return False
        return True

    def info(self) -> dict[str, Any]:
        return {"spec": self.spec, "kind": self.kind, "label": self.label, "steps": self.steps}


@dataclass
class LiveSettings:
    blue: str
    orange: str = "chaser"
    team_size: int = 1
    speed: float = 1.0
    match_seconds: float = 300.0


@dataclass
class LiveSession:
    settings: LiveSettings
    blue: Side
    orange: Side
    frames: deque = field(default_factory=lambda: deque(maxlen=BUFFER))
    events: deque = field(default_factory=lambda: deque(maxlen=40))
    seq: int = 0
    score: list[int] = field(default_factory=lambda: [0, 0])
    match_index: int = 1
    match_clock: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    paused: bool = False
    running: bool = True
    error: str = ""
    last_poll: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def event(self, text: str, kind: str = "info") -> None:
        self.events.append({"seq": self.seq, "t": time.time(), "text": text, "kind": kind})


class LiveManager:
    def __init__(self) -> None:
        self.session: LiveSession | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    # -- control ---------------------------------------------------------
    def start(self, settings: LiveSettings) -> dict[str, Any]:
        if settings.team_size not in (1, 2, 3):
            raise SpecError("Teamgröße muss 1, 2 oder 3 sein")
        settings.speed = float(min(8.0, max(0.25, settings.speed)))
        settings.match_seconds = float(min(1800.0, max(30.0, settings.match_seconds)))
        blue = Side(settings.blue, *resolve_spec(settings.blue))
        orange = Side(settings.orange, *resolve_spec(settings.orange))
        blue.load()
        orange.load()
        self.stop()
        session = LiveSession(settings, blue, orange)
        session.event(f"{blue.label} gegen {orange.label}")
        with self.lock:
            self.session = session
            self.thread = threading.Thread(
                target=self._run, args=(session,), daemon=True, name="rocketai-live"
            )
            self.thread.start()
        return self.state()

    def stop(self) -> None:
        with self.lock:
            session, thread = self.session, self.thread
            if session is not None:
                session.running = False
        if thread is not None:
            thread.join(timeout=3)

    def control(self, *, paused: bool | None = None, speed: float | None = None) -> dict[str, Any]:
        session = self.session
        if session is None:
            raise SpecError("Kein Live-Spiel aktiv")
        if paused is not None:
            session.paused = paused
        if speed is not None:
            session.settings.speed = float(min(8.0, max(0.25, speed)))
        return self.state()

    # -- reading ---------------------------------------------------------
    def state(self, since: int = -1) -> dict[str, Any]:
        session = self.session
        if session is None:
            return {"active": False}
        session.last_poll = time.time()
        with session.lock:
            frames = [f for f in session.frames if f["seq"] > since]
            if since < 0:
                frames = frames[-30:]  # a fresh viewer starts near "now"
            events = (
                [e for e in session.events if e["seq"] > since]
                if since >= 0
                else list(session.events)
            )
            return {
                "active": session.running,
                "error": session.error,
                "paused": session.paused,
                "speed": session.settings.speed,
                "team_size": session.settings.team_size,
                "match_seconds": session.settings.match_seconds,
                "seq": session.seq,
                "score": list(session.score),
                "match": session.match_index,
                "clock": round(session.match_clock, 2),
                "blue": session.blue.info(),
                "orange": session.orange.info(),
                "history": session.history[-10:],
                "frames": frames,
                "events": events,
            }

    # -- loop ------------------------------------------------------------
    def _run(self, session: LiveSession) -> None:
        settings = session.settings
        try:
            env = make_env(
                team_size=settings.team_size,
                episode_seconds=10_000,
                no_touch_seconds=10_000,
                kickoff_probability=1.0,
            )
            obs = env.reset()
            next_tick = time.perf_counter()
            while session.running:
                now = time.time()
                if now - session.last_poll > IDLE_TIMEOUT:
                    session.event("Pausiert: niemand schaut zu", "muted")
                    break
                for side in (session.blue, session.orange):
                    if side.maybe_reload(now):
                        session.event(f"Neuer Trainingsstand geladen: {side.label}", "accent")
                if session.paused:
                    time.sleep(0.1)
                    next_tick = time.perf_counter()
                    continue

                state = env.state
                blue_agents = [a for a in env.agents if state.cars[a].team_num == 0]
                orange_agents = [a for a in env.agents if state.cars[a].team_num == 1]
                actions = {
                    **session.blue.player.act(blue_agents, obs, state),
                    **session.orange.player.act(orange_agents, obs, state),
                }
                obs, _, terminated, _ = env.step({a: np.array([i]) for a, i in actions.items()})
                state = env.state
                session.match_clock += STEP_SECONDS
                frame = frame_of(session.match_clock, state)
                brain = self._brain(session, env.agents)

                goal = None
                if any(terminated.values()):
                    scorer = state.scoring_team
                    if scorer in (0, 1):
                        session.score[scorer] += 1
                        goal = scorer
                        side = session.blue if scorer == 0 else session.orange
                        session.event(
                            f"Tor für {'Blau' if scorer == 0 else 'Orange'} ({side.label}) – "
                            f"{session.score[0]}:{session.score[1]}",
                            "blue" if scorer == 0 else "orange",
                        )
                with session.lock:
                    session.seq += 1
                    session.frames.append(
                        {"seq": session.seq, "f": frame, "brain": brain, "goal": goal}
                    )
                if goal is not None:
                    obs = env.reset()
                if session.match_clock >= settings.match_seconds:
                    self._finish_match(session)
                    obs = env.reset()

                # Pace to real time × speed; never try to "catch up" a long stall.
                next_tick += STEP_SECONDS / session.settings.speed
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:
                    next_tick = time.perf_counter()
            env.close()
        except Exception as error:  # surfaced in the UI
            session.error = f"{type(error).__name__}: {error}"
            session.event(f"Fehler: {session.error}", "error")
        finally:
            session.running = False

    @staticmethod
    def _brain(session: LiveSession, agents: list[str]) -> dict[str, Any] | None:
        """What the first AI-controlled car decided (blue preferred)."""
        for index, side in enumerate((session.blue, session.orange)):
            info = getattr(side.player, "last_info", None)
            if info:
                agent = sorted(info)[0]
                car_index = agents.index(agent) if agent in agents else 0
                return {"team": index, "car": car_index, **info[agent]}
        return None

    @staticmethod
    def _finish_match(session: LiveSession) -> None:
        blue, orange = session.score
        winner = "Unentschieden" if blue == orange else ("Blau" if blue > orange else "Orange")
        session.history.append(
            {"match": session.match_index, "blue": blue, "orange": orange, "winner": winner}
        )
        session.event(f"Spiel {session.match_index} vorbei: {blue}:{orange} – {winner}", "accent")
        session.match_index += 1
        session.score = [0, 0]
        session.match_clock = 0.0


LIVE = LiveManager()
