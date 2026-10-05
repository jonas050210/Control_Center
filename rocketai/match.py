"""Simulated matches between two players: evaluation results and replays for the arena view."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import OBS_BASE_SIZE, TICK_SKIP, TICKS_PER_SECOND
from .env import make_env
from .opponents import Player

#: Replay frames are recorded every decision (15 per second).
REPLAY_FORMAT = "rocketai.replay.v1"  # v1 readers ignore the extra frame fields


@dataclass
class MatchResult:
    blue: str
    orange: str
    team_size: int
    seconds: float
    goals_blue: int = 0
    goals_orange: int = 0
    touches_blue: int = 0
    touches_orange: int = 0
    frames: list[list[Any]] = field(default_factory=list, repr=False)
    goal_times: list[tuple[float, int]] = field(default_factory=list)

    @property
    def winner(self) -> str:
        if self.goals_blue > self.goals_orange:
            return "blue"
        if self.goals_orange > self.goals_blue:
            return "orange"
        return "draw"

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("frames")
        data["winner"] = self.winner
        return data


def frame_of(t: float, state: Any) -> list[Any]:
    """Compact frame: ``[t, [bx, by, bz], cars, pads]``.

    Each car is ``[team, x, y, z, yaw, boost, demoed, pitch, roll]``; ``pads``
    is a bitmask of active boost pads in RLGym's BOOST_LOCATIONS order.
    """
    ball = state.ball.position
    cars = []
    for car in state.cars.values():
        pos = car.physics.position
        pitch, yaw, roll = (float(v) for v in car.physics.euler_angles)
        cars.append(
            [
                int(car.team_num),
                round(float(pos[0])),
                round(float(pos[1])),
                round(float(pos[2])),
                round(yaw, 3),
                round(float(car.boost_amount)),
                1 if car.is_demoed else 0,
                round(pitch, 3),
                round(roll, 3),
            ]
        )
    pads = 0
    for index, timer in enumerate(state.boost_pad_timers):
        if timer <= 0:
            pads |= 1 << index
    return [round(t, 3), [round(float(v)) for v in ball], cars, pads]


# Backwards-compatible name.
_frame = frame_of


def needs_extras(players: list[Player]) -> bool:
    """Braucht einer der Spieler die erweiterte Beobachtung?

    Ein Checkpoint weiß selbst, womit er trainiert wurde (``obs_size``), deshalb
    kann ein Match genau die Beobachtung bauen, die alle Spieler verstehen —
    auch alte Checkpoints mit 172 Werten laufen weiter.
    """
    for player in players:
        model = getattr(player, "model", None)
        size = getattr(model, "obs_size", None)
        if size is not None:
            return int(size) > OBS_BASE_SIZE
    return True


def play_match(
    blue: Player,
    orange: Player,
    *,
    team_size: int = 1,
    seconds: float = 120.0,
    record: bool = False,
    seed: int | None = None,
    obs_extras: bool | None = None,
) -> MatchResult:
    """A timed match: kickoff after every goal, like the real game (no overtime)."""
    if obs_extras is None:
        obs_extras = needs_extras([blue, orange])
    env = make_env(
        team_size=team_size,
        episode_seconds=seconds + 1,
        no_touch_seconds=seconds + 1,
        kickoff_probability=1.0,
        seed=seed,
        obs_extras=obs_extras,
    )
    result = MatchResult(blue.name, orange.name, team_size, seconds)
    obs = env.reset()
    elapsed = 0.0
    step_seconds = TICK_SKIP / TICKS_PER_SECOND
    if record:
        result.frames.append(_frame(0.0, env.state))
    for step in range(1, round(seconds / step_seconds) + 1):
        state = env.state
        blue_agents = [a for a in env.agents if state.cars[a].team_num == 0]
        orange_agents = [a for a in env.agents if state.cars[a].team_num == 1]
        actions = {**blue.act(blue_agents, obs, state), **orange.act(orange_agents, obs, state)}
        obs, _, terminated, truncated = env.step({a: np.array([i]) for a, i in actions.items()})
        elapsed = step * step_seconds  # integer steps: no floating-point drift
        state = env.state
        for car in state.cars.values():
            if car.team_num == 0:
                result.touches_blue += car.ball_touches
            else:
                result.touches_orange += car.ball_touches
        if record:
            result.frames.append(_frame(elapsed, state))
        if any(terminated.values()):
            scorer = state.scoring_team
            if scorer == 0:
                result.goals_blue += 1
            elif scorer == 1:
                result.goals_orange += 1
            result.goal_times.append((round(elapsed, 2), int(scorer if scorer is not None else -1)))
            obs = env.reset()
        elif any(truncated.values()):
            obs = env.reset()
    env.close()
    return result


def evaluate(
    policy: Player,
    opponent: Player,
    *,
    games: int = 6,
    team_size: int = 1,
    seconds: float = 120.0,
    seed: int = 0,
    obs_extras: bool | None = None,
) -> dict[str, Any]:
    """``games`` matches, switching sides each game; results from the policy's point of view."""
    wins = draws = losses = goals_for = goals_against = 0
    started = time.perf_counter()
    for game in range(games):
        policy_blue = game % 2 == 0
        blue, orange = (policy, opponent) if policy_blue else (opponent, policy)
        result = play_match(
            blue,
            orange,
            team_size=team_size,
            seconds=seconds,
            seed=seed + game,
            obs_extras=obs_extras,
        )
        mine = result.goals_blue if policy_blue else result.goals_orange
        theirs = result.goals_orange if policy_blue else result.goals_blue
        goals_for += mine
        goals_against += theirs
        if mine > theirs:
            wins += 1
        elif mine < theirs:
            losses += 1
        else:
            draws += 1
    return {
        "opponent": opponent.name,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        # Draws count half, like chess scoring.
        "score": (wins + 0.5 * draws) / games if games else 0.0,
        "seconds": round(time.perf_counter() - started, 2),
    }


def save_replay(path: Path, result: MatchResult, meta: dict[str, Any] | None = None) -> None:
    payload = {
        "format": REPLAY_FORMAT,
        "fps": TICKS_PER_SECOND / TICK_SKIP,
        "meta": {**result.summary(), **(meta or {})},
        "frames": result.frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
