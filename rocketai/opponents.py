"""Players for matches: trained policies and simple scripted bots used as a skill ladder."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, BLUE_TEAM

from .env import lookup_table
from .model import ActorCritic, load_policy

_TABLE = lookup_table()


def ground_index(throttle: int, steer: int, boost: bool = False, handbrake: bool = False) -> int:
    """Index of the ground action with these controls in the 90-entry table."""
    target = [
        1 if boost else throttle,
        steer,
        0,
        steer,
        0,
        0,
        1 if boost else 0,
        1 if handbrake else 0,
    ]
    for index, row in enumerate(_TABLE):
        if list(row) == target:
            return index
    raise ValueError(f"no ground action for {target}")


IDLE_ACTION = ground_index(0, 0)


class Player(Protocol):
    name: str

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]: ...


class PolicyPlayer:
    def __init__(self, model: ActorCritic, name: str = "policy", deterministic: bool = True):
        self.model = model
        self.name = name
        self.deterministic = deterministic
        #: Set ``explain = True`` to keep, per agent, what the network "thought"
        #: on its last decision (used by the live view).
        self.explain = False
        self.last_info: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_checkpoint(cls, path: Path, deterministic: bool = True) -> PolicyPlayer:
        return cls(load_policy(path), name=path.stem, deterministic=deterministic)

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]:
        if not agents:
            return {}
        batch = np.stack([obs[agent] for agent in agents]).astype(np.float32)
        if not self.explain:
            actions, _, _ = self.model.act(batch, deterministic=self.deterministic)
            return {agent: int(a) for agent, a in zip(agents, actions, strict=True)}
        with torch.no_grad():
            tensor = torch.as_tensor(batch)
            probs = torch.softmax(self.model.logits(tensor), dim=-1).numpy()
            values = self.model.value(tensor).numpy()
        if self.deterministic:
            actions = probs.argmax(axis=-1)
        else:
            rng = np.random.default_rng()
            actions = np.array([rng.choice(len(p), p=p / p.sum()) for p in probs])
        self.last_info = {
            agent: explain_decision(probs[i], float(values[i]), int(actions[i]))
            for i, agent in enumerate(agents)
        }
        return {agent: int(a) for agent, a in zip(agents, actions, strict=True)}


def describe_action(index: int) -> str:
    """Human-readable German label for one of the 90 actions."""
    throttle, steer, pitch, yaw, roll, jump, boost, handbrake = (float(v) for v in _TABLE[index])
    parts: list[str] = []
    if jump:
        if pitch or yaw or roll:
            vertical = {-1.0: "vorwärts", 1.0: "rückwärts"}.get(pitch, "")
            side = ("rechts" if (yaw or roll) > 0 else "links") if (yaw or roll) else ""
            parts.append("Flip " + " ".join(x for x in (vertical, side) if x))
        else:
            parts.append("Sprung")
    else:
        if throttle > 0:
            parts.append("Gas")
        elif throttle < 0:
            parts.append("Rückwärts")
        if pitch:
            parts.append("Nase runter" if pitch < 0 else "Nase hoch")
        if steer:
            parts.append("rechts" if steer > 0 else "links")
        if roll:
            parts.append("Rolle " + ("rechts" if roll > 0 else "links"))
    if boost:
        parts.append("Boost")
    if handbrake:
        parts.append("Luftrolle" if jump or pitch or roll else "Drift")
    return " · ".join(parts) or "Nichts tun"


ACTION_LABELS = [describe_action(i) for i in range(len(_TABLE))]


def explain_decision(probs: np.ndarray, value: float, action: int, top: int = 5) -> dict[str, Any]:
    order = np.argsort(probs)[::-1][:top]
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    row = _TABLE[action]
    return {
        "action": action,
        "label": ACTION_LABELS[action],
        "controls": [
            round(float(v), 2) for v in row
        ],  # throttle steer pitch yaw roll jump boost handbrake
        "top": [[ACTION_LABELS[i], round(float(probs[i]), 4)] for i in order],
        "value": round(value, 3),
        # 1 = completely sure (one action), 0 = uniform over all 90 actions.
        "confidence": round(1.0 - entropy / float(np.log(len(probs))), 3),
    }


class IdleBot:
    name = "idle"

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]:
        return dict.fromkeys(agents, IDLE_ACTION)


class RandomBot:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]:
        return {agent: self.rng.randrange(len(_TABLE)) for agent in agents}


def steer_towards(car: Car, target: np.ndarray) -> tuple[int, float]:
    """(steer, angle) to turn ``car`` towards ``target``."""
    delta = target - car.physics.position
    local_x = float(np.dot(car.physics.forward, delta))
    local_y = float(np.dot(car.physics.right, delta))
    angle = math.atan2(local_y, local_x)
    steer = 0 if abs(angle) < 0.12 else (1 if angle > 0 else -1)
    return steer, angle


def shooting_target(car: Car, ball: np.ndarray) -> np.ndarray:
    """A point just behind the ball as seen from the goal ``car`` attacks.

    Driving through it pushes the ball towards the opponent's net instead of
    wherever the car happened to come from.
    """
    attack_y = BACK_WALL_Y if car.team_num == BLUE_TEAM else -BACK_WALL_Y
    away = ball - np.array([0.0, attack_y, ball[2]], dtype=np.float32)
    norm = float(np.linalg.norm(away))
    direction = away / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)
    car_pos = car.physics.position
    # "Ahead" of the ball = on the wrong side; hitting it now risks an own goal.
    ahead = float(np.dot(car_pos - ball, -direction)) > 0
    if ahead:
        # Loop round: aim well behind the ball and off to the car's side.
        side = np.sign(car_pos[0] - ball[0]) or 1.0
        target = ball + direction * 900.0
        target[0] += side * 700.0
        return np.clip(target, [-3900, -5000, 0], [3900, 5000, 0]).astype(np.float32)
    distance = float(np.linalg.norm(ball - car_pos))
    # Close to the ball, just hit it; the offset matters on the approach.
    return ball + direction * 140.0 * min(1.0, distance / 1500.0)


class ChaserBot:
    """Drives at the ball, boosting when it is lined up: the classic 'ball chaser'."""

    name = "chaser"

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]:
        actions = {}
        for agent in agents:
            car = state.cars[agent]
            steer, angle = steer_towards(car, shooting_target(car, state.ball.position))
            boost = abs(angle) < 0.3 and car.boost_amount > 0
            actions[agent] = ground_index(1, steer, boost=boost, handbrake=abs(angle) > 2.2)
        return actions


class DefenderBot(ChaserBot):
    """Chases in its own half, otherwise falls back between the ball and its own goal."""

    name = "defender"

    def act(
        self, agents: list[str], obs: dict[str, np.ndarray], state: GameState
    ) -> dict[str, int]:
        actions = super().act(agents, obs, state)
        for agent in agents:
            car = state.cars[agent]
            own_goal_y = -BACK_WALL_Y if car.team_num == BLUE_TEAM else BACK_WALL_Y
            ball = state.ball.position
            # Positive when the ball is on our side of midfield; the margin makes
            # a ball sitting on the halfway line (every kickoff) count as ours.
            depth_towards_own_goal = float(ball[1]) * (1 if own_goal_y > 0 else -1)
            if depth_towards_own_goal > -1000:
                continue
            guard = np.array([ball[0] * 0.3, own_goal_y * 0.7, 17.0], dtype=np.float32)
            distance = float(np.linalg.norm(guard - car.physics.position))
            steer, angle = steer_towards(car, guard)
            throttle = 1 if distance > 400 else 0
            actions[agent] = ground_index(
                throttle, steer, boost=distance > 2500 and abs(angle) < 0.3
            )
        return actions


#: The ladder shown in the UI, weakest first.
SCRIPTED_BOTS = {
    "idle": ("Stillstand", IdleBot),
    "random": ("Zufall", RandomBot),
    "chaser": ("Balljäger", ChaserBot),
    "defender": ("Verteidiger", DefenderBot),
}


def make_player(spec: str) -> Player:
    """``idle`` / ``random`` / ``chaser`` / ``defender`` or a path to a checkpoint."""
    if spec in SCRIPTED_BOTS:
        return SCRIPTED_BOTS[spec][1]()
    path = Path(spec)
    if path.is_file():
        return PolicyPlayer.from_checkpoint(path)
    raise ValueError(
        f"unknown opponent {spec!r}: use {', '.join(SCRIPTED_BOTS)} or a checkpoint path"
    )
