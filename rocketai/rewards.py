"""Staged rewards: first reach and touch the ball, then shoot at the goal, then the full game.

Every component is computed from the agent's own perspective (orange agents
attack the blue goal), and the per-component totals are published in
``shared_info["reward_parts"]`` so the UI can show what the bot is paid for.
Weights follow the usual RLGym recipe: dense shaping early, sparse goal reward
growing with the stage. They are tunable here and nowhere else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BACK_NET_Y,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    CAR_MAX_SPEED,
    CEILING_Z,
    ORANGE_TEAM,
)

#: Weight per component and stage. Components with weight 0 are skipped.
STAGE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "speed_to_ball": 1.0,
        "face_ball": 0.25,
        "touch": 10.0,
        "in_air": 0.01,  # random play is airborne ~80% of the time; keep this a tiny nudge
        "ball_to_goal": 0.0,
        "goal": 20.0,
        "boost_keep": 0.0,
        "air_touch": 0.0,
    },
    2: {
        "speed_to_ball": 0.5,
        "face_ball": 0.1,
        "touch": 5.0,
        "in_air": 0.02,
        "ball_to_goal": 2.0,
        "goal": 50.0,
        "boost_keep": 0.0,
        "air_touch": 0.0,
    },
    3: {
        "speed_to_ball": 0.25,
        "face_ball": 0.05,
        "touch": 3.0,
        "in_air": 0.02,
        "ball_to_goal": 3.0,
        "goal": 100.0,
        "boost_keep": 0.2,
        "air_touch": 5.0,
    },
}

COMPONENTS = tuple(STAGE_WEIGHTS[1])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros_like(vector)


def component_values(agent: AgentID, state: GameState) -> dict[str, float]:
    """Unweighted reward components for one agent in one state (pure, testable)."""
    car = state.cars[agent]
    car_pos = car.physics.position
    ball = state.ball
    to_ball = _unit(ball.position - car_pos)
    attack_y = -BACK_NET_Y if car.team_num == ORANGE_TEAM else BACK_NET_Y
    to_goal = _unit(np.array([0.0, attack_y, 0.0], dtype=np.float32) - ball.position)

    touched = car.ball_touches > 0
    height = float(ball.position[2])
    values = {
        "speed_to_ball": max(
            0.0, float(np.dot(car.physics.linear_velocity, to_ball)) / CAR_MAX_SPEED
        ),
        "face_ball": float(np.dot(car.physics.forward, to_ball)),
        "touch": 1.0 if touched else 0.0,
        "in_air": 0.0 if car.on_ground else 1.0,
        "ball_to_goal": float(np.dot(ball.linear_velocity, to_goal)) / BALL_MAX_SPEED,
        "goal": 0.0,
        "boost_keep": float(np.sqrt(max(0.0, car.boost_amount) / 100.0)),
        "air_touch": (
            min(1.0, max(0.0, height - 2 * BALL_RADIUS) / (CEILING_Z / 2))
            if touched and not car.on_ground
            else 0.0
        ),
    }
    scoring = state.scoring_team
    if scoring is not None:
        values["goal"] = 1.0 if scoring == car.team_num else -1.0
    return values


class StagedReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, stage: int = 1):
        if stage not in STAGE_WEIGHTS:
            raise ValueError(f"unknown reward stage {stage}")
        self.stage = stage
        self.weights = {name: w for name, w in STAGE_WEIGHTS[stage].items() if w}

    def reset(
        self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]
    ) -> None:
        shared_info["reward_parts"] = dict.fromkeys(self.weights, 0.0)

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        parts = shared_info.setdefault("reward_parts", dict.fromkeys(self.weights, 0.0))
        rewards: dict[AgentID, float] = {}
        for agent in agents:
            values = component_values(agent, state)
            total = 0.0
            for name, weight in self.weights.items():
                contribution = weight * values[name]
                parts[name] = parts.get(name, 0.0) + contribution
                total += contribution
            rewards[agent] = total
        return rewards
