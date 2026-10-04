"""The RocketSim training environment, built the same way for training, evaluation and replays."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from rlgym.api import RLGym, StateMutator
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BALL_RADIUS,
    BLUE_TEAM,
    CEILING_Z,
    SIDE_WALL_X,
)
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from .config import OBS_PADDING, TICK_SKIP
from .rewards import StagedReward

#: Inner field box used for random spawns (keeps cars and ball off the walls).
_SPAWN_X = SIDE_WALL_X - 600
_SPAWN_Y = 4000


class MixedStartMutator(StateMutator[GameState]):
    """Kickoffs most of the time, random ball/car placements otherwise.

    Random starts show the bot many more ball situations per hour than kickoffs
    alone, which is what speeds up learning to touch the ball. Kickoffs stay
    the majority because every real match starts with one.
    """

    def __init__(self, kickoff_probability: float = 0.5, rng: random.Random | None = None):
        self.kickoff_probability = kickoff_probability
        self.kickoff = KickoffMutator()
        self.rng = rng or random.Random()

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        if self.rng.random() < self.kickoff_probability:
            # RLGym's KickoffMutator shuffles spawns with the global ``random``;
            # drive it from our seeded generator so seeded matches repeat exactly.
            saved = random.getstate()
            random.seed(self.rng.getrandbits(64))
            try:
                self.kickoff.apply(state, shared_info)
            finally:
                random.setstate(saved)
            return
        rng = self.rng
        state.ball.position = np.array(
            [
                rng.uniform(-_SPAWN_X, _SPAWN_X),
                rng.uniform(-_SPAWN_Y, _SPAWN_Y),
                rng.choice([BALL_RADIUS + 2, rng.uniform(BALL_RADIUS, CEILING_Z / 2)]),
            ],
            dtype=np.float32,
        )
        state.ball.linear_velocity = np.array(
            [rng.uniform(-800, 800), rng.uniform(-800, 800), rng.uniform(0, 400)],
            dtype=np.float32,
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            side = -1 if car.team_num == BLUE_TEAM else 1
            car.physics.position = np.array(
                [rng.uniform(-_SPAWN_X, _SPAWN_X), side * rng.uniform(500, _SPAWN_Y), 17],
                dtype=np.float32,
            )
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array(
                [0, rng.uniform(-np.pi, np.pi), 0], dtype=np.float32
            )
            car.boost_amount = rng.uniform(0, 100)


def make_env(
    team_size: int = 1,
    reward_stage: int = 1,
    episode_seconds: float = 300.0,
    no_touch_seconds: float = 30.0,
    kickoff_probability: float = 0.5,
    seed: int | None = None,
) -> RLGym:
    """A RocketSim match: ``team_size`` cars per side, all controlled by the caller."""
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
            MixedStartMutator(kickoff_probability, random.Random(seed)),
        ),
        obs_builder=DefaultObs(zero_padding=OBS_PADDING),
        action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
        reward_fn=StagedReward(reward_stage),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(episode_seconds),
            NoTouchTimeoutCondition(no_touch_seconds),
        ),
        # rlbot_delay=True reproduces RLBot's one-tick input delay, so a policy
        # trained here acts with the same timing in the real game.
        transition_engine=RocketSimEngine(rlbot_delay=True),
    )


def lookup_table() -> np.ndarray:
    """The 90-entry action table (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)."""
    return LookupTableAction.make_lookup_table()
