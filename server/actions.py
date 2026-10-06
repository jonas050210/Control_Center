"""Human control mappings shared by the REST API and the tests.

The Playground sends one *press* per request; a press is translated into the
10-dimensional ``MultiDiscrete`` action consumed by :class:`env.shooter_env.ShooterEnv`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Press -> (controls, repeat) used by the browser D-pad and the test suite.
PLAY_ACTIONS: dict[str, dict[str, Any]] = {
    "forward": {"move": 1, "repeat": 4},
    "back": {"move": -1, "repeat": 4},
    "strafe_left": {"strafe": -1, "repeat": 4},
    "strafe_right": {"strafe": 1, "repeat": 4},
    "sprint_forward": {"move": 1, "sprint": True, "repeat": 6},
    "turn_left": {"yaw": -1, "repeat": 1},
    "turn_right": {"yaw": 1, "repeat": 1},
    "look_up": {"pitch": 1, "repeat": 1},
    "look_down": {"pitch": -1, "repeat": 1},
    "move_fire": {"move": 1, "shoot": True, "repeat": 4},
    "fire": {"shoot": True, "repeat": 8},
    "reload": {"reload": True, "repeat": 30},
    "jump": {"jump": True, "repeat": 1},
    "wait": {"repeat": 1},
}

BOT_BEHAVIORS = {"Tactical": "full", "Shooter": "shooter", "Walker": "walker"}
HEURISTIC = "Heuristic AI"


def make_action(
    *,
    move: int = 0,
    strafe: int = 0,
    yaw: int = 0,
    pitch: int = 0,
    shoot: bool = False,
    jump: bool = False,
    reload: bool = False,
    sprint: bool = False,
    stance: int = 0,
) -> np.ndarray:
    """Encode one human control frame as a ``MultiDiscrete`` action vector."""
    action = np.asarray([1, 1, 1, 1, int(shoot), int(sprint), stance, 1, int(jump), int(reload)],
                        dtype=np.int64)
    action[0] = int(np.clip(move, -1, 1)) + 1
    action[1] = int(np.clip(strafe, -1, 1)) + 1
    action[2] = int(np.clip(yaw, -1, 1)) + 1
    action[3] = int(np.clip(pitch, -1, 1)) + 1
    return action


def action_from_controls(controls: dict[str, Any], stance: int = 0, sprint: bool = False) -> np.ndarray:
    """Build an action from a JSON control payload with defensive defaults."""
    return make_action(
        move=int(controls.get("move", 0)),
        strafe=int(controls.get("strafe", 0)),
        yaw=int(controls.get("yaw", 0)),
        pitch=int(controls.get("pitch", 0)),
        shoot=bool(controls.get("shoot", False)),
        jump=bool(controls.get("jump", False)),
        reload=bool(controls.get("reload", False)),
        sprint=bool(controls.get("sprint", sprint)),
        stance=int(controls.get("stance", stance)),
    )
