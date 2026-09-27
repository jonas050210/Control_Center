"""Canonical action adapter shared by recording, BC, Godot, and RL.

The dataset stores human-readable movement values (-1/0/+1), while Gymnasium's
``MultiDiscrete`` space stores zero-based category indices.  Keeping the
translation in one module prevents the subtle action-order drift that used to
exist between the recorder, BC runner, and sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from data_pipeline.schema import ActionState


ACTION_NAMES: Tuple[str, ...] = (
    "move_x",
    "move_y",
    "jump",
    "crouch",
    "sprint",
    "reload",
    "fire",
    "ads",
    "mouse_dx_bin",
    "mouse_dy_bin",
)
ACTION_DIMS: Tuple[int, ...] = (3, 3, 2, 2, 2, 2, 2, 2, 21, 21)
CENTER_LOOK_BIN = 10


@dataclass(frozen=True)
class SandboxAction:
    """One canonical FPS control command.

    Movement uses ``-1/0/+1`` just like :class:`ActionState`; buttons are
    binary and look is represented by the same 21 bins used by BC.
    """

    move_x: int = 0
    move_y: int = 0
    jump: int = 0
    crouch: int = 0
    sprint: int = 0
    reload: int = 0
    fire: int = 0
    ads: int = 0
    mouse_dx_bin: int = CENTER_LOOK_BIN
    mouse_dy_bin: int = CENTER_LOOK_BIN

    def clipped(self) -> "SandboxAction":
        return SandboxAction(
            move_x=max(-1, min(1, int(self.move_x))),
            move_y=max(-1, min(1, int(self.move_y))),
            jump=int(bool(self.jump)),
            crouch=int(bool(self.crouch)),
            sprint=int(bool(self.sprint)),
            reload=int(bool(self.reload)),
            fire=int(bool(self.fire)),
            ads=int(bool(self.ads)),
            mouse_dx_bin=max(0, min(20, int(self.mouse_dx_bin))),
            mouse_dy_bin=max(0, min(20, int(self.mouse_dy_bin))),
        )

    def to_array(self) -> np.ndarray:
        """Return the zero-based Gymnasium MultiDiscrete representation."""
        action = self.clipped()
        return np.asarray(
            [
                action.move_x + 1,
                action.move_y + 1,
                action.jump,
                action.crouch,
                action.sprint,
                action.reload,
                action.fire,
                action.ads,
                action.mouse_dx_bin,
                action.mouse_dy_bin,
            ],
            dtype=np.int64,
        )

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in ACTION_NAMES}

    @classmethod
    def from_action_state(cls, action: ActionState) -> "SandboxAction":
        return cls(**{name: int(getattr(action, name)) for name in ACTION_NAMES}).clipped()

    @classmethod
    def from_array(cls, values: Sequence[Any]) -> "SandboxAction":
        """Decode a Gym action.

        Ten-element arrays use the canonical full action order.  Five-element
        arrays from SandboxAI <=1.0 remain supported as
        ``move_x, move_y, fire, yaw, pitch`` so old checkpoints and scripts can
        still be inspected and evaluated.
        """
        arr = np.asarray(values).reshape(-1)
        if arr.size >= len(ACTION_DIMS):
            return cls(
                move_x=int(arr[0]) - 1,
                move_y=int(arr[1]) - 1,
                jump=int(arr[2]),
                crouch=int(arr[3]),
                sprint=int(arr[4]),
                reload=int(arr[5]),
                fire=int(arr[6]),
                ads=int(arr[7]),
                mouse_dx_bin=int(arr[8]),
                mouse_dy_bin=int(arr[9]),
            ).clipped()
        if arr.size >= 5:  # legacy compact environment action
            return cls(
                move_x=int(arr[0]) - 1,
                move_y=int(arr[1]) - 1,
                fire=int(arr[2]),
                mouse_dx_bin=int(arr[3]),
                mouse_dy_bin=int(arr[4]),
            ).clipped()
        raise ValueError(
            f"Sandbox action must contain 10 canonical values (or 5 legacy values), got {arr.size}"
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SandboxAction":
        """Decode a canonical mapping (movement is -1/0/+1, not an index)."""
        kwargs = {
            name: values.get(name, CENTER_LOOK_BIN if "bin" in name else 0)
            for name in ACTION_NAMES
        }
        return cls(**{key: int(value) for key, value in kwargs.items()}).clipped()


def coerce_sandbox_action(action: Any) -> SandboxAction:
    if isinstance(action, SandboxAction):
        return action.clipped()
    if isinstance(action, ActionState):
        return SandboxAction.from_action_state(action)
    if isinstance(action, Mapping):
        return SandboxAction.from_mapping(action)
    if isinstance(action, (Sequence, np.ndarray)) and not isinstance(action, (str, bytes)):
        return SandboxAction.from_array(action)
    raise TypeError(f"Unsupported sandbox action type: {type(action)!r}")


def neutral_action() -> np.ndarray:
    return SandboxAction().to_array()
