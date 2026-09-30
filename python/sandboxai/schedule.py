"""Pure PPO rollout-schedule helpers.

Kept small so tests, manifests and future analysis tools can reason about
requested versus full-rollout timesteps without importing SB3.
"""

from __future__ import annotations

import math
from typing import Any


def full_rollout_schedule(
    requested_timesteps: int,
    environment_count: int,
    rollout_length: int,
) -> dict[str, Any]:
    """Returns the number of complete on-policy updates SB3 will schedule."""
    if requested_timesteps < 1:
        raise ValueError("requested_timesteps must be positive")
    if environment_count < 1 or rollout_length < 1:
        raise ValueError("environment_count and rollout_length must be positive")
    rollout_batch = int(environment_count) * int(rollout_length)
    updates = math.ceil(int(requested_timesteps) / rollout_batch)
    scheduled = updates * rollout_batch
    return {
        "rollout_length": int(rollout_length),
        "rollout_batch_size": rollout_batch,
        "expected_updates": updates,
        "requested_timesteps": int(requested_timesteps),
        "scheduled_timesteps": scheduled,
        "overshoot_timesteps": scheduled - int(requested_timesteps),
        "overshoot_fraction": (scheduled - int(requested_timesteps)) / int(requested_timesteps),
    }
