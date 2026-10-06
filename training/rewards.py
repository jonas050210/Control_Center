"""Centralized reward shaping for the arena environment."""

from __future__ import annotations

from dataclasses import dataclass, field


KILL_BONUS = 5.0
"""Reward for removing the opponent."""

KILL_DECAY_FLOOR = 0.25
"""A kill at the time limit is still worth this share of the full bonus.

Killing is the win condition, so the reward has to shrink over the episode: a
policy that only fights at the end (or hides until the health comparison) then
earns clearly less than one that finishes the fight early.
"""


@dataclass
class RewardEvents:
    distance_before: float
    distance_after: float
    aim_error_radians: float
    enemy_visible: bool = True
    elapsed_seconds: float = 0.0
    time_limit_seconds: float = 60.0
    hits: int = 0
    headshots: int = 0
    kills: int = 0
    incoming_hits: int = 0
    deaths: int = 0
    successful_dodges: int = 0
    wasted_ammo: int = 0
    physics_steps: int = 1


@dataclass
class RewardBreakdown:
    total: float
    components: dict[str, float] = field(default_factory=dict)


AIM_BONUS = 0.005
"""Per-step reward for keeping the crosshair on a visible opponent.

The bonus exists for credit assignment, not as a goal of its own: over a 30 s
episode at 15 decisions/s it adds up to ~2.25, which stays below the +5 kill
bonus. It used to be 0.05, i.e. ~22 per episode - farming the crosshair earned
ten times more than actually killing the opponent, and the measured symptom was
a PPO policy that kept winning on health at the time limit without ever firing
a lethal shot (0 % kill rate in 250k steps).
"""


def shape_reward(events: RewardEvents) -> RewardBreakdown:
    """Apply the project reward contract and return auditable components.

    A small aim bonus is continuous inside a ten-degree cone, while approach
    reward is granted once for a clear step toward the rival rather than being
    tied to raw world units.
    """
    components: dict[str, float] = {}
    # Closing in is only worth something while the opponent is actually known -
    # otherwise the shaping pays for walking toward a guessed position.
    if events.enemy_visible and events.distance_after < events.distance_before - 0.02:
        components["approach"] = 0.01
    aim_window = 0.17453292519943295  # ten degrees
    components["aim"] = AIM_BONUS * max(0.0, 1.0 - max(0.0, events.aim_error_radians) / aim_window)
    headshots = min(max(0, events.hits), max(0, events.headshots))
    body_hits = max(0, events.hits) - headshots
    components["hits"] = float(body_hits)
    components["headshots"] = 2.5 * float(headshots)
    limit = max(1e-6, float(events.time_limit_seconds))
    remaining = max(KILL_DECAY_FLOOR, 1.0 - max(0.0, float(events.elapsed_seconds)) / limit)
    components["kills"] = KILL_BONUS * remaining * float(max(0, events.kills))
    components["damage_taken"] = -0.5 * float(max(0, events.incoming_hits))
    components["deaths"] = -2.0 * float(max(0, events.deaths))
    components["dodges"] = 0.1 * float(max(0, events.successful_dodges))
    components["wasted_ammo"] = -0.01 * float(max(0, events.wasted_ammo))
    components["time_pressure"] = -0.001 * max(1, int(events.physics_steps))
    total = sum(components.values())
    return RewardBreakdown(total=total, components=components)
