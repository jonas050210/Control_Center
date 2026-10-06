"""Centralized reward shaping for the arena environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardEvents:
    distance_before: float
    distance_after: float
    aim_error_radians: float
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


def shape_reward(events: RewardEvents) -> RewardBreakdown:
    """Apply the project reward contract and return auditable components.

    A small aim bonus is continuous inside a ten-degree cone, while approach
    reward is granted once for a clear step toward the rival rather than being
    tied to raw world units.
    """
    components: dict[str, float] = {}
    if events.distance_after < events.distance_before - 0.02:
        components["approach"] = 0.01
    aim_window = 0.17453292519943295  # ten degrees
    components["aim"] = 0.05 * max(0.0, 1.0 - max(0.0, events.aim_error_radians) / aim_window)
    headshots = min(max(0, events.hits), max(0, events.headshots))
    body_hits = max(0, events.hits) - headshots
    components["hits"] = float(body_hits)
    components["headshots"] = 2.5 * float(headshots)
    components["kills"] = 5.0 * float(max(0, events.kills))
    components["damage_taken"] = -0.5 * float(max(0, events.incoming_hits))
    components["deaths"] = -2.0 * float(max(0, events.deaths))
    components["dodges"] = 0.1 * float(max(0, events.successful_dodges))
    components["wasted_ammo"] = -0.01 * float(max(0, events.wasted_ammo))
    components["time_pressure"] = -0.001 * max(1, int(events.physics_steps))
    total = sum(components.values())
    return RewardBreakdown(total=total, components=components)
