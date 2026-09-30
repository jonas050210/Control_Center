"""Automatic curriculum: promotion, demotion and the guards around them.

Phase 13. The GDScript side already has ``CurriculumConfig`` (what each
level enables) and ``CurriculumController`` (level state inside a running
environment). What was missing is the *policy* that decides when to move,
expressed so it can be unit-tested without a simulation:

* performance is measured over a **rolling window**, not the last episode;
* a level change needs a **minimum number of episodes** at that level, so
  three lucky episodes cannot promote;
* after any change there is a **cooldown**, which stops the oscillation
  you otherwise get right at the threshold;
* promotion and demotion have **separate thresholds** (hysteresis);
* every parameter is configurable and the whole thing is deterministic:
  the same sequence of results produces the same level trajectory.

Demotion matters as much as promotion. A curriculum that can only go up
turns a temporary regression into permanent failure, because the agent is
left on a level it can no longer solve.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

## Mirrors CurriculumConfig.Level in scripts/core/curriculum_config.gd.
MIN_LEVEL: int = 1
MAX_COMBAT_LEVEL: int = 10


@dataclass
class CurriculumSchedule:
    """Configuration for the automatic curriculum."""

    min_level: int = MIN_LEVEL
    max_level: int = MAX_COMBAT_LEVEL
    start_level: int = 1
    ## Rolling window length, in episodes.
    window: int = 50
    ## Episodes that must be played at the current level before it may change.
    min_episodes_per_level: int = 30
    ## Episodes after a change during which no further change may happen.
    cooldown_episodes: int = 20
    ## Rolling success rate at or above which the level is raised.
    promote_threshold: float = 0.70
    ## Rolling success rate at or below which the level is lowered.
    demote_threshold: float = 0.25
    ## Whether demotion is allowed at all.
    allow_demotion: bool = True

    def __post_init__(self) -> None:
        if self.min_level < 1:
            raise ValueError("min_level must be >= 1")
        if self.max_level < self.min_level:
            raise ValueError("max_level must be >= min_level")
        if not self.min_level <= self.start_level <= self.max_level:
            raise ValueError("start_level must lie within [min_level, max_level]")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.demote_threshold >= self.promote_threshold:
            raise ValueError("demote_threshold must be below promote_threshold (hysteresis)")


@dataclass
class CurriculumState:
    level: int
    episodes_at_level: int = 0
    cooldown_remaining: int = 0
    window: deque[float] = field(default_factory=deque)
    promotions: int = 0
    demotions: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class AutoCurriculum:
    """Rolling-performance curriculum controller.

    Usage is one call per finished episode::

        change = curriculum.record(success=won)

    ``record`` returns the level change that happened on that episode (or
    ``None``), so a caller can log it and reconfigure its environments.
    """

    def __init__(self, schedule: CurriculumSchedule | None = None) -> None:
        self.schedule = schedule or CurriculumSchedule()
        self.state = CurriculumState(level=self.schedule.start_level)

    @property
    def level(self) -> int:
        return self.state.level

    def success_rate(self) -> float:
        window = self.state.window
        return sum(window) / len(window) if window else 0.0

    def ready_to_change(self) -> bool:
        """Whether the guards currently allow a level change at all."""
        schedule = self.schedule
        state = self.state
        return (
            state.cooldown_remaining <= 0
            and state.episodes_at_level >= schedule.min_episodes_per_level
            and len(state.window) >= min(schedule.window, schedule.min_episodes_per_level)
        )

    def record(self, success: bool, reward: float = 0.0) -> dict[str, Any] | None:
        """Records one finished episode and applies the curriculum rules."""
        schedule = self.schedule
        state = self.state
        state.episodes_at_level += 1
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
        state.window.append(1.0 if success else 0.0)
        while len(state.window) > schedule.window:
            state.window.popleft()

        if not self.ready_to_change():
            return None

        rate = self.success_rate()
        if rate >= schedule.promote_threshold and state.level < schedule.max_level:
            return self._change(state.level + 1, "promote", rate, reward)
        if (
            schedule.allow_demotion
            and rate <= schedule.demote_threshold
            and state.level > schedule.min_level
        ):
            return self._change(state.level - 1, "demote", rate, reward)
        return None

    def _change(self, new_level: int, kind: str, rate: float, reward: float) -> dict[str, Any]:
        state = self.state
        event = {
            "from_level": state.level,
            "to_level": new_level,
            "kind": kind,
            "success_rate": rate,
            "reward": reward,
            "episodes_at_level": state.episodes_at_level,
        }
        state.level = new_level
        state.episodes_at_level = 0
        state.cooldown_remaining = self.schedule.cooldown_episodes
        # The window is cleared on purpose: performance measured on the
        # previous level says nothing about the new one, and carrying it
        # over is what makes a curriculum stair-step through several levels
        # on one good streak.
        state.window.clear()
        if kind == "promote":
            state.promotions += 1
        else:
            state.demotions += 1
        state.history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "level": self.state.level,
            "success_rate": self.success_rate(),
            "episodes_at_level": self.state.episodes_at_level,
            "cooldown_remaining": self.state.cooldown_remaining,
            "promotions": self.state.promotions,
            "demotions": self.state.demotions,
            "window_size": len(self.state.window),
        }

    def history(self) -> list[dict[str, Any]]:
        return list(self.state.history)
