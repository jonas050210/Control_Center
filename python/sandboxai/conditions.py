"""Training/evaluation condition space and per-condition performance tracking.

A *condition* is one combination of the things an episode can vary along:
the map, the lighting, the scenario, how many enemies there are, and the
curriculum level. Training across a distribution of conditions instead of a
single arena is what separates "the policy learned this map" from "the
policy learned to fight", and the only way to tell those apart is to keep
the statistics per condition rather than in one pooled average.

Three pieces live here:

``ConditionSpace``
    Declares the axes and produces conditions, either sampled (training) or
    enumerated (evaluation). Sampling is seeded and reproducible: the same
    seed yields the same episode sequence, which is required for replay.

``ConditionTracker``
    Rolling per-condition statistics with a bounded window, so a long run
    does not grow without bound and recent performance is what is reported.

``generalization_report``
    A structured per-condition report (Phase 17): it answers "where does
    this policy fall apart", which a pooled mean reward cannot.

Nothing here touches the simulation. Conditions are applied by the caller
through ``EnvironmentCore.set_map`` / ``set_lighting_mode`` /
``set_scenario`` / ``set_curriculum_level``; the condition itself is never
part of an observation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
import random
from typing import Any, Iterable, Iterator, Sequence

# Mirrors MapLibrary.MAPS ids in scripts/world/map_library.gd. Kept as a
# plain list rather than imported, because the Python side must be usable
# without a running Godot; python/tests/test_conditions.py checks it against
# the GDScript source so drift fails a test.
MAP_IDS: tuple[str, ...] = (
    "open_field",
    "training_yard",
    "blind_corner",
    "cover_field",
    "long_corridor",
    "two_rooms",
    "pillar_hall",
    "catwalks",
    "compound",
    "ambush_alley",
    "echo_maze",
    "combat_complex",
    "crossfire_lab",
    "night_yard",
    "foggy_field",
    "random_ops",
)

# Mirrors LightingProfile.MODE_IDS in scripts/perception/lighting_profile.gd.
LIGHTING_IDS: tuple[str, ...] = (
    "normal",
    "low_light",
    "night",
    "fog",
    "high_contrast",
    "mixed",
)


@dataclass(frozen=True)
class Condition:
    """One fully specified episode setup.

    ``seed`` is part of the condition on purpose: an episode is only
    reproducible if the seed travels with the rest of the setup.
    """

    map_id: str = ""
    lighting: str = ""
    scenario: str = ""
    enemy_count: int = 1
    level: int = 1
    seed: int = 0

    @property
    def key(self) -> str:
        """Stable identity used to group statistics.

        Deliberately excludes the seed: two episodes on the same map with
        the same lighting and enemy count are the same *condition* even
        though they are different episodes.
        """
        return "|".join(
            (
                self.map_id or "-",
                self.lighting or "-",
                self.scenario or "-",
                f"n{self.enemy_count}",
                f"L{self.level}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionSpace:
    """The distribution episodes are drawn from.

    Every axis is a plain sequence, so an experiment can narrow any of them
    (one map, all lightings) without a separate code path. An empty
    sequence means "leave that axis at the environment's default", which is
    how the pre-existing single-arena training keeps working unchanged.
    """

    maps: Sequence[str] = field(default_factory=lambda: list(MAP_IDS))
    lightings: Sequence[str] = field(default_factory=lambda: list(LIGHTING_IDS))
    scenarios: Sequence[str] = field(default_factory=list)
    enemy_counts: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    levels: Sequence[int] = field(default_factory=lambda: [6])
    base_seed: int = 0

    def __post_init__(self) -> None:
        if not self.enemy_counts:
            raise ValueError("enemy_counts must not be empty")
        if not self.levels:
            raise ValueError("levels must not be empty")
        if any(count < 0 for count in self.enemy_counts):
            raise ValueError("enemy_counts must be non-negative")

    def size(self) -> int:
        """Number of distinct conditions in the enumerated grid."""
        return (
            max(1, len(self.maps))
            * max(1, len(self.lightings))
            * max(1, len(self.scenarios))
            * len(self.enemy_counts)
            * len(self.levels)
        )

    def sample(self, rng: random.Random) -> Condition:
        """Draws one condition. Uses the caller's RNG so the episode
        sequence is a pure function of the training seed."""
        return Condition(
            map_id=rng.choice(list(self.maps)) if self.maps else "",
            lighting=rng.choice(list(self.lightings)) if self.lightings else "",
            scenario=rng.choice(list(self.scenarios)) if self.scenarios else "",
            enemy_count=rng.choice(list(self.enemy_counts)),
            level=rng.choice(list(self.levels)),
            seed=rng.randrange(0, 2**31 - 1),
        )

    def sample_many(self, count: int, seed: int | None = None) -> list[Condition]:
        rng = random.Random(self.base_seed if seed is None else seed)
        return [self.sample(rng) for _ in range(count)]

    def enumerate_grid(self, seeds_per_condition: int = 1) -> Iterator[Condition]:
        """Every condition in the space, with deterministic seeds.

        This is the evaluation counterpart of ``sample``: an evaluation run
        must cover the space exactly once (or N times) rather than
        sampling it, otherwise two policies are compared on different
        episodes.
        """
        maps: Iterable[str] = self.maps or [""]
        lightings: Iterable[str] = self.lightings or [""]
        scenarios: Iterable[str] = self.scenarios or [""]
        index = 0
        for map_id in maps:
            for lighting in lightings:
                for scenario in scenarios:
                    for enemy_count in self.enemy_counts:
                        for level in self.levels:
                            for repeat in range(max(1, seeds_per_condition)):
                                yield Condition(
                                    map_id=map_id,
                                    lighting=lighting,
                                    scenario=scenario,
                                    enemy_count=enemy_count,
                                    level=level,
                                    seed=self.base_seed + index * 1009 + repeat,
                                )
                                index += 1


@dataclass
class ConditionStats:
    """Rolling statistics for one condition."""

    key: str
    window: int = 100
    episodes: int = 0
    rewards: deque[float] = field(default_factory=deque)
    wins: deque[float] = field(default_factory=deque)
    lengths: deque[float] = field(default_factory=deque)

    def record(self, reward: float, won: bool, steps: int) -> None:
        self.episodes += 1
        for series, value in (
            (self.rewards, float(reward)),
            (self.wins, 1.0 if won else 0.0),
            (self.lengths, float(steps)),
        ):
            series.append(value)
            while len(series) > self.window:
                series.popleft()

    @staticmethod
    def _mean(values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "condition": self.key,
            "episodes": self.episodes,
            "window": len(self.rewards),
            "mean_reward": self._mean(self.rewards),
            "win_rate": self._mean(self.wins),
            "mean_length": self._mean(self.lengths),
        }


class ConditionTracker:
    """Per-condition performance, plus the pooled totals.

    Tracking per condition is the point: a policy that wins 90% on open
    maps and 10% in the dark averages a respectable 50% that hides the
    actual failure. ``worst_conditions`` exists so that failure is the
    first thing a report shows.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._stats: dict[str, ConditionStats] = {}
        self._order: list[str] = []

    def record(self, condition: Condition, reward: float, won: bool, steps: int) -> None:
        key = condition.key
        stats = self._stats.get(key)
        if stats is None:
            stats = ConditionStats(key=key, window=self.window)
            self._stats[key] = stats
            self._order.append(key)
        stats.record(reward, won, steps)

    def keys(self) -> list[str]:
        return list(self._order)

    def stats_for(self, condition: Condition | str) -> dict[str, Any]:
        key = condition if isinstance(condition, str) else condition.key
        stats = self._stats.get(key)
        return stats.summary() if stats else {}

    def total_episodes(self) -> int:
        return sum(stats.episodes for stats in self._stats.values())

    def overall(self) -> dict[str, Any]:
        rewards: list[float] = []
        wins: list[float] = []
        lengths: list[float] = []
        for stats in self._stats.values():
            rewards.extend(stats.rewards)
            wins.extend(stats.wins)
            lengths.extend(stats.lengths)
        count = len(rewards)
        return {
            "conditions": len(self._stats),
            "episodes": self.total_episodes(),
            "mean_reward": sum(rewards) / count if count else 0.0,
            "win_rate": sum(wins) / len(wins) if wins else 0.0,
            "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        }

    def summaries(self) -> list[dict[str, Any]]:
        return [self._stats[key].summary() for key in self._order]

    def worst_conditions(self, count: int = 5, metric: str = "win_rate") -> list[dict[str, Any]]:
        ordered = sorted(self.summaries(), key=lambda row: row.get(metric, 0.0))
        return ordered[:count]


def generalization_report(tracker: ConditionTracker, top_n: int = 5) -> dict[str, Any]:
    """Structured report of per-condition performance (Phase 17).

    Returns data, not prose, so it can be asserted in a test, diffed
    between runs and rendered by the Control Center.
    """
    summaries = tracker.summaries()
    win_rates = [row["win_rate"] for row in summaries]
    spread = (max(win_rates) - min(win_rates)) if win_rates else 0.0
    return {
        "overall": tracker.overall(),
        "by_condition": summaries,
        "worst": tracker.worst_conditions(top_n),
        "best": list(reversed(tracker.worst_conditions(len(summaries))))[:top_n],
        # A large spread means the policy is condition-dependent, i.e. it
        # has memorized something it should have generalized.
        "win_rate_spread": spread,
        "generalizes": spread <= 0.25 and len(summaries) > 1,
    }


def format_generalization_report(report: dict[str, Any]) -> str:
    """Human-readable rendering of ``generalization_report``."""
    overall = report["overall"]
    lines = [
        "Generalization report",
        "=" * 72,
        (
            f"conditions {overall['conditions']}  episodes {overall['episodes']}  "
            f"mean reward {overall['mean_reward']:.2f}  win rate {overall['win_rate']:.1%}"
        ),
        f"win-rate spread across conditions: {report['win_rate_spread']:.1%}",
        "",
        f"{'condition':<52}{'eps':>5}{'reward':>10}{'win':>8}",
        "-" * 72,
    ]
    for row in report["by_condition"]:
        lines.append(
            f"{row['condition']:<52}{row['episodes']:>5}"
            f"{row['mean_reward']:>10.2f}{row['win_rate']:>8.1%}"
        )
    return "\n".join(lines)
