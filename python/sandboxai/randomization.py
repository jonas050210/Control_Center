"""Deterministic training distribution over maps, conditions and spawns.

Phase 11. Training on one arena produces a policy that knows that arena.
This module turns "the environment" into a *distribution* of environments
and does it deterministically, so a run is reproducible and a replay's
header is enough to rebuild the exact episode it came from.

What is randomized per episode:

* map id
* layout variant (the map's layout seed)
* episode seed
* scenario
* lighting
* enemy count
* enemy spawn configuration (spawn rule + per-enemy spawn seeds)

What is deliberately **not** randomized: anything the policy observes as a
label. The map id never enters the observation vector — the whole point of
the distribution is that the policy cannot condition on "which map is
this", only on what it perceives. ``assert_no_environment_labels`` checks
that structurally against the contract, and a test runs it.

Determinism model: one ``episode_plan(index)`` call is a pure function of
(distribution configuration, master seed, index). No shared RNG state, so
sixteen parallel environments asking for episode 137 all get the same
episode, and a resumed run continues the same stream.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .conditions import LIGHTING_IDS, MAP_IDS, Condition, ConditionTracker
from .contract import OBSERVATION_SPEC

## Observation field names that would let a policy identify the
## environment instead of perceiving it. None of these exist in the
## contract, and `assert_no_environment_labels` keeps it that way.
FORBIDDEN_LABEL_TOKENS: tuple[str, ...] = (
    "map_id",
    "map_index",
    "scenario_id",
    "lighting_mode",
    "layout_id",
    "level_id",
    "episode_index",
    "condition_id",
)

## Spawn rules understood by ScenarioLibrary (scripts/scenario/scenario_library.gd).
## Mirrors the SPAWN_* constants there; python/tests/test_conditions.py parses
## the GDScript source and fails if the two lists drift apart.
SPAWN_RULES: tuple[str, ...] = (
    "ahead",
    "ring",
    "out_of_sight",
    "behind_cover",
    "surround",
    "elevated",
    "random",
)

## Stride of the per-environment episode stream (see
## ``stream_for_environment``). A fixed large prime rather than the
## environment count, so changing the environment count never re-maps which
## stream index a given environment has already played. Named here so the
## training pipeline's per-environment driver derives exactly the same
## indices this module's batch API does.
STREAM_STRIDE: int = 1_000_003


class RandomizationError(RuntimeError):
    """The training distribution is configured in a way that leaks."""


def assert_no_environment_labels() -> list[str]:
    """Fails if the observation contract exposes an environment identity.

    A policy that can read "map 7" solves the environment by lookup. This
    is cheap to check and catastrophic to get wrong, so it is an assertion
    rather than a comment.
    """
    offending = [
        field.name
        for field in OBSERVATION_SPEC
        if any(token in field.name for token in FORBIDDEN_LABEL_TOKENS)
    ]
    if offending:
        raise RandomizationError(
            "the observation contract exposes environment labels, which lets a policy "
            f"memorize instead of perceive: {offending}"
        )
    return [field.name for field in OBSERVATION_SPEC]


def _derive(master_seed: int, index: int, salt: str) -> int:
    """Stable 63-bit derivation. Same inputs -> same number, everywhere.

    ``hash()`` is salted per process and ``random.Random`` differs between
    Python versions for some methods; a digest does not. An episode plan
    must be identical on the machine that trained and the machine that
    replays.
    """
    payload = f"{master_seed}:{index}:{salt}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


@dataclass
class EnemySpawnPlan:
    """How the opposition is placed for one episode."""

    count: int
    rule: str
    seeds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodePlan:
    """A fully specified, reproducible episode setup."""

    index: int
    condition: Condition
    layout_seed: int
    ## None for a "stub" plan: the checkpoint battery and the skill-metrics
    ## sink only need condition + seed, and the engine derives the spawn
    ## layout from that seed itself, so inventing one here would be a
    ## second, disagreeing source of truth.
    spawn: EnemySpawnPlan | None = None

    @property
    def key(self) -> str:
        return self.condition.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "condition": self.condition.to_dict(),
            "layout_seed": self.layout_seed,
            "spawn": self.spawn.to_dict() if self.spawn is not None else None,
        }

    def replay_header_fields(self) -> dict[str, Any]:
        """The subset a :class:`sandboxai.replay.ReplayHeader` needs."""
        return {
            "seed": self.condition.seed,
            "map_id": self.condition.map_id,
            "scenario": self.condition.scenario,
            "lighting": self.condition.lighting,
            "enemy_count": self.condition.enemy_count,
            "curriculum_level": self.condition.level,
        }

    def environment_commands(self) -> list[dict[str, Any]]:
        """Ordered set-up calls for ``EnvironmentCore``.

        Returned as data rather than executed, because the Python side
        must stay engine-agnostic: the Godot bridge, a mock and the
        Control Center all apply these the same way.
        """
        commands: list[dict[str, Any]] = [
            {"call": "set_curriculum_level", "level": self.condition.level}
        ]
        if self.condition.map_id:
            commands.append({"call": "set_map", "map_id": self.condition.map_id})
        if self.condition.lighting:
            commands.append({"call": "set_lighting_mode", "mode_id": self.condition.lighting})
        if self.condition.scenario:
            commands.append({"call": "set_scenario", "scenario_id": self.condition.scenario})
        commands.append({"call": "reset", "seed": self.condition.seed})
        return commands


@dataclass
class TrainingDistribution:
    """The distribution episodes are drawn from during training."""

    maps: Sequence[str] = field(default_factory=lambda: list(MAP_IDS))
    lightings: Sequence[str] = field(default_factory=lambda: list(LIGHTING_IDS))
    scenarios: Sequence[str] = field(default_factory=list)
    enemy_counts: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    levels: Sequence[int] = field(default_factory=lambda: [6])
    spawn_rules: Sequence[str] = field(default_factory=lambda: list(SPAWN_RULES))
    master_seed: int = 1234
    ## Number of distinct layout variants drawn per map. More variants
    ## means less chance of memorizing a specific obstacle arrangement.
    layout_variants: int = 8

    def __post_init__(self) -> None:
        if not self.maps:
            raise ValueError("maps must not be empty")
        if not self.enemy_counts:
            raise ValueError("enemy_counts must not be empty")
        if not self.levels:
            raise ValueError("levels must not be empty")
        if self.layout_variants < 1:
            raise ValueError("layout_variants must be >= 1")
        unknown = [rule for rule in self.spawn_rules if rule not in SPAWN_RULES]
        if unknown:
            raise ValueError(f"unknown spawn rules: {unknown}")

    def size(self) -> int:
        """Distinct condition cells (excluding seeds and spawn detail)."""
        return (
            len(self.maps)
            * max(1, len(self.lightings))
            * max(1, len(self.scenarios))
            * len(self.enemy_counts)
            * len(self.levels)
            * self.layout_variants
        )

    def episode_plan(self, index: int) -> EpisodePlan:
        """Episode ``index`` of the stream. Pure function; no state."""
        pick = lambda values, salt: values[_derive(self.master_seed, index, salt) % len(values)]  # noqa: E731
        map_id = pick(list(self.maps), "map")
        lighting = pick(list(self.lightings), "light") if self.lightings else ""
        scenario = pick(list(self.scenarios), "scenario") if self.scenarios else ""
        enemy_count = int(pick(list(self.enemy_counts), "enemies"))
        level = int(pick(list(self.levels), "level"))
        layout_variant = _derive(self.master_seed, index, "layout") % self.layout_variants
        rule = pick(list(self.spawn_rules), "spawn_rule")
        condition = Condition(
            map_id=map_id,
            lighting=lighting,
            scenario=scenario,
            enemy_count=enemy_count,
            level=level,
            seed=_derive(self.master_seed, index, "seed") % (2**31 - 1),
        )
        spawn = EnemySpawnPlan(
            count=enemy_count,
            rule=rule,
            seeds=[
                _derive(self.master_seed, index, f"spawn{slot}") % (2**31 - 1)
                for slot in range(enemy_count)
            ],
        )
        return EpisodePlan(
            index=index,
            condition=condition,
            layout_seed=layout_variant,
            spawn=spawn,
        )

    def episode_plans(self, count: int, start: int = 0) -> list[EpisodePlan]:
        return [self.episode_plan(start + offset) for offset in range(count)]

    def stream_index(self, environment_index: int, ordinal: int) -> int:
        """Global stream index of environment ``k``'s ``ordinal``-th episode.

        Named so the training pipeline (which draws plans one at a time as
        episodes start) and ``stream_for_environment`` (which materializes a
        batch) never drift apart.
        """
        if environment_index < 0 or ordinal < 0:
            raise ValueError("environment_index and ordinal must be >= 0")
        return environment_index + ordinal * STREAM_STRIDE

    def stream_for_environment(self, environment_index: int, count: int) -> list[EpisodePlan]:
        """Per-environment episode stream for parallel training.

        Environment ``k`` plays episodes ``k, k + 1000003, k + 2*1000003,
        ...`` of the global stream. The stride is a fixed large prime rather
        than the environment count on purpose: this method does not need to
        know how many environments exist, and a fixed stride means changing
        the environment count never re-maps which episodes a given
        environment has already played. Streams stay disjoint (different
        residues mod the stride), each is still a pure function of the
        master seed, and because ``episode_plan`` is hash-derived the
        sampled indices remain an unbiased draw from the distribution.
        """
        if environment_index < 0:
            raise ValueError("environment_index must be >= 0")
        return [
            self.episode_plan(self.stream_index(environment_index, offset))
            for offset in range(count)
        ]

    def coverage(self, count: int) -> dict[str, Any]:
        """How much of the distribution ``count`` episodes actually touch."""
        plans = self.episode_plans(count)
        maps = {plan.condition.map_id for plan in plans}
        lightings = {plan.condition.lighting for plan in plans}
        counts = {plan.condition.enemy_count for plan in plans}
        variants = {(plan.condition.map_id, plan.layout_seed) for plan in plans}
        return {
            "episodes": count,
            "distinct_maps": len(maps),
            "map_coverage": len(maps) / len(self.maps),
            "distinct_lightings": len(lightings),
            "distinct_enemy_counts": len(counts),
            "distinct_layout_variants": len(variants),
            "distinct_conditions": len({plan.key for plan in plans}),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maps"] = list(self.maps)
        payload["lightings"] = list(self.lightings)
        payload["scenarios"] = list(self.scenarios)
        payload["enemy_counts"] = list(self.enemy_counts)
        payload["levels"] = list(self.levels)
        payload["spawn_rules"] = list(self.spawn_rules)
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrainingDistribution:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


class DistributionRunTracker:
    """Per-condition performance over a randomized run.

    Thin wrapper over ``ConditionTracker`` that also remembers which plan
    produced each episode, so a weak condition can be re-run exactly.
    """

    def __init__(self, window: int = 100) -> None:
        self.conditions = ConditionTracker(window=window)
        self._plans: dict[str, list[int]] = {}

    def record(self, plan: EpisodePlan, reward: float, won: bool, steps: int) -> None:
        self.conditions.record(plan.condition, reward, won, steps)
        self._plans.setdefault(plan.key, []).append(plan.index)

    def episodes_for(self, condition_key: str) -> list[int]:
        return list(self._plans.get(condition_key, []))

    def report(self, worst: int = 5) -> dict[str, Any]:
        return {
            "overall": self.conditions.overall(),
            "by_condition": self.conditions.summaries(),
            "worst": self.conditions.worst_conditions(worst),
            "reproduce": {
                row["condition"]: self.episodes_for(row["condition"])[:5]
                for row in self.conditions.worst_conditions(worst)
            },
        }


def sample_conditions(distribution: TrainingDistribution, count: int, seed: int | None = None):
    """Convenience: plain ``Condition`` objects from the distribution.

    ``seed`` overrides the distribution's master seed for a one-off draw
    without mutating the distribution (which a training run shares).
    """
    if seed is None:
        return [plan.condition for plan in distribution.episode_plans(count)]
    clone = TrainingDistribution(**{**distribution.to_dict(), "master_seed": int(seed)})
    return [plan.condition for plan in clone.episode_plans(count)]


def unused_rng_warning(rng: random.Random) -> None:  # pragma: no cover - documentation hook
    """Placeholder documenting why there is no shared RNG here.

    A shared ``random.Random`` would make ``episode_plan(137)`` depend on
    how many episodes were drawn before it, which breaks resume, breaks
    parallel environments and breaks replay. Everything in this module
    derives from a digest instead.
    """
    raise NotImplementedError("the training distribution intentionally holds no RNG state")
