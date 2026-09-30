"""Curriculum integration: stages, gating and the episode stream.

Phase 12. ``auto_curriculum.AutoCurriculum`` already decides *when* to move
a level (rolling window, minimum episodes, cooldown, hysteresis). What was
missing is *what a level means* now that the platform has a randomized
distribution, multi-policy matchups, exploration and teams.

:data:`STAGES` is that table. One row per curriculum level, mirroring
``CurriculumConfig.Level`` in ``scripts/core/curriculum_config.gd``, and a
test asserts the two stay in step. Each row declares:

* which perception/world systems the level exercises (documentation of the
  engine's own gating, not a second implementation of it);
* the enemy counts and maps the level draws from;
* the promotion metric and threshold;
* whether the level uses the randomized distribution at all.

:class:`CurriculumDirector` glues the three pieces together: it owns an
``AutoCurriculum``, turns the current level into a
``randomization.TrainingDistribution``, and hands out reproducible
:class:`~sandboxai.randomization.EpisodePlan` objects. Feeding an episode
result back may promote or demote, which reconfigures the distribution for
the *next* episode — never the one in flight, because changing the setup
mid-episode is how a curriculum corrupts its own statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Sequence

from .auto_curriculum import AutoCurriculum, CurriculumSchedule
from .conditions import LIGHTING_IDS, MAP_IDS, Condition
from .randomization import EpisodePlan, TrainingDistribution

## Promotion metrics a stage may be gated on. "win_rate" is the default;
## exploration stages are gated on coverage instead, because a Map
## Analyzer episode has no opponent to beat.
PROMOTION_METRICS: tuple[str, ...] = ("win_rate", "coverage", "survival_rate")

## Engine-semantics mirrors, needed to predict what an episode plan
## actually runs as. Both mirror CurriculumConfig in
## scripts/core/curriculum_config.gd and are drift-checked by tests:
##
## ``WORLD_MIN_LEVEL``
##     Level from which the engine keeps world geometry (Level 5,
##     OBSTACLES_COVER). Below it, map/scenario/lighting ids are recorded
##     in the sampled plan for provenance but are NOT applied: levels 1-4
##     must stay bit-for-bit the legacy obstacle-free behavior, and
##     handing them a map would silently switch the reset path.
## ``MULTI_ENEMY_MIN``
##     CurriculumConfig.MULTIPLE_ENEMIES_MIN_COUNT: once at level 4-10 the
##     engine never creates fewer than this many enemies.
## ``HANDLING_MIN_LEVEL``
##     CurriculumConfig.weapon_handling_enabled() / hit_zones_enabled():
##     the level from which recoil, bloom, magazines, reloads, fire modes
##     and head/body hit zones are switched on. Below it the weapon is the
##     original cooldown-gated hitscan, so levels 1-4 stay bit-for-bit
##     comparable with checkpoints trained before the handling layer
##     existed.
WORLD_MIN_LEVEL: int = 5
MULTI_ENEMY_MIN: int = 3
MULTI_ENEMY_MAX_LEVEL: int = 10
HANDLING_MIN_LEVEL: int = 5


def weapon_handling_enabled(level: int) -> bool:
    """Whether the engine arms the weapon handling layer at ``level``.

    Mirrors ``CurriculumConfig.weapon_handling_enabled``. Reports and
    dashboards need this to explain why a recoil or reload metric is
    flat: on levels 1-4 it is not a policy failure, the system is off.
    """
    return int(level) >= HANDLING_MIN_LEVEL


def hit_zones_enabled(level: int) -> bool:
    """Whether head/body hit zones are resolved at ``level``.

    Mirrors ``CurriculumConfig.hit_zones_enabled``. Shares a threshold
    with the handling layer: both are the same step up in weapon fidelity
    and splitting them would create a level where headshots exist but
    recoil does not.
    """
    return int(level) >= HANDLING_MIN_LEVEL


def applied_condition(condition: Condition) -> Condition:
    """The condition the engine actually runs for a sampled one.

    The training distribution samples *requested* conditions; the engine
    then applies its own level semantics on top. Reporting (per-condition
    metrics, replay headers, generalization splits) must describe what
    ran, not what was sampled — otherwise a level 3 episode would be
    filed under a map it never loaded, and a 1-enemy request at level 5
    would be filed as 1 enemy while 3 spawned.

    Rules mirrored from CurriculumConfig / EnvironmentCore:

    * levels {WORLD_MIN_LEVEL}..10 apply map/lighting/scenario ids as
      given; below that those ids are dropped (legacy reset path);
    * at levels 4..{MULTI_ENEMY_MAX_LEVEL} the enemy count is raised to
      at least {MULTI_ENEMY_MIN} (``effective_enemy_count``);
    * seed and level always apply verbatim.
    """
    level = condition.level
    enemy_count = condition.enemy_count
    if 4 <= level <= MULTI_ENEMY_MAX_LEVEL:
        enemy_count = max(MULTI_ENEMY_MIN, enemy_count)
    if level < WORLD_MIN_LEVEL:
        return Condition(
            map_id="",
            lighting="",
            scenario="",
            enemy_count=enemy_count,
            level=level,
            seed=condition.seed,
        )
    if enemy_count != condition.enemy_count:
        return replace(condition, enemy_count=enemy_count)
    return condition


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum level, as configuration."""

    level: int
    name: str
    focus: str
    ## Systems the level exercises. Descriptive: the engine's
    ## CurriculumConfig is the authority, this is the Python-side mirror
    ## used for reports and for choosing the episode distribution.
    systems: tuple[str, ...] = ()
    maps: tuple[str, ...] = ()
    lightings: tuple[str, ...] = ("normal",)
    scenarios: tuple[str, ...] = ()
    enemy_counts: tuple[int, ...] = (1,)
    randomized: bool = False
    exploration: bool = False
    self_play: bool = False
    promotion_metric: str = "win_rate"
    ## An episode counts as a success when its promotion metric reaches
    ## this value. For "win_rate" that is simply "it won".
    success_at: float = 0.5
    promote_at: float = 0.70
    demote_at: float = 0.25
    ## Minimum episodes at this level before it may change. Later levels
    ## are noisier, so they need more evidence.
    min_episodes: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


## The progression. Levels 1-11 mirror CurriculumConfig.Level exactly.
STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        level=1,
        name="movement_and_aim",
        focus="move and point the crosshair at a stationary target",
        systems=("movement", "aim"),
        maps=("open_field",),
        enemy_counts=(1,),
        promote_at=0.80,
        min_episodes=20,
    ),
    CurriculumStage(
        level=2,
        name="moving_targets",
        focus="track a target that moves",
        systems=("movement", "aim", "tracking"),
        maps=("open_field", "training_yard"),
        enemy_counts=(1,),
        promote_at=0.75,
        min_episodes=25,
    ),
    CurriculumStage(
        level=3,
        name="basic_combat",
        focus="fight an opponent that shoots back",
        systems=("movement", "aim", "combat", "survival"),
        maps=("open_field", "training_yard"),
        enemy_counts=(1,),
    ),
    CurriculumStage(
        level=4,
        name="multi_enemy_combat",
        focus="target selection under several simultaneous threats",
        systems=("combat", "target_selection", "awareness"),
        maps=("open_field", "training_yard"),
        enemy_counts=(3,),
        promote_at=0.65,
        min_episodes=40,
    ),
    CurriculumStage(
        level=5,
        name="cover",
        focus="use geometry, and control a weapon that now fights back",
        systems=(
            "combat",
            "positioning",
            "cover",
            "navigation",
            "weapon_handling",
            "hit_zones",
        ),
        maps=("cover_field", "pillar_hall", "two_rooms", "combat_complex", "crossfire_lab"),
        scenarios=("cover_fight",),
        enemy_counts=(1, 2),
        promote_at=0.65,
        min_episodes=40,
    ),
    CurriculumStage(
        level=6,
        name="fov_and_occlusion",
        focus="the observation stops being ground truth",
        systems=("perception", "fov", "line_of_sight", "positioning"),
        maps=("blind_corner", "compound", "two_rooms", "combat_complex"),
        scenarios=("corner_fight",),
        enemy_counts=(1, 2),
        promote_at=0.60,
        min_episodes=50,
    ),
    CurriculumStage(
        level=7,
        name="sound",
        focus="act on information you can only hear",
        systems=("perception", "sound", "awareness"),
        maps=("echo_maze", "two_rooms", "compound"),
        scenarios=("sound_only",),
        enemy_counts=(1, 2),
        lightings=("normal", "low_light", "night"),
        promote_at=0.55,
        min_episodes=50,
    ),
    CurriculumStage(
        level=8,
        name="memory",
        focus="keep tracking a contact you can no longer see",
        systems=("perception", "memory", "awareness", "search"),
        maps=("blind_corner", "echo_maze", "long_corridor"),
        scenarios=("target_disappears",),
        enemy_counts=(1, 2),
        lightings=("normal", "low_light", "fog"),
        promote_at=0.55,
        min_episodes=50,
    ),
    CurriculumStage(
        level=9,
        name="navigation_and_vertical",
        focus="cross complex geometry, use height",
        systems=("navigation", "movement", "vertical", "positioning"),
        maps=("catwalks", "compound", "ambush_alley", "combat_complex", "crossfire_lab"),
        scenarios=("vertical_encounter",),
        enemy_counts=(1, 2, 3),
        promote_at=0.55,
        min_episodes=50,
    ),
    CurriculumStage(
        level=10,
        name="randomized_environments",
        focus="generalize across maps, lighting and spawn configurations",
        systems=("perception", "memory", "sound", "navigation", "generalization"),
        maps=tuple(MAP_IDS),
        lightings=tuple(LIGHTING_IDS),
        enemy_counts=(1, 2, 3, 5),
        randomized=True,
        promote_at=0.55,
        demote_at=0.20,
        min_episodes=80,
    ),
    CurriculumStage(
        level=11,
        name="self_play",
        focus="fight another learned policy",
        systems=("perception", "memory", "sound", "navigation", "self_play"),
        maps=tuple(MAP_IDS),
        lightings=tuple(LIGHTING_IDS),
        enemy_counts=(1,),
        randomized=True,
        self_play=True,
        promote_at=0.60,
        demote_at=0.20,
        min_episodes=100,
    ),
)

## Optional, off the main line: the Map Analyzer exploration stage. It is
## not part of the promotion ladder (there is nothing to beat), so it is
## selected explicitly rather than reached by promotion.
EXPLORATION_STAGE: CurriculumStage = CurriculumStage(
    level=10,
    name="map_analyzer",
    focus="cover an unknown map using perception alone",
    systems=("perception", "exploration", "spatial_memory", "navigation"),
    maps=tuple(MAP_IDS),
    lightings=tuple(LIGHTING_IDS),
    enemy_counts=(0,),
    randomized=True,
    exploration=True,
    promotion_metric="coverage",
    success_at=0.70,
    promote_at=0.85,
    demote_at=0.30,
    min_episodes=30,
)

STAGES_BY_LEVEL: dict[int, CurriculumStage] = {stage.level: stage for stage in STAGES}


def stage_for(level: int) -> CurriculumStage:
    """The stage for a level, clamped into the declared range."""
    if not STAGES_BY_LEVEL:  # pragma: no cover - defensive
        raise RuntimeError("no curriculum stages declared")
    clamped = max(min(int(level), max(STAGES_BY_LEVEL)), min(STAGES_BY_LEVEL))
    return STAGES_BY_LEVEL[clamped]


## Highest level standard (single-policy) training runs. Level 11 is the
## self-play stage: its "episodes" are two-policy league matches, not
## distribution-sampled episodes, so it can never be part of the standard
## training stream or the seen-condition evaluation union.
TRAINABLE_MAX_LEVEL: int = 10


def evaluation_levels() -> list[int]:
    """Every curriculum level the runs' evaluation distribution spans."""
    return sorted({stage.level for stage in STAGES if stage.level <= TRAINABLE_MAX_LEVEL})


def evaluation_distribution(
    master_seed: int = 1234, layout_variants: int = 8
) -> TrainingDistribution:
    """The frozen per-run 'seen' evaluation distribution: union of the ladder.

    Drawn with an evaluation-only master seed (``config.seed +
    EVAL_MASTER_SEED_SALT`` in pipeline.py), so the episodes are provably
    never training episodes, and the plan LIST is identical at every
    checkpoint: results are comparable across checkpoints. Difficulty is
    not a single point -- the condition key (L{level}-bucketed rows) keeps
    each level's slice visible instead of collapsing it into one scalar.
    """
    maps = sorted({map_id for stage in STAGES for map_id in stage.maps}) or [MAP_IDS[0]]
    lightings = sorted({lighting for stage in STAGES for lighting in stage.lightings})
    scenarios = sorted({scenario for stage in STAGES for scenario in stage.scenarios})
    enemy_counts = sorted({count for stage in STAGES for count in stage.enemy_counts}) or [1]
    return TrainingDistribution(
        maps=maps,
        lightings=lightings,
        scenarios=scenarios,
        enemy_counts=enemy_counts,
        levels=evaluation_levels(),
        master_seed=master_seed,
        layout_variants=layout_variants,
    )


def distribution_for(
    stage: CurriculumStage, master_seed: int = 1234, layout_variants: int = 8
) -> TrainingDistribution:
    """Builds the episode distribution a stage draws from."""
    return TrainingDistribution(
        maps=list(stage.maps) or [MAP_IDS[0]],
        lightings=list(stage.lightings),
        scenarios=list(stage.scenarios),
        enemy_counts=[count for count in stage.enemy_counts] or [1],
        levels=[stage.level],
        master_seed=master_seed,
        layout_variants=layout_variants if stage.randomized else 1,
    )


@dataclass
class EpisodeOutcome:
    """What the director needs to know about a finished episode."""

    won: bool = False
    reward: float = 0.0
    steps: int = 0
    coverage: float = 0.0
    survived: bool = True

    def metric(self, name: str) -> float:
        if name == "win_rate":
            return 1.0 if self.won else 0.0
        if name == "coverage":
            return float(self.coverage)
        if name == "survival_rate":
            return 1.0 if self.survived else 0.0
        raise ValueError(f"unknown promotion metric: {name!r}")


class CurriculumDirector:
    """Stage table + automatic curriculum + reproducible episode stream.

    One object a training loop can ask two questions of:

        plan = director.next_episode()      # what to run
        change = director.record(outcome)   # what happened, and did we move

    Level changes take effect for the *next* ``next_episode`` call.
    """

    def __init__(
        self,
        start_level: int = 1,
        master_seed: int = 1234,
        schedule: CurriculumSchedule | None = None,
        exploration: bool = False,
    ) -> None:
        self.master_seed = int(master_seed)
        self.exploration = bool(exploration)
        stage = EXPLORATION_STAGE if exploration else stage_for(start_level)
        self.schedule = schedule or CurriculumSchedule(
            start_level=stage.level,
            # Level 11 (self-play) is above AutoCurriculum's default
            # combat ceiling; the integrated ladder explicitly reaches it.
            max_level=stage.level if exploration else max(STAGES_BY_LEVEL),
            # The exploration stage is off the ladder: there is no level 9
            # of "map analysis" to fall back to, so it never moves.
            allow_demotion=not exploration,
            promote_threshold=stage.promote_at,
            demote_threshold=stage.demote_at,
            min_episodes_per_level=stage.min_episodes,
        )
        self.auto = AutoCurriculum(self.schedule)
        self._episode_index = 0
        self._stage = stage
        self._distribution = distribution_for(stage, self._stage_seed(stage.level))
        self.changes: list[dict[str, Any]] = []

    # -- state -------------------------------------------------------------

    def _stage_seed(self, level: int) -> int:
        # Each level gets its own stream, so re-entering level 6 after a
        # demotion does not replay exactly the episodes that were just
        # failed.
        return self.master_seed + level * 7919

    @property
    def level(self) -> int:
        return self.auto.level

    @property
    def stage(self) -> CurriculumStage:
        return self._stage

    @property
    def distribution(self) -> TrainingDistribution:
        return self._distribution

    def _sync_stage(self) -> None:
        stage = EXPLORATION_STAGE if self.exploration else stage_for(self.auto.level)
        if stage is self._stage:
            return
        self._stage = stage
        self._distribution = distribution_for(stage, self._stage_seed(stage.level))
        self.schedule.promote_threshold = stage.promote_at
        self.schedule.demote_threshold = stage.demote_at
        self.schedule.min_episodes_per_level = stage.min_episodes

    # -- episode stream ----------------------------------------------------

    def next_episode(self) -> EpisodePlan:
        """The next episode to run. Reproducible from (seed, level, index)."""
        plan = self._distribution.episode_plan(self._episode_index)
        self._episode_index += 1
        return plan

    def peek(self, offset: int = 0) -> EpisodePlan:
        """The plan that ``next_episode`` would return, without consuming."""
        return self._distribution.episode_plan(self._episode_index + offset)

    def record(self, outcome: EpisodeOutcome) -> dict[str, Any] | None:
        """Records a finished episode; may promote or demote."""
        value = outcome.metric(self._stage.promotion_metric)
        change = self.auto.record(value >= self._stage.success_at, outcome.reward)
        if change is not None:
            change["stage_from"] = self._stage.name
            self._sync_stage()
            change["stage_to"] = self._stage.name
            self.changes.append(change)
        return change

    def snapshot(self) -> dict[str, Any]:
        payload = self.auto.snapshot()
        payload.update(
            {
                "stage": self._stage.name,
                "focus": self._stage.focus,
                "systems": list(self._stage.systems),
                "promotion_metric": self._stage.promotion_metric,
                "episodes_drawn": self._episode_index,
                "maps": len(self._stage.maps),
                "randomized": self._stage.randomized,
                "self_play": self._stage.self_play,
                "exploration": self._stage.exploration,
            }
        )
        return payload

    def history(self) -> list[dict[str, Any]]:
        return list(self.changes)


def describe_progression() -> list[dict[str, Any]]:
    """The full ladder, as data (for docs, the Control Center and tests)."""
    return [stage.to_dict() for stage in STAGES]


def format_progression(stages: Sequence[CurriculumStage] = STAGES) -> str:
    lines = [
        f"{'lvl':>3}  {'stage':<26}{'enemies':>9}{'maps':>6}{'metric':>14}{'promote':>9}",
        "-" * 72,
    ]
    for stage in stages:
        lines.append(
            f"{stage.level:>3}  {stage.name:<26}"
            f"{','.join(str(count) for count in stage.enemy_counts):>9}"
            f"{len(stage.maps):>6}{stage.promotion_metric:>14}{stage.promote_at:>9.0%}"
        )
    return "\n".join(lines) + "\n"
