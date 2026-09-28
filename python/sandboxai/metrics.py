"""Research / skill metrics: what the policy actually did, per episode.

Phase 3. These are **diagnostics**, not rewards. The distinction is the
whole point of the module and is enforced structurally: nothing here is
imported by the reward path, and every value is derived *after* the fact
from information the agent itself had (its observation vector and the
step's event dictionary). Turning "cover usage" into a reward term would
teach the policy to hug walls; measuring it tells you whether it learned
to, which is the question a research platform should answer.

Eight categories, mirroring the way an FPS coach would break a round down:

``aim``          shots, hits, accuracy, aim error, target switching
``reaction``     detection / confirmation / first-shot latency
``awareness``    contacts seen and lost, memory and sound actually used
``positioning``  cover, exposure, distance management, corner usage
``movement``     path efficiency, navigation failures, speed, jumps, stuck
``combat``       damage in/out, kills, deaths, wasted shots
``survival``     time alive, damage avoided, escapes
``exploration``  coverage, discovery rate, efficiency, time-to-coverage

Every metric is computed from a per-tick :class:`StepSample`. The caller
fills one in from what it already has — the observation vector (read by
NAME through ``contract.observation_value``, never by a hardcoded index)
and ``info["events"]`` from the Godot step. That keeps the metrics layer
honest: if the agent could not perceive it, the metric cannot see it
either. The one exception is the explicitly named ``ground_truth`` block,
which is used for nothing except reporting and is never mixed into a
per-category score.

Aggregation supports grouping by episode, policy, map, scenario, lighting,
enemy count, curriculum level and seed, and exports to JSON and CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contract import observation_slice, observation_value

## Category names, in report order.
CATEGORIES: tuple[str, ...] = (
    "aim",
    "reaction",
    "awareness",
    "positioning",
    "movement",
    "combat",
    "survival",
    "exploration",
)

## Labels an episode can be grouped by. Every one of these is *experiment
## configuration*, not something the policy observes.
GROUP_KEYS: tuple[str, ...] = (
    "policy_id",
    "map_id",
    "scenario",
    "lighting",
    "enemy_count",
    "curriculum_level",
    "seed",
    "episode_id",
)

## An exposure longer than this while a contact is visible counts as "bad
## exposure": standing in the open in front of something that can shoot
## back. Seconds.
BAD_EXPOSURE_SECONDS: float = 1.5

## Below this normalized distance the agent is inside brawling range; above
## the far threshold it is disengaged. Used only to describe distance
## management, never to prescribe it.
CLOSE_DISTANCE_NORM: float = 0.15
FAR_DISTANCE_NORM: float = 0.5

## Movement slower than this fraction of full speed, while an action asked
## for movement, counts as "stuck".
STUCK_SPEED_NORM: float = 0.05


@dataclass
class StepSample:
    """One tick of evidence, as seen by the agent.

    Only ``observation`` and ``events`` are required; everything else has
    a neutral default so a caller with less information still produces
    valid (if sparser) metrics rather than wrong ones.
    """

    observation: Sequence[float]
    events: dict[str, Any] = field(default_factory=dict)
    action: Sequence[int] = field(default_factory=tuple)
    dt: float = 1.0 / 60.0
    ## Optional extras the environment can provide cheaply.
    navigation_failed: bool = False
    target_id: int = -1
    ## Strictly for reporting. Never enters a category score.
    ground_truth: dict[str, Any] = field(default_factory=dict)


def _finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(value) else default


class EpisodeMetrics:
    """Accumulates one episode's diagnostics from per-tick samples."""

    def __init__(self, labels: dict[str, Any] | None = None) -> None:
        self.labels: dict[str, Any] = dict(labels or {})
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, labels: dict[str, Any] | None = None) -> None:
        """Clears every counter. Called between episodes.

        A metrics object that is not reset is the classic source of
        "impossible" numbers (accuracy above 1, survival longer than the
        episode), so the reset path is tested explicitly.
        """
        if labels is not None:
            self.labels = dict(labels)
        self.ticks: int = 0
        self.elapsed: float = 0.0

        # aim
        self.shots: int = 0
        self.hits: int = 0
        self._aim_error_sum: float = 0.0
        self._aim_error_samples: int = 0
        self._shot_aim_error_sum: float = 0.0
        self.target_switches: int = 0
        self._last_target: int = -1

        # reaction
        self._first_visible_tick: int = -1
        self._first_confirm_tick: int = -1
        self._first_shot_tick: int = -1
        self._contact_started_tick: int = -1
        self.detection_latency: float = -1.0
        self.confirmation_latency: float = -1.0
        self.shot_latency: float = -1.0

        # awareness
        self.visible_contacts: int = 0
        self.lost_contacts: int = 0
        self.memory_ticks: int = 0
        self.sound_ticks: int = 0
        self.unknown_area_ticks: int = 0
        self._was_visible: bool = False
        self._memory_then_visible: int = 0
        self._memory_streak: bool = False
        self._sound_then_visible: int = 0
        self._sound_streak: bool = False

        # positioning
        self.cover_ticks: int = 0
        self.exposed_ticks: int = 0
        self.bad_exposure_time: float = 0.0
        self._exposure_streak: float = 0.0
        self.corner_ticks: int = 0
        self._distance_sum: float = 0.0
        self._distance_samples: int = 0
        self.close_ticks: int = 0
        self.far_ticks: int = 0

        # movement
        self._speed_sum: float = 0.0
        self.navigation_failures: int = 0
        self.jumps: int = 0
        self.stuck_time: float = 0.0
        self._path_length: float = 0.0
        self._start_position: list[float] | None = None
        self._last_position: list[float] | None = None
        self._end_position: list[float] | None = None

        # combat
        self.damage_dealt: float = 0.0
        self.damage_received: float = 0.0
        self.kills: int = 0
        self.deaths: int = 0
        self.unnecessary_shots: int = 0

        # survival
        self.alive_time: float = 0.0
        self.damage_free_time: float = 0.0
        self._damage_free_streak: float = 0.0
        self.escapes: int = 0
        self._was_exposed: bool = False

        # exploration
        self.explored_fraction: float = 0.0
        self.new_area_events: int = 0
        self._last_explored: float = 0.0
        self.time_to_half_coverage: float = -1.0
        self.time_to_target_coverage: float = -1.0

        self.result: dict[str, Any] = {}

    # -- accumulation ------------------------------------------------------

    def record(self, sample: StepSample) -> None:
        observation = sample.observation
        dt = float(sample.dt)
        self.ticks += 1
        self.elapsed += dt
        events = sample.events

        visible = observation_value(observation, "primary_enemy_visible") > 0.5
        in_fov = observation_value(observation, "primary_enemy_in_fov") > 0.5
        los = observation_value(observation, "primary_enemy_los_clear") > 0.5
        confidence = observation_value(observation, "primary_enemy_confidence")
        bearing = abs(observation_value(observation, "primary_enemy_bearing_norm"))
        distance = observation_value(observation, "primary_enemy_distance_norm")
        in_cover = observation_value(observation, "agent_in_cover") > 0.5
        clearance = observation_value(observation, "agent_forward_clearance_norm")
        sound_loudness = observation_value(observation, "last_sound_loudness")
        from_memory = (not visible) and confidence > 0.05
        alive = bool(events.get("alive", True))

        self._record_aim(events, bearing, visible, sample)
        self._record_reaction(visible, los, events)
        self._record_awareness(observation, visible, from_memory, sound_loudness)
        self._record_positioning(in_cover, visible, in_fov, distance, clearance, dt)
        self._record_movement(observation, sample, dt)
        self._record_combat(events)
        self._record_survival(events, alive, in_cover, visible, dt)
        self._record_exploration(observation)

    def _record_aim(
        self, events: dict[str, Any], bearing: float, visible: bool, sample: StepSample
    ) -> None:
        if visible:
            self._aim_error_sum += bearing
            self._aim_error_samples += 1
        if events.get("shot_fired"):
            self.shots += 1
            self._shot_aim_error_sum += bearing
            if events.get("hit"):
                self.hits += 1
        target = int(sample.target_id)
        if target >= 0:
            if self._last_target >= 0 and target != self._last_target:
                self.target_switches += 1
            self._last_target = target

    def _record_reaction(self, visible: bool, los: bool, events: dict[str, Any]) -> None:
        tick = self.ticks - 1
        if visible:
            if self._contact_started_tick < 0:
                self._contact_started_tick = tick
            if self._first_visible_tick < 0:
                self._first_visible_tick = tick
            if self._first_confirm_tick < 0 and los:
                self._first_confirm_tick = tick
        if events.get("shot_fired") and self._first_shot_tick < 0:
            self._first_shot_tick = tick

    def _record_awareness(
        self,
        observation: Sequence[float],
        visible: bool,
        from_memory: bool,
        sound_loudness: float,
    ) -> None:
        if visible and not self._was_visible:
            self.visible_contacts += 1
            # A sighting that follows a memory-only or sound-only stretch
            # is evidence the non-visual information was USED, not just
            # present. That is the only honest way to measure it from the
            # outside.
            if self._memory_streak:
                self._memory_then_visible += 1
            if self._sound_streak:
                self._sound_then_visible += 1
            self._memory_streak = False
            self._sound_streak = False
        if self._was_visible and not visible:
            self.lost_contacts += 1
        if from_memory:
            self.memory_ticks += 1
            self._memory_streak = True
        if sound_loudness > 0.05:
            self.sound_ticks += 1
            self._sound_streak = True
        if observation_value(observation, "current_area_known") < 0.5:
            self.unknown_area_ticks += 1
        self._was_visible = visible

    def _record_positioning(
        self,
        in_cover: bool,
        visible: bool,
        in_fov: bool,
        distance: float,
        clearance: float,
        dt: float,
    ) -> None:
        if in_cover:
            self.cover_ticks += 1
            self._exposure_streak = 0.0
        else:
            self.exposed_ticks += 1
            if visible or in_fov:
                self._exposure_streak += dt
                if self._exposure_streak > BAD_EXPOSURE_SECONDS:
                    self.bad_exposure_time += dt
            else:
                self._exposure_streak = 0.0
        # A short forward clearance while not in cover is the signature of
        # standing at a corner/doorway rather than in the open.
        if 0.0 < clearance < 0.25 and not in_cover:
            self.corner_ticks += 1
        if distance > 0.0:
            self._distance_sum += distance
            self._distance_samples += 1
            if distance < CLOSE_DISTANCE_NORM:
                self.close_ticks += 1
            elif distance > FAR_DISTANCE_NORM:
                self.far_ticks += 1

    def _record_movement(self, observation: Sequence[float], sample: StepSample, dt: float) -> None:
        # agent_velocity_norm / agent_position_norm are 3-wide fields, so
        # they are read through the slice helper rather than the scalar one.
        velocity = observation_slice(observation, "agent_velocity_norm")
        position = observation_slice(observation, "agent_position_norm")
        speed = math.sqrt(sum(component * component for component in velocity))
        self._speed_sum += speed

        if self._start_position is None:
            self._start_position = list(position)
        if self._last_position is not None:
            self._path_length += math.dist(self._last_position, position)
        self._last_position = list(position)
        self._end_position = list(position)

        action = list(sample.action)
        wants_movement = bool(action) and (
            (len(action) > 0 and action[0] != 1) or (len(action) > 1 and action[1] != 1)
        )
        if wants_movement and speed < STUCK_SPEED_NORM:
            self.stuck_time += dt
        if len(action) > 5 and action[5] == 1:
            self.jumps += 1
        if sample.navigation_failed:
            self.navigation_failures += 1

    def _record_combat(self, events: dict[str, Any]) -> None:
        self.damage_dealt += _finite(float(events.get("damage_dealt", 0.0)))
        self.damage_received += _finite(float(events.get("damage_taken", 0.0)))
        if events.get("kill"):
            self.kills += 1
        if events.get("died"):
            self.deaths += 1
        if events.get("useless_shot"):
            self.unnecessary_shots += 1

    def _record_survival(
        self,
        events: dict[str, Any],
        alive: bool,
        in_cover: bool,
        visible: bool,
        dt: float,
    ) -> None:
        if alive:
            self.alive_time += dt
        took_damage = float(events.get("damage_taken", 0.0)) > 0.0
        if took_damage:
            self._damage_free_streak = 0.0
        else:
            self._damage_free_streak += dt
            self.damage_free_time += dt
        exposed_under_fire = visible and not in_cover
        if self._was_exposed and in_cover:
            self.escapes += 1
        self._was_exposed = exposed_under_fire

    def _record_exploration(self, observation: Sequence[float]) -> None:
        explored = observation_value(observation, "explored_fraction")
        if explored > self._last_explored + 1e-6:
            self.new_area_events += 1
        self._last_explored = max(self._last_explored, explored)
        self.explored_fraction = self._last_explored
        if self.time_to_half_coverage < 0.0 and self.explored_fraction >= 0.5:
            self.time_to_half_coverage = self.elapsed
        if self.time_to_target_coverage < 0.0 and self.explored_fraction >= 0.9:
            self.time_to_target_coverage = self.elapsed

    def finish(self, result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Closes the episode and returns its summary."""
        self.result = dict(result or {})
        return self.summary()

    # -- reporting ---------------------------------------------------------

    def _dt(self) -> float:
        return self.elapsed / self.ticks if self.ticks else 0.0

    def aim(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "hits": self.hits,
            "accuracy": self.hits / self.shots if self.shots else 0.0,
            "mean_aim_error": (
                self._aim_error_sum / self._aim_error_samples if self._aim_error_samples else 0.0
            ),
            "mean_shot_aim_error": self._shot_aim_error_sum / self.shots if self.shots else 0.0,
            "target_switches": self.target_switches,
        }

    def reaction(self) -> dict[str, Any]:
        dt = self._dt()
        detection = self._first_visible_tick * dt if self._first_visible_tick >= 0 else -1.0
        confirmation = (
            (self._first_confirm_tick - self._first_visible_tick) * dt
            if self._first_confirm_tick >= 0 and self._first_visible_tick >= 0
            else -1.0
        )
        shot = (
            (self._first_shot_tick - self._first_visible_tick) * dt
            if self._first_shot_tick >= 0 and self._first_visible_tick >= 0
            else -1.0
        )
        return {
            # -1 means "never happened"; a 0 would claim an instant reaction.
            "detection_latency": detection,
            "confirmation_latency": confirmation,
            "shot_latency": shot,
        }

    def awareness(self) -> dict[str, Any]:
        ticks = max(1, self.ticks)
        return {
            "visible_contacts": self.visible_contacts,
            "lost_contacts": self.lost_contacts,
            "memory_tick_fraction": self.memory_ticks / ticks,
            "memory_reacquisitions": self._memory_then_visible,
            "sound_tick_fraction": self.sound_ticks / ticks,
            "sound_reacquisitions": self._sound_then_visible,
            "unknown_area_fraction": self.unknown_area_ticks / ticks,
        }

    def positioning(self) -> dict[str, Any]:
        ticks = max(1, self.ticks)
        return {
            "cover_fraction": self.cover_ticks / ticks,
            "exposure_time": self.exposed_ticks * self._dt(),
            "bad_exposure_time": self.bad_exposure_time,
            "mean_target_distance": (
                self._distance_sum / self._distance_samples if self._distance_samples else 0.0
            ),
            "close_range_fraction": self.close_ticks / ticks,
            "long_range_fraction": self.far_ticks / ticks,
            "corner_fraction": self.corner_ticks / ticks,
        }

    def movement(self) -> dict[str, Any]:
        displacement = (
            math.dist(self._start_position, self._end_position)
            if self._start_position is not None and self._end_position is not None
            else 0.0
        )
        return {
            # 1.0 = walked in a straight line; near 0 = wandered in circles.
            "path_efficiency": displacement / self._path_length if self._path_length > 1e-9 else 0.0,
            "path_length": self._path_length,
            "displacement": displacement,
            "navigation_failures": self.navigation_failures,
            "mean_speed": self._speed_sum / self.ticks if self.ticks else 0.0,
            "jumps": self.jumps,
            "stuck_time": self.stuck_time,
        }

    def combat(self) -> dict[str, Any]:
        return {
            "damage_dealt": self.damage_dealt,
            "damage_received": self.damage_received,
            "damage_ratio": (
                self.damage_dealt / self.damage_received if self.damage_received > 0.0 else float(self.damage_dealt > 0.0)
            ),
            "kills": self.kills,
            "deaths": self.deaths,
            "unnecessary_shots": self.unnecessary_shots,
            "target_selection_changes": self.target_switches,
        }

    def survival(self) -> dict[str, Any]:
        return {
            "survival_time": self.alive_time,
            "damage_free_time": self.damage_free_time,
            "damage_avoidance": self.damage_free_time / self.elapsed if self.elapsed else 0.0,
            "escapes": self.escapes,
        }

    def exploration(self) -> dict[str, Any]:
        return {
            "map_coverage": self.explored_fraction,
            "new_area_events": self.new_area_events,
            "exploration_efficiency": (
                self.explored_fraction / self._path_length if self._path_length > 1e-9 else 0.0
            ),
            "time_to_half_coverage": self.time_to_half_coverage,
            "time_to_target_coverage": self.time_to_target_coverage,
        }

    def categories(self) -> dict[str, dict[str, Any]]:
        return {
            "aim": self.aim(),
            "reaction": self.reaction(),
            "awareness": self.awareness(),
            "positioning": self.positioning(),
            "movement": self.movement(),
            "combat": self.combat(),
            "survival": self.survival(),
            "exploration": self.exploration(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "labels": dict(self.labels),
            "ticks": self.ticks,
            "elapsed": self.elapsed,
            "result": dict(self.result),
            "categories": self.categories(),
        }

    def flat(self) -> dict[str, Any]:
        """``category.metric`` -> value, plus labels. The CSV row shape."""
        row: dict[str, Any] = {f"label.{key}": value for key, value in self.labels.items()}
        row["ticks"] = self.ticks
        row["elapsed"] = self.elapsed
        for category, values in self.categories().items():
            for name, value in values.items():
                row[f"{category}.{name}"] = value
        for key, value in self.result.items():
            if isinstance(value, (int, float, str, bool)):
                row[f"result.{key}"] = value
        return row


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flattens a :meth:`EpisodeMetrics.summary` payload into a CSV row."""
    row: dict[str, Any] = {f"label.{key}": value for key, value in summary.get("labels", {}).items()}
    row["ticks"] = summary.get("ticks", 0)
    row["elapsed"] = summary.get("elapsed", 0.0)
    for category, values in summary.get("categories", {}).items():
        for name, value in values.items():
            row[f"{category}.{name}"] = value
    for key, value in summary.get("result", {}).items():
        if isinstance(value, (int, float, str, bool)):
            row[f"result.{key}"] = value
    return row


class MetricsAggregator:
    """Collects episode summaries and aggregates them along any label axis."""

    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, summary: dict[str, Any]) -> None:
        if "categories" not in summary:
            raise ValueError("expected an EpisodeMetrics.summary() payload")
        self._episodes.append(summary)

    def add_episode(self, metrics: EpisodeMetrics) -> None:
        self.add(metrics.summary())

    def episodes(self) -> list[dict[str, Any]]:
        return list(self._episodes)

    def reset(self) -> None:
        self._episodes.clear()

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _mean_of(rows: Sequence[dict[str, Any]], category: str, metric: str) -> float:
        """Mean over episodes, ignoring the -1 "never happened" sentinel.

        Averaging a -1 in with real latencies produces a number that is
        not a latency at all, so those episodes are excluded rather than
        silently poisoning the mean.
        """
        values = [
            float(row["categories"][category][metric])
            for row in rows
            if metric in row["categories"].get(category, {})
        ]
        values = [value for value in values if value >= 0.0 or metric.endswith("_norm")]
        return sum(values) / len(values) if values else 0.0

    def aggregate(self, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        selected = list(rows if rows is not None else self._episodes)
        aggregate: dict[str, Any] = {"episodes": len(selected), "categories": {}}
        if not selected:
            aggregate["categories"] = {category: {} for category in CATEGORIES}
            return aggregate
        for category in CATEGORIES:
            names: list[str] = []
            for row in selected:
                for name in row["categories"].get(category, {}):
                    if name not in names:
                        names.append(name)
            aggregate["categories"][category] = {
                name: self._mean_of(selected, category, name) for name in names
            }
        return aggregate

    def group_by(self, key: str) -> dict[str, dict[str, Any]]:
        """Aggregates by one label (``policy_id``, ``map_id``, ...)."""
        if key not in GROUP_KEYS:
            raise ValueError(f"unknown grouping key {key!r}; known: {GROUP_KEYS}")
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for row in self._episodes:
            value = str(row.get("labels", {}).get(key, ""))
            if value not in buckets:
                buckets[value] = []
                order.append(value)
            buckets[value].append(row)
        return {value: self.aggregate(buckets[value]) for value in order}

    def group_by_many(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Aggregates by a composite key, e.g. (policy, map, lighting)."""
        for key in keys:
            if key not in GROUP_KEYS:
                raise ValueError(f"unknown grouping key {key!r}; known: {GROUP_KEYS}")
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for row in self._episodes:
            labels = row.get("labels", {})
            value = "|".join(f"{key}={labels.get(key, '')}" for key in keys)
            if value not in buckets:
                buckets[value] = []
                order.append(value)
            buckets[value].append(row)
        return {value: self.aggregate(buckets[value]) for value in order}

    # -- export ------------------------------------------------------------

    def to_json(self, path: str | Path, group_keys: Sequence[str] = ("policy_id", "map_id")) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall": self.aggregate(),
            "groups": {key: self.group_by(key) for key in group_keys},
            "episodes": self._episodes,
        }
        target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return target

    def to_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [flatten_summary(summary) for summary in self._episodes]
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return target


def format_metrics(aggregate: dict[str, Any]) -> str:
    """Human-readable rendering of an aggregate. Diagnostics, not a score."""
    lines = [
        "SandboxAI research metrics (diagnostic; not reward terms)",
        "=" * 64,
        f"episodes: {aggregate.get('episodes', 0)}",
    ]
    for category in CATEGORIES:
        values: dict[str, Any] = aggregate.get("categories", {}).get(category, {})
        if not values:
            continue
        lines.append("")
        lines.append(category.upper())
        for name, value in values.items():
            lines.append(f"  {name:<28}{float(value):>12.4f}")
    return "\n".join(lines) + "\n"


def iter_flat_rows(summaries: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for summary in summaries:
        yield flatten_summary(summary)
