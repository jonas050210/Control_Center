"""Tests for the research/skill metrics subsystem (Phase 3)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from sandboxai.contract import OBSERVATION_FIELD_COUNT, OBSERVATION_INDEX
from sandboxai.metrics import (
    BAD_EXPOSURE_SECONDS,
    CATEGORIES,
    GROUP_KEYS,
    EpisodeMetrics,
    MetricsAggregator,
    StepSample,
    flatten_summary,
    format_metrics,
)

DT = 1.0 / 60.0


def obs(**fields: float) -> list[float]:
    """Builds an observation vector by FIELD NAME.

    Tests that hardcode indices are exactly the thing the contract's
    append-only rule is supposed to protect against, so they are built
    through OBSERVATION_INDEX here too.
    """
    vector = [0.0] * OBSERVATION_FIELD_COUNT
    for name, value in fields.items():
        index, width = OBSERVATION_INDEX[name]
        if width == 1:
            vector[index] = float(value)
        else:  # pragma: no cover - only used via obs_vec below
            raise ValueError(f"{name} is {width}-wide; set it explicitly")
    return vector


def obs_with_position(position, velocity, **fields: float) -> list[float]:
    vector = obs(**fields)
    p_index, _ = OBSERVATION_INDEX["agent_position_norm"]
    v_index, _ = OBSERVATION_INDEX["agent_velocity_norm"]
    vector[p_index : p_index + 3] = [float(value) for value in position]
    vector[v_index : v_index + 3] = [float(value) for value in velocity]
    return vector


IDLE_ACTION = (1, 1, 1, 1, 0, 0)
FORWARD_ACTION = (2, 1, 1, 1, 0, 0)


class AimAndReactionTests(unittest.TestCase):
    def test_accuracy_and_aim_error(self):
        metrics = EpisodeMetrics()
        for index in range(4):
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=1.0, primary_enemy_bearing_norm=0.1),
                    events={"shot_fired": True, "hit": index < 3, "damage_dealt": 10.0 if index < 3 else 0.0},
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        aim = metrics.aim()
        self.assertEqual(aim["shots"], 4)
        self.assertEqual(aim["hits"], 3)
        self.assertAlmostEqual(aim["accuracy"], 0.75)
        self.assertAlmostEqual(aim["mean_aim_error"], 0.1, places=6)

    def test_target_switches_are_counted_once_per_change(self):
        metrics = EpisodeMetrics()
        for target in (0, 0, 1, 1, 2):
            metrics.record(
                StepSample(observation=obs(), action=IDLE_ACTION, dt=DT, target_id=target)
            )
        self.assertEqual(metrics.aim()["target_switches"], 2)

    def test_reaction_latencies_are_measured_from_first_sighting(self):
        metrics = EpisodeMetrics()
        for _ in range(5):  # nothing visible yet
            metrics.record(StepSample(observation=obs(), action=IDLE_ACTION, dt=DT))
        metrics.record(
            StepSample(observation=obs(primary_enemy_visible=1.0), action=IDLE_ACTION, dt=DT)
        )
        metrics.record(
            StepSample(
                observation=obs(primary_enemy_visible=1.0, primary_enemy_los_clear=1.0),
                action=IDLE_ACTION,
                dt=DT,
            )
        )
        metrics.record(
            StepSample(
                observation=obs(primary_enemy_visible=1.0, primary_enemy_los_clear=1.0),
                events={"shot_fired": True},
                action=IDLE_ACTION,
                dt=DT,
            )
        )
        reaction = metrics.reaction()
        self.assertAlmostEqual(reaction["detection_latency"], 5 * DT, places=6)
        self.assertAlmostEqual(reaction["confirmation_latency"], DT, places=6)
        self.assertAlmostEqual(reaction["shot_latency"], 2 * DT, places=6)

    def test_latency_is_minus_one_when_it_never_happened(self):
        metrics = EpisodeMetrics()
        metrics.record(StepSample(observation=obs(), action=IDLE_ACTION, dt=DT))
        reaction = metrics.reaction()
        self.assertEqual(reaction["detection_latency"], -1.0)
        self.assertEqual(reaction["shot_latency"], -1.0)


class AwarenessTests(unittest.TestCase):
    def test_contacts_seen_and_lost(self):
        metrics = EpisodeMetrics()
        pattern = [1, 1, 0, 0, 1, 0]
        for visible in pattern:
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=float(visible)),
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        awareness = metrics.awareness()
        self.assertEqual(awareness["visible_contacts"], 2)
        self.assertEqual(awareness["lost_contacts"], 2)

    def test_memory_and_sound_use_require_a_reacquisition(self):
        metrics = EpisodeMetrics()
        # remembered contact (not visible but confident) then re-sighted
        for _ in range(3):
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_confidence=0.6, last_sound_loudness=0.4),
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        metrics.record(
            StepSample(observation=obs(primary_enemy_visible=1.0), action=IDLE_ACTION, dt=DT)
        )
        awareness = metrics.awareness()
        self.assertEqual(awareness["memory_reacquisitions"], 1)
        self.assertEqual(awareness["sound_reacquisitions"], 1)
        self.assertAlmostEqual(awareness["memory_tick_fraction"], 0.75)

    def test_unknown_area_fraction(self):
        metrics = EpisodeMetrics()
        for known in (1.0, 1.0, 0.0, 0.0):
            metrics.record(
                StepSample(observation=obs(current_area_known=known), action=IDLE_ACTION, dt=DT)
            )
        self.assertAlmostEqual(metrics.awareness()["unknown_area_fraction"], 0.5)


class PositioningAndMovementTests(unittest.TestCase):
    def test_cover_and_bad_exposure(self):
        metrics = EpisodeMetrics()
        steps = int(BAD_EXPOSURE_SECONDS / DT) + 20
        for _ in range(steps):
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=1.0, agent_in_cover=0.0),
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        positioning = metrics.positioning()
        self.assertEqual(positioning["cover_fraction"], 0.0)
        self.assertGreater(positioning["bad_exposure_time"], 0.0)
        self.assertLess(positioning["bad_exposure_time"], steps * DT)

    def test_cover_resets_the_exposure_streak(self):
        metrics = EpisodeMetrics()
        for index in range(200):
            in_cover = 1.0 if index % 2 == 0 else 0.0
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=1.0, agent_in_cover=in_cover),
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        # Alternating in/out of cover never accumulates a long exposure.
        self.assertEqual(metrics.positioning()["bad_exposure_time"], 0.0)
        self.assertAlmostEqual(metrics.positioning()["cover_fraction"], 0.5)

    def test_path_efficiency_straight_line_versus_circle(self):
        straight = EpisodeMetrics()
        for step in range(10):
            straight.record(
                StepSample(
                    observation=obs_with_position((0.1 * step, 0.0, 0.0), (1.0, 0.0, 0.0)),
                    action=FORWARD_ACTION,
                    dt=DT,
                )
            )
        self.assertAlmostEqual(straight.movement()["path_efficiency"], 1.0, places=5)

        loop = EpisodeMetrics()
        cycle = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.0, 0.1), (0.0, 0.0, 0.1)]
        for step in range(12):
            loop.record(
                StepSample(
                    observation=obs_with_position(cycle[step % 4], (1.0, 0.0, 0.0)),
                    action=FORWARD_ACTION,
                    dt=DT,
                )
            )
        self.assertLess(loop.movement()["path_efficiency"], 0.2)

    def test_stuck_time_needs_movement_intent(self):
        idle = EpisodeMetrics()
        moving = EpisodeMetrics()
        for _ in range(30):
            sample = obs_with_position((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            idle.record(StepSample(observation=sample, action=IDLE_ACTION, dt=DT))
            moving.record(StepSample(observation=sample, action=FORWARD_ACTION, dt=DT))
        self.assertEqual(idle.movement()["stuck_time"], 0.0)
        self.assertAlmostEqual(moving.movement()["stuck_time"], 30 * DT, places=6)

    def test_jumps_and_navigation_failures(self):
        metrics = EpisodeMetrics()
        metrics.record(
            StepSample(observation=obs(), action=(1, 1, 1, 1, 0, 1), dt=DT, navigation_failed=True)
        )
        metrics.record(StepSample(observation=obs(), action=(1, 1, 1, 1, 0, 1), dt=DT))
        movement = metrics.movement()
        self.assertEqual(movement["jumps"], 2)
        self.assertEqual(movement["navigation_failures"], 1)


class CombatSurvivalExplorationTests(unittest.TestCase):
    def test_combat_counters(self):
        metrics = EpisodeMetrics()
        metrics.record(
            StepSample(
                observation=obs(),
                events={"shot_fired": True, "hit": True, "damage_dealt": 30.0, "kill": True},
                action=IDLE_ACTION,
                dt=DT,
            )
        )
        metrics.record(
            StepSample(
                observation=obs(),
                events={"damage_taken": 12.0, "useless_shot": True, "shot_fired": True},
                action=IDLE_ACTION,
                dt=DT,
            )
        )
        combat = metrics.combat()
        self.assertEqual(combat["kills"], 1)
        self.assertAlmostEqual(combat["damage_dealt"], 30.0)
        self.assertAlmostEqual(combat["damage_received"], 12.0)
        self.assertEqual(combat["unnecessary_shots"], 1)
        self.assertAlmostEqual(combat["damage_ratio"], 2.5)

    def test_survival_and_escapes(self):
        metrics = EpisodeMetrics()
        # exposed under fire, then breaks line of sight into cover
        metrics.record(
            StepSample(
                observation=obs(primary_enemy_visible=1.0, agent_in_cover=0.0),
                events={"damage_taken": 5.0},
                action=IDLE_ACTION,
                dt=DT,
            )
        )
        metrics.record(
            StepSample(observation=obs(agent_in_cover=1.0), action=IDLE_ACTION, dt=DT)
        )
        survival = metrics.survival()
        self.assertEqual(survival["escapes"], 1)
        self.assertAlmostEqual(survival["survival_time"], 2 * DT, places=6)
        self.assertAlmostEqual(survival["damage_avoidance"], 0.5, places=6)

    def test_exploration_progress_and_time_to_coverage(self):
        metrics = EpisodeMetrics()
        for step in range(20):
            metrics.record(
                StepSample(
                    observation=obs_with_position(
                        (0.05 * step, 0.0, 0.0), (1.0, 0.0, 0.0), explored_fraction=0.05 * step
                    ),
                    action=FORWARD_ACTION,
                    dt=DT,
                )
            )
        exploration = metrics.exploration()
        self.assertAlmostEqual(exploration["map_coverage"], 0.95, places=6)
        self.assertEqual(exploration["new_area_events"], 19)
        self.assertGreater(exploration["time_to_half_coverage"], 0.0)
        self.assertGreater(exploration["time_to_target_coverage"], exploration["time_to_half_coverage"])
        self.assertGreater(exploration["exploration_efficiency"], 0.0)


class ResetTests(unittest.TestCase):
    def test_reset_clears_every_category(self):
        metrics = EpisodeMetrics({"policy_id": "brain_a"})
        for _ in range(5):
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=1.0, explored_fraction=0.4),
                    events={"shot_fired": True, "hit": True, "damage_dealt": 5.0, "kill": True},
                    action=(2, 1, 1, 1, 1, 1),
                    dt=DT,
                )
            )
        metrics.finish({"win": True})
        metrics.reset()
        self.assertEqual(metrics.ticks, 0)
        self.assertEqual(metrics.result, {})
        for category, values in metrics.categories().items():
            for name, value in values.items():
                if name.endswith("latency") or name.startswith("time_to_"):
                    self.assertEqual(value, -1.0, f"{category}.{name}")
                else:
                    self.assertEqual(float(value), 0.0, f"{category}.{name}")
        # Labels survive a reset unless new ones are supplied.
        self.assertEqual(metrics.labels["policy_id"], "brain_a")
        metrics.reset({"policy_id": "brain_b"})
        self.assertEqual(metrics.labels["policy_id"], "brain_b")

    def test_two_episodes_through_one_object_do_not_leak(self):
        metrics = EpisodeMetrics()
        for _ in range(10):
            metrics.record(
                StepSample(observation=obs(), events={"shot_fired": True}, action=IDLE_ACTION, dt=DT)
            )
        first = metrics.finish({})
        metrics.reset()
        for _ in range(2):
            metrics.record(
                StepSample(observation=obs(), events={"shot_fired": True}, action=IDLE_ACTION, dt=DT)
            )
        second = metrics.finish({})
        self.assertEqual(first["categories"]["aim"]["shots"], 10)
        self.assertEqual(second["categories"]["aim"]["shots"], 2)


class AggregationAndExportTests(unittest.TestCase):
    def _episode(self, policy: str, map_id: str, shots: int, hits: int) -> dict:
        metrics = EpisodeMetrics(
            {
                "policy_id": policy,
                "map_id": map_id,
                "scenario": "cover_fight",
                "lighting": "normal",
                "enemy_count": 2,
                "curriculum_level": 6,
                "seed": 11,
                "episode_id": f"{policy}-{map_id}-{shots}",
            }
        )
        for index in range(shots):
            metrics.record(
                StepSample(
                    observation=obs(primary_enemy_visible=1.0),
                    events={"shot_fired": True, "hit": index < hits},
                    action=IDLE_ACTION,
                    dt=DT,
                )
            )
        return metrics.finish({"win": hits > shots / 2})

    def test_group_by_policy_and_map(self):
        aggregator = MetricsAggregator()
        aggregator.add(self._episode("brain_a", "open_field", 10, 8))
        aggregator.add(self._episode("brain_a", "night_yard", 10, 2))
        aggregator.add(self._episode("brain_b", "open_field", 10, 5))
        self.assertEqual(len(aggregator), 3)

        by_policy = aggregator.group_by("policy_id")
        self.assertAlmostEqual(by_policy["brain_a"]["categories"]["aim"]["accuracy"], 0.5)
        self.assertAlmostEqual(by_policy["brain_b"]["categories"]["aim"]["accuracy"], 0.5)
        self.assertEqual(by_policy["brain_a"]["episodes"], 2)

        by_map = aggregator.group_by("map_id")
        self.assertAlmostEqual(by_map["night_yard"]["categories"]["aim"]["accuracy"], 0.2)

    def test_group_by_many_builds_composite_keys(self):
        aggregator = MetricsAggregator()
        aggregator.add(self._episode("brain_a", "open_field", 4, 4))
        groups = aggregator.group_by_many(["policy_id", "map_id"])
        self.assertIn("policy_id=brain_a|map_id=open_field", groups)

    def test_every_documented_group_key_is_usable(self):
        aggregator = MetricsAggregator()
        aggregator.add(self._episode("brain_a", "open_field", 2, 1))
        for key in GROUP_KEYS:
            self.assertEqual(sum(g["episodes"] for g in aggregator.group_by(key).values()), 1)

    def test_unknown_group_key_raises(self):
        aggregator = MetricsAggregator()
        with self.assertRaises(ValueError):
            aggregator.group_by("secret_enemy_position")

    def test_aggregate_ignores_never_happened_latencies(self):
        aggregator = MetricsAggregator()
        never = EpisodeMetrics({"policy_id": "p"})
        never.record(StepSample(observation=obs(), action=IDLE_ACTION, dt=DT))
        aggregator.add(never.finish({}))

        happened = EpisodeMetrics({"policy_id": "p"})
        happened.record(
            StepSample(observation=obs(primary_enemy_visible=1.0), action=IDLE_ACTION, dt=DT)
        )
        aggregator.add(happened.finish({}))
        # Only the episode where detection happened contributes; a -1
        # sentinel averaged in would report a negative latency.
        self.assertGreaterEqual(aggregator.aggregate()["categories"]["reaction"]["detection_latency"], 0.0)

    def test_empty_aggregate_is_well_formed(self):
        aggregate = MetricsAggregator().aggregate()
        self.assertEqual(aggregate["episodes"], 0)
        self.assertEqual(sorted(aggregate["categories"]), sorted(CATEGORIES))

    def test_json_and_csv_export(self):
        aggregator = MetricsAggregator()
        aggregator.add(self._episode("brain_a", "open_field", 6, 3))
        aggregator.add(self._episode("brain_b", "compound", 6, 6))
        with tempfile.TemporaryDirectory() as tmp:
            json_path = aggregator.to_json(Path(tmp) / "metrics.json")
            csv_path = aggregator.to_csv(Path(tmp) / "metrics.csv")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall"]["episodes"], 2)
            self.assertIn("brain_a", payload["groups"]["policy_id"])
            with csv_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertIn("aim.accuracy", rows[0])
        self.assertIn("label.policy_id", rows[0])

    def test_flatten_matches_the_object_method(self):
        metrics = EpisodeMetrics({"policy_id": "brain_a"})
        metrics.record(StepSample(observation=obs(), action=IDLE_ACTION, dt=DT))
        summary = metrics.finish({"win": True})
        self.assertEqual(flatten_summary(summary), metrics.flat())

    def test_add_rejects_a_non_metrics_payload(self):
        with self.assertRaises(ValueError):
            MetricsAggregator().add({"episode_reward": 1.0})

    def test_format_metrics_renders_every_non_empty_category(self):
        aggregator = MetricsAggregator()
        aggregator.add(self._episode("brain_a", "open_field", 3, 2))
        text = format_metrics(aggregator.aggregate())
        for category in CATEGORIES:
            self.assertIn(category.upper(), text)
        self.assertIn("diagnostic", text)


class InformationBoundaryTests(unittest.TestCase):
    def test_ground_truth_never_reaches_a_category(self):
        clean = EpisodeMetrics()
        leaky = EpisodeMetrics()
        for _ in range(5):
            observation = obs(primary_enemy_visible=0.0)
            clean.record(StepSample(observation=observation, action=IDLE_ACTION, dt=DT))
            leaky.record(
                StepSample(
                    observation=observation,
                    action=IDLE_ACTION,
                    dt=DT,
                    ground_truth={"enemy_position": [5.0, 0.0, 3.0], "enemy_health": 100.0},
                )
            )
        self.assertEqual(clean.categories(), leaky.categories())


if __name__ == "__main__":
    unittest.main()
