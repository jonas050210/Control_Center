"""Tests for the condition space and per-condition tracking.

Also the drift guard the ``conditions.py`` header promises: the map and
lighting id lists are hand-mirrored from GDScript, so a test has to hold
them to the engine source.
"""

from __future__ import annotations

import random
import re
import unittest
from pathlib import Path

from sandboxai.conditions import (
    LIGHTING_IDS,
    MAP_IDS,
    Condition,
    ConditionSpace,
    ConditionStats,
    ConditionTracker,
    format_generalization_report,
    generalization_report,
)
from sandboxai.randomization import SPAWN_RULES

REPO_ROOT = Path(__file__).resolve().parents[2]


class SourceDriftTests(unittest.TestCase):
    def test_map_ids_match_map_library(self):
        source = (REPO_ROOT / "scripts/world/map_library.gd").read_text(encoding="utf-8")
        block = re.search(r"const MAPS: Array = \[(.*?)\n\]", source, re.S)
        self.assertIsNotNone(block, "could not locate const MAPS")
        ids = re.findall(r'"id":\s*"([a-z0-9_]+)"', block.group(1))
        self.assertEqual(list(MAP_IDS), ids)

    def test_lighting_ids_match_lighting_profile(self):
        source = (REPO_ROOT / "scripts/perception/lighting_profile.gd").read_text(encoding="utf-8")
        match = re.search(r"const MODE_IDS: Array = \[(.*?)\]", source, re.S)
        self.assertIsNotNone(match, "could not locate MODE_IDS")
        ids = re.findall(r'"([a-z_]+)"', match.group(1))
        self.assertEqual(list(LIGHTING_IDS), ids)

    def test_spawn_rules_match_scenario_library(self):
        """SPAWN_RULES must mirror ScenarioLibrary's SPAWN_* constants.

        Regression: the list used to name three rules that do not exist in
        the engine ("spread", "flank", "cover") and miss four that do, so a
        plan carrying one of those rules would silently degrade to default
        placement and the real rules could never be drawn.
        """
        source = (REPO_ROOT / "scripts/scenario/scenario_library.gd").read_text(encoding="utf-8")
        matches = re.findall(r'^const SPAWN_(\w+): String = "([a-z_]+)"', source, re.M)
        self.assertTrue(matches, "could not locate SPAWN_* constants")
        declared = [value for _name, value in matches]
        self.assertEqual(sorted(declared), sorted(set(declared)), "duplicate SPAWN_* value")
        self.assertEqual(sorted(SPAWN_RULES), sorted(declared))

    def test_ids_are_unique(self):
        self.assertEqual(len(set(MAP_IDS)), len(MAP_IDS))
        self.assertEqual(len(set(LIGHTING_IDS)), len(LIGHTING_IDS))
        self.assertEqual(len(set(SPAWN_RULES)), len(SPAWN_RULES))


class ConditionTests(unittest.TestCase):
    def test_key_excludes_the_seed(self):
        first = Condition(map_id="compound", lighting="fog", enemy_count=2, level=6, seed=1)
        second = Condition(map_id="compound", lighting="fog", enemy_count=2, level=6, seed=99999)
        self.assertEqual(first.key, second.key)

    def test_key_distinguishes_every_other_axis(self):
        base = Condition(
            map_id="compound", lighting="fog", scenario="ambush", enemy_count=2, level=6
        )
        variants = [
            Condition(map_id="catwalks", lighting="fog", scenario="ambush", enemy_count=2, level=6),
            Condition(
                map_id="compound", lighting="night", scenario="ambush", enemy_count=2, level=6
            ),
            Condition(
                map_id="compound", lighting="fog", scenario="sound_only", enemy_count=2, level=6
            ),
            Condition(map_id="compound", lighting="fog", scenario="ambush", enemy_count=3, level=6),
            Condition(map_id="compound", lighting="fog", scenario="ambush", enemy_count=2, level=7),
        ]
        for variant in variants:
            self.assertNotEqual(base.key, variant.key)

    def test_empty_axes_render_as_placeholders(self):
        self.assertEqual(Condition().key, "-|-|-|n1|L1")

    def test_to_dict_round_trips(self):
        condition = Condition(map_id="two_rooms", lighting="night", seed=7)
        self.assertEqual(Condition(**condition.to_dict()), condition)


class ConditionSpaceTests(unittest.TestCase):
    def test_default_space_covers_the_real_ids(self):
        space = ConditionSpace()
        self.assertEqual(list(space.maps), list(MAP_IDS))
        self.assertEqual(list(space.lightings), list(LIGHTING_IDS))

    def test_size_matches_the_enumerated_grid(self):
        space = ConditionSpace(
            maps=["open_field", "compound"],
            lightings=["normal", "fog"],
            scenarios=[],
            enemy_counts=[1, 3],
            levels=[6],
        )
        self.assertEqual(space.size(), len(list(space.enumerate_grid())))
        self.assertEqual(space.size(), 8)

    def test_enumerate_grid_repeats_seeds_per_condition(self):
        space = ConditionSpace(
            maps=["open_field"], lightings=["normal"], enemy_counts=[1], levels=[1]
        )
        conditions = list(space.enumerate_grid(seeds_per_condition=4))
        self.assertEqual(len(conditions), 4)
        self.assertEqual(len({condition.seed for condition in conditions}), 4)
        self.assertEqual(len({condition.key for condition in conditions}), 1)

    def test_enumerate_grid_is_deterministic(self):
        space = ConditionSpace(maps=["open_field", "compound"], base_seed=17)
        first = [condition.to_dict() for condition in space.enumerate_grid()]
        second = [condition.to_dict() for condition in space.enumerate_grid()]
        self.assertEqual(first, second)

    def test_sampling_is_a_pure_function_of_the_rng(self):
        space = ConditionSpace()
        left = space.sample(random.Random(5))
        right = space.sample(random.Random(5))
        self.assertEqual(left, right)
        self.assertNotEqual(left, space.sample(random.Random(6)))

    def test_sample_many_respects_the_seed(self):
        space = ConditionSpace()
        self.assertEqual(space.sample_many(10, seed=3), space.sample_many(10, seed=3))
        self.assertNotEqual(space.sample_many(10, seed=3), space.sample_many(10, seed=4))

    def test_samples_stay_inside_the_space(self):
        space = ConditionSpace(maps=["open_field"], lightings=["fog"], enemy_counts=[2], levels=[6])
        for condition in space.sample_many(20, seed=1):
            self.assertEqual(condition.map_id, "open_field")
            self.assertEqual(condition.lighting, "fog")
            self.assertEqual(condition.enemy_count, 2)
            self.assertEqual(condition.level, 6)

    def test_empty_axis_means_environment_default(self):
        space = ConditionSpace(maps=[], lightings=[], scenarios=[])
        condition = space.sample(random.Random(0))
        self.assertEqual(condition.map_id, "")
        self.assertEqual(condition.lighting, "")
        self.assertEqual(condition.scenario, "")

    def test_invalid_spaces_are_rejected(self):
        with self.assertRaises(ValueError):
            ConditionSpace(enemy_counts=[])
        with self.assertRaises(ValueError):
            ConditionSpace(levels=[])
        with self.assertRaises(ValueError):
            ConditionSpace(enemy_counts=[-1])


class ConditionStatsTests(unittest.TestCase):
    def test_rolling_window_drops_old_episodes(self):
        stats = ConditionStats(key="k", window=3)
        for reward in (1.0, 2.0, 3.0, 4.0):
            stats.record(reward, won=True, steps=10)
        self.assertEqual(stats.episodes, 4)
        summary = stats.summary()
        self.assertEqual(summary["window"], 3)
        self.assertAlmostEqual(summary["mean_reward"], 3.0)

    def test_empty_stats_do_not_divide_by_zero(self):
        self.assertEqual(ConditionStats(key="k").summary()["mean_reward"], 0.0)


class ConditionTrackerTests(unittest.TestCase):
    def _tracker(self) -> ConditionTracker:
        tracker = ConditionTracker(window=50)
        easy = Condition(map_id="open_field", lighting="normal", enemy_count=1, level=3)
        hard = Condition(map_id="echo_maze", lighting="night", enemy_count=3, level=8)
        for index in range(10):
            tracker.record(easy, reward=5.0, won=True, steps=100)
            tracker.record(hard, reward=-1.0, won=index == 0, steps=200)
        return tracker

    def test_tracks_each_condition_separately(self):
        tracker = self._tracker()
        self.assertEqual(len(tracker.keys()), 2)
        self.assertEqual(tracker.total_episodes(), 20)
        easy = tracker.stats_for("open_field|normal|-|n1|L3")
        self.assertAlmostEqual(easy["win_rate"], 1.0)
        hard = tracker.stats_for("echo_maze|night|-|n3|L8")
        self.assertAlmostEqual(hard["win_rate"], 0.1)

    def test_unknown_condition_returns_empty(self):
        self.assertEqual(ConditionTracker().stats_for("nope"), {})

    def test_overall_pools_everything(self):
        overall = self._tracker().overall()
        self.assertEqual(overall["conditions"], 2)
        self.assertEqual(overall["episodes"], 20)
        self.assertAlmostEqual(overall["win_rate"], 0.55)

    def test_worst_conditions_surfaces_the_failure_first(self):
        worst = self._tracker().worst_conditions(1)
        self.assertEqual(worst[0]["condition"], "echo_maze|night|-|n3|L8")

    def test_the_average_hides_what_the_split_shows(self):
        # The point of per-condition tracking, asserted rather than claimed.
        tracker = self._tracker()
        overall = tracker.overall()["win_rate"]
        worst = tracker.worst_conditions(1)[0]["win_rate"]
        self.assertGreater(overall - worst, 0.4)


class ReportTests(unittest.TestCase):
    def test_report_flags_a_condition_dependent_policy(self):
        tracker = ConditionTracker()
        for _index in range(20):
            tracker.record(Condition(map_id="open_field"), 1.0, True, 50)
            tracker.record(Condition(map_id="echo_maze"), 0.0, False, 50)
        report = generalization_report(tracker)
        self.assertAlmostEqual(report["win_rate_spread"], 1.0)
        self.assertFalse(report["generalizes"])
        self.assertEqual(report["worst"][0]["condition"], Condition(map_id="echo_maze").key)
        self.assertEqual(report["best"][0]["condition"], Condition(map_id="open_field").key)

    def test_report_accepts_an_even_policy(self):
        tracker = ConditionTracker()
        for index in range(20):
            tracker.record(Condition(map_id="open_field"), 1.0, index % 2 == 0, 50)
            tracker.record(Condition(map_id="echo_maze"), 1.0, index % 2 == 0, 50)
        report = generalization_report(tracker)
        self.assertTrue(report["generalizes"])

    def test_a_single_condition_never_counts_as_generalizing(self):
        tracker = ConditionTracker()
        for _ in range(20):
            tracker.record(Condition(map_id="open_field"), 1.0, True, 50)
        self.assertFalse(generalization_report(tracker)["generalizes"])

    def test_formatting_lists_every_condition(self):
        tracker = ConditionTracker()
        for _ in range(3):
            tracker.record(Condition(map_id="open_field"), 1.0, True, 50)
            tracker.record(Condition(map_id="compound"), 0.0, False, 50)
        text = format_generalization_report(generalization_report(tracker))
        self.assertIn("open_field", text)
        self.assertIn("compound", text)
        self.assertIn("win-rate spread", text)


if __name__ == "__main__":
    unittest.main()
