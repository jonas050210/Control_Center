"""Tests for generalization evaluation (Phase 7) and the randomized
training distribution (Phase 11)."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from sandboxai.conditions import LIGHTING_IDS, MAP_IDS
from sandboxai.generalization import (
    ENEMY_COUNT_BUCKETS,
    EVALUATION_LIGHTINGS,
    MAP_BUCKETS,
    SCENARIO_FAMILIES,
    EvaluationEpisode,
    GeneralizationSuite,
    MapSplit,
    compare_policies,
    format_report,
)
from sandboxai.randomization import (
    FORBIDDEN_LABEL_TOKENS,
    SPAWN_RULES,
    DistributionRunTracker,
    RandomizationError,
    TrainingDistribution,
    assert_no_environment_labels,
    sample_conditions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class MapSplitTests(unittest.TestCase):
    def test_a_map_cannot_be_both_trained_and_held_out(self):
        with self.assertRaises(ValueError):
            MapSplit(train_maps=["open_field"], holdout_maps=["open_field"])

    def test_a_seed_cannot_be_both_trained_and_unseen(self):
        with self.assertRaises(ValueError):
            MapSplit(train_seeds=[1], unseen_seeds=[1])

    def test_bucket_classification(self):
        split = MapSplit(
            train_maps=["open_field"],
            holdout_maps=["compound"],
            train_seeds=[1, 2],
            unseen_seeds=[99],
            train_scenarios=["single_target"],
            unseen_scenarios=["ambush"],
        )
        self.assertEqual(split.bucket_of("open_field", 1, "single_target"), "known")
        self.assertEqual(split.bucket_of("open_field", 99, "single_target"), "unseen_seeds")
        self.assertEqual(split.bucket_of("open_field", 1, "ambush"), "unseen_variants")
        self.assertEqual(split.bucket_of("compound", 1, "single_target"), "unseen_maps")

    def test_default_split_holds_maps_out(self):
        split = MapSplit()
        self.assertTrue(split.holdout_maps)
        self.assertFalse(set(split.train_maps) & set(split.holdout_maps))
        self.assertTrue(set(split.train_maps) | set(split.holdout_maps) <= set(MAP_IDS))


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = GeneralizationSuite(episodes_per_cell=1)

    def test_plan_is_deterministic(self):
        first = [episode.condition for episode in self.suite.plan()]
        second = [episode.condition for episode in GeneralizationSuite(episodes_per_cell=1).plan()]
        self.assertEqual(first, second)

    def test_plan_covers_every_axis_and_bucket(self):
        plan = self.suite.plan()
        buckets = {episode.map_bucket for episode in plan}
        self.assertEqual(buckets, set(MAP_BUCKETS))
        axes = {episode.axis for episode in plan}
        self.assertEqual(axes, {"maps", "conditions", "combat", "scenarios"})
        lighting_buckets = {e.bucket for e in plan if e.axis == "conditions"}
        self.assertEqual(lighting_buckets, set(EVALUATION_LIGHTINGS))
        combat_buckets = {e.bucket for e in plan if e.axis == "combat"}
        self.assertEqual(combat_buckets, {f"{n}_enemies" for n in ENEMY_COUNT_BUCKETS})
        scenario_buckets = {e.bucket for e in plan if e.axis == "scenarios"}
        self.assertEqual(scenario_buckets, set(SCENARIO_FAMILIES))

    def test_unseen_maps_are_actually_held_out(self):
        split = self.suite.split
        for episode in self.suite.plan():
            if episode.map_bucket == "unseen_maps":
                self.assertIn(episode.condition.map_id, split.holdout_maps)
                self.assertNotIn(episode.condition.map_id, split.train_maps)

    def test_every_lighting_id_is_real(self):
        for lighting in EVALUATION_LIGHTINGS:
            self.assertIn(lighting, LIGHTING_IDS)

    def test_every_scenario_family_maps_to_a_real_scenario(self):
        source = (REPO_ROOT / "scripts" / "scenario" / "scenario_library.gd").read_text(
            encoding="utf-8"
        )
        declared = set(re.findall(r'"id":\s*"([a-z_]+)"', source))
        for family, scenario_id in SCENARIO_FAMILIES.items():
            self.assertIn(
                scenario_id, declared, f"{family} -> {scenario_id} is not a real scenario"
            )

    def test_episodes_per_cell_scales_the_plan(self):
        single = GeneralizationSuite(episodes_per_cell=1).plan_size()
        double = GeneralizationSuite(episodes_per_cell=2).plan_size()
        self.assertEqual(double, single * 2)


class ReportTests(unittest.TestCase):
    def _run(self, unseen_win_rate: float) -> GeneralizationSuite:
        suite = GeneralizationSuite(episodes_per_cell=1)
        counter = {"unseen": 0}

        def play(episode: EvaluationEpisode) -> dict:
            if episode.map_bucket == "unseen_maps":
                counter["unseen"] += 1
                won = (counter["unseen"] % 100) < unseen_win_rate * 100
            else:
                won = True
            return {"won": won, "reward": 1.0 if won else -1.0, "steps": 100}

        suite.run(play)
        return suite

    def test_report_shape(self):
        report = self._run(0.0).report()
        for key in ("episodes", "overall", "map_buckets", "by_axis", "transfer_gap", "split"):
            self.assertIn(key, report)
        self.assertEqual(sorted(report["map_buckets"]), sorted(MAP_BUCKETS))

    def test_transfer_gap_detects_memorization(self):
        memorizer = self._run(0.0).report()
        generalizer = self._run(1.0).report()
        self.assertAlmostEqual(memorizer["transfer_gap"], 1.0)
        self.assertAlmostEqual(generalizer["transfer_gap"], 0.0)

    def test_nothing_is_reported_for_episodes_that_were_not_played(self):
        suite = GeneralizationSuite(episodes_per_cell=1)
        report = suite.report()
        self.assertEqual(report["episodes"], 0)
        for bucket in MAP_BUCKETS:
            self.assertEqual(report["map_buckets"][bucket]["episodes"], 0)
            self.assertEqual(report["map_buckets"][bucket]["win_rate"], 0.0)

    def test_extra_keys_from_the_runner_are_carried_through(self):
        suite = GeneralizationSuite(episodes_per_cell=1)
        suite.run(lambda episode: {"won": True, "reward": 1.0, "steps": 10, "accuracy": 0.42})
        self.assertAlmostEqual(suite.rows()[0]["accuracy"], 0.42)

    def test_export_writes_json_csv_and_text(self):
        suite = self._run(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            paths = suite.export(tmp)
            payload = json.loads(paths["report"].read_text(encoding="utf-8"))
            with paths["episodes"].open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            text = paths["text"].read_text(encoding="utf-8")
        self.assertEqual(payload["episodes"], len(rows))
        self.assertIn("transfer gap", text)

    def test_format_report_renders_every_axis(self):
        text = format_report(self._run(0.5).report())
        for axis in ("CONDITIONS", "COMBAT", "SCENARIOS"):
            self.assertIn(axis, text)

    def test_compare_policies_orders_by_win_rate(self):
        strong = self._run(1.0).report()
        weak = self._run(0.0).report()
        comparison = compare_policies({"weak": weak, "strong": strong})
        self.assertEqual(comparison["policies"][0]["policy_id"], "strong")
        self.assertEqual(comparison["count"], 2)

    def test_reset_clears_results(self):
        suite = self._run(1.0)
        suite.reset()
        self.assertEqual(suite.report()["episodes"], 0)


class NoLabelLeakTests(unittest.TestCase):
    def test_the_observation_contract_exposes_no_environment_label(self):
        names = assert_no_environment_labels()
        self.assertGreater(len(names), 50)
        for name in names:
            for token in FORBIDDEN_LABEL_TOKENS:
                self.assertNotIn(token, name)

    def test_the_check_would_actually_fail(self):
        # Proves the guard is not vacuous: patch in a labelled field.
        import sandboxai.randomization as randomization
        from sandboxai.contract import ObservationField

        original = randomization.OBSERVATION_SPEC
        randomization.OBSERVATION_SPEC = original + (
            ObservationField(999, 1, "map_id_onehot", "leak", "leak"),
        )
        try:
            with self.assertRaises(RandomizationError):
                randomization.assert_no_environment_labels()
        finally:
            randomization.OBSERVATION_SPEC = original


class TrainingDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.distribution = TrainingDistribution(master_seed=99)

    def test_episode_plan_is_a_pure_function(self):
        first = self.distribution.episode_plan(137)
        second = TrainingDistribution(master_seed=99).episode_plan(137)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_out_of_order_access_does_not_change_the_stream(self):
        forward = [self.distribution.episode_plan(index).to_dict() for index in range(10)]
        shuffled_source = TrainingDistribution(master_seed=99)
        backward = [shuffled_source.episode_plan(index).to_dict() for index in reversed(range(10))]
        self.assertEqual(forward, list(reversed(backward)))

    def test_a_different_master_seed_produces_a_different_stream(self):
        other = TrainingDistribution(master_seed=100)
        self.assertNotEqual(
            [plan.condition.seed for plan in self.distribution.episode_plans(20)],
            [plan.condition.seed for plan in other.episode_plans(20)],
        )

    def test_the_distribution_is_actually_varied(self):
        coverage = self.distribution.coverage(400)
        self.assertEqual(coverage["distinct_maps"], len(MAP_IDS))
        self.assertGreaterEqual(coverage["distinct_lightings"], len(LIGHTING_IDS) - 1)
        self.assertGreater(coverage["distinct_layout_variants"], 20)
        self.assertGreater(coverage["distinct_conditions"], 20)

    def test_parallel_environments_get_different_episodes(self):
        first = [plan.condition.seed for plan in self.distribution.stream_for_environment(0, 8)]
        second = [plan.condition.seed for plan in self.distribution.stream_for_environment(1, 8)]
        self.assertNotEqual(first, second)
        # ...but each environment's own stream is reproducible.
        self.assertEqual(
            first, [plan.condition.seed for plan in self.distribution.stream_for_environment(0, 8)]
        )

    def test_spawn_plan_matches_the_enemy_count(self):
        for index in range(20):
            plan = self.distribution.episode_plan(index)
            self.assertEqual(plan.spawn.count, plan.condition.enemy_count)
            self.assertEqual(len(plan.spawn.seeds), plan.condition.enemy_count)
            self.assertIn(plan.spawn.rule, SPAWN_RULES)

    def test_environment_commands_are_ordered_and_complete(self):
        plan = self.distribution.episode_plan(3)
        commands = plan.environment_commands()
        self.assertEqual(commands[0]["call"], "set_curriculum_level")
        self.assertEqual(commands[-1]["call"], "reset")
        self.assertEqual(commands[-1]["seed"], plan.condition.seed)
        calls = [command["call"] for command in commands]
        self.assertIn("set_map", calls)
        self.assertIn("set_lighting_mode", calls)

    def test_replay_header_fields_match_the_plan(self):
        plan = self.distribution.episode_plan(5)
        fields = plan.replay_header_fields()
        self.assertEqual(fields["seed"], plan.condition.seed)
        self.assertEqual(fields["map_id"], plan.condition.map_id)
        self.assertEqual(fields["enemy_count"], plan.condition.enemy_count)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingDistribution(maps=[])
        with self.assertRaises(ValueError):
            TrainingDistribution(layout_variants=0)
        with self.assertRaises(ValueError):
            TrainingDistribution(spawn_rules=["teleport_behind_you"])

    def test_round_trip_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.distribution.save(Path(tmp) / "distribution.json")
            reloaded = TrainingDistribution.load(path)
        self.assertEqual(reloaded.to_dict(), self.distribution.to_dict())
        self.assertEqual(
            reloaded.episode_plan(7).to_dict(), self.distribution.episode_plan(7).to_dict()
        )

    def test_sample_conditions_can_override_the_seed_without_mutating(self):
        before = self.distribution.master_seed
        conditions = sample_conditions(self.distribution, 5, seed=4321)
        self.assertEqual(len(conditions), 5)
        self.assertEqual(self.distribution.master_seed, before)


class DistributionTrackerTests(unittest.TestCase):
    def test_per_condition_performance_and_reproduction_indices(self):
        distribution = TrainingDistribution(master_seed=7, maps=["open_field", "night_yard"])
        tracker = DistributionRunTracker(window=10)
        for index in range(40):
            plan = distribution.episode_plan(index)
            won = plan.condition.map_id == "open_field"
            tracker.record(plan, 1.0 if won else -1.0, won, 100)
        report = tracker.report()
        self.assertGreater(report["overall"]["episodes"], 0)
        self.assertTrue(report["worst"])
        worst_key = report["worst"][0]["condition"]
        self.assertIn("night_yard", worst_key)
        self.assertTrue(report["reproduce"][worst_key])
        # The recorded indices really do reproduce that condition.
        index = report["reproduce"][worst_key][0]
        self.assertEqual(distribution.episode_plan(index).key, worst_key)


if __name__ == "__main__":
    unittest.main()
