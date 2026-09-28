"""Tests for the condition space, the self-play league and the automatic
curriculum.

These three modules decide *what* the policy is trained on and *who* it is
trained against, so the properties tested here are mostly about honesty and
reproducibility rather than about learning: the same seed must produce the
same episodes and the same opponents, a frozen policy must stay frozen, and
per-condition statistics must not be silently pooled into a flattering
average.
"""
from __future__ import annotations

import random
import re
import unittest
from pathlib import Path

from sandboxai.auto_curriculum import AutoCurriculum, CurriculumSchedule
from sandboxai.conditions import (
    LIGHTING_IDS,
    MAP_IDS,
    Condition,
    ConditionSpace,
    ConditionTracker,
    format_generalization_report,
    generalization_report,
)
from sandboxai.league import (
    CheckpointRegistry,
    League,
    expected_score,
    format_standings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConditionSpaceTests(unittest.TestCase):
    def test_map_and_lighting_ids_match_the_godot_source(self):
        """The Python mirror must not drift from the GDScript catalog."""
        map_source = (REPO_ROOT / "scripts/world/map_library.gd").read_text(encoding="utf-8")
        godot_map_ids = re.findall(r'^\t\t"id": "([a-z_]+)",', map_source, flags=re.MULTILINE)
        self.assertEqual(tuple(godot_map_ids), MAP_IDS)

        lighting_source = (
            REPO_ROOT / "scripts/perception/lighting_profile.gd"
        ).read_text(encoding="utf-8")
        match = re.search(r"const MODE_IDS: Array = \[(.*?)\]", lighting_source, flags=re.DOTALL)
        self.assertIsNotNone(match)
        godot_lighting_ids = tuple(re.findall(r'"([a-z_]+)"', match.group(1)))
        self.assertEqual(godot_lighting_ids, LIGHTING_IDS)

    def test_sampling_is_reproducible_for_a_seed(self):
        space = ConditionSpace(base_seed=7)
        first = space.sample_many(25)
        second = space.sample_many(25)
        self.assertEqual([c.to_dict() for c in first], [c.to_dict() for c in second])

    def test_different_seeds_produce_different_episode_streams(self):
        space = ConditionSpace()
        self.assertNotEqual(
            [c.to_dict() for c in space.sample_many(25, seed=1)],
            [c.to_dict() for c in space.sample_many(25, seed=2)],
        )

    def test_sampled_conditions_stay_inside_the_declared_axes(self):
        space = ConditionSpace(
            maps=["two_rooms", "night_yard"],
            lightings=["night"],
            enemy_counts=[2, 5],
            levels=[7],
        )
        rng = random.Random(3)
        for _ in range(50):
            condition = space.sample(rng)
            self.assertIn(condition.map_id, ("two_rooms", "night_yard"))
            self.assertEqual(condition.lighting, "night")
            self.assertIn(condition.enemy_count, (2, 5))
            self.assertEqual(condition.level, 7)

    def test_enumerated_grid_covers_the_space_exactly_once(self):
        space = ConditionSpace(
            maps=["open_field", "two_rooms"],
            lightings=["normal", "fog"],
            enemy_counts=[1, 3],
            levels=[6],
        )
        grid = list(space.enumerate_grid())
        self.assertEqual(len(grid), space.size())
        self.assertEqual(len({c.key for c in grid}), space.size())
        self.assertEqual(len({c.seed for c in grid}), space.size(), "seeds must be distinct")

    def test_grid_is_stable_across_calls(self):
        space = ConditionSpace(maps=["compound"], lightings=["mixed"], enemy_counts=[3])
        self.assertEqual(
            [c.to_dict() for c in space.enumerate_grid(seeds_per_condition=3)],
            [c.to_dict() for c in space.enumerate_grid(seeds_per_condition=3)],
        )

    def test_empty_axes_mean_environment_default(self):
        space = ConditionSpace(maps=[], lightings=[], enemy_counts=[1], levels=[1])
        condition = space.sample(random.Random(0))
        self.assertEqual(condition.map_id, "")
        self.assertEqual(condition.lighting, "")

    def test_invalid_space_is_rejected(self):
        with self.assertRaises(ValueError):
            ConditionSpace(enemy_counts=[])
        with self.assertRaises(ValueError):
            ConditionSpace(levels=[])


class ConditionTrackerTests(unittest.TestCase):
    def _tracker(self) -> ConditionTracker:
        tracker = ConditionTracker(window=10)
        easy = Condition(map_id="open_field", lighting="normal", enemy_count=1, level=6)
        hard = Condition(map_id="night_yard", lighting="night", enemy_count=3, level=6)
        for _ in range(10):
            tracker.record(easy, reward=8.0, won=True, steps=200)
        for index in range(10):
            tracker.record(hard, reward=-2.0, won=index == 0, steps=600)
        return tracker

    def test_statistics_are_kept_per_condition(self):
        tracker = self._tracker()
        self.assertEqual(len(tracker.keys()), 2)
        easy = tracker.stats_for("open_field|normal|-|n1|L6")
        hard = tracker.stats_for("night_yard|night|-|n3|L6")
        self.assertEqual(easy["win_rate"], 1.0)
        self.assertAlmostEqual(hard["win_rate"], 0.1)
        self.assertEqual(tracker.total_episodes(), 20)

    def test_pooled_average_hides_what_per_condition_reveals(self):
        tracker = self._tracker()
        overall = tracker.overall()
        self.assertAlmostEqual(overall["win_rate"], 0.55)
        worst = tracker.worst_conditions(1)[0]
        self.assertEqual(worst["condition"], "night_yard|night|-|n3|L6")
        self.assertAlmostEqual(worst["win_rate"], 0.1)

    def test_window_bounds_memory(self):
        tracker = ConditionTracker(window=5)
        condition = Condition(map_id="open_field")
        for index in range(50):
            tracker.record(condition, reward=float(index), won=True, steps=10)
        stats = tracker.stats_for(condition)
        self.assertEqual(stats["episodes"], 50)
        self.assertEqual(stats["window"], 5)
        self.assertAlmostEqual(stats["mean_reward"], sum(range(45, 50)) / 5)

    def test_generalization_report_flags_condition_dependence(self):
        report = generalization_report(self._tracker())
        self.assertGreater(report["win_rate_spread"], 0.25)
        self.assertFalse(report["generalizes"])
        self.assertEqual(len(report["by_condition"]), 2)
        self.assertIn("night_yard", format_generalization_report(report))

    def test_generalization_report_accepts_a_uniform_policy(self):
        tracker = ConditionTracker(window=10)
        for map_id in ("open_field", "two_rooms", "compound"):
            condition = Condition(map_id=map_id, lighting="normal", enemy_count=1, level=6)
            for index in range(10):
                tracker.record(condition, reward=5.0, won=index < 6, steps=300)
        report = generalization_report(tracker)
        self.assertTrue(report["generalizes"])
        self.assertAlmostEqual(report["win_rate_spread"], 0.0)


class CheckpointRegistryTests(unittest.TestCase):
    def test_registration_and_lookup(self):
        registry = CheckpointRegistry()
        registry.register("brain_a", checkpoint="a.zip", frozen=False)
        registry.register("brain_b", checkpoint="b.zip", frozen=False)
        self.assertEqual(len(registry), 2)
        self.assertIn("brain_a", registry)
        self.assertEqual(registry.get("brain_b").checkpoint, "b.zip")
        with self.assertRaises(KeyError):
            registry.get("brain_c")

    def test_duplicate_policy_ids_are_rejected(self):
        registry = CheckpointRegistry()
        registry.register("brain_a")
        with self.assertRaises(ValueError):
            registry.register("brain_a")

    def test_snapshots_are_separate_policies_with_their_own_weights(self):
        registry = CheckpointRegistry()
        registry.register("brain_a", checkpoint="live.zip", frozen=False)
        first = registry.snapshot("brain_a", checkpoint="snap_1000.zip", step=1000)
        second = registry.snapshot("brain_a", checkpoint="snap_2000.zip", step=2000)
        self.assertNotEqual(first.policy_id, second.policy_id)
        self.assertNotEqual(first.checkpoint, second.checkpoint)
        self.assertTrue(first.frozen and second.frozen)
        self.assertEqual(registry.get("brain_a").checkpoint, "live.zip")
        self.assertEqual(registry.latest("brain_a").policy_id, second.policy_id)

    def test_registry_round_trips_through_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "league" / "registry.json"
            registry = CheckpointRegistry()
            registry.register("brain_a", checkpoint="a.zip", frozen=False, tags=["baseline"])
            registry.snapshot("brain_a", checkpoint="a_500.zip", step=500)
            registry.save(path)

            reloaded = CheckpointRegistry(path)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.get("brain_a").tags, ["baseline"])
            self.assertEqual(reloaded.get("brain_a@500").step, 500)


class LeagueTests(unittest.TestCase):
    def _league(self, strategy: str = "uniform", seed: int = 11) -> League:
        registry = CheckpointRegistry()
        registry.register("learner", checkpoint="live.zip", frozen=False)
        for step in (100, 200, 300):
            registry.snapshot("learner", checkpoint=f"snap_{step}.zip", step=step)
        return League(registry, strategy=strategy, seed=seed)

    def test_only_frozen_policies_are_sampled_as_opponents(self):
        league = self._league()
        for _ in range(30):
            opponent = league.sample_opponent("learner")
            self.assertIsNotNone(opponent)
            self.assertTrue(league.registry.get(opponent).frozen)

    def test_opponent_sampling_is_reproducible(self):
        first = [self._league().sample_opponent("learner") for _ in range(20)]
        second = [self._league().sample_opponent("learner") for _ in range(20)]
        self.assertEqual(first, second)

        league = self._league()
        drawn = [league.sample_opponent("learner") for _ in range(20)]
        league.reset_sampling()
        self.assertEqual(drawn, [league.sample_opponent("learner") for _ in range(20)])

    def test_latest_strategy_always_picks_the_newest_snapshot(self):
        league = self._league(strategy="latest")
        self.assertEqual(
            {league.sample_opponent("learner") for _ in range(10)}, {"learner@300"}
        )

    def test_uniform_strategy_eventually_uses_the_whole_pool(self):
        league = self._league(strategy="uniform")
        drawn = {league.sample_opponent("learner") for _ in range(100)}
        self.assertEqual(drawn, {"learner@100", "learner@200", "learner@300"})

    def test_prioritized_strategy_is_valid_and_deterministic(self):
        first = self._league(strategy="prioritized")
        second = self._league(strategy="prioritized")
        draws = [first.sample_opponent("learner") for _ in range(20)]
        self.assertEqual(draws, [second.sample_opponent("learner") for _ in range(20)])
        for policy_id in draws:
            self.assertIn(policy_id, ("learner@100", "learner@200", "learner@300"))

    def test_empty_pool_returns_no_opponent(self):
        registry = CheckpointRegistry()
        registry.register("only_learner", frozen=False)
        league = League(registry)
        self.assertIsNone(league.sample_opponent("only_learner"))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            League(CheckpointRegistry(), strategy="nonsense")

    def test_match_results_update_records_and_ratings(self):
        league = self._league()
        league.record_match("learner@100", "learner@200", 1.0)
        winner = league.registry.get("learner@100")
        loser = league.registry.get("learner@200")
        self.assertEqual((winner.wins, winner.losses), (1, 0))
        self.assertEqual((loser.wins, loser.losses), (0, 1))
        self.assertGreater(winner.elo, loser.elo)
        self.assertAlmostEqual(winner.elo + loser.elo, 2400.0, places=6)

    def test_draws_and_score_validation(self):
        league = self._league()
        league.record_match("learner@100", "learner@200", 0.5)
        self.assertEqual(league.registry.get("learner@100").draws, 1)
        self.assertAlmostEqual(league.registry.get("learner@100").elo, 1200.0)
        with self.assertRaises(ValueError):
            league.record_match("learner@100", "learner@200", 1.5)

    def test_elo_can_be_disabled_without_losing_win_loss_records(self):
        registry = CheckpointRegistry()
        registry.register("a")
        registry.register("b")
        league = League(registry, enable_elo=False)
        league.record_match("a", "b", 1.0)
        self.assertEqual(registry.get("a").wins, 1)
        self.assertAlmostEqual(registry.get("a").elo, 1200.0)

    def test_head_to_head_counts_both_orderings(self):
        league = self._league()
        league.record_match("learner@100", "learner@200", 1.0)
        league.record_match("learner@200", "learner@100", 1.0)
        record = league.head_to_head("learner@100", "learner@200")
        self.assertEqual((record["wins"], record["losses"]), (1, 1))

    def test_tournament_pairings_are_deterministic_and_ordered(self):
        pairings = League.tournament_pairings(["a", "b", "c"])
        self.assertEqual(len(pairings), 6)
        self.assertEqual(pairings, League.tournament_pairings(["a", "b", "c"]))
        self.assertIn(("a", "b"), pairings)
        self.assertIn(("b", "a"), pairings)
        self.assertNotIn(("a", "a"), pairings)
        self.assertEqual(len(League.tournament_pairings(["a", "b"], rounds=3)), 6)

    def test_standings_and_report(self):
        league = self._league()
        league.record_match("learner@100", "learner@200", 1.0)
        league.record_match("learner@100", "learner@300", 1.0)
        standings = league.standings()
        self.assertEqual(standings[0]["policy_id"], "learner@100")
        report = league.report()
        self.assertEqual(report["matches"], 2)
        self.assertEqual(report["policies"], 4)
        self.assertIn("learner@100", format_standings(standings))

    def test_expected_score_is_symmetric(self):
        self.assertAlmostEqual(expected_score(1200, 1200), 0.5)
        self.assertAlmostEqual(
            expected_score(1400, 1200) + expected_score(1200, 1400), 1.0, places=9
        )


class AutoCurriculumTests(unittest.TestCase):
    def _schedule(self, **overrides) -> CurriculumSchedule:
        defaults = dict(
            start_level=3,
            window=10,
            min_episodes_per_level=10,
            cooldown_episodes=5,
            promote_threshold=0.7,
            demote_threshold=0.3,
        )
        defaults.update(overrides)
        return CurriculumSchedule(**defaults)

    def test_invalid_schedules_are_rejected(self):
        with self.assertRaises(ValueError):
            CurriculumSchedule(promote_threshold=0.5, demote_threshold=0.5)
        with self.assertRaises(ValueError):
            CurriculumSchedule(start_level=99)
        with self.assertRaises(ValueError):
            CurriculumSchedule(window=0)

    def test_sustained_success_promotes_once_not_repeatedly(self):
        curriculum = AutoCurriculum(self._schedule())
        events = [curriculum.record(success=True) for _ in range(10)]
        changes = [event for event in events if event]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["kind"], "promote")
        self.assertEqual(curriculum.level, 4)
        self.assertEqual(curriculum.state.cooldown_remaining, 5)

    def test_a_short_lucky_streak_cannot_promote(self):
        curriculum = AutoCurriculum(self._schedule(min_episodes_per_level=30))
        for _ in range(20):
            self.assertIsNone(curriculum.record(success=True))
        self.assertEqual(curriculum.level, 3)

    def test_cooldown_blocks_an_immediate_second_promotion(self):
        curriculum = AutoCurriculum(self._schedule())
        for _ in range(10):
            curriculum.record(success=True)
        self.assertEqual(curriculum.level, 4)
        promotions_during_cooldown = [
            curriculum.record(success=True) for _ in range(4)
        ]
        self.assertTrue(all(event is None for event in promotions_during_cooldown))
        self.assertEqual(curriculum.level, 4)

    def test_sustained_failure_demotes(self):
        curriculum = AutoCurriculum(self._schedule())
        for _ in range(10):
            curriculum.record(success=False)
        self.assertEqual(curriculum.level, 2)
        self.assertEqual(curriculum.state.demotions, 1)

    def test_demotion_can_be_disabled(self):
        curriculum = AutoCurriculum(self._schedule(allow_demotion=False))
        for _ in range(40):
            curriculum.record(success=False)
        self.assertEqual(curriculum.level, 3)

    def test_levels_are_clamped_at_both_ends(self):
        top = AutoCurriculum(self._schedule(start_level=10, max_level=10))
        for _ in range(40):
            top.record(success=True)
        self.assertEqual(top.level, 10)

        bottom = AutoCurriculum(self._schedule(start_level=1, min_level=1))
        for _ in range(40):
            bottom.record(success=False)
        self.assertEqual(bottom.level, 1)

    def test_middling_performance_holds_the_level(self):
        curriculum = AutoCurriculum(self._schedule())
        for index in range(60):
            curriculum.record(success=index % 2 == 0)
        self.assertEqual(curriculum.level, 3)
        self.assertEqual(curriculum.state.promotions, 0)
        self.assertEqual(curriculum.state.demotions, 0)

    def test_identical_result_sequences_produce_identical_trajectories(self):
        pattern = [True, True, False, True, True, True, False, True, True, True] * 6

        def run() -> list[dict]:
            curriculum = AutoCurriculum(self._schedule())
            for success in pattern:
                curriculum.record(success=success)
            return curriculum.history()

        self.assertEqual(run(), run())

    def test_snapshot_exposes_the_decision_inputs(self):
        curriculum = AutoCurriculum(self._schedule())
        for _ in range(5):
            curriculum.record(success=True)
        snapshot = curriculum.snapshot()
        self.assertEqual(snapshot["level"], 3)
        self.assertEqual(snapshot["episodes_at_level"], 5)
        self.assertAlmostEqual(snapshot["success_rate"], 1.0)
        self.assertEqual(snapshot["window_size"], 5)


if __name__ == "__main__":
    unittest.main()
