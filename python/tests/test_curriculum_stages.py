"""Tests for the integrated curriculum ladder (Phase 12)."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from sandboxai.conditions import LIGHTING_IDS, MAP_IDS
from sandboxai.curriculum_stages import (
    EXPLORATION_STAGE,
    PROMOTION_METRICS,
    STAGES,
    STAGES_BY_LEVEL,
    CurriculumDirector,
    EpisodeOutcome,
    describe_progression,
    distribution_for,
    format_progression,
    stage_for,
)
from sandboxai.randomization import EpisodePlan

REPO_ROOT = Path(__file__).resolve().parents[2]


class StageTableTests(unittest.TestCase):
    def test_eleven_levels_numbered_one_to_eleven(self):
        self.assertEqual([stage.level for stage in STAGES], list(range(1, 12)))

    def test_stage_names_are_unique(self):
        names = [stage.name for stage in STAGES]
        self.assertEqual(len(set(names)), len(names))

    def test_levels_match_the_gdscript_curriculum_enum(self):
        source = (REPO_ROOT / "scripts/core/curriculum_config.gd").read_text(encoding="utf-8")
        block = re.search(r"enum Level \{(.*?)\}", source, re.S)
        self.assertIsNotNone(block, "could not find the Level enum")
        levels = {
            int(value): name.lower()
            for name, value in re.findall(r"(\w+)\s*=\s*(\d+)", block.group(1))
        }
        self.assertEqual(sorted(levels), sorted(STAGES_BY_LEVEL))

    def test_first_stage_is_movement_and_last_is_self_play(self):
        self.assertIn("movement", STAGES[0].systems)
        self.assertTrue(STAGES[-1].self_play)
        self.assertEqual(STAGES[-1].level, 11)
        # Nothing before the final stage claims self-play.
        self.assertFalse(any(stage.self_play for stage in STAGES[:-1]))

    def test_thresholds_are_sane(self):
        for stage in STAGES + (EXPLORATION_STAGE,):
            with self.subTest(stage=stage.name):
                self.assertIn(stage.promotion_metric, PROMOTION_METRICS)
                self.assertGreater(stage.promote_at, stage.demote_at)
                self.assertTrue(0.0 < stage.promote_at <= 1.0)
                self.assertTrue(0.0 <= stage.demote_at < 1.0)
                self.assertGreaterEqual(stage.min_episodes, 20)
                self.assertTrue(stage.maps)
                self.assertTrue(stage.lightings)

    def test_every_map_and_lighting_id_is_real(self):
        for stage in STAGES + (EXPLORATION_STAGE,):
            for map_id in stage.maps:
                self.assertIn(map_id, MAP_IDS, f"{stage.name}: {map_id}")
            for lighting in stage.lightings:
                self.assertIn(lighting, LIGHTING_IDS, f"{stage.name}: {lighting}")

    def test_scenarios_exist_in_the_scenario_library(self):
        source = (REPO_ROOT / "scripts/scenario/scenario_library.gd").read_text(encoding="utf-8")
        for stage in STAGES + (EXPLORATION_STAGE,):
            for scenario in stage.scenarios:
                self.assertIn(f'"{scenario}"', source, f"{stage.name}: {scenario}")

    def test_difficulty_broadly_increases(self):
        # Later levels must not be gated more leniently on evidence.
        episodes = [stage.min_episodes for stage in STAGES]
        self.assertEqual(episodes, sorted(episodes))
        # And the promotion bar must never rise as levels get harder.
        bars = [stage.promote_at for stage in STAGES[2:-1]]
        self.assertEqual(bars, sorted(bars, reverse=True))

    def test_perception_systems_appear_at_the_documented_levels(self):
        self.assertNotIn("perception", STAGES_BY_LEVEL[4].systems)
        self.assertIn("fov", STAGES_BY_LEVEL[6].systems)
        self.assertIn("sound", STAGES_BY_LEVEL[7].systems)
        self.assertIn("memory", STAGES_BY_LEVEL[8].systems)

    def test_exploration_stage_is_off_the_ladder(self):
        self.assertTrue(EXPLORATION_STAGE.exploration)
        self.assertEqual(EXPLORATION_STAGE.promotion_metric, "coverage")
        self.assertNotIn(EXPLORATION_STAGE, STAGES)
        self.assertEqual(EXPLORATION_STAGE.enemy_counts, (0,))

    def test_stage_for_clamps(self):
        self.assertEqual(stage_for(0).level, 1)
        self.assertEqual(stage_for(99).level, 11)
        self.assertEqual(stage_for(7).level, 7)

    def test_describe_and_format(self):
        described = describe_progression()
        self.assertEqual(len(described), len(STAGES))
        self.assertEqual(described[0]["level"], 1)
        text = format_progression()
        self.assertIn("self_play", text)
        self.assertEqual(len(text.strip().splitlines()), len(STAGES) + 2)


class DistributionTests(unittest.TestCase):
    def test_distribution_only_produces_the_stage_level(self):
        distribution = distribution_for(STAGES_BY_LEVEL[6], master_seed=11)
        for index in range(50):
            plan = distribution.episode_plan(index)
            self.assertEqual(plan.condition.level, 6)
            self.assertIn(plan.condition.map_id, STAGES_BY_LEVEL[6].maps)
            self.assertIn(plan.condition.enemy_count, STAGES_BY_LEVEL[6].enemy_counts)

    def test_non_randomized_stage_uses_one_layout_variant(self):
        self.assertEqual(distribution_for(STAGES_BY_LEVEL[1]).layout_variants, 1)
        self.assertGreater(distribution_for(STAGES_BY_LEVEL[10]).layout_variants, 1)

    def test_randomized_stage_actually_varies_the_map(self):
        distribution = distribution_for(STAGES_BY_LEVEL[10], master_seed=5)
        maps = {distribution.episode_plan(index).condition.map_id for index in range(200)}
        self.assertGreater(len(maps), 5)


class DirectorTests(unittest.TestCase):
    def test_starts_at_level_one_by_default(self):
        director = CurriculumDirector()
        self.assertEqual(director.level, 1)
        self.assertEqual(director.stage.name, "movement_and_aim")

    def test_plans_are_reproducible_for_the_same_seed(self):
        first = [CurriculumDirector(master_seed=99).next_episode().key for _ in range(1)]
        second = CurriculumDirector(master_seed=99)
        self.assertEqual(first[0], second.next_episode().key)

    def test_peek_does_not_consume(self):
        director = CurriculumDirector(master_seed=3)
        peeked = director.peek().to_dict()
        self.assertEqual(peeked, director.peek().to_dict())
        self.assertEqual(peeked, director.next_episode().to_dict())
        self.assertNotEqual(peeked, director.next_episode().to_dict())

    def test_next_episode_returns_a_plan_with_engine_commands(self):
        plan = CurriculumDirector(master_seed=3).next_episode()
        self.assertIsInstance(plan, EpisodePlan)
        commands = plan.environment_commands()
        self.assertTrue(commands)
        self.assertEqual(commands[-1]["call"], "reset")

    def test_no_promotion_from_a_single_lucky_episode(self):
        director = CurriculumDirector(master_seed=1)
        change = director.record(EpisodeOutcome(won=True, reward=10.0))
        self.assertIsNone(change)
        self.assertEqual(director.level, 1)

    def test_promotion_requires_the_minimum_episode_count(self):
        director = CurriculumDirector(master_seed=1)
        minimum = director.stage.min_episodes
        for _ in range(minimum - 1):
            self.assertIsNone(director.record(EpisodeOutcome(won=True, reward=1.0)))
        self.assertEqual(director.level, 1)
        change = director.record(EpisodeOutcome(won=True, reward=1.0))
        self.assertIsNotNone(change)
        self.assertEqual(change["kind"], "promote")
        self.assertEqual(director.level, 2)

    def test_promotion_switches_the_stage_and_the_distribution(self):
        director = CurriculumDirector(master_seed=1)
        before = director.distribution
        for _ in range(director.stage.min_episodes):
            director.record(EpisodeOutcome(won=True, reward=1.0))
        self.assertEqual(director.stage.name, "moving_targets")
        self.assertIsNot(director.distribution, before)
        self.assertEqual(director.next_episode().condition.level, 2)

    def test_a_streak_cannot_skip_two_levels_at_once(self):
        director = CurriculumDirector(master_seed=1)
        levels = []
        for _ in range(60):
            director.record(EpisodeOutcome(won=True, reward=1.0))
            levels.append(director.level)
        # Monotone and never jumping by more than one.
        for previous, current in zip(levels, levels[1:]):
            self.assertLessEqual(current - previous, 1)

    def test_demotion_on_sustained_failure(self):
        director = CurriculumDirector(start_level=5, master_seed=1)
        self.assertEqual(director.level, 5)
        change = None
        for _ in range(200):
            change = director.record(EpisodeOutcome(won=False, reward=-1.0))
            if change:
                break
        self.assertIsNotNone(change)
        self.assertEqual(change["kind"], "demote")
        self.assertEqual(director.level, 4)
        self.assertEqual(change["stage_from"], "cover")
        self.assertEqual(change["stage_to"], "multi_enemy_combat")

    def test_level_one_cannot_be_demoted(self):
        director = CurriculumDirector(master_seed=1)
        for _ in range(200):
            self.assertIsNone(director.record(EpisodeOutcome(won=False)))
        self.assertEqual(director.level, 1)

    def test_self_play_level_is_reachable(self):
        director = CurriculumDirector(start_level=10, master_seed=1)
        for _ in range(400):
            director.record(EpisodeOutcome(won=True, reward=1.0))
            if director.level == 11:
                break
        self.assertEqual(director.level, 11)
        self.assertTrue(director.stage.self_play)
        # And the ladder stops there.
        for _ in range(400):
            director.record(EpisodeOutcome(won=True, reward=1.0))
        self.assertEqual(director.level, 11)

    def test_re_entering_a_level_does_not_replay_the_same_episodes(self):
        director = CurriculumDirector(start_level=5, master_seed=1)
        first_visit = [director.next_episode().to_dict() for _ in range(5)]
        for _ in range(200):
            director.next_episode()
            if director.record(EpisodeOutcome(won=False)):
                break
        self.assertEqual(director.level, 4)
        for _ in range(400):
            director.next_episode()
            if director.record(EpisodeOutcome(won=True, reward=1.0)):
                break
        self.assertEqual(director.level, 5)
        second_visit = [director.next_episode().to_dict() for _ in range(5)]
        self.assertNotEqual(first_visit, second_visit)
        self.assertNotEqual(
            [entry["index"] for entry in first_visit],
            [entry["index"] for entry in second_visit],
        )

    def test_exploration_director_is_gated_on_coverage(self):
        director = CurriculumDirector(exploration=True, master_seed=2)
        self.assertEqual(director.stage.promotion_metric, "coverage")
        for _ in range(200):
            self.assertIsNone(director.record(EpisodeOutcome(won=True, coverage=0.1)))
        # Winning does nothing here; only coverage counts, and the stage
        # is off the ladder so it never promotes or demotes.
        self.assertEqual(director.stage.name, "map_analyzer")
        self.assertEqual(director.level, EXPLORATION_STAGE.level)
        self.assertEqual(director.history(), [])
        for _ in range(200):
            self.assertIsNone(director.record(EpisodeOutcome(coverage=0.99)))
        self.assertEqual(director.stage.name, "map_analyzer")

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            EpisodeOutcome().metric("vibes")

    def test_snapshot_reports_the_stage_and_the_gate(self):
        director = CurriculumDirector(master_seed=1)
        director.next_episode()
        director.record(EpisodeOutcome(won=True))
        snapshot = director.snapshot()
        self.assertEqual(snapshot["level"], 1)
        self.assertEqual(snapshot["stage"], "movement_and_aim")
        self.assertEqual(snapshot["episodes_drawn"], 1)
        self.assertEqual(snapshot["promotion_metric"], "win_rate")
        self.assertFalse(snapshot["self_play"])

    def test_history_records_every_change(self):
        director = CurriculumDirector(master_seed=1)
        for _ in range(120):
            director.record(EpisodeOutcome(won=True, reward=1.0))
        history = director.history()
        self.assertTrue(history)
        self.assertTrue(all(entry["kind"] == "promote" for entry in history))
        self.assertEqual(
            [entry["to_level"] for entry in history],
            list(range(2, 2 + len(history))),
        )


if __name__ == "__main__":
    unittest.main()
