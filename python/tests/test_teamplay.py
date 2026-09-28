"""Tests for the teamplay foundation (Phase 8)."""
from __future__ import annotations

import unittest

from sandboxai.teamplay import (
    COMMS_SYMBOLS,
    FORBIDDEN_REPORT_FIELDS,
    SOLO_TEAM,
    CommsChannel,
    CommsMessage,
    TeamConfig,
    TeamError,
    TeamOutcome,
    TeamRewardHooks,
    TeammateReport,
    build_teammate_reports,
    team_summary,
    validate_teammate_report,
)


class DefaultsTests(unittest.TestCase):
    def test_teamplay_is_disabled_by_default(self):
        config = TeamConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.teams, [SOLO_TEAM])
        self.assertFalse(config.comms_enabled)
        self.assertEqual(config.team_reward_weight, 0.0)

    def test_disabling_canonicalizes_every_slot_onto_the_solo_team(self):
        config = TeamConfig(enabled=False, slot_teams=[0, 1, 2], comms_enabled=True,
                            team_reward_weight=0.9)
        self.assertEqual(config.slot_teams, [SOLO_TEAM] * 3)
        self.assertFalse(config.comms_enabled)
        self.assertEqual(config.team_reward_weight, 0.0)

    def test_solo_reward_is_untouched_when_disabled(self):
        hooks = TeamRewardHooks(TeamConfig.solo())
        outcomes = [TeamOutcome(slot=0, reward=3.5, kills=2)]
        self.assertEqual(hooks.shaped_rewards(outcomes), {0: 3.5})

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            TeamConfig(slot_teams=[])
        with self.assertRaises(ValueError):
            TeamConfig(enabled=True, slot_teams=[0, -1])
        with self.assertRaises(ValueError):
            TeamConfig(objective="win_at_all_costs")
        with self.assertRaises(ValueError):
            TeamConfig(enabled=True, team_reward_weight=2.0)


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TeamConfig.versus(2)  # 2v2: slots 0,1 vs 2,3

    def test_versus_layout(self):
        self.assertTrue(self.config.enabled)
        self.assertEqual(self.config.slot_teams, [0, 0, 1, 1])
        self.assertEqual(self.config.teams, [0, 1])
        self.assertEqual(self.config.slots_of(0), [0, 1])
        self.assertEqual(self.config.slots_of(1), [2, 3])

    def test_teammates_exclude_self(self):
        self.assertEqual(self.config.teammates_of(0), [1])
        self.assertEqual(self.config.teammates_of(3), [2])

    def test_are_teammates(self):
        self.assertTrue(self.config.are_teammates(0, 1))
        self.assertFalse(self.config.are_teammates(1, 2))

    def test_three_versus_three(self):
        config = TeamConfig.versus(3)
        self.assertEqual(len(config.slot_teams), 6)
        self.assertEqual(config.slots_of(1), [3, 4, 5])

    def test_slot_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            self.config.team_of(9)

    def test_round_trip_through_dict(self):
        restored = TeamConfig.from_dict(self.config.to_dict())
        self.assertEqual(restored.slot_teams, self.config.slot_teams)
        self.assertEqual(restored.enabled, self.config.enabled)


class FriendlyFireTests(unittest.TestCase):
    def test_friendly_fire_off_blocks_teammate_damage(self):
        config = TeamConfig.versus(2, friendly_fire=False)
        self.assertFalse(config.can_damage(0, 1))
        self.assertTrue(config.can_damage(0, 2))

    def test_friendly_fire_on_allows_it(self):
        config = TeamConfig.versus(2, friendly_fire=True)
        self.assertTrue(config.can_damage(0, 1))

    def test_self_damage_is_never_allowed(self):
        config = TeamConfig.versus(2, friendly_fire=True)
        self.assertFalse(config.can_damage(1, 1))

    def test_penalty_only_exists_when_friendly_fire_is_on(self):
        on = TeamRewardHooks(TeamConfig.versus(2, friendly_fire=True))
        off = TeamRewardHooks(TeamConfig.versus(2, friendly_fire=False))
        self.assertEqual(on.friendly_fire_penalty(0, 1, 30.0), 30.0)
        self.assertEqual(on.friendly_fire_penalty(0, 2, 30.0), 0.0)
        self.assertEqual(off.friendly_fire_penalty(0, 1, 30.0), 0.0)


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TeamConfig.versus(2, comms_enabled=True, comms_budget=2)
        self.channel = CommsChannel(self.config)

    def test_messages_never_cross_teams(self):
        self.channel.send(0, "contact", 0.3)
        self.channel.send(2, "help", -0.2)
        team_zero = self.channel.inbox(1)
        team_one = self.channel.inbox(3)
        self.assertEqual([m.sender_slot for m in team_zero], [0])
        self.assertEqual([m.sender_slot for m in team_one], [2])

    def test_a_sender_does_not_receive_its_own_message(self):
        self.channel.send(0, "contact")
        self.assertEqual(self.channel.inbox(0), [])

    def test_budget_is_enforced_per_tick(self):
        self.assertIsNotNone(self.channel.send(0, "contact"))
        self.assertIsNotNone(self.channel.send(0, "moving"))
        self.assertIsNone(self.channel.send(0, "help"))
        self.channel.advance()
        self.assertIsNotNone(self.channel.send(0, "help"))

    def test_messages_are_transient(self):
        self.channel.send(0, "contact")
        self.assertEqual(self.channel.pending(), 1)
        self.channel.advance()
        self.assertEqual(self.channel.pending(), 0)
        self.assertEqual(self.channel.inbox(1), [])

    def test_comms_are_silent_when_disabled(self):
        channel = CommsChannel(TeamConfig.versus(2, comms_enabled=False))
        self.assertIsNone(channel.send(0, "contact"))
        self.assertEqual(channel.inbox(1), [])

    def test_comms_are_silent_when_teamplay_is_off(self):
        channel = CommsChannel(TeamConfig())
        self.assertIsNone(channel.send(0, "contact"))

    def test_unknown_symbol_is_rejected(self):
        with self.assertRaises(ValueError):
            CommsMessage(sender_slot=0, team=0, symbol="flank_left_in_three_seconds")

    def test_bearing_is_range_checked(self):
        with self.assertRaises(ValueError):
            CommsMessage(sender_slot=0, team=0, symbol="contact", bearing_norm=4.0)

    def test_vocabulary_is_small(self):
        # A large vocabulary is a continuous channel in disguise.
        self.assertLessEqual(len(COMMS_SYMBOLS), 16)

    def test_reset_clears_the_channel(self):
        self.channel.send(0, "contact")
        self.channel.advance()
        self.channel.reset()
        self.assertEqual(self.channel.tick, 0)
        self.assertEqual(self.channel.pending(), 0)


class TeammateInformationTests(unittest.TestCase):
    def test_reports_only_reach_teammates(self):
        config = TeamConfig.versus(2)
        perceptions = {
            1: {"contact_visible": True, "contact_bearing_norm": 0.4, "contact_confidence": 0.9},
            2: {"contact_visible": True, "contact_bearing_norm": -0.9, "contact_confidence": 1.0},
        }
        reports = build_teammate_reports(config, 0, perceptions)
        self.assertEqual([report.slot for report in reports], [1])

    def test_no_reports_when_teamplay_is_off(self):
        self.assertEqual(build_teammate_reports(TeamConfig(), 0, {1: {}}), [])

    def test_a_report_carrying_ground_truth_is_rejected(self):
        config = TeamConfig.versus(2)
        for forbidden in sorted(FORBIDDEN_REPORT_FIELDS):
            with self.assertRaises(TeamError, msg=forbidden):
                build_teammate_reports(config, 0, {1: {forbidden: 1.0}})

    def test_an_unknown_report_field_is_rejected(self):
        with self.assertRaises(TeamError):
            validate_teammate_report({"wallhack": True})

    def test_confidence_decays_with_age(self):
        fresh = TeammateReport(slot=1, team=0, contact_confidence=1.0, age=0.0)
        stale = TeammateReport(slot=1, team=0, contact_confidence=1.0, age=8.0)
        self.assertAlmostEqual(fresh.decayed(4.0).contact_confidence, 1.0)
        self.assertAlmostEqual(stale.decayed(4.0).contact_confidence, 0.25)

    def test_report_has_no_field_for_absolute_enemy_state(self):
        fields = set(TeammateReport(slot=1, team=0).to_dict())
        self.assertFalse(fields & FORBIDDEN_REPORT_FIELDS)


class TeamRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outcomes = [
            TeamOutcome(slot=0, reward=1.0, kills=1, alive=True),
            TeamOutcome(slot=1, reward=-1.0, kills=0, alive=False, deaths=1),
            TeamOutcome(slot=2, reward=0.5, kills=0, alive=True),
            TeamOutcome(slot=3, reward=0.5, kills=0, alive=True),
        ]

    def test_weight_zero_leaves_individual_rewards_alone(self):
        hooks = TeamRewardHooks(TeamConfig.versus(2, team_reward_weight=0.0))
        self.assertEqual(
            hooks.shaped_rewards(self.outcomes), {0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5}
        )

    def test_shaping_blends_individual_and_team_score(self):
        hooks = TeamRewardHooks(TeamConfig.versus(2, team_reward_weight=0.5))
        shaped = hooks.shaped_rewards(self.outcomes)
        # team 0 scored 1 kill, team 1 scored none
        self.assertAlmostEqual(shaped[0], 0.5 * 1.0 + 0.5 * 1.0)
        self.assertAlmostEqual(shaped[1], 0.5 * -1.0 + 0.5 * 1.0)
        self.assertAlmostEqual(shaped[2], 0.5 * 0.5 + 0.5 * 0.0)

    def test_survive_objective_scores_alive_fraction(self):
        hooks = TeamRewardHooks(TeamConfig.versus(2, objective="survive"))
        self.assertAlmostEqual(hooks.team_score(self.outcomes, 0), 0.5)
        self.assertAlmostEqual(hooks.team_score(self.outcomes, 1), 1.0)

    def test_control_and_explore_objectives_use_progress(self):
        outcomes = [
            TeamOutcome(slot=0, objective_progress=0.4),
            TeamOutcome(slot=1, objective_progress=0.6),
            TeamOutcome(slot=2),
            TeamOutcome(slot=3),
        ]
        control = TeamRewardHooks(TeamConfig.versus(2, objective="control"))
        explore = TeamRewardHooks(TeamConfig.versus(2, objective="explore"))
        self.assertAlmostEqual(control.team_score(outcomes, 0), 0.5)
        self.assertAlmostEqual(explore.team_score(outcomes, 0), 1.0)

    def test_summary_reports_per_team(self):
        summary = team_summary(TeamConfig.versus(2), self.outcomes)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["teams"]["0"]["slots"], [0, 1])
        self.assertEqual(summary["teams"]["0"]["kills"], 1)
        self.assertEqual(summary["teams"]["0"]["alive"], 1)
        self.assertEqual(summary["teams"]["1"]["alive"], 2)


if __name__ == "__main__":
    unittest.main()
