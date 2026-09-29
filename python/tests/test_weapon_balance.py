"""Weapon balance regression tests.

These assert the *design intent* of the weapon roster rather than the
literal numbers: which weapon owns which engagement band, that damage
falloff is well-formed, that handling makes sprays worse than bursts where
it should, and that no profile quietly dominates the others.

They read the numbers straight out of the GDScript via
``sandboxai.weapons``, so retuning a profile in the engine either keeps
these properties or fails here with a specific explanation.
"""
from __future__ import annotations

import math
import unittest

from sandboxai.weapons import (
    RANGE_BANDS,
    load_handling_constants,
    load_profiles,
    role_ranking,
    ttk_table,
)

ENEMY_HEALTH = 100.0


class WeaponTableParsingTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles()

    def test_parses_the_four_shipped_profiles(self):
        self.assertEqual({"rifle", "shotgun", "pistol", "smg"}, set(self.profiles))

    def test_every_profile_is_internally_consistent(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertGreater(weapon.damage, 0.0, "damage must be positive")
                self.assertGreater(weapon.range_m, 0.0)
                self.assertGreater(weapon.cooldown_time, 0.0)
                self.assertGreater(weapon.hit_radius, 0.0)
                self.assertGreaterEqual(weapon.projectile_count, 1)
                self.assertIn(weapon.fire_mode, ("auto", "semi", "pump"))
                self.assertGreater(weapon.magazine_size, 0)
                self.assertGreater(weapon.reload_time, 0.0)
                self.assertGreaterEqual(weapon.headshot_multiplier, 1.0)
                self.assertGreaterEqual(weapon.minimum_damage_scale, 0.0)
                self.assertLessEqual(weapon.minimum_damage_scale, 1.0)
                self.assertLessEqual(
                    weapon.falloff_start_m,
                    weapon.range_m,
                    "falloff cannot start beyond maximum range",
                )
                self.assertGreater(
                    0.0 if weapon.move_speed_scale_firing <= 0 else weapon.move_speed_scale_firing,
                    0.0,
                )
                self.assertLessEqual(
                    weapon.move_speed_scale_firing,
                    1.0,
                    "firing must never make you faster",
                )

    def test_handling_constants_are_present_and_sane(self):
        consts = load_handling_constants()
        for key in (
            "RECOIL_RECOVERY_DELAY",
            "RECOIL_MAX_DEG",
            "RECOIL_PATTERN_LENGTH",
            "RECOIL_SUSTAIN_SCALE",
            "SPREAD_SETTLED_EPSILON",
            "HANDLING_GOLDEN_ANGLE",
        ):
            self.assertIn(key, consts, f"{key} missing from SandboxConfig")
        self.assertGreater(consts["RECOIL_MAX_DEG"], 0.0)
        self.assertGreater(consts["RECOIL_RECOVERY_DELAY"], 0.0)
        self.assertGreaterEqual(consts["RECOIL_PATTERN_LENGTH"], 1.0)
        self.assertGreater(consts["SPREAD_SETTLED_EPSILON"], 0.0)
        self.assertLess(
            consts["SPREAD_SETTLED_EPSILON"],
            0.1,
            "the settled threshold must be tight enough to mean 'pinpoint'",
        )


class DamageFalloffTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles()

    def test_falloff_is_monotonically_non_increasing(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                previous = math.inf
                for step in range(0, 301):
                    distance = step * 0.1
                    scale = weapon.damage_scale_at(distance)
                    self.assertLessEqual(
                        scale, previous + 1e-9, f"damage rose at {distance:.1f} m"
                    )
                    self.assertGreaterEqual(scale, weapon.minimum_damage_scale - 1e-9)
                    self.assertLessEqual(scale, 1.0 + 1e-9)
                    previous = scale

    def test_full_damage_inside_the_falloff_start(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertAlmostEqual(weapon.damage_scale_at(0.0), 1.0)
                self.assertAlmostEqual(weapon.damage_scale_at(weapon.falloff_start_m), 1.0)

    def test_minimum_damage_at_and_beyond_maximum_range(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertAlmostEqual(
                    weapon.damage_scale_at(weapon.range_m), weapon.minimum_damage_scale
                )
                self.assertAlmostEqual(
                    weapon.damage_scale_at(weapon.range_m * 4.0),
                    weapon.minimum_damage_scale,
                )

    def test_shotgun_falls_off_hardest_and_soonest(self):
        # The shotgun is the roster's close-range specialist: it must lose
        # its damage faster than anything else or it becomes a rifle.
        shotgun = self.profiles["shotgun"]
        others = [w for name, w in self.profiles.items() if name != "shotgun"]
        for other in others:
            with self.subTest(other=other.profile_id):
                self.assertLessEqual(shotgun.range_m, other.range_m)
                self.assertLessEqual(shotgun.minimum_damage_scale, other.minimum_damage_scale)


class RoleSeparationTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles()
        self.ranking = role_ranking(self.profiles, ENEMY_HEALTH)

    def test_shotgun_wins_point_blank(self):
        best = self.ranking["point_blank"][0][0]
        self.assertEqual(best, "shotgun", f"point blank ranking: {self.ranking['point_blank']}")

    def test_shotgun_one_shots_at_contact_range(self):
        shotgun = self.profiles["shotgun"]
        self.assertEqual(shotgun.shots_to_kill(1.0, ENEMY_HEALTH), 1)
        self.assertAlmostEqual(shotgun.ttk(1.0, ENEMY_HEALTH), 0.0)

    def test_rifle_is_the_only_long_range_option(self):
        _, low, high = RANGE_BANDS[-1]
        probe = (low + high) / 2.0
        reachable = sorted(
            name for name, weapon in self.profiles.items() if weapon.range_m >= probe
        )
        self.assertEqual(
            reachable, ["rifle"], f"only the rifle should reach {probe:.1f} m"
        )
        far = sorted(
            name for name, weapon in self.profiles.items() if weapon.range_m >= high
        )
        self.assertEqual(far, ["rifle"], f"only the rifle should reach {high:.1f} m")

    def test_no_profile_wins_every_band(self):
        winners = {band: entries[0][0] for band, entries in self.ranking.items()}
        self.assertGreater(
            len(set(winners.values())),
            1,
            f"a single weapon dominates every range band: {winners}",
        )

    def test_every_profile_is_the_best_choice_somewhere(self):
        # Not necessarily per band -- the roster is small -- but every
        # weapon must beat every other weapon at *some* legal distance, or
        # it is strictly dominated and there is no reason to ever pick it.
        distances = [d * 0.5 for d in range(1, 31)]
        for name, weapon in self.profiles.items():
            for other_name, other in self.profiles.items():
                if name == other_name:
                    continue
                with self.subTest(weapon=name, versus=other_name):
                    wins = any(
                        d <= weapon.range_m
                        and weapon.effective_ttk(d, ENEMY_HEALTH)
                        < other.effective_ttk(d, ENEMY_HEALTH)
                        for d in distances
                    )
                    self.assertTrue(
                        wins, f"{name} never beats {other_name} at any distance"
                    )

    def test_close_range_weapons_cannot_reach_long_range(self):
        for name in ("shotgun", "smg", "pistol"):
            with self.subTest(weapon=name):
                self.assertTrue(
                    math.isinf(self.profiles[name].effective_ttk(14.0, ENEMY_HEALTH)),
                    f"{name} should not be able to kill at 14 m",
                )

    def test_fire_modes_are_distinct_across_the_roster(self):
        modes = {weapon.fire_mode for weapon in self.profiles.values()}
        self.assertGreaterEqual(
            len(modes), 3, f"roster should exercise several fire modes, got {modes}"
        )


class TimeToKillTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles()

    def test_ttk_is_never_negative_and_finite_in_range(self):
        for name, weapon in self.profiles.items():
            for step in range(0, int(weapon.range_m * 2) + 1):
                distance = step * 0.5
                with self.subTest(weapon=name, distance=distance):
                    value = weapon.ttk(distance, ENEMY_HEALTH)
                    self.assertGreaterEqual(value, 0.0)
                    self.assertTrue(math.isfinite(value))

    def test_ttk_is_non_decreasing_with_distance(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                previous = -1.0
                for step in range(0, int(weapon.range_m * 2) + 1):
                    value = weapon.ttk(step * 0.5, ENEMY_HEALTH)
                    self.assertGreaterEqual(
                        value, previous - 1e-9, "TTK improved with distance"
                    )
                    previous = value

    def test_in_range_ttk_stays_inside_a_playable_window(self):
        # Too fast and the episode is decided by who peeks first; too slow
        # and credit assignment over a 60 s episode falls apart.
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                worst = weapon.ttk(weapon.range_m, ENEMY_HEALTH)
                self.assertLessEqual(
                    worst, 6.0, f"{name} takes {worst:.2f}s to kill at its own max range"
                )

    def test_headshots_are_faster_than_body_shots(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                if weapon.headshot_multiplier <= 1.0:
                    self.skipTest("profile has no headshot bonus")
                body = weapon.ttk(2.0, ENEMY_HEALTH)
                head = weapon.ttk(2.0, ENEMY_HEALTH, headshot=True)
                self.assertLessEqual(head, body)
                far_body = weapon.ttk(weapon.range_m, ENEMY_HEALTH)
                far_head = weapon.ttk(weapon.range_m, ENEMY_HEALTH, headshot=True)
                self.assertLess(
                    far_head,
                    far_body,
                    f"{name} headshots must pay off at range",
                )

    def test_headshot_multiplier_cannot_trivialise_a_fight(self):
        # A headshot should reward aim, not delete the skill ceiling: no
        # profile may go from a multi-shot kill to a one-shot kill at its
        # maximum range purely from the head multiplier.
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                if weapon.profile_id == "shotgun":
                    continue  # already a one-shot up close by design
                shots = weapon.shots_to_kill(weapon.range_m, ENEMY_HEALTH, headshot=True)
                self.assertGreaterEqual(
                    shots, 2, f"{name} one-shots with a headshot at max range"
                )

    def test_sustained_ttk_never_beats_ideal_ttk(self):
        for name, weapon in self.profiles.items():
            for step in range(0, int(weapon.range_m * 2) + 1):
                distance = step * 0.5
                with self.subTest(weapon=name, distance=distance):
                    self.assertGreaterEqual(
                        weapon.sustained_ttk(distance, ENEMY_HEALTH),
                        weapon.ttk(distance, ENEMY_HEALTH) - 1e-9,
                    )

    def test_magazine_forces_a_reload_only_on_long_fights(self):
        # Every weapon must be able to secure at least one kill on a full
        # magazine at its optimal range, otherwise the reload is not a
        # tactical decision, it is a tax.
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                optimal = min(weapon.falloff_start_m, weapon.range_m) * 0.5
                shots = weapon.shots_to_kill(optimal, ENEMY_HEALTH)
                self.assertLessEqual(
                    shots,
                    weapon.magazine_size,
                    f"{name} cannot kill at {optimal:.1f} m without reloading",
                )


class HandlingTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles()
        self.constants = load_handling_constants()

    def test_bloom_only_ever_widens_the_cone(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                previous = -1.0
                for shots in range(0, 40):
                    cone = weapon.spread_after(shots)
                    self.assertGreaterEqual(cone, previous - 1e-9)
                    self.assertLessEqual(cone, max(weapon.spread_max_deg, 1e-9) + 1e-9)
                    previous = cone

    def test_first_shot_from_rest_is_pinpoint(self):
        epsilon = self.constants["SPREAD_SETTLED_EPSILON"]
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertLessEqual(
                    weapon.spread_after(0, speed_fraction=0.0),
                    epsilon,
                    f"{name} is not pinpoint on the first settled shot",
                )

    def test_moving_while_firing_costs_accuracy(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertGreater(
                    weapon.spread_after(0, speed_fraction=1.0),
                    weapon.spread_after(0, speed_fraction=0.0),
                    f"{name} shoots just as well at a sprint",
                )

    def test_firing_slows_the_shooter(self):
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                self.assertLess(
                    weapon.move_speed_scale_firing,
                    1.0,
                    f"{name} does not plant the shooter while firing",
                )

    def test_recoil_climbs_then_saturates(self):
        pattern = self.constants["RECOIL_PATTERN_LENGTH"]
        sustain = self.constants["RECOIL_SUSTAIN_SCALE"]
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                first_pitch, _ = weapon.recoil_kick(0, self.constants)
                late_pitch, _ = weapon.recoil_kick(int(pattern) * 3, self.constants)
                self.assertAlmostEqual(first_pitch, weapon.recoil_vertical_deg, places=6)
                self.assertAlmostEqual(
                    late_pitch, weapon.recoil_vertical_deg * sustain, places=6
                )
                self.assertLessEqual(
                    late_pitch,
                    first_pitch + 1e-9,
                    "sustained fire should not kick harder than the first shot",
                )

    def test_horizontal_recoil_is_deterministic_and_two_sided(self):
        # No RNG anywhere: the same shot index always produces the same
        # kick, and the pattern must wander both ways rather than drifting
        # in one direction the policy could simply pre-compensate once.
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                if weapon.recoil_horizontal_deg <= 0.0:
                    continue
                kicks = [weapon.recoil_kick(i, self.constants)[1] for i in range(16)]
                repeat = [weapon.recoil_kick(i, self.constants)[1] for i in range(16)]
                self.assertEqual(kicks, repeat, "recoil is not reproducible")
                self.assertTrue(any(k > 0 for k in kicks), "never kicks right")
                self.assertTrue(any(k < 0 for k in kicks), "never kicks left")

    def test_accumulated_recoil_is_bounded(self):
        limit = self.constants["RECOIL_MAX_DEG"]
        for name, weapon in self.profiles.items():
            with self.subTest(weapon=name):
                accumulated = 0.0
                for index in range(200):
                    pitch, _ = weapon.recoil_kick(index, self.constants)
                    accumulated = min(limit, max(-limit, accumulated + pitch))
                self.assertLessEqual(abs(accumulated), limit + 1e-9)

    def test_effective_ttk_is_never_better_than_ideal(self):
        for name, weapon in self.profiles.items():
            for step in range(1, int(weapon.range_m * 2) + 1):
                distance = step * 0.5
                with self.subTest(weapon=name, distance=distance):
                    self.assertGreaterEqual(
                        weapon.effective_ttk(distance, ENEMY_HEALTH),
                        weapon.ttk(distance, ENEMY_HEALTH) - 1e-9,
                        "bloom made the weapon better than a perfect cone",
                    )

    def test_moving_fire_is_never_better_than_planted_fire(self):
        for name, weapon in self.profiles.items():
            for step in range(1, int(weapon.range_m * 2) + 1):
                distance = step * 0.5
                with self.subTest(weapon=name, distance=distance):
                    planted = weapon.effective_ttk(distance, ENEMY_HEALTH, speed_fraction=0.0)
                    moving = weapon.effective_ttk(distance, ENEMY_HEALTH, speed_fraction=1.0)
                    self.assertGreaterEqual(moving, planted - 1e-9)

    def test_spraying_the_smg_at_range_is_punished(self):
        # The SMG is the only profile whose cooldown is shorter than the
        # recoil recovery delay, so it is the one weapon that can actually
        # out-run its own bloom. At the edge of its range that must cost
        # real time, otherwise "hold the trigger" is the optimal policy.
        smg = self.profiles["smg"]
        edge = smg.range_m
        ideal = smg.ttk(edge, ENEMY_HEALTH)
        sprayed = smg.effective_ttk(edge, ENEMY_HEALTH)
        self.assertGreater(
            sprayed,
            ideal * 1.25,
            f"spraying the SMG at {edge} m costs almost nothing ({sprayed:.2f}s vs {ideal:.2f}s)",
        )

    def test_burst_discipline_beats_spraying_at_the_smg_range_edge(self):
        smg = self.profiles["smg"]
        edge = smg.range_m
        sprayed = smg.effective_ttk(edge, ENEMY_HEALTH)
        bursted = min(
            smg.burst_ttk(edge, burst, pause, ENEMY_HEALTH)
            for burst in (3, 4, 5, 6)
            for pause in (0.2, 0.3, 0.4)
        )
        self.assertLess(
            bursted,
            sprayed,
            f"no burst pattern beats spraying ({bursted:.2f}s vs {sprayed:.2f}s)",
        )

    def test_slow_firing_profiles_fully_recover_between_shots(self):
        # The rifle and pistol fire slowly enough that their cone collapses
        # between rounds; that is what makes them the precision options.
        for name in ("rifle", "pistol"):
            weapon = self.profiles[name]
            with self.subTest(weapon=name):
                decay = weapon._bloom_decay_per_cycle(self.constants)
                self.assertGreaterEqual(
                    decay,
                    weapon.spread_per_shot_deg,
                    f"{name} accumulates bloom at its own rate of fire",
                )


class TtkTableTests(unittest.TestCase):
    def test_table_reports_range_limits_rather_than_fake_numbers(self):
        table = ttk_table([2.0, 14.0])
        shotgun = table["profiles"]["shotgun"]["by_distance"]
        self.assertTrue(shotgun["2m"]["in_range"])
        self.assertFalse(shotgun["14m"]["in_range"])
        rifle = table["profiles"]["rifle"]["by_distance"]
        self.assertTrue(rifle["14m"]["in_range"])

    def test_table_covers_every_profile(self):
        table = ttk_table()
        self.assertEqual(set(table["profiles"]), set(load_profiles()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
