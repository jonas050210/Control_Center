"""Perception tests: the agent must not know more than it can see.

These tests are the contract for the observation layout version 2:
bearings are quantised to sectors, distances and health to bands, cover really
hides the opponent and only sightings are remembered.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from env.physics import can_see, in_field_of_view
from env.shooter_env import (
    BEARING_SECTORS,
    DEFAULT_VISION_MODE,
    DISTANCE_BAND_EDGES,
    MEMORY_SECONDS,
    OBSERVATION_FIELDS,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    UNKNOWN_BEARING,
    VISION_MODES,
    ShooterEnv,
    bearing_sector,
    distance_band_value,
    hp_band_value,
    sector_angles,
)


def env_with(mode: str, map_name: str = "Arena", **kwargs) -> ShooterEnv:
    env = ShooterEnv(map_name=map_name, seed=11, vision_mode=mode, **kwargs)
    env.reset(seed=11)
    return env


def hide_behind_cover(env: ShooterEnv) -> float:
    """Place both fighters so a pillar blocks the line of sight; return its x."""
    pillar = max(env.arena_map.objects, key=lambda item: getattr(item, "height", 0) or 0)
    me, enemy = env.player.body, env.opponent.body
    me.x, me.y, me.yaw = pillar.x, pillar.y - 3.0, 0.0
    enemy.x, enemy.y = pillar.x, pillar.y + 2.0
    return pillar.x


class LayoutTests(unittest.TestCase):
    def test_observation_layout_is_documented_and_complete(self) -> None:
        self.assertEqual(OBSERVATION_SIZE, 31)
        self.assertEqual(len(OBSERVATION_FIELDS), OBSERVATION_SIZE)
        self.assertEqual(len(set(OBSERVATION_FIELDS)), OBSERVATION_SIZE)
        self.assertEqual(OBSERVATION_VERSION, 2)
        # The perception block replaces the old "exact enemy state" values.
        self.assertEqual(OBSERVATION_FIELDS[5:13], (
            "enemy_bearing_sin", "enemy_bearing_cos", "enemy_distance_band", "enemy_visible",
            "enemy_hp_band", "enemy_time_since_seen", "enemy_memory_sin", "enemy_memory_cos",
        ))
        self.assertEqual(OBSERVATION_FIELDS[22], "enemy_alive")

    def test_default_vision_mode_is_the_honest_one(self) -> None:
        self.assertEqual(DEFAULT_VISION_MODE, "coarse_los")
        self.assertIn(DEFAULT_VISION_MODE, VISION_MODES)

    def test_unknown_vision_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShooterEnv(map_name="Dust", vision_mode="wallhack")


class QuantisationTests(unittest.TestCase):
    def test_sectors_cover_the_circle_without_gaps(self) -> None:
        centres = sector_angles()
        self.assertEqual(len(centres), BEARING_SECTORS)
        self.assertAlmostEqual(centres[0], math.radians(-180.0), places=9)
        width = math.radians(360.0 / BEARING_SECTORS)
        for index, centre in enumerate(centres):
            expected = bearing_sector(centre)
            self.assertAlmostEqual(centre, expected, places=9)
            # everything inside the sector snaps to its centre
            for offset in (-width * 0.45, 0.0, width * 0.45):
                self.assertAlmostEqual(bearing_sector(centre + offset), centre, places=9)
            _ = index

    def test_sector_error_never_exceeds_half_a_sector(self) -> None:
        from env.physics import wrap_angle

        limit = math.radians(360.0 / BEARING_SECTORS) * 0.5 + 1e-9
        for step in range(-360, 360):
            angle = math.radians(step * 0.5)
            error = abs(wrap_angle(bearing_sector(angle) - angle))
            self.assertLessEqual(error, limit, f"angle {step * 0.5}° snapped too far")

    def test_distance_bands_are_monotonic_and_last_band_is_open(self) -> None:
        max_distance = 70.0
        values = [distance_band_value(value, max_distance)
                  for value in (1.0, 4.9, 5.1, 9.9, 10.1, 19.9, 20.1, 34.9, 35.1, 59.9, 90.0)]
        self.assertEqual(values, sorted(values))
        edges = len(DISTANCE_BAND_EDGES) + 1
        distinct = len({round(value, 6) for value in values})
        self.assertEqual(distinct, edges)
        self.assertLessEqual(max(values), 1.0)
        self.assertGreaterEqual(min(values), -1.0)

    def test_health_bands_are_four_buckets(self) -> None:
        self.assertAlmostEqual(hp_band_value(0.9), 0.75)
        self.assertAlmostEqual(hp_band_value(0.6), 0.25)
        self.assertAlmostEqual(hp_band_value(0.4), -0.25)
        self.assertAlmostEqual(hp_band_value(0.1), -0.75)


class VisionTests(unittest.TestCase):
    def test_field_of_view_is_120_degrees(self) -> None:
        env = env_with("coarse_los")
        me, enemy = env.player.body, env.opponent.body
        me.yaw = 0.0
        for degrees, expected in ((0.0, True), (59.0, True), (61.0, False), (180.0, False)):
            enemy.x, enemy.y = me.x + math.sin(math.radians(degrees)) * 5.0, \
                me.y + math.cos(math.radians(degrees)) * 5.0
            self.assertEqual(in_field_of_view(me, enemy), expected, f"{degrees}°")
        env.close()

    def test_cover_hides_the_opponent(self) -> None:
        env = env_with("coarse_los")
        hide_behind_cover(env)
        self.assertFalse(can_see(env.player.body, env.opponent.body, env.arena_map))
        view = env.enemy_view(0)
        self.assertFalse(view["visible"])
        self.assertFalse(view["memory"])
        self.assertIsNone(view["seconds_since_seen"])
        self.assertEqual(view["bearing_sin"], 0.0)
        self.assertEqual(view["bearing_cos"], 0.0)
        self.assertEqual(view["distance_band"], 0.0)
        self.assertEqual(view["hp_band"], 0.0)
        env.close()

    def test_open_line_of_sight_is_reported(self) -> None:
        env = env_with("coarse_los", map_name="Dust")
        me, enemy = env.player.body, env.opponent.body
        me.z, enemy.z = 0.0, 0.0
        enemy.x, enemy.y = me.x + 6.0, me.y + 6.0
        me.yaw = math.atan2(enemy.x - me.x, enemy.y - me.y)
        view = env.enemy_view(0)
        self.assertTrue(view["visible"])
        self.assertTrue(view["memory"])
        self.assertAlmostEqual(view["seconds_since_seen"], 0.0)
        self.assertGreater(abs(view["bearing_sin"]) + abs(view["bearing_cos"]), 0.5)
        env.close()

    def test_other_modes_keep_the_legacy_tracking(self) -> None:
        for mode in ("exact", "noisy", "coarse"):
            with self.subTest(mode=mode):
                env = env_with(mode)
                hide_behind_cover(env)
                view = env.enemy_view(0)
                self.assertTrue(view["visible"], f"{mode} must not respect cover")
                env.close()


class LeakTests(unittest.TestCase):
    """Two hidden opponents in the same sector and band must be indistinguishable."""

    def _observation_for_hidden_enemy(self, env: ShooterEnv, offset: tuple[float, float]) -> np.ndarray:
        hide_behind_cover(env)
        env.opponent.body.x += offset[0]
        env.opponent.body.y += offset[1]
        return env.get_observation(0)

    def test_hidden_enemy_has_no_precise_coordinates(self) -> None:
        offsets = ((0.0, 0.0), (0.6, 0.4), (-0.5, 0.8))
        observations = [
            self._observation_for_hidden_enemy(env_with("coarse_los"), offset) for offset in offsets
        ]
        for observation in observations[1:]:
            np.testing.assert_array_equal(observations[0], observation)
        # Enemy health is unknown while hidden (index 9) and marked alive at 22.
        self.assertEqual(observations[0][9], 0.0)
        self.assertEqual(observations[0][22], 1.0)

    def test_exact_mode_still_reveals_the_position(self) -> None:
        first = self._observation_for_hidden_enemy(env_with("exact"), (0.0, 0.0))
        second = self._observation_for_hidden_enemy(env_with("exact"), (0.6, 0.4))
        self.assertFalse(np.array_equal(first, second))

    def test_enemy_health_is_not_leaked_when_hidden(self) -> None:
        env = env_with("coarse_los")
        hide_behind_cover(env)
        healthy = env.get_observation(0)
        env.opponent.body.hp = 12.0
        hurt = env.get_observation(0)
        np.testing.assert_array_equal(healthy, hurt,
                                      "a hidden enemy may not leak its current health")
        self.assertEqual(hurt[9], 0.0, "enemy health band stays unknown behind cover")

    def test_bearing_is_coarse_in_coarse_modes(self) -> None:
        env = env_with("coarse", map_name="Dust")
        me, enemy = env.player.body, env.opponent.body
        me.x, me.y, me.yaw = 0.0, 0.0, 0.0
        angles = []
        for degrees in (2.0, 6.0, 13.0):
            enemy.x = math.sin(math.radians(degrees)) * 10.0
            enemy.y = math.cos(math.radians(degrees)) * 10.0
            view = env.enemy_view(0)
            angles.append(math.atan2(view["bearing_sin"], view["bearing_cos"]))
        self.assertEqual(len({round(value, 9) for value in angles}), 1,
                         "angles inside one sector must report the same direction")
        env.close()


class MemoryTests(unittest.TestCase):
    def _see_then_hide(self, env: ShooterEnv) -> dict:
        me, enemy = env.player.body, env.opponent.body
        me.z, enemy.z = 0.0, 0.0
        enemy.x, enemy.y = me.x + 5.0, me.y + 5.0
        me.yaw = math.atan2(enemy.x - me.x, enemy.y - me.y)
        first = env.enemy_view(0)
        assert first["visible"]
        hide_behind_cover(env)
        return first

    def test_memory_survives_cover_and_fades_after_five_seconds(self) -> None:
        env = env_with("coarse_los")
        seen = self._see_then_hide(env)
        memory = env.enemy_view(0)
        self.assertFalse(memory["visible"])
        self.assertTrue(memory["memory"], "the last sighting must be remembered")
        self.assertGreater(abs(memory["bearing_sin"]) + abs(memory["bearing_cos"]), 0.5)
        self.assertAlmostEqual(memory["seconds_since_seen"], 0.0, places=6)

        env.elapsed += MEMORY_SECONDS * 0.5
        halfway = env.enemy_view(0)
        self.assertTrue(halfway["memory"])
        self.assertAlmostEqual(halfway["seconds_since_seen"], MEMORY_SECONDS * 0.5, places=6)

        env.elapsed += MEMORY_SECONDS
        faded = env.enemy_view(0)
        self.assertFalse(faded["memory"], "a sighting older than MEMORY_SECONDS is gone")
        self.assertIsNone(faded["seconds_since_seen"])
        self.assertEqual(faded["bearing_sin"], 0.0)
        self.assertEqual(faded["bearing_cos"], 0.0)
        # The sighting itself had a real direction (straight ahead => cos = 1).
        self.assertAlmostEqual(math.hypot(seen["bearing_sin"], seen["bearing_cos"]), 1.0, places=6)
        env.close()

    def test_time_since_seen_is_encoded_from_unknown_to_old(self) -> None:
        env = env_with("coarse_los", map_name="Dust")
        self._see_then_hide(env)
        just_seen = env.get_observation(0)[10]
        env.elapsed += MEMORY_SECONDS
        old = env.get_observation(0)[10]
        self.assertAlmostEqual(just_seen, 0.0, places=5)
        self.assertAlmostEqual(old, 1.0, places=5)

    def test_reset_forgets_everything(self) -> None:
        # Arena spawns are blocked by a pillar, so a fresh episode starts dim.
        env = env_with("coarse_los", map_name="Arena")
        env.player.last_seen_bearing = 42.0
        env.player.last_seen_time = 12.0
        env.player.blind_steps = 99
        env.reset(seed=11)
        self.assertIsNone(env.player.last_seen_bearing)
        self.assertIsNone(env.player.last_seen_time)
        self.assertEqual(env.player.blind_steps, 1, "only the reset frame counts")
        self.assertEqual(env.player.visible_steps, 0)
        self.assertFalse(env.enemy_view(0)["memory"])
        env.close()


class RewardTests(unittest.TestCase):
    def test_no_aim_bonus_without_visual_contact(self) -> None:
        env = env_with("coarse_los")
        hide_behind_cover(env)
        self.assertFalse(env._tracked(0))
        self.assertFalse(env._tracked(1))
        # The aim error used for the reward is neutralised while blind.
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("perception", info)
        self.assertFalse(info["perception"]["player"]["visible"])
        env.close()

    def test_aim_bonus_stays_active_with_contact(self) -> None:
        env = env_with("coarse_los", map_name="Dust")
        hide_behind_cover(env)
        # Turn towards the opponent so the reward gate opens only if it sees.
        env.player.body.yaw = math.atan2(env.opponent.body.x - env.player.body.x,
                                         env.opponent.body.y - env.player.body.y)
        env.opponent.body.x = env.player.body.x + 5.0
        env.opponent.body.y = env.player.body.y
        env.player.body.yaw = math.radians(90.0)
        self.assertLess(abs(env._aim_error(0)), math.radians(5.0))
        self.assertTrue(env._tracked(0), "the open sightline must open the reward gate")
        env.close()


class EpisodeStatTests(unittest.TestCase):
    def test_episode_summary_reports_blind_ratio(self) -> None:
        env = env_with("coarse_los", max_episode_seconds=3.0)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
        metrics = info["episode_metrics"]
        self.assertIn("player_blind_ratio", metrics)
        self.assertGreaterEqual(metrics["player_blind_ratio"], 0.0)
        self.assertLessEqual(metrics["player_blind_ratio"], 1.0)
        self.assertEqual(metrics["vision_mode"], "coarse_los")
        env.close()


class CurriculumOpponentTests(unittest.TestCase):
    """The opponent ladder must add one new lesson per phase, not two."""

    def test_phase_ladder_and_modes_are_consistent(self) -> None:
        from env.shooter_env import CURRICULUM_OPPONENTS, OPPONENT_MODES

        self.assertEqual(CURRICULUM_OPPONENTS[1], "stationary")
        self.assertEqual(CURRICULUM_OPPONENTS[2], "mover")
        self.assertEqual(CURRICULUM_OPPONENTS[3], "walker")
        self.assertEqual(CURRICULUM_OPPONENTS[4], "full")
        for mode in CURRICULUM_OPPONENTS.values():
            self.assertIn(mode, OPPONENT_MODES)
        # Phase 2 introduces movement, phase 3 return fire: the harmless mode has
        # to stay harmless, otherwise both lessons arrive at once again.
        self.assertNotIn("mover", {"walker", "shooter", "full"})


class MetricsCsvSchemaTests(unittest.TestCase):
    """The metrics file must survive new columns without losing values."""

    def test_new_column_is_appended_and_old_rows_are_kept(self) -> None:
        import csv
        import tempfile
        import threading
        from pathlib import Path

        from training.train import _append_csv

        with tempfile.TemporaryDirectory(prefix="metrics-csv-") as temporary:
            path = Path(temporary) / "training_metrics.csv"
            lock = threading.Lock()
            _append_csv(path, ["steps", "win_rate"], {"steps": 1, "win_rate": 0.5}, lock)
            _append_csv(path, ["steps", "win_rate", "kill_rate"],
                        {"steps": 2, "win_rate": 0.6, "kill_rate": 0.25}, lock)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), ["steps", "win_rate", "kill_rate"])
            self.assertEqual(rows[0]["win_rate"], "0.5")
            self.assertEqual(rows[0]["kill_rate"], "")
            self.assertEqual(rows[1]["kill_rate"], "0.25")
            self.assertEqual(rows[1]["win_rate"], "0.6")

    def test_unknown_extra_columns_of_a_file_are_kept(self) -> None:
        import csv
        import tempfile
        import threading
        from pathlib import Path

        from training.train import _append_csv

        with tempfile.TemporaryDirectory(prefix="metrics-csv-") as temporary:
            path = Path(temporary) / "training_metrics.csv"
            path.write_text("steps,renamed_metric\n1,7\n", encoding="utf-8")
            _append_csv(path, ["steps"], {"steps": 2}, threading.Lock())
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), ["steps", "renamed_metric"])
            self.assertEqual(rows[0]["renamed_metric"], "7")
            self.assertEqual(rows[1]["steps"], "2")


class CurriculumBacktrackTests(unittest.TestCase):
    """A phase that only produces defeats must be undone, not endured."""

    def test_phase_is_undone_after_a_collapse(self) -> None:
        from training.train import curriculum_backtrack_needed

        self.assertTrue(curriculum_backtrack_needed(
            current_phase=3, episodes_in_phase=45, recent_kill_rate=0.0, gate=0.25))
        self.assertTrue(curriculum_backtrack_needed(
            current_phase=2, episodes_in_phase=40, recent_kill_rate=0.11, gate=0.25))

    def test_healthy_phase_and_first_phase_are_kept(self) -> None:
        from training.train import curriculum_backtrack_needed

        self.assertFalse(curriculum_backtrack_needed(
            current_phase=1, episodes_in_phase=200, recent_kill_rate=0.0, gate=0.25))
        self.assertFalse(curriculum_backtrack_needed(
            current_phase=3, episodes_in_phase=20, recent_kill_rate=0.0, gate=0.25))
        self.assertFalse(curriculum_backtrack_needed(
            current_phase=3, episodes_in_phase=80, recent_kill_rate=0.3, gate=0.25))


class DemoArchiveTests(unittest.TestCase):
    """Demos recorded with another layout must never be silently mixed."""

    def test_outdated_demo_file_is_moved_aside(self) -> None:
        import json
        import tempfile
        from pathlib import Path as _Path

        from training.imitation import archive_outdated_demos, write_demo_meta

        with tempfile.TemporaryDirectory() as folder:
            path = _Path(folder) / "demos.csv"
            path.write_text("state_0,action_0\n1,2\n", encoding="utf-8")
            write_demo_meta(path, vision_mode="coarse_los", samples=1)
            meta = path.with_name("demos_meta.json")
            meta.write_text(json.dumps({"observation_version": 1}), encoding="utf-8")
            archived = archive_outdated_demos(path)
            self.assertIsNotNone(archived)
            self.assertFalse(path.exists())
            self.assertTrue(archived.exists())
            self.assertTrue(archived.with_name(f"{archived.stem}_meta.json").exists())

    def test_current_demo_file_stays_untouched(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from training.imitation import archive_outdated_demos, write_demo_meta

        with tempfile.TemporaryDirectory() as folder:
            path = _Path(folder) / "demos.csv"
            path.write_text("state_0,action_0\n1,2\n", encoding="utf-8")
            write_demo_meta(path, vision_mode="coarse_los", samples=1)
            self.assertIsNone(archive_outdated_demos(path))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

class VisionModeSwitchTests(unittest.TestCase):
    """The trainer's perception curriculum switches modes between phases."""

    def test_switching_mode_clears_memory_and_keeps_the_layout(self) -> None:
        env = ShooterEnv(map_name="Dust", curriculum=False, frame_skip=4,
                         vision_mode="exact", seed=4)
        observation, _ = env.reset(seed=4)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        # exact mode always knows the opponent, so a sighting must exist
        self.assertIsNotNone(env._combatants[0].last_seen_time)

        env.set_vision_mode("coarse_los")
        self.assertEqual(env.vision_mode, "coarse_los")
        self.assertIsNone(env._combatants[0].last_seen_time,
                          "switching perception must drop the old memory")
        switched, _ = env.reset(seed=4)
        self.assertEqual(switched.shape, (OBSERVATION_SIZE,))
        self.assertNotEqual(float(np.abs(observation).sum()), float(np.abs(switched).sum()))
        env.close()

    def test_unknown_mode_is_rejected(self) -> None:
        env = ShooterEnv(map_name="Dust", frame_skip=4, seed=4)
        with self.assertRaises(ValueError):
            env.set_vision_mode("clairvoyant")
        env.close()

    def test_trainer_vision_curriculum_gets_stricter(self) -> None:
        from training.train import vision_mode_for_phase

        modes = [vision_mode_for_phase(phase) for phase in (1, 2, 3, 4)]
        self.assertEqual(modes, ["noisy", "coarse", "coarse_los", "coarse_los"])
        # the last phase must train exactly what the arena plays
        self.assertEqual(modes[-1], DEFAULT_VISION_MODE)


class CurriculumSpawnTests(unittest.TestCase):
    """Early curriculum phases must start inside weapon range.

    Measured on Dust at the full spawn separation of 36 m: a perfectly aiming
    fighter lands 0 % (Pistol) and 1.8 % (AK-47) of its shots - a phase out there
    cannot teach a kill, the policy just farms the aim bonus for the time limit.
    """

    def _distance(self, env) -> float:
        first, second = env._combatants
        return math.hypot(first.body.x - second.body.x, first.body.y - second.body.y)

    def test_early_phases_start_closer_and_phase_four_uses_the_map_spawns(self) -> None:
        distances = {}
        for phase in (1, 2, 3, 4):
            env = ShooterEnv(map_name="Dust", curriculum=True, curriculum_phase=phase,
                             frame_skip=4, seed=11)
            distances[phase] = self._distance(env)
            env.close()
        self.assertLess(distances[1], distances[2])
        self.assertLess(distances[2], distances[3])
        self.assertLess(distances[3], distances[4])
        self.assertLess(distances[1], 12.0)
        # A first run used 45 % (16.2 m) for phase 2 and then produced ~500
        # episodes in a row without a single win or kill - the step from the
        # passive phase-1 opponent to one that shoots back needs a flatter ramp.
        self.assertLess(distances[2], 15.0)

        plain = ShooterEnv(map_name="Dust", curriculum=False, frame_skip=4, seed=11)
        self.assertAlmostEqual(self._distance(plain), distances[4], places=3)
        plain.close()

    def test_shortened_spawns_are_not_inside_cover(self) -> None:
        from env.physics import position_is_free

        for map_name in ("Dust", "Warehouse", "Highrise", "Arena"):
            for phase in (1, 2, 3):
                env = ShooterEnv(map_name=map_name, curriculum=True, curriculum_phase=phase,
                                 frame_skip=4, seed=5)
                for fighter in env._combatants:
                    with self.subTest(map=map_name, phase=phase):
                        self.assertTrue(position_is_free(env.arena_map, fighter.body.x,
                                                         fighter.body.y))
                env.close()
