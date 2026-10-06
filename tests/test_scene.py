"""Scene-geometry tests for the browser renderer payloads (replaces Plotly tests)."""

from __future__ import annotations

import json
import unittest

from env.maps import ArenaMap, ArenaObject, MAP_NAMES, create_map
from server.actions import PLAY_ACTIONS, make_action
from server.config import DETAIL_PRESETS, OBJECT_COLORS
from server.minigames import new_dodge_game, step_dodge_game
from server.scene import (
    aim_index_scene,
    ambient_details,
    ambient_props,
    dodge_arena_scene,
    map_key,
    map_scene,
    match_frame,
    scene_key,
    target_states,
)


class StaticSceneTests(unittest.TestCase):
    def test_every_map_exports_cover_grid_and_details(self) -> None:
        for name in MAP_NAMES:
            with self.subTest(map=name):
                payload = map_scene(create_map(name), detail_count=DETAIL_PRESETS["Balanced"])
                self.assertEqual(payload["name"], name)
                self.assertGreater(payload["width"], 0)
                self.assertGreater(payload["depth"], 0)
                self.assertTrue(payload["objects"])
                self.assertTrue(all(item["color"] in OBJECT_COLORS.values()
                                    for item in payload["objects"]))
                self.assertEqual(len(payload["details"]["x"]), DETAIL_PRESETS["Balanced"])
                self.assertEqual(len(payload["spawns"]), 2)
                self.assertLess(len(json.dumps(payload)), 260_000)

    def test_hundreds_of_cover_objects_stay_a_single_payload(self) -> None:
        props = [
            ArenaObject((index % 20) - 10, (index // 20) - 2, 0,
                        0.7, 0.8 + (index % 4) * 0.15, 0.7, "crate", f"Crate {index}")
            for index in range(400)
        ]
        arena_map = ArenaMap("Custom", 50, 50, props, ((-20, 0), (20, 0)))
        payload = map_scene(arena_map, detail_count=0)
        self.assertEqual(len(payload["objects"]), 400)
        self.assertEqual(payload["details"]["x"], [])
        # Renderer groups by kind+colour, so 400 crates become one instanced mesh.
        groups = {(item["kind"], item["color"]) for item in payload["objects"]}
        self.assertEqual(len(groups), 1)
        self.assertLess(len(json.dumps(payload)), 400_000)

    def test_detail_levels_scale_ambient_objects(self) -> None:
        arena_map = create_map("Dust")
        low = map_scene(arena_map, detail_count=DETAIL_PRESETS["Performance"])
        high = map_scene(arena_map, detail_count=DETAIL_PRESETS["Ultra"])
        self.assertLess(len(low["details"]["x"]), len(high["details"]["x"]))
        # Set dressing (decorative crates/barrels) scales with detail but caps at 24.
        self.assertLessEqual(len(low["objects"]), len(high["objects"]))
        self.assertLessEqual(len(high["objects"]) - len(create_map("Dust").objects), 24)

    def test_ambient_geometry_is_deterministic_and_collision_free(self) -> None:
        arena_map = create_map("Warehouse")
        first = map_scene(arena_map, detail_count=120)
        second = map_scene(arena_map, detail_count=120)
        self.assertEqual(first["details"], second["details"])
        self.assertEqual(first["key"], second["key"])
        for spawn_x, spawn_y in arena_map.spawn_points:
            for x, y in zip(first["details"]["x"], first["details"]["y"]):
                self.assertGreater(
                    ((x - spawn_x) ** 2 + (y - spawn_y) ** 2) ** 0.5, 1.0,
                    "floor detail must not cover a spawn point",
                )
        self.assertTrue(ambient_props.cache_info().currsize >= 1)
        self.assertTrue(ambient_details.cache_info().currsize >= 1)

    def test_scene_keys_track_layout_and_detail_changes(self) -> None:
        arena_map = create_map("Dust")
        self.assertNotEqual(scene_key(arena_map, 60), scene_key(arena_map, 320))
        edited = arena_map.copy()
        edited.objects.append(ArenaObject(0, 0, 0, 2, 2, 2, "crate", "Extra"))
        self.assertNotEqual(map_key(arena_map), map_key(edited))
        self.assertNotEqual(scene_key(arena_map, 60), scene_key(edited, 60))


class MatchFrameTests(unittest.TestCase):
    def _frame(self) -> dict:
        from env.shooter_env import ShooterEnv

        env = ShooterEnv(map_name="Dust", weapon_name="Pistol", opponent_weapon="AK-47",
                         curriculum=False, opponent_mode="full", frame_skip=4, seed=11)
        env.reset(seed=11)
        for _ in range(12):
            env.step_duel(env.heuristic_action(0), env.heuristic_action(1))
        frame = match_frame(env)
        env.close()
        return frame

    def test_frame_contains_agents_trails_and_timers(self) -> None:
        frame = self._frame()
        self.assertEqual(len(frame["agents"]), 2)
        agent = frame["agents"][0]
        for key in ("x", "y", "z", "yaw", "pitch", "hp", "height", "weapon", "ammo",
                    "mag_size", "color", "label"):
            self.assertIn(key, agent)
        self.assertEqual(frame["agents"][0]["color"], "#00cc33")
        self.assertEqual(frame["agents"][1]["color"], "#ff0040")
        self.assertGreater(frame["elapsed"], 0)
        self.assertGreater(frame["frame"], 0)
        self.assertIsInstance(frame["trails"], list)

    def test_frame_is_json_serializable_and_compact(self) -> None:
        frame = self._frame()
        encoded = json.dumps(frame)
        self.assertLess(len(encoded), 30_000)
        self.assertNotIn("NaN", encoded)


class MiniGameSceneTests(unittest.TestCase):
    def test_aim_range_exposes_nine_targets_and_camera_hint(self) -> None:
        payload = aim_index_scene("aim-lab")
        self.assertEqual(len(payload["targets"]), 9)
        cells = [target["cell"] for target in payload["targets"]]
        self.assertEqual(cells, list(range(9)))
        self.assertIn("camera_radius", payload)
        states = target_states(4, hits=3, misses=1, streak=2)
        self.assertEqual(sum(1 for state in states if state["active"]), 1)
        self.assertTrue(states[4]["active"])

    def test_dodge_grid_tracks_player_and_projectiles(self) -> None:
        payload = dodge_arena_scene("dodge-grid")
        self.assertEqual(len(payload["cells"]), 25)
        self.assertEqual(len(payload["turrets"]), 5)
        game = new_dodge_game(seed=4)
        for _ in range(6):
            step_dodge_game(game, dx=-1)
        self.assertIn("player", game)
        column, row = game["player"]
        self.assertIn((column, row), [(cell["column"], cell["row"]) for cell in payload["cells"]])
        for projectile_x, projectile_y in game["projectiles"]:
            self.assertTrue(0 <= projectile_x < 5 and 0 <= projectile_y < 5)


class ControlEncodingTests(unittest.TestCase):
    def test_play_action_presets_match_the_action_space(self) -> None:
        for name, preset in PLAY_ACTIONS.items():
            with self.subTest(press=name):
                controls = {key: value for key, value in preset.items() if key != "repeat"}
                action = make_action(**controls)
                self.assertEqual(action.shape, (10,))
                self.assertTrue(1 <= preset.get("repeat", 1) <= 35)

    def test_sprint_and_stance_are_optional_controls(self) -> None:
        action = make_action(move=1, shoot=True, sprint=True, stance=2, jump=True, reload=True)
        self.assertEqual(action.tolist(), [2, 1, 1, 1, 1, 1, 2, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
