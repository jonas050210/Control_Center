"""Quality and trace-budget checks for the detailed, batched 3D scenes."""

from __future__ import annotations

import unittest

from env.maps import ArenaMap, ArenaObject, MAP_NAMES, create_map
from gui.games import new_dodge_game, step_dodge_game
from gui.visuals import build_aim_range_figure, build_dodge_arena_figure, build_map_figure


class ThreeDVisualTests(unittest.TestCase):
    def test_every_map_renders_with_a_small_webgl_trace_budget(self) -> None:
        for name in MAP_NAMES:
            with self.subTest(map=name):
                figure = build_map_figure(create_map(name), detail_count=100)
                self.assertLessEqual(len(figure.data), 18)
                self.assertTrue(any(trace.name == "Armored floor" for trace in figure.data))
                self.assertTrue(any("Scattered floor detail" in str(trace.name) for trace in figure.data))
                self.assertLess(len(figure.to_json()), 250_000)

    def test_hundreds_of_cover_objects_are_batched_by_material(self) -> None:
        props = [
            ArenaObject((index % 20) - 10, (index // 20) - 2, 0,
                        0.7, 0.8 + (index % 4) * 0.15, 0.7, "crate", f"Crate {index}")
            for index in range(400)
        ]
        arena_map = ArenaMap("Custom", 50, 50, props, ((-20, 0), (20, 0)))
        figure = build_map_figure(arena_map, show_spawns=False, detail_count=0)
        crate_traces = [trace for trace in figure.data if trace.name == "Crate props · 400"]
        self.assertEqual(len(crate_traces), 1)
        self.assertLessEqual(len(figure.data), 10)
        self.assertLess(len(figure.to_json()), 800_000)

    def test_both_minigames_have_detailed_rotatable_3d_scenes(self) -> None:
        aim = build_aim_range_figure(target_cell=4, hits=6, misses=2, streak=3, detail_count=90)
        self.assertTrue(any(trace.name == "Active target lock" for trace in aim.data))
        self.assertTrue(any(trace.name == "Target cores" for trace in aim.data))
        self.assertLessEqual(len(aim.data), 20)
        self.assertIn("eye", aim.layout.scene.camera)

        dodge = new_dodge_game(seed=4)
        for _ in range(3):
            step_dodge_game(dodge, dx=-1)
        arena = build_dodge_arena_figure(dodge, detail_count=90)
        self.assertTrue(any(trace.name == "Automated turrets" for trace in arena.data))
        self.assertTrue(any(trace.name == "Player beacon" for trace in arena.data))
        self.assertLessEqual(len(arena.data), 20)
        self.assertIn("eye", arena.layout.scene.camera)

    def test_live_match_has_low_poly_fighters_batched_shot_fx_and_hud(self) -> None:
        agents = [
            {"x": -3.0, "y": 0.0, "z": 0.0, "yaw": 0.1, "height": 1.8,
             "hp": 85.0, "weapon": "Pistol", "ammo": 8},
            {"x": 3.0, "y": 0.0, "z": 0.0, "yaw": -0.1, "height": 1.8,
             "hp": 60.0, "weapon": "AK-47", "ammo": 21},
        ]
        trails = [
            {"start": (-3.0, 0.0, 1.4), "end": (3.0, 0.0, 1.4), "hit": index % 2 == 0}
            for index in range(60)
        ]
        figure = build_map_figure(create_map("Dust"), agents=agents, trails=trails, detail_count=90)
        names = [str(trace.name) for trace in figure.data]
        self.assertEqual(sum("low-poly fighter" in name for name in names), 2)
        self.assertIn("Shot trails", names)
        self.assertIn("Hit sparks", names)
        self.assertLessEqual(len(figure.data), 24)
        self.assertLess(len(figure.to_json()), 300_000)

    def test_ultra_detail_increases_objects_without_adding_webgl_layers(self) -> None:
        arena_map = create_map("Dust")
        balanced = build_map_figure(arena_map, detail_count=100)
        ultra = build_map_figure(arena_map, detail_count=300)
        balanced_detail = next(trace for trace in balanced.data
                               if str(trace.name).startswith("Scattered floor detail"))
        ultra_detail = next(trace for trace in ultra.data
                            if str(trace.name).startswith("Scattered floor detail"))
        self.assertGreater(len(ultra_detail.x), len(balanced_detail.x))
        self.assertEqual(len(ultra.data), len(balanced.data))


if __name__ == "__main__":
    unittest.main()
