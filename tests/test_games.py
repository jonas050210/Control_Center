"""Unit tests for the small session-state playground games."""

from __future__ import annotations

import unittest

from gui.games import aim_tap, new_aim_game, new_dodge_game, step_dodge_game
from gui.tabs.playground import _make_action


class PlaygroundGameTests(unittest.TestCase):
    def test_dodge_run_is_seeded_bounded_and_stops_on_collision(self) -> None:
        first = new_dodge_game(seed=123)
        second = new_dodge_game(seed=123)
        moves = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
        for tick in range(120):
            dx, dy = moves[tick % len(moves)]
            step_dodge_game(first, dx, dy)
            step_dodge_game(second, dx, dy)
            visible_first = {key: value for key, value in first.items() if key != "rng"}
            visible_second = {key: value for key, value in second.items() if key != "rng"}
            self.assertEqual(visible_first, visible_second)
            self.assertTrue(0 <= first["player"][0] < 5)
            self.assertTrue(0 <= first["player"][1] < 5)
            if not first["active"]:
                break
        self.assertFalse(first["active"], "seeded test run should eventually hit an incoming projectile")
        final_score = first["score"]
        step_dodge_game(first, 1, 1)
        self.assertEqual(first["score"], final_score)
        self.assertTrue(first["hit"])

    def test_aim_drill_hits_misses_and_expires(self) -> None:
        game = new_aim_game(duration_seconds=15, seed=19)
        first_target = game["target"]
        start = game["started_at"]
        self.assertTrue(aim_tap(game, first_target, now=start + 0.2))
        self.assertEqual(game["hits"], 1)
        self.assertNotEqual(game["target"], first_target)
        self.assertAlmostEqual(game["reaction_ms"][0], 200.0, delta=10.0)
        self.assertFalse(aim_tap(game, game["target"] + 1, now=start + 0.3))
        self.assertEqual(game["misses"], 1)
        self.assertEqual(game["streak"], 0)
        self.assertFalse(aim_tap(game, game["target"], now=start + 16.0))
        self.assertFalse(game["active"])
        self.assertEqual(game["hits"], 1)

    def test_human_shooter_action_matches_multidiscrete_encoding(self) -> None:
        action = _make_action(move=1, strafe=-1, yaw=-1, pitch=1, shoot=True,
                              sprint=True, stance=2, jump=True)
        self.assertEqual(action.tolist(), [2, 0, 0, 2, 1, 1, 2, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
