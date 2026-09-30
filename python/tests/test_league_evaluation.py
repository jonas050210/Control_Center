"""Tests for the completed self-play league (Phase 6).

The existing sampling/Elo behaviour is covered by
``test_training_infrastructure.py``; this file covers what Phase 6 added:
rich match results, per-map/per-condition breakdowns, frozen-opponent
enforcement, persisted history and deterministic tournaments.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from sandboxai.league import (
    MATCH_STAT_KEYS,
    CheckpointRegistry,
    EvaluationLeague,
    FrozenOpponentError,
    FrozenOpponentGuard,
    MatchResult,
    format_match_history,
)


def _registry(tmp: Path | None = None) -> CheckpointRegistry:
    registry = CheckpointRegistry()
    registry.register("learner", checkpoint=str((tmp or Path(".")) / "learner.zip"), frozen=False)
    registry.register("learner@100", checkpoint=str((tmp or Path(".")) / "s100.zip"), step=100)
    registry.register("learner@200", checkpoint=str((tmp or Path(".")) / "s200.zip"), step=200)
    registry.register("scripted_baseline", checkpoint="", frozen=True, tags=["baseline"])
    return registry


def _result(a: str, b: str, score: float, **kwargs) -> MatchResult:
    payload = {
        "map_id": "open_field",
        "lighting": "normal",
        "scenario": "duel",
        "enemy_count": 1,
        "stats_a": {key: 1.0 for key in MATCH_STAT_KEYS},
        "stats_b": {key: 0.5 for key in MATCH_STAT_KEYS},
    }
    payload.update(kwargs)
    return MatchResult(policy_a=a, policy_b=b, score_a=score, **payload)


class MatchResultTests(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(_result("a", "b", 1.0).outcome, "win_a")
        self.assertEqual(_result("a", "b", 0.0).outcome, "win_b")
        self.assertEqual(_result("a", "b", 0.5).outcome, "draw")
        self.assertEqual(_result("a", "b", 0.5, truncated=True).outcome, "truncated")

    def test_score_is_range_checked(self):
        with self.assertRaises(ValueError):
            _result("a", "b", 1.5)

    def test_condition_key_excludes_the_seed(self):
        first = _result("a", "b", 1.0, seed=1)
        second = _result("a", "b", 1.0, seed=2)
        self.assertEqual(first.condition_key, second.condition_key)
        self.assertNotEqual(
            first.condition_key, _result("a", "b", 1.0, map_id="night_yard").condition_key
        )

    def test_round_trip(self):
        original = _result("a", "b", 1.0, seed=42)
        restored = MatchResult.from_dict(original.to_dict())
        self.assertEqual(restored.policy_a, "a")
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.stats_a, original.stats_a)


class BreakdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.league = EvaluationLeague(_registry(), seed=5)
        self.league.record_result(_result("learner", "learner@100", 1.0, map_id="open_field"))
        self.league.record_result(
            _result("learner", "learner@100", 0.0, map_id="night_yard", lighting="night")
        )
        self.league.record_result(
            _result(
                "learner", "learner@200", 0.5, map_id="night_yard", lighting="night", truncated=True
            )
        )

    def test_policy_summary_counts_every_outcome_class(self):
        summary = self.league.policy_summary("learner")
        self.assertEqual(summary["matches"], 3)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["draws"], 0)
        self.assertEqual(summary["truncations"], 1)

    def test_diagnostics_are_averaged_per_policy(self):
        summary = self.league.policy_summary("learner")
        for key in MATCH_STAT_KEYS:
            self.assertAlmostEqual(summary[f"mean_{key}"], 1.0)
        opponent = self.league.policy_summary("learner@100")
        for key in MATCH_STAT_KEYS:
            self.assertAlmostEqual(opponent[f"mean_{key}"], 0.5)

    def test_per_map_breakdown(self):
        by_map = self.league.by_map("learner")
        self.assertEqual(by_map["open_field"]["win_rate"], 1.0)
        self.assertEqual(by_map["night_yard"]["matches"], 2)
        self.assertEqual(by_map["night_yard"]["win_rate"], 0.0)

    def test_per_condition_and_per_lighting_breakdowns(self):
        self.assertEqual(len(self.league.by_condition("learner")), 2)
        by_light = self.league.by_lighting("learner")
        self.assertEqual(by_light["night"]["matches"], 2)

    def test_per_opponent_breakdown(self):
        by_opponent = self.league.by_opponent("learner")
        self.assertEqual(by_opponent["learner@100"]["matches"], 2)
        self.assertEqual(by_opponent["learner@200"]["truncations"], 1)

    def test_full_report_labels_elo_as_experimental(self):
        report = self.league.full_report()
        self.assertIn("EXPERIMENTAL", report["elo_note"])
        self.assertEqual(report["matches_recorded"], 3)
        self.assertIn("by_map", report["policies"]["learner"])

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(KeyError):
            self.league.record_result(_result("ghost", "learner", 1.0))

    def test_history_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.league.save_history(Path(tmp) / "history.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 3)
            reloaded = EvaluationLeague(_registry(), seed=5)
            self.assertEqual(reloaded.load_history(path), 3)
        self.assertEqual(reloaded.by_map("learner")["open_field"]["win_rate"], 1.0)
        # Loading must not double-count into the registry by default.
        self.assertEqual(reloaded.registry.get("learner").matches, 0)

    def test_history_can_be_replayed_into_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.league.save_history(Path(tmp) / "history.json")
            reloaded = EvaluationLeague(_registry(), seed=5)
            reloaded.load_history(path, replay_into_registry=True)
        self.assertEqual(reloaded.registry.get("learner").matches, 3)

    def test_format_match_history_renders(self):
        text = format_match_history(self.league.results)
        self.assertIn("learner", text)
        self.assertIn("night_yard", text)


class FrozenOpponentTests(unittest.TestCase):
    def test_guard_detects_a_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "s100.zip"
            frozen.write_bytes(b"weights-v1")
            registry = _registry(root)
            guard = FrozenOpponentGuard()
            watched = guard.arm(registry.records())
            self.assertIn("learner@100", watched)
            guard.verify(registry.records())  # unchanged: fine
            time.sleep(0.01)
            frozen.write_bytes(b"weights-v2-different-length")
            with self.assertRaises(FrozenOpponentError):
                guard.verify(registry.records())

    def test_guard_detects_a_deleted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "s100.zip"
            frozen.write_bytes(b"weights")
            registry = _registry(root)
            guard = FrozenOpponentGuard()
            guard.arm(registry.records())
            frozen.unlink()
            with self.assertRaises(FrozenOpponentError):
                guard.verify(registry.records())

    def test_guard_ignores_the_learning_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            learner = root / "learner.zip"
            learner.write_bytes(b"v1")
            registry = _registry(root)
            guard = FrozenOpponentGuard()
            self.assertNotIn("learner", guard.arm(registry.records()))
            time.sleep(0.01)
            learner.write_bytes(b"v2-longer")
            guard.verify(registry.records())  # learning weights may move


class TournamentTests(unittest.TestCase):
    def _play(self, calls: list[tuple[str, str, dict]]):
        def play_match(policy_a: str, policy_b: str, condition: dict) -> MatchResult:
            calls.append((policy_a, policy_b, dict(condition)))
            # Deterministic pseudo-outcome derived only from the inputs.
            score = 1.0 if (hash((policy_a, policy_b, condition["seed"])) % 2 == 0) else 0.0
            return _result(
                policy_a,
                policy_b,
                score,
                map_id=condition.get("map_id", "open_field"),
                lighting=condition.get("lighting", "normal"),
                seed=condition["seed"],
            )

        return play_match

    def test_tournament_plays_every_ordered_pair(self):
        calls: list[tuple[str, str, dict]] = []
        league = EvaluationLeague(_registry(), seed=3)
        played = league.run_tournament(["learner", "learner@100", "learner@200"], self._play(calls))
        self.assertEqual(len(played), 6)  # 3 * 2 ordered pairs
        self.assertEqual(len({(a, b) for a, b, _c in calls}), 6)

    def test_tournament_is_deterministic_for_the_same_seed(self):
        first_calls: list[tuple[str, str, dict]] = []
        second_calls: list[tuple[str, str, dict]] = []
        EvaluationLeague(_registry(), seed=17).run_tournament(
            ["learner", "learner@100"], self._play(first_calls)
        )
        EvaluationLeague(_registry(), seed=17).run_tournament(
            ["learner", "learner@100"], self._play(second_calls)
        )
        self.assertEqual(first_calls, second_calls)

    def test_a_different_league_seed_changes_the_match_seeds(self):
        first_calls: list[tuple[str, str, dict]] = []
        second_calls: list[tuple[str, str, dict]] = []
        EvaluationLeague(_registry(), seed=1).run_tournament(
            ["learner", "learner@100"], self._play(first_calls)
        )
        EvaluationLeague(_registry(), seed=2).run_tournament(
            ["learner", "learner@100"], self._play(second_calls)
        )
        self.assertNotEqual(
            [call[2]["seed"] for call in first_calls],
            [call[2]["seed"] for call in second_calls],
        )

    def test_tournament_covers_every_condition(self):
        calls: list[tuple[str, str, dict]] = []
        league = EvaluationLeague(_registry(), seed=3)
        conditions = [
            {"map_id": "open_field", "lighting": "normal"},
            {"map_id": "night_yard", "lighting": "night"},
        ]
        league.run_tournament(["learner", "learner@100"], self._play(calls), conditions=conditions)
        self.assertEqual(len(calls), 4)
        self.assertEqual({call[2]["map_id"] for call in calls}, {"open_field", "night_yard"})
        self.assertEqual(len(league.by_map("learner")), 2)

    def test_tournament_fails_when_a_frozen_opponent_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s100.zip").write_bytes(b"v1")
            (root / "s200.zip").write_bytes(b"v1")
            registry = _registry(root)
            league = EvaluationLeague(registry, seed=3)

            def sabotage(policy_a: str, policy_b: str, condition: dict) -> MatchResult:
                (root / "s100.zip").write_bytes(b"v2-much-longer-weights")
                return _result(policy_a, policy_b, 1.0, seed=condition["seed"])

            with self.assertRaises(FrozenOpponentError):
                league.run_tournament(["learner", "learner@100"], sabotage)

    def test_play_match_must_return_a_match_result(self):
        league = EvaluationLeague(_registry(), seed=3)
        with self.assertRaises(TypeError):
            league.run_tournament(
                ["learner", "learner@100"], lambda a, b, condition: {"score": 1.0}
            )


if __name__ == "__main__":
    unittest.main()
