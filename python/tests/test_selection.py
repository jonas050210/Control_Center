"""Checkpoint-selection rule: semantics, provenance and resume behaviour."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sandboxai.config import TrainingConfig
from sandboxai.selection import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_GOALS,
    CheckpointSelectionRule,
)


class SelectionRuleTests(unittest.TestCase):
    def test_default_rule_reproduces_historical_behaviour(self):
        rule = CheckpointSelectionRule()
        self.assertEqual(rule.metric, DEFAULT_SELECTION_METRIC)
        self.assertEqual(rule.goal, "max")
        self.assertEqual(rule.min_delta, 0.0)
        self.assertEqual(rule.initial_score(), float("-inf"))
        # Strictly greater, exactly as the old `reward > best_score`.
        self.assertTrue(rule.is_improvement(1.0, float("-inf")))
        self.assertTrue(rule.is_improvement(1.0, 0.999))
        self.assertFalse(rule.is_improvement(1.0, 1.0))
        self.assertFalse(rule.is_improvement(0.5, 1.0))

    def test_minimisation_inverts_the_comparison(self):
        rule = CheckpointSelectionRule(metric="mean_deaths", goal="min")
        self.assertEqual(rule.initial_score(), float("inf"))
        self.assertTrue(rule.is_improvement(3.0, float("inf")))
        self.assertTrue(rule.is_improvement(1.0, 2.0))
        self.assertFalse(rule.is_improvement(2.0, 2.0))
        self.assertFalse(rule.is_improvement(3.0, 2.0))

    def test_min_delta_suppresses_near_noise_churn(self):
        rule = CheckpointSelectionRule(min_delta=0.5)
        self.assertFalse(rule.is_improvement(1.4, 1.0))
        self.assertFalse(rule.is_improvement(1.5, 1.0))
        self.assertTrue(rule.is_improvement(1.6, 1.0))
        low = CheckpointSelectionRule(metric="mean_ttk", goal="min", min_delta=0.5)
        self.assertFalse(low.is_improvement(0.6, 1.0))
        self.assertTrue(low.is_improvement(0.4, 1.0))

    def test_score_reads_nested_and_rejects_non_numeric(self):
        rule = CheckpointSelectionRule(metric="condition_evaluation.mean_win_rate")
        summary = {"condition_evaluation": {"mean_win_rate": 0.25}}
        self.assertAlmostEqual(rule.score(summary), 0.25)
        self.assertIsNone(rule.score({"condition_evaluation": {}}))
        self.assertIsNone(rule.score({}))
        self.assertIsNone(CheckpointSelectionRule(metric="a").score({"a": "text"}))
        self.assertIsNone(CheckpointSelectionRule(metric="a").score({"a": True}))
        self.assertIsNone(CheckpointSelectionRule(metric="a").score({"a": float("nan")}))
        self.assertIsNone(CheckpointSelectionRule(metric="a").score({"a": float("inf")}))

    def test_non_finite_candidate_is_never_an_improvement(self):
        rule = CheckpointSelectionRule()
        self.assertFalse(rule.is_improvement(float("nan"), 0.0))
        self.assertFalse(rule.is_improvement(float("-inf"), float("-inf")))

    def test_invalid_rules_are_rejected(self):
        with self.assertRaises(ValueError):
            CheckpointSelectionRule(metric="  ")
        with self.assertRaises(ValueError):
            CheckpointSelectionRule(goal="highest")
        with self.assertRaises(ValueError):
            CheckpointSelectionRule(min_delta=-0.1)
        for goal in SELECTION_GOALS:
            CheckpointSelectionRule(goal=goal)

    def test_provenance_roundtrip_and_comparability(self):
        rule = CheckpointSelectionRule(metric="win_rate", goal="max", min_delta=0.01)
        payload = rule.as_dict()
        self.assertEqual(CheckpointSelectionRule.from_dict(payload), rule)
        # A pre-rule best.json was selected by the historical default.
        self.assertTrue(CheckpointSelectionRule().matches(None))
        self.assertFalse(rule.matches(None))
        # min_delta only tightens the threshold: still the same quantity.
        self.assertTrue(
            rule.matches({"metric": "win_rate", "goal": "max", "min_delta": 0.5})
        )
        self.assertFalse(rule.matches({"metric": "win_rate", "goal": "min"}))
        self.assertFalse(rule.matches({"metric": "mean_episode_reward", "goal": "max"}))

    def test_describe_is_human_readable(self):
        self.assertEqual(
            CheckpointSelectionRule().describe(), "maximise mean_episode_reward"
        )
        self.assertEqual(
            CheckpointSelectionRule(metric="mean_ttk", goal="min", min_delta=0.25).describe(),
            "minimise mean_ttk by more than 0.25",
        )


class SelectionConfigTests(unittest.TestCase):
    def test_training_config_defaults_are_the_historical_rule(self):
        rule = TrainingConfig().checkpoint_selection_rule()
        self.assertEqual(rule, CheckpointSelectionRule())

    def test_training_config_carries_the_rule(self):
        config = TrainingConfig(
            checkpoint_selection_metric="win_rate",
            checkpoint_selection_goal="max",
            checkpoint_selection_min_delta=0.02,
        ).validate()
        self.assertEqual(
            config.checkpoint_selection_rule(),
            CheckpointSelectionRule(metric="win_rate", goal="max", min_delta=0.02),
        )

    def test_invalid_rule_fails_config_validation_not_the_first_evaluation(self):
        with self.assertRaises(ValueError):
            TrainingConfig(checkpoint_selection_goal="best").validate()
        with self.assertRaises(ValueError):
            TrainingConfig(checkpoint_selection_min_delta=-1.0).validate()
        with self.assertRaises(ValueError):
            TrainingConfig(checkpoint_selection_metric="").validate()

    def test_rule_survives_a_config_json_roundtrip(self):
        config = TrainingConfig(
            checkpoint_selection_metric="condition_evaluation.mean_win_rate",
            checkpoint_selection_goal="max",
            checkpoint_selection_min_delta=0.05,
        ).validate()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            restored = TrainingConfig.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        self.assertEqual(
            restored.checkpoint_selection_rule(), config.checkpoint_selection_rule()
        )


class SelectionResumeTests(unittest.TestCase):
    """The resume path in train_ppo, exercised without running training."""

    @staticmethod
    def _inherit(rule: CheckpointSelectionRule, record: dict) -> float:
        """Mirror of train_ppo's best.json inheritance logic."""
        if not rule.matches(record.get("selection_rule")):
            return rule.initial_score()
        inherited = record.get("score", record.get("mean_reward"))
        if isinstance(inherited, (int, float)) and not isinstance(inherited, bool):
            return float(inherited)
        return rule.initial_score()

    def test_legacy_best_json_is_inherited_by_the_default_rule(self):
        rule = CheckpointSelectionRule()
        record = {"mean_reward": 12.5, "timesteps": 1000, "checkpoint": "best_eval.zip"}
        self.assertEqual(self._inherit(rule, record), 12.5)

    def test_incomparable_rule_restarts_selection(self):
        rule = CheckpointSelectionRule(metric="win_rate")
        record = {
            "mean_reward": 12.5,
            "score": 12.5,
            "selection_rule": CheckpointSelectionRule().as_dict(),
        }
        self.assertEqual(self._inherit(rule, record), float("-inf"))

    def test_matching_rule_inherits_its_own_score(self):
        rule = CheckpointSelectionRule(metric="win_rate")
        record = {"mean_reward": 12.5, "score": 0.4, "selection_rule": rule.as_dict()}
        self.assertAlmostEqual(self._inherit(rule, record), 0.4)


if __name__ == "__main__":
    unittest.main()
