"""Tests for multi-seed experiment management, statistics, and regression detection."""
from __future__ import annotations

import math
import unittest

from sandboxai.experiment import (
    ExperimentConfig,
    aggregate_seed_runs,
    compare_experiments,
    compute_statistics,
    format_experiment_report,
)


class ExperimentStatisticsTests(unittest.TestCase):
    def test_compute_statistics_empty(self):
        stats = compute_statistics([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["std"], 0.0)

    def test_compute_statistics_known_values(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_statistics(values)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], 30.0)
        self.assertAlmostEqual(stats["median"], 30.0)
        self.assertAlmostEqual(stats["min"], 10.0)
        self.assertAlmostEqual(stats["max"], 50.0)
        # Sample standard deviation: sqrt(((10-30)^2 + (20-30)^2 + 0 + (40-30)^2 + (50-30)^2)/4) = sqrt((400+100+100+400)/4) = sqrt(250) = 15.8113883
        self.assertAlmostEqual(stats["std"], math.sqrt(250.0))
        self.assertTrue(stats["ci95_low"] < stats["mean"] < stats["ci95_high"])

    def test_aggregate_seed_runs(self):
        runs = [
            {"win_rate": 0.80, "mean_episode_reward": 12.5, "mean_accuracy": 0.45},
            {"win_rate": 0.85, "mean_episode_reward": 14.0, "mean_accuracy": 0.50},
            {"win_rate": 0.75, "mean_episode_reward": 11.0, "mean_accuracy": 0.40},
        ]
        agg = aggregate_seed_runs(runs)
        self.assertEqual(agg["runs_count"], 3)
        self.assertIn("win_rate", agg["metrics"])
        self.assertAlmostEqual(agg["metrics"]["win_rate"]["mean"], 0.80)
        self.assertAlmostEqual(agg["metrics"]["mean_episode_reward"]["mean"], 12.5)

    def test_compare_experiments_regression_detection(self):
        baseline = {
            "metrics": {
                "win_rate": {"mean": 0.85, "std": 0.02, "count": 5},
                "mean_episode_reward": {"mean": 15.0, "std": 1.0, "count": 5},
                "mean_accuracy": {"mean": 0.55, "std": 0.03, "count": 5},
            }
        }
        candidate_regressed = {
            "metrics": {
                "win_rate": {"mean": 0.60, "std": 0.03, "count": 5},
                "mean_episode_reward": {"mean": 8.0, "std": 1.5, "count": 5},
                "mean_accuracy": {"mean": 0.52, "std": 0.03, "count": 5},
            }
        }
        res = compare_experiments(baseline, candidate_regressed, threshold=0.05)
        self.assertEqual(res["status"], "regression_warning")
        self.assertTrue(any(r["metric"] == "win_rate" for r in res["regressions"]))
        self.assertTrue(any(r["metric"] == "mean_episode_reward" for r in res["regressions"]))

    def test_compare_experiments_improvement(self):
        baseline = {
            "metrics": {
                "win_rate": {"mean": 0.60, "std": 0.02, "count": 5},
                "mean_episode_reward": {"mean": 10.0, "std": 0.5, "count": 5},
            }
        }
        candidate_improved = {
            "metrics": {
                "win_rate": {"mean": 0.85, "std": 0.02, "count": 5},
                "mean_episode_reward": {"mean": 16.0, "std": 0.5, "count": 5},
            }
        }
        res = compare_experiments(baseline, candidate_improved, threshold=0.05)
        self.assertEqual(res["status"], "improved")
        self.assertEqual(len(res["regressions"]), 0)
        self.assertTrue(len(res["improvements"]) >= 1)

    def test_format_experiment_report(self):
        summary = {
            "runs_count": 3,
            "metrics": {
                "win_rate": {"mean": 0.80, "std": 0.05, "median": 0.80, "ci95_low": 0.74, "ci95_high": 0.86},
                "mean_accuracy": {"mean": 0.45, "std": 0.02, "median": 0.45, "ci95_low": 0.42, "ci95_high": 0.48},
            },
        }
        report = format_experiment_report(summary)
        self.assertIn("SANDBOXAI MULTI-SEED EXPERIMENT REPORT", report)
        self.assertIn("win_rate", report)
        self.assertIn("0.800", report)

    def test_experiment_config_validation(self):
        cfg = ExperimentConfig(name="test_exp", seeds=[1, 2, 3])
        cfg.validate()
        self.assertEqual(cfg.name, "test_exp")
        self.assertEqual(len(cfg.seeds), 3)

        with self.assertRaises(ValueError):
            ExperimentConfig(name="", seeds=[1, 2]).validate()

        with self.assertRaises(ValueError):
            ExperimentConfig(name="dup_seed", seeds=[1, 1]).validate()


if __name__ == "__main__":
    unittest.main()
