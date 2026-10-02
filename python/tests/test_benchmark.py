"""Unit tests for the pure (Godot-free) parts of the benchmark module."""

import unittest

from sandboxai.benchmark import DEFAULT_ENVIRONMENT_COUNTS, percentile, summarize_scaling


class PercentileTests(unittest.TestCase):
    def test_empty_and_interpolated_percentiles(self):
        self.assertEqual(percentile([], 0.95), 0.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.0), 1.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 1.0), 4.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_invalid_quantile_is_rejected(self):
        with self.assertRaises(ValueError):
            percentile([1.0], 1.1)


class SummarizeScalingTests(unittest.TestCase):
    def test_empty_results_return_empty_summary(self):
        self.assertEqual(summarize_scaling([]), {})

    def test_picks_the_environment_count_with_highest_throughput(self):
        results = [
            {"environments": 1, "steps_per_second": 100.0},
            {"environments": 8, "steps_per_second": 700.0},
            {"environments": 16, "steps_per_second": 750.0},
        ]
        summary = summarize_scaling(results)
        self.assertEqual(summary["best_environment_count"], 16)
        self.assertAlmostEqual(summary["best_steps_per_second"], 750.0)

    def test_flags_diminishing_returns_when_throughput_growth_lags_env_growth(self):
        results = [
            {"environments": 1, "steps_per_second": 100.0},
            {"environments": 8, "steps_per_second": 750.0},  # near-linear, no flag yet
            {"environments": 64, "steps_per_second": 800.0},  # 8x envs, ~flat throughput
        ]
        summary = summarize_scaling(results)
        self.assertEqual(summary["diminishing_returns_at_environment_count"], 64)

    def test_default_environment_counts_cover_the_recommended_sweep(self):
        self.assertEqual(DEFAULT_ENVIRONMENT_COUNTS, (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128))
        self.assertEqual(DEFAULT_ENVIRONMENT_COUNTS, tuple(sorted(DEFAULT_ENVIRONMENT_COUNTS)))


if __name__ == "__main__":
    unittest.main()
