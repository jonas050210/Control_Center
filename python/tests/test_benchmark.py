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


class FixedWindowMeasurementTests(unittest.TestCase):
    def measure(self, *, steps=None, warmup_steps=2, cancel_after=None):
        from unittest.mock import patch

        import numpy as np

        from sandboxai.benchmark import benchmark_simulation

        clock = {"now": 0.0}

        class Client:
            calls = 0
            closed = False

            def reset(self, seed):
                pass

            def step(self, actions):
                self.calls += 1
                clock["now"] += 0.5
                return None, None, np.zeros(len(actions)), []

            def close(self):
                self.closed = True

        client = Client()
        self.client = client
        progress = []
        with (
            patch("sandboxai.benchmark.make_batch_client", return_value=client),
            patch("sandboxai.benchmark.resource_snapshot", return_value={}),
            patch("sandboxai.benchmark.time.perf_counter", side_effect=lambda: clock["now"]),
        ):
            rows = benchmark_simulation(
                project_path=".",
                environment_counts=[2],
                steps=steps,
                max_seconds_per_config=20.0,
                warmup_steps=warmup_steps,
                on_step_progress=progress.append,
                cancel=(lambda: client.calls >= cancel_after) if cancel_after else None,
            )
        return rows[0], progress

    def test_fixed_window_cannot_end_at_an_early_step_target(self):
        row, progress = self.measure()
        self.assertEqual(row["measurement_mode"], "fixed_time")
        self.assertEqual(row["measurement_seconds"], 20.0)
        self.assertEqual(row["elapsed_seconds"], 20.0)
        self.assertEqual(row["warmup_seconds"], 1.0)
        self.assertEqual(row["steps_per_environment"], 40)
        self.assertEqual(row["total_steps"], 80)
        self.assertEqual(row["steps_per_second"], 4.0)
        self.assertTrue(all(event["target_steps"] is None for event in progress))
        self.assertTrue(self.client.closed)

    def test_explicit_steps_keep_the_scripted_step_budget(self):
        row, _ = self.measure(steps=3)
        self.assertEqual(row["steps_per_environment"], 3)
        self.assertEqual(row["elapsed_seconds"], 1.5)
        self.assertEqual(row["measurement_mode"], "steps")
        self.assertFalse(row["time_boxed"])

    def test_cancellation_closes_the_in_flight_client(self):
        from sandboxai.benchmark import BenchmarkCancelled

        with self.assertRaises(BenchmarkCancelled):
            self.measure(cancel_after=3)
        self.assertEqual(self.client.calls, 3)
        self.assertTrue(self.client.closed)
