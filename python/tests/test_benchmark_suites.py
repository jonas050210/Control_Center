"""Tests for the benchmark suite plan (Phase 13).

These tests verify the *plan* and the report arithmetic. They deliberately
do not verify throughput numbers: those require a real Godot executable,
and this suite asserts that the module refuses to invent them when one is
absent.
"""
from __future__ import annotations

import unittest
from unittest import mock

from sandboxai.benchmark import DEFAULT_ENVIRONMENT_COUNTS
from sandboxai.benchmark_suites import (
    SUITE_ENVIRONMENT_COUNTS,
    SUITES,
    SUITES_BY_NAME,
    BenchmarkEnvironmentReport,
    BenchmarkUnavailableError,
    describe_plan,
    format_report,
    godot_available,
    overhead_table,
    run_suite,
    run_suites,
)
from sandboxai.contract import ACTION_NVEC


class PlanTests(unittest.TestCase):
    def test_required_environment_counts(self):
        self.assertEqual(SUITE_ENVIRONMENT_COUNTS, (1, 4, 8, 16, 32, 64))

    def test_five_distinct_suites(self):
        self.assertEqual(len(SUITES), 5)
        self.assertEqual(len(SUITES_BY_NAME), 5)
        self.assertEqual(
            sorted(SUITES_BY_NAME),
            [
                "curriculum_1_4",
                "curriculum_5_10",
                "map_analyzer",
                "perception_combat",
                "weapon_handling",
            ],
        )

    def test_suites_cover_the_requested_workloads(self):
        self.assertLessEqual(SUITES_BY_NAME["curriculum_1_4"].curriculum_level, 4)
        self.assertGreaterEqual(SUITES_BY_NAME["curriculum_5_10"].curriculum_level, 5)
        self.assertLessEqual(SUITES_BY_NAME["curriculum_5_10"].curriculum_level, 10)
        self.assertIn("perception", SUITES_BY_NAME["perception_combat"].stresses)
        self.assertIn("exploration", SUITES_BY_NAME["map_analyzer"].stresses)
        self.assertGreater(
            SUITES_BY_NAME["perception_combat"].enemy_count,
            SUITES_BY_NAME["curriculum_1_4"].enemy_count,
        )

    def test_weapon_handling_suite_isolates_the_handling_layer(self):
        handling = SUITES_BY_NAME["weapon_handling"]
        baseline = SUITES_BY_NAME["curriculum_5_10"]
        # The handling layer only switches on from level 5, so the suite
        # has to sit at or above it to measure anything at all.
        self.assertGreaterEqual(handling.curriculum_level, 5)
        self.assertIn("weapon_handling", handling.stresses)
        self.assertNotIn("weapon_handling", baseline.stresses)
        self.assertEqual(handling.environment_counts, baseline.environment_counts)

    def test_every_suite_uses_the_same_counts_so_they_are_comparable(self):
        for suite in SUITES:
            self.assertEqual(suite.environment_counts, SUITE_ENVIRONMENT_COUNTS)

    def test_plan_contains_no_measurements(self):
        plan = describe_plan()
        self.assertFalse(plan["measured"])
        self.assertEqual(plan["environment_counts"], list(SUITE_ENVIRONMENT_COUNTS))
        serialized = str(plan)
        for forbidden in ("steps_per_second", "elapsed_seconds"):
            self.assertNotIn(forbidden, serialized)


class RefusalTests(unittest.TestCase):
    def test_running_without_godot_raises(self):
        with mock.patch("sandboxai.benchmark_suites.shutil.which", return_value=None):
            self.assertFalse(godot_available("godot"))
            with self.assertRaises(BenchmarkUnavailableError):
                run_suite(SUITES[0], project_path=".")
            with self.assertRaises(BenchmarkUnavailableError):
                run_suites(project_path=".")

    def test_environment_report_states_that_nothing_was_measured(self):
        with mock.patch("sandboxai.benchmark_suites.shutil.which", return_value=None):
            report = BenchmarkEnvironmentReport().build()
        self.assertFalse(report["godot_available"])
        self.assertFalse(report["can_measure"])
        self.assertTrue(any("not found" in note for note in report["notes"]))

    def test_environment_report_when_godot_exists(self):
        with mock.patch("sandboxai.benchmark_suites.shutil.which", return_value="/usr/bin/godot"):
            report = BenchmarkEnvironmentReport().build()
        self.assertTrue(report["can_measure"])
        self.assertEqual(report["suites"], [suite.name for suite in SUITES])

    def test_formatting_an_unmeasured_report_says_so(self):
        text = format_report(describe_plan())
        self.assertIn("No benchmark measurements available", text)


class OverheadTests(unittest.TestCase):
    def _result(self, name: str, throughputs: dict[int, float]) -> dict:
        return {
            "suite": name,
            "config": {"description": name},
            "rows": [
                {
                    "environments": count,
                    "steps_per_second": value,
                    "episodes_per_second": 1.0,
                    "elapsed_seconds": 10.0,
                }
                for count, value in throughputs.items()
            ],
            "scaling": {},
        }

    def test_slowdown_is_relative_to_the_cheapest_suite(self):
        results = [
            self._result("curriculum_1_4", {1: 1000.0, 4: 2000.0}),
            self._result("perception_combat", {1: 500.0, 4: 500.0}),
        ]
        table = overhead_table(results)
        entry = table["by_suite"]["perception_combat"]
        self.assertAlmostEqual(entry["slowdown_by_environment_count"][1], 2.0)
        self.assertAlmostEqual(entry["slowdown_by_environment_count"][4], 4.0)
        self.assertAlmostEqual(entry["mean_slowdown"], 3.0)

    def test_only_shared_environment_counts_are_compared(self):
        results = [
            self._result("curriculum_1_4", {1: 1000.0}),
            self._result("map_analyzer", {1: 500.0, 64: 100.0}),
        ]
        entry = overhead_table(results)["by_suite"]["map_analyzer"]
        self.assertEqual(list(entry["slowdown_by_environment_count"]), [1])

    def test_missing_baseline_yields_no_table(self):
        self.assertEqual(overhead_table([self._result("map_analyzer", {1: 1.0})]), {})

    def test_formatting_a_measured_report(self):
        report = {
            "measured": True,
            "suites": [self._result("curriculum_1_4", {1: 1000.0, 4: 2000.0})],
            "overhead": {},
        }
        text = format_report(report)
        self.assertIn("curriculum_1_4", text)
        self.assertIn("1000.0", text)


class ContractDriftTests(unittest.TestCase):
    def test_the_benchmark_idle_action_matches_the_action_contract(self):
        # A regression guard: the benchmark used to send a 5-wide action
        # against a 6-wide contract.
        source = (
            __import__("sandboxai.benchmark", fromlist=["benchmark"]).__file__  # type: ignore[attr-defined]
        )
        text = open(source, encoding="utf-8").read()
        self.assertIn("ACTION_NVEC", text)
        self.assertNotIn("[1, 1, 1, 1, 0]", text)
        self.assertEqual(len(ACTION_NVEC), 6)

    def test_default_sweep_includes_every_suite_count(self):
        for count in SUITE_ENVIRONMENT_COUNTS:
            self.assertIn(count, DEFAULT_ENVIRONMENT_COUNTS)


if __name__ == "__main__":
    unittest.main()
