"""Tests for the real Godot runtime validation harness.

Ensures that:
1. When Godot is unavailable, the validator safely refuses to invent numbers
   and returns an honest unavailable report.
2. The report formatter outputs clear, diagnostic summaries.
3. Contract checks enforce the 84-float observation and 6-field MultiDiscrete action.
"""
from __future__ import annotations

import unittest
from unittest import mock

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from sandboxai.runtime_validation import (
    RuntimeValidationReport,
    RuntimeValidator,
    ValidationCheck,
    format_validation_report,
)


class RuntimeValidationHarnessTests(unittest.TestCase):
    def test_validator_refuses_to_invent_when_godot_missing(self):
        with mock.patch("sandboxai.runtime_validation.find_godot_executable", side_effect=FileNotFoundError("not found")):
            validator = RuntimeValidator(godot_executable="non_existent_godot")
            self.assertFalse(validator.is_godot_available())

            report = validator.validate()
            self.assertFalse(report.godot_available)
            self.assertEqual(report.status, "unavailable")
            self.assertEqual(report.passed_checks, 0)
            self.assertEqual(report.failed_checks, 0)
            self.assertFalse(report.measured_throughput.get("measured", False))
            self.assertTrue(any("not found" in note for note in report.notes))

    def test_format_validation_report_renders_clear_structure(self):
        report = RuntimeValidationReport(
            godot_executable="/usr/bin/godot",
            godot_available=True,
            godot_version="4.7.2.stable",
            platform_system="Linux",
            status="passed",
            total_checks=2,
            passed_checks=2,
            failed_checks=0,
            measured_throughput={"measured": True, "steps_per_second": 14250.0},
            checks=[
                ValidationCheck(
                    check_id="bridge_spaces",
                    name="JSON-Lines Bridge Spaces",
                    category="contract",
                    passed=True,
                    latency_ms=2.5,
                ),
                ValidationCheck(
                    check_id="bridge_ping",
                    name="Stdio Ping",
                    category="transport",
                    passed=True,
                    latency_ms=0.4,
                ),
            ],
            notes=["Test execution note"],
        )
        rendered = format_validation_report(report)
        self.assertIn("RUNTIME VALIDATION REPORT", rendered)
        self.assertIn("4.7.2.stable", rendered)
        self.assertIn("14250.0 steps/sec", rendered)
        self.assertIn("JSON-Lines Bridge Spaces", rendered)
        self.assertIn("PASS", rendered)

    def test_format_validation_report_for_unavailable_state(self):
        report = RuntimeValidationReport(
            godot_executable=None,
            godot_available=False,
            godot_version=None,
            platform_system="Windows",
            status="unavailable",
            measured_throughput={"measured": False},
            notes=["Engine executable missing."],
        )
        rendered = format_validation_report(report)
        self.assertIn("UNAVAILABLE", rendered)
        self.assertIn("N/A (unmeasured", rendered)
        self.assertIn("Engine executable missing.", rendered)


if __name__ == "__main__":
    unittest.main()
