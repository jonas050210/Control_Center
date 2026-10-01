"""Tests for the source-backed TTK Testing mechanics manifest."""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from sandboxai.cli import main
from sandboxai.ttk_testing import (
    TTK_TESTING_EVIDENCE,
    EvidenceStatus,
    calibration_required,
    format_status,
    status_summary,
    verified_mechanics,
)


class EvidenceManifestTests(unittest.TestCase):
    def test_manifest_has_unique_mechanics_and_official_sources(self) -> None:
        names = [item.mechanic for item in TTK_TESTING_EVIDENCE]
        self.assertEqual(len(names), len(set(names)))
        for item in TTK_TESTING_EVIDENCE:
            with self.subTest(mechanic=item.mechanic):
                self.assertTrue(item.source_url.startswith("https://"))
                self.assertTrue(item.source_label)
                self.assertTrue(item.implementation_rule)

    def test_current_controls_are_verified_without_inventing_numeric_rules(self) -> None:
        verified = {item.mechanic for item in verified_mechanics()}
        self.assertTrue({"fire", "aim", "crouch", "lean", "manual_weapon_swap"}.issubset(verified))
        pending = {item.mechanic for item in calibration_required()}
        self.assertTrue(
            {
                "recoil_values_and_pattern",
                "reload_behavior_and_timing",
                "movement_physics",
            }.issubset(pending)
        )

    def test_manual_swap_and_no_helmet_camera_are_explicit_product_rules(self) -> None:
        items = {item.mechanic: item for item in TTK_TESTING_EVIDENCE}
        self.assertEqual(items["automatic_weapon_switch"].status, EvidenceStatus.EXCLUDED)
        self.assertIn(
            "Never implement automatic", items["automatic_weapon_switch"].implementation_rule
        )
        self.assertEqual(items["helmet_camera"].status, EvidenceStatus.EXCLUDED)
        self.assertIn("normal first-person", items["helmet_camera"].implementation_rule)

    def test_summary_is_json_ready_and_honest_about_calibration_gaps(self) -> None:
        summary = status_summary()
        self.assertEqual(summary["target"], "Roblox TTK Testing")
        self.assertEqual(summary["rules"]["weapon_switch"], "manual_only")
        self.assertEqual(summary["rules"]["helmet_camera"], "excluded")
        encoded = json.dumps(summary)
        self.assertIn("calibration_required", encoded)
        self.assertIn("wound_painting_and_bleeding", encoded)

    def test_formatted_status_includes_each_evidence_class(self) -> None:
        text = format_status()
        self.assertIn("Verified mechanics", text)
        self.assertIn("Needs screenshot/manual calibration", text)
        self.assertIn("Excluded from this project", text)
        self.assertIn("manual only", text)


class CliTests(unittest.TestCase):
    def test_ttk_status_prints_text_and_json(self) -> None:
        for arguments, expected in (
            (["ttk-status"], "Verified mechanics"),
            (["ttk-status", "--json"], '"target"'),
        ):
            with self.subTest(arguments=arguments):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(arguments), 0)
                self.assertIn(expected, output.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
