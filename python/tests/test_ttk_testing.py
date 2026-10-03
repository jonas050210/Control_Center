"""Tests for the source-backed TTK Testing mechanics manifest."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import unittest.mock

from sandboxai.cli import main
from sandboxai.ttk_testing import (
    TTK_TESTING_EVIDENCE,
    EvidenceStatus,
    apply_ttk_calibration_preset,
    calculate_ttk_metrics,
    calibration_required,
    format_status,
    load_ttk_calibration,
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

    def test_calculate_ttk_metrics_computes_exact_stk_and_ttk_ms(self) -> None:
        metrics = calculate_ttk_metrics(damage=34.0, rpm=750.0, target_hp=100.0)
        self.assertEqual(metrics["shots_to_kill"], 3)
        self.assertAlmostEqual(metrics["ttk_ms"], 160.0, places=1)
        self.assertAlmostEqual(metrics["burst_dps"], 425.0, places=1)
        self.assertEqual(metrics["pace_class"], "INSTANT_LETHAL")

    def test_apply_ttk_calibration_preset_persists_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = apply_ttk_calibration_preset(tmp, "sable_cqb_carbine")
            self.assertTrue(res["ok"])
            cal = load_ttk_calibration(tmp)
            self.assertIn("weapon_damage_and_rpm_ttk_curve", cal)
            self.assertIn("160.0 ms TTK", cal["weapon_damage_and_rpm_ttk_curve"]["measured_value"])

    def test_probe_latest_roblox_log_parses_json_place_id_and_large_log_head(self) -> None:
        from pathlib import Path

        from sandboxai.ttk_testing import TTK_TESTING_PLACE_ID, _probe_latest_roblox_log

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "0.600.0_Player_20261002.log"
            head = (
                f"[FLog::Output] ! Joining game 'abc' place {TTK_TESTING_PLACE_ID} at 10.0.0.1\n"
                f'[DFLog::GameJoinLoadTime] {{"placeId":{TTK_TESTING_PLACE_ID},"universeId":9292879893}}\n'
            )
            filler = "FLog::Network telemetry tick\n" * 25000
            log_path.write_text(head + filler, encoding="utf-8")
            info = _probe_latest_roblox_log(tmp)
            self.assertEqual(info["detected_place_id"], TTK_TESTING_PLACE_ID)
            self.assertTrue(info["in_ttk_testing"])
            self.assertEqual(info["session_state"], "in_ttk_testing")


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

    def test_analyze_roblox_ttk_screenshot_extracts_png_dimensions(self) -> None:
        import struct
        import tempfile
        from pathlib import Path

        from sandboxai.ttk_testing import analyze_roblox_ttk_screenshot

        with tempfile.TemporaryDirectory() as tmp:
            shot_dir = Path(tmp) / ".sandboxai" / "ttk_screenshots"
            shot_dir.mkdir(parents=True)
            png = shot_dir / "ttk_1080p.png"
            ihdr = struct.pack(">IIBBBBB", 1920, 1080, 8, 6, 0, 0, 0)
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr)
            res = analyze_roblox_ttk_screenshot(tmp)
            self.assertTrue(res["ok"])
            self.assertEqual((res["width"], res["height"]), (1920, 1080))
            self.assertEqual(res["hud_layout"], "1080p-native")


class WindowsHostBridgeTests(unittest.TestCase):
    """Focus/probe/capture must work from WSL, not only from native Windows.

    The maintainer runs the Control Center inside WSL and the Roblox client
    on the Windows host: launching already worked (it goes through
    ``cmd.exe``), but every window helper bailed out on
    ``sys.platform.startswith("win")`` and did nothing. These tests drive the
    PowerShell route with a fake subprocess so the behaviour is pinned
    without a Windows machine.
    """

    def setUp(self) -> None:
        from sandboxai import ttk_testing

        self.ttk = ttk_testing
        self.calls: list[list[str]] = []
        self.results: list[tuple[int, str]] = []
        self.platform = self.ttk.sys.platform
        self.wsl = True
        self.powershell = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
        self._patchers = {
            "is_wsl": unittest.mock.patch.object(ttk_testing, "is_wsl", lambda: self.wsl),
            "powershell": unittest.mock.patch.object(
                ttk_testing, "_powershell", self._fake_powershell
            ),
            "run": unittest.mock.patch.object(ttk_testing, "_run_host_command", self._fake_run),
            "platform": unittest.mock.patch.object(ttk_testing.sys, "platform", "linux"),
        }
        for patcher in self._patchers.values():
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fake_powershell(self, *arguments: str) -> list[str] | None:
        if not self.wsl:
            return None
        return [*self.powershell, *arguments]

    def _fake_run(self, command: list[str], *, timeout: float) -> tuple[int, str]:
        del timeout
        self.calls.append(list(command))
        if not self.results:
            return 1, "unexpected call"
        return self.results.pop(0)

    def test_host_bridge_is_powershell_under_wsl(self) -> None:
        self.assertEqual(self.ttk.host_bridge(), "powershell")

    def test_host_bridge_is_empty_without_a_windows_host(self) -> None:
        self.wsl = False
        self.assertEqual(self.ttk.host_bridge(), "")

    def test_focus_reports_success_and_runs_the_host_script(self) -> None:
        self.results = [(0, "OK")]
        result = self.ttk.focus_roblox_window()
        self.assertTrue(result["ok"])
        self.assertTrue(result["focused"])
        self.assertEqual(len(self.calls), 1)
        self.assertIn("SetForegroundWindow", self.calls[0][-1])

    def test_focus_names_the_missing_client_instead_of_the_platform(self) -> None:
        self.results = [(1, "NO_WINDOW")]
        result = self.ttk.focus_roblox_window()
        self.assertFalse(result["ok"])
        self.assertIn("Start Roblox Player", result["message"])
        self.assertNotIn("requires Windows", result["message"])

    def test_probe_parses_the_window_rectangle(self) -> None:
        self.results = [(0, "WINDOW|4242|10,20,1930,1100|1|TTK Testing [MAP VOTING]")]
        probe = self.ttk._probe_roblox_window()
        self.assertTrue(probe["window_found"])
        self.assertEqual(probe["window_pid"], 4242)
        self.assertEqual(probe["window_rect"], [10, 20, 1930, 1100])
        self.assertEqual((probe["window_width"], probe["window_height"]), (1920, 1080))
        self.assertTrue(probe["window_focused"])
        self.assertEqual(probe["window_title"], "TTK Testing [MAP VOTING]")

    def test_probe_reports_no_window_without_crashing(self) -> None:
        self.results = [(0, "NONE")]
        probe = self.ttk._probe_roblox_window()
        self.assertFalse(probe["window_found"])
        self.assertIsNone(probe.get("window_rect"))

    def test_capture_writes_into_the_captures_directory(self) -> None:
        import tempfile
        from pathlib import Path

        from sandboxai.wsl import windows_to_wsl_path

        def run(command: list[str], *, timeout: float) -> tuple[int, str]:
            del timeout
            self.calls.append(list(command))
            # Stand in for the Windows host: create the file PowerShell would
            # have written, at the path the script was handed. A
            # ``wsl.localhost`` UNC target is not a POSIX filename,
            # so translate it back before simulating the host write.
            script = command[-1]
            marker = "$bmp.Save('"
            start = script.index(marker) + len(marker)
            end = script.index("',", start)
            target = script[start:end].replace("''", "'")
            host_target = target if Path(target).is_absolute() else windows_to_wsl_path(target)
            Path(host_target).write_bytes(b"\x89PNG")
            return 0, target

        self._patchers["run"].stop()
        self._patchers["run"] = unittest.mock.patch.object(self.ttk, "_run_host_command", run)
        self._patchers["run"].start()

        with tempfile.TemporaryDirectory() as tmp:
            self.results = [(0, "NONE")]  # no window -> whole screen
            result = self.ttk.capture_roblox_screenshot(tmp)
            self.assertTrue(result["ok"], result)
            self.assertEqual(Path(result["path"]).parent.name, "ttk_captures")
            self.assertTrue(Path(result["path"]).is_file())
            self.assertFalse(result["window_captured"])

    def test_capture_reports_the_host_error_instead_of_a_platform_hint(self) -> None:
        import tempfile

        self.results = [(0, "NONE"), (1, "EMPTY_RECT")]  # probe, then capture
        with tempfile.TemporaryDirectory() as tmp:
            result = self.ttk.capture_roblox_screenshot(tmp)
        self.assertFalse(result["ok"])
        self.assertIn("EMPTY_RECT", result["error"])

    def test_capture_says_what_is_missing_without_a_host(self) -> None:
        import tempfile

        self.wsl = False
        with tempfile.TemporaryDirectory() as tmp:
            result = self.ttk.capture_roblox_screenshot(tmp)
        self.assertFalse(result["ok"])
        self.assertIn("PowerShell", result["error"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
