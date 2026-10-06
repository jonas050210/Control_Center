"""Headless Streamlit smoke test for the wired Playground controls."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class PlaygroundDashboardTests(unittest.TestCase):
    def test_shooter_aim_and_dodge_controls_render_and_advance(self) -> None:
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit AppTest is unavailable in this installation.")

        app_path = Path(__file__).resolve().parents[1] / "gui" / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        self.assertFalse(app.exception)
        navigation = next(radio for radio in app.radio if radio.label == "NAVIGATION")
        navigation.set_value("🔫  PLAYGROUND").run(timeout=30)
        self.assertFalse(app.exception)

        # The shooter is a working Gymnasium match, and taps really advance it.
        self.assertIn("🔴 FIRE", [button.label for button in app.button])
        app.session_state["human_demo_recording"] = True
        app.run(timeout=30)
        next(button for button in app.button if button.label == "🔴 FIRE").click().run(timeout=30)
        self.assertGreater(app.session_state["playground_env"].elapsed, 0.0)
        self.assertTrue(app.session_state["playground_demo_rows"])
        self.assertFalse(app.exception)

        # Target taps score, while the seeded dodge pad advances the run.
        next(box for box in app.selectbox if box.label == "PLAY MODE").set_value("🎯 Aim Trainer").run(timeout=30)
        next(button for button in app.button if button.label == "▶ Start / Restart Drill").click().run(timeout=30)
        next(button for button in app.button if button.label.startswith("🎯 ")).click().run(timeout=30)
        self.assertEqual(app.session_state["aim_game"]["hits"], 1)
        self.assertFalse(app.exception)

        next(box for box in app.selectbox if box.label == "PLAY MODE").set_value("⚡ Dodge Survival").run(timeout=30)
        next(button for button in app.button if button.label == "▶ New Run").click().run(timeout=30)
        next(button for button in app.button if button.label == "WAIT").click().run(timeout=30)
        self.assertEqual(app.session_state["dodge_game"]["tick"], 1)
        self.assertFalse(app.exception)

    def test_saved_policy_choice_runs_duel_path_and_fails_over_safely(self) -> None:
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit AppTest is unavailable in this installation.")
        import gui.tabs.arena as arena_tab
        import gui.tabs.playground as playground_tab

        app_path = Path(__file__).resolve().parents[1] / "gui" / "app.py"
        with TemporaryDirectory(prefix="neural-arena-policy-test-") as directory:
            models_dir = Path(directory)
            (models_dir / "test_policy.zip").write_bytes(b"invalid policy fixture")
            with patch.object(arena_tab, "MODELS_DIR", models_dir), \
                    patch.object(playground_tab, "MODELS_DIR", models_dir):
                app = AppTest.from_file(str(app_path)).run(timeout=30)
                next(radio for radio in app.radio if radio.label == "NAVIGATION").set_value(
                    "🔫  PLAYGROUND"
                ).run(timeout=30)
                next(selectbox for selectbox in app.selectbox
                     if selectbox.label == "BOT BEHAVIOR / PPO MODEL").set_value(
                    "test_policy.zip"
                ).run(timeout=30)
                next(button for button in app.button if button.label == "🔴 FIRE").click().run(timeout=30)
                self.assertGreater(app.session_state["playground_env"].elapsed, 0.0)
                self.assertTrue(any("Could not load test_policy.zip" in warning.value
                                    for warning in app.warning))
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
