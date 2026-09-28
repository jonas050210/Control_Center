"""Static guards that the Control Center stays OUT of the RL training path.

The Control Center is an operator/inspection tool. Training must be able to
run exactly as before — headless, with no GUI code constructed, no telemetry
built and no logging buffered. These tests parse the sources instead of
launching Godot so they run anywhere (Godot is not required in CI).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Files that are executed by (or that define) the headless RL hot path.
TRAINING_PATH_SOURCES = (
    "scripts/rl/rl_server.gd",
    "scripts/rl/rl_adapter.gd",
    "scripts/core/simulation_manager.gd",
    "scripts/env/environment_core.gd",
    "scripts/core/observation.gd",
    "scripts/core/action.gd",
    "scripts/reward/reward_system.gd",
    "scripts/input/controller_base.gd",
    "scripts/input/human_controller.gd",
    "scripts/input/ai_stub_controller.gd",
)

CONTROL_CENTER_DIR = PROJECT_ROOT / "scripts" / "control_center"


def read(relative: str) -> str:
    path = PROJECT_ROOT / relative
    assert path.is_file(), f"missing source file: {path}"
    return path.read_text(encoding="utf-8")


class TrainingPathIsolationTests(unittest.TestCase):
    def test_training_sources_never_reference_the_control_center(self):
        for relative in TRAINING_PATH_SOURCES:
            source = read(relative)
            self.assertNotIn(
                "control_center",
                source,
                f"{relative} must not depend on the Control Center; the GUI consumes "
                "simulation state, never the other way around",
            )

    def test_python_trainer_still_launches_the_headless_rl_server(self):
        source = read("python/sandboxai/godot_env.py")
        self.assertIn("--headless", source)
        self.assertIn("res://scripts/rl/rl_server.gd", source)
        self.assertNotIn(
            "control_center",
            source,
            "training must never launch the Control Center scene",
        )

    def test_default_main_scene_is_unchanged(self):
        source = read("project.godot")
        self.assertIn('run/main_scene="res://scenes/main.tscn"', source)
        self.assertIn("run/flush_stdout_on_print=true", source)

    def test_control_center_entry_point_guards_on_the_display_server(self):
        source = read("scripts/control_center/control_center_main.gd")
        self.assertIn('DisplayServer.get_name() == "headless"', source)
        self.assertIn("presentation_enabled", source)
        # The GUI must be constructed only behind that guard.
        gui_construction = source.find("ControlCenterUI.new()")
        guard = source.find("if not presentation_enabled:")
        self.assertGreater(gui_construction, -1, "the UI is created in the entry point")
        self.assertGreater(guard, -1, "an early return protects headless runs")
        self.assertLess(
            guard,
            gui_construction,
            "the headless early-return must come before any GUI construction",
        )

    def test_session_disables_telemetry_and_logging_in_training_mode(self):
        source = read("scripts/control_center/control_center_session.gd")
        self.assertIn("func telemetry_enabled() -> bool:", source)
        self.assertIn("return presentation_enabled and not is_training_mode()", source)
        self.assertIn("event_log.enabled = telemetry_enabled()", source)
        self.assertIn("if not telemetry_enabled():\n\t\treturn {}", source)

    def test_no_control_center_file_calls_engine_time_scale(self):
        for path in sorted(CONTROL_CENTER_DIR.rglob("*.gd")):
            source = path.read_text(encoding="utf-8")
            code = "\n".join(
                line for line in source.splitlines() if not line.strip().startswith("#")
            )
            self.assertNotIn(
                "Engine.time_scale =",
                code,
                f"{path.name} must not change Engine.time_scale; speed is implemented "
                "as simulation steps per frame so physics stay deterministic",
            )

    def test_event_log_is_bounded_and_throttled_by_default(self):
        source = read("scripts/control_center/control_center_event_log.gd")
        capacity = re.search(r"const DEFAULT_CAPACITY:\s*int\s*=\s*(\d+)", source)
        throttle = re.search(r"const DEFAULT_THROTTLE_SECONDS:\s*float\s*=\s*([\d.]+)", source)
        budget = re.search(r"const DEFAULT_MAX_EVENTS_PER_SECOND:\s*float\s*=\s*([\d.]+)", source)
        self.assertIsNotNone(capacity)
        self.assertIsNotNone(throttle)
        self.assertIsNotNone(budget)
        self.assertLessEqual(int(capacity.group(1)), 2000, "the log buffer must stay small")
        self.assertGreater(float(throttle.group(1)), 0.0, "repeat events must be throttled")
        self.assertGreater(float(budget.group(1)), 0.0, "an event storm must be bounded")

    def test_control_center_does_not_reimplement_simulation_logic(self):
        """The GUI may read state; it must not step or mutate the simulation."""
        forbidden = (
            "func step(",
            "agent.health =",
            "enemy.health =",
            "episode.cumulative_reward =",
            "agent.position =",
        )
        for path in sorted(CONTROL_CENTER_DIR.rglob("*.gd")):
            source = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(
                    needle,
                    source,
                    f"{path.name} must not duplicate or mutate simulation state ({needle})",
                )

    def test_ui_layer_is_separated_from_the_data_layer(self):
        """Data-layer Control Center files must not import UI/Control types."""
        data_layer = (
            "control_center_config.gd",
            "control_center_event_log.gd",
            "control_center_results.gd",
            "control_center_telemetry.gd",
            "observation_inspector.gd",
            "perception_model.gd",
            "system_monitor.gd",
            "training_agent_manager.gd",
            "training_run_history.gd",
        )
        for name in data_layer:
            source = (CONTROL_CENTER_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("/ui/", source, f"{name} must stay independent of the UI layer")
            self.assertNotIn(
                "extends Control", source, f"{name} must be a plain data/logic class"
            )


class ControlCenterSurfaceTests(unittest.TestCase):
    """The scene, entry point and panels the launch instructions promise."""

    def test_scene_exists_and_points_at_the_entry_script(self):
        scene = read("scenes/control_center.tscn")
        self.assertIn("res://scripts/control_center/control_center_main.gd", scene)

    def test_every_documented_panel_exists(self):
        expected = (
            "ui/status_bar.gd",
            "ui/agent_panel.gd",
            "ui/perception_panel.gd",
            "ui/observation_panel.gd",
            "ui/results_panel.gd",
            "ui/settings_panel.gd",
            "ui/controls_panel.gd",
            "ui/log_panel.gd",
            "ui/hud.gd",
            "ui/control_center_ui.gd",
            "ui/perception_map.gd",
            "ui/ui_theme.gd",
            "ui/home_panel.gd",
            "ui/agents_panel.gd",
            "ui/headless_panel.gd",
            "ui/headless_monitor_panel.gd",
            "ui/agent_card.gd",
            "ui/system_status_panel.gd",
            "ui/analytics_panel.gd",
            "ui/history_panel.gd",
            "ui/training_launch_panel.gd",
            "spectator_camera.gd",
            "perception_overlay_3d.gd",
        )
        for relative in expected:
            self.assertTrue(
                (CONTROL_CENTER_DIR / relative).is_file(),
                f"missing Control Center component: {relative}",
            )

    def test_unavailable_features_are_marked_not_faked(self):
        perception = read("scripts/control_center/perception_model.gd")
        self.assertIn("unavailable_note", perception)
        self.assertIn("has_method", perception)
        config = read("scripts/control_center/control_center_config.gd")
        self.assertIn("policy_source_available", config)

    def test_system_monitor_is_honest_and_hardware_agnostic(self):
        """PC telemetry may only relay measured values, for any GPU model."""
        source = read("scripts/control_center/system_monitor.gd")
        # Unavailable metrics must surface as N/A through one shared helper.
        self.assertIn('return "N/A"', source)
        self.assertIn("empty_snapshot", source)
        # GPU metrics come exclusively from real NVIDIA telemetry.
        self.assertIn("nvidia-smi", source)
        # No GPU model may be hard-coded (the RTX 4060 Ti must work because
        # nvidia-smi reports it, not because the code special-cases it).
        for needle in ("4060", "RTX", "GeForce"):
            self.assertNotIn(
                needle,
                source,
                f"system_monitor.gd must not hard-code the GPU model ({needle})",
            )
        # Polling stays slow and off the training path.
        match = re.search(r"const POLL_INTERVAL_SECONDS:\s*float\s*=\s*([\d.]+)", source)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(float(match.group(1)), 1.0, "system polling must stay cheap")

    def test_history_and_agent_registry_relay_backend_data_only(self):
        history = read("scripts/control_center/training_run_history.gd")
        self.assertIn("status.json", history)
        manager = read("scripts/control_center/training_agent_manager.gd")
        # The registry must not compute training metrics of its own; it
        # relays controller snapshots.
        self.assertIn("controller.snapshot()", manager)
        for needle in ("mean_episode_reward", "accuracy", "kills"):
            self.assertNotIn(
                needle,
                manager,
                "the agent registry must not synthesize backend metrics",
            )


if __name__ == "__main__":
    unittest.main()
