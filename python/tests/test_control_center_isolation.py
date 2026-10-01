"""Static guards for the headless-only Control Center architecture.

The Control Center is now one thing: the desktop application
(``python3 main.py`` / ``sandboxai control-center-desktop``) that operates
the headless training stack. There is no in-simulator operator scene any
more, and the training path must stay GUI-free exactly as before. These
tests parse sources instead of launching Godot or Tk so they run anywhere.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "python" / "sandboxai"

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

# Python modules on the training/bridge path: they must never import the
# GUI. The desktop Control Center imports them (through the adapter), never
# the other way around.
PYTHON_TRAINING_MODULES = (
    "ppo.py",
    "pipeline.py",
    "godot_env.py",
    "sharded_env.py",
    "benchmark.py",
    "benchmark_pipeline.py",
    "agents.py",
    "hardware_profile.py",
)

GUI_MODULE_NAMES = ("control_center_desktop", "control_center_pages", "control_center_widgets")


def read(relative: str) -> str:
    path = PROJECT_ROOT / relative
    assert path.is_file(), f"missing source file: {path}"
    return path.read_text(encoding="utf-8")


def _module_level_imports(path: Path) -> set[str]:
    """Modules imported at the top level of ``path`` (deferred imports excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


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
            "training must never launch a Control Center scene",
        )

    def test_default_main_scene_is_unchanged(self):
        source = read("project.godot")
        self.assertIn('run/main_scene="res://scenes/main.tscn"', source)
        self.assertIn("run/flush_stdout_on_print=true", source)

    def test_python_training_modules_never_import_the_gui(self):
        """A training run must stay importable and runnable without Tk."""
        for name in PYTHON_TRAINING_MODULES:
            path = PACKAGE / name
            assert path.is_file(), f"missing module: {path}"
            imported = _module_level_imports(path)
            for gui in GUI_MODULE_NAMES:
                self.assertNotIn(
                    gui,
                    imported,
                    f"{name} must not import {gui} at module level; the training "
                    "path must never require a GUI toolkit",
                )

    def test_the_in_simulator_control_center_scene_is_gone(self):
        """Headless-only: the rendered operator scene must not come back."""
        self.assertFalse(
            (PROJECT_ROOT / "scenes" / "control_center.tscn").exists(),
            "the Control Center is the desktop application; a rendered "
            "in-simulator operator scene would be a second, visual Control Center",
        )
        self.assertFalse(
            (PROJECT_ROOT / "scripts" / "control_center").exists(),
            "scripts/control_center/ was removed with the rendered operator UI",
        )


class HeadlessOnlyStartupTests(unittest.TestCase):
    """`python3 main.py` must launch the Control Center, nothing else."""

    def test_root_main_launches_the_desktop_control_center(self):
        source = read("main.py")
        self.assertIn("control_center_desktop", source)
        self.assertIn("def main() -> int:", source)
        # The bootstrap must work without a pip install of the package.
        self.assertIn("sys.path.insert", source)
        self.assertTrue((PROJECT_ROOT / "python" / "sandboxai" / "__init__.py").is_file())

    def test_desktop_control_center_stays_a_thin_view_over_the_adapter(self):
        """No RL logic may grow inside the GUI modules."""
        for name in ("control_center_desktop.py", "control_center_pages.py"):
            source = (PACKAGE / name).read_text(encoding="utf-8")
            for needle in (
                "import torch",
                "train_ppo(",
                "benchmark_simulation(",
                "PPO(",
            ):
                self.assertNotIn(
                    needle,
                    source,
                    f"{name} must not implement training/benchmark logic ({needle}); "
                    "it belongs to the adapter and the modules below it",
                )

    def test_gui_renders_unavailable_values_as_na(self):
        source = (PACKAGE / "control_center_viewmodel.py").read_text(encoding="utf-8")
        self.assertIn('"n/a"', source, "missing data renders as n/a, never estimated")


if __name__ == "__main__":
    unittest.main()
