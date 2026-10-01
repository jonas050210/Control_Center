"""Static guards for the headless-only Control Center architecture.

The Control Center is now one thing: the desktop application
(``python3 main.py`` / ``sandboxai control-center-desktop``) that operates
the headless training stack. There is no in-simulator operator scene any
more, and the training path must stay GUI-free exactly as before. These
tests parse sources instead of launching Godot or Tk so they run anywhere.
"""

from __future__ import annotations

import ast
import shutil
import tempfile
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

# Leftovers that Windows/OneDrive, Godot and editors drop into a directory
# whose tracked files are already deleted. None of them carries GDScript or
# scene data, so none of them can resurrect the operator UI; they only mark a
# working tree that still has to be swept. ``.uid``/``.import`` are Godot
# sidecars that point at resources which no longer exist, and the editor
# regenerates or drops them on the next import.
INERT_LEFTOVER_NAMES = frozenset(
    {"desktop.ini", "thumbs.db", ".ds_store", ".gitkeep", ".gitignore", ".directory"}
)
INERT_LEFTOVER_SUFFIXES = frozenset(
    {".uid", ".import", ".tmp", ".temp", ".swp", ".swo", ".bak", ".orig", ".rej", ".pyc", ".log"}
)
# Directories that only ever hold regenerated caches.
INERT_LEFTOVER_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})


def read(relative: str) -> str:
    path = PROJECT_ROOT / relative
    assert path.is_file(), f"missing source file: {path}"
    return path.read_text(encoding="utf-8")


def _is_inert_leftover(path: Path) -> bool:
    """True when ``path`` is sync/editor cruft rather than restorable source."""
    if any(part.lower() in INERT_LEFTOVER_DIRS for part in path.parts):
        return True
    return (
        path.name.lower() in INERT_LEFTOVER_NAMES or path.suffix.lower() in INERT_LEFTOVER_SUFFIXES
    )


def _restorable_entries(directory: Path) -> list[str]:
    """Files under ``directory`` that could put the operator UI back.

    Git does not track empty directories, and a Windows/OneDrive checkout
    keeps the folder (plus a ``desktop.ini``) around long after its tracked
    files are gone. Those machine-local remnants are not a second Control
    Center, so the guard looks for real content: any file that is neither
    inert cruft nor inside a regenerated cache directory.
    """
    if not directory.is_dir():
        return []
    found: list[str] = []
    for child in sorted(directory.rglob("*")):
        if child.is_dir():
            continue
        relative = child.relative_to(directory)
        if _is_inert_leftover(relative):
            continue
        found.append(relative.as_posix())
    return found


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
        scene = PROJECT_ROOT / "scenes" / "control_center.tscn"
        self.assertFalse(
            scene.exists(),
            "the Control Center is the desktop application; a rendered "
            "in-simulator operator scene would be a second, visual Control Center",
        )
        directory = PROJECT_ROOT / "scripts" / "control_center"
        # A plain file at that path is never legitimate, whatever it holds.
        self.assertFalse(
            directory.exists() and not directory.is_dir(),
            f"{directory} must not exist; the Control Center is the desktop application",
        )
        # Guard against restorable content, not against a machine-local
        # directory entry: git cannot track an empty directory, and a
        # Windows/OneDrive working copy routinely keeps the folder (and a
        # desktop.ini inside it) after the tracked GDScript files are deleted.
        restorable = _restorable_entries(directory)
        self.assertEqual(
            restorable,
            [],
            "scripts/control_center/ was removed with the rendered operator UI, "
            f"but {directory} still contains {', '.join(restorable)}. "
            "Do not restore the directory: delete those files (git clean -xdf "
            "scripts/control_center, or rm -rf scripts/control_center in a "
            "non-git copy of the sources)",
        )


class RestorableEntryDetectionTests(unittest.TestCase):
    """The leftover filter must stay strict about anything that is source."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp()) / "control_center"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory.parent, True)

    def write(self, relative: str) -> None:
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    def test_missing_or_empty_directory_has_no_restorable_entries(self):
        self.assertEqual(_restorable_entries(self.directory / "absent"), [])
        self.assertEqual(_restorable_entries(self.directory), [])

    def test_sync_and_cache_leftovers_are_not_restorable(self):
        for relative in (
            "desktop.ini",
            "Thumbs.db",
            ".DS_Store",
            "control_center_main.gd.uid",
            "icon.png.import",
            "__pycache__/stale.pyc",
            "notes.txt.bak",
        ):
            self.write(relative)
        self.assertEqual(_restorable_entries(self.directory), [])

    def test_operator_ui_sources_are_reported(self):
        self.write("control_center_main.gd")
        self.write("panels/inspector.tscn")
        self.write("desktop.ini")
        self.assertEqual(
            _restorable_entries(self.directory),
            ["control_center_main.gd", "panels/inspector.tscn"],
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
