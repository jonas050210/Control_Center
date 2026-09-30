"""Desktop Control Center GUI tests.

These construct the *real* Tk window against a *real* SandboxAIAdapter
pointed at a throwaway ``tmp_path`` project (never a mock/fake adapter that
would risk masking a real wiring bug between the GUI and the adapter) - but
never launch ``mainloop()``: every check drives the widgets and drains the
background-work queue directly, which is the "injectable adapter at the
unit-test boundary" pattern for GUI code without needing an event loop.

Skipped when Tkinter is not importable (this sandbox) or when no display
server is available to open a Tk window (headless CI without Xvfb); both are
environment facts, not regressions, per the project's testing rules.
"""
import json
import time
import unittest
from pathlib import Path

from optional_deps import HAS_TKINTER, TKINTER_REASON

from sandboxai.adapter import SandboxAIAdapter
from sandboxai.config import TrainingConfig

if HAS_TKINTER:
    import tkinter as tk

    from sandboxai.control_center_desktop import PAGE_CLASSES, ControlCenter


def _make_app(project_root: Path) -> "ControlCenter":
    adapter = SandboxAIAdapter(project_root=project_root, output_root=project_root / "training")
    return ControlCenter(adapter=adapter)


def _drain_background(app: "ControlCenter", attempts: int = 20, delay: float = 0.05) -> None:
    """Runs the background-thread -> Tk-queue hand-off without ``mainloop()``.

    Background work is genuinely asynchronous (real thread pool), so this
    polls the same private queue ``BackgroundRunner._pump`` would drain on a
    timer, giving submitted work a bounded chance to finish before a test
    asserts on its result.
    """
    for _ in range(attempts):
        app.background._pump()
        app.update()
        time.sleep(delay)


@unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
class ControlCenterConstructionTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project_root = Path(self._tmp.name)
        try:
            self.app = _make_app(self.project_root)
        except tk.TclError as exc:
            self.skipTest(f"no display available for Tk: {exc}")
        self.addCleanup(self._safe_destroy)

    def _safe_destroy(self):
        try:
            self.app._on_close()
        except tk.TclError:
            pass

    def test_every_required_page_is_registered(self):
        titles = {page_class.title for page_class in PAGE_CLASSES}
        assert titles == {
            "Dashboard", "Training", "Agents", "Benchmarks", "Evaluations",
            "Runs / Checkpoints", "System / Telemetry", "Settings",
        }

    def test_every_page_builds_and_refreshes_without_raising(self):
        for page_class in PAGE_CLASSES:
            self.app.show_page(page_class.title)
            _drain_background(self.app)
        # No exception means every page's build()/refresh() survived an
        # empty (no runs yet) project directory - the most common state a
        # fresh user will actually see.

    def test_dashboard_reflects_a_real_run_directory(self):
        run_dir = self.project_root / "training" / "runs" / "run-a"
        run_dir.mkdir(parents=True)
        TrainingConfig(total_training_steps=1000, output_root=str(self.project_root / "training"),
                        run_id="run-a").save(run_dir / "config.json")
        (run_dir / "status.json").write_text(json.dumps({
            "state": "Running", "timesteps": 250, "total_training_steps": 1000, "updated_at": time.time(),
        }))
        self.app.show_page("Dashboard")
        _drain_background(self.app)
        page = self.app.pages["Dashboard"]
        assert page._last_run_dir == str(run_dir)

    def test_training_form_rejects_invalid_input_without_starting_a_process(self):
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.field_vars["environment_count"].set("not-a-number")
        # _start() shows a messagebox on invalid input; patch it out so the
        # test does not block on a real dialog.
        page.__class__.__module__  # sanity: page is the real class
        from unittest import mock

        with mock.patch("sandboxai.control_center_desktop.messagebox.showerror") as mocked:
            page._start()
        mocked.assert_called_once()
        assert page.process_id is None

    def test_settings_page_shows_the_real_project_and_output_roots(self):
        self.app.show_page("Settings")
        page = self.app.pages["Settings"]
        assert str(self.project_root) in page.project_root_label.cget("text")

    def test_close_shuts_down_background_worker(self):
        self.app._on_close()
        assert self.app.background._closed is True


if __name__ == "__main__":
    unittest.main()
