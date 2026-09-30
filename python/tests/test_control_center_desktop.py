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

import contextlib
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
        with contextlib.suppress(tk.TclError):
            self.app._on_close()

    def test_every_required_page_is_registered(self):
        titles = {page_class.title for page_class in PAGE_CLASSES}
        assert titles == {
            "Dashboard",
            "Training",
            "Agents",
            "Benchmarks",
            "Evaluations",
            "Runs / Checkpoints",
            "System / Telemetry",
            "Settings",
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
        TrainingConfig(
            total_training_steps=1000,
            output_root=str(self.project_root / "training"),
            run_id="run-a",
        ).save(run_dir / "config.json")
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "state": "Running",
                    "timesteps": 250,
                    "total_training_steps": 1000,
                    "updated_at": time.time(),
                }
            )
        )
        self.app.show_page("Dashboard")
        _drain_background(self.app)
        page = self.app.pages["Dashboard"]
        # SandboxAIAdapter resolves project_root/output_root (see
        # adapter.py), so discovered run directories come back canonicalized
        # too; run_dir here is built from the raw self.project_root, which on
        # Windows can be an 8.3 short name (RUNNER~1) that resolves to a
        # different-looking but identical directory (runneradmin). Compare
        # against the resolved form the adapter actually reports.
        assert page._last_run_dir == str(run_dir.resolve())

    def test_training_form_rejects_invalid_input_without_starting_a_process(self):
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.field_vars["environment_count"].set("not-a-number")
        # _start() shows a messagebox on invalid input; patch it out so the
        # test does not block on a real dialog.
        from unittest import mock

        with mock.patch("sandboxai.control_center_desktop.messagebox.showerror") as mocked:
            page._start()
        mocked.assert_called_once()
        assert page.process_id is None

    def test_settings_page_shows_the_real_project_and_output_roots(self):
        self.app.show_page("Settings")
        page = self.app.pages["Settings"]
        # SandboxAIAdapter.__init__ stores project_root.resolve() (a
        # deliberate canonicalization). self.project_root here is the raw,
        # unresolved tempdir path; on Windows temp paths can come back in
        # 8.3 short form (RUNNER~1) while .resolve() reports the real long
        # name (runneradmin) for the identical directory, so compare the
        # resolved form the label actually shows.
        assert str(self.project_root.resolve()) in page.project_root_label.cget("text")

    def test_close_shuts_down_background_worker(self):
        self.app._on_close()
        assert self.app.background._closed is True


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
class BackgroundRunnerShutdownTests(unittest.TestCase):
    """``close()`` must not return while a worker is still running.

    Regression for a Windows-only CI crash: the suite died with
    ``Windows fatal exception: code 0x80000003`` - no traceback, no
    failing assertion, the interpreter simply stopped - while a worker
    thread was garbage-collecting inside ``discover_run_directories``
    and the main thread was reconfiguring a Tk widget.

    The cause was ``shutdown(wait=False)``: ``_on_close`` closed the
    runner and called ``destroy()`` immediately, so a worker still in
    flight was left holding the closure that submitted it, and those
    closures reach Tk widgets. The worker then dropped the last
    reference and ran a Tk finaliser off the interpreter's own thread.

    No real Tk window is needed to pin this, only something with
    ``after``; the crash was about thread ownership, not about widgets.
    """

    class _StubRoot:
        """Stands in for the Tk root: records timers, never fires them."""

        def __init__(self) -> None:
            self.scheduled = 0

        def after(self, _delay_ms: int, _callback) -> str:
            self.scheduled += 1
            return "timer"

    def _runner(self):
        from sandboxai.control_center_widgets import BackgroundRunner

        return BackgroundRunner(self._StubRoot())  # type: ignore[arg-type]

    def test_close_waits_for_work_that_already_started(self) -> None:
        import threading

        runner = self._runner()
        started = threading.Event()
        finished = threading.Event()

        def slow() -> str:
            started.set()
            time.sleep(0.3)
            finished.set()
            return "done"

        runner.submit(slow, lambda _result, _error: None)
        self.assertTrue(started.wait(5), "the worker never started")
        runner.close()
        self.assertTrue(
            finished.is_set(),
            "close() returned while a worker was still running - that worker can "
            "finalise Tk objects off the Tk thread, which kills the process on Windows",
        )

    def test_close_leaves_no_live_worker_threads(self) -> None:
        import threading

        runner = self._runner()
        for _ in range(3):
            runner.submit(lambda: time.sleep(0.05), lambda _result, _error: None)
        runner.close()
        alive = [t.name for t in threading.enumerate() if "control-center-bg" in t.name]
        self.assertEqual(alive, [], f"worker threads outlived close(): {alive}")

    def test_close_drops_undelivered_results(self) -> None:
        """Their callbacks hold widgets; they must not sit in the queue."""
        runner = self._runner()
        runner.submit(lambda: "value", lambda _result, _error: None)
        runner.close()
        self.assertTrue(runner._queue.empty())

    def test_submit_after_close_is_ignored(self) -> None:
        runner = self._runner()
        runner.close()
        calls: list[str] = []
        runner.submit(lambda: calls.append("ran"), lambda _result, _error: None)
        time.sleep(0.1)
        self.assertEqual(calls, [])
