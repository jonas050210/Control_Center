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
            "Benchmarks",
            "Evaluations",
            "Runs / Checkpoints",
            "System / Telemetry",
            "Settings",
        }
        assert "Agents" not in titles, "the Agents page is replaced by Training"

    def test_every_page_builds_and_refreshes_without_raising(self):
        for page_class in PAGE_CLASSES:
            self.app.show_page(page_class.title)
            _drain_background(self.app)
        # No exception means every page's build()/refresh() survived an
        # empty (no runs yet) project directory - the most common state a
        # fresh user will actually see.

    def test_dense_tables_keep_every_column_reachable_without_permanent_scrollbars(self):
        """Tables keep right-hand data reachable through *overlay* bars.

        The old UI pinned two native scrollbars under every table (the
        loudest complaint about the layout). The contract now is: the data
        stays reachable, the bars are overlay widgets that only appear while
        something is actually scrollable, and a wide window stretches the
        columns instead of leaving dead space.
        """
        for title, attribute in (
            ("Training", "tree"),
            ("Benchmarks", "tree"),
            ("Evaluations", "checkpoint_tree"),
            ("Evaluations", "eval_tree"),
            ("Runs / Checkpoints", "tree"),
        ):
            self.app.show_page(title)
            page = self.app.pages[title]
            table = getattr(page, attribute)
            self.assertTrue(
                table.cget("xscrollcommand"),
                f"{title}'s {attribute} reports horizontal movement to a scrollbar",
            )
            scrollbar = table._horizontal_scrollbar
            self.assertEqual(scrollbar._orient, "horizontal")
            # An overlay bar is placed, never gridded into the layout, and it
            # hides itself while everything fits.
            self.assertNotEqual(scrollbar.winfo_manager(), "grid")
            self.assertFalse(scrollbar._overflow() and scrollbar.winfo_manager() == "")

    def test_every_widget_follows_the_application_theme_bus(self):
        """A widget built without a bus keeps the default palette forever.

        That is invisible without a display: the widget subscribes to the
        module's Corz default instead of the running window's bus, so it
        never repaints when the operator switches theme or accent. The same
        walk runs in the smoke harness; this is the real-Tk counterpart.
        """
        from sandboxai.control_center_ui import ThemeBus

        for page_class in PAGE_CLASSES:
            self.app.show_page(page_class.title)
            offenders = []
            pending = [self.app, *self.app.pages.values()]
            while pending:
                root = pending.pop()
                children = list(root.winfo_children())
                pending.extend(children)
                for widget in children:
                    for attribute in ("_bus", "bus"):
                        value = getattr(widget, attribute, None)
                        if isinstance(value, ThemeBus) and value is not self.app.bus:
                            offenders.append(f"{widget.winfo_class()}.{attribute}")
            self.assertEqual(offenders, [], f"{page_class.title}: {offenders}")

    def test_accent_and_preset_transfer_work_on_the_real_window(self):
        """Every page must repaint on the new accent, and presets must travel."""
        import json
        import tempfile

        from sandboxai.control_center_theme import normalize_accent

        self.app.show_page("Settings")
        self.assertTrue(self.app.set_accent("#F5A524"))
        self.assertEqual(self.app.palette.accent.lower(), "#f5a524")
        self.assertEqual(normalize_accent(self.app.prefs.accent), "#f5a524")
        # Every page repaints against the new palette without raising.
        for title in self.app.pages:
            self.app.show_page(title)
        self.assertFalse(self.app.set_accent("not a colour"))
        self.assertEqual(normalize_accent(self.app.prefs.accent), "#f5a524")
        self.assertTrue(self.app.set_accent(""))
        self.assertNotEqual(self.app.palette.accent.lower(), "#f5a524")

        self.app.show_page("Settings")
        settings = self.app.pages["Settings"]
        settings.preset_name_var.set("window test")
        settings._save_preset()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "preset.json"
            self.assertTrue(self.app.export_layout_preset("window test", target))
            document = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn("layout", document)
            self.assertIn("accent", document["appearance"])
            document["name"] = "window import"
            target.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(self.app.preset_name_for_import(target), "window import")
            self.assertEqual(self.app.import_layout_preset(target), "window import")
            self.assertIn("window import", self.app.list_layout_presets())
            # An existing name is not replaced by surprise.
            self.assertIsNone(self.app.import_layout_preset(target))
            self.assertEqual(self.app.import_layout_preset(target, overwrite=True), "window import")
        self.assertTrue(self.app.apply_layout_preset("window import"))
        for name in ("window test", "window import"):
            self.assertTrue(self.app.delete_layout_preset(name))

    def test_command_palette_opens_once_and_moves_with_the_arrow_keys(self):
        """Ctrl+K twice must not stack palettes, and Up/Down must pick a page."""
        self.app.open_command_palette()
        window = self.app._palette_window
        self.assertTrue(window.winfo_exists())
        self.app.open_command_palette()
        self.assertIs(self.app._palette_window, window)

        # Tk can only deliver generated key events once the toplevel is
        # mapped and owns the keyboard focus; that is also what the <Map>
        # handler guarantees in the real application.
        self.app.update()
        entry = next(child for child in window.winfo_children() if child.winfo_class() == "TEntry")
        listing = next(
            child for child in window.winfo_children() if child.winfo_class() == "Listbox"
        )
        self.assertGreater(listing.size(), 1)
        self.assertEqual(listing.curselection(), (0,))
        entry.focus_force()
        self.app.update()

        def press(widget, sequence: str) -> None:
            widget.event_generate(sequence, when="now")
            self.app.update()

        press(entry, "<Down>")
        self.assertEqual(listing.curselection(), (1,))
        # The key *release* must not undo the move: refresh() may only reset
        # the highlight when the filtered page list actually changed.
        press(entry, "<KeyRelease>")
        self.assertEqual(listing.curselection(), (1,))
        press(entry, "<Up>")
        self.assertEqual(listing.curselection(), (0,))
        # The arrows also work while the list itself holds the focus, and the
        # list's own binding keeps Tk's Listbox cursor step out of the way so
        # one keypress moves exactly one row (it used to move two).
        listing.focus_force()
        self.app.update()
        press(listing, "<Down>")
        self.assertEqual(listing.curselection(), (1,))
        # Escape closes it; the next Ctrl+K opens a fresh one, Ctrl+K closes
        # that one again and a page accelerator jumps straight to the page.
        press(window, "<Escape>")
        self.assertFalse(window.winfo_exists())
        self.app.open_command_palette()
        self.app.update()
        second = self.app._palette_window
        self.assertIsNot(second, window)
        press(second, "<Control-Key-k>")
        self.assertFalse(second.winfo_exists())
        self.app.open_command_palette()
        self.app.update()
        press(self.app._palette_window, "<Control-Key-4>")
        self.assertFalse(self.app._palette_window.winfo_exists())
        self.assertEqual(self.app._current.title, "Evaluations")

    def test_the_wheel_scrolls_a_page_from_anywhere_over_its_content(self):
        """The wheel must not need the pointer to be over the bare canvas.

        Tk delivers the wheel to the widget under the pointer, so a page that
        only binds it on its canvas does not move when the pointer is over a
        card's labels - the scrollbar then looks like the only way down. A
        widget that scrolls itself (a Text in a card) and anything outside
        the area must keep the page still.
        """
        from sandboxai.control_center_ui import ScrollArea

        host = tk.Frame(self.app)
        self.addCleanup(host.destroy)
        host.pack(fill="both", expand=True)
        area = ScrollArea(host, self.app.bus, scale_px=self.app.px)
        area.pack(fill="both", expand=True)
        for index in range(120):
            tk.Label(area.body, text=f"row {index}").pack()
        self.app.update()
        self.assertLess(area.canvas.yview()[1], 1.0, "the scratch page must overflow")

        def wheel_down(widget: object) -> bool:
            """One downwards wheel notch, spelled however this system does."""
            before = area.canvas.yview()[0]
            for sequence, options in (("<MouseWheel>", {"delta": -120}), ("<Button-5>", {})):
                with contextlib.suppress(tk.TclError):
                    widget.event_generate(sequence, **options)
                self.app.update()
                if area.canvas.yview()[0] > before:
                    return True
            return False

        rows = area.body.winfo_children()
        self.assertTrue(wheel_down(rows[5]), "the wheel over a card's label must scroll the page")

        text = tk.Text(area.body, height=5)
        text.pack()
        self.app.update()
        before = area.canvas.yview()[0]
        with contextlib.suppress(tk.TclError):
            text.event_generate("<MouseWheel>", delta=-120)
        self.app.update()
        self.assertEqual(area.canvas.yview()[0], before, "a Text keeps the wheel for itself")

        outside = tk.Label(host, text="outside")
        outside.pack()
        self.app.update()
        before = area.canvas.yview()[0]
        self.assertFalse(wheel_down(outside), "a wheel outside the area must not scroll it")
        self.assertEqual(area.canvas.yview()[0], before)

    def test_window_state_is_remembered_without_breaking_startup(self):
        """Maximized-ness is a preference, and saving it must never raise."""
        self.app.save_preferences()
        self.assertIsInstance(self.app.prefs.zoomed, bool)
        document = json.loads(self.app.prefs_store.path.read_text(encoding="utf-8"))
        self.assertIn("zoomed", document)
        self.assertIn("accent", document)

    def test_density_change_rebuilds_the_page_without_losing_the_view(self):
        """A density change re-lays the widgets out; the view must survive.

        The rebuild destroys and recreates the page's children, which is how
        the new paddings take effect. Only a real Tk build can verify the
        consequence: the log the operator was reading, its poll cursors
        (without them the next poll would duplicate everything) and the page
        staying usable afterwards.
        """
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.log_panel.apply_log(
            {"stdout": ["density marker line"], "stdout_cursor": 7, "stderr_cursor": 3}
        )

        self.app.set_density("compact")

        page = self.app.pages["Training"]  # the rebuild replaces the widgets
        self.assertIn("density marker line", page.log_panel.text.get("1.0", "end-1c"))
        self.assertEqual(page.log_panel.stdout_after, 7)
        self.assertEqual(page.log_panel.stderr_after, 3)
        # The page still works after being rebuilt underneath itself.
        page.refresh()
        _drain_background(self.app, attempts=3)
        self.app.set_density("comfort")

    def test_page_poll_gate_coalesces_slow_refreshes_without_losing_the_latest_one(
        self,
    ) -> None:
        """An overlapping poll has one latest-only retry, never an unbounded queue."""
        self.app.show_page("Dashboard")
        _drain_background(self.app)
        page = self.app.pages["Dashboard"]
        results: list[str] = []
        page.submit_poll(
            "test-poll", lambda: "first", lambda result, _error: results.append(result)
        )
        page.submit_poll(
            "test-poll", lambda: "superseded", lambda result, _error: results.append(result)
        )
        page.submit_poll(
            "test-poll", lambda: "latest", lambda result, _error: results.append(result)
        )
        _drain_background(self.app)
        self.assertEqual(results, ["first", "latest"])
        self.assertNotIn("test-poll", page._polls_in_flight)
        self.assertNotIn("test-poll", page._pending_polls)

        page.submit_poll("test-poll", lambda: "next", lambda result, _error: results.append(result))
        _drain_background(self.app)
        self.assertEqual(results, ["first", "latest", "next"])

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

    def test_launch_form_rejects_invalid_input_without_starting_an_agent(self):
        """The launch slot must go INVALID before anything can be launched."""
        from unittest import mock

        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.field_vars["environment_count"].set("not-a-number")
        _drain_background(self.app)
        # The inline launch slot reports the parse error and disables the
        # launch button; no dialog, no process.
        slot = page._launch_slot
        assert slot["state"] == "INVALID"
        assert slot["errors"]
        assert str(page.launch_button.cget("state")) == "disabled"
        with (
            mock.patch("sandboxai.control_center_pages.messagebox.showerror") as dialog,
            mock.patch.object(page.adapter.agents, "launch_training") as launch,
        ):
            page._launch()
            _drain_background(self.app)
        dialog.assert_not_called()
        launch.assert_not_called()

    def test_benchmark_page_runs_the_planned_workflow_and_applies_the_winner(self):
        """Auto is the default mode: Start plans a host-scaled sweep, hands
        only that plan to the pipeline, and applies the winning
        configuration automatically - there is no separate apply step."""
        from unittest import mock

        # Build the Training page first so the automatic apply step has a
        # live launch form to mirror the winning configuration into.
        self.app.show_page("Training")
        self.app.show_page("Benchmarks")
        page = self.app.pages["Benchmarks"]
        # No hidden budget/grid form: the mode selector drives the plan.
        self.assertFalse(hasattr(page, "pipeline_vars"))
        self.assertEqual(page.mode_control.get(), "auto")
        plan = page.current_plan()
        self.assertEqual(plan["errors"], [])
        self.assertTrue(plan["environments"])
        self.assertTrue(plan["workers"])
        recommendation = {
            "environment_count": 8,
            "env_workers": 2,
            "device": "cpu",
            "inference_device": "cpu",
            "expected_steps_per_second": 100.0,
            "basis": "validated_training_slice",
            "rationale": ["measured"],
            "warnings": [],
        }
        report = {
            "status": "completed",
            "elapsed_seconds": 1.0,
            "stages": [
                {"name": "discovery", "status": "completed"},
                {"name": "screening", "status": "completed", "configurations": []},
            ],
            "recommendation": recommendation,
        }
        applied = dict(recommendation, applied_utc="2026-01-01T00:00:00Z")
        with (
            mock.patch.object(page.adapter, "run_benchmark_pipeline", return_value=report) as run,
            mock.patch.object(
                page.adapter, "apply_recommended_configuration", return_value=applied
            ) as apply_call,
        ):
            page._start()
            _drain_background(self.app)
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        # Auto mode plans the host-scaled sweep itself and passes no
        # user-entered budget, grid or finalist parameters.
        self.assertEqual(kwargs["budget_mode"], plan["budget_mode"])
        self.assertIn(kwargs["budget_mode"], {"steps", "time"})
        self.assertEqual(kwargs["minutes"], plan["minutes"])
        self.assertEqual(sorted(kwargs["environment_counts"]), sorted(plan["environments"]))
        self.assertEqual(sorted(kwargs["worker_counts"]), sorted(plan["workers"]))
        self.assertEqual(
            set(kwargs)
            - {"budget_mode", "steps", "minutes", "environment_counts", "worker_counts"},
            {"cancel", "on_progress"},
        )
        apply_call.assert_called_once()
        self.assertIs(page._applied, True)
        # The Training launch form mirrors the applied topology.
        agents = self.app.pages["Training"]
        self.assertEqual(agents.field_vars["environment_count"].get(), "8")
        self.assertEqual(agents.field_vars["env_workers"].get(), "2")

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

    def test_log_autoscroll_resumes_after_the_operator_returns_to_the_bottom(self):
        from sandboxai.control_center_widgets import LogPanel

        panel = LogPanel(self.app, max_lines=300)
        panel.pack(fill="both", expand=True)
        self.app.update()
        panel.apply_log({"stdout": [f"line {index}" for index in range(120)]})
        panel._on_manual_scroll(None)
        assert panel._user_scrolled_up is True

        # The checkbox remains enabled, so returning to the newest output
        # should resume live-follow rather than leave it silently stuck.
        panel._on_text_scroll("0.0", "1.0")
        assert panel._user_scrolled_up is False
        panel._on_autoscroll_toggled()
        self.app.update()
        # Tk's terminal newline may make the final fraction just shy of 1.0;
        # it must nevertheless be at the bottom rather than leave the reader
        # at its prior historical position.
        self.assertGreater(panel.text.yview()[1], 0.98)

    def test_log_scrollbar_and_keyboard_reading_pause_follow_without_hiding_wide_output(self):
        """The log must not snap a reader away from a traceback they are inspecting.

        Wheel events already had a regression test, but actual desktop readers
        also drag the visible scrollbar or use Home/Page Up. This pins both
        paths and verifies the horizontal scrollbar that makes unwrapped
        command lines/tracebacks reachable.
        """
        from sandboxai.control_center_widgets import LogPanel

        panel = LogPanel(self.app, max_lines=300)
        panel.pack(fill="both", expand=True)
        self.app.update()
        panel.apply_log(
            {
                "stdout": [
                    f"line {index}: " + ("very-wide-traceback-segment " * 24)
                    for index in range(120)
                ]
            }
        )
        self.app.update()
        # Wrapping is on by default, which is what removes the horizontal bar
        # from a log full of wide tracebacks in the first place.
        self.assertTrue(panel._wrap.get())
        self.assertEqual(str(panel.text.cget("wrap")), "word")
        # Turning wrapping off keeps the wide lines intact and reveals the
        # overlay bar (an overlay, not a permanently gridded native bar).
        panel._wrap.set(False)
        panel._on_wrap_toggled()
        self.app.update()
        self.assertEqual(str(panel.text.cget("wrap")), "none")
        self.assertEqual(panel._xscroll._orient, "horizontal")
        self.assertNotEqual(panel._xscroll.winfo_manager(), "grid")
        self.assertTrue(
            panel.text.cget("xscrollcommand"),
            "unwrapped process output reports horizontal movement to its scrollbar",
        )

        # This is the command path used by scrollbar arrows/track dragging,
        # not a synthetic wheel event.
        panel._scroll_text_y("moveto", "0.0")
        self.assertTrue(panel._user_scrolled_up)
        panel.apply_log({"stdout": ["a newer line must not steal the reading position"]})
        self.app.update()
        self.assertLess(
            panel.text.yview()[1],
            0.98,
            "new process output preserves a scrollbar reader's historical position",
        )

        # Keyboard navigation uses the same delayed follow-state check. A
        # reader returning to the end resumes live-follow on the next update.
        panel._on_manual_scroll()
        panel._scroll_text_y("moveto", "1.0")
        self.assertFalse(panel._user_scrolled_up)
        panel.apply_log({"stdout": ["follow resumes at the newest line"]})
        self.app.update()
        self.assertGreater(panel.text.yview()[1], 0.98)

    def test_evaluation_and_run_details_reject_late_selections(self):
        """Slow disk reads must not overwrite the newer selected detail pane."""
        self.app.show_page("Evaluations")
        evaluation = self.app.pages["Evaluations"]
        evaluation._selected_evaluation_paths = ("evaluation-new.json",)
        evaluation._evaluation_detail_generation = 4
        evaluation._on_details(("evaluation-old.json",), 3, [{}], None)
        self.assertEqual(evaluation.detail_text.get("1.0", "end-1c"), "")

        self.app.show_page("Runs / Checkpoints")
        runs = self.app.pages["Runs / Checkpoints"]
        runs._selected_run_dir = "run-new"
        runs._run_detail_generation = 4
        runs._on_detail("run-old", 3, {}, None)
        self.assertEqual(runs.detail_text.get("1.0", "end-1c"), "")

    def test_training_page_rejects_late_logs_and_binds_actions_to_the_clicked_run(self):
        """Background output/commands must stay attached to the selected agent.

        A slow file read for Agent A may finish after the operator selected
        Agent B. Rendering A's output under B would make the telemetry
        misleading; resolving an action lambda after the selection changed
        could act on B instead of the clicked A.
        """
        from unittest import mock

        self.app.show_page("Training")
        _drain_background(self.app)
        page = self.app.pages["Training"]
        page._last_views = [
            {"agent_id": "agent-a", "process_id": "proc-a"},
            {"agent_id": "agent-b", "process_id": "proc-b"},
        ]
        page._selected_agent_id = "agent-b"
        page._log_selection_generation = 2
        page._on_log("proc-a", 1, {"stdout": ["stale agent-a output"]}, None)
        self.assertNotIn("stale agent-a output", page.log_panel.text.get("1.0", "end"))
        page._on_log("proc-b", 2, {"stdout": ["current agent-b output"]}, None)
        self.assertIn("current agent-b output", page.log_panel.text.get("1.0", "end"))

        # A deselection clears the now-unattributed output and disables
        # commands rather than leaving an old process apparently actionable.
        page._clear_selection()
        self.assertEqual(page.log_panel.text.get("1.0", "end-1c"), "")
        self.assertEqual(str(page.stop_button.cget("state")), "disabled")
        self.assertEqual(str(page.pause_button.cget("state")), "disabled")
        self.assertEqual(str(page.restart_button.cget("state")), "disabled")

        # The queued action captures the agent id that was selected when the
        # operator pressed the button, rather than resolving the mutable
        # selection later on a worker thread.
        page._selected_agent_id = "agent-a"
        with mock.patch.object(page.adapter.agents, "stop") as stop:
            page._stop()
            page._selected_agent_id = "agent-b"
            _drain_background(self.app)
        stop.assert_called_once_with("agent-a")

    def test_tooltip_cancels_its_delayed_callback_when_a_page_widget_is_destroyed(self):
        from sandboxai.control_center_widgets import ToolTip

        button = tk.Button(self.app, text="temporary")
        button.pack()
        tip = ToolTip(button, "temporary help")
        tip._schedule()
        button.destroy()
        self.app.update()
        assert tip._after_id is None
        assert tip._window is None


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

    class _TimerRoot(_StubRoot):
        """Timer-aware root used to prove manual pumps cannot multiply work."""

        def __init__(self) -> None:
            super().__init__()
            self.cancelled: list[str] = []

        def after(self, _delay_ms: int, _callback) -> str:
            self.scheduled += 1
            return f"timer-{self.scheduled}"

        def after_cancel(self, timer_id: str) -> None:
            self.cancelled.append(timer_id)

    def _runner(self):
        from sandboxai.control_center_widgets import BackgroundRunner

        runner = BackgroundRunner(self._StubRoot())  # type: ignore[arg-type]
        # Unconditionally, not just on the happy path: an unclosed runner
        # leaves automatic collection off for every test that follows.
        self.addCleanup(runner.close)
        return runner

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

    def test_close_gives_up_on_a_wedged_worker(self) -> None:
        """A stuck adapter call must not be able to hang the window shut.

        Waiting is the right default, but unbounded waiting turns one
        wedged process listing into a window that will not close. The
        GC suspension is the guard that still holds in that case.
        """
        import threading

        from sandboxai.control_center_widgets import BackgroundRunner

        runner = self._runner()
        runner.CLOSE_TIMEOUT_S = 0.2  # type: ignore[misc]
        release = threading.Event()
        self.addCleanup(release.set)
        runner.submit(lambda: release.wait(30), lambda _result, _error: None)
        time.sleep(0.1)

        started = time.monotonic()
        runner.close()
        elapsed = time.monotonic() - started

        self.assertLess(
            elapsed, 5.0, f"close() waited {elapsed:.1f}s on a worker that never finishes"
        )
        self.assertTrue(BackgroundRunner.CLOSE_TIMEOUT_S >= 1.0, "the real timeout stays generous")

    def test_close_leaves_no_live_worker_threads(self) -> None:
        """No worker may still be around once the pool has wound down.

        Deliberately not asserted the instant ``close()`` returns. What
        ``close()`` guarantees is that the submitted *work* has finished
        or been abandoned; the pool's threads then exit on their own,
        and that last step is not instantaneous. Asserting on the
        instant made this pass locally and fail in CI, which is a flaky
        test rather than a real guarantee.
        """
        import threading

        def live_workers() -> list[str]:
            return [t.name for t in threading.enumerate() if "control-center-bg" in t.name]

        runner = self._runner()
        for _ in range(3):
            runner.submit(lambda: time.sleep(0.05), lambda _result, _error: None)
        runner.close()

        deadline = time.monotonic() + 10.0
        while live_workers() and time.monotonic() < deadline:
            time.sleep(0.05)

        alive = live_workers()
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
        self.assertFalse(
            runner.submit(lambda: calls.append("ran"), lambda _result, _error: None),
            "callers that own in-flight UI state must be able to release it on shutdown",
        )
        time.sleep(0.1)
        self.assertEqual(calls, [])

    def test_manual_pumps_keep_one_pending_timer_and_close_cancels_it(self) -> None:
        from sandboxai.control_center_widgets import BackgroundRunner

        root = self._TimerRoot()
        runner = BackgroundRunner(root)  # type: ignore[arg-type]
        self.addCleanup(runner.close)
        self.assertEqual(root.scheduled, 1, "runner starts with one future pump")
        for _ in range(8):
            runner._pump()
        self.assertEqual(
            root.scheduled,
            1,
            "a test/manual drain does not build an unbounded Tk callback chain",
        )
        runner.close()
        self.assertEqual(root.cancelled, ["timer-1"], "close removes the pending Tk callback")


@unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
class BackgroundRunnerOwnsGarbageCollectionTests(unittest.TestCase):
    """Cycle collection must happen on the Tk thread, not on a worker.

    Second half of the same Windows crash. Waiting for workers at
    shutdown was necessary but not sufficient: the fatal collection
    happened *while* a worker was running, mid-``pathlib`` walk, because
    CPython collects in whichever thread trips the allocation threshold.
    Discarded widget trees are cyclic, so a worker that trips it runs
    ``tkinter.Variable.__del__`` against the Tcl interpreter it does not
    own, and the process dies without a traceback.

    The runner therefore suspends automatic collection while it is alive
    and drives the collector from ``_pump``, which is on the Tk thread.
    """

    class _StubRoot:
        def after(self, _delay_ms: int, _callback) -> str:
            return "timer"

    def _runner(self, poll_ms: int = 100):
        from sandboxai.control_center_widgets import BackgroundRunner

        runner = BackgroundRunner(self._StubRoot(), poll_ms=poll_ms)  # type: ignore[arg-type]
        self.addCleanup(runner.close)
        return runner

    def test_automatic_collection_is_suspended_while_a_runner_is_alive(self) -> None:
        import gc

        self.assertTrue(gc.isenabled(), "precondition: the suite runs with gc on")
        runner = self._runner()
        self.assertFalse(
            gc.isenabled(),
            "automatic collection is still on, so a worker thread can finalise Tk objects",
        )
        runner.close()
        self.assertTrue(gc.isenabled(), "close() did not hand the collector back")

    def test_the_last_runner_out_restores_the_collector(self) -> None:
        """set_output_root builds a second runner before dropping the first."""
        import gc

        first = self._runner()
        second = self._runner()
        first.close()
        self.assertFalse(gc.isenabled(), "the first close() freed a collector it did not own")
        second.close()
        self.assertTrue(gc.isenabled())

    def test_closing_twice_does_not_unbalance_the_counter(self) -> None:
        import gc

        runner = self._runner()
        runner.close()
        runner.close()
        other = self._runner()
        self.assertFalse(
            gc.isenabled(),
            "a double close() decremented the counter twice and left a live runner "
            "running with automatic collection on",
        )
        other.close()
        self.assertTrue(gc.isenabled())

    def test_pump_collects_cycles_on_the_calling_thread(self) -> None:
        runner = self._runner(poll_ms=1000)  # collect on every pump

        collected: list[str] = []

        class Cyclic:
            def __del__(self) -> None:
                collected.append("finalised")

        node = Cyclic()
        node.self_reference = node  # type: ignore[attr-defined]
        del node

        self.assertEqual(collected, [], "a cycle should survive until something collects it")
        runner._pump()
        self.assertEqual(
            collected,
            ["finalised"],
            "_pump did not collect, so cycles are left for whichever thread trips the "
            "allocation threshold - which is the bug",
        )


@unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
class TrainingAndBenchmarkPlanTests(unittest.TestCase):
    """The reworked launch path: Time|Steps on Training, Auto|Push|Custom on Benchmarks."""

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

    def test_training_page_is_reachable_with_every_documented_control(self):
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        self.assertEqual(page.budget_mode.get(), "steps")
        self.assertTrue(page.launch_button.winfo_exists())
        self.assertIn("Time", page.budget_mode.choices())
        self.assertIn("Steps", page.budget_mode.choices())

    def test_time_budget_switches_the_plan_to_minutes_and_validates_the_entry(self):
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.budget_mode.set("time")
        page._on_budget_mode("time")
        budget = page.current_budget()
        self.assertEqual(budget["mode"], "time")
        self.assertEqual(budget["errors"], [])
        self.assertIsNotNone(budget["minutes"])
        self.assertIn("cooperative stop", budget["budget_line"])

        page.budget_minutes_var.set("nonsense")
        budget = page.current_budget()
        self.assertTrue(budget["errors"])
        self.assertEqual(budget["mode"], "time")

        page.budget_minutes_var.set("1441")
        self.assertTrue(page.current_budget()["errors"], "an absurd budget is refused")

    def test_steps_preset_button_fills_the_step_field(self):
        self.app.show_page("Training")
        page = self.app.pages["Training"]
        page.budget_mode.set("steps")
        page._on_budget_mode("steps")
        page.field_vars["total_training_steps"].set("1")
        page._apply_step_preset("25000")
        self.assertEqual(page.field_vars["total_training_steps"].get(), "25000")

    def test_benchmark_custom_mode_requires_valid_lists(self):
        self.app.show_page("Benchmarks")
        page = self.app.pages["Benchmarks"]
        page.mode_control.set("custom")
        page._on_mode_changed("custom")
        self.assertEqual(page.current_plan()["errors"], [])

        page.custom_env_var.set("16, 32")
        page.custom_worker_var.set("nonsense")
        plan = page.current_plan()
        self.assertTrue(plan["errors"])
        self.assertEqual(str(page.run_button.cget("state")), "disabled")

        page.custom_worker_var.set("8, 64")
        plan = page.current_plan()
        # worker > envs is dropped with a warning rather than erroring out.
        self.assertEqual(plan["errors"], [])
        self.assertEqual(plan["expected_configurations"], 2)
        self.assertTrue(plan["warnings"])
        self.assertNotEqual(str(page.run_button.cget("state")), "disabled")

    def test_benchmark_auto_plan_covers_the_whole_host(self):
        self.app.show_page("Benchmarks")
        page = self.app.pages["Benchmarks"]
        page.mode_control.set("auto")
        page._on_mode_changed("auto")
        plan = page.current_plan()
        self.assertEqual(plan["mode"], "auto")
        self.assertEqual(plan["errors"], [])
        if not plan["environments"]:
            self.skipTest("no environment ladder is available on this host")
        self.assertEqual(plan["environments"], sorted(plan["environments"]))
        self.assertLessEqual(max(plan["environments"]), 128 * 4)
