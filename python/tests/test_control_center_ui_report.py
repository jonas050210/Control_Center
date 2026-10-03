"""Tests for ``tools/control_center_ui_report.py``.

The tool itself needs a real window, which no headless machine can give it,
so what is pinned here is the part that must work regardless: the traversal
that finds the widgets, the arithmetic that decides whether a label is
clipped or a font is below the floor, and the rendering that turns findings
into something a human can act on. A report that silently renders nothing is
worse than no report, and that failure is invisible from the tool side.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

from optional_deps import HAS_TKINTER, TKINTER_REASON

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY_ROOT / "tools" / "control_center_ui_report.py"


def _load_module():
    """Load the tool by path - ``tools/`` is a folder of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("control_center_ui_report", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_tool = _load_module()


class _FakeWidget:
    """The smallest thing with ``winfo_children()`` - see :func:`_walk`."""

    def __init__(self, name, children=()):
        self.name = name
        self._children = list(children)

    def winfo_children(self):
        return list(self._children)


class WalkTests(unittest.TestCase):
    def test_every_descendant_is_visited(self):
        tree = _FakeWidget("root", [_FakeWidget("a", [_FakeWidget("a1")]), _FakeWidget("b")])
        found = {widget.name for widget in report_tool._walk(tree)}
        self.assertEqual(found, {"root", "a", "a1", "b"})

    def test_a_widget_without_children_is_not_an_error(self):
        # The traversal is duck-typed so it also works against the smoke
        # harness' fake tkinter, whose widgets are not tkinter.Misc.
        self.assertEqual([widget.name for widget in report_tool._walk(_FakeWidget("x"))], ["x"])

    def test_a_widget_that_raises_mid_walk_is_skipped(self):
        class _Broken(_FakeWidget):
            def winfo_children(self):
                raise RuntimeError("destroyed while walking")

        found = report_tool._walk(_FakeWidget("root", [_Broken("dead"), _FakeWidget("alive")]))
        self.assertIn("root", [widget.name for widget in found])


class LabelFindingTests(unittest.TestCase):
    def test_a_label_that_needs_more_room_than_it_has_is_clipped(self):
        finding = report_tool.LabelFinding(
            text="Environment count", font_px=12, needed_px=140, available_px=100
        )
        self.assertTrue(finding.clipped)
        self.assertFalse(finding.tiny)

    def test_a_wrapping_label_is_not_reported_as_clipped(self):
        # A wrapped label reflows instead of cutting the text, so reporting
        # it would be noise on every subtitle in the window.
        finding = report_tool.LabelFinding(
            text="A long subtitle that wraps",
            font_px=12,
            needed_px=400,
            available_px=200,
            wraps=True,
        )
        self.assertFalse(finding.clipped)

    def test_sub_pixel_rounding_is_not_a_finding(self):
        finding = report_tool.LabelFinding(
            text="Steps/s", font_px=12, needed_px=61, available_px=60
        )
        self.assertFalse(finding.clipped)

    def test_text_below_the_floor_is_a_finding(self):
        finding = report_tool.LabelFinding(text="tiny", font_px=9, needed_px=20, available_px=200)
        self.assertTrue(finding.tiny)
        self.assertFalse(finding.clipped)

    def test_an_unmeasured_font_is_not_reported_as_tiny(self):
        # font_px 0 means "could not measure", which must not read as "too
        # small" - the report would then accuse the window of something the
        # tool failed to check.
        finding = report_tool.LabelFinding(text="?", font_px=0, needed_px=0, available_px=10)
        self.assertFalse(finding.tiny)


class TableColumnTests(unittest.TestCase):
    def test_a_header_wider_than_its_column_is_reported(self):
        column = report_tool.TableColumn(
            name="frames_per_second", header="FPS/env", width_px=70, header_px=78
        )
        self.assertTrue(column.header_clipped)

    def test_a_column_with_room_to_spare_is_not_reported(self):
        column = report_tool.TableColumn(name="envs", header="Envs", width_px=60, header_px=40)
        self.assertFalse(column.header_clipped)


class RenderingTests(unittest.TestCase):
    def _page(self):
        return report_tool.PageReport(
            title="Benchmarks",
            cards=("Automatic benchmark", "Measurements"),
            buttons=(
                report_tool.ButtonFinding(text="Start Benchmark", state="normal"),
                report_tool.ButtonFinding(text="Cancel", state="disabled"),
            ),
            tables=(
                report_tool.TableFinding(
                    columns=(
                        report_tool.TableColumn(
                            name="fps", header="FPS/env", width_px=70, header_px=78
                        ),
                    ),
                    rows=12,
                ),
            ),
            labels=(
                report_tool.LabelFinding(
                    text="Measures this machine and applies the best stable configuration",
                    font_px=11,
                    needed_px=900,
                    available_px=400,
                ),
            ),
            refresh_ms=1.25,
            viewport_px=800,
            content_px=1240,
        )

    def test_a_report_names_what_it_found(self):
        page = self._page()
        markdown = report_tool.render_markdown(report_tool.Report(pages=(page,)))
        self.assertIn("### Benchmarks", markdown)
        self.assertIn("Disabled: Cancel", markdown)
        self.assertIn("clipped", markdown.lower())
        self.assertIn("`FPS/env` needs 78px", markdown)
        self.assertIn("scrolls (1240 px in 800 px)", markdown)

    def test_long_text_is_shortened_not_wrapped(self):
        page = report_tool.PageReport(
            title="Stats",
            labels=(
                report_tool.LabelFinding(
                    text="x " * 200, font_px=11, needed_px=4000, available_px=300
                ),
            ),
        )
        markdown = report_tool.render_markdown(report_tool.Report(pages=(page,)))
        self.assertTrue(
            all(len(line) < 200 for line in markdown.splitlines()),
            "one finding should stay on one line",
        )

    def test_a_clean_page_says_so(self):
        # "Nothing to report." is the evidence that the walk actually looked:
        # an empty section is indistinguishable from a broken one.
        markdown = report_tool.render_markdown(
            report_tool.Report(
                pages=(report_tool.PageReport(title="Settings", cards=("Presets",)),)
            )
        )
        self.assertIn("Nothing to report.", markdown)
        self.assertIn("0 finding(s)", markdown)

    def test_the_report_is_json_serialisable(self):
        page = self._page()
        data = report_tool.report_to_dict(report_tool.Report(pages=(page,)))
        encoded = json.dumps(data)
        self.assertEqual(json.loads(encoded)["findings"], page.findings)
        self.assertEqual(data["pages"][0]["title"], "Benchmarks")
        self.assertTrue(data["pages"][0]["tables"][0]["columns"][0]["header_clipped"])

    def test_findings_are_counted_across_pages(self):
        report = report_tool.Report(pages=(self._page(), self._page()))
        self.assertEqual(report.findings, 2 * self._page().findings)
        self.assertGreaterEqual(report.findings, 2)


class SlugTests(unittest.TestCase):
    def test_a_page_title_becomes_a_file_name(self):
        self.assertEqual(report_tool._slug("Runs / Checkpoints"), "runs-checkpoints")
        self.assertEqual(report_tool._slug("Dashboard"), "dashboard")
        self.assertEqual(report_tool._slug("///"), "page")


class NoTkinterTests(unittest.TestCase):
    """The tool is useless without Tk - it must say so instead of crashing."""

    @unittest.skipIf(HAS_TKINTER, "this machine has Tkinter")
    def test_main_explains_the_missing_package(self):
        import io
        from contextlib import redirect_stderr

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            code = report_tool.main([])
        self.assertEqual(code, 2, TKINTER_REASON)
        self.assertIn("python3-tk", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
