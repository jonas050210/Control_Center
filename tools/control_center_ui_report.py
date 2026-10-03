"""Measure and photograph the real Control Center window on this machine.

The headless smoke harness proves that the window *builds*; it proves nothing
about how it looks. This script is the other half: it opens the real Tk
window, walks every page and writes down what a human would otherwise have to
describe in words - clipped labels, text under the 11 px floor, table headers
wider than their column, buttons that are disabled, dead theme listeners after
three density changes, and what a refresh costs per page. Where Pillow can
grab the screen it also saves one PNG per page.

It is a measuring instrument, not a gate and not a test: it reports what it
finds and always says what it could not do (no Tkinter, no Pillow, no way to
screenshot a headless display) instead of inventing a number.

    python3 tools/control_center_ui_report.py
    python3 tools/control_center_ui_report.py --out .sandboxai/ui_report --settle 1.5
    python3 tools/control_center_ui_report.py --no-screenshots --pages Dashboard,Benchmarks

Exit codes: ``0`` report written, ``2`` the prerequisites are missing (the
message names the package to install), ``1`` the window itself failed to
build - which is a real finding, and the traceback is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

#: The project's own floor. Anything below this was readable on the
#: developer's display and is not readable on a 125 % laptop panel.
MIN_FONT_PX = 11

#: Sub-pixel rounding makes an exact comparison of "text width" against
#: "widget width" useless; two pixels is where a label starts to look cut off.
CLIP_TOLERANCE_PX = 2

DEFAULT_OUT = ".sandboxai/ui_report"

#: Densities cycled for the listener check. A density change destroys and
#: rebuilds every widget, which is exactly when dead repaint callbacks used
#: to accumulate.
DENSITY_SWEEP = ("compact", "ultra", "comfort")


# ---------------------------------------------------------------------------
# Findings (plain data - no Tk anywhere in this section, so it is testable
# on a machine without python3-tk)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelFinding:
    """One label, with what it needs against what it got."""

    text: str
    font_px: int
    needed_px: int
    available_px: int
    wraps: bool = False

    @property
    def clipped(self) -> bool:
        """Text wider than its widget, and not allowed to wrap."""
        return not self.wraps and self.needed_px > self.available_px + CLIP_TOLERANCE_PX

    @property
    def tiny(self) -> bool:
        return 0 < self.font_px < MIN_FONT_PX


@dataclass(frozen=True)
class ButtonFinding:
    """One button: what it says and whether it can be pressed."""

    text: str
    state: str

    @property
    def enabled(self) -> bool:
        return "disabled" not in self.state


@dataclass(frozen=True)
class TableColumn:
    name: str
    header: str
    width_px: int
    header_px: int

    @property
    def header_clipped(self) -> bool:
        return self.header_px > self.width_px + CLIP_TOLERANCE_PX


@dataclass(frozen=True)
class TableFinding:
    columns: tuple[TableColumn, ...] = ()
    rows: int = 0
    empty_text: str = ""

    @property
    def clipped_headers(self) -> tuple[TableColumn, ...]:
        return tuple(column for column in self.columns if column.header_clipped)


@dataclass(frozen=True)
class PageReport:
    """Everything one page had to say about itself."""

    title: str
    cards: tuple[str, ...] = ()
    buttons: tuple[ButtonFinding, ...] = ()
    tables: tuple[TableFinding, ...] = ()
    labels: tuple[LabelFinding, ...] = ()
    refresh_ms: float = 0.0
    viewport_px: int = 0
    content_px: int = 0
    notes: tuple[str, ...] = ()
    screenshot: str = ""

    @property
    def disabled_buttons(self) -> tuple[ButtonFinding, ...]:
        return tuple(button for button in self.buttons if not button.enabled)

    @property
    def clipped_labels(self) -> tuple[LabelFinding, ...]:
        return tuple(label for label in self.labels if label.clipped)

    @property
    def tiny_labels(self) -> tuple[LabelFinding, ...]:
        return tuple(label for label in self.labels if label.tiny)

    @property
    def scrolls(self) -> bool:
        return self.content_px > self.viewport_px + 4

    @property
    def findings(self) -> int:
        """How many things on this page are worth a look."""
        return (
            len(self.clipped_labels)
            + len(self.tiny_labels)
            + sum(len(table.clipped_headers) for table in self.tables)
        )


@dataclass(frozen=True)
class Report:
    host: dict[str, str] = field(default_factory=dict)
    window: dict[str, object] = field(default_factory=dict)
    pages: tuple[PageReport, ...] = ()
    checks: dict[str, object] = field(default_factory=dict)
    screenshots: tuple[str, ...] = ()

    @property
    def findings(self) -> int:
        return sum(page.findings for page in self.pages)


def _quote(text: str, limit: int = 72) -> str:
    """One line of a finding, shortened rather than wrapped."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _page_lines(page: PageReport) -> list[str]:
    """The bullet list for one page, without the heading."""
    lines: list[str] = []
    if page.cards:
        lines.append("- Cards: " + ", ".join(page.cards))
    else:
        lines.append("- Cards: none")
    lines.append(
        f"- Buttons: {len(page.buttons)}"
        f" ({len(page.disabled_buttons)} disabled)"
        f" · refresh {page.refresh_ms:.1f} ms"
        + (f" · scrolls ({page.content_px} px in {page.viewport_px} px)" if page.scrolls else "")
    )
    disabled = [button.text for button in page.disabled_buttons]
    if disabled:
        lines.append(f"- Disabled: {', '.join(disabled)}")
    for table in page.tables:
        widths = ", ".join(f"{column.header} {column.width_px}px" for column in table.columns)
        lines.append(f"- Table: {table.rows} row(s) · {widths or 'no columns'}")
        for column in table.clipped_headers:
            lines.append(
                f"  - header `{column.header}` needs {column.header_px}px, "
                f"column is {column.width_px}px"
            )
    for label in page.clipped_labels:
        lines.append(
            f'- Clipped: "{_quote(label.text)}" needs {label.needed_px}px, '
            f"has {label.available_px}px ({label.font_px}px font)"
        )
    for label in page.tiny_labels:
        lines.append(
            f'- Small text: "{_quote(label.text)}" at {label.font_px}px (floor {MIN_FONT_PX}px)'
        )
    for note in page.notes:
        lines.append(f"- Note: {note}")
    if page.screenshot:
        lines.append(f"- Screenshot: `{page.screenshot}`")
    if not page.findings and not page.notes:
        lines.append("- Nothing to report.")
    return lines


def render_markdown(report: Report) -> str:
    """Render the report as Markdown.

    Kept out of the collection code so it can be tested on a machine that
    has no Tk: a report that cannot be rendered is a report nobody reads.
    """
    lines: list[str] = ["# Control Center UI report", ""]
    if report.host:
        lines.append(
            "**Host** — " + ", ".join(f"{key}: {value}" for key, value in report.host.items()) + "."
        )
        lines.append("")
    if report.window:
        lines.append(
            "**Window** — "
            + ", ".join(f"{key}: {value}" for key, value in report.window.items())
            + "."
        )
        lines.append("")

    if report.checks:
        lines.append("## Window checks")
        lines.append("")
        for key, value in report.checks.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.append(f"## Pages ({report.findings} finding(s))")
    lines.append("")
    for page in report.pages:
        lines.append(f"### {page.title}")
        lines.append("")
        lines.extend(_page_lines(page))
        lines.append("")

    if report.screenshots:
        lines.append("## Screenshots")
        lines.append("")
        for path in report.screenshots:
            lines.append(f"- `{path}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def report_to_dict(report: Report) -> dict[str, object]:
    """The same report as JSON-serialisable data.

    The Markdown is for reading; this is for pasting into an issue or
    comparing two machines without retyping anything.
    """

    def page_to_dict(page: PageReport) -> dict[str, object]:
        return {
            "title": page.title,
            "cards": list(page.cards),
            "buttons": [{"text": button.text, "state": button.state} for button in page.buttons],
            "tables": [
                {
                    "rows": table.rows,
                    "empty_text": table.empty_text,
                    "columns": [
                        {
                            "name": column.name,
                            "header": column.header,
                            "width_px": column.width_px,
                            "header_px": column.header_px,
                            "header_clipped": column.header_clipped,
                        }
                        for column in table.columns
                    ],
                }
                for table in page.tables
            ],
            "labels": [
                {
                    "text": label.text,
                    "font_px": label.font_px,
                    "needed_px": label.needed_px,
                    "available_px": label.available_px,
                    "wraps": label.wraps,
                    "clipped": label.clipped,
                    "tiny": label.tiny,
                }
                for label in page.labels
            ],
            "refresh_ms": round(page.refresh_ms, 3),
            "viewport_px": page.viewport_px,
            "content_px": page.content_px,
            "notes": list(page.notes),
            "screenshot": page.screenshot,
            "findings": page.findings,
        }

    return {
        "host": dict(report.host),
        "window": dict(report.window),
        "checks": dict(report.checks),
        "findings": report.findings,
        "pages": [page_to_dict(page) for page in report.pages],
        "screenshots": list(report.screenshots),
    }


# ---------------------------------------------------------------------------
# Collection (needs a real Tk)
# ---------------------------------------------------------------------------


def _walk(widget: object) -> list[object]:
    """Every descendant of ``widget``, depth first.

    Duck-typed on purpose: anything with ``winfo_children()`` can be walked,
    which is what lets this run against the smoke harness' fake Tk (and what
    keeps it working if a page ever hosts a widget from another toolkit).
    """
    found: list[object] = []
    stack: list[object] = [widget]
    while stack:
        current = stack.pop()
        found.append(current)
        children = getattr(current, "winfo_children", None)
        if children is None:
            continue
        try:
            found.extend(_walk_below(list(children())))
        except Exception:  # pragma: no cover - a destroyed child mid-walk
            continue
    return found


def _walk_below(children: list[object]) -> list[object]:
    """Flatten a child list, skipping anything already destroyed."""
    flat: list[object] = []
    for child in children:
        flat.append(child)
        grandchildren = getattr(child, "winfo_children", None)
        if grandchildren is None:
            continue
        try:
            flat.extend(_walk_below(list(grandchildren())))
        except Exception:  # pragma: no cover
            continue
    return flat


def _font_of(widget: object, root: object) -> tuple[int, object]:
    """Resolve a widget's font to ``(pixels, the font object to measure with)``."""
    from tkinter import font as tkfont

    try:
        value = widget.cget("font")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - a widget without -font
        value = ""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[1]), tkfont.Font(root=root, font=tuple(value))
        except Exception:  # pragma: no cover - a malformed font spec
            pass
    try:
        resolved = tkfont.Font(root=root, font=value or "TkDefaultFont")
        return int(resolved.actual().get("size") or 0), resolved
    except Exception:  # pragma: no cover - no font support in this build
        return 0, None


def _text_px(font: object, text: str) -> int:
    if font is None:
        return 0
    try:
        return int(font.measure(text))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return 0


def _state_of(widget: object) -> str:
    try:
        return str(widget.cget("state"))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - a widget without -state
        return "normal"


def _slug(title: str) -> str:
    """A page title as a file name: "Runs / Checkpoints" -> "runs-checkpoints"."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "page"


def _collect_label(widget: object, root: object) -> LabelFinding | None:
    """One label's measurement, or ``None`` if it cannot be measured honestly."""
    try:
        text = str(widget.cget("text"))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - a widget without -text
        return None
    if not text.strip():
        return None
    font_px, font = _font_of(widget, root)
    available = int(widget.winfo_width())  # type: ignore[attr-defined]
    if available <= 1:
        # Not mapped yet (or packed into a scroll area that has not laid
        # out): measuring it now would report a clipping that is not there.
        return None
    try:
        wraplength = int(float(widget.cget("wraplength") or 0))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        wraplength = 0
    return LabelFinding(
        text=text,
        font_px=font_px,
        needed_px=_text_px(font, text),
        available_px=available,
        wraps=wraplength > 1,
    )


def _collect_table(widget: object, root: object) -> TableFinding:
    """One table's columns, with what each header needs."""
    _font_px, font = _font_of(widget, root)
    columns: list[TableColumn] = []
    for name in widget.cget("columns") or ():  # type: ignore[attr-defined]
        name = str(name)
        if name == "#0":
            continue
        try:
            header = str(widget.heading(name, "text"))  # type: ignore[attr-defined]
            width = int(widget.column(name, "width"))  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - a heading-less column
            continue
        columns.append(
            TableColumn(
                name=name,
                header=header,
                width_px=width,
                # +16: the sort arrow gutter Tk keeps beside a heading.
                header_px=_text_px(font, header) + 16,
            )
        )
    try:
        rows = len(widget.get_children())  # type: ignore[attr-defined]
        empty_text = str(widget.cget("empty_text") or "")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        rows, empty_text = 0, ""
    return TableFinding(columns=tuple(columns), rows=rows, empty_text=empty_text)


def collect_page(app: object, page: object, root: object) -> PageReport:
    """Walk one page and write down what it shows."""
    cards: list[str] = []
    buttons: list[ButtonFinding] = []
    labels: list[LabelFinding] = []
    tables: list[TableFinding] = []

    for widget in _walk(page):
        kind = str(widget.winfo_class())
        title = getattr(widget, "_title", None)
        if isinstance(title, str) and title:
            cards.append(title)
        if kind in {"TButton", "Button"}:
            text = ""
            try:
                text = str(widget.cget("text"))  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                text = ""
            buttons.append(ButtonFinding(text=text, state=_state_of(widget)))
        elif kind in {"TLabel", "Label"}:
            finding = _collect_label(widget, root)
            if finding is not None:
                labels.append(finding)
        elif kind == "Treeview":
            tables.append(_collect_table(widget, root))

    try:
        start = time.perf_counter()
        page.refresh()  # type: ignore[attr-defined]
        refresh_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:  # pragma: no cover - a page that cannot poll
        refresh_ms = -1.0
        notes = [f"refresh() raised {type(exc).__name__}: {exc}"]
    else:
        notes = []

    return PageReport(
        title=str(getattr(page, "title", "")),
        cards=tuple(cards),
        buttons=tuple(buttons),
        tables=tuple(tables),
        labels=tuple(labels),
        refresh_ms=refresh_ms,
        viewport_px=int(page.winfo_height()),  # type: ignore[attr-defined]
        content_px=int(page.winfo_reqheight()),  # type: ignore[attr-defined]
        notes=tuple(notes),
    )


def _screenshot_path(directory: Path, title: str) -> Path:
    return directory / f"{_slug(title)}.png"


def _grab(path: Path, bbox: tuple[int, int, int, int]) -> str:
    """Save one window screenshot. Returns "" on success, the reason on failure."""
    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]
    except Exception as exc:
        return f"Pillow is not importable ({exc.__class__.__name__}) - pip install pillow"
    try:
        if sys.platform.startswith("win"):
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
        else:
            # Pillow's Linux backend shells out; without $DISPLAY there is
            # nothing to photograph and that is worth saying out loud.
            display = os.environ.get("DISPLAY")
            if not display:
                return "no $DISPLAY - nothing to screenshot on a headless machine"
            image = ImageGrab.grab(bbox=bbox, xdisplay=display)
        image.save(path)
    except Exception as exc:
        return f"{exc.__class__.__name__}: {exc}"
    return ""


def _pump(app: object, seconds: float) -> None:
    """Let real background work finish and run its completion callbacks."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            app.update_idletasks()  # type: ignore[attr-defined]
            app.update()  # type: ignore[attr-defined]
            app.background._pump()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - the window went away
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def build_report(app: object, *, settle: float, screenshots: bool, out: Path) -> Report:
    """Walk every page of a live window and collect the findings."""
    import tkinter as tk

    root = app
    _pump(app, max(0.3, settle))

    pages: list[PageReport] = []
    shot_paths: list[str] = []
    shot_problem = ""
    for title, page in app.pages.items():  # type: ignore[attr-defined]
        try:
            app.show_page(title)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - a page that will not open
            pages.append(
                PageReport(
                    title=str(title), notes=(f"show_page raised {type(exc).__name__}: {exc}",)
                )
            )
            continue
        _pump(app, max(0.2, settle))
        report_page = collect_page(app, page, root)
        if screenshots and not shot_problem:
            target = _screenshot_path(out, str(title))
            bbox = (
                int(app.winfo_rootx()),  # type: ignore[attr-defined]
                int(app.winfo_rooty()),  # type: ignore[attr-defined]
                int(app.winfo_rootx() + app.winfo_width()),  # type: ignore[attr-defined]
                int(app.winfo_rooty() + app.winfo_height()),  # type: ignore[attr-defined]
            )
            problem = _grab(target, bbox)
            if problem:
                shot_problem = problem
            else:
                report_page = PageReport(**{**report_page.__dict__, "screenshot": target.name})
                shot_paths.append(target.name)
        pages.append(report_page)

    # A density change destroys and rebuilds every widget: the check that
    # made dead repaint callbacks visible in the first place. Counting before
    # and after is not enough on its own - the bus drops a dead owner on the
    # next notification - so both counts are taken right after a publish,
    # the way the smoke harness measures it.
    before, after = -1, -1
    try:
        app.set_theme(app.prefs.theme)  # type: ignore[attr-defined]
        before = len(app.bus._listeners)  # type: ignore[attr-defined]
        for density in DENSITY_SWEEP:
            app.set_density(density)  # type: ignore[attr-defined]
            _pump(app, 0.2)
        app.set_theme(app.prefs.theme)  # type: ignore[attr-defined]
        after = len(app.bus._listeners)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - a window that cannot restyle
        pages.append(
            PageReport(
                title="__density__",
                notes=(f"the density sweep failed: {type(exc).__name__}: {exc}",),
            )
        )

    foreign: list[str] = []
    for widget in _walk(app):
        bus = getattr(widget, "_cc_bus", None)
        if bus is not None and bus is not app.bus:  # type: ignore[attr-defined]
            foreign.append(str(widget.winfo_class()))

    checks: dict[str, object] = {
        "theme listeners": (
            f"{before} settled, {after} after three density changes"
            + ("" if after <= before else f" - grows by {after - before}, which is a leak")
        ),
        "widgets on a foreign theme bus": ", ".join(sorted(set(foreign))) if foreign else "none",
        "screenshots": ", ".join(shot_paths) if shot_paths else (shot_problem or "disabled"),
    }

    window: dict[str, object] = {
        "size": f"{app.winfo_width()}x{app.winfo_height()}",  # type: ignore[attr-defined]
        "display": f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}",  # type: ignore[attr-defined]
        "viewport scale": round(float(app.scale.viewport_scale), 3),  # type: ignore[attr-defined]
        "theme": str(app.prefs.theme),  # type: ignore[attr-defined]
        "density": str(app.prefs.density),  # type: ignore[attr-defined]
    }
    return Report(
        host={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tk": str(tk.TkVersion),
        },
        window=window,
        pages=tuple(pages),
        checks=checks,
        screenshots=tuple(shot_paths),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure and photograph the real Control Center window."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"where the report and the screenshots go (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.6,
        help="seconds to wait per page for background polls to land (default: 0.6)",
    )
    parser.add_argument(
        "--no-screenshots", action="store_true", help="skip the PNGs (the report is still written)"
    )
    parser.add_argument("--pages", default="", help="comma-separated page titles, default: all")
    parser.add_argument("--project-path", default=None, help="project root, default: this checkout")
    parser.add_argument("--output-root", default="training", help="run artifacts root")
    args = parser.parse_args(argv)

    try:
        import tkinter  # noqa: F401  # imported for the error message below
    except ImportError:
        print(
            "Tkinter is not available in this Python installation. Install it with\n"
            "  Debian/Ubuntu: sudo apt-get install python3-tk\n"
            "  or use a Python build with Tk support.",
            file=sys.stderr,
        )
        return 2

    repository = Path(__file__).resolve().parent.parent
    if str(repository / "python") not in sys.path:
        sys.path.insert(0, str(repository / "python"))

    from sandboxai.adapter import SandboxAIAdapter
    from sandboxai.control_center_desktop import ControlCenter

    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)

    app = None
    try:
        adapter = SandboxAIAdapter(
            project_root=args.project_path or repository, output_root=args.output_root
        )
        app = ControlCenter(adapter=adapter)
        app.geometry("1800x980")
        app.update()
        if args.pages:
            wanted = {name.strip() for name in args.pages.split(",") if name.strip()}
            unknown = wanted - set(app.pages)
            if unknown:
                print(f"unknown page(s): {', '.join(sorted(unknown))}", file=sys.stderr)
                return 2
            app.pages = {name: page for name, page in app.pages.items() if name in wanted}
        report = build_report(app, settle=args.settle, screenshots=not args.no_screenshots, out=out)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            # The window is going away anyway; a TclError here would mask
            # the real finding the report was written for.
            with contextlib.suppress(Exception):
                app.destroy()

    markdown = render_markdown(report)
    (out / "report.md").write_text(markdown, encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(report_to_dict(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(markdown)
    print(f"report: {out / 'report.md'}")
    print(f"data:   {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
