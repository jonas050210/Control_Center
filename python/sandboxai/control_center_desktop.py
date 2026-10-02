"""Tk desktop Control Center backed exclusively by :mod:`sandboxai.adapter`.

Architecture (kept intentionally unchanged from the first version):

    Desktop Control Center (this file)
        -> SandboxAIAdapter (python/sandboxai/adapter.py)
            -> existing CLI/training/evaluation/benchmark infrastructure
                -> headless Godot workers

This is the Control Center: a normal movable/resizable desktop application
that operates the *headless* training stack. There is no rendering, no
environment visualisation and no human-play mode — the window only reads
what the backends actually measured (agents, workers, environments, steps,
throughput, rewards/losses, resources, checkpoints, logs) and issues
lifecycle commands through the cooperative control protocol.

This module owns *presentation only*: widget layout, polling cadence and
turning user input into calls on :class:`~sandboxai.adapter.SandboxAIAdapter`.
Every dict-shaping/formatting rule lives in
:mod:`sandboxai.control_center_viewmodel` so it is unit-testable without Tk
and cannot drift between pages. No page invents a metric: a value the
adapter did not return is shown as "n/a", never estimated.

Tk is a small, supported local desktop dependency that ships with CPython on
Windows; no additional GUI toolkit is required. All adapter calls that touch
disk or a subprocess run on a background thread pool
(:class:`BackgroundRunner`) and hand their result back to the Tk thread
through a queue, so a slow poll (a big run directory, a stuck process list)
never freezes the window.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import (  # messagebox re-export keeps the public test/embedding seam stable
    messagebox,
    ttk,
)

from .adapter import SandboxAIAdapter
from .control_center_pages import PAGE_CLASSES, Page
from .control_center_widgets import (
    _FONT_FAMILY,
    _MONO_FONT,
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_HOVER,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_SURFACE,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_WARN,
    BackgroundRunner,
    ToolTip,
)

# ---------------------------------------------------------------------------
# Application shell
# ---------------------------------------------------------------------------

__all__ = ["ControlCenter", "PAGE_CLASSES", "main", "messagebox"]


class ControlCenter(tk.Tk):
    POLL_MS = 600

    def __init__(self, adapter: SandboxAIAdapter | None = None) -> None:
        super().__init__()
        self.title("SandboxAI Studio")
        self.geometry("1420x900")
        self.minsize(1100, 700)
        self.adapter = adapter or SandboxAIAdapter()
        self.background = BackgroundRunner(self)
        self._configure_style()
        self.pages: dict[str, Page] = {}
        self._current: Page | None = None
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._build_shell()
        self.after(self.POLL_MS, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        self.configure(background=COLOR_BG)
        self.option_add("*TCombobox*Listbox.background", COLOR_SURFACE_RAISED)
        self.option_add("*TCombobox*Listbox.foreground", COLOR_TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", "#1d4ed8")
        self.option_add("*TCombobox*Listbox.selectForeground", COLOR_TEXT)
        style.configure(
            ".",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_SURFACE_RAISED,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            font=(_FONT_FAMILY, 10),
        )
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Shell.TFrame", background=COLOR_BG)
        style.configure("Header.TFrame", background=COLOR_SURFACE)
        style.configure("Nav.TFrame", background=COLOR_SURFACE)
        style.configure("Content.TFrame", background=COLOR_BG)
        style.configure("Surface.TFrame", background=COLOR_SURFACE)
        style.configure("Raised.TFrame", background=COLOR_SURFACE_RAISED)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure(
            "BrandEyebrow.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 9),
        )
        style.configure(
            "BrandTitle.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 14, "bold"),
        )
        style.configure(
            "OperatorStatus.TLabel",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_OK,
            font=(_FONT_FAMILY, 9),
            padding=(12, 5),
        )
        style.configure(
            "PageTitle.TLabel",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 18, "bold"),
        )
        style.configure(
            "PageSubtitle.TLabel",
            background=COLOR_BG,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 11, "bold"),
        )
        style.configure(
            "FieldTitle.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 9, "bold"),
        )
        style.configure(
            "FieldHelp.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 8),
        )
        style.configure(
            "Card.TFrame",
            background=COLOR_SURFACE_RAISED,
            bordercolor=COLOR_BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "CardInner.TFrame",
            background=COLOR_SURFACE_RAISED,
        )
        style.configure(
            "CardLabel.TLabel",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 9),
        )
        style.configure(
            "CardValue.TLabel",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            font=(_MONO_FONT, 13, "bold"),
        )
        style.configure(
            "Leader.TLabel",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            padding=(12, 8),
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.configure(
            "TLabelframe",
            background=COLOR_SURFACE,
            bordercolor=COLOR_BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.configure(
            "TButton",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            padding=(12, 7),
            font=(_FONT_FAMILY, 9),
        )
        style.map(
            "TButton",
            background=[("active", COLOR_HOVER), ("pressed", "#2d333f")],
            foreground=[("disabled", COLOR_MUTED)],
        )
        style.configure(
            "Primary.TButton",
            background=COLOR_ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            padding=(16, 8),
            font=(_FONT_FAMILY, 9, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#2563eb"), ("pressed", "#1d4ed8"), ("disabled", COLOR_BORDER)],
            foreground=[("disabled", COLOR_MUTED)],
        )
        style.configure(
            "Danger.TButton",
            background="#2b1215",
            foreground=COLOR_ERROR,
            bordercolor="#5c1f24",
            borderwidth=1,
            padding=(12, 7),
            font=(_FONT_FAMILY, 9),
        )
        style.map("Danger.TButton", background=[("active", "#3f171b"), ("pressed", "#521c22")])
        style.configure(
            "NavGroup.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 8, "bold"),
        )
        style.configure(
            "NavFootnote.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 8),
        )
        style.configure(
            "Nav.TButton",
            anchor="w",
            padding=(14, 9),
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            borderwidth=0,
            font=(_FONT_FAMILY, 10),
        )
        style.map(
            "Nav.TButton",
            background=[("active", COLOR_HOVER), ("pressed", COLOR_SURFACE_RAISED)],
            foreground=[("active", COLOR_TEXT)],
        )
        style.configure(
            "NavSelected.TButton",
            anchor="w",
            padding=(14, 9),
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            borderwidth=0,
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.map(
            "NavSelected.TButton",
            background=[("active", COLOR_HOVER), ("pressed", COLOR_SURFACE_RAISED)],
            foreground=[("active", COLOR_TEXT)],
        )
        style.configure(
            "Warning.TLabel",
            background="#2c2008",
            foreground=COLOR_WARN,
            padding=(12, 8),
            font=(_FONT_FAMILY, 9),
        )
        style.configure(
            "Error.TLabel",
            background="#331115",
            foreground=COLOR_ERROR,
            padding=(12, 8),
            font=(_FONT_FAMILY, 9),
        )
        style.configure(
            "Treeview",
            background=COLOR_SURFACE,
            fieldbackground=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            rowheight=28,
            borderwidth=0,
            font=(_MONO_FONT, 9),
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_MUTED,
            relief="flat",
            padding=(10, 7),
            font=(_FONT_FAMILY, 9, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", "#1e3a8a")],
            foreground=[("selected", COLOR_TEXT)],
        )
        style.configure(
            "TEntry",
            fieldbackground=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            insertcolor=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            padding=6,
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            padding=5,
        )
        style.configure("TSeparator", background=COLOR_BORDER)
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=COLOR_SURFACE, foreground=COLOR_MUTED, padding=(12, 8)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLOR_SURFACE_RAISED)],
            foreground=[("selected", COLOR_TEXT)],
        )

    def _build_shell(self) -> None:
        outer = ttk.Frame(self, style="Shell.TFrame", padding=0)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Header.TFrame", padding=(24, 12))
        header.pack(fill="x")
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="SandboxAI", style="BrandTitle.TLabel").pack(side="left")
        ttk.Label(
            brand,
            text="   Training & Calibration Studio",
            style="BrandEyebrow.TLabel",
        ).pack(side="left", pady=(2, 0))
        self.status_label = ttk.Label(
            header, text="Ready", style="OperatorStatus.TLabel"
        )
        self.status_label.pack(side="right", pady=2)
        self.telemetry_badge = ttk.Label(
            header,
            text="Bridge v3  ·  84-Obs  ·  6-Head",
            style="BrandEyebrow.TLabel",
            padding=(12, 5),
        )
        self.telemetry_badge.pack(side="right", padx=(0, 8), pady=2)
        tk.Frame(outer, height=1, background=COLOR_BORDER, borderwidth=0).pack(fill="x")

        body = ttk.Frame(outer, style="Shell.TFrame")
        body.pack(fill="both", expand=True)
        nav = ttk.Frame(body, style="Nav.TFrame", width=220, padding=(12, 16))
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        tk.Frame(body, width=1, background=COLOR_BORDER, borderwidth=0).pack(
            side="left", fill="y"
        )
        self.content = ttk.Frame(body, style="Content.TFrame", padding=(20, 18, 24, 18))
        self.content.pack(side="left", fill="both", expand=True)

        nav_groups = {
            0: "OVERVIEW",
            1: "WORKFLOWS",
            4: "ARTIFACTS",
            5: "SYSTEM",
        }
        for index, page_class in enumerate(PAGE_CLASSES):
            if index in nav_groups:
                if index:
                    ttk.Separator(nav).pack(fill="x", pady=(14, 8))
                ttk.Label(nav, text=nav_groups[index], style="NavGroup.TLabel").pack(
                    fill="x", padx=10, pady=(0, 4)
                )
            button = ttk.Button(
                nav,
                text=page_class.title,
                style="Nav.TButton",
                command=lambda name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
            button.pack(fill="x", pady=1)
            ToolTip(button, f"Open {page_class.title} (Ctrl+{index + 1})")
            self._nav_buttons[page_class.title] = button
            page = page_class(self.content, self)
            self.pages[page_class.title] = page
            self.bind(
                f"<Control-Key-{index + 1}>",
                lambda _evt, name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )

        ttk.Separator(nav).pack(fill="x", pady=14)
        ttk.Label(
            nav,
            text="Shortcuts: Ctrl+1 .. Ctrl+7\nHeadless Godot Bridge v3",
            style="NavFootnote.TLabel",
            justify="left",
        ).pack(anchor="w", padx=10)

        self.show_page(PAGE_CLASSES[0].title)

    def show_page(self, name: str) -> None:
        if self._current is not None:
            self._current.pack_forget()
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        self._current = page
        for title, button in self._nav_buttons.items():
            selected = title == name
            button.configure(
                style="NavSelected.TButton" if selected else "Nav.TButton",
                text=title,
            )
        page.show()

    def set_status(self, message: str, error: bool = False) -> None:
        self.status_label.configure(
            text=message,
            foreground=COLOR_ERROR if error else COLOR_OK,
        )

    def set_output_root(self, output_root: str) -> None:
        self.background.close()
        self.adapter = SandboxAIAdapter(
            project_root=self.adapter.project_root, output_root=output_root
        )
        self.background = BackgroundRunner(self)
        for page in self.pages.values():
            # close() deliberately discards queued delivery callbacks. A page
            # must therefore not retain a coalesced-poll guard that belonged
            # to the retired runner, or a later visit would never refresh.
            page.reset_polls()
            page.adapter = self.adapter

    def _tick(self) -> None:
        if self._current is not None:
            try:
                self._current.refresh()
            except Exception as exc:  # noqa: BLE001 - a page bug must not stop polling entirely
                self.set_status(
                    f"internal error refreshing {self._current.title}: {exc}", error=True
                )
        self.after(self.POLL_MS, self._tick)

    def _on_close(self) -> None:
        self.background.close()
        self.adapter.close()
        self.destroy()


def main(project_root: str | None = None, output_root: str = "training") -> int:
    app = ControlCenter(
        adapter=SandboxAIAdapter(project_root=project_root, output_root=output_root)
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
