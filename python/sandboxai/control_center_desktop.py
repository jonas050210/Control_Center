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
    COLOR_ACCENT,
    COLOR_ACCENT_SECONDARY,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_HOVER,
    COLOR_MUTED,
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
    POLL_MS = 1500

    def __init__(self, adapter: SandboxAIAdapter | None = None) -> None:
        super().__init__()
        self.title("SandboxAI Control Center")
        self.geometry("1360x860")
        self.minsize(1080, 680)
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
        self.option_add("*TCombobox*Listbox.selectBackground", "#164e63")
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
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure(
            "BrandEyebrow.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_ACCENT,
            font=(_FONT_FAMILY, 9, "bold"),
        )
        style.configure(
            "BrandTitle.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 17, "bold"),
        )
        style.configure(
            "OperatorStatus.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 9, "bold"),
            padding=(10, 6),
        )
        style.configure(
            "PageTitle.TLabel",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 20, "bold"),
        )
        style.configure(
            "PageSubtitle.TLabel",
            background=COLOR_BG,
            foreground=COLOR_ACCENT_SECONDARY,
            font=(_FONT_FAMILY, 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=(_FONT_FAMILY, 12, "bold"),
        )
        style.configure(
            "Card.TFrame",
            background=COLOR_SURFACE_RAISED,
            bordercolor=COLOR_BORDER,
            relief="solid",
            borderwidth=1,
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
            font=(_FONT_FAMILY, 15, "bold"),
        )
        style.configure(
            "TLabelframe", background=COLOR_BG, bordercolor=COLOR_BORDER, relief="solid"
        )
        style.configure(
            "TLabelframe.Label",
            background=COLOR_BG,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.configure(
            "TButton",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            borderwidth=0,
            padding=(12, 8),
        )
        style.map(
            "TButton",
            background=[("active", COLOR_HOVER), ("pressed", "#243b53")],
            foreground=[("disabled", COLOR_MUTED)],
        )
        style.configure(
            "Primary.TButton",
            background=COLOR_ACCENT,
            foreground="#06131d",
            borderwidth=0,
            padding=(14, 9),
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#75e8ff"), ("pressed", "#179dc1"), ("disabled", COLOR_BORDER)],
        )
        style.configure(
            "Danger.TButton",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_ERROR,
            borderwidth=0,
            padding=(12, 8),
        )
        style.map("Danger.TButton", background=[("active", "#4c1d2a"), ("pressed", "#881337")])
        style.configure(
            "NavGroup.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 9, "bold"),
        )
        style.configure(
            "NavFootnote.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_MUTED,
            font=(_FONT_FAMILY, 9),
        )
        style.configure("Nav.TButton", anchor="w", padding=(14, 10), background=COLOR_SURFACE)
        style.map(
            "Nav.TButton",
            background=[("active", COLOR_HOVER), ("pressed", "#193a55")],
            foreground=[("active", COLOR_TEXT)],
        )
        style.configure(
            "NavSelected.TButton",
            anchor="w",
            padding=(14, 10),
            background=COLOR_HOVER,
            foreground=COLOR_ACCENT,
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.map(
            "NavSelected.TButton",
            background=[("active", "#193a55"), ("pressed", "#193a55")],
            foreground=[("active", COLOR_ACCENT)],
        )
        style.configure(
            "Warning.TLabel",
            background="#302711",
            foreground=COLOR_WARN,
            padding=(12, 9),
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.configure(
            "Error.TLabel",
            background="#351923",
            foreground=COLOR_ERROR,
            padding=(12, 9),
            font=(_FONT_FAMILY, 10, "bold"),
        )
        style.configure(
            "Treeview",
            background=COLOR_SURFACE,
            fieldbackground=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SURFACE_RAISED,
            foreground=COLOR_MUTED,
            relief="flat",
            padding=(8, 8),
        )
        style.map(
            "Treeview", background=[("selected", "#164e63")], foreground=[("selected", COLOR_TEXT)]
        )
        style.configure(
            "TEntry",
            fieldbackground=COLOR_SURFACE_RAISED,
            foreground=COLOR_TEXT,
            insertcolor=COLOR_TEXT,
            padding=7,
        )
        style.configure(
            "TCombobox", fieldbackground=COLOR_SURFACE_RAISED, foreground=COLOR_TEXT, padding=6
        )
        style.configure("TSeparator", background=COLOR_BORDER)
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=COLOR_SURFACE, foreground=COLOR_MUTED, padding=(12, 8)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLOR_HOVER)],
            foreground=[("selected", COLOR_ACCENT)],
        )

    def _build_shell(self) -> None:
        outer = ttk.Frame(self, style="Shell.TFrame", padding=0)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Header.TFrame", padding=(18, 11))
        header.pack(fill="x")
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="SANDBOXAI // TTK", style="BrandEyebrow.TLabel").pack(anchor="w")
        ttk.Label(brand, text="CONTROL CENTER", style="BrandTitle.TLabel").pack(anchor="w")
        self.status_label = ttk.Label(header, text="SYSTEM // READY", style="OperatorStatus.TLabel")
        self.status_label.pack(side="right", pady=4)
        ttk.Separator(outer).pack(fill="x")

        body = ttk.Frame(outer, style="Shell.TFrame")
        body.pack(fill="both", expand=True)
        nav = ttk.Frame(body, style="Nav.TFrame", width=224, padding=(14, 18))
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        self.content = ttk.Frame(body, style="Content.TFrame", padding=(12, 14, 20, 18))
        self.content.pack(side="left", fill="both", expand=True)

        nav_groups = {0: "OVERVIEW", 1: "OPERATIONS", 4: "ANALYSIS", 5: "SYSTEM"}
        for index, page_class in enumerate(PAGE_CLASSES):
            if index in nav_groups:
                if index:
                    ttk.Separator(nav).pack(fill="x", pady=(12, 8))
                ttk.Label(nav, text=nav_groups[index], style="NavGroup.TLabel").pack(
                    fill="x", padx=8, pady=(0, 4)
                )
            button = ttk.Button(
                nav,
                text=page_class.title,
                style="Nav.TButton",
                command=lambda name=page_class.title: self.show_page(name),  # type: ignore[misc]
            )
            button.pack(fill="x", pady=2)
            ToolTip(button, f"Open {page_class.title}")
            self._nav_buttons[page_class.title] = button
            page = page_class(self.content, self)
            self.pages[page_class.title] = page

        ttk.Separator(nav).pack(fill="x", pady=10)
        ttk.Label(
            nav,
            text="HEADLESS CONTROL\nMeasured backend data only.",
            style="NavFootnote.TLabel",
            justify="left",
        ).pack(anchor="w")

        self.show_page(PAGE_CLASSES[0].title)

    def show_page(self, name: str) -> None:
        if self._current is not None:
            self._current.pack_forget()
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        self._current = page
        for title, button in self._nav_buttons.items():
            button.configure(style="NavSelected.TButton" if title == name else "Nav.TButton")
        page.show()

    def set_status(self, message: str, error: bool = False) -> None:
        self.status_label.configure(text=message, foreground=COLOR_ERROR if error else COLOR_MUTED)

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
