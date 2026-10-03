"""Design tokens, themes, DPI scaling and persisted UI preferences for the Control Center.

The desktop Control Center used to hard-code its palette, font sizes and
paddings across three modules. That made a coherent restyle impossible and
let a handful of 8-9 point labels survive on a 1080p screen. This module is
the single source of truth for every visual decision instead:

* :class:`Theme` - one named palette (surface ramp, text ramp, accents,
  status colours, glow) as plain strings so it can be diffed and tested.
* :class:`UiScale` - DPI awareness plus the type scale. No text role maps
  below :data:`MIN_FONT_PX`, which is what keeps the UI readable at 100 %
  and at 150 % Windows scaling alike.
* :class:`Density` - row height and padding steps for the three density
  presets.
* :class:`UiPreferences` / :class:`PreferencesStore` - the machine-local
  ``.sandboxai/ui/preferences.json`` (theme, layout, density, motion,
  radius, effects, window geometry, last page).
* :func:`apply_ttk_styles` - one function that configures every ttk style
  the pages use, so a theme switch is a real restyle and not a patch.

Nothing here imports a page or a widget, so it stays importable from tests
without a display or an adapter.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

__all__ = [
    "DENSITIES",
    "Density",
    "FONT_ROLES",
    "LAYOUT_MODES",
    "MIN_FONT_PX",
    "MOTION_LEVELS",
    "PreferencesStore",
    "THEME_NAMES",
    "THEMES",
    "Theme",
    "FontSpec",
    "UiPreferences",
    "UiScale",
    "apply_ttk_styles",
    "enable_dpi_awareness",
    "get_theme",
    "preferences_path",
]

# ---------------------------------------------------------------------------
# Type scale
# ---------------------------------------------------------------------------

#: Font roles and their nominal pixel size at 100 % scaling. ``display`` is
#: reserved for the brand, ``h1`` for page titles, ``h2`` for card headers.
FONT_ROLES: dict[str, int] = {
    "display": 26,
    "h1": 19,
    "h2": 14,
    "body": 12,
    "small": 11,
    "micro": 10,
    "mono": 12,
}

#: Hard floor for every rendered font. The desktop app's core complaint was
#: 8-9 pt help text; this constant is the contract that it cannot come back.
MIN_FONT_PX = 11

#: Font stacks per platform. The first entry that Tk resolves wins.
_FONT_STACKS: tuple[tuple[str, ...], ...] = (
    ("Segoe UI Variable Text", "Segoe UI", "Inter", "DejaVu Sans"),
    ("JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono"),
)

STANDARD_FONTS: tuple[str, ...] = _FONT_STACKS[0]
MONO_FONTS: tuple[str, ...] = _FONT_STACKS[1]


#: A Tk font specification: family + size, optionally with a weight/slant.
FontSpec = tuple[str, int] | tuple[str, int, str]


def _pick_font(candidates: tuple[str, ...]) -> str:
    """Return the first platform-plausible family without constructing Tk.

    Tk resolves families lazily; picking a family that does not exist makes
    Tk silently fall back to a default that may be much wider. The
    platform-specific first choices here are the ones that exist on the
    supported Windows/Linux desktops, and the final entry of every stack is
    a family that ships with the platform's default Tk install.
    """
    if sys.platform.startswith("win"):
        return candidates[0]
    for name in candidates:
        if name in ("DejaVu Sans", "DejaVu Sans Mono"):
            return name
    return candidates[-1]


FONT_FAMILY = _pick_font(STANDARD_FONTS)
MONO_FONT_FAMILY = _pick_font(MONO_FONTS)


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


def _lighten(color: str, amount: float) -> str:
    """Mix a ``#rrggbb`` colour towards white by ``amount`` (0..1)."""
    color = color.lstrip("#")
    channels = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    mixed = [min(255, round(channel + (255 - channel) * amount)) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _relative_luminance(color: str) -> float:
    """WCAG relative luminance of a ``#rrggbb`` colour."""

    def linear(channel: int) -> float:
        value = channel / 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    color = color.lstrip("#")
    return (
        0.2126 * linear(int(color[0:2], 16))
        + 0.7152 * linear(int(color[2:4], 16))
        + 0.0722 * linear(int(color[4:6], 16))
    )


def normalize_accent(value: str | None) -> str:
    """Return a usable ``#rrggbb`` accent, or ``""`` for "use the theme's own".

    Accepts ``#rgb``, ``#rrggbb``, the same without the hash, and surrounding
    whitespace. Anything else is refused (``""``), because an accent the
    operator typed is applied to every page and a silently invented colour
    would be worse than falling back to the theme's designed one.
    """
    if not value:
        return ""
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(channel * 2 for channel in text)
    if len(text) != 6:
        return ""
    try:
        int(text, 16)
    except ValueError:
        return ""
    return "#" + text.lower()


def readable_on(color: str) -> str:
    """Label text for a filled ``color`` surface: white, or black if needed.

    White is kept whenever it reaches the 3:1 WCAG minimum for large/bold
    text (which is what an accent-coloured button label is), so a
    hand-picked accent looks like the theme's own design; only colours that
    cannot carry white text at all - amber, lime, silver - switch to the
    dark ink. Deriving the inverse (black first) would have flipped the
    Corz blue of the built-in themes to dark text and made the same colour
    look different depending on how it was chosen.
    """
    dark_ink = "#0B0C0E"
    white_contrast = 1.05 / (_relative_luminance(color) + 0.05)
    return "#FFFFFF" if white_contrast >= 3.0 else dark_ink


@dataclass(frozen=True)
class Theme:
    """One complete colour palette."""

    name: str
    label: str
    bg: str
    shell: str
    panel: str
    card: str
    card_hover: str
    card_active: str
    border: str
    border_strong: str
    text: str
    text_dim: str
    text_muted: str
    accent: str
    accent_soft: str
    accent_second: str
    ok: str
    warn: str
    error: str
    info: str
    on_accent: str = "#FFFFFF"

    def color(self, role: str) -> str:
        """Look up a colour by token name (``theme.color("accent")``)."""
        value = getattr(self, role, None)
        if not isinstance(value, str):
            raise KeyError(f"unknown theme colour: {role}")
        return value

    def with_accent(self, color: str) -> Theme:
        """A copy of this theme with ``color`` as the accent.

        The soft tint and the text colour *on* the accent are derived here,
        so a hand-picked accent cannot produce unreadable button labels:
        the same rule the built-in themes follow is applied to it.
        """
        from dataclasses import replace

        if color.lower() == self.accent.lower():
            # Picking the theme's own accent is a no-op, so a designed pair
            # (cyan with its deep-teal label ink, lime with its dark green)
            # stays exactly as the theme author chose it.
            return self
        return replace(
            self,
            accent=color,
            accent_soft=_lighten(color, 0.22),
            on_accent=readable_on(color),
        )

    def contrast_ratio(self, first: str, second: str) -> float:
        """WCAG contrast ratio between two of this theme's own tokens.

        Always ``>= 1``: the brighter colour is the numerator. (The first
        version of this helper unpacked ``sorted(...)`` the wrong way round,
        so it reported the reciprocal - a contrast of 3.7:1 came back as
        0.27. It had no caller at the time, which is how it survived.)
        """
        darker, lighter = sorted(
            (_relative_luminance(self.color(first)), _relative_luminance(self.color(second)))
        )
        return (lighter + 0.05) / (darker + 0.05)


def _theme(
    name: str,
    label: str,
    *,
    bg: str,
    shell: str,
    panel: str,
    card: str,
    accent: str,
    accent_second: str,
    text: str = "#F2F4F8",
    text_dim: str = "#A7AEBD",
    text_muted: str = "#767E8C",
    ok: str = "#34D399",
    warn: str = "#F5A524",
    error: str = "#F26D6D",
    info: str = "#60A5FA",
    on_accent: str = "#FFFFFF",
) -> Theme:
    """Build a theme from the surface/accent inputs, deriving the rest."""

    return Theme(
        name=name,
        label=label,
        bg=bg,
        shell=shell,
        panel=panel,
        card=card,
        card_hover=_lighten(card, 0.05),
        card_active=_lighten(card, 0.10),
        border=_lighten(card, 0.10),
        border_strong=_lighten(card, 0.20),
        text=text,
        text_dim=text_dim,
        text_muted=text_muted,
        accent=accent,
        accent_soft=_lighten(accent, 0.22),
        accent_second=accent_second,
        ok=ok,
        warn=warn,
        error=error,
        info=info,
        on_accent=on_accent,
    )


THEMES: dict[str, Theme] = {
    "corz": _theme(
        "corz",
        "Corz Dark",
        bg="#07080A",
        shell="#0C0E12",
        panel="#12151B",
        card="#181C24",
        accent="#4F7CFF",
        accent_second="#8B5CF6",
    ),
    "cyan": _theme(
        "cyan",
        "Midnight Cyan",
        bg="#05090B",
        shell="#0A1014",
        panel="#101A20",
        card="#16232B",
        accent="#22D3EE",
        accent_second="#3B82F6",
        on_accent="#04121A",
    ),
    "lime": _theme(
        "lime",
        "Neon Lime",
        bg="#080B07",
        shell="#0D120C",
        panel="#151C14",
        card="#1C261B",
        accent="#A3E635",
        accent_second="#22C55E",
        on_accent="#0B1405",
    ),
    "graphite": _theme(
        "graphite",
        "Graphite Mono",
        bg="#0A0A0B",
        shell="#111113",
        panel="#17171A",
        card="#1F1F23",
        accent="#E5E7EB",
        accent_second="#9CA3AF",
        on_accent="#101012",
    ),
    "light": _theme(
        "light",
        "Studio Light",
        bg="#EEF1F6",
        shell="#F7F9FC",
        panel="#FFFFFF",
        card="#F2F5FA",
        accent="#2563EB",
        accent_second="#7C3AED",
        text="#111827",
        text_dim="#3F4653",
        text_muted="#6B7280",
        ok="#0F9D63",
        warn="#B45309",
        error="#DC2626",
        info="#1D4ED8",
    ),
}

THEME_NAMES: tuple[str, ...] = tuple(THEMES)

#: Accent colours an operator can pick without typing a hex value. Kept
#: deliberately few: an accent colour is a taste decision, and every one of
#: these keeps its label text readable (see :func:`readable_on`).
ACCENT_PRESETS: tuple[tuple[str, str], ...] = (
    ("#4F7CFF", "Corz Blue"),
    ("#22D3EE", "Cyan"),
    ("#34D399", "Mint"),
    ("#A3E635", "Lime"),
    ("#F5A524", "Amber"),
    ("#F26D6D", "Coral"),
    ("#EC4899", "Pink"),
    ("#8B5CF6", "Violet"),
    ("#E5E7EB", "Silver"),
)


def get_theme(name: str | None, accent: str = "") -> Theme:
    """Return a theme by name, falling back to the default instead of raising.

    A preferences file written by a newer/older build must never stop the
    window from opening, so an unknown name is a fallback, not an error.
    ``accent`` (when it parses) replaces the theme's own accent colour.
    """
    theme = THEMES[name] if name is not None and name in THEMES else THEMES["corz"]
    accent = normalize_accent(accent)
    return theme.with_accent(accent) if accent else theme


# ---------------------------------------------------------------------------
# Density, layout, motion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Density:
    """Spacing steps for one density preset."""

    name: str
    label: str
    row_height: int
    pad: int
    gap: int
    card_pad: int
    font_delta: int = 0


DENSITIES: dict[str, Density] = {
    "comfort": Density("comfort", "Comfort", row_height=34, pad=14, gap=10, card_pad=16),
    "compact": Density("compact", "Compact", row_height=30, pad=11, gap=8, card_pad=13),
    "ultra": Density("ultra", "Ultra", row_height=27, pad=9, gap=6, card_pad=11, font_delta=-1),
}

LAYOUT_MODES: dict[str, str] = {
    "rail": "Rail",
    "topbar": "Topbar",
    "board": "Command Board",
}

MOTION_LEVELS: dict[str, str] = {
    "off": "Off",
    "reduced": "Reduced",
    "normal": "Normal",
    "cinematic": "Cinematic",
}

#: Speed multiplier per motion level: 0 disables animation entirely.
MOTION_SPEED: dict[str, float] = {"off": 0.0, "reduced": 1.6, "normal": 1.0, "cinematic": 0.75}


# ---------------------------------------------------------------------------
# DPI
# ---------------------------------------------------------------------------


def enable_dpi_awareness() -> bool:
    """Make the process DPI-aware on Windows before Tk is created.

    Without this, Windows scales the window bitmap after the fact: Tk
    reports 96 dpi, lays the UI out for 96 dpi and the compositor blurs the
    result. Returns whether awareness was actually enabled, so a caller can
    report it honestly instead of pretending.
    """
    if not sys.platform.startswith("win"):
        return False
    import ctypes

    with contextlib.suppress(OSError, AttributeError):
        # PROCESS_PER_MONITOR_DPI_AWARE = 2, available since Windows 8.1.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        return True
    with contextlib.suppress(OSError, AttributeError):
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        return True
    return False


def compute_viewport_scale(width: int, height: int) -> float:
    """Compute a proportional UI scale factor from the live window viewport size.

    At 1800x980 (a 1920x1080 screen) the factor is 1.0. Smaller windows scale
    elements, fonts, paddings, and row heights down proportionally (down to 0.68x),
    while larger windows scale up gently (up to 1.15x). Quantized in 0.04 steps
    so dragging a window border does not thrash style rebuilds on 1-pixel jitter.
    """
    w = max(480, int(width or 1800))
    h = max(360, int(height or 980))
    raw = min(w / 1800.0, h / 980.0)
    clamped = max(0.68, min(1.15, raw))
    return round(round(clamped / 0.04) * 0.04, 2)


@dataclass(frozen=True)
class UiScale:
    """Pixel scaling and the type scale derived from one DPI value and viewport scale."""

    dpi: float = 96.0
    font_family: str = FONT_FAMILY
    mono_family: str = MONO_FONT_FAMILY
    viewport_scale: float = 1.0

    @property
    def factor(self) -> float:
        dpi_factor = max(0.85, min(2.5, float(self.dpi) / 96.0))
        vp_factor = max(0.65, min(1.25, float(self.viewport_scale)))
        return max(0.60, min(2.5, dpi_factor * vp_factor))

    def with_viewport(self, width: int, height: int) -> UiScale:
        """Return a copy of this UiScale adjusted for the current window dimensions."""
        vp = compute_viewport_scale(width, height)
        if abs(vp - self.viewport_scale) < 0.03:
            return self
        return UiScale(
            dpi=self.dpi,
            font_family=self.font_family,
            mono_family=self.mono_family,
            viewport_scale=vp,
        )

    @classmethod
    def from_root(cls, root: Any, override: float | None = None) -> UiScale:
        """Measure the display from a live Tk root (``winfo_fpixels('1i')``)."""
        dpi = override
        if dpi is None:
            with contextlib.suppress(Exception):
                dpi = float(root.winfo_fpixels("1i"))
        return cls(dpi=float(dpi or 96.0))

    def px(self, value: float, *, minimum: int = 0) -> int:
        eff_min = minimum
        if minimum > 1 and self.viewport_scale < 1.0:
            eff_min = max(1, int(round(minimum * self.viewport_scale)))
        return max(eff_min, int(round(value * self.factor)))

    def font(self, role: str, *, bold: bool = False, mono: bool = False) -> FontSpec:
        """Resolve a font role into a Tk font tuple, never below 11 px at default viewport scale."""
        base = FONT_ROLES.get(role, FONT_ROLES["body"])
        floor = (
            MIN_FONT_PX
            if self.viewport_scale >= 1.0
            else max(9, int(round(MIN_FONT_PX * self.viewport_scale)))
        )
        size = max(floor, self.px(base, minimum=floor))
        family = self.mono_family if mono else self.font_family
        return (family, size, "bold") if bold else (family, size)


# ---------------------------------------------------------------------------
# Persisted preferences
# ---------------------------------------------------------------------------


@dataclass
class UiPreferences:
    """Everything the GUI remembers between sessions."""

    theme: str = "corz"
    layout: str = "rail"
    density: str = "comfort"
    motion: str = "normal"
    radius: int = 10
    show_glow: bool = True
    show_grid: bool = True
    geometry: str = ""
    last_page: str = "Dashboard"
    active_preset: str = ""
    ## "" = the theme's own designed accent; otherwise a #rrggbb override
    ## that every theme then wears (see ``Theme.with_accent``).
    accent: str = ""
    ## Remembered window state: a maximized window should come back
    ## maximized instead of shrinking to its last restored size.
    zoomed: bool = False

    def normalized(self) -> UiPreferences:
        """Clamp every field to a supported value (never raise)."""
        return UiPreferences(
            theme=self.theme if self.theme in THEMES else "corz",
            layout=self.layout if self.layout in LAYOUT_MODES else "rail",
            density=self.density if self.density in DENSITIES else "comfort",
            motion=self.motion if self.motion in MOTION_LEVELS else "normal",
            radius=max(0, min(20, int(self.radius))),
            show_glow=bool(self.show_glow),
            show_grid=bool(self.show_grid),
            geometry=str(self.geometry or ""),
            last_page=str(self.last_page or "Dashboard"),
            active_preset=str(self.active_preset or ""),
            accent=normalize_accent(self.accent),
            zoomed=bool(self.zoomed),
        )


def preferences_path(project_root: str | Path) -> Path:
    """``<project>/.sandboxai/ui/preferences.json``."""
    return Path(project_root) / ".sandboxai" / "ui" / "preferences.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temp file + rename so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


class PreferencesStore:
    """Load/save :class:`UiPreferences` from a project's ``.sandboxai``."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.path = preferences_path(self.project_root)
        self.error: str | None = None

    def load(self) -> UiPreferences:
        """Read the stored preferences; a corrupt file yields the defaults."""
        self.error = None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return UiPreferences()
        except (OSError, json.JSONDecodeError) as exc:
            self.error = f"{self.path}: {exc}"
            return UiPreferences()
        if not isinstance(raw, dict):
            self.error = f"{self.path}: expected a JSON object"
            return UiPreferences()
        known = {spec.name for spec in fields(UiPreferences)}
        values = {key: value for key, value in raw.items() if key in known}
        try:
            return UiPreferences(**values).normalized()
        except (TypeError, ValueError) as exc:
            self.error = f"{self.path}: {exc}"
            return UiPreferences()

    def save(self, preferences: UiPreferences) -> None:
        try:
            _atomic_write_json(self.path, asdict(preferences.normalized()))
            self.error = None
        except OSError as exc:
            self.error = f"{self.path}: {exc}"


# ---------------------------------------------------------------------------
# ttk style application
# ---------------------------------------------------------------------------


def apply_ttk_styles(
    root: Any,
    theme: Theme,
    *,
    density: Density = DENSITIES["comfort"],
    scale: UiScale | None = None,
) -> Any:
    """Configure every ttk style the Control Center uses for one theme.

    Called once at startup and again on every theme/density switch, so a
    theme change is a complete restyle. Returns the ``ttk.Style`` object so
    callers (and tests) can assert on it.
    """
    from tkinter import ttk

    scale = scale or UiScale()
    style = ttk.Style(root)
    with contextlib.suppress(Exception):
        style.theme_use("clam")

    pad = scale.px(density.pad, minimum=6)
    gap = scale.px(density.gap, minimum=4)
    row_pad = max(4, gap - 2)

    def font(role: str, *, bold: bool = False, mono: bool = False) -> tuple[Any, ...]:
        resolved = scale.font(role, bold=bold, mono=mono)
        if density.font_delta:
            family, size, *rest = resolved
            floor = (
                MIN_FONT_PX
                if scale.viewport_scale >= 1.0
                else max(9, int(round(MIN_FONT_PX * scale.viewport_scale)))
            )
            size = max(floor, int(size) + density.font_delta)
            return (family, size, *rest)
        return resolved

    root.configure(background=theme.bg)
    root.option_add("*TCombobox*Listbox.background", theme.card)
    root.option_add("*TCombobox*Listbox.foreground", theme.text)
    root.option_add("*TCombobox*Listbox.selectBackground", theme.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", theme.on_accent)
    root.option_add("*Text.selectBackground", theme.accent)
    root.option_add("*Text.selectForeground", theme.on_accent)

    style.configure(
        ".",
        background=theme.bg,
        foreground=theme.text,
        fieldbackground=theme.card,
        bordercolor=theme.border,
        lightcolor=theme.border,
        darkcolor=theme.border,
        font=font("body"),
    )
    for name, background in (
        ("TFrame", theme.bg),
        ("Shell.TFrame", theme.bg),
        ("Header.TFrame", theme.shell),
        ("Nav.TFrame", theme.shell),
        ("Content.TFrame", theme.bg),
        ("Surface.TFrame", theme.panel),
        ("Raised.TFrame", theme.card),
    ):
        style.configure(name, background=background)
    style.configure("TLabel", background=theme.bg, foreground=theme.text, font=font("body"))

    label_styles: dict[str, dict[str, Any]] = {
        "BrandEyebrow.TLabel": {
            "background": theme.shell,
            "foreground": theme.text_dim,
            "font": font("small"),
        },
        "BrandTitle.TLabel": {
            "background": theme.shell,
            "foreground": theme.text,
            "font": font("display", bold=True),
        },
        "OperatorStatus.TLabel": {
            "background": theme.card,
            "foreground": theme.ok,
            "font": font("small"),
            "padding": (scale.px(12), scale.px(5)),
        },
        "PageTitle.TLabel": {
            "background": theme.bg,
            "foreground": theme.text,
            "font": font("h1", bold=True),
        },
        "PageSubtitle.TLabel": {
            "background": theme.bg,
            "foreground": theme.text_dim,
            "font": font("body"),
        },
        "Section.TLabel": {
            "background": theme.bg,
            "foreground": theme.text,
            "font": font("h2", bold=True),
        },
        "FieldTitle.TLabel": {
            "background": theme.panel,
            "foreground": theme.text,
            "font": font("small", bold=True),
        },
        "FieldHelp.TLabel": {
            "background": theme.panel,
            "foreground": theme.text_muted,
            "font": font("micro"),
        },
        # A one-line explanation under a page heading ("No runs yet - launch
        # a training run..."). Sits on the page background, quieter than the
        # subtitle but not as faint as the disabled text colour.
        "Hint.TLabel": {
            "background": theme.bg,
            "foreground": theme.text_dim,
            "font": font("small"),
        },
        # The overlay a table shows while it has no rows. Its background has
        # to be the table's own field background (``theme.panel``) or the hint
        # would sit in a visible rectangle on top of the empty table.
        "Empty.TLabel": {
            "background": theme.panel,
            "foreground": theme.text_dim,
            "font": font("small"),
            "padding": (scale.px(14), scale.px(10, minimum=6)),
        },
    }
    for name, options in label_styles.items():
        style.configure(name, **options)

    card_styles: dict[str, dict[str, Any]] = {
        "Card.TFrame": {
            "background": theme.card,
            "bordercolor": theme.border,
            "relief": "solid",
            "borderwidth": 1,
        },
        "CardInner.TFrame": {"background": theme.card},
        "CardLabel.TLabel": {
            "background": theme.card,
            "foreground": theme.text_dim,
            "font": font("small"),
        },
        "CardValue.TLabel": {
            "background": theme.card,
            "foreground": theme.text,
            "font": font("h2", bold=True, mono=True),
        },
        "Leader.TLabel": {
            "background": theme.card,
            "foreground": theme.text,
            "padding": (scale.px(12), scale.px(8)),
            "font": font("body", bold=True),
        },
    }
    for name, options in card_styles.items():
        style.configure(name, **options)

    style.configure(
        "TLabelframe",
        background=theme.panel,
        bordercolor=theme.border,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=theme.bg,
        foreground=theme.text_dim,
        font=font("small", bold=True),
    )

    style.configure(
        "TButton",
        background=theme.card,
        foreground=theme.text,
        bordercolor=theme.border,
        borderwidth=1,
        padding=(pad, scale.px(7, minimum=5)),
        font=font("small"),
    )
    style.map(
        "TButton",
        background=[("active", theme.card_hover), ("pressed", theme.card_active)],
        foreground=[("disabled", theme.text_muted)],
    )
    style.configure(
        "Primary.TButton",
        background=theme.accent,
        foreground=theme.on_accent,
        borderwidth=0,
        padding=(scale.px(16), scale.px(9, minimum=6)),
        font=font("small", bold=True),
    )
    style.map(
        "Primary.TButton",
        background=[
            ("active", theme.accent_soft),
            ("pressed", theme.accent),
            ("disabled", theme.border),
        ],
        foreground=[("disabled", theme.text_muted)],
    )
    style.configure(
        "Ghost.TButton",
        background=theme.panel,
        foreground=theme.text_dim,
        borderwidth=0,
        padding=(scale.px(10), scale.px(6, minimum=4)),
        font=font("small"),
    )
    style.map("Ghost.TButton", background=[("active", theme.card_hover)])
    style.configure(
        "Danger.TButton",
        background=theme.card,
        foreground=theme.error,
        bordercolor=theme.error,
        borderwidth=1,
        padding=(pad, scale.px(7, minimum=5)),
        font=font("small"),
    )
    style.map(
        "Danger.TButton",
        background=[("active", theme.card_hover), ("pressed", theme.card_active)],
    )
    style.configure(
        "Nav.TButton",
        anchor="w",
        padding=(scale.px(14), scale.px(9, minimum=6)),
        background=theme.shell,
        foreground=theme.text_dim,
        borderwidth=0,
        font=font("body"),
    )
    style.map(
        "Nav.TButton",
        background=[("active", theme.card), ("pressed", theme.card_active)],
        foreground=[("active", theme.text)],
    )
    style.configure(
        "NavSelected.TButton",
        anchor="w",
        padding=(scale.px(14), scale.px(9, minimum=6)),
        background=theme.card_active,
        foreground=theme.text,
        borderwidth=0,
        font=font("body", bold=True),
    )
    style.map(
        "NavSelected.TButton",
        background=[("active", theme.card_hover)],
        foreground=[("active", theme.text)],
    )
    style.configure(
        "NavGroup.TLabel",
        background=theme.shell,
        foreground=theme.text_muted,
        font=font("micro", bold=True),
    )
    style.configure(
        "NavFootnote.TLabel",
        background=theme.shell,
        foreground=theme.text_muted,
        font=font("micro"),
    )
    style.configure(
        "Warning.TLabel",
        background=theme.card,
        foreground=theme.warn,
        padding=(scale.px(12), scale.px(8)),
        font=font("small"),
    )
    style.configure(
        "Error.TLabel",
        background=theme.card,
        foreground=theme.error,
        padding=(scale.px(12), scale.px(8)),
        font=font("small"),
    )
    style.configure(
        "Sidebar.TLabel",
        background=theme.shell,
        foreground=theme.text_dim,
        font=font("small"),
    )
    style.configure(
        "Pill.TLabel",
        background=theme.card,
        foreground=theme.text_dim,
        padding=(scale.px(10), scale.px(4, minimum=3)),
        font=font("micro"),
    )
    style.configure(
        "PillOk.TLabel",
        background=theme.card,
        foreground=theme.ok,
        padding=(scale.px(10), scale.px(4, minimum=3)),
        font=font("micro", bold=True),
    )
    style.configure(
        "PillWarn.TLabel",
        background=theme.card,
        foreground=theme.warn,
        padding=(scale.px(10), scale.px(4, minimum=3)),
        font=font("micro", bold=True),
    )
    style.configure(
        "PillError.TLabel",
        background=theme.card,
        foreground=theme.error,
        padding=(scale.px(10), scale.px(4, minimum=3)),
        font=font("micro", bold=True),
    )

    style.configure(
        "Treeview",
        background=theme.panel,
        fieldbackground=theme.panel,
        foreground=theme.text,
        rowheight=scale.px(density.row_height, minimum=24),
        borderwidth=0,
        font=font("mono"),
    )
    style.configure(
        "Treeview.Heading",
        background=theme.card,
        foreground=theme.text_dim,
        relief="flat",
        padding=(scale.px(10), scale.px(7, minimum=4)),
        font=font("small", bold=True),
    )
    style.map(
        "Treeview",
        background=[("selected", theme.card_active)],
        foreground=[("selected", theme.text)],
    )
    style.configure(
        "TEntry",
        fieldbackground=theme.card,
        foreground=theme.text,
        insertcolor=theme.text,
        bordercolor=theme.border,
        padding=(row_pad, max(4, row_pad - 1)),
    )
    style.configure(
        "TCombobox",
        fieldbackground=theme.card,
        foreground=theme.text,
        arrowcolor=theme.text_dim,
        bordercolor=theme.border,
        padding=(row_pad, max(4, row_pad - 1), row_pad, max(4, row_pad - 1)),
    )
    style.configure("TSeparator", background=theme.border)
    style.configure(
        "TCheckbutton", background=theme.panel, foreground=theme.text, font=font("small")
    )
    style.map(
        "TCheckbutton",
        background=[("active", theme.panel)],
        foreground=[("disabled", theme.text_muted)],
    )
    style.configure("TNotebook", background=theme.bg, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=theme.panel,
        foreground=theme.text_dim,
        padding=(scale.px(14), scale.px(8)),
        font=font("small"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", theme.card)],
        foreground=[("selected", theme.text)],
    )
    style.configure(
        "TPanedwindow",
        background=theme.bg,
        sashwidth=scale.px(6, minimum=4),
    )
    # The overlay scrollbars the tables use are Canvas widgets, but Tk still
    # asks for a default scrollbar style in a few places (Text widgets).
    style.configure(
        "Vertical.TScrollbar",
        background=theme.panel,
        troughcolor=theme.bg,
        bordercolor=theme.bg,
        arrowcolor=theme.text_muted,
        width=scale.px(10, minimum=8),
    )
    return style


def prefs_as_dict(preferences: UiPreferences) -> dict[str, Any]:
    """Serialise preferences for tests and diagnostics."""
    return asdict(preferences.normalized())
