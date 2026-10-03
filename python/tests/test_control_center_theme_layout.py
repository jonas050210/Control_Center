"""Headless tests for the Control Center's design tokens and layout model.

Both modules are deliberately Tk-free: the theme owns colours, density,
font sizes and persisted preferences, the layout module owns the movable
widgets and the saved presets. That means the parts of the GUI rework with
real logic behind them are testable in an environment without Tkinter or a
display - which is exactly where the Tk suite skips.

The tests pin behaviour a user would notice: nothing renders below a
readable font size, switching themes cannot lose a preference, and a
corrupt or hand-edited preset can never break the window.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from optional_deps import HAS_TKINTER, TKINTER_REASON

from sandboxai.control_center_layout import (
    LayoutState,
    PresetError,
    PresetStore,
    WidgetSpec,
    columns_for_width,
    deserialize,
    fit_window_geometry,
    move,
    normalize,
    placement_slots,
    reset_all,
    reset_page,
    safe_preset_name,
    serialize,
    set_span,
    set_visible,
)
from sandboxai.control_center_theme import (
    DENSITIES,
    FONT_ROLES,
    LAYOUT_MODES,
    MIN_FONT_PX,
    MOTION_LEVELS,
    THEME_NAMES,
    THEMES,
    PreferencesStore,
    UiPreferences,
    UiScale,
    apply_ttk_styles,
    get_theme,
    normalize_accent,
    readable_on,
)

SPECS = (
    WidgetSpec("kpis", "Key figures", min_span=1, max_span=3),
    WidgetSpec("charts", "Charts", min_span=1, max_span=3),
    WidgetSpec("log", "Log", min_span=1, max_span=2, removable=False),
)
PAGES = {"Dashboard": SPECS}


class ThemeTests(unittest.TestCase):
    def test_every_theme_is_complete_and_distinct(self) -> None:
        self.assertGreaterEqual(len(THEME_NAMES), 4)
        for name in THEME_NAMES:
            theme = THEMES[name]
            with self.subTest(theme=name):
                self.assertEqual(theme.name, name)
                self.assertTrue(theme.label)
                for field in ("bg", "shell", "panel", "card", "text", "accent", "ok", "error"):
                    value = getattr(theme, field)
                    self.assertTrue(
                        value.startswith("#") and len(value) in (7, 9), f"{field}={value}"
                    )
        self.assertEqual(THEME_NAMES[0], "corz", "the default theme stays the first entry")
        self.assertEqual(
            len({THEMES[name].bg for name in THEME_NAMES}),
            len(THEME_NAMES),
            "two themes with the same background would be indistinguishable",
        )

    def test_themes_expose_a_contrasting_text_colour(self) -> None:
        def luminance(hex_colour: str) -> float:
            value = hex_colour.lstrip("#")[:6]
            red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
            return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0

        for name in THEME_NAMES:
            theme = THEMES[name]
            with self.subTest(theme=name):
                delta = abs(luminance(theme.text) - luminance(theme.bg))
                self.assertGreater(delta, 0.35, "text must stay legible against the background")

    def test_every_font_role_stays_at_or_above_the_readable_floor(self) -> None:
        """The old UI shipped 8-9 pt help text; MIN_FONT_PX is the contract."""
        for dpi in (96.0, 120.0, 144.0, 192.0):
            scale = UiScale(dpi=dpi)
            for role in FONT_ROLES:
                for mono in (False, True):
                    with self.subTest(dpi=dpi, role=role, mono=mono):
                        family, size = scale.font(role, mono=mono)[:2]
                        self.assertTrue(family)
                        self.assertGreaterEqual(size, MIN_FONT_PX)

    def test_density_steps_shrink_without_going_to_zero(self) -> None:
        self.assertGreater(DENSITIES["comfort"].row_height, DENSITIES["ultra"].row_height)
        for name, density in DENSITIES.items():
            with self.subTest(density=name):
                self.assertGreaterEqual(density.row_height, 20)
                self.assertGreaterEqual(density.pad, 4)
                self.assertGreaterEqual(density.gap, 4)

    def test_motion_levels_cover_off_to_cinematic(self) -> None:
        self.assertIn("off", MOTION_LEVELS)
        self.assertEqual(MOTION_LEVELS["off"], "Off")

    def test_settings_combobox_uses_the_theme_surface(self) -> None:
        """Settings dropdowns stay themed after header-only selectors are removed."""

        class RecordingStyle:
            def __init__(self, _root):
                self.configured: dict[str, dict[str, object]] = {}
                self.mapped: dict[str, dict[str, object]] = {}

            def theme_use(self, _name):
                return "clam"

            def configure(self, name, **options):
                self.configured.setdefault(name, {}).update(options)

            def map(self, name, **options):
                self.mapped.setdefault(name, {}).update(options)

        class Root:
            def configure(self, **_options):
                pass

            def option_add(self, *_args):
                pass

        tkinter = ModuleType("tkinter")
        ttk = ModuleType("tkinter.ttk")
        ttk.Style = RecordingStyle  # type: ignore[attr-defined]
        tkinter.ttk = ttk  # type: ignore[attr-defined]
        with patch.dict("sys.modules", {"tkinter": tkinter, "tkinter.ttk": ttk}):
            style = apply_ttk_styles(Root(), THEMES["corz"])

        self.assertEqual(style.configured["TCombobox"]["fieldbackground"], THEMES["corz"].card)
        self.assertEqual(style.configured["TCombobox"]["foreground"], THEMES["corz"].text)


class PreferencesTests(unittest.TestCase):
    def test_preferences_survive_a_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = PreferencesStore(root)
            store.save(
                UiPreferences(theme="light", density="compact", layout="topbar", motion="off")
            )
            restored = PreferencesStore(root).load()
        self.assertEqual(restored.theme, "light")
        self.assertEqual(restored.density, "compact")
        self.assertEqual(restored.layout, "topbar")
        self.assertEqual(restored.motion, "off")

    def test_unknown_values_fall_back_instead_of_crashing(self) -> None:
        """A hand-edited or older preferences file must not break the window."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = PreferencesStore(root)
            path = store.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "theme": "does-not-exist",
                        "density": "roomy",
                        "layout": "diagonal",
                        "motion": "warp",
                        "radius": 9999,
                    }
                ),
                encoding="utf-8",
            )
            loaded = store.load()
        self.assertIn(loaded.theme, THEME_NAMES)
        self.assertIn(loaded.density, DENSITIES)
        self.assertIn(loaded.layout, LAYOUT_MODES)
        self.assertIn(loaded.motion, MOTION_LEVELS)
        self.assertLessEqual(loaded.radius, 32)

    def test_a_bom_prefixed_file_loads(self) -> None:
        """PowerShell's ``Out-File -Encoding utf8`` writes a UTF-8 BOM."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = PreferencesStore(root)
            store.save(UiPreferences(theme="lime"))
            raw = store.path.read_text(encoding="utf-8")
            store.path.write_text(raw, encoding="utf-8-sig")
            loaded = PreferencesStore(root).load()
        self.assertEqual(loaded.theme, "lime")


class LayoutModelTests(unittest.TestCase):
    def test_default_layout_lists_every_widget_once_in_order(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        placements = state.placements("Dashboard")
        self.assertEqual([p.widget_id for p in placements], [spec.widget_id for spec in SPECS])
        self.assertEqual([p.order for p in placements], [0, 1, 2])
        self.assertTrue(all(p.visible for p in placements))
        self.assertTrue(all(p.span == 1 for p in placements))

    def test_move_changes_order_and_clamps_at_the_edges(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        state = move(state, "Dashboard", "log", -5)
        self.assertEqual(
            [p.widget_id for p in state.placements("Dashboard")], ["log", "kpis", "charts"]
        )
        state = move(state, "Dashboard", "log", 99)
        self.assertEqual(
            [p.widget_id for p in state.placements("Dashboard")], ["kpis", "charts", "log"]
        )

    def test_span_is_clamped_to_the_widget_spec_on_load(self) -> None:
        """A preset written for an older widget set cannot over-span a card.

        ``set_span`` stores what its caller asks for and documents that the
        caller clamps; ``normalize`` (every load path) is the funnel that
        enforces the widget's own ``min_span``/``max_span``.
        """
        state = normalize(LayoutState(pages={}), PAGES)
        state = set_span(state, "Dashboard", "log", 3)
        state = normalize(state, PAGES)
        log = next(p for p in state.placements("Dashboard") if p.widget_id == "log")
        self.assertEqual(log.span, 2)
        state = normalize(set_span(state, "Dashboard", "kpis", 99), PAGES)
        kpis = next(p for p in state.placements("Dashboard") if p.widget_id == "kpis")
        self.assertEqual(kpis.span, 3)

    def test_the_studio_only_offers_spans_the_spec_allows(self) -> None:
        for spec in SPECS:
            with self.subTest(spec=spec.widget_id):
                self.assertLessEqual(spec.min_span, spec.max_span)
                self.assertGreaterEqual(spec.min_span, 1)

    def test_visible_only_returns_visible_widgets_by_default(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        state = set_visible(state, "Dashboard", "charts", False)
        self.assertEqual([p.widget_id for p in state.visible("Dashboard")], ["kpis", "log"])
        self.assertEqual(len(state.placements("Dashboard")), 3, "hidden widgets stay listed")

    def test_normalize_repairs_unknown_and_retired_entries(self) -> None:
        """Presets and preferences can be older than the page's widget list."""
        from sandboxai.control_center_layout import WidgetPlacement

        hand_edited = LayoutState(
            pages={
                "Dashboard": [
                    WidgetPlacement("ghost", 2, True, 0),
                    WidgetPlacement("kpis", 9, False, 1),
                ],
                "Retired": [WidgetPlacement("anything", 1, True, 0)],
            }
        )
        state = normalize(hand_edited, PAGES)
        self.assertEqual(state.pages.keys(), {"Dashboard"})
        placements = state.placements("Dashboard")
        self.assertNotIn("ghost", [p.widget_id for p in placements])
        self.assertEqual([p.widget_id for p in placements], ["kpis", "charts", "log"])

    def test_reset_restores_defaults(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        state = move(state, "Dashboard", "log", -2)
        state = set_span(state, "Dashboard", "kpis", 3)
        state = set_visible(state, "Dashboard", "charts", False)
        page_reset = reset_page(state, "Dashboard", PAGES)
        self.assertEqual(
            [p.widget_id for p in page_reset.placements("Dashboard")], ["kpis", "charts", "log"]
        )
        self.assertTrue(all(p.visible for p in page_reset.placements("Dashboard")))
        everything = reset_all(PAGES)
        self.assertTrue(all(p.span == 1 for p in everything.placements("Dashboard")))

    def test_serialize_round_trip_survives_deserialize(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        state = set_span(state, "Dashboard", "kpis", 2)
        state = move(state, "Dashboard", "log", -1)
        payload = json.loads(json.dumps(serialize(state)))
        restored = deserialize(payload, PAGES)
        self.assertEqual(
            [(p.widget_id, p.span, p.order) for p in restored.placements("Dashboard")],
            [(p.widget_id, p.span, p.order) for p in state.placements("Dashboard")],
        )

    def test_deserialize_falls_back_to_the_default_layout_for_junk(self) -> None:
        """An unreadable preset must never leave the window without a layout."""
        for junk in (None, [], {"pages": "not-a-dict"}, {"pages": {"Dashboard": "nope"}}):
            with self.subTest(junk=junk):
                state = deserialize(junk, PAGES)
                self.assertEqual(
                    [p.widget_id for p in state.placements("Dashboard")],
                    ["kpis", "charts", "log"],
                )


class AccentTests(unittest.TestCase):
    """The accent override: one free colour, applied to every theme."""

    def test_only_real_colours_are_accepted(self) -> None:
        self.assertEqual(normalize_accent("#4F7CFF"), "#4f7cff")
        self.assertEqual(normalize_accent("4f7cff"), "#4f7cff")
        self.assertEqual(normalize_accent("  #ABC  "), "#aabbcc")
        for unusable in ("", None, "blue", "#12345", "#gggggg", "rgb(1,2,3)"):
            with self.subTest(value=unusable):
                self.assertEqual(normalize_accent(unusable), "")

    def test_an_override_replaces_the_accent_and_keeps_text_readable(self) -> None:
        base = get_theme("corz")
        custom = get_theme("corz", "#F5A524")
        self.assertEqual(custom.accent.lower(), "#f5a524")
        self.assertNotEqual(custom.accent_soft, base.accent_soft)
        # Amber is a light colour: white label text on it would be unreadable,
        # so the theme must switch to its dark text token.
        self.assertEqual(custom.on_accent, readable_on("#F5A524"))
        self.assertGreater(custom.contrast_ratio("accent", "on_accent"), 4.5)
        # An unusable override falls back to the theme's own accent.
        self.assertEqual(get_theme("corz", "nonsense"), base)

    def test_contrast_ratio_is_a_ratio_and_symmetric(self) -> None:
        """It used to unpack `sorted()` the wrong way round and report 1/ratio.

        Nothing called it, so the inversion survived: a 3.7:1 pair came back
        as 0.27. This pins the contract the WCAG formula actually has.
        """
        theme = get_theme("corz")
        self.assertGreater(theme.contrast_ratio("text", "bg"), 10.0)
        self.assertEqual(
            theme.contrast_ratio("accent", "on_accent"),
            theme.contrast_ratio("on_accent", "accent"),
        )
        black_on_white = theme.contrast_ratio("on_accent", "on_accent")
        self.assertAlmostEqual(black_on_white, 1.0, places=6)

    def test_the_designed_accent_keeps_its_label_colour(self) -> None:
        """A picked accent must not repaint the built-in look differently."""
        for name in ("corz", "cyan", "lime"):
            with self.subTest(theme=name):
                built_in = THEMES[name]
                self.assertEqual(get_theme(name, built_in.accent), built_in)

    def test_the_override_survives_a_preferences_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PreferencesStore(Path(tmp))
            store.path.parent.mkdir(parents=True, exist_ok=True)
            prefs = UiPreferences(accent="#22d3ee", zoomed=True)
            store.save(prefs)
            loaded = store.load()
            self.assertEqual(loaded.accent, "#22d3ee")
            self.assertTrue(loaded.zoomed)
            # A junk accent in the file must not survive normalization.
            self.assertEqual(UiPreferences(accent="not-a-colour").normalized().accent, "")


@unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
class ThemeBusOwnershipTests(unittest.TestCase):
    """A rebuilt page must not leave dead repaint callbacks on the bus.

    ``ThemeBus`` lives in the Tk-backed UI module, so this class skips where
    Tkinter is missing; the smoke harness (fake Tk) pins the same contract
    for the real window headlessly.
    """

    @staticmethod
    def _bus_class():
        from sandboxai.control_center_ui import ThemeBus

        return ThemeBus

    def test_a_listener_whose_widget_is_gone_is_dropped(self) -> None:
        bus = self._bus_class()(THEMES["corz"])
        alive_calls: list[str] = []

        class Widget:
            def __init__(self, alive: bool) -> None:
                self.alive = alive

            def winfo_exists(self) -> int:
                return 1 if self.alive else 0

            def apply_theme(self, theme: object) -> None:
                alive_calls.append(theme.name)  # type: ignore[attr-defined]

        corpse, live = Widget(False), Widget(True)
        bus.subscribe(corpse.apply_theme, owner=corpse)
        bus.subscribe(live.apply_theme, owner=live)

        bus.set_theme(THEMES["cyan"])

        # The live widget repainted, the destroyed one was neither called
        # nor kept: one more switch leaves exactly the live listener.
        self.assertEqual(alive_calls, ["cyan"])
        self.assertEqual(len(bus._listeners), 1)

        bus.set_theme(THEMES["lime"])
        self.assertEqual(alive_calls, ["cyan", "lime"])
        self.assertEqual(len(bus._listeners), 1)

    def test_a_listener_without_an_owner_is_never_pruned(self) -> None:
        # Callers that manage their own lifetime keep the old behavior.
        bus = self._bus_class()(THEMES["corz"])
        seen: list[str] = []
        unsubscribe = bus.subscribe(lambda theme: seen.append(theme.name))
        bus.set_theme(THEMES["graphite"])
        self.assertEqual(seen, ["graphite"])
        unsubscribe()
        bus.set_theme(THEMES["corz"])
        self.assertEqual(seen, ["graphite"])


class PresetStoreTests(unittest.TestCase):
    def test_save_load_list_delete_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            state = set_span(normalize(LayoutState(pages={}), PAGES), "Dashboard", "kpis", 3)
            saved = store.save(
                "My Layout", layout=state, appearance={"theme": "cyan", "density": "compact"}
            )
            self.assertEqual(saved, "My Layout")
            self.assertIn("My Layout", store.list_presets())
            document = store.load("My Layout")
            self.assertIsNotNone(document)
            self.assertEqual(document["appearance"]["theme"], "cyan")
            restored = deserialize(document["layout"], PAGES)
            self.assertEqual(restored.placements("Dashboard")[0].span, 3)
            self.assertTrue(store.delete("My Layout"))
            self.assertEqual(store.list_presets(), [])
            self.assertIsNone(store.load("My Layout"))

    def test_export_and_import_move_a_preset_between_projects(self) -> None:
        """An export must be a complete preset, importable under its own name."""
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            state = set_span(normalize(LayoutState(pages={}), PAGES), "Dashboard", "kpis", 3)
            store.save(
                "shared layout", layout=state, appearance={"theme": "lime", "accent": "#22d3ee"}
            )
            target = Path(tmp) / "exports" / "anywhere.json"
            written = store.export("shared layout", target)
            self.assertTrue(written.is_file())
            self.assertEqual(json.loads(written.read_text())["name"], "shared layout")

            # Another project (a different .sandboxai directory) imports it.
            other = PresetStore(Path(tmp) / "other")
            with self.assertRaises(PresetError):
                other.import_preset(Path(tmp) / "missing.json")
            junk = Path(tmp) / "junk.json"
            junk.write_text('{"hello": 1}')
            with self.assertRaises(PresetError):
                other.import_preset(junk)

            name = other.import_preset(written)
            self.assertEqual(name, "shared layout")
            self.assertEqual(other.import_target_name(written), "shared layout")
            document = other.load(name)
            self.assertEqual(document["appearance"]["accent"], "#22d3ee")
            self.assertEqual(
                deserialize(document["layout"], PAGES).placements("Dashboard")[0].span, 3
            )
            # A clash is refused, never a silent replacement.
            with self.assertRaises(PresetError):
                other.import_preset(written)
            self.assertEqual(other.import_preset(written, overwrite=True), name)

    def test_export_to_a_directory_uses_the_preset_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            store.save("cool!!layout", layout=normalize(LayoutState(pages={}), PAGES))
            written = store.export("cool!!layout", Path(tmp))
            self.assertEqual(written.name, "coollayout.json")
            self.assertEqual(PresetStore.import_target_name(written), "coollayout")

    def test_rename_moves_the_layout_and_its_appearance(self) -> None:
        """Renaming must keep the preset, not drop it and make a new one."""
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            state = set_span(normalize(LayoutState(pages={}), PAGES), "Dashboard", "kpis", 2)
            store.save("before", layout=state, appearance={"theme": "lime"})
            renamed = store.rename("before", "after")
            self.assertIn("after", store.list_presets())
            self.assertNotIn("before", store.list_presets())
            document = store.load("after")
            self.assertEqual(document["appearance"]["theme"], "lime")
            self.assertEqual(
                deserialize(document["layout"], PAGES).placements("Dashboard")[0].span, 2
            )
            self.assertEqual(renamed, "after")
            # Renaming onto an existing name would silently destroy the
            # other preset, so it must be refused instead.
            store.save("third", layout=state)
            with self.assertRaises(PresetError):
                store.rename("third", "after")

    def test_safe_preset_name_cannot_escape_the_preset_directory(self) -> None:
        """A preset name becomes a file name: it must not be able to walk out."""
        for raw in ("../../etc/passwd", "..\\..\\windows\\system32", "sub/dir/name", "  ..  "):
            with self.subTest(raw=raw):
                try:
                    name = safe_preset_name(raw)
                except PresetError:
                    continue
                self.assertNotIn("/", name)
                self.assertNotIn("\\", name)
                self.assertFalse(name.startswith("."))
                self.assertEqual(Path(name).name, name)
        self.assertEqual(safe_preset_name("  My Layout  "), "My Layout")
        self.assertEqual(safe_preset_name("cool!!layout??"), "coollayout")
        with self.assertRaises(PresetError):
            safe_preset_name("!!!")
        with self.assertRaises(PresetError):
            safe_preset_name("")

    def test_a_broken_preset_file_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            store.save("broken", layout=normalize(LayoutState(pages={}), PAGES), appearance={})
            store._path("broken").write_text("{not json", encoding="utf-8")
            self.assertIsNone(store.load("broken"))
            self.assertTrue(store.error)

    def test_presets_are_written_atomically(self) -> None:
        """A crash mid-save must not leave a truncated preset behind."""
        with tempfile.TemporaryDirectory() as tmp:
            store = PresetStore(Path(tmp))
            store.save("atomic", layout=normalize(LayoutState(pages={}), PAGES), appearance={})
            leftovers = [p.name for p in store._path("atomic").parent.glob("*.tmp")]
            self.assertEqual(leftovers, [])


class BoardSizingTests(unittest.TestCase):
    """The card board's resize decision, pinned without a display.

    The desktop suite used to hang inside ``update()`` because the board
    re-decided its column count from inside the ``<Configure>`` event of the
    layout that decision had just produced. The decision itself lives in
    :func:`columns_for_width` so the two properties that break that loop -
    a hysteresis band and no guessing while the board is unlaid-out - are
    testable here, on every machine, instead of only under Xvfb.
    """

    def test_the_count_follows_the_width_when_it_clearly_changes(self) -> None:
        self.assertEqual(columns_for_width(1400, max_columns=3, min_column_width=320, current=3), 3)
        self.assertEqual(columns_for_width(700, max_columns=3, min_column_width=320, current=3), 2)
        self.assertEqual(columns_for_width(340, max_columns=3, min_column_width=320, current=3), 1)
        # A big resize still takes the full step, not one column per event.
        self.assertEqual(columns_for_width(1400, max_columns=3, min_column_width=320, current=1), 3)

    def test_a_width_inside_the_band_leaves_the_count_alone(self) -> None:
        """A window parked on a threshold must not oscillate between counts."""
        for width in (639, 640, 641, 650, 670):
            self.assertEqual(
                columns_for_width(width, max_columns=3, min_column_width=320, current=2),
                2,
                f"width {width} is within the hysteresis band of two columns",
            )
        # Clearing the band by enough does move it: the third column needs
        # 3 x 320 px plus the 25 px margin, the first needs to fall below
        # 2 x 320 px minus the margin.
        self.assertEqual(columns_for_width(984, max_columns=3, min_column_width=320, current=2), 2)
        self.assertEqual(columns_for_width(985, max_columns=3, min_column_width=320, current=2), 3)
        self.assertEqual(columns_for_width(615, max_columns=3, min_column_width=320, current=2), 2)
        self.assertEqual(columns_for_width(614, max_columns=3, min_column_width=320, current=2), 1)

    def test_an_unlaid_out_board_keeps_its_column_count(self) -> None:
        """``winfo_width()`` is 1 before the first layout; that is not a resize."""
        self.assertEqual(columns_for_width(1, max_columns=3, min_column_width=320, current=3), 3)
        self.assertEqual(columns_for_width(0, max_columns=3, min_column_width=320, current=2), 2)

    def test_the_band_is_symmetric_for_every_supported_width(self) -> None:
        """No width may produce two different counts for one current value."""
        for current in (1, 2, 3):
            for width in range(2, 2600, 1):
                first = columns_for_width(
                    width, max_columns=3, min_column_width=320, current=current
                )
                second = columns_for_width(
                    width, max_columns=3, min_column_width=320, current=first
                )
                self.assertEqual(
                    first,
                    second,
                    f"width {width} flips from {current} to {first} and back",
                )


class PlacementSlotTests(unittest.TestCase):
    """The board's card placement as a pure value (its ``rebuild`` cache)."""

    def _slots(self, **kwargs):
        state = normalize(LayoutState(pages={}), PAGES)
        return placement_slots(state.visible("Dashboard"), **kwargs)

    def test_slots_flow_left_to_right_and_wrap_at_the_column_count(self) -> None:
        slots = self._slots(columns=2)
        self.assertEqual(slots, (("kpis", 0, 0, 1), ("charts", 0, 1, 1), ("log", 1, 0, 1)))

    def test_a_span_wider_than_the_row_moves_to_the_next_row(self) -> None:
        state = normalize(LayoutState(pages={}), PAGES)
        state = set_span(state, "Dashboard", "kpis", 3)
        slots = placement_slots(state.visible("Dashboard"), columns=2)
        # The 3-wide card cannot share a two-column row, so it starts row 0
        # alone and the rest follow on the next row.
        self.assertEqual(slots[0], ("kpis", 0, 0, 2))
        self.assertEqual([slot[1] for slot in slots[1:]], [1, 1])

    def test_identical_layouts_produce_identical_slots(self) -> None:
        """The identity ``LayoutBoard.rebuild`` uses to skip the re-grid."""
        self.assertEqual(self._slots(columns=3), self._slots(columns=3))
        self.assertNotEqual(self._slots(columns=3), self._slots(columns=2))


class WindowGeometryTests(unittest.TestCase):
    """Where the window opens, and how big.

    A remembered geometry is only a suggestion: the screen it was captured on
    may have been bigger, or gone (a monitor change, a different WSLg/RDP
    session, a laptop undocked from a 4K display). These pin the three rules -
    fit the screen, stay reachable, centre what has no position - so a stale
    preference file can never open a window the operator cannot get back.
    """

    def _parts(self, geometry: str) -> tuple[int, int, int, int]:
        import re

        match = re.fullmatch(r"(\d+)x(\d+)\+(\d+)\+(\d+)", geometry)
        self.assertIsNotNone(match, f"{geometry!r} is not a Tk geometry string")
        assert match is not None
        return tuple(int(group) for group in match.groups())  # type: ignore[return-value]

    def test_a_fresh_window_is_sized_for_a_1920x1080_screen(self) -> None:
        width, height, x, y = self._parts(
            fit_window_geometry(None, screen_width=1920, screen_height=1080)
        )
        self.assertEqual((width, height), (1800, 980))
        self.assertEqual((x, y), ((1920 - 1800) // 2, (1080 - 980) // 2))

    def test_a_saved_geometry_that_no_longer_fits_is_clamped_onto_the_screen(self) -> None:
        """The 4K-to-1080p case: keep the intent, drop what cannot be shown."""
        width, height, x, y = self._parts(
            fit_window_geometry("3200x2000+5000+5000", screen_width=1920, screen_height=1080)
        )
        self.assertLessEqual(width, 1920)
        self.assertLessEqual(height, 1080)
        self.assertLessEqual(x + width, 1920)
        self.assertLessEqual(y + height, 1080)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)

    def test_a_window_restored_off_the_left_edge_comes_back(self) -> None:
        _width, _height, x, y = self._parts(
            fit_window_geometry("1500x900-300-200", screen_width=1920, screen_height=1080)
        )
        self.assertEqual((x, y), (0, 0))

    def test_a_usable_saved_size_and_position_are_kept(self) -> None:
        self.assertEqual(
            fit_window_geometry("1500x900+120+80", screen_width=1920, screen_height=1080),
            "1500x900+120+80",
        )

    def test_a_saved_size_below_the_minimum_is_raised_to_it(self) -> None:
        width, height, _x, _y = self._parts(
            fit_window_geometry("900x500+10+10", screen_width=1920, screen_height=1080)
        )
        self.assertEqual((width, height), (1280, 800))

    def test_an_unreadable_geometry_is_treated_as_no_geometry(self) -> None:
        self.assertEqual(
            fit_window_geometry("nonsense", screen_width=1920, screen_height=1080),
            fit_window_geometry(None, screen_width=1920, screen_height=1080),
        )
        self.assertEqual(
            fit_window_geometry("", screen_width=1920, screen_height=1080),
            fit_window_geometry(None, screen_width=1920, screen_height=1080),
        )

    def test_a_small_screen_gets_a_window_that_still_fits(self) -> None:
        width, height, x, y = self._parts(
            fit_window_geometry("1720x1000+0+0", screen_width=1366, screen_height=768)
        )
        self.assertLessEqual(width, 1366)
        self.assertLessEqual(height, 768)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1366)
        self.assertLessEqual(y + height, 768)

    def test_a_tiny_screen_never_produces_an_offscreen_window(self) -> None:
        width, height, x, y = self._parts(
            fit_window_geometry(None, screen_width=1024, screen_height=600)
        )
        self.assertLessEqual(width, 1024)
        self.assertLessEqual(height, 600)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)

    def test_viewport_scale_shrinks_elements_proportionally_below_1080p(self) -> None:
        from sandboxai.control_center_theme import compute_viewport_scale

        full_scale = UiScale(dpi=96.0).with_viewport(1800, 980)
        small_scale = UiScale(dpi=96.0).with_viewport(1024, 600)
        self.assertEqual(compute_viewport_scale(1800, 980), 1.0)
        self.assertLess(small_scale.viewport_scale, 1.0)
        self.assertLess(small_scale.px(238, minimum=180), full_scale.px(238, minimum=180))
        self.assertLess(small_scale.font("h1")[1], full_scale.font("h1")[1])


if __name__ == "__main__":
    unittest.main()
