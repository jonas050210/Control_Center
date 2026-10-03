"""Static contract checks for the reworked Control Center pages.

The Tk suite can only run where Tkinter *and* a display exist (the CI
``desktop-ui-tests`` job); on a headless machine it skips. These checks
parse the page module instead, so a broken wiring between a page and its
layout board fails everywhere - including on the machine of whoever edits
the file next.

What is pinned:

* every exported page class declares the API the shell calls,
* a page that declares movable widgets actually builds a board and
  registers exactly the widgets it declared (a spec without a factory
  would render an empty cell; a factory without a spec raises),
* page titles are unique and ``PAGE_WIDGETS`` only mentions real pages,
* the old Agents page is gone and no page resurrects a "Tests" tab,
* a label names what it shows (``steps/s``, not ``fps``) and a control
  lives in the one page that owns it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from optional_deps import HAS_TKINTER, TKINTER_REASON

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sandboxai"
PAGES_SOURCE = PACKAGE / "control_center_pages.py"

#: Every module that builds real Tk widgets.
GUI_MODULES = (
    PACKAGE / "control_center_desktop.py",
    PACKAGE / "control_center_pages.py",
    PACKAGE / "control_center_ui.py",
    PACKAGE / "control_center_widgets.py",
)

#: Names every Tk widget already carries from ``Misc``/``BaseWidget``. An
#: instance attribute with one of these names shadows Tk's own machinery,
#: and that failure only surfaces on a machine with a real Tk:
#: ``self._options = (...)`` turned ``Canvas.__init__`` into
#: ``TypeError: 'tuple' object is not callable`` *inside tkinter*, which the
#: headless suite never saw. This check runs everywhere.
TKINTER_INTERNALS = frozenset(
    {
        "_bind",
        "_configure",
        "_displayof",
        "_getboolean",
        "_getconfigure",
        "_getdoubles",
        "_getints",
        "_grid_configure",
        "_last_child_ids",
        "_name",
        "_nametowidget",
        "_options",
        "_register",
        "_report_exception",
        "_root",
        "_setup",
        "_subst_format",
        "_subst_format_dyn",
        "_tclCommands",
        "_w",
        "_windowingsystem",
        "children",
        "master",
        "tk",
        "widgetName",
    }
)

#: The shell (``control_center_desktop.ControlCenter``) calls these.
PAGE_API = (
    "build",
    "refresh",
    "on_theme",
    "rebuild_after_restyle",
    "reset_polls",
    "show",
)


def _module_tree() -> ast.Module:
    return ast.parse(PAGES_SOURCE.read_text(encoding="utf-8"))


def _page_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    classes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Page"):
            classes[node.name] = node
    return classes


def _class_attr(node: ast.ClassDef, name: str) -> ast.expr | None:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            if isinstance(target, ast.Name) and target.id == name and statement.value is not None:
                return statement.value
    return None


def _find_method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    return None


def _declared_widget_ids(node: ast.ClassDef) -> list[str] | None:
    """The ``widgets`` tuple's ids, or ``None`` when the page declares none."""
    value = _class_attr(node, "widgets")
    if value is None:
        return None
    if not isinstance(value, ast.Tuple):
        raise AssertionError(f"{node.name}.widgets must be a tuple of WidgetSpec entries")
    ids: list[str] = []
    for element in value.elts:
        if not isinstance(element, ast.Call):
            raise AssertionError(f"{node.name}.widgets entries must be WidgetSpec(...) calls")
        widget_id: str | None = None
        if element.args and isinstance(element.args[0], ast.Constant):
            widget_id = str(element.args[0].value)
        for keyword in element.keywords:
            if keyword.arg == "widget_id" and isinstance(keyword.value, ast.Constant):
                widget_id = str(keyword.value.value)
        if widget_id is None:
            raise AssertionError(f"{node.name}.widgets entry without a widget id")
        ids.append(widget_id)
    return ids


def _registered_widget_ids(node: ast.ClassDef) -> set[str]:
    """Every ``board.add("id", ...)`` / ``self.board(...)`` call in build()."""
    build = _find_method(node, "build")
    if build is None:
        return set()
    ids: set[str] = set()
    for sub in ast.walk(build):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if (
            sub.func.attr == "add"
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and isinstance(sub.args[0].value, str)
        ):
            ids.add(sub.args[0].value)
    return ids


def _board_is_attached(node: ast.ClassDef) -> bool:
    """True when the board ``Page.board()`` builds is handed to a manager.

    A board that is created and filled but never packed/gridded renders
    nothing: every card ends up inside an unmapped frame, which is exactly
    how the Dashboard/Training/Benchmarks pages shipped - heading, then a
    blank page.
    """
    method = _find_method(node, "board")
    if method is None:
        return False
    for sub in ast.walk(method):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        target = sub.func.value
        if (
            sub.func.attr in {"pack", "grid", "place"}
            and isinstance(target, ast.Name)
            and target.id == "board"
        ):
            return True
    return False


def _uses_a_board(node: ast.ClassDef) -> bool:
    build = _find_method(node, "build")
    if build is None:
        return False
    for sub in ast.walk(build):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"board", "board_columns"}
        ):
            return True
    return False


class PageContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _module_tree()
        self.classes = _page_classes(self.tree)

    def test_expected_pages_exist_and_the_agents_page_is_gone(self) -> None:
        self.assertIn("TrainingPage", self.classes)
        self.assertNotIn("AgentsPage", self.classes)
        titles = {
            name: _class_attr(node, "title").value  # type: ignore[union-attr]
            for name, node in self.classes.items()
            if _class_attr(node, "title") is not None
        }
        self.assertEqual(titles["TrainingPage"], "Training")
        self.assertNotIn("Agents", set(titles.values()))

    def test_no_page_declares_a_tests_tab(self) -> None:
        """A Tests tab was explicitly rejected; no page may add one back."""
        titles = [
            str(_class_attr(node, "title").value)  # type: ignore[union-attr]
            for node in self.classes.values()
            if _class_attr(node, "title") is not None
        ]
        self.assertNotIn("Tests", titles)
        self.assertNotIn("Test", titles)

    def test_every_page_implements_the_shell_api_and_a_subtitle(self) -> None:
        for name, node in self.classes.items():
            if name == "Page":
                continue
            with self.subTest(page=name):
                self.assertIsNotNone(_class_attr(node, "title"), f"{name} needs a title")
                self.assertIsNotNone(_class_attr(node, "subtitle"), f"{name} needs a subtitle")
                self.assertIsNotNone(_find_method(node, "build"), f"{name} needs build()")
                for method in PAGE_API:
                    # Provided by the shared base class, overridden or not.
                    self.assertTrue(
                        _find_method(self.classes["Page"], method) is not None
                        or _find_method(node, method) is not None,
                        f"neither {name} nor Page defines {method}()",
                    )

    def test_declared_widgets_are_registered_on_the_pages_board(self) -> None:
        checked = 0
        for name, node in self.classes.items():
            declared = _declared_widget_ids(node)
            if not declared:
                continue
            checked += 1
            with self.subTest(page=name):
                self.assertEqual(
                    len(declared), len(set(declared)), f"{name} declares a widget twice"
                )
                registered = _registered_widget_ids(node)
                self.assertEqual(
                    set(declared),
                    registered,
                    f"{name}: declared {sorted(set(declared))} but registered "
                    f"{sorted(registered)} on its layout board",
                )
                self.assertTrue(_uses_a_board(node), f"{name} declares widgets but builds no board")
        self.assertGreaterEqual(checked, 3, "the movable-widget pages disappeared")

    def test_the_shared_board_is_attached_to_its_page(self) -> None:
        """``Page.board()`` must give the board a geometry manager.

        The pages built and filled their board but never attached it, so the
        three movable pages (Dashboard, Training, Benchmarks) rendered their
        heading and then nothing. This is the one regression the real-Tk
        suite could not report on a machine without Tk, so it is pinned here.
        """
        self.assertTrue(
            _board_is_attached(self.classes["Page"]),
            "Page.board() must pack/grid/place the board it creates",
        )

    def test_page_titles_are_unique(self) -> None:
        titles = [
            str(_class_attr(node, "title").value)  # type: ignore[union-attr]
            for node in self.classes.values()
            if _class_attr(node, "title") is not None
        ]
        self.assertEqual(len(titles), len(set(titles)))

    @unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
    def test_page_widgets_registry_matches_the_page_classes(self) -> None:
        """The Settings layout studio iterates ``PAGE_WIDGETS``."""
        import sandboxai.control_center_pages as pages_module

        registry = pages_module.PAGE_WIDGETS
        titles = {page_class.title for page_class in pages_module.PAGE_CLASSES}
        self.assertTrue(set(registry), "PAGE_WIDGETS is empty")
        self.assertTrue(set(registry) <= titles, "PAGE_WIDGETS names a page that does not exist")
        for title, specs in registry.items():
            with self.subTest(page=title):
                self.assertTrue(specs, f"{title} is registered with no widgets")
                ids = [spec.widget_id for spec in specs]
                self.assertEqual(len(ids), len(set(ids)), f"{title} lists a widget twice: {ids}")
                for spec in specs:
                    self.assertLessEqual(spec.min_span, spec.max_span)
                    self.assertLessEqual(spec.default_span, spec.max_span)
                    self.assertGreaterEqual(spec.default_span, spec.min_span)

    @unittest.skipUnless(HAS_TKINTER, TKINTER_REASON)
    def test_pages_sharing_a_title_with_the_registry_declare_widgets(self) -> None:
        import sandboxai.control_center_pages as pages_module

        for page_class in pages_module.PAGE_CLASSES:
            with self.subTest(page=page_class.title):
                if page_class.title in pages_module.PAGE_WIDGETS:
                    self.assertTrue(page_class.widgets, "registry entry without class widgets")
                else:
                    self.assertEqual(
                        page_class.widgets,
                        (),
                        f"{page_class.title} declares widgets but is not in PAGE_WIDGETS, "
                        "so the layout studio cannot show them",
                    )


#: ttk style names look like ``FieldTitle.TLabel``; anything ending in one
#: of these suffixes is a style reference, no matter how many dots precede it.
_STYLE_SUFFIXES = (
    "TLabel",
    "TFrame",
    "TButton",
    "TCheckbutton",
    "TCombobox",
    "TEntry",
    "Treeview",
    "TScrollbar",
    "TSeparator",
    "TPanedwindow",
    "TRadiobutton",
    "TProgressbar",
    "TNotebook",
    "TScale",
    "TLabelFrame",
    "TSpinbox",
    "TMenubutton",
)

_THEME = "control_center_theme.py"


class PageSurfaceTests(unittest.TestCase):
    """Labels and controls the operator reads, and where they belong.

    Both failures here were real: a dashboard tile labelled *fps* showed the
    trainer's steps/s, and the Roblox card carried a "CPU turbo" button that
    Settings already owned. Neither is a wiring error, so nothing else in
    this file would notice them.
    """

    def setUp(self) -> None:
        self.tree = _module_tree()
        self.classes = _page_classes(self.tree)

    def _class_constants(self, name: str, attribute: str) -> list[str]:
        node = self.classes[name]
        assigned = _class_attr(node, attribute)
        assert assigned is not None, f"{name}.{attribute} is gone"
        return [str(item.value) for item in assigned.elts]  # type: ignore[union-attr]

    def test_the_dashboard_names_throughput_steps_per_second(self) -> None:
        labels = self._class_constants("DashboardPage", "STAT_LABELS")
        self.assertIn(
            "steps/s",
            labels,
            "the dashboard tile shows the trainer's steps_per_second - calling it fps "
            "sets it next to the benchmark's FPS/env as if they were one quantity",
        )
        self.assertNotIn("fps", labels)

    def test_host_settings_live_on_settings_not_in_the_roblox_card(self) -> None:
        # The Roblox card is about the game client. A CPU-governor button there
        # was a second control for something Settings already owns - and the
        # duplication is what makes a page unreadable, not the extra click.
        self.assertIsNone(
            _find_method(self.classes["DashboardPage"], "_enable_ubuntu_cpu_turbo"),
            "CPU turbo is a host setting: it belongs to Settings, which shows the "
            "governor state beside it, not to the Roblox card",
        )
        self.assertIsNotNone(
            _find_method(self.classes["SettingsPage"], "_activate_ubuntu_cpu_turbo")
        )


_GUI_MODULES = (
    "control_center_desktop.py",
    "control_center_pages.py",
    "control_center_ui.py",
    "control_center_widgets.py",
)


def _style_name(value: str) -> bool:
    if "." not in value or value.startswith("."):
        return False
    return value.endswith(_STYLE_SUFFIXES) and " " not in value


def _style_strings(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _style_name(node.value)
    }


class StyleNameTests(unittest.TestCase):
    """Every ``style=...`` a widget asks for must be one the theme defines.

    ttk accepts an unknown style silently and paints the widget with the
    default look, so a renamed style is invisible in the code and nearly
    invisible on screen - but not to this check. It also keeps the theme
    module the single owner of the visual vocabulary.
    """

    def test_every_referenced_ttk_style_is_configured(self) -> None:
        defined = _style_strings(PACKAGE / _THEME)
        self.assertTrue(defined, "the theme module must define ttk styles")
        missing: dict[str, list[str]] = {}
        for name in _GUI_MODULES:
            for style in _style_strings(PACKAGE / name):
                if style not in defined:
                    missing.setdefault(style, []).append(name)
        self.assertEqual(
            missing,
            {},
            "styles referenced but not configured by the theme: " + repr(missing),
        )


class AdapterGuardTests(unittest.TestCase):
    """``hasattr(self.adapter, "x")`` must name a method the adapter has.

    The pages guard a few optional-looking adapter calls with ``hasattr``.
    A typo in that string turns the guard into a permanent ``False``: the
    button then does nothing, silently, on every machine - no traceback and
    no test failure, because the branch is simply never taken.
    """

    def setUp(self) -> None:
        self.tree = _module_tree()

    def test_guarded_adapter_attributes_exist(self) -> None:
        from sandboxai.adapter import SandboxAIAdapter

        guarded: list[str] = []
        for sub in ast.walk(self.tree):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
                continue
            if sub.func.id != "hasattr" or len(sub.args) != 2:
                continue
            target, name = sub.args
            if not (isinstance(target, ast.Attribute) and target.attr == "adapter"):
                continue
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                guarded.append(name.value)
        self.assertTrue(guarded, "the pages no longer guard any adapter call")
        missing = sorted(set(guarded) - set(dir(SandboxAIAdapter)))
        self.assertEqual(missing, [], f"guarded but missing on the adapter: {missing}")


class ShellApiTests(unittest.TestCase):
    """``self.app.<name>`` on a page must exist on the shell.

    The pages type their ``app`` as ``Any`` (it is the Tk shell), so mypy
    cannot see a renamed or mistyped shell attribute; a page that calls a
    method the shell no longer has only fails when a user opens that page.
    This compares the names the pages use against the names the shell
    defines - assignments, methods and class attributes - so the rename
    fails here instead.
    """

    SHELL = "control_center_desktop.py"

    def setUp(self) -> None:
        self.tree = _module_tree()

    def _used_by_pages(self) -> set[str]:
        used: set[str] = set()
        for sub in ast.walk(self.tree):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "app"
                and isinstance(sub.value.value, ast.Name)
                and sub.value.value.id == "self"
            ):
                used.add(sub.attr)
        return used

    def _defined_on_the_shell(self) -> set[str]:
        tree = ast.parse((PACKAGE / self.SHELL).read_text(encoding="utf-8"))
        control_center = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ControlCenter"
        )
        defined: set[str] = set()
        for node in control_center.body:
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and isinstance(sub.ctx, ast.Store)
                ):
                    defined.add(sub.attr)
        return defined

    def test_every_app_attribute_a_page_uses_exists(self) -> None:
        used = self._used_by_pages()
        defined = self._defined_on_the_shell()
        missing = sorted(used - defined)
        self.assertEqual(
            missing,
            [],
            "the pages call shell attributes that ControlCenter does not define: "
            + ", ".join(missing),
        )


class FontFloorTests(unittest.TestCase):
    """The 'too small' complaint must not creep back in as a literal font size."""

    def test_no_page_hard_codes_a_tiny_font(self) -> None:
        offenders: list[str] = []
        for path in sorted(PACKAGE.glob("control_center_*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "font=" not in line:
                    continue
                for size in range(6, 11):
                    if f", {size})" in line or f'"{size}"' in line or f", {size}," in line:
                        offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "font sizes below the 11 px floor belong in the theme, not in a widget: "
            + "; ".join(offenders),
        )


def _base_names(node: ast.ClassDef) -> list[str]:
    """Base classes as written: ``tk.Canvas`` stays qualified, ``Page`` bare."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            names.append(f"{base.value.id}.{base.attr}")
        elif isinstance(base, ast.Name):
            names.append(base.id)
    return names


def _tk_widget_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """Classes that inherit from a ``tk.``/``ttk.`` widget, transitively."""
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    widgets = {
        name
        for name, node in classes.items()
        if any(base.startswith(("tk.", "ttk.")) for base in _base_names(node))
    }
    growing = True
    while growing:
        growing = False
        for name, node in classes.items():
            if name in widgets or not any(base in widgets for base in _base_names(node)):
                continue
            widgets.add(name)
            growing = True
    return {name: classes[name] for name in widgets}


def _assigned_self_attributes(node: ast.ClassDef) -> set[str]:
    """Every ``self.<name> = ...`` inside a class body."""
    names: set[str] = set()
    for statement in ast.walk(node):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return names


class TkInternalsShadowTests(unittest.TestCase):
    """A widget subclass must not take over names Tk puts on every widget."""

    def test_widget_subclasses_leave_tkinter_internals_alone(self) -> None:
        offenders: list[str] = []
        for path in GUI_MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name, node in _tk_widget_classes(tree).items():
                for attribute in sorted(_assigned_self_attributes(node) & TKINTER_INTERNALS):
                    offenders.append(f"{path.name}:{node.lineno}: {name}.{attribute}")
        self.assertEqual(
            offenders,
            [],
            "these instance attributes shadow names tkinter already puts on every "
            "widget; they only fail on a real Tk build: " + "; ".join(offenders),
        )


class GeometryFeedbackTests(unittest.TestCase):
    """Source-level guards against the geometry loop that froze the desktop suite.

    The real-Tk suite caught the *symptom* (the window resizing itself inside
    ``update()``, which only returns when the event queue drains). These
    checks pin the two structures that caused it, so a future edit cannot
    quietly reintroduce one on a machine where the Tk suite only skips:

    * a ``ttk.Panedwindow`` re-arranges its panes whenever a child reports a
      new requested size - with a table that fits its columns to the width it
      was given, that is a feedback loop. The Control Center has no use for a
      movable sash: it lays its split cards out with ``grid`` weights.
    * the card surface must be a *decoration*. ``RoundedPanel`` paints its
      rounded surface on a placed canvas behind a content-sized frame; the
      earlier version hosted the content as a canvas window item and derived
      its own height from the body on every ``<Configure>``, which is the
      loop itself.
    """

    def test_no_page_builds_a_panedwindow(self) -> None:
        offenders: list[str] = []
        for path in GUI_MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or node.attr != "Panedwindow":
                    continue
                if isinstance(node.value, ast.Name) and node.value.id in {"tk", "ttk"}:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "a ttk.Panedwindow re-arranges its panes on every requested-size change "
            "of a child and loops with self-sizing content; use grid weights instead: "
            + ", ".join(offenders),
        )

    def test_the_card_surface_is_a_decoration_not_the_container(self) -> None:
        source = (PACKAGE / "control_center_ui.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        panel = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "RoundedPanel"
            ),
            None,
        )
        self.assertIsNotNone(panel, "RoundedPanel is gone")
        assert panel is not None
        bases = _base_names(panel)
        self.assertIn(
            "tk.Frame",
            bases,
            "RoundedPanel must stay a frame whose size comes from its content",
        )
        self.assertNotIn("tk.Canvas", bases, "a canvas card re-derives its own size")
        body = ast.get_source_segment(source, panel) or ""
        self.assertNotIn(
            "itemconfigure",
            body,
            "the card must not resize a hosted window item from its own geometry",
        )
        self.assertNotIn(
            "winfo_reqheight",
            body,
            "the card must not derive a requested height from its body's request",
        )

    def test_every_table_explains_its_empty_state(self) -> None:
        """A table with only headings reads like a broken one."""
        tree = _module_tree()
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "_scrollable_table":
                continue
            if not any(keyword.arg == "empty_text" for keyword in node.keywords):
                offenders.append(f"line {node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "every _scrollable_table(...) call must pass empty_text= (a page with no "
            "data has to say why the table is empty): " + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
