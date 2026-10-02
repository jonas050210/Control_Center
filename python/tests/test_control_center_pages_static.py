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
* the old Agents page is gone and no page resurrects a "Tests" tab.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from optional_deps import HAS_TKINTER, TKINTER_REASON

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sandboxai"
PAGES_SOURCE = PACKAGE / "control_center_pages.py"

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
                self.assertEqual(ids, sorted(set(ids)), f"{title} lists a widget twice")
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


if __name__ == "__main__":
    unittest.main()
