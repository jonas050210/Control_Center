"""Drift tests between the code and everything that documents it.

Versions in this repository have exactly one source of truth each, and
those sources are named in CONTRIBUTING.md:

* the engine version lives in ``sandboxai.contract.GODOT_VERSION``
* the package version lives in ``pyproject.toml``

Prose, CI workflows and ``project.godot`` all repeat those numbers, and a
repeated number drifts. These tests are the reason the repetition is
safe: change the source of truth, and whatever still disagrees fails
here with the file name in the message.
"""

from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path

from sandboxai import __version__
from sandboxai.contract import GODOT_VERSION, OBSERVATION_FIELD_COUNT

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY_ROOT / "docs"
PACKAGE = REPOSITORY_ROOT / "python" / "sandboxai"
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"

# Files that quote the engine version in prose or configuration. Each one
# is read as text and every "Godot <x.y.z>" mention in it must agree with
# contract.GODOT_VERSION.
ENGINE_VERSION_DOCUMENTS = (
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "PROJECT.md",
    "docs/ARCHITECTURE.md",
    "docs/CONTROL_CENTER.md",
    "docs/RUN_LOCAL_VALIDATION.md",
)

_GODOT_MENTION = re.compile(r"Godot[\s_v]+(\d+\.\d+\.\d+)")

# Living documents and comments that restate the observation width in
# prose. Historical reports (AUDIT_REPORT.md, RESEARCH_SANDBOX_REPORT.md) are
# deliberately absent: they describe the state of a past revision.
OBSERVATION_DIMENSION_DOCUMENTS = (
    "README.md",
    "PROJECT.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md",
    "docs/CURRICULUM_AND_COMBAT.md",
    "docs/DEBUG_GUI_AND_BENCHMARKING.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    # Code comments that quote the width; a stale one is just as misleading
    # as a stale README line.
    "scripts/core/simulation_manager.gd",
    "scripts/weapon/weapon_state.gd",
    "tools/control_center_smoke.py",
)

#: Every way the width is spelled in those files.
_OBSERVATION_WIDTH_MENTIONS = (
    re.compile(r"(\d+)-float"),
    re.compile(r"(\d+)\s+\**float"),
    re.compile(r"(\d+)-field"),
    re.compile(r"(\d+)\s*(?:->|\u2192)\s*128\s*(?:->|\u2192)\s*128"),
)


def _read(relative: str) -> str:
    return (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")


class EngineVersionTests(unittest.TestCase):
    """``contract.GODOT_VERSION`` is the only place the engine version lives."""

    def test_the_contract_version_is_a_full_three_part_version(self) -> None:
        self.assertRegex(GODOT_VERSION, r"^\d+\.\d+\.\d+$")

    def test_the_godot_ci_workflow_downloads_the_contract_version(self) -> None:
        workflow = (WORKFLOWS / "godot-tests.yml").read_text(encoding="utf-8")
        declared = re.search(r'^\s*GODOT_VERSION:\s*"([^"]+)"', workflow, re.MULTILINE)
        self.assertIsNotNone(declared, "godot-tests.yml declares no GODOT_VERSION")
        assert declared is not None  # narrowing for type checkers
        self.assertEqual(
            declared.group(1),
            GODOT_VERSION,
            "CI would download a different engine than the contract pins",
        )

    def test_project_godot_targets_the_same_feature_version(self) -> None:
        project = _read("project.godot")
        features = re.search(r"config/features\s*=\s*PackedStringArray\(([^)]*)\)", project)
        self.assertIsNotNone(features, "project.godot declares no config/features")
        assert features is not None  # narrowing for type checkers
        major_minor = ".".join(GODOT_VERSION.split(".")[:2])
        self.assertIn(
            f'"{major_minor}"',
            features.group(1),
            "project.godot's feature version does not match contract.GODOT_VERSION",
        )

    def test_the_documented_engine_version_never_drifts(self) -> None:
        for relative in ENGINE_VERSION_DOCUMENTS:
            with self.subTest(document=relative):
                mentions = set(_GODOT_MENTION.findall(_read(relative)))
                self.assertTrue(mentions, f"{relative} no longer names a Godot version")
                self.assertEqual(
                    mentions,
                    {GODOT_VERSION},
                    f"{relative} names an engine version other than {GODOT_VERSION}",
                )

    def test_the_probed_executable_names_include_the_engine_version(self) -> None:
        from sandboxai.config import _EXACT_VERSION_CANDIDATES

        self.assertTrue(_EXACT_VERSION_CANDIDATES)
        for name in _EXACT_VERSION_CANDIDATES:
            with self.subTest(executable=name):
                self.assertIn(GODOT_VERSION, name)


class ObservationDimensionTests(unittest.TestCase):
    """Prose must repeat the observation width, not invent its own.

    The v4 object block changed the width from 84 to 106 and nine living
    documents still claimed 84. ``AGENTS.md`` asks that a number written
    into prose be checkable, so this test is the check: every
    "<N>-float" / "<N> float" / "<N>-field" / "<N> to 128 to 128" mention
    in a living document must use ``contract.OBSERVATION_FIELD_COUNT``.
    """

    def test_living_documents_use_the_contract_width(self) -> None:
        for relative in OBSERVATION_DIMENSION_DOCUMENTS:
            with self.subTest(document=relative):
                text = _read(relative)
                for pattern in _OBSERVATION_WIDTH_MENTIONS:
                    for match in pattern.finditer(text):
                        self.assertEqual(
                            int(match.group(1)),
                            OBSERVATION_FIELD_COUNT,
                            f"{relative} claims a {match.group(1)}-wide observation, "
                            f"the contract is {OBSERVATION_FIELD_COUNT}",
                        )

    def test_the_contract_document_leads_with_the_current_width(self) -> None:
        """``docs/OBSERVATION_ACTION_CONTRACT.md`` keeps its own history.

        The version sections at the bottom deliberately quote the old
        widths (33, 65, 84), so only the first mention - the one in the
        current-state header and data-flow diagram - is checked.
        """
        text = _read("docs/OBSERVATION_ACTION_CONTRACT.md")
        first = min(
            (match for pattern in _OBSERVATION_WIDTH_MENTIONS for match in pattern.finditer(text)),
            key=lambda match: match.start(),
        )
        self.assertEqual(int(first.group(1)), OBSERVATION_FIELD_COUNT)


class PackageVersionTests(unittest.TestCase):
    """``pyproject.toml`` is the only place the package version lives."""

    def setUp(self) -> None:
        self.pyproject = tomllib.loads(_read("pyproject.toml"))

    def test_the_package_reports_the_packaged_version(self) -> None:
        self.assertEqual(__version__, self.pyproject["project"]["version"])

    def test_the_cli_reports_the_packaged_version(self) -> None:
        import contextlib
        import io

        from sandboxai import cli

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as exit_info:
            cli.build_parser().parse_args(["--version"])
        self.assertEqual(exit_info.exception.code, 0)
        printed = stream.getvalue()
        self.assertIn(__version__, printed)
        self.assertIn(GODOT_VERSION, printed)

    def test_the_declared_python_floor_is_the_one_ruff_targets(self) -> None:
        requires = self.pyproject["project"]["requires-python"]
        target = self.pyproject["tool"]["ruff"]["target-version"]
        floor = re.search(r"(\d+)\.(\d+)", requires)
        assert floor is not None
        self.assertEqual(target, f"py{floor.group(1)}{floor.group(2)}")


class DocumentationIndexTests(unittest.TestCase):
    """Every document is reachable, and every index entry resolves."""

    def setUp(self) -> None:
        self.index_path = DOCS / "README.md"
        self.index = self.index_path.read_text(encoding="utf-8")
        self.documents = {
            path.name for path in sorted(DOCS.glob("*.md")) if path.name != "README.md"
        }

    def test_every_document_is_listed_in_the_index(self) -> None:
        linked = set(re.findall(r"\]\((?!\.\./|https?:)([A-Z0-9_]+\.md)\)", self.index))
        missing = self.documents - linked
        self.assertEqual(
            missing,
            set(),
            f"docs/README.md does not link: {sorted(missing)}",
        )

    def test_every_index_entry_points_at_a_real_file(self) -> None:
        for target in re.findall(r"\]\((?!https?:)([^)#]+)\)", self.index):
            with self.subTest(link=target):
                self.assertTrue(
                    (self.index_path.parent / target).resolve().is_file(),
                    f"docs/README.md links to a missing file: {target}",
                )

    def test_relative_links_inside_the_documents_resolve(self) -> None:
        for document in sorted(DOCS.glob("*.md")):
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\]\((?!https?:|mailto:|#)([^)#]+)\)", text):
                with self.subTest(document=document.name, link=target):
                    self.assertTrue(
                        (document.parent / target).resolve().exists(),
                        f"{document.name} links to a missing path: {target}",
                    )


def _package_modules() -> dict[str, str]:
    """Map every public module name to the first line of its docstring."""
    modules: dict[str, str] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name.startswith("__"):
            continue
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        if docstring is None:
            modules[path.stem] = ""
        else:
            modules[path.stem] = docstring.splitlines()[0].strip()
    return modules


def _module_level_imports(path: Path) -> set[str]:
    """Sibling modules imported when ``path`` is imported.

    Only module-level statements count, plus the bodies of ``if`` blocks
    (that is where ``if TYPE_CHECKING:`` lives, and a cycle there still
    breaks tooling even though it never runs). Imports inside functions
    are deliberately excluded: deferring an import into a function is the
    standard way to break a cycle, and several modules here do exactly
    that on purpose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    siblings: set[str] = set()
    for node in tree.body:
        candidates = ast.walk(node) if isinstance(node, ast.If) else [node]
        for candidate in candidates:
            relative = isinstance(candidate, ast.ImportFrom) and candidate.level == 1
            if relative and candidate.module:
                siblings.add(candidate.module.split(".")[0])
    return siblings


class DependencyDeclarationTests(unittest.TestCase):
    """pyproject.toml is the only place dependencies are declared.

    There used to be a requirements.txt as well. Nothing referenced it -
    not the README, not CI, not the helper scripts; the pip-audit job
    builds its own list from pyproject - and it had already drifted,
    listing ``torch>=2.1`` where pyproject says ``torch>=2.1,<3``. A
    second copy of a fact nobody reads is a second copy that goes wrong
    quietly, so it was deleted. This test keeps it deleted.
    """

    def test_there_is_no_second_dependency_list(self) -> None:
        stray = REPOSITORY_ROOT / "requirements.txt"
        self.assertFalse(
            stray.is_file(),
            "requirements.txt is back. Dependencies belong in pyproject.toml; "
            "if a plain list is needed, generate it rather than maintaining it.",
        )

    def test_the_lock_file_is_labelled_as_platform_specific(self) -> None:
        lock = REPOSITORY_ROOT / "requirements-lock-linux-py311-cpu.txt"
        self.assertTrue(lock.is_file(), "the generated CI lock file is missing")
        header = lock.read_text(encoding="utf-8")[:2000]
        self.assertIn("linux", header.lower())

    def test_every_extra_resolves_to_a_declared_extra(self) -> None:
        data = tomllib.loads(_read("pyproject.toml"))
        extras = data["project"]["optional-dependencies"]
        for name, requirements in extras.items():
            for requirement in requirements:
                match = re.fullmatch(r"sandboxai\[([\w,\s-]+)\]", requirement)
                if match is None:
                    continue
                for referenced in match.group(1).split(","):
                    with self.subTest(extra=name, references=referenced.strip()):
                        self.assertIn(referenced.strip(), extras)


class ModuleMapTests(unittest.TestCase):
    """docs/PYTHON_MODULE_MAP.md is generated from the package, so it must match it.

    The package is flat by choice (see the document's own preamble). The
    price of that choice is that nothing about the grouping is enforced
    by the directory layout, so it is enforced here instead: a new module
    that nobody classified fails this, and a description that no longer
    matches the module docstring fails this.
    """

    ROW = re.compile(
        r"^\| \[`([a-z0-9_]+)`\]\(\.\./python/sandboxai/([a-z0-9_]+)\.py\) \| (.+?) \|$"
    )

    def setUp(self) -> None:
        self.map_path = DOCS / "PYTHON_MODULE_MAP.md"
        self.rows = [
            match.groups()
            for match in (
                self.ROW.match(line)
                for line in self.map_path.read_text(encoding="utf-8").splitlines()
            )
            if match is not None
        ]
        self.modules = _package_modules()

    def test_every_module_is_classified_exactly_once(self) -> None:
        listed = [name for name, _, _ in self.rows]
        self.assertEqual(
            sorted(listed),
            sorted(self.modules),
            "docs/PYTHON_MODULE_MAP.md disagrees with python/sandboxai/ about which modules exist",
        )
        self.assertEqual(len(listed), len(set(listed)), "a module is listed twice")

    def test_each_link_target_matches_its_label(self) -> None:
        for name, target, _ in self.rows:
            with self.subTest(module=name):
                self.assertEqual(name, target)

    def test_descriptions_are_the_module_docstring_summaries(self) -> None:
        for name, _, description in self.rows:
            with self.subTest(module=name):
                self.assertEqual(
                    description,
                    self.modules[name],
                    f"the map describes {name} with a line that is not its "
                    f"docstring summary; edit the docstring instead",
                )

    def test_the_stated_module_count_is_right(self) -> None:
        text = self.map_path.read_text(encoding="utf-8")
        self.assertIn(f"all {len(self.modules)} modules sit directly under", text)

    def test_the_module_level_import_graph_is_acyclic(self) -> None:
        graph = {
            path.stem: _module_level_imports(path)
            for path in sorted(PACKAGE.glob("*.py"))
            if not path.name.startswith("__")
        }
        # Iteratively strip modules with no un-visited sibling imports. A
        # non-empty remainder is a cycle, and its members are named.
        remaining = {name: set(deps) & set(graph) for name, deps in graph.items()}
        while True:
            resolved = {name for name, deps in remaining.items() if not deps}
            if not resolved:
                break
            remaining = {
                name: deps - resolved for name, deps in remaining.items() if name not in resolved
            }
        # What survives is the cycle plus everything that depends on it.
        # Strip the dependants too, so the message names only the modules
        # that are actually in a loop.
        while True:
            imported = {dep for deps in remaining.values() for dep in deps}
            dependants = set(remaining) - imported
            if not dependants:
                break
            remaining = {name: deps for name, deps in remaining.items() if name not in dependants}
        self.assertEqual(
            remaining,
            {},
            "module-level import cycle in sandboxai: "
            + ", ".join(f"{name} -> {sorted(deps)}" for name, deps in sorted(remaining.items())),
        )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
