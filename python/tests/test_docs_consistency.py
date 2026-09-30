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

import re
import tomllib
import unittest
from pathlib import Path

from sandboxai import __version__
from sandboxai.contract import GODOT_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY_ROOT / "docs"
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"

# Files that quote the engine version in prose or configuration. Each one
# is read as text and every "Godot <x.y.z>" mention in it must agree with
# contract.GODOT_VERSION.
ENGINE_VERSION_DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "PROJECT.md",
    "docs/ARCHITECTURE.md",
    "docs/CONTROL_CENTER.md",
    "docs/RUN_LOCAL_VALIDATION.md",
)

_GODOT_MENTION = re.compile(r"Godot[\s_v]+(\d+\.\d+\.\d+)")


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


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
