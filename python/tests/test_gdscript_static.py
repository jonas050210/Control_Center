"""Runs the GDScript static analyzer over the repository as a unit test.

The Godot engine cannot be executed in every environment this repo is
developed in (CI containers, sandboxes without a GPU or without the engine
binary), so `tests/run_tests.gd` is not always runnable. This test closes
part of that gap: it parses every `.gd` file and fails on the classes of
defect that a type-checked language would catch at compile time — unknown
`preload` targets, calls to methods that do not exist on a preloaded
script, wrong argument counts, calls of instance members through a script
class, unknown enum members and duplicate definitions. A second test runs
gdtoolkit's `gdlint` style rules.

It is deliberately a *static* check. It does NOT prove the simulation
behaves correctly; only `godot --headless --path . --script
res://tests/run_tests.gd` does that.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from optional_deps import GDTOOLKIT_REASON, HAS_GDTOOLKIT
from sandboxai.gdscript_analysis import (
    ProjectIndex,
    analyze,
    check_local_method_calls,
    check_static_calls,
    lint_all,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

## Findings that only report a MISSING TOOL rather than a defect in the
## repository. Without gdtoolkit the analyzer cannot verify syntax, and it
## says so explicitly instead of pretending the files parsed.
_TOOLING_KINDS = frozenset({"parser-unavailable", "linter-unavailable"})


def _format(findings) -> str:
    return "\n".join(
        f"  {finding.path}:{finding.line}: [{finding.kind}] {finding.message}"
        for finding in findings
    )


class GDScriptStaticAnalysisTests(unittest.TestCase):
    def test_repository_has_no_static_analysis_findings(self):
        findings = [f for f in analyze(REPO_ROOT) if f.kind not in _TOOLING_KINDS]
        if findings:
            self.fail(
                f"GDScript static analysis reported {len(findings)} finding(s):\n"
                f"{_format(findings)}"
            )

    @unittest.skipUnless(HAS_GDTOOLKIT, GDTOOLKIT_REASON)
    def test_every_script_parses_with_the_real_gdscript_grammar(self):
        findings = [f for f in analyze(REPO_ROOT) if f.kind == "syntax"]
        if findings:
            self.fail(f"GDScript syntax errors:\n{_format(findings)}")

    @unittest.skipUnless(HAS_GDTOOLKIT, GDTOOLKIT_REASON)
    def test_repository_is_gdlint_clean(self):
        findings = [f for f in lint_all(REPO_ROOT) if f.kind not in _TOOLING_KINDS]
        if findings:
            self.fail(f"gdlint reported {len(findings)} problem(s):\n{_format(findings)}")

    def test_analyzer_actually_scans_the_gdscript_sources(self):
        # Guards against the analyzer silently finding zero files (e.g. a
        # path change) and reporting a vacuous pass.
        scripts = list((REPO_ROOT / "scripts").rglob("*.gd"))
        tests = list((REPO_ROOT / "tests").rglob("*.gd"))
        self.assertGreater(len(scripts), 40)
        self.assertGreater(len(tests), 20)


class StaticCallThroughScriptClassTests(unittest.TestCase):
    """Regression tests for the self-play `observations: []` failure class.

    The real Godot 4.7.2 runtime failed the self-play validation with
    ``Self play reset observation shape invalid: []`` because
    ``self_play_environment.gd`` called ``LightingProfile.mode(...)`` — an
    instance *variable* — through the preloaded script class. Godot rejects
    that at COMPILE time ("Static function ... not found in base ..."),
    which invalidates the whole script. An invalid script still loads as a
    resource, so ``preload`` chains survive: ``SelfPlayEnvironmentCore.new()``
    silently returned ``null``, ``SelfPlayAdapter._init`` aborted on the
    null ``reset()`` call, ``environments`` stayed empty, and every reset
    answered ``{"ok": true, "observations": []}``. The single-agent checks
    never load that script, so 6/7 checks passed.

    These tests pin the static check that catches the compile-error class
    without needing the engine.
    """

    def _fixture_project(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "profile.gd").write_text(
            "class_name Profile\n"
            "extends RefCounted\n"
            "\n"
            "var mode: int = 0\n"
            "\n"
            "static func create() -> Profile:\n"
            "\treturn load(\"res://scripts/profile.gd\").new()\n"
            "\n"
            "static func from_id(id: String, seed: int = 0) -> Profile:\n"
            "\treturn create()\n",
            encoding="utf-8",
        )
        (root / "scripts" / "env.gd").write_text(
            "class_name Env\n"
            "extends RefCounted\n"
            "\n"
            "const Profile = preload(\"res://scripts/profile.gd\")\n"
            "\n"
            "var lighting = null\n"
            "\n"
            "func reset() -> void:\n"
            "\tlighting = Profile.mode(\"low_light\")\n",
            encoding="utf-8",
        )
        return root

    def test_instance_member_called_through_script_class_is_flagged(self):
        index = ProjectIndex(self._fixture_project())
        findings = check_static_calls(index)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.kind, "nonstatic-call")
        self.assertEqual(finding.path, "res://scripts/env.gd")
        self.assertEqual(finding.line, 9)
        self.assertIn("Profile.mode", finding.message)
        self.assertIn("instance member", finding.message)

    def test_static_and_constructor_calls_are_not_flagged(self):
        root = self._fixture_project()
        (root / "scripts" / "env.gd").write_text(
            "class_name Env\n"
            "extends RefCounted\n"
            "\n"
            "const Profile = preload(\"res://scripts/profile.gd\")\n"
            "\n"
            "var lighting = null\n"
            "\n"
            "func reset() -> void:\n"
            "\tlighting = Profile.from_id(\"low_light\")\n"
            "\tvar fresh = Profile.new()\n",
            encoding="utf-8",
        )
        self.assertEqual(check_static_calls(ProjectIndex(root)), [])

    def test_self_play_environment_static_calls_are_clean(self):
        # Direct, named regression guard: the file that broke the real
        # runtime must contain zero class-level calls of instance members.
        index = ProjectIndex(REPO_ROOT)
        sp_findings = [
            f for f in check_static_calls(index) if "self_play" in f.path
        ]
        self.assertEqual(
            sp_findings,
            [],
            "self-play scripts must not call instance members through a "
            "script class (this is a Godot compile error that makes the "
            "self-play reset return [] over the bridge)",
        )


class UndefinedLocalCallTests(unittest.TestCase):
    """Regression tests for the broken-``system_monitor.gd`` failure class.

    ``scripts/control_center/system_monitor.gd`` called
    ``_needs_cpu_command()`` and ``_parse_optional_number()`` without
    declaring either. Godot rejects undeclared bare calls at COMPILE time
    ("Function ... not found in base self"), which invalidates the whole
    script — yet the invalid script still loads as a resource, so every
    ``preload`` chain survived and the damage only appeared at runtime:
    ``ControlCenterSystemMonitor.new()`` aborted with "Nonexistent function
    'new' in base 'GDScript'", ``ControlCenterSession._init`` died mid-way
    (leaving ``system_monitor`` and ``simulation_manager`` null), and ~37
    Control Center tests failed across session/scene/dashboard files.

    These tests pin the static check that catches this class of defect
    without needing the engine.
    """

    def _fixture_project(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "monitor.gd").write_text(
            "class_name Monitor\n"
            "extends Node\n"
            "\n"
            "func poll() -> void:\n"
            "\tvar jobs := {\"cpu_command\": _needs_cpu_command()}\n"
            "\t_ready()\n"
            "\tsuper._init()\n",
            encoding="utf-8",
        )
        return root

    def test_undefined_local_call_is_flagged(self):
        findings = check_local_method_calls(ProjectIndex(self._fixture_project()))
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.kind, "unknown-local-call")
        self.assertEqual(finding.path, "res://scripts/monitor.gd")
        self.assertEqual(finding.line, 5)
        self.assertIn("_needs_cpu_command()", finding.message)

    def test_virtuals_and_super_calls_are_not_flagged(self):
        root = self._fixture_project()
        # Remove the only genuine offender; _ready() is an engine virtual
        # and super._init() targets the base class, so neither may be
        # reported even though neither is declared here.
        (root / "scripts" / "monitor.gd").write_text(
            "class_name Monitor\n"
            "extends Node\n"
            "\n"
            "func poll() -> void:\n"
            "\t_ready()\n"
            "\tsuper._init()\n",
            encoding="utf-8",
        )
        self.assertEqual(check_local_method_calls(ProjectIndex(root)), [])

    def test_declared_and_inherited_local_calls_are_not_flagged(self):
        root = self._fixture_project()
        (root / "scripts" / "base.gd").write_text(
            "class_name Base\n"
            "extends RefCounted\n"
            "\n"
            "func _inherited_helper() -> bool:\n"
            "\treturn true\n",
            encoding="utf-8",
        )
        (root / "scripts" / "monitor.gd").write_text(
            "class_name Monitor\n"
            "extends Base\n"
            "\n"
            "func _own_helper() -> bool:\n"
            "\treturn true\n"
            "\n"
            "func poll() -> void:\n"
            "\tif _own_helper() and _inherited_helper():\n"
            "\t\t_ready()\n",
            encoding="utf-8",
        )
        self.assertEqual(check_local_method_calls(ProjectIndex(root)), [])

    def test_system_monitor_local_calls_are_clean(self):
        # Direct, named regression guard for the file that broke the real
        # runtime: it must not call any helper it does not declare.
        index = ProjectIndex(REPO_ROOT)
        findings = [
            f for f in check_local_method_calls(index) if "system_monitor" in f.path
        ]
        self.assertEqual(
            findings,
            [],
            "system_monitor.gd must not reference helpers it does not define "
            "(this is the Godot compile error that invalidated the script and "
            "cascaded through the Control Center suite)",
        )


if __name__ == "__main__":
    unittest.main()
