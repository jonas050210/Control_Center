"""Runs the GDScript static analyzer over the repository as a unit test.

The Godot engine cannot be executed in every environment this repo is
developed in (CI containers, sandboxes without a GPU or without the engine
binary), so `tests/run_tests.gd` is not always runnable. This test closes
part of that gap: it parses every `.gd` file and fails on the classes of
defect that a type-checked language would catch at compile time — unknown
`preload` targets, calls to methods that do not exist on a preloaded
script, wrong argument counts, unknown enum members and duplicate
definitions. A second test runs gdtoolkit's `gdlint` style rules.

It is deliberately a *static* check. It does NOT prove the simulation
behaves correctly; only `godot --headless --path . --script
res://tests/run_tests.gd` does that.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from optional_deps import GDTOOLKIT_REASON, HAS_GDTOOLKIT
from sandboxai.gdscript_analysis import analyze, lint_all

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


if __name__ == "__main__":
    unittest.main()
