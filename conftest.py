"""Pytest bootstrap for the SandboxAI Python test-suite.

The package lives under ``python/`` (see ``[tool.setuptools.package-dir]``
in pyproject.toml) so that the Godot project root stays free of Python
package clutter. Without this file a bare ``pytest`` run from the
repository root cannot import ``sandboxai`` unless the package was
pip-installed first.

Adding the directory here means the exact same command works whether or
not ``pip install -e .`` has been run, which keeps the "run the tests"
instruction in the README a single line.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PYTHON_ROOT = _REPO_ROOT / "python"

for _candidate in (_PYTHON_ROOT, _PYTHON_ROOT / "tests"):
    _path = str(_candidate)
    if _candidate.is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)
