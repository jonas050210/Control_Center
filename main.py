#!/usr/bin/env python3
"""SandboxAI Control Center - the headless-only control application.

Running this file from the repository root launches the desktop Control
Center directly; no Godot editor and no manual ``.tscn`` launching is
required. The window is a normal movable/resizable desktop application
that operates the headless training stack (agents, workers, benchmarks,
evaluations, runs and telemetry) over the existing ``sandboxai`` package.

    python3 main.py

The equivalent installed entry point is ``sandboxai control-center-desktop``
(see ``tools/windows/start_control_center.bat`` for the Windows launcher).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parent
_PACKAGE_ROOT = _REPOSITORY_ROOT / "python"


def _ensure_importable() -> None:
    """Make ``sandboxai`` importable without a prior ``pip install -e .``.

    The package lives under ``python/`` (see ``pyproject.toml``), so a
    source checkout is not on ``sys.path`` by default. Inserting it here
    keeps ``python3 main.py`` a single self-contained command, exactly like
    the test suite's ``conftest.py`` bootstrap.
    """
    if _PACKAGE_ROOT.is_dir() and str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))


def main() -> int:
    _ensure_importable()
    try:
        from sandboxai.control_center_desktop import main as control_center_main
    except ImportError as exc:
        # A Python without Tk fails in one of two ways: `import tkinter`
        # itself (no Tkinter package at all) or `import _tkinter` from
        # inside it (the package exists but its C extension does not - a
        # common case with conda/embedded builds and broken venvs). Both
        # mean the same thing to the user, and both deserve the fix rather
        # than a traceback that ends in a module name they never typed.
        if exc.name in {"tkinter", "_tkinter"}:
            print(
                "The Control Center needs Tkinter, which this Python "
                "installation does not provide. Install the python3-tk "
                "system package (Debian/Ubuntu), repair the Python install "
                "with Tk support (Windows installer: Modify -> tcl/tk), or "
                "use a Python build with Tk, then run `python3 main.py` "
                "again.",
                file=sys.stderr,
            )
            return 1
        raise
    return control_center_main(
        project_root=str(_REPOSITORY_ROOT),
        output_root=str(_REPOSITORY_ROOT / "training"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
