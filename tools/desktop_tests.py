#!/usr/bin/env python3
"""Run the real-Tk Control Center tests with one command, on any machine.

`python/tests/test_control_center_desktop.py` only means something with a
real Tk interpreter and a display: those tests caught two bugs (a shadowed
tkinter method, a benchmark button that never disabled) that no headless
check could see. Until now the only place they ran was the CI job
`desktop-ui-tests` under Xvfb, so a contributor's own machine silently
skipped them - exactly the gap that let those bugs through.

This script picks the strongest check the machine can actually run:

1. Tkinter + a display  -> the desktop suite on the real window.
2. Tkinter, no display  -> the same suite under `xvfb-run`, when installed.
3. neither              -> the static contracts and the fake-Tk smoke
                           harness, with the exact command needed for the
                           real suite printed.

Exit status is the status of whatever ran. Use ``--strict`` to fail instead
of falling back (that is the CI behaviour, and what a pre-release check
should use).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_TESTS = "python/tests/test_control_center_desktop.py"
STATIC_TESTS = (
    "python/tests/test_control_center_pages_static.py",
    "python/tests/test_control_center_theme_layout.py",
    "python/tests/test_control_center_viewmodel.py",
)
SMOKE = "tools/control_center_smoke.py"

INSTALL_HINTS = {
    "tkinter": "python3-tk (Debian/Ubuntu), python3-tkinter (Fedora), or a Python build with Tk",
    "xvfb": "xvfb (Debian/Ubuntu: apt install xvfb)",
}


def _run(command: list[str], *, extra_env: dict[str, str] | None = None) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "python")
    env.update(extra_env or {})
    return subprocess.call(command, cwd=str(REPOSITORY_ROOT), env=env)


def _python_imports(module: str) -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env={**os.environ, "PYTHONPATH": "python"},
        cwd=str(REPOSITORY_ROOT),
    )
    return probe.returncode == 0


def _has_display() -> bool:
    if sys.platform in {"win32", "darwin"}:
        return True  # the native window system is the display
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _headless_fallback(strict: bool) -> int:
    if strict:
        print(
            "No real Tk run is possible here. Install Tkinter and, on a headless\n"
            f"machine, Xvfb:\n  tkinter: {INSTALL_HINTS['tkinter']}\n"
            f"  xvfb:    {INSTALL_HINTS['xvfb']}",
            file=sys.stderr,
        )
        return 1
    print(
        "No real Tk run is possible here; running the checks that do work on this\n"
        "machine instead. The real window is covered by CI job `desktop-ui-tests`."
    )
    status = _run([sys.executable, "-m", "pytest", "-q", *STATIC_TESTS])
    status |= _run([sys.executable, SMOKE])
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when the real-Tk suite cannot run instead of falling back",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="extra arguments for pytest (e.g. -k density)",
    )
    args = parser.parse_args(argv)

    if not _python_imports("tkinter"):
        return _headless_fallback(args.strict)

    base = [sys.executable, "-m", "pytest", "-q", DESKTOP_TESTS, *args.pytest_args]
    if _has_display():
        print("Tkinter and a display are available: running the real window suite.")
        return _run(base)

    if shutil.which("xvfb-run"):
        print("Tkinter is available but there is no display: using xvfb-run.")
        return _run(["xvfb-run", "-a", *base])

    print(
        "Tkinter is available but neither a display nor xvfb-run is; install\n"
        f"Xvfb to run the window suite here:\n  {INSTALL_HINTS['xvfb']}"
    )
    return _headless_fallback(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
