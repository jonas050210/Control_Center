#!/usr/bin/env python3
"""Start SandboxAI - always with the project's own virtual environment.

    python start.py                    # open the Control Center (desktop window)
    python start.py view               # watch the newest checkpoint play in 3D
    python start.py view --map compound --checkpoint training/runs/<run>
    python start.py train --steps 200000
    python start.py <any sandboxai command> [options]   (see: python start.py help)

Run ``python install.py`` once first; if you forget, this script offers to
do it. You never need to activate the virtual environment yourself: when
started with any other Python, start.py re-launches itself with
``.venv``'s interpreter. Only the standard library is used before that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
INSTALL_MARKER = ROOT / ".sandboxai" / "install.json"
REEXEC_FLAG = "SANDBOXAI_START_REEXEC"

USAGE = """\
SandboxAI launcher

  python start.py                 open the Control Center window
  python start.py view [options]  watch a checkpoint play in the 3D viewer
                                  (python start.py view --help for all options)
  python start.py train ...       train PPO (all `sandboxai` commands work:
  python start.py evaluate ...     train, resume, evaluate, record, bc-train,
  python start.py help             benchmark, validate-runtime, ...)

First time? Run:  python install.py
"""


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_in_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def offer_install() -> bool:
    """Asks to run install.py when the environment is missing."""
    print("SandboxAI is not set up yet (no .venv found).")
    if not sys.stdin or not sys.stdin.isatty():
        print("Run:  python install.py")
        return False
    answer = input("Run the setup now? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes", "j", "ja"):
        print("Okay. Run  python install.py  whenever you are ready.")
        return False
    completed = subprocess.run([sys.executable, str(ROOT / "install.py")], cwd=str(ROOT))
    return completed.returncode == 0 and venv_python().is_file()


def relaunch_in_venv(argv: list[str]) -> int:
    python = venv_python()
    if not python.is_file() and not offer_install():
        return 1
    if os.environ.get(REEXEC_FLAG):
        print(f"start.py: {python} did not activate the virtual environment.", file=sys.stderr)
        return 1
    environment = dict(os.environ, **{REEXEC_FLAG: "1"})
    try:
        return subprocess.call([str(python), str(Path(__file__).resolve()), *argv], env=environment)
    except KeyboardInterrupt:
        return 130


def dispatch(argv: list[str]) -> int:
    """Runs inside .venv: maps the launcher arguments onto the sandboxai CLI."""
    package_root = ROOT / "python"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    os.chdir(ROOT)

    if argv and argv[0] in ("help", "-h", "--help"):
        print(USAGE)
        if argv[0] == "help":
            from sandboxai.cli import main as cli_main

            return cli_main(["--help"])
        return 0

    if not INSTALL_MARKER.is_file():
        print("Note: install.py has not completed here yet; run it if something is missing.")

    if not argv or argv[0].startswith("-"):
        # Bare `start.py` (optionally with Control Center flags such as
        # --output-root) opens the desktop Control Center.
        from sandboxai.cli import main as cli_main

        return cli_main(["control-center-desktop", *argv])

    from sandboxai.cli import main as cli_main

    return cli_main(argv)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not running_in_venv():
        return relaunch_in_venv(args)
    try:
        return int(dispatch(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
