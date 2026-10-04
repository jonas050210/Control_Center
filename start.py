#!/usr/bin/env python3
"""Start the RocketAI web app (after ``python install.py``).

python start.py                 # opens http://127.0.0.1:8765 in the browser
python start.py --port 9000 --no-browser
python start.py --host 0.0.0.0  # reachable from other devices in your network
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    except OSError:
        return False


def main() -> int:
    if not in_venv():
        if not VENV_PYTHON.exists():
            print("RocketAI ist noch nicht installiert. Bitte zuerst ausführen:  python install.py")
            return 1
        # Re-run inside the virtual environment.
        return subprocess.call(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT
        )
    sys.path.insert(0, str(ROOT))
    from rocketai.cli import main as cli_main

    try:
        return cli_main(["serve", *sys.argv[1:]])
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
