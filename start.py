#!/usr/bin/env python3
"""NEURAL ARENA launcher: starts the FastAPI control center and opens the browser.

Usage (from the repository root):

    python start.py                    # http://127.0.0.1:8501 + Browser öffnen
    python start.py --port 8600        # anderer Port (bei Belegung automatisch +1)
    python start.py --no-browser       # nur Server starten
    python start.py --host 0.0.0.0     # im LAN erreichbar (z. B. aus Windows heraus)
    python start.py --reload           # Auto-Reload für die Entwicklung

If the virtual environment from install.py exists, the script re-executes itself
with that interpreter so no manual activation is required. In WSL the browser
opening falls back to wslview/cmd.exe/powershell.exe automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIRED_PACKAGES = ("fastapi", "uvicorn", "numpy", "gymnasium", "psutil")
DEFAULT_PORT = 8501


def print_banner(host: str, port: int, url: str, reload: bool) -> None:
    print()
    print("  ⚡ NEURAL ARENA · Control Center")
    print("  " + "-" * 62)
    print(f"  Server      : http://{host}:{port}")
    print(f"  Browser     : {url}")
    print(f"  API-Docs     : http://{host}:{port}/api/docs")
    print(f"  Auto-Reload : {'an' if reload else 'aus'}")
    print("  Beenden     : STRG+C")
    print("  " + "-" * 62)
    print()


def in_project_venv() -> bool:
    """True when the running interpreter *is* the project virtual environment.

    ``Path(sys.executable).resolve()`` is useless here: ``.venv/bin/python`` is a
    symlink chain that ends at the system interpreter. ``sys.prefix`` is the
    reliable marker because venv always points it at the environment folder.
    """
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def missing_packages() -> list[str]:
    import importlib.util

    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]


def reexec_in_venv() -> None:
    """Re-run start.py with .venv/bin/python when the venv exists but is not active."""
    if os.environ.get("NEURAL_ARENA_NO_REEXEC") == "1":
        return
    if not VENV_PYTHON.exists():
        return
    if in_project_venv():
        return
    if not missing_packages():
        return  # current interpreter is already usable, no restart needed
    target = VENV_PYTHON
    print(f"[NEURAL ARENA] Wechsle in die virtuelle Umgebung: {target}")
    sys.stdout.flush()
    sys.stderr.flush()
    env = dict(os.environ, NEURAL_ARENA_NO_REEXEC="1")
    os.execve(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def free_port(host: str, port: int, attempts: int = 12) -> int:
    """Return the first free TCP port starting at ``port``."""
    for offset in range(attempts):
        candidate = port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host if host not in {"0.0.0.0", "::"} else "127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    print(f"[NEURAL ARENA][warn] Ports {port}-{port + attempts - 1} sind belegt; nutze {port + attempts}.")
    return port + attempts


def browser_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}/"


def open_browser_when_ready(url: str, timeout: float = 25.0) -> None:
    """Poll the health endpoint, then open the default browser (WSL aware)."""
    health = url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    json.loads(response.read().decode("utf-8"))
                    break
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(0.4)
    else:
        print("[NEURAL ARENA][warn] Server war nicht rechtzeitig bereit – URL bitte manuell öffnen.")
    if not _launch_browser(url):
        print(f"[NEURAL ARENA] Browser konnte nicht automatisch geöffnet werden: {url}")


def _launch_browser(url: str) -> bool:
    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:  # noqa: BLE001 - fall through to the WSL helpers
        pass
    candidates: list[list[str]] = []
    if shutil.which("wslview"):
        candidates.append(["wslview", url])
    if shutil.which("cmd.exe"):
        candidates.append(["cmd.exe", "/c", "start", "", url])
    if shutil.which("powershell.exe"):
        candidates.append(["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"])
    if shutil.which("xdg-open"):
        candidates.append(["xdg-open", url])
    for command in candidates:
        try:
            result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=15)
            if result.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Startet das NEURAL ARENA Control Center.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP-Port (Standard {DEFAULT_PORT}, belegte Ports werden übersprungen)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind-Adresse (0.0.0.0 für Zugriff aus Windows/LAN)")
    parser.add_argument("--no-browser", action="store_true", help="Browser nicht automatisch öffnen")
    parser.add_argument("--reload", action="store_true", help="Auto-Reload für die Entwicklung")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.version_info < (3, 10):
        print(f"[NEURAL ARENA][error] Python {sys.version.split()[0]} ist zu alt (>= 3.10 nötig).")
        return 1

    # Switch into .venv first: it may be the only interpreter with the packages.
    reexec_in_venv()

    missing = missing_packages()
    if missing:
        print("[NEURAL ARENA][error] Fehlende Pakete: " + ", ".join(missing))
        print("[NEURAL ARENA][hint]  Zuerst installieren:  python install.py")
        return 1

    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    port = free_port(args.host, args.port)
    url = browser_url(args.host, port)
    print_banner(args.host, port, url, args.reload)

    from server.app import app  # imported late so import errors surface after the banner

    if not args.no_browser:
        opener = threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True)
        opener.start()

    import uvicorn

    reload_dirs = [str(PROJECT_ROOT / "server"), str(PROJECT_ROOT / "web")] if args.reload else None
    uvicorn.run(
        "server.app:app" if args.reload else app,
        host=args.host,
        port=port,
        log_level=args.log_level,
        reload=args.reload,
        reload_dirs=reload_dirs,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[NEURAL ARENA] Server gestoppt.")
        raise SystemExit(0)
