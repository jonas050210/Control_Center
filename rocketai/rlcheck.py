"""Rocket League & RLBot environment check (Windows), mirroring RLBot's own logic.

Everything that parses files is a pure function so it is testable on any OS;
only :func:`check` touches the registry and the process list.

Findings:

* Is Rocket League installed (Steam or Epic) and where?
* Is it running — and if so, was it started by RLBot (``-rlbot`` argument)?
  A normally started RL blocks RLBot: it must be closed first.
* Is RLBotServer available / running, is its port open?
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STEAM_APP_ID = "252950"
EPIC_APP_NAME = "Sugar"
RLBOT_PORT = 23234
EXE_RELATIVE = Path("Binaries") / "Win64" / "RocketLeague.exe"


# ---------------------------------------------------------------- parsing
def parse_vdf_paths(text: str) -> list[str]:
    """Library folders from Steam's ``libraryfolders.vdf`` (``"path" "..."`` entries)."""
    return [m.replace("\\\\", "\\") for m in re.findall(r'"path"\s+"([^"]+)"', text)]


def parse_acf_installdir(text: str) -> str | None:
    match = re.search(r'"installdir"\s+"([^"]+)"', text)
    return match.group(1) if match else None


def parse_epic_manifest(text: str) -> str | None:
    """InstallLocation of an Epic ``.item`` manifest if it is Rocket League."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if data.get("AppName") == EPIC_APP_NAME or data.get("MainGameAppName") == EPIC_APP_NAME:
        return data.get("InstallLocation") or None
    return None


def classify_process(command_line: str | None) -> str:
    """``rlbot`` if started for bots, ``normal`` otherwise."""
    if command_line and re.search(r"(^|\s)-rlbot(\s|$)", command_line):
        return "rlbot"
    return "normal"


# ---------------------------------------------------------------- finding
def steam_root() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            path, _ = winreg.QueryValueEx(key, "SteamPath")
        return Path(path)
    except OSError:
        return None


def find_steam_install(root: Path | None) -> Path | None:
    if root is None:
        return None
    libraries = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        libraries += [Path(p) for p in parse_vdf_paths(vdf.read_text("utf-8", "replace"))]
    for library in libraries:
        manifest = library / "steamapps" / f"appmanifest_{STEAM_APP_ID}.acf"
        if manifest.is_file():
            installdir = parse_acf_installdir(manifest.read_text("utf-8", "replace"))
            if installdir:
                exe = library / "steamapps" / "common" / installdir / EXE_RELATIVE
                if exe.is_file():
                    return exe
    return None


def find_epic_install(manifests: Path | None = None) -> Path | None:
    if manifests is None:
        program_data = os.environ.get("PROGRAMDATA")
        if not program_data:
            return None
        manifests = Path(program_data) / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    if not manifests.is_dir():
        return None
    for item in sorted(manifests.glob("*.item")):
        location = parse_epic_manifest(item.read_text("utf-8", "replace"))
        if location:
            exe = Path(location) / EXE_RELATIVE
            if exe.is_file():
                return exe
    return None


def find_rlbot_server(cwd: Path | None = None) -> Path | None:
    name = "RLBotServer.exe" if sys.platform == "win32" else "RLBotServer"
    from .play import server_path

    candidates = [server_path(), (cwd or Path.cwd()) / name]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "RLBot5" / "bin" / name)
    found = next((c for c in candidates if c.is_file()), None)
    if found:
        return found
    which = shutil.which(name)
    return Path(which) if which else None


def process_command_lines(name: str) -> list[str] | None:
    """Command lines of running processes called ``name``; None if unknown."""
    try:
        if sys.platform == "win32":
            script = (
                f"Get-CimInstance Win32_Process -Filter \"Name='{name}'\" | "
                "ForEach-Object { $_.CommandLine }"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if out.returncode != 0:
                return None
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5)
        stem = name.removesuffix(".exe")
        return [line for line in out.stdout.splitlines() if stem in line.split(" ")[0]]
    except (OSError, subprocess.SubprocessError):
        return None


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- report
@dataclass
class RLStatus:
    platform: str
    supported: bool
    install: str | None = None
    store: str | None = None
    game: str = "unknown"  # not_running | rlbot | normal | unknown
    server: str | None = None
    server_running: bool = False
    port_open: bool = False
    ready: bool = False
    verdict: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(status: RLStatus) -> RLStatus:
    """Fill ``steps``, ``ready`` and ``verdict`` from the raw findings (pure)."""
    steps = []

    def step(key: str, label: str, state: str, detail: str) -> None:
        steps.append({"key": key, "label": label, "state": state, "detail": detail})

    if not status.supported:
        step(
            "os",
            "Windows",
            "bad",
            "Echtes Rocket League läuft nur unter Windows. "
            "Training und Live-Ansicht funktionieren trotzdem.",
        )
    if status.install:
        step("install", "Rocket League installiert", "ok", f"{status.store}: {status.install}")
    elif status.supported:
        step(
            "install",
            "Rocket League installiert",
            "bad",
            "Nicht gefunden (weder Steam noch Epic). Ohne Spiel nur Simulation.",
        )
    if status.game == "normal":
        step(
            "game",
            "Spiel-Zustand",
            "warn",
            "Rocket League läuft normal gestartet. Bitte schließen – RLBot startet es "
            "selbst mit den richtigen Parametern (offline, ohne Anti-Cheat).",
        )
    elif status.game == "rlbot":
        step("game", "Spiel-Zustand", "ok", "Läuft im RLBot-Modus – bereit für Bots.")
    elif status.game == "not_running":
        step("game", "Spiel-Zustand", "ok", "Läuft nicht – RLBot startet es beim Spielstart.")
    elif status.supported:
        step("game", "Spiel-Zustand", "warn", "Konnte nicht ermittelt werden (PowerShell).")
    if status.server:
        detail = status.server + (" (läuft)" if status.server_running else "")
        step("server", "RLBotServer", "ok", detail)
    elif status.supported:
        step(
            "server",
            "RLBotServer",
            "warn",
            "Fehlt – einmal 'python install.py' ausführen (lädt RLBotServer).",
        )

    status.steps = steps
    status.ready = bool(status.supported and status.install and status.game != "normal")
    if not status.supported:
        status.verdict = "Nur Simulation (kein Windows)"
    elif not status.install:
        status.verdict = "Rocket League nicht gefunden"
    elif status.game == "normal":
        status.verdict = "Rocket League schließen"
    elif status.game == "rlbot":
        status.verdict = "Bereit · RL im Bot-Modus"
    else:
        status.verdict = "Bereit"
    return status


def check() -> RLStatus:
    supported = sys.platform == "win32"
    status = RLStatus(platform=sys.platform, supported=supported)
    if supported:
        exe = find_steam_install(steam_root())
        store = "Steam"
        if exe is None:
            exe, store = find_epic_install(), "Epic"
        if exe is not None:
            status.install, status.store = str(exe), store
        lines = process_command_lines("RocketLeague.exe")
        if lines is None:
            status.game = "unknown"
        elif not lines:
            status.game = "not_running"
        else:
            status.game = (
                "rlbot" if any(classify_process(c) == "rlbot" for c in lines) else "normal"
            )
    server = find_rlbot_server()
    if server is not None:
        status.server = str(server)
        running = process_command_lines(server.name)
        status.server_running = bool(running)
    status.port_open = port_open(RLBOT_PORT)
    status.server_running = status.server_running or status.port_open
    return summarize(status)
