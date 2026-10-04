"""Installation checks shown on the Setup page and by ``rocketai doctor``."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Any

from .config import PYTHON_CMD
from .play import server_path


def _module(name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False, "nicht installiert"
        module = __import__(name, fromlist=["*"])
    except Exception as error:  # missing or broken install
        return False, f"Fehler beim Import: {type(error).__name__}: {error}"
    return True, str(getattr(module, "__version__", "installiert"))


def _python_check(version: tuple[int, int] | None = None) -> tuple[bool, str]:
    actual_version = sys.version_info[:2] if version is None else version
    display_version = (
        platform.python_version() if version is None else f"{version[0]}.{version[1]}"
    )
    detail = f"{display_version} ({sys.executable})"
    ok = (3, 11) <= actual_version < (3, 14)
    if not ok:
        detail += "; unterstützt werden Python 3.11–3.13 (empfohlen: 3.12)"
    return ok, detail


def teacher_check() -> dict[str, Any]:
    from .teacher import describe_teacher

    info = describe_teacher()
    if info["ready"]:
        detail = f"bereit in {info['directory']}"
    else:
        detail = f"nicht geladen - '{PYTHON_CMD} -m rocketai teacher' holt ihn (nur offline nutzen)"
    return {
        "label": f"Lehrer ({info['name']})",
        "detail": detail,
        "ok": info["ready"],
        "required": False,
    }


def run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(key: str, label: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append(
            {"key": key, "label": label, "ok": ok, "detail": detail, "required": required}
        )

    python_ok, python_detail = _python_check()
    add("python", "Python 3.11–3.13", python_ok, python_detail)

    for key, label in (
        ("torch", "PyTorch (CPU)"),
        ("rlgym.rocket_league.api", "RLGym-Simulation"),
        ("RocketSim", "RocketSim"),
    ):
        ok, detail = _module(key)
        add(key, label, ok, detail)
    ok, detail = _module("rlbot")
    add("rlbot", "RLBot-Python (für das echte Spiel)", ok, detail, required=False)
    server = server_path()
    add(
        "rlbot_server",
        "RLBotServer (für das echte Spiel)",
        server.exists(),
        str(server) if server.exists() else f"fehlt: {server} ({PYTHON_CMD} install.py)",
        required=False,
    )
    is_windows = sys.platform == "win32"
    add(
        "windows",
        "Windows (Rocket League läuft nur dort)",
        is_windows,
        platform.platform()
        if is_windows
        else "Training geht überall, Spielen im echten RL nur unter Windows",
        required=False,
    )
    if is_windows:
        from .rlcheck import check

        rl = check()
        add(
            "rocket_league",
            "Rocket League installiert",
            bool(rl.install),
            f"{rl.store}: {rl.install}" if rl.install else "nicht gefunden (Steam/Epic)",
            required=False,
        )
        if rl.game == "normal":
            add(
                "rl_running",
                "Rocket League geschlossen",
                False,
                "läuft normal gestartet – vor dem Spielstart schließen",
                required=False,
            )
    add("cpu", "CPU-Kerne", True, f"{os.cpu_count()} (mehr Kerne = schnelleres Training)")
    try:
        checks.append(teacher_check())
    except Exception as error:
        add(
            "teacher",
            "Lehrer (Nexto)",
            False,
            f"Prüfung fehlgeschlagen: {type(error).__name__}: {error}",
            required=False,
        )
    return checks
