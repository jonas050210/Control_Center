#!/usr/bin/env python3
"""One-time setup for RocketAI.

    python3 install.py              # everything (recommended)
    python3 install.py --no-rlbot   # only training/simulation, no real-game support

Unter Windows heißt der Befehl ``python`` statt ``python3``.

Creates ``.venv``, installs CPU-only PyTorch and RocketAI with all
dependencies, and downloads RLBotServer (checked by SHA-256) into
``tools/rlbot`` so the trained bot can play in the real Rocket League.
Only uses the Python standard library, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

RLBOT_VERSION = "v5.0.0-rc17"
RLBOT_ASSETS = {
    # platform: (file name, size in bytes, sha256)
    "win32": (
        "RLBotServer.exe",
        5563392,
        "b64dc026d61b9f2b2d5645e80dbc6a9d7391231ff49246d29028018dcb036ab8",
    ),
    "linux": (
        "RLBotServer",
        6284192,
        "6ecf4905e777daa4d1c0bd09e982bc3449083c169778940be752ff23b3fa1154",
    ),
}
RLBOT_URL = "https://github.com/RLBot/core/releases/download/{version}/{name}"


def say(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args: str | Path) -> None:
    command = [str(a) for a in args]
    print("    $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def python_312_command() -> str:
    return "py -3.12" if os.name == "nt" else "python3.12"


def check_python(version: tuple[int, int] | None = None) -> None:
    """Stop before installing when the current interpreter is outside our tested range."""
    version = sys.version_info[:2] if version is None else version
    if version < (3, 11):
        sys.exit(
            f"Python 3.11–3.13 wird benötigt (gefunden: {version[0]}.{version[1]})."
        )
    if version >= (3, 14):
        sys.exit(
            f"Python {version[0]}.{version[1]} wird noch nicht unterstützt. "
            "RocketAI benötigt Python 3.11–3.13; die verwendete RLGym-Version "
            f"bricht unter Python 3.14 beim Import ab. Bitte Python 3.12 installieren "
            f"und `{python_312_command()} install.py` ausführen."
        )


def interpreter_version(python: Path) -> tuple[int, int]:
    """Read the version of an existing venv without relying on activation."""
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        major, minor = result.stdout.strip().split(".", maxsplit=1)
        return int(major), int(minor)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        sys.exit(
            f"Die vorhandene virtuelle Umgebung kann nicht geprüft werden ({error}). "
            "Bitte den Ordner .venv löschen und install.py erneut ausführen."
        )


def create_venv() -> Path:
    python = venv_python()
    if python.exists():
        version = interpreter_version(python)
        if version < (3, 11) or version >= (3, 14):
            sys.exit(
                f"Die vorhandene virtuelle Umgebung verwendet Python {version[0]}.{version[1]} "
                "und wird nicht unterstützt. Bitte .venv löschen und danach mit Python 3.12 "
                f"`{python_312_command()} install.py` erneut ausführen."
            )
        say(f"Virtuelle Umgebung vorhanden: {VENV} (Python {version[0]}.{version[1]})")
    else:
        say(f"Lege virtuelle Umgebung an: {VENV}")
        venv.EnvBuilder(with_pip=True).create(VENV)
    run(python, "-m", "pip", "install", "--upgrade", "pip", "--quiet")
    return python


def install_packages(python: Path, with_rlbot: bool, dev: bool) -> None:
    say("Installiere PyTorch (nur CPU, keine Grafikkarte nötig)")
    run(python, "-m", "pip", "install", "torch", "--index-url", TORCH_CPU_INDEX)
    extras = [name for name, wanted in (("rlbot", with_rlbot), ("dev", dev)) if wanted]
    target = f".[{','.join(extras)}]" if extras else "."
    say("Installiere RocketAI und Abhängigkeiten (RLGym, RocketSim, Web-App)")
    run(python, "-m", "pip", "install", "-e", target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_rlbot_server() -> None:
    platform = (
        "win32"
        if sys.platform == "win32"
        else "linux"
        if sys.platform.startswith("linux")
        else None
    )
    if platform is None:
        say(
            "RLBotServer gibt es nur für Windows und Linux – übersprungen (Training funktioniert trotzdem)."
        )
        return
    name, size, expected = RLBOT_ASSETS[platform]
    folder = ROOT / "tools" / "rlbot"
    target = folder / name
    if target.exists() and sha256(target) == expected:
        say(f"RLBotServer {RLBOT_VERSION} bereits vorhanden")
        return
    folder.mkdir(parents=True, exist_ok=True)
    url = RLBOT_URL.format(version=RLBOT_VERSION, name=name)
    say(f"Lade RLBotServer {RLBOT_VERSION} herunter")
    print(f"    {url}")
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RocketAI-installer"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 16):
            out.write(chunk)
    if partial.stat().st_size != size or sha256(partial) != expected:
        partial.unlink(missing_ok=True)
        sys.exit(
            "Download von RLBotServer ist beschädigt (Prüfsumme falsch). Bitte erneut versuchen."
        )
    os.replace(partial, target)
    if platform == "linux":
        target.chmod(0o755)
    print(f"    gespeichert: {target} (Prüfsumme ok)")


PYTHON_CMD = "python" if sys.platform == "win32" else "python3"


def main() -> None:
    parser = argparse.ArgumentParser(description="RocketAI einrichten")
    parser.add_argument(
        "--no-rlbot", action="store_true", help="ohne Unterstützung für das echte Spiel"
    )
    parser.add_argument(
        "--dev", action="store_true", help="zusätzlich Test-Werkzeuge (pytest, ruff)"
    )
    args = parser.parse_args()

    check_python()
    python = create_venv()
    install_packages(python, with_rlbot=not args.no_rlbot, dev=args.dev)
    if not args.no_rlbot:
        try:
            download_rlbot_server()
        except OSError as error:
            print(f"\nRLBotServer konnte nicht geladen werden ({error}).")
            print(
                f"Training geht trotzdem; für das echte Spiel '{PYTHON_CMD} install.py' "
                "später erneut ausführen."
            )
    say("Prüfe die Installation")
    subprocess.run([str(python), "-m", "rocketai", "doctor"], cwd=ROOT, check=True)
    say(f"Fertig! Starte die App mit:  {PYTHON_CMD} start.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(
            f"\nEin Schritt ist fehlgeschlagen (Exit-Code {error.returncode}). Details stehen oben."
        )
