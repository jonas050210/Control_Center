#!/usr/bin/env python3
"""NEURAL ARENA installer: prepares a virtual environment, dependencies, and web assets.

Usage (from the repository root):

    python install.py                 # full install: venv + requirements + three.js
    python install.py --skip-torch    # everything except PyTorch/PPO training
    python install.py --no-venv       # install into the current interpreter
    python install.py --force-assets  # re-download the vendored Three.js build

The script is idempotent: it reuses an existing virtual environment and skips
downloads that are already present unless ``--force-assets`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import venv


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
TRAINING_REQUIREMENTS_FILE = PROJECT_ROOT / "requirements-training.txt"
VENV_DIR = PROJECT_ROOT / ".venv"
VENDOR_DIR = PROJECT_ROOT / "web" / "vendor"

THREE_VERSION = "0.160.1"
THREE_TARGET = VENDOR_DIR / "three.module.min.js"
# Direct file URLs first, npm registry tarball as the fallback that also works
# behind restrictive proxies (only registry.npmjs.org has to be reachable).
THREE_URLS = [
    f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.min.js",
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.min.js",
]
THREE_TARBALL_URLS = [
    f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz",
]
TARBALL_MEMBER = "package/build/three.module.min.js"
MINIMUM_ASSET_BYTES = 100_000

CORE_REQUIREMENTS = ["numpy", "gymnasium", "fastapi", "uvicorn[standard]", "httpx", "psutil"]
TRAINING_REQUIREMENTS = ["torch", "stable-baselines3"]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class Installer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = Path(args.python) if args.python else Path(sys.executable)
        self.steps: list[str] = []

    # -------------------------------------------------------------- reporting
    def log(self, message: str) -> None:
        print(f"[NEURAL ARENA] {message}", flush=True)

    def warn(self, message: str) -> None:
        print(f"[NEURAL ARENA][warn] {message}", flush=True)

    def fail(self, message: str, hint: str | None = None) -> None:
        print(f"[NEURAL ARENA][error] {message}", file=sys.stderr, flush=True)
        if hint:
            print(f"[NEURAL ARENA][hint] {hint}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    # ------------------------------------------------------------------ steps
    def check_python(self) -> None:
        if sys.version_info < (3, 10):
            self.fail(
                f"Python {sys.version.split()[0]} ist zu alt.",
                "Bitte Python 3.10 oder neuer installieren (getestet mit 3.11).",
            )
        self.log(f"Python {sys.version.split()[0]} gefunden ({sys.executable}).")

    def ensure_venv(self) -> Path:
        if self.args.no_venv:
            self.log("--no-venv gesetzt: Installation erfolgt in den aktuellen Interpreter.")
            return Path(sys.executable)
        if VENV_DIR.exists() and not VENV_DIR.joinpath("pyvenv.cfg").exists():
            self.log("Unvollständiges .venv gefunden – wird neu erstellt.")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        if not VENV_DIR.exists():
            self.log(f"Erstelle virtuelle Umgebung in {VENV_DIR} …")
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(VENV_DIR)
        else:
            self.log(f"Vorhandene virtuelle Umgebung wird verwendet: {VENV_DIR}")
        python = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            self.fail(f"Interpreter {python} fehlt.", "Bitte .venv löschen und install.py erneut ausführen.")
        self.python = python
        return self.python

    def run(self, command: list[str], *, quiet: bool = False) -> None:
        pretty = " ".join(str(part) for part in command)
        self.log(f"$ {pretty}")
        result = subprocess.run(command, cwd=PROJECT_ROOT,
                                stdout=subprocess.DEVNULL if quiet else None,
                                stderr=subprocess.STDOUT if quiet else None)
        if result.returncode != 0 and not quiet:
            self.fail(f"Befehl fehlgeschlagen: {pretty}")

    def upgrade_pip(self) -> None:
        self.run([str(self.python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    def install_requirements(self) -> None:
        """Install the runtime requirements, plus the optional PPO training stack."""
        if self.args.skip_torch:
            self.log("--skip-torch: PyTorch/Stable-Baselines3 werden ausgelassen (kein PPO-Training).")
            if REQUIREMENTS.exists():
                self.run([str(self.python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
            else:
                self.run([str(self.python), "-m", "pip", "install", *CORE_REQUIREMENTS])
            return
        self.log("Installiere PyTorch als CPU-Wheel (kleiner als das CUDA-Paket) …")
        self.run([str(self.python), "-m", "pip", "install", "torch", "--index-url", TORCH_CPU_INDEX])
        if TRAINING_REQUIREMENTS_FILE.exists():
            self.run([str(self.python), "-m", "pip", "install", "-r", str(TRAINING_REQUIREMENTS_FILE)])
        else:
            self.run([str(self.python), "-m", "pip", "install", *TRAINING_REQUIREMENTS])
        if REQUIREMENTS.exists():
            self.log(f"Installiere Laufzeit-Abhängigkeiten aus {REQUIREMENTS.name} …")
            self.run([str(self.python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        else:
            self.warn("requirements.txt nicht gefunden – installiere Kernpakete direkt.")
            self.run([str(self.python), "-m", "pip", "install", *CORE_REQUIREMENTS])

    def handle_legacy_packages(self) -> None:
        """Offer to remove Streamlit/Plotly/pandas – they are no longer used."""
        installed = self._installed_legacy_packages()
        if not installed:
            return
        if self.args.keep_legacy:
            self.log(f"Behalte bereits installierte Altlasten: {', '.join(installed)} (--keep-legacy).")
            return
        remove = self.args.remove_legacy
        if not remove and sys.stdin.isatty():
            answer = input(
                f"[NEURAL ARENA] Diese Pakete werden nicht mehr gebraucht: {', '.join(installed)}.\n"
                "               Jetzt deinstallieren? [J/n] "
            ).strip().lower()
            remove = answer in {"", "j", "ja", "y", "yes"}
        if remove:
            self.log(f"Entferne nicht mehr benötigte Pakete: {', '.join(installed)} …")
            self.run([str(self.python), "-m", "pip", "uninstall", "-y", *installed], quiet=True)
        else:
            self.log(f"Streamlit/Plotly/pandas bleiben installiert ({', '.join(installed)}).")

    def download_assets(self) -> None:
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        if THREE_TARGET.exists() and THREE_TARGET.stat().st_size >= MINIMUM_ASSET_BYTES \
                and not self.args.force_assets:
            self.log(f"Three.js bereits vorhanden ({THREE_TARGET.stat().st_size // 1024} KB) – übersprungen.")
            return
        last_error: Exception | None = None
        for url in THREE_URLS:
            self.log(f"Lade Three.js {THREE_VERSION} von {url} …")
            try:
                payload = self._fetch(url)
                self._store_asset(payload, url)
                return
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
                self.warn(f"Download fehlgeschlagen: {exc}")
        for url in THREE_TARBALL_URLS:
            self.log(f"Versuche npm-Registry-Tarball: {url} …")
            try:
                payload = self._extract_from_tarball(self._fetch(url, binary=True))
                self._store_asset(payload, url)
                return
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                last_error = exc
                self.warn(f"Tarball-Extraktion fehlgeschlagen: {exc}")
        self.warn(
            "Three.js konnte nicht heruntergeladen werden. Das Control Center versucht dann beim "
            "Start automatisch das CDN; ohne Internet bleibt die 3D-Ansicht leer. "
            f"Letzter Fehler: {last_error}"
        )

    @staticmethod
    def _fetch(url: str, binary: bool = False) -> bytes:
        with urllib.request.urlopen(url, timeout=90) as response:
            return response.read()

    @staticmethod
    def _extract_from_tarball(tarball: bytes) -> bytes:
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
            member = archive.extractfile(TARBALL_MEMBER)
            if member is None:
                raise KeyError(f"{TARBALL_MEMBER} nicht im Tarball")
            return member.read()

    def _store_asset(self, payload: bytes, source: str) -> None:
        if len(payload) < MINIMUM_ASSET_BYTES:
            raise ValueError(f"Antwort zu klein ({len(payload)} Bytes)")
        if b"THREE" not in payload[:400_000]:
            raise ValueError("Antwort enthält kein Three.js")
        THREE_TARGET.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()[:12]
        self.log(f"Gespeichert: web/vendor/three.module.min.js ({len(payload) // 1024} KB, "
                 f"sha256:{digest}…, Quelle: {source})")

    def create_runtime_dirs(self) -> None:
        for relative in ("models", "logs", "data", "web/vendor"):
            (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
        self.log("Laufzeitordner models/, logs/, data/, web/vendor/ sind vorhanden.")

    def verify(self) -> None:
        self.log("Prüfe Installation …")
        script = (
            "import importlib.util as u, sys;"
            "mods=['numpy','gymnasium','fastapi','uvicorn','httpx','psutil'];"
            "missing=[m for m in mods if u.find_spec(m) is None];"
            "print('MISSING=' + ','.join(missing));"
            "print('TRAINING=' + str(bool(u.find_spec('stable_baselines3') and u.find_spec('torch'))));"
            "sys.exit(0)"
        )
        result = subprocess.run([str(self.python), "-c", script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        output = result.stdout.strip()
        if result.returncode != 0:
            self.fail("Abhängigkeitsprüfung fehlgeschlagen.", result.stderr.strip()[:400])
        missing = ""
        training = False
        for line in output.splitlines():
            if line.startswith("MISSING="):
                missing = line.split("=", 1)[1]
            if line.startswith("TRAINING="):
                training = line.split("=", 1)[1].strip() == "True"
        if missing:
            self.warn(f"Fehlende Pakete: {missing}. install.py erneut ohne --skip-torch ausführen.")
        else:
            self.log("Alle Kernpakete importierbar.")
        self.log(f"PPO-Training verfügbar: {'ja' if training else 'nein (--skip-torch?)'}")

        smoke = (
            "from env.shooter_env import ShooterEnv;"
            "env=ShooterEnv(map_name='Dust', seed=7);"
            "obs,_=env.reset();"
            "[env.step(env.action_space.sample()) for _ in range(3)];"
            "env.close();print('SMOKE_OK')"
        )
        result = subprocess.run([str(self.python), "-c", smoke], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        if "SMOKE_OK" in result.stdout:
            self.log("Headless-Umgebungs-Smoke-Test erfolgreich.")
        else:
            self.warn(f"Umgebungs-Smoke-Test fehlgeschlagen: {result.stderr.strip()[:300]}")

    def summary(self) -> None:
        launcher = "start.py"
        self.log("")
        self.log("=" * 68)
        self.log(" INSTALLATION ABGESCHLOSSEN")
        self.log("=" * 68)
        self.log(f" Python für den Start: {self.python}")
        self.log(f" Control Center starten: {str(self.python)} {launcher}")
        self.log(" Der Browser öffnet sich automatisch auf http://127.0.0.1:8501")
        self.log(" Optionen: --port 8600 · --no-browser · --host 0.0.0.0")
        self.log(" Tests:   " + str(self.python) + " -m unittest discover -s tests -v")
        self.log("")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Installiert Abhängigkeiten und Web-Assets für NEURAL ARENA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-venv", action="store_true",
                        help="Pakete in den aktuellen Interpreter installieren statt in .venv")
    parser.add_argument("--python", default=None,
                        help="Interpreter für eine vorhandene Umgebung (überspringt die venv-Erstellung)")
    parser.add_argument("--skip-torch", action="store_true",
                        help="PyTorch und Stable-Baselines3 auslassen (kein PPO-Training)")
    parser.add_argument("--remove-legacy", action="store_true",
                        help="Streamlit/Plotly/pandas ohne Rückfrage deinstallieren")
    parser.add_argument("--keep-legacy", action="store_true",
                        help="Nicht nach dem Entfernen von Streamlit/Plotly/pandas fragen")
    parser.add_argument("--force-assets", action="store_true",
                        help="Three.js auch dann neu laden, wenn die Datei bereits existiert")
    parser.add_argument("--no-assets", action="store_true",
                        help="Keine Web-Assets herunterladen (3D nutzt dann das CDN)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    installer = Installer(args)
    print()
    print("  ⚡ NEURAL ARENA · Installer")
    print("  Headless RL-Sandbox mit FastAPI-Backend und WebGL-Frontend")
    print()
    installer.check_python()
    if args.python:
        installer.python = Path(args.python)
        installer.log(f"Nutze vorhandenen Interpreter: {installer.python}")
    else:
        installer.ensure_venv()
    installer.upgrade_pip()
    installer.install_requirements()
    if not args.no_assets:
        installer.download_assets()
    installer.create_runtime_dirs()
    installer.verify()
    installer.summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
