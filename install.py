#!/usr/bin/env python3
"""One-command setup for a SandboxAI checkout.

    python install.py            # everything: venv, CPU PyTorch, packages, Godot
    python start.py              # afterwards: open the Control Center

What it does, in order (every step is skipped when it is already done, so
running it again is cheap and safe):

1. checks the Python version (3.11+) and whether Tkinter is available;
2. creates a private virtual environment in ``.venv``;
3. installs the **CPU-only** PyTorch build (no CUDA download - the policy is
   a tiny MLP and CPU is the supported training device) and then the
   project itself with its training and test extras;
4. finds a matching Godot 4.7.2 or downloads the official build from
   GitHub into ``tools/godot/`` and verifies its SHA-256;
5. remembers the Godot path in ``.sandboxai/settings.json`` and runs
   ``sandboxai validate-runtime`` as a final end-to-end check.

Only the standard library is used: this file has to run *before* anything
is installed. It is intentionally not called ``setup.py`` - setuptools would
execute a file with that name during ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
SETTINGS_DIR = ROOT / ".sandboxai"
INSTALL_MARKER = SETTINGS_DIR / "install.json"
GODOT_DIR = ROOT / "tools" / "godot"

MIN_PYTHON = (3, 11)
GODOT_VERSION = "4.7.2"
GODOT_RELEASE_URL = (
    f"https://github.com/godotengine/godot/releases/download/{GODOT_VERSION}-stable/"
)
TORCH_REQUIREMENT = "torch>=2.1,<3"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Official release assets and their published SHA-256 digests (GitHub
# release metadata for 4.7.2-stable). Pinning the digest means a corrupted
# or substituted download is refused instead of executed.
GODOT_ASSETS = {
    ("linux", "x86_64"): (
        f"Godot_v{GODOT_VERSION}-stable_linux.x86_64.zip",
        "cadd3204e728a35d3f13adb7fd0d7902636b79f6b95c40c265eb73b6c35329e4",
    ),
    ("linux", "arm64"): (
        f"Godot_v{GODOT_VERSION}-stable_linux.arm64.zip",
        "5dd0d86405cf7e8adf79fb6377b38ba682a2846cb378ffe5364f38c01ad29b9d",
    ),
    ("windows", "x86_64"): (
        f"Godot_v{GODOT_VERSION}-stable_win64.exe.zip",
        "731980f9608d61333e5baf54a2ef17210acc7a538446c0cb9969f002aca1e953",
    ),
    ("windows", "arm64"): (
        f"Godot_v{GODOT_VERSION}-stable_windows_arm64.exe.zip",
        "2d206e0b1d2366eb3508971c670c52333dcd7218b818102d77ca7ceaa86892ba",
    ),
    ("macos", "universal"): (
        f"Godot_v{GODOT_VERSION}-stable_macos.universal.zip",
        "c58a24e31d720be9d62f60cb5627c4e695fb72f21b0cfe1bc9ccaa9a3b3ba63e",
    ),
}

IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


class Report:
    """Collects one line per step for the final summary."""

    def __init__(self) -> None:
        self.rows = []  # (status, name, detail)

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def print(self) -> None:
        print()
        print("=" * 64)
        print(" SandboxAI setup summary")
        print("=" * 64)
        for status, name, detail in self.rows:
            print(f" [{status:^4}] {name:<18} {detail}")
        print("=" * 64)


def step(title: str) -> None:
    print()
    print(f"==> {title}")


def info(message: str) -> None:
    print(f"    {message}")


def warn(message: str) -> None:
    print(f"    WARNING: {message}")


def fail(message: str, hint: str = "") -> None:
    print()
    print(f"ERROR: {message}")
    if hint:
        for line in hint.splitlines():
            print(f"       {line}")
    sys.exit(1)


def run(command, *, check: bool = True, capture: bool = False, timeout: float | None = None):
    """Runs a command, echoing it; returns the CompletedProcess."""
    info("$ " + " ".join(str(part) for part in command))
    try:
        return subprocess.run(
            [str(part) for part in command],
            cwd=str(ROOT),
            check=check,
            text=True,
            capture_output=capture,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        if capture and (exc.stdout or exc.stderr):
            print((exc.stdout or "") + (exc.stderr or ""))
        raise


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def host_platform() -> tuple[str, str]:
    """(os, arch) in the GODOT_ASSETS key vocabulary."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system.startswith("win"):
        os_name = "windows"
    elif system == "darwin":
        return ("macos", "universal")
    else:
        os_name = "linux"
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64", "armv8l"):
        arch = "arm64"
    else:
        arch = machine
    return (os_name, arch)


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def linux_package_hint(package: str) -> str:
    if shutil.which("apt-get"):
        return f"sudo apt-get install {package}"
    if shutil.which("dnf"):
        return f"sudo dnf install {package}"
    if shutil.which("pacman"):
        return f"sudo pacman -S {package}"
    return f"install the '{package}' package with your system package manager"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def check_python(report: Report) -> None:
    step("Checking Python")
    version = sys.version_info
    info(f"Python {version.major}.{version.minor}.{version.micro} at {sys.executable}")
    if version[:2] < MIN_PYTHON:
        fail(
            f"SandboxAI needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.",
            "Windows: install it from https://www.python.org/downloads/ (tick 'tcl/tk').\n"
            "Linux:   " + linux_package_hint("python3.11 python3.11-venv python3.11-tk"),
        )
    report.add("ok", "Python", f"{version.major}.{version.minor}.{version.micro}")


def check_tkinter(report: Report) -> bool:
    step("Checking Tkinter (needed by the Control Center window)")
    try:
        import tkinter  # noqa: F401
    except ImportError:
        if IS_WINDOWS:
            hint = "re-run the python.org installer and enable 'tcl/tk and IDLE'"
        elif platform.system() == "Darwin":
            hint = "use the python.org installer or 'brew install python-tk'"
        else:
            minor = f"{sys.version_info.major}.{sys.version_info.minor}"
            hint = linux_package_hint(f"python{minor}-tk")
        warn("Tkinter is missing - training/CLI work, the Control Center window does not.")
        info(f"Fix: {hint}")
        report.add("warn", "Tkinter", f"missing - {hint}")
        return False
    info("Tkinter is available.")
    report.add("ok", "Tkinter", "available")
    return True


def ensure_venv(report: Report, recreate: bool) -> Path:
    step("Preparing the virtual environment (.venv)")
    python = venv_python()
    if recreate and VENV_DIR.exists():
        info("--repair: removing the existing .venv")
        shutil.rmtree(VENV_DIR)
    if python.is_file():
        probe = subprocess.run(
            [str(python), "-c", "import sys; print(sys.version_info[:2] >= (3, 11))"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "True":
            info(f"Reusing {VENV_DIR}")
            report.add("ok", "Virtualenv", str(VENV_DIR))
            return python
        warn("The existing .venv is broken or too old; recreating it.")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        fail(
            "Could not create the virtual environment.",
            "On Debian/Ubuntu the venv module is a separate package:\n"
            + linux_package_hint(f"python{minor}-venv"),
        )
    if not python.is_file():
        fail(f"venv was created but {python} does not exist.")
    run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    report.add("ok", "Virtualenv", f"created {VENV_DIR}")
    return python


def _torch_state(python: Path) -> dict | None:
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import json, torch; print(json.dumps({'version': torch.__version__, "
            "'cuda': torch.version.cuda}))",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    try:
        return json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _remove_cuda_torch(python: Path) -> None:
    """Uninstalls a CUDA PyTorch build and the nvidia-* wheels it pulled in."""
    listing = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json"], capture_output=True, text=True
    )
    names = ["torch"]
    try:
        for entry in json.loads(listing.stdout or "[]"):
            name = str(entry.get("name", ""))
            if name.lower().startswith(("nvidia-", "triton")):
                names.append(name)
    except ValueError:
        pass
    run([python, "-m", "pip", "uninstall", "-y", *names])


def ensure_torch(report: Report, python: Path, allow_pypi: bool = False) -> None:
    step("Installing PyTorch (CPU-only build)")
    state = _torch_state(python)
    if state and not state.get("cuda"):
        info(f"PyTorch {state['version']} (CPU) is already installed.")
        report.add("ok", "PyTorch", f"{state['version']} (CPU)")
        return
    if state and state.get("cuda"):
        if allow_pypi:
            info(f"PyTorch {state['version']} is installed (CUDA build, kept: --allow-pypi-torch).")
            report.add("ok", "PyTorch", f"{state['version']} (CUDA build kept)")
            return
        info(f"Found a CUDA build ({state['version']}); replacing it with the CPU build.")
        _remove_cuda_torch(python)
    try:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                TORCH_REQUIREMENT,
                "--index-url",
                TORCH_CPU_INDEX,
            ]
        )
    except subprocess.CalledProcessError:
        # On Windows and macOS the regular PyPI wheel *is* the CPU build, so
        # it is a safe fallback there. On Linux the PyPI wheel bundles ~3 GB
        # of CUDA libraries - exactly what this setup avoids - so it is only
        # used when explicitly allowed.
        if host_platform()[0] == "linux" and not allow_pypi:
            fail(
                f"Could not install the CPU-only PyTorch from {TORCH_CPU_INDEX}.",
                "Check the network/proxy and run install.py again.\n"
                "Alternatives: --no-training (no PyTorch), or --allow-pypi-torch\n"
                "(regular PyPI build, which on Linux includes the large CUDA libraries).",
            )
        warn("The CPU index was not usable; installing PyTorch from PyPI instead.")
        run([python, "-m", "pip", "install", TORCH_REQUIREMENT])
    state = _torch_state(python)
    if not state:
        fail("PyTorch was installed but cannot be imported.")
    report.add("ok", "PyTorch", f"{state['version']} ({'CUDA' if state['cuda'] else 'CPU'})")


def ensure_project(report: Report, python: Path, extras: list[str]) -> None:
    step("Installing SandboxAI and its Python dependencies")
    spec = "." + (f"[{','.join(extras)}]" if extras else "")
    run([python, "-m", "pip", "install", "-e", spec])
    probe = subprocess.run(
        [str(python), "-c", "import sandboxai; print(sandboxai.__version__)"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        fail("sandboxai was installed but cannot be imported.", probe.stderr.strip())
    report.add("ok", "Packages", f"sandboxai {probe.stdout.strip()} [{', '.join(extras) or '-'}]")


# -- Godot -------------------------------------------------------------------


def godot_version_of(executable: str) -> str:
    """'4.7.2.stable.official...' or '' when it cannot be run."""
    try:
        completed = subprocess.run(
            [executable, "--headless", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (completed.stdout or "").splitlines():
        line = line.strip()
        if line[:1].isdigit():
            return line
    return ""


def version_matches(text: str) -> bool:
    return text.startswith(GODOT_VERSION + ".")


def _downloaded_godot(target_dir: Path) -> Path | None:
    """The bridge executable inside an extracted release, if present."""
    if not target_dir.is_dir():
        return None
    if IS_WINDOWS:
        consoles = sorted(target_dir.rglob("*_console.exe"))
        if consoles:
            return consoles[0]
        exes = sorted(target_dir.rglob("Godot*.exe"))
        return exes[0] if exes else None
    mac = target_dir / "Godot.app" / "Contents" / "MacOS" / "Godot"
    if mac.is_file():
        return mac
    for candidate in sorted(target_dir.glob(f"Godot_v{GODOT_VERSION}-stable_*")):
        if candidate.is_file() and not candidate.name.endswith(".zip"):
            return candidate
    return None


def _existing_godot_candidates() -> list[str]:
    candidates = []
    for variable in ("GODOT_PATH", "GODOT_EXECUTABLE"):
        if os.environ.get(variable):
            candidates.append(os.environ[variable])
    settings = SETTINGS_DIR / "settings.json"
    try:
        remembered = json.loads(settings.read_text(encoding="utf-8-sig")).get("godot_executable")
        if remembered:
            candidates.append(str(remembered))
    except (OSError, ValueError, AttributeError):
        pass
    for name in ("godot", "godot4", f"Godot_v{GODOT_VERSION}-stable_win64_console.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    unique = []
    for candidate in candidates:
        if candidate not in unique and (shutil.which(candidate) or Path(candidate).is_file()):
            unique.append(candidate)
    return unique


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SandboxAI-installer"})
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last > 0.25 or done == total:
                last = now
                if total:
                    percent = done * 100 // total
                    sys.stdout.write(
                        f"\r    downloading {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({percent:3d}%)"
                    )
                else:
                    sys.stdout.write(f"\r    downloading {done / 1e6:6.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    partial.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GodotDownloadError(RuntimeError):
    """The Godot build could not be fetched; the rest of the setup still runs."""


def download_godot() -> Path:
    key = host_platform()
    if key not in GODOT_ASSETS:
        fail(
            f"No official Godot {GODOT_VERSION} build is known for {key[0]}/{key[1]}.",
            "Install Godot manually and re-run with --godot /path/to/godot",
        )
    asset, expected_sha = GODOT_ASSETS[key]
    target_dir = GODOT_DIR / GODOT_VERSION
    archive = GODOT_DIR / asset
    if not (archive.is_file() and _sha256(archive) == expected_sha):
        info(f"Downloading {asset} from the official Godot GitHub release")
        try:
            _download(GODOT_RELEASE_URL + asset, archive)
        except OSError as exc:
            raise GodotDownloadError(
                f"download failed ({exc}). Download {GODOT_RELEASE_URL + asset} "
                "manually, unzip it and re-run with --godot PATH"
            ) from exc
    actual_sha = _sha256(archive)
    if actual_sha != expected_sha:
        archive.unlink(missing_ok=True)
        fail(
            "The Godot download failed its SHA-256 check and was deleted.",
            f"expected {expected_sha}\nreceived {actual_sha}",
        )
    info("SHA-256 verified.")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            extracted = Path(bundle.extract(member, target_dir))
            mode = (member.external_attr >> 16) & 0o777
            if mode:
                extracted.chmod(mode)
    archive.unlink(missing_ok=True)  # the extracted build is what we keep
    executable = _downloaded_godot(target_dir)
    if executable is None:
        fail(f"Could not find the Godot executable inside {target_dir}.")
    if not IS_WINDOWS:
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return executable


def ensure_godot(report: Report, explicit: str | None) -> str | None:
    step(f"Setting up Godot {GODOT_VERSION}")
    if explicit:
        version = godot_version_of(explicit)
        if not version:
            fail(f"--godot {explicit!r} could not be started.")
        if not version_matches(version):
            warn(f"{explicit} reports {version}; SandboxAI is tested with {GODOT_VERSION}.")
        report.add("ok" if version_matches(version) else "warn", "Godot", f"{explicit} ({version})")
        return explicit

    downloaded = _downloaded_godot(GODOT_DIR / GODOT_VERSION)
    if downloaded is not None and version_matches(godot_version_of(str(downloaded))):
        info(f"Using the previously downloaded build: {downloaded}")
        report.add("ok", "Godot", f"{downloaded}")
        return str(downloaded)

    for candidate in _existing_godot_candidates():
        version = godot_version_of(candidate)
        if version_matches(version):
            info(f"Found a matching installation: {candidate} ({version})")
            report.add("ok", "Godot", f"{candidate}")
            return candidate
        if version:
            info(f"Ignoring {candidate}: it is Godot {version}")

    try:
        executable = download_godot()
    except GodotDownloadError as exc:
        warn(str(exc))
        report.add("fail", "Godot", "download failed - see the warning above")
        return None
    version = godot_version_of(str(executable))
    if not version_matches(version):
        warn(f"The downloaded Godot reports {version or 'nothing'}; it may not run here.")
        report.add("warn", "Godot", f"{executable} (could not verify)")
    else:
        info(f"Installed {executable} ({version})")
        report.add("ok", "Godot", f"downloaded to {executable.relative_to(ROOT)}")
    return str(executable)


def validate_runtime(report: Report, python: Path, godot: str) -> None:
    step("Remembering Godot and running a short end-to-end check")
    # validate-runtime both stores the executable in .sandboxai/settings.json
    # (every later command uses it) and boots the real headless bridge.
    completed = run(
        [python, "-m", "sandboxai", "validate-runtime", "--godot-executable", godot],
        check=False,
        capture=True,
        timeout=600,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    tail = [line for line in output.strip().splitlines() if line.strip()][-6:]
    for line in tail:
        info(line)
    if completed.returncode == 0:
        report.add("ok", "Runtime check", "headless Godot bridge answered")
    else:
        report.add("fail", "Runtime check", "see the output above")


def write_marker(python: Path, godot: str | None, extras: list[str]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_MARKER.write_text(
        json.dumps(
            {
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": str(python),
                "extras": extras,
                "godot_executable": godot,
                "godot_version": GODOT_VERSION,
                "torch": "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up SandboxAI: virtualenv, CPU PyTorch, packages and Godot "
        f"{GODOT_VERSION}. Safe to run again; finished steps are skipped."
    )
    parser.add_argument(
        "--godot", metavar="PATH", help="use this Godot executable instead of finding/downloading"
    )
    parser.add_argument("--no-godot", action="store_true", help="skip the Godot step")
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="skip PyTorch/SB3 (Control Center and data tools only)",
    )
    parser.add_argument(
        "--allow-pypi-torch",
        action="store_true",
        help="if the CPU-only PyTorch index is unreachable, accept the regular PyPI build "
        "(on Linux that one bundles the large CUDA libraries)",
    )
    parser.add_argument("--dev", action="store_true", help="also install lint and GDScript tooling")
    parser.add_argument(
        "--repair", action="store_true", help="delete and recreate .venv before installing"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="skip the final validate-runtime check"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    # Our own lines and the pip/venv child output share one terminal; line
    # buffering keeps them in order when the output is piped or logged.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    os.chdir(ROOT)
    print(f"SandboxAI setup in {ROOT}")
    report = Report()

    check_python(report)
    check_tkinter(report)
    python = ensure_venv(report, args.repair)

    extras = ["test"]
    if not args.no_training:
        ensure_torch(report, python, args.allow_pypi_torch)
        extras.insert(0, "training")
    if args.dev:
        extras += ["lint", "gdscript"]
    ensure_project(report, python, extras)

    godot = None
    if args.no_godot:
        report.add("skip", "Godot", "--no-godot")
    else:
        godot = ensure_godot(report, args.godot)
        if godot and not args.skip_check:
            validate_runtime(report, python, godot)

    write_marker(python, godot, extras)
    report.print()
    failed = any(status == "fail" for status, _, _ in report.rows)
    print()
    if failed:
        print("Setup finished with problems (see above).")
    else:
        print("Setup complete. Start SandboxAI with:  python start.py")
        print("Watch a checkpoint in 3D with:         python start.py view")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
