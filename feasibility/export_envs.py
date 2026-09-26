#!/usr/bin/env python3
"""SandboxAI feasibility: import + export the godot_rl example environments.

godot-rl (0.8.2) can only launch a REAL exported game binary as the RL
environment (it appends `--port=... --env_seed=...` and on Windows requires
an `.exe` that CreateProcess can execute - a renamed .bat does NOT work).
This script produces those binaries with the Godot 4.3 export presets that
`feasibility/setup_examples.py` copied into the example projects:

  fps           -> build/fps_windows.exe          (Test A/B: raycast + PPO)
  virtualcamera -> build/virtualcamera_windows.exe (Test C: 84x84 pixels)

Usage (from the repo root, with the venv active):
  python feasibility/export_envs.py                          # both, current OS
  python feasibility/export_envs.py --example fps            # just the FPS env
  python feasibility/export_envs.py --platform windows       # force platform
  python feasibility/export_envs.py --import-only            # no export (dev)

Godot binary resolution order:
  --godot <path>  >  GODOT_BIN env var  >  Godot exe in the repo root.

Export templates must be installed ONCE per machine (any Godot 4.3):
  open the editor -> Editor/Manage Export Templates... -> Download and Install
  (or place them in %APPDATA%\\Godot\\export_templates\\4.3.stable\\ on Windows,
   ~/.local/share/godot/export_templates/4.3.stable/ on Linux).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

EXAMPLES = {
    "fps": {
        "project": ROOT / "examples" / "examples" / "FPS",
        "preset": {"windows": "Windows Desktop", "linux": "Linux/X11"},
        "output": {"windows": "fps_windows.exe", "linux": "fps_linux.x86_64"},
    },
    "virtualcamera": {
        "project": ROOT / "examples" / "examples" / "VirtualCamera",
        "preset": {"windows": "Windows Desktop", "linux": "Linux/X11"},
        "output": {"windows": "virtualcamera_windows.exe", "linux": "virtualcamera_linux.x86_64"},
    },
}

GODOT_NAMES = [
    "Godot_v4.3-stable_win64_console.exe",  # Windows console build (best for CLI)
    "Godot_v4.3-stable_win64.exe",          # Windows standard build
    "Godot_v4.3-stable_linux.x86_64",       # Linux official build
]


def find_godot(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("GODOT_BIN"):
        candidates.append(Path(os.environ["GODOT_BIN"]))
    candidates += [ROOT / n for n in GODOT_NAMES]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    sys.exit(
        "Could not find a Godot 4.3 executable.\n"
        "Pass --godot <path> or set GODOT_BIN, or place one of these in the repo root:\n  "
        + "\n  ".join(GODOT_NAMES)
    )


def run_godot(godot: Path, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    cmd = [str(godot), "--headless", *args]
    print(f"+ {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.perf_counter() - t0
    out = (proc.stdout + "\n" + proc.stderr).strip()
    if out:
        # show the last lines that matter, drop pure progress spam
        lines = [ln for ln in out.splitlines() if ln.strip()]
        print("\n".join(lines[-12:]))
    print(f"  (exit {proc.returncode}, {dt:.0f}s)")
    return proc


def import_project(godot: Path, project: Path) -> None:
    print(f"\n== importing {project.name} (first-time asset import) ==")
    # Run the import pass twice: the first pass can log script errors for
    # resources that preload files which are still being imported (e.g. the
    # godot_rl addon preloading its icon.png). The second pass must be clean.
    for attempt in (1, 2):
        proc = run_godot(godot, ["--path", str(project), "--import", "--quit"])
        if proc.returncode != 0:
            sys.exit(f"Godot import failed for {project} (pass {attempt}, exit {proc.returncode}).")


def export_project(godot: Path, project: Path, preset: str, out_name: str) -> Path:
    out_path = BUILD / out_name
    print(f'\n== exporting {project.name} [{preset}] -> {out_path} ==')
    proc = run_godot(godot, ["--path", str(project), "--export-release", preset, str(out_path)])
    out = (proc.stdout + "\n" + proc.stderr).lower()
    if "no export template found" in out:
        sys.exit(
            "Export templates for Godot 4.3 are not installed.\n"
            "Fix: open the Godot editor once -> Editor/Manage Export Templates... -> "
            "Download and Install (4.3.stable), then re-run this script."
        )
    if proc.returncode != 0 or not out_path.is_file():
        sys.exit(f"Export failed for {project.name} (exit {proc.returncode}).")
    size_mb = out_path.stat().st_size / 1e6
    print(f"exported OK: {out_path} ({size_mb:.0f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--godot", default=None, help="path to the Godot 4.3 executable")
    ap.add_argument("--example", choices=sorted(EXAMPLES), default=None, help="example to export (default: both)")
    ap.add_argument("--platform", choices=["windows", "linux"], default=None,
                    help="export platform (default: current OS)")
    ap.add_argument("--import-only", action="store_true", help="only run the first-time import, do not export")
    args = ap.parse_args()

    godot = find_godot(args.godot)
    platform = args.platform or ("windows" if sys.platform == "win32" else "linux")
    names = [args.example] if args.example else sorted(EXAMPLES)

    print(f"godot binary : {godot}")
    print(f"platform     : {platform}")

    for name in names:
        cfg = EXAMPLES[name]
        project = cfg["project"]
        if not project.is_dir():
            sys.exit(
                f"{project} does not exist.\n"
                f"Run: python feasibility/setup_examples.py"
            )
        import_project(godot, project)
        if args.import_only:
            continue
        export_project(godot, project, cfg["preset"][platform], cfg["output"][platform])

    if args.import_only:
        print("\nimport-only done.")
    else:
        print(
            f"\nAll done. Environments are in {BUILD}. Run e.g.:\n"
            f"  python feasibility/benchmark_env.py --env_path build{os.sep}{EXAMPLES[names[0]]['output'][platform]}"
        )


if __name__ == "__main__":
    main()
