"""Import and export the self-contained Godot tactical sandbox."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_godot(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for name in ("godot4", "godot", "Godot_v4.3-stable_win64.exe", "Godot_v4.3-stable_win64_console.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
        candidates.append(Path(name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Godot 4 executable not found; pass --godot or add Godot to PATH")


def export_sandbox(godot: Optional[str] = None, preset: Optional[str] = None) -> Path:
    executable = find_godot(godot)
    project = Path(__file__).resolve().parent / "godot_project"
    if preset is None:
        import platform

        preset = "Windows Desktop" if platform.system() == "Windows" else "Linux"
    output = (
        Path("build/sandbox_windows.exe")
        if preset == "Windows Desktop"
        else Path("build/sandbox_linux.x86_64")
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(executable), "--headless", "--path", str(project), "--editor", "--quit-after", "2"],
        check=True,
    )
    subprocess.run(
        [str(executable), "--headless", "--path", str(project), "--export-release", preset, str(output)],
        check=True,
    )
    if not output.is_file():
        raise RuntimeError(f"Godot reported success but export is missing: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SandboxAI's Godot arena")
    parser.add_argument("--godot", default=None)
    parser.add_argument("--preset", choices=["Windows Desktop", "Linux"], default=None)
    args = parser.parse_args()
    print(export_sandbox(args.godot, args.preset))


if __name__ == "__main__":
    main()
