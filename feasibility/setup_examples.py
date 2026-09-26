#!/usr/bin/env python3
"""SandboxAI feasibility: reproducible checkout of the godot_rl_agents_examples.

The examples repository is third-party and intentionally NOT tracked in git
(see .gitignore: `examples/`).  To keep the feasibility tests reproducible,
this script:

  1. clones `edbeeching/godot_rl_agents_examples` at a PINNED commit,
  2. verifies the checked-out commit hash,
  3. copies the SandboxAI overlay files (export presets, 84x84 camera) from
     `feasibility/godot_overlays/` over the checkout.

Pinned commit: d65963648439167f4902043376321c15d3df0e3a (2026-01-22)
  - `examples/FPS`          last changed upstream 2023-12-03 (Godot 4.1-era
    project, GDScript, verified working with Godot 4.3-stable and godot-rl
    0.8.2 - see PROJECT.md "Feasibility status").
  - `examples/VirtualCamera` last changed upstream 2024-01-17.
  - Newer upstream `main` may migrate examples to Godot 4.4/4.5; do NOT
    track main, always use the pinned commit (or re-verify a new one).

Usage:
  python feasibility/setup_examples.py            # clone / verify / overlay
  python feasibility/setup_examples.py --force    # discard local clone, re-clone

Requires: git on PATH.  No Python dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/edbeeching/godot_rl_agents_examples.git"
PINNED_SHA = "d65963648439167f4902043376321c15d3df0e3a"

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "examples"
OVERLAY = Path(__file__).resolve().parent / "godot_overlays"

# Sanity checks that the checkout actually contains what the tests need.
REQUIRED_PATHS = [
    "examples/FPS/project.godot",
    "examples/FPS/train.tscn",
    "examples/FPS/addons/godot_rl_agents/sync.gd",
    "examples/VirtualCamera/project.godot",
    "examples/VirtualCamera/Env.tscn",
    "examples/VirtualCamera/addons/godot_rl_agents/sync.gd",
]


def run_git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)
    return proc


def head_sha(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    proc = run_git(["rev-parse", "HEAD"], cwd=path, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def clone_pinned() -> None:
    print(f"cloning {REPO_URL} at pinned commit {PINNED_SHA[:12]} ...")
    run_git(["clone", "--no-checkout", REPO_URL, str(DEST)])
    run_git(["checkout", PINNED_SHA], cwd=DEST)
    print("clone complete.")


def reclone_with_backup() -> None:
    backup = ROOT / f"examples_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"moving existing {DEST} -> {backup}")
    DEST.rename(backup)
    clone_pinned()


def verify() -> None:
    sha = head_sha(DEST)
    if sha != PINNED_SHA:
        sys.exit(
            f"examples checkout is at {sha}, expected pinned {PINNED_SHA}.\n"
            f"Run: python {Path(__file__).name} --force"
        )
    missing = [p for p in REQUIRED_PATHS if not (DEST / p).is_file()]
    if missing:
        sys.exit(f"examples checkout is incomplete, missing:\n  " + "\n  ".join(missing))
    print(f"verified: examples checkout is exactly {PINNED_SHA[:12]} and complete.")


def apply_overlay() -> None:
    n = 0
    for src in sorted(OVERLAY.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(OVERLAY)
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"overlay: {rel}")
        n += 1
    print(f"applied {n} overlay file(s).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="discard the existing examples/ clone and re-clone")
    args = ap.parse_args()

    if args.force and DEST.exists():
        reclone_with_backup()
    elif not DEST.exists():
        clone_pinned()
    elif head_sha(DEST) is None:
        print(f"{DEST} exists but is not a git checkout of the examples repo.")
        reclone_with_backup()
    else:
        sha = head_sha(DEST)
        if sha != PINNED_SHA:
            print(f"examples checkout is at {sha}, fetching pinned {PINNED_SHA[:12]} ...")
            run_git(["fetch", "origin", PINNED_SHA], cwd=DEST)
            run_git(["checkout", PINNED_SHA], cwd=DEST)
        else:
            print("examples already at pinned commit.")

    verify()
    apply_overlay()

    print(
        "\nNext steps (see feasibility/README.md):\n"
        "  1. python feasibility/export_envs.py --example fps --platform windows\n"
        "     (needs Godot 4.3 + export templates installed once)\n"
        "  2. python feasibility/benchmark_env.py --env_path build\\fps_windows.exe\n"
    )


if __name__ == "__main__":
    main()
