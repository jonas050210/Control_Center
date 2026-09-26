# SandboxAI — Godot 4 + godot_rl_agents feasibility test

Goal: validate the Godot 4.3 + godot_rl_agents stack (Python 3.11, PyTorch,
SB3) on the actual Windows 11 / RTX 4060 Ti machine before building the real
SandboxAI environment. **Read `PROJECT.md` first** for project context,
boundaries and the measured-results summary.

## Why an exported `.exe` is required (root cause of the first failed attempt)

`godot-rl` 0.8.2 launches the environment itself:

```
GodotEnv(env_path=...)   # Windows: path suffix is rewritten to .exe and must exist
  -> Popen([env_path, "--port=11008", "--env_seed=0",
            "--disable-render-loop", "--headless"])      # when the window is hidden
```

- A **renamed `.bat` does not work**: `CreateProcess` needs a real PE
  executable, not a batch script with a renamed suffix.
- The **Godot editor binary is not an environment**: launched without a
  project it opens the project manager, never starts the `Sync` node, and
  Python times out on port 11008 after 60 s.
- Therefore the example project must be **exported as a standalone game**
  (`.exe` on Windows, `.x86_64` on Linux). This is also the officially
  documented godot_rl workflow. `feasibility/export_envs.py` does it.

## Components (verified combination)

| Piece | Version |
|---|---|
| OS | Windows 11 (target), Debian 12 (CI verification sandbox) |
| GPU | RTX 4060 Ti 8 GB (target), none (CI sandbox) |
| Godot | 4.3-stable (`4.3.stable.official.77dcf97d8`) |
| godot-rl (pip) | 0.8.2 (latest release) |
| stable-baselines3 | 2.4.0 |
| gymnasium | 1.0.0 |
| torch | 2.6.0+cu124 on Windows (cu124 index); CPU torch in the CI sandbox |
| Python | 3.11 |
| examples | `edbeeching/godot_rl_agents_examples` @ `d659636` (pinned, see below) |

The bundled GDScript addon inside the examples (Nov 2023 version,
`MINOR_VERSION 3` handshake) is **newer** than the one godot-rl 0.8.2 pins in
its submodule (Jan 2023) and connects fine — the version mismatch only
prints a warning. Do not "update" the addon independently of the pinned
examples commit.

## Setup (one-time, Windows 11)

```powershell
cd <repo>
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r feasibility\requirements.txt

# Godot 4.3 single exe already in repo root (Godot_v4.3-stable_win64.exe /
# Godot_v4.3-stable_win64_console.exe). Export templates must be installed
# ONCE: start the editor -> Editor / Manage Export Templates... ->
# Download and Install (4.3.stable official).

# clone the pinned examples + apply SandboxAI overlay (export presets, 84x84 camera):
python feasibility\setup_examples.py
```

`setup_examples.py` clones the examples repo at pinned commit `d659636`
(FPS example unchanged upstream since 2023-12-03, VirtualCamera since
2024-01-17; newer upstream `main` migrates other examples to Godot 4.4/4.5
and must not be used with 4.3). It then copies `feasibility/godot_overlays/`
over the checkout:

- `examples/FPS/export_presets.cfg` (new) — Windows + Linux export presets
- `examples/VirtualCamera/export_presets.cfg` (new) — same
- `examples/VirtualCamera/VirtualCamera.tscn` — SubViewport 36x36 -> **84x84**

## Export the environments

```powershell
python feasibility\export_envs.py                # both examples, current OS
python feasibility\export_envs.py --example fps  # just the FPS env
```

This runs the first-time asset import (required: the repo ships no
`.import` files — **running the project before importing produces a wall of
"missing Player / missing texture / failed scene load" errors**, which is
what broke the first manual attempt) and then exports:

- `build\fps_windows.exe` (Test A/B: raycast observations, 8 agents)
- `build\virtualcamera_windows.exe` (Test C: 84x84 RGB camera observations)

## Test A — raycast benchmark (no rendering needed)

```powershell
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --n_parallel 4 --seconds 30
```

Measures: total env steps/sec (8 agents per instance), Python-side step
calls/sec, per-instance throughput, system RAM, Godot process RSS, GPU VRAM
(via `nvidia-smi`, when present).

## Test B — short PPO training (SB3, MultiInputPolicy)

```powershell
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
```

Model is saved to `logs/ppo_feasibility.zip`. Random-policy rewards stay
near zero in this example (reward only fires on hits) — that is env design,
not a broken channel.

## Test C — pixel observations (84x84, REQUIRES the GPU window)

```powershell
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\virtualcamera_windows.exe --viz --timesteps 20000
```

`--viz` is **required** here: without it godot-rl passes
`--headless --disable-render-loop` and nothing is rendered, so the camera
sensor cannot capture. With `--viz` the game window stays visible (it can be
small/unfocused). The camera path does a GPU->CPU readback per step and
ships pixels hex-encoded over TCP — the known godot_rl_agents bottleneck;
this test measures how bad it really is on the RTX 4060 Ti.

If the connection fails on Windows with a permission error, retry with a
high port: `--port 51008` (upstream issue #225).

## Results

MEASURED values only — no estimates in this table. "CI sandbox" = Debian 12,
2 vCPU, no GPU, Godot 4.3 built from source (headless, OpenXR disabled —
slower than official builds, treat as a lower bound).

| Test | Machine | steps/sec | notes |
|---|---|---|---|
| FPS raycast, 1 instance (8 agents), editor-run | CI sandbox (prev. session) | 246 total / 31 calls | Godot process CPU-bound |
| FPS raycast, 2 instances (16 agents) | CI sandbox (prev. session) | 471 total | ~1.9x scaling on 2 vCPUs |
| PPO raycast, 28,672 steps, 2 instances | CI sandbox (prev. session) | ~410 (SB3 fps) | MultiInputPolicy, clean training |
| FPS raycast, 1 instance (8 agents), editor-run | CI sandbox (this session) | 255 total / 32 calls | Godot RSS 141 MB |
| FPS raycast, 2 instances (16 agents), editor-run | CI sandbox (this session) | 483 total / 241 per inst | RSS 284 MB combined |
| PPO raycast, 28,672 steps, 2 instances, editor-run | CI sandbox (this session) | 393 incl. learner (SB3 fps ~398) | 73 s; model saved to logs/ |
| FPS raycast, 1 instance, exported binary | CI sandbox (this session) | *fill in* | |
| FPS raycast, 2 instances, exported binary | CI sandbox (this session) | *fill in* | |
| PPO raycast, exported binary | CI sandbox (this session) | *fill in* | |
| VirtualCamera 84x84, headless (no --viz) | CI sandbox (this session) | fails by design | `get_data` on null texture — rendering must stay enabled |
| VirtualCamera 84x84 | CI sandbox | not possible | no GPU/GL — must run on the 4060 Ti |
| FPS raycast, 1 instance | Windows 4060 Ti | *fill in* | |
| FPS raycast, 4 instances | Windows 4060 Ti | *fill in* | |
| PPO 50k raycast | Windows 4060 Ti | *fill in* | |
| VirtualCamera 84x84, 1 instance | Windows 4060 Ti | *fill in* | |

RAM in the CI sandbox (previous session): ~140 MB per headless Godot FPS
instance, ~580 MB Python/SB3 side.

## Linux dev note (not needed on Windows)

On Linux the same trick as a `.bat` wrapper *does* work, because a shell
script with a shebang is executable via `Popen`:

```sh
mkdir -p build && printf '#!/bin/sh\nexec /path/to/godot --path "$PWD/examples/examples/FPS" "$@"\n' \
  > build/fps_linux.x86_64 && chmod +x build/fps_linux.x86_64
python feasibility/benchmark_env.py --env_path build/fps_linux.x86_64
```

This runs the project through the editor binary without exporting (handy for
fast iteration); exported binaries remain the canonical environments.
