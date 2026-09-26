# SandboxAI — Godot 4 + godot_rl_agents feasibility test

Goal: validate the Godot 4 + godot_rl_agents stack (Python 3.11, PyTorch, SB3)
before building the real SandboxAI environment.

## Components

| Piece | Version |
|---|---|
| Godot | 4.3-stable |
| godot-rl (pip) | 0.8.2 |
| stable-baselines3 | 2.4.x |
| gymnasium | 1.0.0 |
| Python | 3.11 |
| Env (raycast test) | official `FPS` example from `edbeeching/godot_rl_agents_examples` |
| Env (pixel test) | official `VirtualCamera` example (RGBCameraSensor3D + SubViewport) |

## Setup (Windows 11, RTX 4060 Ti)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r feasibility\requirements.txt

# Godot 4.3 (no install, single exe):
# https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_win64.exe.zip

# examples:
git clone https://github.com/edbeeching/godot_rl_agents_examples
```

Open `examples/FPS` once in the Godot editor (imports assets), or run:

```powershell
Godot_v4.3-stable_win64_console.exe --headless --path examples\FPS --import --quit
```

Create `run_fps.bat` (used as `--env_path`):

```bat
@echo off
"C:\path\to\Godot_v4.3-stable_win64_console.exe" --path "C:\path\to\examples\FPS" %*
```

## Test 1 — connection + raw throughput (raycast obs, headless)

```powershell
python feasibility\benchmark_env.py --env_path run_fps.bat --speedup 30 --seconds 30
python feasibility\benchmark_env.py --env_path run_fps.bat --speedup 30 --n_parallel 4
```

## Test 2 — short PPO training (raycast obs)

```powershell
python feasibility\train_ppo.py --env_path run_fps.bat --timesteps 50000 --n_parallel 2
```

## Test 3 — pixel observations (~84x84)

Pixel obs need actual rendering — do NOT pass `--headless`; the game window
must exist (it may be small/unfocused). In `examples/VirtualCamera`:

1. Open `VirtualCamera.tscn`, set the `SubViewport` size from `36x36` to `84x84`.
2. Create `run_cam.bat` like `run_fps.bat` but pointing at `examples\VirtualCamera`.
3. Run with the window visible:

```powershell
python feasibility\benchmark_env.py --env_path run_cam.bat --viz --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path run_cam.bat --viz --timesteps 20000
```

Note: the camera path does a GPU->CPU readback per step and ships pixels
hex-encoded over TCP — this is the known godot_rl_agents bottleneck. Expect
pixel steps/sec well below raycast steps/sec.

## Results

Reference numbers from the Linux CI sandbox the stack was first validated in
(Debian 12, 2 weak vCPUs, NO GPU, Godot 4.3 built from source, headless).
These are a *lower bound* — a desktop CPU + RTX 4060 Ti should be several
times faster per instance and scale to more parallel instances.

| Test | Machine | steps/sec | notes |
|---|---|---|---|
| FPS raycast, 1 instance (8 agents) | sandbox 2 vCPU | 246 total / 31 calls | Godot process 91% of one core (GDScript sensors + JSON) |
| FPS raycast, 2 instances (16 agents) | sandbox 2 vCPU | 471 total | ~1.9x scaling, near-linear |
| PPO raycast, 25k steps, 2 instances | sandbox 2 vCPU | ~410 (SB3 fps) | trains cleanly, MultiInputPolicy, obs 2250-dim |
| VirtualCamera 84x84 | sandbox | n/a | impossible without GPU/GL - run on Windows |
| FPS raycast, 1 instance | Windows 4060 Ti | *fill in* | |
| FPS raycast, 4 instances | Windows 4060 Ti | *fill in* | |
| PPO 50k raycast | Windows 4060 Ti | *fill in* | |
| VirtualCamera 84x84, 1 instance | Windows 4060 Ti | *fill in* | |

RAM footprint measured in sandbox: ~140 MB per headless Godot FPS instance,
~580 MB for the Python/SB3 side. Total under 1 GB for 2 instances + trainer.

Known quirks found:
- `godot_rl_agents` requires the env executable to end in `.x86_64` (Linux),
  `.exe` (Windows) - a wrapper script/bat must carry that suffix.
- `gdrl.env_from_hub` needs git-lfs; cloning `godot_rl_agents_examples` from
  GitHub is simpler.
- In the FPS example, reward fires only on hits and `done` only on death
  (`player.gd`) - random-policy smoke tests show all-zero rewards; that is
  env design, not a broken channel.
- godot-rl 0.8.2 + gymnasium 1.0 + sb3 2.4 + Python 3.11: no compatibility
  issues observed.
