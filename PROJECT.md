# SandboxAI — PROJECT.md (single source of truth)

> Read this first. This file is the authoritative context for all AI
> coding/research sessions on this repository. Update it when facts change;
> keep results labeled VERIFIED / MEASURED / ESTIMATED / UNKNOWN and never
> invent measurements.

Last updated: 2026-09-26 (after Windows-launch fix + re-verification of the M0
feasibility kit).

## 1. What this project is

SandboxAI is a vision-based AI learning project. The pipeline:

```
human plays (Roblox "TTK Testing [HARDPOINT]")
      │  screen recording + input logging (manual data capture only)
      ▼
imitation learning / behavioral cloning (PyTorch)
      │  BC policy weights
      ▼
AI plays inside OUR OWN controlled sandbox (Godot 4 + godot_rl_agents)
      │  PPO / later APPO fine-tuning
      ▼
reinforcement learning improves the policy
      ▼
later: GUI / control center for running, watching and managing the AI
```

The sandbox is a small, controlled, tactical-FPS-like environment *inspired by*
the gameplay structure of TTK (positioning, pre-aim, angle discipline, ADS,
recoil, movement, mouse control) — **not** a 1:1 copy of its maps/assets.

## 2. TTK's role — strict boundaries (DO NOT UNDO)

TTK Testing [HARDPOINT] on Roblox is **only a manual gameplay/data source**
for imitation learning (screen + human input recordings).

Explicitly OUT OF SCOPE, permanently unless the owner says otherwise:
- Automating or botting the public Roblox game
- Injecting into the Roblox client, reading its memory, bypassing anti-cheat
- Using Roblox as the online RL environment

Why Roblox can't be the RL env anyway: real-time only, nondeterministic, no
reset/state/time-acceleration interfaces — unusable for high-throughput RL.

## 3. Hardware / software constraints

| Constraint | Value |
|---|---|
| OS (target machine) | Windows 11 |
| CPU | Intel i7-12700F |
| GPU | RTX 4060 Ti, **8 GB VRAM** |
| RAM | 32 GB |
| Python | 3.11 |
| Budget | Local / free / open-source. No paid APIs/subscriptions unless explicitly justified. |

## 4. Technology choices (decided, with reasons)

Chosen after a comparative study (Godot vs Unity ML-Agents vs Unreal vs custom
Python vs ViZDoom vs GPU-batch simulators):

| Piece | Choice | Why |
|---|---|---|
| Engine / sandbox | **Godot 4.3-stable** (`official.77dcf97d8`) | MIT, free, light on 8 GB GPU, low-poly visuals close to Roblox style (helps BC transfer), sandbox doubles as later GUI/game |
| RL bridge | **godot_rl_agents** (pip `godot-rl` **0.8.2**, latest release) | Gymnasium interface, Python 3.11, SB3/SampleFactory/CleanRL wrappers, ONNX export path |
| RL algo | **Stable-Baselines3 PPO 2.4.0** first; Sample Factory/APPO to evaluate later if throughput demands | simplicity first |
| ML framework | **PyTorch** (2.6.0+cu124 on Windows) | project preference |
| BC model | Custom PyTorch (suggested start: ResNet-18 or IMPALA-style CNN + ConvGRU) trained on Roblox recordings | research finding |
| Deployment | ONNX later, for in-engine inference / GUI phase | optional |

Rejected alternatives (do not revisit without new evidence): Unity ML-Agents
(Python 3.10 pin, rigid trainer), Unreal (too heavy for solo dev), ViZDoom
(visual domain gap vs Roblox, dead end for GUI), custom 3D Python env (months
of renderer work), Madrona-class GPU sims (C++/Linux research tooling).

Version-matching note (VERIFIED): the GDScript addon bundled inside the
examples repo (Nov 2023, handshake `MINOR_VERSION 3`) is **newer** than the
one the `godot-rl` 0.8.2 PyPI package pins in its own submodule (Jan 2023).
They connect fine — the handshake prints `WARNING: minor version mismatch 7 3`
and continues. Do not "update" the addon independently of the pinned examples
commit.

### Observation/action design (initial, from TTK research — estimates, not yet validated)
- Observation: ~160x120 RGB at ~15 Hz for BC on real recordings; sandbox RL
  benchmarked first at 84x84. Hybrid visual + vector/raycast observations are
  acceptable during development (debugging, reward shaping, RL fallback).
- Mouse movement: discretized into bins initially, not direct regression.
- Restricted action space initially. Audio is a later/optional modality.

## 5. Feasibility status (M0)

### What was broken on Windows, and why (root causes, VERIFIED)

The first Windows attempt (`python feasibility\benchmark_env.py --env_path
run_fps.bat ...`) failed for three separate reasons, all now verified against
source and reproduced in the CI sandbox:

1. **godot-rl requires a real exported game executable.**
   `GodotEnv` (godot-rl 0.8.2) rewrites the env path suffix to `.exe` on
   Windows, requires the file to exist, and launches it with
   `Popen([env_path, "--port=...", "--env_seed=...", "--disable-render-loop",
   "--headless"])` and `shell=False`. Consequences (VERIFIED from source):
   - a `.bat` renamed to `.exe` cannot execute (`CreateProcess` needs a PE
     binary) — the Linux wrapper-script trick does not translate to Windows;
   - passing the **Godot editor exe** passes the file check but opens the
     editor/project manager with no project and no RL `Sync` node → nothing
     connects to 127.0.0.1:11008 → Python times out after 60 s.
   The only correct launch shape is a **standalone exported build of the
   example project** (this is also the officially documented godot_rl flow).
2. **The example project was run before Godot's first-time asset import.**
   The examples repo ships zero `.import` files (upstream gitignores them).
   Running the project un-imported produces exactly the reported error wall:
   `Could not find type "Player"/"PlayerHitBox"/"CharacterModel"/"Projectile"/
   "AIController3D"/"RayCastSensor3D"` (scripts fail to load → their
   `class_name`s never register) plus `No loader found for resource:
   ...texture_xx.png` and cascading scene-load failures. REPRODUCED in the CI
   sandbox: un-imported run → identical error wall; after `--import` → clean
   boot, Sync node connects.
3. **Unpinned examples checkout.** `godot_rl_agents_examples` `main` has moved
   on (other examples now target Godot 4.4/4.5, C#). The two examples we use
   are untouched upstream (`FPS` since 2023-12-03, `VirtualCamera` since
   2024-01-17) and work with 4.3, but the checkout is now pinned to commit
   `d65963648439167f4902043376321c15d3df0e3a` for reproducibility.

### What was fixed / implemented (this session)

- `feasibility/setup_examples.py` (new): clones the examples repo at the
  pinned commit, verifies the hash and required files, and applies the
  tracked SandboxAI overlay (`feasibility/godot_overlays/`) onto it.
- `feasibility/godot_overlays/` (new): export presets for FPS and
  VirtualCamera (Windows `.exe` + Linux `.x86_64`, embedded PCK) and the
  VirtualCamera 84x84 camera (SubViewport 36x36 → 84x84, obs shape [3,84,84]).
  The VirtualCamera presets also exclude the orphan `Model.tscn`, which
  references `res://90s_dad/scene.gltf` — a file never committed upstream
  (VERIFIED: nothing references `Model.tscn`; without the exclusion every
  export logs a resource-not-found error).
- `feasibility/export_envs.py` (new): runs the required two-pass headless
  import and exports `build\fps_windows.exe` / `build\virtualcamera_windows.exe`
  (the launchable RL environments godot-rl expects). Auto-finds the Godot
  binary (`--godot` / `GODOT_BIN` / repo-root exes) and explains the
  export-template requirement if missing.
- `feasibility/benchmark_env.py` (updated): now also reports agents/instance,
  per-instance throughput, Godot process RSS, GPU name, and accepts `--port`
  (Windows low-port permission workaround, upstream issue #225).
- `feasibility/train_ppo.py` (updated): fixed crash when saving to `logs/`
  (dir did not exist on a fresh clone); accepts `--port`.
- `feasibility/requirements.txt`: pinned to the verified combination
  (godot-rl 0.8.2, sb3 2.4.0, gymnasium 1.0.0).
- `.gitignore`: `build/` added (exported binaries are generated artifacts).
- Removed the broken `run_fps.bat` guidance from the docs (a `.bat` can never
  satisfy godot-rl on Windows).

### MEASURED results

All rows MEASURED in the CI sandbox (Debian 12, 2 weak vCPUs, **no GPU**;
Godot 4.3 built from source at the same commit as official 4.3-stable,
`77dcf97d8`, OpenXR disabled) — a lower bound, NOT the target machine.

Editor-binary runs (project run through the editor binary via wrapper):

| Test | steps/sec | notes |
|---|---|---|
| FPS raycast, 1 instance (8 agents) | 255 total / 32 calls | Godot RSS 141 MB |
| FPS raycast, 2 instances (16 agents) | 483 total / 241 per inst | RSS 284 MB, ~1.9x scaling |
| PPO 28,672 steps, 2 instances | 393 incl. learner (SB3 fps ~398) | 73 s, MultiInputPolicy, saved to logs/ |

Exported-binary runs (release template build — **the canonical reference**,
this is the exact launch path Windows uses):

| Test (exported binary) | steps/sec | notes |
|---|---|---|
| FPS raycast, 1 instance (8 agents) | 299 total / 37 calls | Godot RSS 118 MB |
| FPS raycast, 2 instances (16 agents) | 464 total / 232 per inst | RSS 235 MB |
| PPO 28,672 steps, 2 instances | 480 incl. learner (~490 SB3 fps) | 60 s |
| VirtualCamera 84x84, connection + obs space | n/a (pixels need GPU) | connects; `obs = Dict('camera_2d': Box(0,255,(3,84,84),uint8))`, 16 agents/instance |

Previous-session numbers from the same sandbox class (246 / 471 / ~410 fps,
~140 MB per instance, ~580 MB Python side) are consistent with the above.

### NOT yet measured (UNKNOWN — the current next step)
- **All Windows / RTX 4060 Ti numbers.** The owner must run Tests A-C on the
  target machine (exact commands below). The GPU is *expected* to be
  sufficient (ESTIMATE) — unproven until measured.
- **Pixel observations (Test C)** could not run in the CI sandbox (no
  GPU/GL/Vulkan). VERIFIED: headless pixel obs fail with `Cannot call method
  'get_data' on a null value` (ViewportTexture has no image without
  rendering) → the pixel benchmark **must** run with `--viz` on Windows.
- VirtualCamera 84x84 throughput, PPO-on-pixels speed, and VRAM usage are
  all UNKNOWN until the Windows run.

### Known quirks (VERIFIED)
- godot-rl 0.8.2 handshake prints `WARNING: minor version mismatch 7 3`
  (Python 0.8.2 vs examples' Nov-2023 addon). Harmless; do not "fix" it.
- godot-rl's FPS step = 8 physics ticks by env design (`action_repeat`
  default in `sync.gd`).
- In the FPS example, reward fires only on hits and `done` only on death —
  near-zero rewards under a random policy are env design, not a bug.
- Windows: if the TCP connection fails with a permission error, retry with a
  high port (`--port 51008`, upstream issue #225).
- Launching an exported env exe manually (no Python server) runs the game in
  human-play mode — the Sync node falls back to `heuristic: human`.
- Pixel path risk (ESTIMATE): per-step GPU→CPU readback + hex-encoded pixels
  over TCP is the known godot_rl_agents bottleneck; Test C measures it.

## 6. Repository structure

```
PROJECT.md                    <- this file (single source of truth)
README.md                     <- stub
.gitignore                    <- excludes .venv/, tools/, examples/, logs/, build/
feasibility/
  README.md                   <- RUNBOOK: setup + Tests A/B/C + results table
  requirements.txt            <- pinned Python deps (Python 3.11)
  setup_examples.py           <- clone pinned examples + apply overlay
  export_envs.py              <- headless import + export env executables
  benchmark_env.py            <- env steps/sec + RAM/VRAM benchmark
  train_ppo.py                <- short SB3 PPO training run
  godot_overlays/examples/    <- files copied over the examples clone:
        FPS/export_presets.cfg          (Windows + Linux export presets)
        VirtualCamera/export_presets.cfg
        VirtualCamera/VirtualCamera.tscn (SubViewport 36x36 -> 84x84)
```

Not in git (transient, per-machine): `.venv/`, `tools/`, `examples/` (pinned
clone of godot_rl_agents_examples), `build/` (exported env binaries),
`logs/`, the Godot editor executables in the repo root.

## 7. Exact commands (Windows 11, from the repo root)

One-time setup:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r feasibility\requirements.txt
# Export templates, once: start Godot_v4.3-stable_win64.exe ->
#   Editor / Manage Export Templates... -> Download and Install (4.3.stable)
python feasibility\setup_examples.py
python feasibility\export_envs.py
```

Test A — raycast benchmark (FPS example, 8 agents/instance):

```powershell
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --n_parallel 4 --seconds 30
```

Test B — PPO (SB3, MultiInputPolicy):

```powershell
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
```

Test C — pixel observations 84x84 (VirtualCamera; `--viz` REQUIRED):

```powershell
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\virtualcamera_windows.exe --viz --timesteps 20000
```

## 8. Roadmap (phases may change — document changes here with reasons)

- **M0 Feasibility** — Linux side verified twice (connection, raycast
  throughput, PPO, parallel instances, export pipeline). Remaining: the
  Windows/RTX 4060 Ti measurements above, above all Test C (pixel).
- **M1 Minimal sandbox** — small greybox tactical-FPS env in Godot (own map,
  TTK-inspired layout). DO NOT START before M0 Windows results are in.
- **M2 Human data capture** — screen + input recording pipeline for TTK
  gameplay (manual play only), dataset format.
- **M3 Behavioral cloning** — train BC model on recordings.
- **M4 Closed-loop evaluation** — BC policy plays in sandbox.
- **M5 DAgger / RL** — PPO (later possibly APPO) fine-tuning in sandbox.
- **M6 GUI / control center** — visualization and management.

## 9. Decisions future agents must NOT undo without explicit justification

1. No automation/injection/memory-reading of the public Roblox game. Ever.
2. Sandbox = our own Godot environment, not Roblox, not a TTK map copy.
3. Engine choice: Godot 4 + godot_rl_agents (revisit only with new evidence,
   e.g. pixel throughput on Windows proving unworkable).
4. Python 3.11 + PyTorch + SB3-first. Local/free/open-source; no paid services.
5. Keep feasibility scripts minimal — no premature framework-building.
6. The RL environment is always an **exported game binary**, never a wrapper
   script/bat and never the editor executable.
7. The examples checkout stays pinned (`d659636…`); changing the pin requires
   re-verifying the compatibility notes in section 4.
8. This session's git branch workflow: work happens on the Arena-managed
   branch; don't force-push or rewrite history.

## 10. Open questions / risks

- Pixel obs steps/sec on Windows + 4060 Ti (M0 blocker). Fallback if too slow:
  hybrid obs (pixels for BC, raycasts for RL) or lower res / frame-skip.
- Visual domain gap Roblox↔Godot sandbox: BC transfer will likely need
  augmentation/fine-tuning; zero-shot transfer is not assumed.
- godot_rl_agents maintenance pace is slow (single maintainer, MIT — forkable).
- Godot physics is resettable/seedable but not guaranteed bit-exact
  deterministic across runs/machines; don't build evaluation on exact replay.

## 11. For the next AI session

- **State**: The feasibility kit is complete and re-verified on Linux,
  including the export pipeline. The Windows runbook (`feasibility/README.md`)
  is ready. Windows numbers are still UNKNOWN.
- **Immediate task**: the owner runs Tests A-C on the Windows 11 / RTX 4060 Ti
  machine with the exact commands in section 7 and fills in the results table
  in `feasibility/README.md`.
- **Then**: with real numbers, decide pixel-RL vs hybrid-obs strategy and
  green-light M1 (minimal greybox sandbox env in Godot).
- **Don'ts**: don't build the full TTK-inspired sandbox yet; don't add
  dependencies; don't touch Roblox automation; don't restructure the repo;
  don't replace the exported-binary launch flow with wrapper scripts.
- When you change anything material (versions, results, decisions), update
  this file in the same commit.
