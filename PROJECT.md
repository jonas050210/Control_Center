# SandboxAI — PROJECT.md (single source of truth)

> Read this first. This file is the authoritative context for all AI
> coding/research sessions on this repository. Update it when facts change;
> keep results labeled VERIFIED / MEASURED / ESTIMATED / UNKNOWN and never
> invent measurements.

Last updated: 2026-09-27 (M1 completion: versioned dataset schema, external screen capture,
OS-level input logging, microsecond timestamp synchronization, mouse discretization binning,
synthetic mock recording, dataset validation/inspection tools, and 37 automated tests; see section 6).

## 1. What this project is

SandboxAI is a vision-based AI learning project. The pipeline:

```
human plays (Roblox "TTK Testing [HARDPOINT]")
      │  screen recording + input logging (manual data capture only — M1 Pipeline)
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
| Data capture | **MSS + Pynput + PIL** (pure Python, cross-platform, non-invasive) | zero-overhead, no C++ compilation, native Windows hook support |
| Deployment | ONNX later, for in-engine inference / GUI phase | optional |

---

## 5. M1 Architecture: Human Data Capture Pipeline

### Pipeline Overview ("KI guckt erst zu und lernt bei mir dazu")

The M1 data pipeline records human gameplay from the outside (desktop screen grabber + OS-level input hooks) without interfering with the game process or anti-cheat.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Screen Capture (MSS)        │       │   OS Input Listener (Pynput)    │
│  - Captures viewport or ROI     │       │  - Asynchronous event queue     │
│  - Downsamples to 160x120 RGB   │       │  - Microsecond perf_counter     │
│  - Precise monotonic timestamp  │       │  - WASD, Space, Shift, C, R     │
│  - Throttled to ~15 Hz          │       │  - Mouse dx/dy, LMB, RMB, Wheel │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Timestamp Synchronization      │
                   │  - Windowed mouse dx/dy sum       │
                   │  - Key press/release state track  │
                   │  - Single-frame tap preservation  │
                   │  - Symmetric log mouse binning    │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │   Session Dataset Writer (v1.0)   │
                   │  - metadata.json                  │
                   │  - samples.jsonl (sync stream)    │
                   │  - frames/frame_XXXXXXXX.jpg      │
                   └───────────────────────────────────┘
```

### Initial Restricted Action Space (VERIFIED)

| Action Key | Type | Domain / Values | Description |
|---|---|---|---|
| `move_x` | Discrete | `{-1, 0, +1}` | Lateral strafe: -1 = A (Left), 0 = None, +1 = D (Right). Cancellation applies (A+D = 0). |
| `move_y` | Discrete | `{-1, 0, +1}` | Longitudinal movement: -1 = S (Backward), 0 = None, +1 = W (Forward). Cancellation applies (W+S = 0). |
| `jump` | Binary | `{0, 1}` | Space key. Active if held or tapped during frame window. |
| `crouch` | Binary | `{0, 1}` | Ctrl / C key. |
| `sprint` | Binary | `{0, 1}` | Shift key. |
| `reload` | Binary | `{0, 1}` | R key. |
| `fire` | Binary | `{0, 1}` | Left Mouse Button (Primary Fire). |
| `ads` | Binary | `{0, 1}` | Right Mouse Button (Aim Down Sights). |
| `mouse_dx` | Continuous | Float (pixels) | Accumulated horizontal mouse delta over $[t_{i-1}, t_i]$. |
| `mouse_dy` | Continuous | Float (pixels) | Accumulated vertical mouse delta over $[t_{i-1}, t_i]$. |
| `mouse_dx_bin` | Discrete | `[0, num_bins_x - 1]` | Discretized horizontal look/aim bin index (default 21 bins). |
| `mouse_dy_bin` | Discrete | `[0, num_bins_y - 1]` | Discretized vertical look/aim bin index (default 21 bins). |
| `wheel_dy` | Integer | `int` | Mouse scroll wheel delta. |
| `active_keys` | List[str] | e.g. `["w", "shift"]` | Normalized raw active keys list for debugging/inspection. |

### Mouse Discretization & Binning Strategy

Human FPS mouse movements are heavy-tailed: high density of sub-pixel and micro-adjustments around zero, with occasional large flick turns.
- **Symmetric Log (`symmetric_log`, default 21 bins)**:
  - Center bin (`bin 10`): deadzone / zero `[-0.2, +0.2]` pixels.
  - Intermediate bins: exponentially spaced thresholds (`[-0.87, -0.42]`, `[-1.82, -0.87]`, `[-3.79, -1.82]`, ..., `[+0.42, +0.87]`, `[+0.87, +1.82]`, ...).
  - Outer bins (`bin 0` and `bin 20`): clamp extreme flicks `[-inf, -150.0]` and `[+150.0, +inf]`.
- **Quantile Binning (`quantile`)**: `MouseBinner.fit_quantile_edges(dataset_deltas, num_bins=21)` fits empirical quantiles directly from recorded human demonstrations.
- **Dequantization**: `binner.dequantize(bin_idx)` maps discrete predictions back to representative continuous deltas for BC policy execution.

### Dataset Storage Layout (Schema v1.0.0)

A dataset session directory has the following structure:

```
datasets/<session_id>/
  ├── metadata.json       <- Session config, schema version, action space, stats summary
  ├── samples.jsonl       <- Line-by-line synchronized steps (JSONL)
  └── frames/             <- Compressed observation images
        ├── frame_00000000.jpg
        ├── frame_00000001.jpg
        └── ...
```

#### `metadata.json` Schema:
```json
{
  "session_id": "session_20260927_073021_12599a",
  "schema_version": "1.0.0",
  "created_at": "2026-09-27T07:30:21.054713+00:00",
  "source": "ttk_testing",
  "platform": {
    "system": "Windows",
    "release": "11",
    "machine": "AMD64",
    "python_version": "3.11.2"
  },
  "capture_config": {
    "target_fps": 15.0,
    "frame_width": 160,
    "frame_height": 120,
    "color_mode": "RGB",
    "image_format": "jpg",
    "jpeg_quality": 90,
    "window_title": "Roblox",
    "roi": null
  },
  "mouse_config": {
    "sensitivity_scale": 1.0,
    "binning_strategy": "symmetric_log",
    "num_bins_x": 21,
    "num_bins_y": 21,
    "bin_edges_x": [-Infinity, -150.0, ..., 150.0, Infinity],
    "bin_edges_y": [-Infinity, -150.0, ..., 150.0, Infinity]
  },
  "action_space": { ... },
  "summary_stats": {
    "duration_seconds": 60.0,
    "total_steps": 900,
    "effective_fps": 15.0,
    "dropped_frames": 0,
    "total_bytes": 2100000
  }
}
```

#### `samples.jsonl` Line Schema:
```json
{
  "step_idx": 42,
  "timestamp": 124.51234,
  "iso_timestamp": "2026-09-27T07:31:02.123456+00:00",
  "dt": 0.0667,
  "frame_file": "frames/frame_00000042.jpg",
  "actions": {
    "move_x": 1,
    "move_y": 1,
    "jump": 0,
    "crouch": 0,
    "sprint": 1,
    "reload": 0,
    "fire": 1,
    "ads": 0,
    "mouse_dx": 4.5,
    "mouse_dy": -1.2,
    "mouse_dx_bin": 14,
    "mouse_dy_bin": 8,
    "wheel_dy": 0,
    "active_keys": ["d", "w", "shift"],
    "mouse_buttons": {"left": true, "right": false, "middle": false}
  },
  "is_valid": true,
  "metadata": {}
}
```

---

## 6. Repository Structure

```
PROJECT.md                    <- this file (single source of truth)
README.md                     <- short overview + quickstart
setup_windows.ps1             <- ONE-COMMAND Windows bootstrap (fresh clone -> exported envs)
requirements.txt              <- root requirements (Python 3.11)
pytest.ini                    <- pytest configuration
.gitignore                    <- excludes .venv/, tools/, /examples/, logs/, build/, datasets/
data_pipeline/                <- M1: Human Data Capture Pipeline
  __init__.py                 <- package exports
  schema.py                   <- versioned schema, dataclasses, serialization
  actions.py                  <- canonical action space, MouseBinner (symmetric log/quantile)
  capture.py                  <- ScreenCapture (MSS, downsampling, ROI cropping)
  input_listener.py           <- InputListener (Pynput async event queue)
  sync.py                     <- ActionSynchronizer (timestamp alignment, tap preservation)
  recorder.py                 <- SessionRecorder coordinator
  mock.py                     <- Synthetic MockScreenCapture & MockInputGenerator
  validate.py                 <- Dataset validation tool (schema, bounds, frame integrity)
  stats.py                    <- Dataset inspection & ASCII histogram generator
  inspect.py                  <- CLI alias for dataset inspection
  record.py                   <- CLI entry point for live & mock recording
  ttk_adapter.py              <- Roblox/TTK & Godot window isolation layer
tests/                        <- Automated test suite (37 tests)
  test_schema.py
  test_actions.py
  test_sync.py
  test_validator.py
  test_stats.py
  test_recorder_mock.py
  test_ttk_adapter.py
feasibility/                  <- M0: Godot-RL feasibility & benchmarks
  README.md                   <- RUNBOOK: setup + Tests A/B/C + results table
  requirements.txt            <- pinned Python deps (Python 3.11)
  gdrl_common.py              <- shared helpers (port pre-check, process cleanup, path resolution)
  setup_examples.py           <- clone pinned examples + apply overlay
  export_envs.py              <- headless import + export env executables
  benchmark_env.py            <- env steps/sec + RAM/VRAM benchmark
  train_ppo.py                <- short SB3 PPO training run
  godot_overlays/examples/    <- overlays for godot_rl_agents_examples
```

---

## 7. Exact Commands (Windows 11, from Repo Root)

Activate environment:
```powershell
.venv\Scripts\activate
```

### M1: Test the Pipeline with Synthetic Mock Data (No Roblox Required)
```powershell
python -m data_pipeline.record --mock --duration 5 --output datasets/mock_test
python -m data_pipeline.validate datasets/mock_test/<session_id>
python -m data_pipeline.inspect datasets/mock_test/<session_id>
```

### M1: Record Real Manual Gameplay from TTK Testing
1. Launch Roblox and enter **TTK Testing [HARDPOINT]**.
2. Run the recorder:
```powershell
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --width 160 --height 120 --output datasets/ttk_sessions
```
3. Play normally. Press `Ctrl+C` in the terminal when done.
4. Validation and summary statistics are printed automatically upon exit.

### M1: Run Automated Tests
```powershell
pytest -v
```

### M0: Feasibility Benchmarks (from previous phase)
```powershell
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
```

---

## 8. Roadmap

- **M0 Feasibility** — Godot 4.3 + godot-rl 0.8.2 + export pipeline verified. Windows baseline benchmarks (Tests A–C).
- **M1 Human Data Pipeline** — [COMPLETED & VERIFIED] Real-time screen capture (~15 Hz, 160x120), OS-level input logging, microsecond timestamp synchronization, symmetric log/quantile mouse binning, synthetic mock generator, validation tool, inspection tool, 37 automated tests.
- **M2 Minimal Sandbox & Dataset Collection** —
  - 1. Record 30–60 minutes of real manual TTK gameplay sessions using M1 recorder.
  - 2. Build small greybox tactical-FPS sandbox map in Godot (TTK-inspired layout, low-poly textures).
- **M3 Behavioral Cloning** — Train visual BC policy (ResNet-18 / IMPALA + ConvGRU) on captured dataset; offline evaluation.
- **M4 Closed-loop Evaluation** — BC policy plays inside the Godot sandbox.
- **M5 DAgger / RL** — PPO fine-tuning in sandbox.
- **M6 GUI / Control Center** — Visualization and management.

---

## 9. Decisions future agents must NOT undo without explicit justification

1. No automation/injection/memory-reading of the public Roblox game. Ever.
2. TTK Testing is ONLY a manual gameplay data source.
3. Sandbox = our own Godot environment, not Roblox, not a TTK map copy.
4. Engine choice: Godot 4 + godot_rl_agents.
5. Python 3.11 + PyTorch + SB3-first. Local/free/open-source; no paid services.
6. Dataset schema versioning is mandatory for all recording formats.
7. Mouse delta representation: store both continuous `dx, dy` and discretized `dx_bin, dy_bin`.
8. The RL environment is always an **exported game binary**, never a wrapper script and never the editor executable.

---

## 10. Known Limitations

1. **OS Window Occlusion**: If another window completely covers the game viewport during recording, screen capture will record the occluding window unless Roblox is focused.
2. **Dynamic In-Game Sensitivity**: If the player changes in-game sensitivity mid-match, continuous mouse `dx/dy` magnitude changes relative to in-game angular rotation. Sensitivity should remain constant during capture sessions.
3. **Headless Linux Capture**: Real screen capture with `mss` requires an active X11/Wayland display server; in headless CI environments, use `--mock` for full synthetic end-to-end testing.

---

## 11. Recommendations for M2

1. **Jonas records initial 15–30 min manual dataset**: Play 3–5 rounds of TTK Testing [HARDPOINT] with `python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --output datasets/ttk_pilot`.
2. **Fit Quantile Mouse Bins**: Use empirical mouse delta distributions from the pilot recording to fine-tune `MouseBinner` quantiles if desired.
3. **Build Minimal Godot Sandbox Map**: Create the greybox tactical FPS map with matching visual style (low poly, 160x120 camera view, player character controller with matching WASD speed, ADS FOV transition, weapon recoil, and hitboxes).
