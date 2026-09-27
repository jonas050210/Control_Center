# SandboxAI — PROJECT.md (single source of truth)

> Read this first. This file is the authoritative context for all AI
> coding/research sessions on this repository. Update it when facts change;
> keep results labeled VERIFIED / MEASURED / ESTIMATED / UNKNOWN and never
> invent measurements.

Last updated: 2026-09-27 (Full Rough End-to-End Architecture Complete: M1 Data Pipeline,
Phase 2 Behavioral Cloning, Phase 3 Godot Tactical Sandbox, Phase 4 Closed-Loop BC Inference,
Phase 5 PPO Reinforcement Learning, Phase 6 Evaluation Benchmark, Phase 7 System Telemetry,
and 54 automated tests; see sections 5–8).

## 1. What this project is

SandboxAI is a vision-based AI learning project. The pipeline:

```
human plays (Roblox "TTK Testing [HARDPOINT]" / Godot Sandbox)
      │  screen recording + input logging (manual data capture only — M1 Data Pipeline)
      ▼
imitation learning / behavioral cloning (PyTorch IMPALA CNN + GRU — Phase 2)
      │  BC policy weights (bc_best.pt)
      ▼
AI plays inside OUR OWN controlled sandbox (Godot 4 Tactical Arena — Phases 3 & 4)
      │  PPO reinforcement learning fine-tuning (Stable-Baselines3 — Phase 5)
      ▼
evaluation & benchmark harness (Random vs BC vs PPO — Phase 6)
      │
      ▼
technical monitoring & control center telemetry (Phase 7)
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

| Piece | Choice | Why |
|---|---|---|
| Engine / sandbox | **Godot 4.3-stable** (`official.77dcf97d8`) | MIT, free, light on 8 GB GPU, low-poly visuals close to Roblox style (helps BC transfer), sandbox doubles as later GUI/game |
| RL bridge | **godot_rl_agents** (pip `godot-rl` **0.8.2**, latest release) | Gymnasium interface, Python 3.11, SB3 wrappers, ONNX export path |
| RL algo | **Stable-Baselines3 PPO 2.4.0** | simplicity first, robust on visual observations |
| ML framework | **PyTorch** (2.6.0+cu124 on Windows / 2.14 on Linux) | project preference |
| BC model | Custom PyTorch IMPALA-style residual CNN + optional GRU | lightweight, fast inference, multi-head action distribution |
| Data capture | **MSS + Pynput + PIL** (pure Python, cross-platform, non-invasive) | zero-overhead, no C++ compilation, native Windows hook support |
| Monitoring | Structured JSON telemetry (`SystemTelemetry`) | decoupled, CLI status tool, consumable by future GUI |

---

## 5. End-to-End Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PIPELINE (M1)                                                       │
│    - ScreenCapture (MSS): 160x120 RGB @ ~15 Hz                              │
│    - InputListener (Pynput): Async WASD, Space, Shift, C, R, LMB, RMB, dx/dy│
│    - ActionSynchronizer: Microsecond timestamp integration & tap preservation│
│    - Versioned Dataset Schema v1.0.0 (metadata.json + samples.jsonl + frames)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. BEHAVIORAL CLONING (Phase 2)                                             │
│    - GameplayDataset: Session-level train/val split, temporal sequence window│
│    - BCVisionNetwork: IMPALA residual CNN + multi-head action prediction    │
│    - BCTrainer: AdamW, AMP (CUDA/CPU), CrossEntropy + Mouse Bin Accuracies   │
│    - Checkpointing: bc_best.pt / bc_latest.pt                                │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. GODOT TACTICAL SANDBOX & CLOSED-LOOP INFERENCE (Phases 3 & 4)            │
│    - Godot 4.3 Arena: Greybox arena, walls, cover pillars, target dummies   │
│    - BCPolicy: Observation -> Model -> Predicted ActionState                │
│    - SandboxGymEnv: Visual Gymnasium bridge (84x84 / 160x120 RGB)           │
│    - Closed-Loop Runner: bc.sandbox_runner evaluates policy in sandbox      │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. REINFORCEMENT LEARNING (Phase 5)                                         │
│    - Stable-Baselines3 PPO with CnnPolicy on visual sandbox observation     │
│    - Fine-tunes tactical movement, target acquisition, and shooting accuracy│
│    - Checkpoint: ppo_sandbox.zip                                            │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. EVALUATION & TELEMETRY (Phases 6 & 7)                                    │
│    - evaluation.benchmark: Standardized comparison (Random vs BC vs PPO)    │
│    - Tracks mean reward, hit rate %, fire rate %, survival steps            │
│    - monitoring.status: Real-time CLI telemetry and system state dashboard  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Action Space & Mouse Discretization Specification

| Action Key | Type | Domain / Values | Description |
|---|---|---|---|
| `move_x` | Discrete | `{-1, 0, +1}` | Lateral strafe: -1 = A (Left), 0 = None, +1 = D (Right). Cancellation applies. |
| `move_y` | Discrete | `{-1, 0, +1}` | Longitudinal movement: -1 = S (Backward), 0 = None, +1 = W (Forward). Cancellation applies. |
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

### Mouse Discretization Model (`symmetric_log`)
- Center bin (`bin 10`): deadzone / zero `[-0.2, +0.2]` pixels.
- Intermediate bins: exponentially spaced thresholds for fine sub-pixel aiming adjustments.
- Outer bins (`bin 0` and `bin 20`): clamp extreme outer flick turns `[-inf, -150.0]` and `[+150.0, +inf]`.
- Dequantization: `binner.dequantize(bin_idx)` maps discrete predictions back to continuous pixel rotations for policy replay.

---

## 7. Repository Structure

```
PROJECT.md                    <- single source of truth
README.md                     <- quickstart & runbook
setup_windows.ps1             <- ONE-COMMAND Windows bootstrap
requirements.txt              <- pinned root requirements (Python 3.11)
pytest.ini                    <- pytest configuration
data_pipeline/                <- M1 Data Capture & Dataset Tooling
  __init__.py
  schema.py                   <- versioned schema, dataclasses, serialization
  actions.py                  <- canonical action space, MouseBinner
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
bc/                           <- Phase 2 & 4: Behavioral Cloning
  __init__.py
  dataset.py                  <- GameplayDataset (session-level split, sequence windows)
  models.py                   <- BCVisionNetwork (IMPALA CNN + GRU + multi-head actions)
  train.py                    <- BCTrainer (AdamW, AMP, multi-task cross-entropy loss)
  policy.py                   <- BCPolicy inference wrapper (predict ActionState)
  infer.py                    <- CLI tool for single-frame inference
  sandbox_runner.py           <- Phase 4: Closed-loop BC runner in tactical sandbox
sandbox/                      <- Phase 3: Godot Tactical Sandbox & Gym Bridge
  __init__.py
  env.py                      <- MockTacticalArenaEnv & GodotSandboxEnv Gymnasium wrappers
  godot_project/              <- Godot 4.3 project
    project.godot
    export_presets.cfg        <- Windows (.exe) and Linux export presets
    scenes/Arena.tscn         <- Greybox 3D tactical arena
    scenes/Player.tscn        <- CharacterBody3D, Camera3D, SubViewport (84x84)
    scenes/TargetDummy.tscn   <- TargetDummy entity with hit detection
    scripts/player.gd         <- WASD, mouse look, raycast weapon shooting
    scripts/ai_controller.gd  <- godot_rl_agents AIController3D interface
    scripts/target_dummy.gd   <- Health, hit callback, randomized respawn
rl/                           <- Phase 5: Reinforcement Learning
  __init__.py
  train_ppo.py                <- Stable-Baselines3 PPO training in Sandbox
  train.py                    <- CLI alias for PPO training
evaluation/                   <- Phase 6: Evaluation & Benchmarking
  __init__.py
  benchmark.py                <- PolicyBenchmark (Random vs BC vs PPO comparison)
monitoring/                   <- Phase 7: System Telemetry & Control Layer
  __init__.py
  state.py                    <- SystemTelemetry persistent JSON tracker
  status.py                   <- CLI status & telemetry dashboard
tests/                        <- Automated Test Suite (54 tests)
  test_schema.py
  test_actions.py
  test_sync.py
  test_validator.py
  test_stats.py
  test_recorder_mock.py
  test_smoke_pipeline.py
  test_ttk_adapter.py
  test_bc_dataset.py
  test_bc_model.py
  test_bc_policy.py
  test_sandbox_env.py
  test_rl_ppo.py
  test_evaluation_benchmark.py
  test_monitoring.py
feasibility/                  <- M0 Godot-RL feasibility & benchmarks
  README.md
  requirements.txt
  benchmark_env.py
  train_ppo.py
  setup_examples.py
  export_envs.py
  gdrl_common.py
```

---

## 8. Exact Commands (Windows 11, from Repo Root)

Activate environment:
```powershell
.venv\Scripts\activate
```

### 1. Run Automated Tests (54 tests)
```powershell
pytest -v
```

### 2. Record Gameplay Data
```powershell
# Synthetic mock recording (5 seconds, test without Roblox)
python -m data_pipeline.record --mock --duration 5 --output datasets/mock_session

# Real manual gameplay from TTK Testing on Roblox
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --output datasets/ttk_pilot
```

### 3. Validate and Inspect Dataset
```powershell
python -m data_pipeline.validate datasets/mock_session/<session_id>
python -m data_pipeline.inspect datasets/mock_session/<session_id>
```

### 4. Train Behavioral Cloning (BC) Policy
```powershell
python -m bc.train --data_dir datasets/ --epochs 10 --batch_size 32 --checkpoint_dir checkpoints/
```

### 5. Run BC Inference on a Single Frame
```powershell
python -m bc.infer --checkpoint checkpoints/bc_best.pt
```

### 6. Run BC Policy Closed-Loop inside Tactical Sandbox
```powershell
python -m bc.sandbox_runner --checkpoint checkpoints/bc_best.pt --episodes 5
```

### 7. Train Reinforcement Learning (PPO) in Tactical Sandbox
```powershell
python -m rl.train --timesteps 10000 --checkpoint_dir checkpoints/
```

### 8. Benchmark & Compare Policies (Random vs BC vs PPO)
```powershell
python -m evaluation.benchmark --bc_checkpoint checkpoints/bc_best.pt --ppo_checkpoint checkpoints/ppo_sandbox.zip --episodes 5
```

### 9. View System Telemetry Dashboard
```powershell
python -m monitoring.status
```

---

## 9. Roadmap

- **M0 Feasibility** — [COMPLETED & HARDENED] Godot 4.3 + godot-rl 0.8.2 + export flow verified.
- **M1 Human Data Pipeline** — [COMPLETED & VERIFIED] External screen capture (~15 Hz, 160x120), OS-level input listener, microsecond timestamp synchronization, symmetric log mouse binning, validation and inspection tools.
- **M2 Behavioral Cloning & Sandbox End-to-End** — [COMPLETED & VERIFIED]
  - PyTorch IMPALA residual CNN + GRU model
  - Session-level train/validation split
  - Closed-loop BC inference in tactical sandbox
  - Stable-Baselines3 PPO training in tactical sandbox
  - 3-way evaluation benchmark harness (Random vs BC vs PPO)
  - System telemetry & state tracking dashboard
  - 54 automated unit and integration tests
- **M3 Real Demonstration Collection & Scaling** —
  - Jonas records 1–2 hours of manual TTK Testing gameplay across several sessions
  - Scale BC model training on real human demonstration data
  - Quantile bin fitting from human mouse distributions
- **M4 Sandbox Visual Tuning & DAgger** —
  - Align Godot sandbox lighting / textures closer to TTK style to minimize visual domain gap
  - DAgger / interactive imitation fine-tuning in sandbox
- **M5 PPO Self-Play & Advanced RL** —
  - Multi-agent target / opponent bots in Godot sandbox
  - High-throughput PPO fine-tuning initialized from BC weights
- **M6 GUI / Control Center Application** —
  - Godot-based or web-based live control center consuming `monitoring.state` telemetry

---

## 10. Decisions future agents must NOT undo without explicit justification

1. No automation/injection/memory-reading of the public Roblox game. Ever.
2. TTK Testing is ONLY a manual gameplay data source.
3. Sandbox = our own Godot environment, not Roblox, not a TTK map copy.
4. Engine choice: Godot 4 + godot_rl_agents.
5. Python 3.11 + PyTorch + SB3-first. Local/free/open-source; no paid services.
6. Dataset schema versioning is mandatory for all recording formats.
7. Mouse delta representation: store both continuous `dx, dy` and discretized `dx_bin, dy_bin`.
8. The RL environment is always an **exported game binary**, never a wrapper script and never the editor executable.

---

## 11. Known Limitations

1. **Visual Domain Gap**: Roblox TTK and the minimal Godot greybox arena have visual differences; zero-shot transfer without sandbox domain adaptation or color augmentation is limited.
2. **OS Window Occlusion**: Screen capture records whatever is displayed in the game client rect; if another application window is overlaid on top of Roblox while playing, it will be captured in the frames.
3. **In-game Sensitivity Shifts**: In-game sensitivity must be kept constant across manual recording sessions so mouse pixel deltas map consistently to angular rotation.

---

## 12. Recommended Immediate Next Step

Jonas records an initial 15–30 minute manual gameplay demonstration dataset in **TTK Testing [HARDPOINT]** on Roblox using:
```powershell
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --width 160 --height 120 --output datasets/ttk_pilot
```
Then trains the first real BC model on his own demonstrations:
```powershell
python -m bc.train --data_dir datasets/ttk_pilot --epochs 15 --batch_size 32
```
And benchmarks it inside the sandbox:
```powershell
python -m evaluation.benchmark --bc_checkpoint checkpoints/bc_best.pt --episodes 10
```
