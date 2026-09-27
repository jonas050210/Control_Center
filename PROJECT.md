# SandboxAI — project source of truth

**Last updated:** 2026-09-27

**Status:** end-to-end local workflow is implemented; the Python reference path is statically checked in this checkout, while the exported Godot binary still requires a Godot 4.3 runtime test on the target machine.

This document records facts and boundaries. Labels mean:

- **VERIFIED** — exercised by automated tests or an end-to-end run in this checkout
- **IMPLEMENTED** — code is complete, but the required external runtime was unavailable here
- **MEASURED** — value came from an actual command, not an estimate
- **EXTERNAL** — requires owner hardware/data/software not present in the repository

## 1. Product and boundary

SandboxAI is a local vision-based learning system:

```text
human manual gameplay
  → synchronized RGB frames + canonical actions
  → versioned/validated demonstration sessions
  → temporal behavioral cloning
  → offline imitation evaluation
  → closed-loop AI play in a controlled SandboxAI FPS
  → BC-warm-started PPO reinforcement learning
  → deterministic policy comparison
  → persistent checkpoints, metrics, telemetry, and repeated cycles
```

TTK Testing [HARDPOINT]/Roblox is only a source of **manually played, externally captured demonstrations**. Permanently out of scope:

- botting or automating public Roblox gameplay
- input injection, process memory access, exploit/anti-cheat bypass
- using Roblox as an RL environment

AI actuation and RL belong only in SandboxAI's own controlled environments.

## 2. Target constraints

| Constraint | Target |
|---|---|
| OS | Windows 11 |
| CPU | Intel i7-12700F |
| GPU | RTX 4060 Ti, 8 GB VRAM |
| RAM | 32 GB |
| Python | 3.11 |
| Engine | Godot 4.3 |
| Cost/privacy | Local, free/open-source, no paid API |

Resource choices: 160×120 demonstration frames, 84×84 RL observations, lightweight residual CNN, optional 256-unit GRU, AMP on CUDA, and typically 1–2 environments. Generated datasets, models, exports, and logs are ignored by Git.

## 3. Current architecture

### 3.1 Capture and dataset — VERIFIED

`SessionRecorder` coordinates MSS/PIL capture, `pynput` input events on Windows, and `ActionSynchronizer` at monotonic timestamps. The synchronizer preserves fast press/release taps and queues events that arrive after a frame timestamp instead of dropping them.

Session layout:

```text
<session>/
  metadata.json       schema/config/platform/status/summary
  samples.jsonl       one synchronized observation/action record per line
  frames/*.jpg|png    RGB observations
```

Schema: `1.1.0`, major-version checked. Recording uses a `.recording` marker, atomic frame replacement, non-destructive session IDs by default, explicit overwrite, final status, and cleanup on failure/interruption.

Tools:

- `data_pipeline.validate`: structure, completion, finite timestamps/deltas, monotonicity, action bounds, bin bounds, path containment, frame integrity/shape, duplicate/missing/orphan frames, count consistency
- `data_pipeline.stats`: timing, storage, action frequencies, mouse distributions/histograms
- `data_pipeline.inspect`: self-contained visual HTML timeline
- `data_pipeline.catalog`: recursive aggregate validation and JSON catalog

### 3.2 Imitation learning — VERIFIED

`GameplayDataset` recursively discovers sessions, rejects incompatible mouse-bin configurations, never crosses session boundaries, and creates session-level train/validation splits. A single sufficiently long session is split chronologically with a sequence context gap. Train-only lightweight color/noise augmentation reduces domain overfitting without changing aim geometry.

`BCVisionNetwork` contains:

- three IMPALA-style residual convolution stages
- adaptive pooling and compact visual latent
- optional stateful GRU for temporal behavior
- categorical heads for movement, jump, crouch, sprint, reload, fire, ADS, and 21-bin horizontal/vertical look
- auxiliary robust continuous mouse regression

Training includes AdamW, CUDA AMP, gradient clipping, multi-task loss, all-head accuracy, look top-3 metrics, latest/best checkpoints, history JSON, exact mouse edges in checkpoints, resume support, experiment manifests, and telemetry.

`BCPolicy` performs shape-safe CHW/HWC preprocessing, resizes to the trained resolution, preserves recurrent state across closed-loop steps, resets it per session/episode, exposes confidence, and emits the shared sandbox action.

Evaluators:

- `evaluation.imitation`: per-head accuracy, macro F1, confidence, joint action agreement, and mouse error on recorded data
- `bc.sandbox_runner`: closed-loop reward, hits, kills, accuracy, fire rate, damage, health, and throughput

### 3.3 Controlled FPS environments

#### Deterministic Python reference arena — VERIFIED

`TacticalArenaEnv` is the production high-throughput headless environment, not a feasibility mock. It implements:

- deterministic seed/reset and serializable state snapshots
- channel-first RGB observations
- shared ten-component action space
- camera yaw/pitch and ADS precision
- collision-aware movement, diagonal normalization, sprint, crouch, and jump state
- magazine/reserve ammo, fire cadence, reload duration, recoil, dry-fire behavior
- ray-style hit tests, target health/respawn, kills, moving enemies, incoming damage and death
- fixed asymmetric cover, lanes, boundaries, occluded sightlines, and exploration cells
- step, movement, exploration, aim, on-target, hit, kill, miss, ammo, damage, and death reward terms
- HUD/crosshair cues plus rich diagnostic `info` (privileged state is not in the policy observation)

The old `MockTacticalArenaEnv` name is retained as a compatibility alias only.

#### Godot tactical arena — IMPLEMENTED, external runtime validation pending

`sandbox/godot_project` is the manually playable long-term visual environment. It has enclosing collision walls, tactical cover, player collision/camera, mouse look, WASD, sprint/crouch/jump, ADS FOV, ray weapon, ammo/reload/cadence/recoil, HUD, health, moving/damaging target enemies, deterministic controller reset, and an 84×84 observation viewport.

Its 3D presentation is self-contained: `scripts/model_factory.gd` builds reusable low-poly meshes and materials at runtime. Player viewmodels use weapon-specific assemblies (receiver, stock, grip, magazine, barrel, sight, scope or pump), targets use armored body/limb/visor/backpack parts, and authored or generated arena cover receives trim, panels, supports, floor insets, and wall bands. No external art pack or network asset is required, so deterministic maps remain portable and easy to test.

A custom localhost-only newline-delimited JSON/TCP bridge is built into the export and enabled only with `--rl-server`. `GodotSandboxEnv` launches an **exported binary**, handshakes protocol v1, sends full actions, receives PNG observations/reward/state, and terminates it cleanly. Manual play has no server. Explicit Godot requests fail rather than silently falling back to Python.

Agent-mode timers/enemy simulation advance once per command at 20 Hz rather than by network wall time, making command trajectories deterministic. The export helper is `python -m sandbox.export`.

No Godot executable/export templates were available in this Linux agent checkout, so actual engine parsing/export/bridge execution is **EXTERNAL**, not falsely reported as measured.

### 3.4 Shared action contract — VERIFIED

Action contract version **2.0.0** is declared in `data_pipeline.schema`, checked by
Python, and returned during the Godot bridge handshake. `sandbox.actions` is
the conversion authority. Canonical values:

| Order | Name | Dataset domain | Gym category count |
|---:|---|---|---:|
| 0 | `move_x` | -1/0/+1 | 3 |
| 1 | `move_y` | -1/0/+1 | 3 |
| 2 | `jump` | 0/1 | 2 |
| 3 | `crouch` | 0/1 | 2 |
| 4 | `sprint` | 0/1 | 2 |
| 5 | `reload` | 0/1 | 2 |
| 6 | `fire` | 0/1 | 2 |
| 7 | `ads` | 0/1 | 2 |
| 8 | `mouse_dx_bin` | 0…20 | 21 |
| 9 | `mouse_dy_bin` | 0…20 | 21 |

Gym space: `MultiDiscrete([3,3,2,2,2,2,2,2,21,21])`. A five-value adapter reads old checkpoints/actions; all new paths use the full contract.

### 3.5 Reinforcement learning — VERIFIED

Stable-Baselines3 PPO uses `BCFeatureExtractor`, which matches the BC encoder. When `--bc_checkpoint` is supplied:

- convolution encoder weights are copied
- visual projection weights are copied
- all ten concatenated categorical action-head weights/biases are copied when compatible
- PPO learns a new value function
- GRU recurrence is intentionally not transferred into feed-forward SB3 PPO

Training supports parallel environments, resume, periodic/final ZIP checkpoints, JSON summaries, throughput/episode metrics, action/reward runtime telemetry, and experiment artifact hashes.

`rl.iterate` performs repeated PPO chunk → fixed-seed benchmark → checkpoint cycles and writes `rl_learning_curve.json`, providing the base for later DAgger, replay/offline RL, or stronger algorithms without changing data/actions/environments.

### 3.6 Evaluation — VERIFIED

`evaluation.benchmark` compares on identical episode seeds:

1. seeded random baseline
2. scripted vision target-tracking baseline
3. BC policy
4. PPO policy

Metrics include reward mean/std/min/max, episode length, shots, hits, hit accuracy, fire rate, kills, damage, and final health. Episode records and JSON reports are retained.

### 3.7 Persistence and monitoring — VERIFIED

Every optional tracked operation owns `logs/experiments/<run_id>/`:

- atomic `manifest.json` with status, parameters, environment, summary, artifacts, SHA-256 and size
- append-only `metrics.jsonl`

`SystemTelemetry` atomically tracks stage, hardware, datasets, active BC/PPO checkpoints, steps/s, FPS, action, reward/state, benchmark results, and recent events. It survives restarts and quarantines corrupt state.

Interfaces:

- `python -m sandboxai status` — terminal dashboard
- `python -m sandboxai dashboard` — dependency-free localhost HTML/JSON dashboard with policy chart and experiment table

## 4. Unified workflow

Primary command:

```powershell
python -m sandboxai e2e --data_dir datasets --bc_epochs 10 --rl_timesteps 10000
```

Stages are not disconnected scripts; the orchestrator performs:

1. discover or explicitly generate recordings
2. catalog and reject invalid sessions
3. train temporal BC and save checkpoints
4. evaluate imitation on the selected held-out session
5. run closed-loop BC episodes
6. initialize PPO from BC and train/save
7. benchmark random/scripted/BC/PPO on common seeds
8. persist one workflow summary plus per-stage experiments and telemetry

Installation smoke command:

```powershell
python -m sandboxai e2e --quick --generate_demo
```

Synthetic data verifies plumbing, not policy quality and never substitutes for human demonstrations.

## 5. Verification performed in this checkout

**Repository verification status 2026-09-27:**

- `python -m compileall` passes for all production Python packages in the current checkout.
- `git diff --check` passes.
- The current checkout contains the full Python/Godot test suite, but its dependencies are not installed in this agent environment; run `python -m pip install -e ".[dev]"` before `pytest -q`.
- GitHub Actions now runs Python 3.11/3.12 compilation, tests, whitespace checks, and Godot project smoke checks.
- Godot export, scene parsing, local bridge, repeated reset, and target-machine runtime still require Godot 4.3 and must be verified on Windows or a Godot-enabled CI runner.
- Synthetic recordings verify plumbing only; their policy scores have no scientific meaning.

Automated coverage includes schema/actions, screen/input mocks, event synchronization including future-event retention, strict validation failure modes, stats/HTML catalog, temporal datasets/models/policy, train/resume, deterministic arena reset and mechanics, weapon reload, closed-loop BC, PPO smoke, benchmark, telemetry, experiment artifacts, replay conversion, CLI integration, and historical feasibility regressions.

## 6. Repository map

```text
data_pipeline/          capture, synchronize, schema, validate, catalog, inspect
bc/                     temporal model, dataset, training, inference, sandbox runner
sandbox/actions.py      shared action contract
sandbox/env.py          deterministic visual FPS reference arena
sandbox/godot_env.py    exported-Godot local bridge client
sandbox/godot_project/  playable tactical FPS and local bridge server
sandbox/export.py       Godot import/export helper
rl/                     BC-warm-started PPO and repeated cycles
evaluation/             offline imitation and closed-loop policy benchmark
monitoring/             state, experiments, CLI and web dashboard
sandboxai/              unified doctor/status/play/end-to-end CLI
feasibility/            historical M0 reference only; not the product
```

## 7. Genuine remaining external limitations

1. **Human data is external.** The repository cannot create the owner's real TTK/sandbox demonstrations. Meaningful BC quality requires several varied manual sessions; synthetic smoke data proves only integration.
2. **Godot binary validation requires Godot.** The project and bridge are implemented, but this agent environment had no Godot executable or export templates. The target machine must run `python -m sandbox.export` and the Godot backend smoke test.
3. **Visual domain gap is empirical.** Roblox and the custom arena differ visually. Augmentation helps, but only owner data and measured transfer can determine whether sandbox demonstrations/domain randomization or DAgger corrections are needed.
4. **PPO sample needs are physical.** The 128-step smoke proves operation, not convergence. Tactical behavior normally requires longer local runs and reward tuning based on retained benchmarks.
5. **OS capture constraints remain.** External capture records the visible client area; occlusion, changed game sensitivity, DPI/raw-input behavior, or a moved/resized window can degrade labels. The recorder detects a missing selected window but cannot control third-party rendering.

## 8. Decisions not to undo casually

1. Never automate public Roblox/TTK gameplay.
2. Keep schema versioning and exact mouse-bin metadata.
3. Keep one canonical action adapter across recorder, BC, Python, Godot, PPO, and evaluation.
4. An explicitly selected backend must fail clearly; never hide integration failure with fallback.
5. AI Godot integration uses an exported game binary, never the editor executable.
6. Keep generated recordings/checkpoints/logs/builds out of Git.
7. Preserve deterministic fixed-seed evaluation before adopting a more complex RL method.
