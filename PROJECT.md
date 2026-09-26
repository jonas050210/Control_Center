# SandboxAI — PROJECT.md (single source of truth)

> Read this first. This file is the authoritative context for all AI
> coding/research sessions on this repository. Update it when facts change;
> distinguish **measured** results from **estimates**.

Last updated: 2026-09-26 (after M0 feasibility work).

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
| Engine / sandbox | **Godot 4.x** (4.3-stable currently) | MIT, free, light on 8 GB GPU, low-poly visuals close to Roblox style (helps BC transfer), sandbox doubles as later GUI/game |
| RL bridge | **godot_rl_agents** (pip `godot-rl`, currently 0.8.2) | Gym/Gymnasium interface, works on Python 3.11 (Unity ML-Agents is pinned to 3.10), SB3/SampleFactory/CleanRL wrappers, ONNX export path |
| RL algo | **Stable-Baselines3 PPO** first; Sample Factory/APPO to evaluate later if throughput demands | simplicity first |
| ML framework | **PyTorch** | project preference |
| BC model | Custom PyTorch (suggested start: ResNet-18 or IMPALA-style CNN + ConvGRU) trained on Roblox recordings | research finding |
| Deployment | ONNX later, for in-engine inference / GUI phase | optional |

Rejected alternatives (do not revisit without new evidence): Unity ML-Agents
(Python 3.10 pin, rigid trainer), Unreal (too heavy for solo dev), ViZDoom
(visual domain gap vs Roblox, dead end for GUI), custom 3D Python env (months
of renderer work), Madrona-class GPU sims (C++/Linux research tooling).

### Observation/action design (initial, from TTK research — estimates, not yet validated)
- Observation: ~160x120 RGB at ~15 Hz for BC on real recordings; sandbox RL
  benchmarked first at 84x84. Hybrid visual + vector/raycast observations are
  acceptable during development (debugging, reward shaping, RL fallback).
- Mouse movement: discretized into bins initially, not direct regression.
- Restricted action space initially. Audio is a later/optional modality.

## 5. Feasibility status (M0)

### Measured (Linux CI sandbox — Debian 12, 2 weak vCPUs, NO GPU/display; lower bound, NOT the target machine)
- Python 3.11.2 venv: `godot-rl 0.8.2`, `stable-baselines3 2.4.0`,
  `gymnasium 1.0.0`, `torch 2.14.0` — installed and ran with no conflicts.
- Godot 4.3-stable built from source (headless-only build) and ran the
  official `FPS` example from `edbeeching/godot_rl_agents_examples`.
- **Godot ↔ Python TCP connection works** (handshake, hybrid
  discrete+continuous action space, 2250-dim raycast obs, 8 agents/instance).
- Raycast throughput: **246 steps/s** (1 instance), **471 steps/s**
  (2 parallel instances, ~1.9x = near-linear scaling on 2 cores).
- PPO trained cleanly: **28,672 steps at ~410 fps** (SB3-reported),
  MultiInputPolicy, losses/gradients healthy.
- RAM: ~140 MB per headless Godot instance, ~580 MB Python/SB3 side.
- Bottleneck identified: Godot process single-core bound (GDScript sensors +
  JSON serialization over TCP), Python mostly idle.

### NOT yet measured (open validation — the current next step)
- **Pixel observations could not be benchmarked** in the Linux sandbox
  (no GPU/GL/Vulkan there). Windows + RTX 4060 Ti pixel performance is the
  key unresolved question.
- All Windows/4060 Ti numbers in `feasibility/README.md` are blank *fill in*
  rows. The GPU is *expected* to be sufficient (estimate) — treat as
  unproven until measured.

### Known quirks (verified)
- godot_rl_agents requires env executable suffix `.x86_64` (Linux) / `.exe`
  (Windows); use a wrapper script/bat with that suffix.
- `gdrl.env_from_hub` needs git-lfs; cloning the examples repo from GitHub
  is simpler.
- In the FPS example, reward fires only on hits and `done` only on death —
  all-zero rewards under a short random policy are env design, not a bug.
- Pixel path risk: per-step GPU→CPU readback + hex-encoded pixels over TCP
  is the known godot_rl_agents camera bottleneck.

## 6. Repository structure

```
PROJECT.md                    <- this file (single source of truth)
README.md                     <- stub
.gitignore                    <- excludes .venv/, tools/, examples/, logs/
feasibility/
  README.md                   <- Windows test procedure + results table
  requirements.txt            <- pinned Python deps (Python 3.11)
  benchmark_env.py            <- env steps/sec + RAM/VRAM benchmark
  train_ppo.py                <- short SB3 PPO training run
```

Not in git (transient, per-machine): `.venv/`, `tools/` (Godot binary/build),
`examples/` (clone of godot_rl_agents_examples), `logs/`.

## 7. Roadmap (phases may change — document changes here with reasons)

- **M0 Feasibility** — mostly done; Windows pixel/raycast benchmarks pending.
- **M1 Minimal sandbox** — small greybox tactical-FPS env in Godot (own map,
  TTK-inspired layout). DO NOT START before M0 Windows results are in.
- **M2 Human data capture** — screen + input recording pipeline for TTK
  gameplay (manual play only), dataset format.
- **M3 Behavioral cloning** — train BC model on recordings.
- **M4 Closed-loop evaluation** — BC policy plays in sandbox.
- **M5 DAgger / RL** — PPO (later possibly APPO) fine-tuning in sandbox.
- **M6 GUI / control center** — visualization and management.

## 8. Decisions future agents must NOT undo without explicit justification

1. No automation/injection/memory-reading of the public Roblox game. Ever.
2. Sandbox = our own Godot environment, not Roblox, not a TTK map copy.
3. Engine choice: Godot 4 + godot_rl_agents (revisit only with new evidence,
   e.g. pixel throughput on Windows proving unworkable).
4. Python 3.11 + PyTorch + SB3-first. Local/free/open-source; no paid services.
5. Keep feasibility scripts minimal — no premature framework-building.
6. This session's git branch workflow: work happens on the Arena-managed
   branch; don't force-push or rewrite history.

## 9. Open questions / risks

- Pixel obs steps/sec on Windows + 4060 Ti (M0 blocker). Fallback if too slow:
  hybrid obs (pixels for BC, raycasts for RL) or lower res / frame-skip.
- Visual domain gap Roblox↔Godot sandbox: BC transfer will likely need
  augmentation/fine-tuning; zero-shot transfer is not assumed.
- godot_rl_agents maintenance pace is slow (single maintainer, MIT — forkable).
- Godot physics is resettable/seedable but not guaranteed bit-exact
  deterministic across runs/machines; don't build evaluation on exact replay.
- Windows quirk reports exist for the GDRL TCP connection (permissions);
  workarounds documented upstream.

## 10. For the next AI session

- **State**: Stack validated end-to-end on Linux (connection, PPO, parallel
  envs, raycast obs). Repo contains only the feasibility kit + this file.
- **Immediate task**: the owner runs `feasibility/README.md` Tests 1–3 on the
  Windows 11 / RTX 4060 Ti machine (raycast benchmark, PPO benchmark, 84x84
  VirtualCamera pixel benchmark) and fills in the results table.
- **Then**: with real numbers, decide pixel-RL vs hybrid-obs strategy and
  green-light M1 (minimal greybox sandbox env in Godot).
- **Don'ts**: don't build the full TTK-inspired sandbox yet; don't add
  dependencies; don't touch Roblox automation; don't restructure the repo.
- When you change anything material (versions, results, decisions), update
  this file in the same commit.
