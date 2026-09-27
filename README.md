# SandboxAI

SandboxAI is a **local-first, free/open-source FPS learning platform**. It records a human player's screen and controls, builds a versioned dataset, trains a temporal vision policy by behavioral cloning, runs that policy in a controlled tactical FPS, improves it with PPO, and compares all policies on deterministic episodes.

```text
manual play → synchronized dataset → validate/inspect → BC + offline evaluation
            → closed-loop sandbox play → BC-warm-started PPO → benchmark → repeat
```

TTK Testing/Roblox is **manual demonstration data only**. SandboxAI never controls, injects into, reads memory from, or performs RL in Roblox. All AI control and RL run in SandboxAI's own Python/Godot environments.

## What is included

- MSS screen capture and non-invasive OS input logging at synchronized monotonic timestamps
- Versioned session format (`metadata.json`, `samples.jsonl`, `frames/`)
- Validation, aggregate catalogs, statistics, and visual HTML inspection
- Lightweight IMPALA-style CNN with optional GRU and ten action heads
- Offline imitation metrics and stateful real-time inference
- A deterministic, visual tactical FPS reference environment for fast headless training
- A manually playable Godot 4 arena with movement, cover, camera control, ADS, ammo, reload, recoil, health, moving enemies, shooting, deterministic reset, and a local-only AI bridge
- Stable-Baselines3 PPO with transferred BC encoder and action-head weights
- Random, scripted-vision, BC, and PPO comparison on identical seeds
- Repeated train/evaluate cycles, checkpoints, manifests, metric JSONL, telemetry, CLI status, and a local web dashboard

The full design and verification record is in [PROJECT.md](PROJECT.md).

## Install (Windows 11 / RTX 4060 Ti)

Install Godot 4.3 and Python 3.11, place the Godot executable in the repository root (or on `PATH`), then:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
.venv\Scripts\activate
python -m sandboxai doctor
```

The setup script installs CUDA PyTorch first, installs the Python dependencies and Godot export templates, and exports `build\sandbox_windows.exe`. Add `-IncludeFeasibility` only when you also need the historical M0 godot-rl examples.

Portable/manual setup:

```bash
python -m venv .venv
# Activate .venv, then install a platform-appropriate PyTorch build first.
pip install -r requirements.txt
python -m sandboxai doctor
```

## 1. Play and record

Export and launch the custom sandbox:

```powershell
python -m sandbox.export --godot C:\path\to\Godot_v4.3-stable_win64_console.exe
python -m sandboxai play --env_path build\sandbox_windows.exe
```

Manual controls: `WASD`, `Shift` sprint, `Ctrl/C` crouch, `Space` jump, mouse look, left-click fire, right-click ADS, `R` reload, `Esc` release mouse, then click to recapture it.

Record manual SandboxAI gameplay:

```powershell
python -m data_pipeline.record --source godot_sandbox --window "SandboxAI" --fps 15 --output datasets\sandbox_manual
```

Record manual TTK gameplay (capture only; no game automation):

```powershell
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --output datasets\ttk_manual
```

A synthetic installation check is also available:

```powershell
python -m data_pipeline.record --mock --duration 5 --output datasets\smoke
```

## 2. Validate and inspect data

```powershell
python -m data_pipeline.validate --dataset datasets\ttk_manual\<session_id>
python -m data_pipeline.stats --dataset datasets\ttk_manual\<session_id>
python -m data_pipeline.inspect --dataset datasets\ttk_manual\<session_id> --output logs\inspection.html
python -m data_pipeline.catalog --data datasets --output logs\dataset_catalog.json
```

The validator detects incomplete recordings, malformed samples, timestamp/action errors, missing/corrupt/duplicate frames, unsafe paths, incompatible schemas, and metadata/count mismatches. Training splits demonstrations by session; a single session uses a chronological split with a context gap.

## 3. Train and evaluate imitation learning

Temporal training is the default CLI mode:

```powershell
python -m bc.train --data_dir datasets --epochs 10 --batch_size 32 --seq_len 4 --gru
python -m evaluation.imitation --checkpoint checkpoints\bc_best.pt --data datasets\held_out_session
python -m bc.sandbox_runner --checkpoint checkpoints\bc_best.pt --episodes 5 --steps 300
```

Checkpoints include architecture, image size, sequence settings, and exact mouse-bin edges. Resume with `python -m bc.train ... --resume checkpoints\bc_latest.pt`.

## 4. Reinforcement learning and comparison

PPO can start from the learned BC visual encoder **and all ten compatible action heads**:

```powershell
python -m rl.train --timesteps 10000 --bc_checkpoint checkpoints\bc_best.pt
python -m evaluation.benchmark --bc_checkpoint checkpoints\bc_best.pt --ppo_checkpoint checkpoints\ppo_sandbox.zip --episodes 10
```

Run repeated train → evaluate → checkpoint cycles:

```powershell
python -m rl.iterate --bc_checkpoint checkpoints\bc_best.pt --cycles 3 --timesteps 10000
```

By default these commands use the deterministic Python reference sandbox for throughput. To use an exported Godot process through the local TCP bridge, add:

```powershell
--backend godot --env_path build\sandbox_windows.exe
```

An explicitly requested Godot backend fails clearly if it cannot start; it never silently trains in a different environment.

## One-command end-to-end workflow

With existing manual datasets:

```powershell
python -m sandboxai e2e --data_dir datasets --bc_epochs 10 --rl_timesteps 10000
```

Fast installation verification, including synthetic recording, BC, closed-loop execution, BC-warm-started PPO, and the four-policy benchmark:

```powershell
python -m sandboxai e2e --quick --generate_demo
```

Outputs are stored under `checkpoints/` and `logs/`. These directories are intentionally ignored by Git.

## Monitoring

```powershell
python -m sandboxai status
python -m sandboxai dashboard --port 8765
```

Open `http://127.0.0.1:8765`. The dashboard reads the same stable JSON state/experiment formats intended for a future control-center application. It displays datasets, active checkpoints, training speed, runtime actions/rewards, benchmark results, hardware, and run history.

## Shared action contract

Gym actions are `MultiDiscrete([3,3,2,2,2,2,2,2,21,21])` in this order:

1. lateral movement
2. forward/back movement
3. jump
4. crouch
5. sprint
6. reload
7. fire
8. ADS
9. horizontal look bin
10. vertical look bin

The recorder, BC model, Python environment, Godot bridge, PPO, and evaluators all use this contract. Dataset movement remains human-readable `-1/0/+1`; `sandbox.actions` is the single conversion layer.

## Test

```powershell
pytest -q
```

The suite covers recording, synchronization, schema/validation, visual data inspection, temporal BC, inference, deterministic FPS mechanics, closed-loop execution, PPO, benchmarks, monitoring, experiment persistence, and integration behavior.

## Hardware profile

Defaults are intentionally conservative for an RTX 4060 Ti with 8 GB VRAM: 84×84 RL observations, 160×120 demonstrations, a small residual CNN/GRU, AMP on CUDA, one or two environments, and incremental checkpoints. No cloud service or paid API is required.
