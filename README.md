# SandboxAI

Vision-based AI learning project: a human plays **TTK Testing [HARDPOINT]**
(Roblox) *manually*; screen + input recordings are captured; an imitation
learning / behavioral-cloning policy (PyTorch) is trained from them; the
learned agent is evaluated and improved with reinforcement learning inside
**our own controlled Godot sandbox**.

**TTK Testing is ONLY a manual gameplay/data source.** No automation,
injection, memory reading or anti-cheat bypass of the public Roblox game —
ever. The RL environment is our own Godot project, not Roblox.

- **Read [`PROJECT.md`](PROJECT.md) first** — it is the single source of truth
  (context, decisions, measured results, boundaries).
- Current status:
  - **M0 Feasibility**: Godot 4.3 + godot-rl 0.8.2 + SB3 PPO export/benchmark pipeline.
  - **M1 Data Pipeline**: Real-time screen capture + OS-level input listener + timestamp synchronization + versioned dataset schema + validation and inspection tools (*"KI guckt erst zu und lernt bei mir dazu"*).

---

## Quickstart (Windows 11, from a fresh clone)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This creates `.venv`, installs PyTorch (CUDA 12.4), data pipeline dependencies, and pinned RL tools (godot-rl 0.8.2, SB3 2.4.0, Gymnasium 1.0.0), clones the examples with the SandboxAI overlay, installs Godot export templates, and exports the benchmark binaries.

---

## M1 Data Pipeline Usage

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

### 1. Run a Synthetic Mock Recording (Test without Roblox)

Verify the entire capture, synchronization, serialization, and validation pipeline in seconds:

```powershell
python -m data_pipeline.record --mock --duration 5 --output datasets/mock_session
```

### 2. Record Real Manual Gameplay (TTK Testing on Roblox)

Launch Roblox TTK Testing, then run:

```powershell
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --output datasets/ttk_run
```

- Target FPS: ~15 Hz (configurable via `--fps 15`)
- Target Resolution: 160x120 RGB (configurable via `--width 160 --height 120`)
- Stop recording at any time: press `Ctrl+C`.

### 3. Validate a Dataset Session

Check schema conformance, frame integrity, monotonic timestamps, and action bounds:

```powershell
python -m data_pipeline.validate datasets/mock_session/<session_id>
```

### 4. Inspect Dataset Metrics & Action Histograms

View detailed step statistics, keyboard frequencies, and mouse rotation histograms:

```powershell
python -m data_pipeline.inspect datasets/mock_session/<session_id>
```

### 5. Run the Automated Test Suite

```powershell
pytest -v
```

---

## M0 Feasibility Benchmarks

```powershell
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
```

The full runbook with all feasibility tests and the results table is in
[`feasibility/README.md`](feasibility/README.md).
