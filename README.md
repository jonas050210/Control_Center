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
- Core pipeline:
  ```text
  Manual human gameplay (TTK on Roblox / Godot)
        │
        ▼ (M1: data_pipeline)
  Screen capture (160x120) + OS input logging + microsecond timestamp sync
        │
        ▼ (M1: schema v1.0.0)
  Versioned dataset (metadata.json + samples.jsonl + frames/)
        │
        ▼ (Phase 2: bc)
  Behavioral Cloning training (PyTorch ResNet/IMPALA CNN + multi-head actions)
        │
        ▼ (Phase 4: bc.sandbox_runner)
  BC policy closed-loop execution inside Godot tactical sandbox
        │
        ▼ (Phase 5: rl.train)
  PPO reinforcement learning fine-tuning inside Godot tactical sandbox
        │
        ▼ (Phase 6: evaluation.benchmark)
  Evaluation & benchmarking (Random vs BC vs PPO)
        │
        ▼ (Phase 7: monitoring)
  System telemetry & state tracking dashboard
  ```

---

## Quickstart (Windows 11, from a fresh clone)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This creates `.venv`, installs PyTorch (CUDA 12.4), data pipeline dependencies, and pinned RL tools (godot-rl 0.8.2, SB3 2.4.0, Gymnasium 1.0.0), clones the examples with the SandboxAI overlay, installs Godot export templates, and exports the benchmark binaries.

Activate the virtual environment:
```powershell
.venv\Scripts\activate
```

---

## Complete End-to-End Workflow & Commands

### 1. Data Pipeline (Recording & Verification)

#### Synthetic Mock Recording (Test without Roblox):
```powershell
python -m data_pipeline.record --mock --duration 5 --output datasets/mock_session
```

#### Real Manual Recording (TTK Testing on Roblox):
Launch Roblox TTK Testing, then run:
```powershell
python -m data_pipeline.record --source ttk_testing --window Roblox --fps 15 --output datasets/ttk_pilot
```
*(Press `Ctrl+C` in the terminal when you finish playing)*

#### Validate & Inspect Dataset:
```powershell
python -m data_pipeline.validate --dataset datasets/mock_session/<session_id>
python -m data_pipeline.stats --dataset datasets/mock_session/<session_id>
```

---

### 2. Behavioral Cloning (BC Training & Inference)

#### Train BC Policy from Recorded Datasets:
```powershell
python -m bc.train --data_dir datasets/ --epochs 10 --batch_size 32 --checkpoint_dir checkpoints/
```

#### Test BC Action Inference on a Frame:
```powershell
python -m bc.infer --checkpoint checkpoints/bc_best.pt
```

#### Run BC Policy Closed-Loop inside Tactical Sandbox:
```powershell
python -m bc.sandbox_runner --checkpoint checkpoints/bc_best.pt --episodes 5
```

---

### 3. Reinforcement Learning (PPO in Sandbox)

#### Train PPO Agent in the Tactical Arena:
```powershell
python -m rl.train --timesteps 10000 --checkpoint_dir checkpoints/
```

---

### 4. Benchmark & Policy Comparison

Compare Random Policy vs BC Policy vs PPO Policy on identical evaluation seeds:
```powershell
python -m evaluation.benchmark --bc_checkpoint checkpoints/bc_best.pt --ppo_checkpoint checkpoints/ppo_sandbox.zip --episodes 5
```

---

### 5. Telemetry & Monitoring Dashboard

```powershell
python -m monitoring.status
```

---

### 6. Run Automated Test Suite (59 Tests)

```powershell
pytest -v
```

---

## M0 Feasibility Benchmarks (Reference)

```powershell
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
```

The full runbook with all feasibility tests and the results table is in
[`feasibility/README.md`](feasibility/README.md).
