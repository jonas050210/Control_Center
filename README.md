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
- Current phase: **M0 feasibility** — proving Godot 4.3 + godot-rl 0.8.2 +
  SB3 + PyTorch run reliably and fast enough on the target machine
  (Windows 11, RTX 4060 Ti 8 GB, Python 3.11).

## Quickstart (Windows 11, from a fresh clone)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This creates the venv, installs the pinned stack (torch cu124, godot-rl
0.8.2, SB3 2.4.0, Gymnasium 1.0.0), clones the pinned godot_rl example
projects with the SandboxAI overlay, installs the Godot 4.3 export templates
and exports the RL environment executables. Safe to re-run; see the script
header for options.

Then run the benchmarks:

```powershell
.venv\Scripts\activate
python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30
python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2
python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30
```

The full runbook with all tests and the results table is in
[`feasibility/README.md`](feasibility/README.md).
