# Debug GUI and benchmarking

> **See also:** the small overlay documented here is the *minimal* in-scene
> debug GUI used by `scenes/main.tscn`. The full operator interface — modes,
> simulation controls, agent/perception/observation inspection, results and
> logs — is the Control Center: [`CONTROL_CENTER.md`](CONTROL_CENTER.md).
> Both are presentation-only and neither is created in headless runs; the
> Control Center reuses `DebugOverlay.build_telemetry_dict()` rather than
> duplicating it.

## Debug / test GUI

`scripts/debug/debug_overlay.gd` (`DebugOverlay`) is a small, presentation-only
`CanvasLayer` used only by the graphical scene (`scripts/core/main.gd` /
`scenes/main.tscn`). It is **never** created by the headless RL bridge
(`scripts/rl/rl_server.gd`), so it has zero effect on headless training or
the JSON-lines protocol — this is verified by
`tests/test_headless_entrypoints.gd` staying green and by the overlay's own
logic living in two pure, dependency-free functions:

- `DebugOverlay.build_telemetry_dict(simulation_manager, env_index) ->
  Dictionary` — reads current state into a plain Dictionary. No Control/Label
  node is touched; this is what `tests/test_debug_overlay.gd` exercises.
- `DebugOverlay.format_lines(telemetry: Dictionary) -> PackedStringArray` —
  pure text formatting.

`_process()` just calls both and assigns the result to a `Label`, so the only
engine-coupled part is the label text assignment and the button wiring
(both trivial, no gameplay logic).

### What it shows

- Render FPS, simulation steps/second, time scale (pause indicator).
- Active environment count, focused environment index, seed, curriculum
  level/name.
- Episode number, timestep (step count within the episode).
- Agent HP, position, aim yaw/pitch, weapon-ready flag.
- Per-enemy list: index, AI state (idle/chase/attack/dead), HP, position.
- Shots fired/hit, accuracy, kills (episode + lifetime total), deaths
  (episode + lifetime total).
- Episode cumulative reward, last-step reward, reward breakdown (from
  `EpisodeState.get_reward_breakdown()`), survival time.
- The last resolved `Action` applied to the focused environment
  (`SimulationManager.get_last_actions()`).
- A condensed observation summary (agent health, primary-enemy
  distance/bearing/aliveness, alive-enemy-count fraction, in-combat flag) —
  not the full 33-float raw vector, to stay readable.

### Controls

All controls call existing public `SimulationManager`/`EnvironmentCore`
methods — the overlay adds no new simulation behavior:

- **Pause / Resume** — toggles `Engine.time_scale` between `0.0` and `1.0`.
- **Reset Episode** — calls `EnvironmentCore.reset(-1)` on the focused
  environment (continues its deterministic seeded RNG stream, same as the
  Python-driven auto-reset path).
- **Enemies -/+** — calls `SimulationManager.build(environment_count,
  new_enemy_count)`. This is a full rebuild (all environments restart);
  there is no lighter-weight "resize enemies without touching anything
  else" API in `SimulationManager`, and adding one purely for the debug GUI
  was judged not worth the extra surface area.
- **Level -/+** — calls `SimulationManager.set_curriculum_level(new_level)`.
- **`< Env` / `Env >`** — changes which environment's telemetry is
  displayed; does not touch simulation state.

### Headless independence

`sandboxai train` / `sandboxai benchmark` / `sandboxai evaluate` all launch
`godot --headless --script res://scripts/rl/rl_server.gd`, which never
instantiates `Main`, `DebugOverlay`, `EnvironmentView`, or any other
presentation node. `SimulationManager.create_visuals = false` is set
explicitly by the bridge.

## Benchmarking

`python/sandboxai/benchmark.py` (`sandboxai benchmark`) measures headless
Godot throughput across a sweep of parallel-environment counts inside a
single Godot process.

```bash
sandboxai benchmark --env-counts 1,2,4,8,16,24,32,48,64 \
  --steps 2000 --max-seconds-per-config 20 \
  --output-dir training/benchmarks/4060ti
```

Each environment count is time-boxed (`--max-seconds-per-config`, default
20 s) so the full recommended sweep finishes in a few minutes instead of
being proportional to `steps * len(env_counts)`. The result JSON/CSV
includes, per environment count: `steps_per_second`, `episodes_per_second`,
whether the run was time-boxed, and a best-effort resource snapshot (CPU%,
process RSS, CUDA allocated/reserved memory if PyTorch+CUDA are available).
`summarize_scaling()` (also printed by the CLI) picks the environment count
with the best measured throughput and flags where returns start
diminishing relative to environment-count growth.

### Where the bottleneck actually is

This project could not run the benchmark against a real Godot binary or the
target RTX 4060 Ti 8GB GPU inside this development sandbox (no Godot
executable or NVIDIA GPU is available here — see the Limitations section of
the final report). The architecture itself, however, makes the likely
bottleneck predictable and worth stating honestly instead of guessing at
numbers:

- **Not VRAM/GPU compute.** The policy is a tiny 2-layer MLP (default
  `net_arch=(128,128)`) over a 33-float vector. This needs a few hundred KB
  of parameters; an RTX 4060 Ti 8GB is enormous overkill for this network,
  and training is very unlikely to be GPU-compute-bound. GPU time is mostly
  idle waiting for environment steps.
- **Likely bottleneck: the single-process JSON-lines bridge and Python's
  GIL.** All N environments run inside *one* Godot process, and Python talks
  to it over one stdin/stdout pipe, one request/response round-trip per PPO
  rollout step (batched: all N actions go in one `step` request, all N
  observations come back in one response). This means:
  - Godot's own per-step simulation cost (analytic, no physics — cheap) is
    likely not the limiter at low environment counts.
  - JSON parsing/serialization cost per step scales with N and with
    observation width (33 floats × N enemies' worth of extra fields is
    still small, but not zero).
  - Both the Godot process and the Python process are single-threaded for
    this bridge traffic (only stderr draining runs on a helper thread), so
    very large N eventually bottlenecks on serialization + IPC rather than
    on actual simulation math.
- **CPU** (single-core-bound serialization/IPC, and Godot's own script
  execution) is the most likely practical ceiling, well before RAM or VRAM.
- **RAM** should stay modest — analytic state per environment is a handful
  of small objects, not full physics/render scenes (`create_visuals=false`
  headless mode never allocates render nodes).

### Practical guidance

Given the above, the previously-validated successful configuration
(24 environments) is a reasonable *starting point*, not necessarily the
ceiling or the optimum — the goal of the benchmark sweep is to find the
environment count where `steps_per_second` stops growing roughly linearly
with environment count (the "diminishing returns" point `summarize_scaling`
reports), not to maximize environment count for its own sake. Run
`sandboxai benchmark` on the target machine (i7-12700F / RTX 4060 Ti 8GB /
32GB RAM) before committing to a larger environment count for a long
training run.
