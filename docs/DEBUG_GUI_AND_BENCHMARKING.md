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
  not the full 84-float raw vector, to stay readable.

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

### Profiling the complete PPO path

Use the opt-in profiler for a representative training run:

```bash
sandboxai train --env-count 4 --enemy-count 1 --seed 42 \
  --steps 106496 --device cpu --profile-training
```

The run writes `logs/training_profile.json`. It measures:

- rollout collection versus the blocking PPO update after each rollout;
- environment step time, NumPy conversion and integrated-pipeline hook time;
- Python JSON encode, pipe write/flush, response wait and JSON decode time,
  plus byte totals per bridge command;
- Godot request parse, command handling (including `command_step`), response
  encode and stdout write time when the bridge supports the profiling command;
- synchronous evaluation, checkpoint saves and resource-snapshot lookup;
- a full evaluation-boundary breakdown: `eval.env_startup` (bridge process
  spawns), `eval.normal.*` (the frozen normal evaluation: total wall time,
  policy prediction, environment stepping, and its bridge's transport
  timings under `eval.normal.bridge.*`), and `eval.battery.*` (the
  checkpoint condition/generalization battery, split per section plus its
  own predict/step/bridge buckets). Evaluation buckets are namespaced, so
  they never fold into the training bridge's `bridge.step.*` numbers;
- one row per PPO iteration, which makes periodic throughput cliffs visible.

Profiling is aggregate rather than a per-step trace, and is disabled by
default. Timings are nested: for example, `env.step_total` includes all
`bridge.step.*` buckets, while `bridge.step.wait_response` includes Godot and
pipe transit. The Godot-side section is what separates simulation from that
wait bucket. `eval.env_startup` firing more than twice in a whole run
(normal evaluation + battery) means evaluation bridge processes are being
respawned per boundary again — that is the single most expensive regression
to watch for, since each spawn pays full engine and project startup.

### What causes the reported FPS drops

The displayed SB3 FPS is a cumulative training rate, not a pure environment
step rate. With 4 environments and the default rollout length 2048, each PPO
iteration collects 8192 actual timesteps and then stops stepping Godot while
PPO performs its default 10 epochs: `10 * 8192 / 256 = 320` minibatch updates.
The first SB3 FPS row has not yet paid an update; later rows have. A large
stepwise fall after each rollout is therefore expected even with an infinitely
fast environment.

The compact `(84 -> 128 -> 128)` MLP produces many very small matrix and
distribution operations. On CUDA, launch, host/device transfer and framework
synchronization overhead dominate these tiny kernels; low utilization and
~0.75 GB allocated VRAM are evidence of under-filled hardware, not a request
for a larger GPU. CPU being about twice as fast for the reported workload is
consistent with that shape.

Other intentional stop-the-world work is easy to distinguish in the profile:

- evaluation is synchronous at every `evaluation_frequency`; auto-curriculum
  mode also runs the configured checkpoint condition/generalization battery;
- checkpoint serialization is synchronous;
- an auto-curriculum episode boundary stages the next plans through one extra
  bridge request, while episode logging/replay selection runs in the trainer
  thread;
- a Control Center-managed run polls its cooperative command file from the
  per-vector-step callback.

Evaluation is synchronous by design, not by accident: its results feed
best-checkpoint selection, early stopping (`early_stopping_patience`,
`min_eval_reward`) and the battery's frozen `policy.zip`/curriculum snapshot,
all of which must be strictly ordered against training progress for a run to
be reproducible. What has been removed is the *avoidable* cost inside that
synchronous block (see below), not the ordering.

### Removed avoidable overhead

Seven hot-path costs were unnecessary and are now avoided without changing
observations, actions, rewards, episode boundaries, evaluation coverage,
checkpoint selection, early stopping, RNG use or PPO data:

1. Training telemetry used to launch `nvidia-smi` synchronously every 100
   vector steps (every 400 actual timesteps with 4 environments). A 106,496
   timestep run therefore launched it about 266 times. It now runs in a
   bounded background `ResourceMonitor`; the callback only copies the latest
   sample. A slow/missing probe can no longer pause rollout collection.
2. Every non-terminal step used to construct and serialize a full cumulative
   episode-metrics dictionary (including reward breakdown) and a per-step
   reward-components dictionary. PPO, skill metrics and replay recording use
   the event dictionary on those steps and consume cumulative metrics only at
   episode end. `compact_training_infos` (on by default, independently
   disableable in config/CLI) retains events and exact terminal metrics while
   omitting only those redundant non-terminal dictionaries. The public
   `GodotBatchClient` default remains full-info mode.
3. Every evaluation boundary spawned fresh Godot bridge processes: one for
   the normal evaluation and one for the checkpoint battery (three with the
   league enabled). Each spawn pays full engine and project startup. Both
   processes now live for the whole run and are re-seeded over the bridge at
   every boundary (`reset(seed)` / staged episode plans), which reproduces
   bit-identical episodes because the engine derives the world
   deterministically from the seed. `eval.env_startup` in the profile shows
   exactly two spawns per run instead of two per boundary.
4. Both evaluation roles now batch deterministic inference. The battery runs
   `checkpoint_eval_environment_count` (default 8, CLI
   `--checkpoint-eval-env-count`). Normal evaluation now explicitly stages
   the historical episode set (`seed + episode_index`) as complete plans and
   sorts results back into episode-index order, so it can safely use
   `evaluation_environment_count` (now default 8) without changing seeds,
   coverage or summary order. Serial and vector paths are regression-tested
   row for row.
5. The battery bridge requests compact infos (it consumes only events and
   terminal metrics, both retained in compact mode), cutting its per-step
   response payload like the training bridge already does.
6. Normal and battery evaluation use separate persistent Godot processes, so
   they now execute at the same time. Access to the shared frozen policy is
   serialized (the short inference/save calls cannot race), while the costly
   independent simulation steps overlap. Both jobs are joined before reward
   comparison, `best_eval.zip`, patience updates or reward-threshold stopping;
   those semantics and their ordering are unchanged. Profiles expose
   `eval.parallel.wall` and the measured `eval.parallel.overlap`.
7. A plan executor now pre-stages each slot's next plan. Godot consumes that
   plan in the terminal step's existing auto-reset instead of first resetting
   a throwaway world, returning to Python, staging, issuing `reset_indices`,
   and resetting the requested world again. Condition and generalization
   plans also share one continuous executor call, eliminating a second batch
   reset and a separate partially-filled vector tail. The normal rows use
   `normal_episodes.csv`; generalization retains the historical
   `episodes.csv`, avoiding concurrent file overwrites.

For CUDA runs there is additionally `--inference-device cpu`: rollout
collection and frozen evaluation then run their policy forward passes on CPU
while PPO updates stay on the GPU. For this tiny policy the per-step
host<->device round trip and kernel-launch overhead exceed the matmuls
themselves, which is why CUDA measured ~2.5x slower than CPU on this
workload. This is opt-in (default `auto` keeps inference coupled to the
training device) because the sampled rollout actions then follow the CPU RNG
stream — deterministic and reproducible, but not bit-identical to a
coupled-device run.

A schema-representative 4-environment JSON sample was 13,576 bytes in full
mode and 8,592 bytes in compact mode (36.7% smaller); Python `json.loads` on
this sandbox fell from 103.2 us to 73.7 us per response (28.6%). These are
serialization microbenchmarks, not claimed Godot/Windows end-to-end FPS.
Run the profile above on the target Windows/WSL machine for authoritative
before/after wall time.

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

## Cost of the world/perception layer

The world, perception, sound and memory systems are all **opt-in per
curriculum level**, and the headless hot path is written so the cheap
levels stay exactly as cheap as they were:

| Level range | Per-step work added vs. the original implementation |
| --- | --- |
| 1–4 | None. `world` is `null`, `AgentPerception.update()` is never called, `SoundBus.tick()` is never called, and `Observation.build()` takes the original 3-argument path. The only difference is that the observation array is 84 floats instead of 33 (the extra 51 are written from already-computed values or left at their "no information" defaults). |
| 5 | Collision + ground queries per character (O(obstacles) axis-separated box tests), `EnemyBrain` instead of `update_ai()`. |
| 6 | Adds per-enemy FOV + line-of-sight: 2 ray samples per enemy per tick, each O(obstacles). |
| 7 | Adds sound: bounded at `SOUND_MAX_ACTIVE = 24` events, each sampled with one occluder count per listener. |
| 8 | Adds memory decay: O(tracked contacts), capped at `MEMORY_MAX_TRACKS = 8`. |
| 9–10 | Adds jump/ground resolution for enemies; same order of cost as level 5. |

Deliberate design choices that keep this bounded:

- Layouts are flat arrays of axis-aligned boxes (8–24 of them), tested with
  the slab method. There is no BVH because at this obstacle count a linear
  scan is faster than traversing one.
- `EnemyBrain` receives a **reused** context dictionary owned by
  `EnvironmentCore`, so the tactical path allocates nothing per tick.
- Sound events and memory tracks both have hard caps, so a long episode
  cannot grow the per-step cost.
- `debug_perception` is off by default and is only ever set by the Control
  Center, for the selected environment, outside TRAINING mode.

### Measuring it on the target machine

No throughput numbers are published here, because no Godot executable was
available in the environment these changes were written in and **fabricated
benchmark numbers are worse than none**. To get real ones:

```bash
sandboxai benchmark --env-counts 1,4,8,16,32,64 --steps 2000
```

Run it once per curriculum level you intend to train at (levels 1–4 and
levels 5–10 have materially different per-step costs, so a single sweep
does not characterise both), and pick the environment count where
steps/second stops scaling linearly. On an i7-12700F / RTX 4060 Ti 8 GB /
32 GB / Win11 box the binding constraint is expected to be CPU-side
simulation and the JSON bridge rather than GPU memory — the policy network
is a small MLP — but that expectation must be confirmed with the sweep
above before it is used to size a training run.
