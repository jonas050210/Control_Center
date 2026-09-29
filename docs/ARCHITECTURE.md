# SandboxAI architecture

## Runtime split

SandboxAI has two deliberately independent runtimes:

1. **Godot simulation**: deterministic Agent/Enemy/Weapon state, action
   resolution, reward calculation, episode lifecycle and optional rendering.
2. **Python ML tooling**: Gymnasium/SB3 PPO, PyTorch BC, evaluation,
   checkpoints, telemetry, datasets and benchmarking.

The bridge is newline-delimited JSON over the Godot process's stdin/stdout.
It avoids a renderer, native plugins, browser networking and machine-specific
paths. One Godot process owns a batch of independent environments.

```
Godot rl_server.gd
  -> SimulationManager
    -> EnvironmentCore[0..N)
      -> ArenaWorld            (seeded geometry, collision + ray queries)
      -> CharacterMotor        (shared gravity/jump/collision integration)
      -> SoundBus              (transient audible events)
      -> AgentPerception       (what the POLICY may know: FOV/LOS/memory)
      -> EnemyBrain            (what the OPPONENTS know and do)
      -> AgentState / EnemyState[] / WeaponState / EpisodeState
  -> RLAdapter
  <==== JSON lines ====>
Python GodotBatchClient
  -> GodotVecEnv (SB3) / GodotGymEnv (evaluation)
```

To inspect the protocol manually, launch the server and send one JSON object
per line:

```text
{"cmd":"spaces"}
{"cmd":"reset","seed":1234}
{"cmd":"step","actions":[[1,1,1,1,0,0]]}
{"cmd":"close"}
```

## Action contract

The canonical Godot `Action` has four ternary fields and two binary triggers:

- `move_axis`, `strafe_axis`, `look_yaw_axis`, `look_pitch_axis`: `-1, 0, 1`
- `shoot`: boolean
- `jump`: boolean (added with curriculum level 9, vertical combat)
- `look_delta`: optional continuous mouse delta, retained for human logs

The ML space is `MultiDiscrete([3, 3, 3, 3, 2, 2])`. Each ternary field is
shifted by one (`-1 -> 0`, `0 -> 1`, `1 -> 2`). `Action.from_discrete()`
continues to support Agent 1's ten single-choice actions for compatibility.
Human, stub AI, external PPO and demonstrations all pass through `Action`.

## Observation contract

`Observation.to_array()` always returns **84** float32-compatible values
(contract v3). The layout is strictly additive across three generations:

- **0–32 (v1)** agent state, the primary/nearest-alive enemy, weapon-ready
  and in-combat flags, two more individually tracked enemies.
- **33–64 (v2)** vertical state, perception (FOV/LOS/visibility), memory
  age and confidence, sound, and world context.
- **65–83 (v3)** perceived brightness at the agent, statistics for contacts
  beyond the three tracked slots, target-selection priority and switch
  recency, second-sound bearing/loudness plus hearing uncertainty, and the
  agent's own map knowledge (explored fraction, whether the current area is
  known, time since it was last here, remembered cover and remembered
  danger).

No map id, lighting mode, enemy count, spawn list or geometry dump is ever
part of the vector; a map is an environment, not a label. See
[`docs/OBSERVATION_ACTION_CONTRACT.md`](OBSERVATION_ACTION_CONTRACT.md) for
the full field-by-field table and normalization rules — it is the canonical
reference; this section only summarizes it.

The primary enemy is the nearest alive target, with a stable dead-target
fallback. RGB and temporal stacking are interfaces only; there are no image
processing dependencies or vision weights in this milestone.

## Environment lifecycle and metrics

`reset(seed)` starts a new episode and seeds a local Godot
`RandomNumberGenerator`. `step(action, dt)` returns:

```gdscript
{
  "observation": Observation,
  "reward": float,
  "done": bool,
  "info": {
    "events": Dictionary,
    "done_reason": String,
    "metrics": Dictionary
  }
}
```

Metrics include reward, episode length, kills, deaths, damage dealt,
damage received, survival time, accuracy, shots fired/hit and win/loss.
`SimulationManager` can auto-reset a completed environment while preserving
its terminal observation under `terminal_observation`; this matches vector
Gym semantics. The Python wrapper returns a fresh reset observation and keeps
terminal info in the `info` dictionary.

The simulation is analytic rather than PhysicsServer-driven. This is
intentional: it is deterministic and fast in headless mode. Views mirror
state but never drive it. A `create_visuals=false` manager creates no visual
nodes and can be used without a window.

## PPO pipeline

`python/sandboxai/ppo.py` creates `GodotVecEnv`, starts `rl_server.gd`, and
constructs SB3 PPO with:

- MLP policy with two 128-unit hidden layers by default
- `MultiDiscrete` action space
- configurable learning rate, rollout length, batch size, gamma, GAE lambda,
  entropy coefficient, clip range, seed, total steps and device
- TensorBoard and JSONL telemetry
- periodic SB3 checkpoints
- isolated evaluation callback and best-evaluation checkpoint

`TrainingConfig` is the central serializable configuration. `device=auto`
selects CUDA only when PyTorch reports it available; CPU is the fallback.
The Godot process receives environment count, enemy count, curriculum level
and seed explicitly.

A checkpoint resume loads the SB3 archive with its optimizer state and calls
`learn(reset_num_timesteps=false)`. The latest checkpoint is written after a
successful run, while `best_eval.zip` is only replaced by a strictly better
evaluation.

At an auto-curriculum evaluation boundary, normal evaluation and the
condition/generalization battery run on two independent persistent bridge
processes concurrently. Both use explicit per-episode plans and batched
`deterministic=True` inference; access to the shared frozen policy is locked.
The callback joins both workers before checkpoint selection, patience updates
or reward-threshold stopping, so parallel execution cannot reorder training
control decisions. The plan executor pre-stages the next episode for Godot's
terminal auto-reset and runs condition plus generalization as one ordered
batch, avoiding duplicate resets and vector-tail work.

## BC and BC-to-PPO

`DemonstrationRecorder` in Godot attaches to `SimulationManager` and logs the
existing human action pipeline. The first JSONL line is dataset metadata;
remaining lines are transition records:

```json
{
  "observation": [...], "action": [...], "next_observation": [...],
  "reward": 0.01, "done": false, "timestamp": 123.4,
  "episode_id": 0, "environment_id": 0, "info": {...}
}
```

`python/sandboxai/dataset.py` validates and loads this format without
PyTorch. The BC model has a two-layer tanh backbone and five categorical
heads. Training uses a deterministic train/validation split, Adam,
checkpointed optimizer state, CSV loss curves and JSONL accuracy metrics.

The BC-to-PPO transfer is explicit and conservative. It copies BC hidden
layers and concatenated categorical heads only if SB3 parameter names and
shapes are compatible. A mismatch raises an error; no fake or partial warm
start is reported.

## Curriculum and self-play

`CurriculumConfig` changes existing `EnemyState` behavior instead of making
parallel hardcoded environments. Eleven levels
(`CurriculumConfig.Level`), each teaching one new capability:

1. stationary target, fixed spawn directly ahead, no movement/attacks
2. one moving enemy, varied spawn distance/angle
3. moving + attacking enemy, spawn variety, strafing while engaging
4. 3+ enemies, spawn variety, strafing, faster/more aggressive
5. obstacles and cover (`EnemyBrain` takes over)
6. FOV + line-of-sight gating (the observation stops being ground truth)
7. sound
8. memory of lost targets
9. vertical combat
10. mixed randomized layouts/scenarios
11. agent-vs-agent self-play hook

The Python mirror is `python/sandboxai/curriculum_stages.py`
(`CurriculumDirector`): a stage table, an `AutoCurriculum` gate and a
deterministic episode distribution per stage. Promotion is
performance-gated over a rolling window with a per-stage minimum episode
count and a cooldown, so no single episode can promote and levels move one
step at a time.

See [`docs/CURRICULUM_AND_COMBAT.md`](CURRICULUM_AND_COMBAT.md) for the full
per-level table, the multi-enemy spawn-variety algorithm, and the
deterministic strafing/movement-pattern design.

`SelfPlayEnvironmentCore` has two `AgentState` slots, independent RNG seeds,
mirrored observations, per-agent rewards and per-agent metrics. Python's
`SelfPlayCoordinator` can load a frozen opponent checkpoint. This is a
foundation for population/self-play training, not a population algorithm.

## Replay, metrics, generalization and benchmarks

[`docs/REPLAY_AND_METRICS.md`](REPLAY_AND_METRICS.md) documents the
subsystems layered on top of the simulator:

- **Deterministic replay** (format v1, JSON Lines), implemented twice and
  byte-compatible: `python/sandboxai/replay.py` and
  `scripts/replay/*.gd`. Light recordings store actions/rewards only,
  because an episode is reproducible from seed + conditions + actions.
- **Research metrics** in eight diagnostic categories
  (`python/sandboxai/metrics.py`, `scripts/metrics/skill_metrics.gd`).
  Never rewards, and ground truth is returned in a separate dictionary.
- **Conditions / randomization / generalization**
  (`conditions.py`, `randomization.py`, `generalization.py`): deterministic
  per-episode plans derived with blake2b rather than shared RNG state, and
  per-condition reporting that surfaces the win-rate spread.
- **Multi-policy and league** (`policies.py`, `league.py`): independent
  weights are asserted, not assumed; `assert_independent_weights` fails if
  two "different" brains share a parameter tensor or a checkpoint path.
- **Team play foundation**, disabled by default (`teamplay.py`,
  `scripts/team/team_config.gd`).
- **External adapter boundary** (`external_adapter.py`) — see
  [`docs/ROBLOX_ADAPTER.md`](ROBLOX_ADAPTER.md).
- **Map Analyzer 2.0** (`scripts/exploration/exploration_report.gd`):
  perception-only heatmaps, regions, routes, sightlines and metrics, with
  unknown cells reported as unknown.
- **Benchmark suites** (`benchmark_suites.py`): four comparable workloads
  at 1/4/8/16/32/64 environments. Raises rather than estimating when Godot
  is absent.

## Control Center

[`docs/CONTROL_CENTER.md`](CONTROL_CENTER.md) documents the interactive
front-end (`scenes/control_center.tscn` + `scripts/control_center/`). Its
place in the architecture:

```text
ControlCenterUI (CanvasLayer, Controls)      presentation only, 10 Hz refresh
        | snapshots (read)      | method calls (play/pause/reset/select/settings)
ControlCenterSession (Node)                  orchestration, owns stepping
        | step_all() / reset_indices() / set_controller()
SimulationManager -> EnvironmentCore * N     unchanged simulation
```

- The session sets `auto_tick = false` and advances the batch itself, so
  pause/step/speed are exact and never use `Engine.time_scale`.
- HUMAN mode rebinds only the selected environment to the existing
  `HumanController`; there is no second gameplay implementation.
- Telemetry is built by `ControlCenterTelemetry` from
  `DebugOverlay.build_telemetry_dict`, `ObservationInspector` (driven by
  `Observation.FIELD_SPEC`) and `PerceptionModel`, which keeps ground truth
  and observation-derived data in separate branches.
- Nothing is constructed when `DisplayServer.get_name() == "headless"`, so
  training keeps its exact previous cost.
- Tabs: Perception, Observation, Results, **Metrics**, **Replay**,
  Settings. The Metrics tab separates AI-available metrics from a
  red-labelled ground-truth block; the Replay tab scrubs a recorded
  episode and never steps the live simulation.

## Debug GUI, benchmarking and the future Roblox boundary

- [`docs/DEBUG_GUI_AND_BENCHMARKING.md`](DEBUG_GUI_AND_BENCHMARKING.md)
  documents the optional, presentation-only `DebugOverlay` (telemetry +
  pause/reset/enemy-count/curriculum controls, never used by headless
  training) and how to run/interpret `sandboxai benchmark`.
- [`docs/OBSERVATION_ACTION_CONTRACT.md`](OBSERVATION_ACTION_CONTRACT.md) is
  the canonical Observation/Action reference, kept independent of any
  specific game engine.
- [`docs/ROBLOX_ADAPTER.md`](ROBLOX_ADAPTER.md) defines the abstract
  `GameAdapter` interface boundary a future Roblox Player adapter would
  implement. No Roblox integration exists yet.

## Files

```text
scripts/
  core/       Action, Observation, config, curriculum, curriculum
              controller, episode, manager
  world/      Obstacle, ArenaWorld, WorldGenerator, CharacterMotor,
              NavigationGraph + NavigationAgent (A* over walkable cells),
              MapLibrary (authored map catalog)
  perception/ PerceptionSystem (FOV/LOS), SoundBus, EnemyMemory,
              ReactionProfile, AgentPerception, LightingProfile,
              TargetSelector
  exploration/ SpatialMemory (perception-built map knowledge with decay),
              MapAnalyzer (exploration mode + Control Center payload)
  scenario/   ScenarioLibrary (twelve seedable encounters)
  env/        EnvironmentCore, EnvironmentReset (episode setup),
              EnvironmentIntrospection (read-only Control Center views),
              optional EnvironmentView
  agent/      Agent state/view
  enemy/      Enemy state/view, EnemyBrain (tactical behavior)
  weapon/     Weapon state
  reward/     Reward calculation
  rl/         RLAdapter and headless JSON-lines server
  recording/  JSONL recorder and graphical human recording entry point
  self_play/  two-agent match foundation
  input/      human and stub controllers
  debug/      optional presentation-only debug overlay
  control_center/
              Control Center data layer (config, session, event log,
              results, telemetry, observation inspector, perception model,
              spectator camera, 3D perception overlay)
    ui/       Control Center presentation layer (status bar, agent panel,
              perception/observation/results/settings tabs, controls, log,
              HUD, perception map, shared theme)
python/sandboxai/
  config.py       Training/BC/evaluation configuration
  contract.py     Observation/Action contract description + GameAdapter hook
  godot_env.py    subprocess, Gymnasium and SB3 adapters
  ppo.py          PPO, evaluation callbacks, checkpoints
  dataset.py      demonstrations and action validation
  bc.py           PyTorch behavior cloning and compatible warm start
  evaluation.py   frozen evaluation and JSON/CSV summaries
  benchmark.py    throughput measurements and scaling analysis
  telemetry.py    structured metrics/resource snapshots
  self_play.py    policy slots and frozen-opponent lifecycle
  league.py       checkpoint registry, opponent sampling, internal Elo
  conditions.py   seeded condition space + per-condition tracking/report
  auto_curriculum.py  rolling-window promotion/demotion with hysteresis
  cli.py          complete command line
```

## Known limits and next milestone

- Only structured observations are trained; RGB and frame stacking are not
  implemented yet.
- Self-play now has a league layer (`python/sandboxai/league.py`:
  checkpoint registry, policy ids, snapshots, uniform/latest/prioritized
  opponent sampling, deterministic tournaments, optional research-only
  Elo), but it is not yet wired into `sandboxai train` — matches must be
  driven by the caller.
- The arena has seeded geometry, axis-separated collision response,
  gravity, jumping, standable platforms and a deterministic navigation
  graph (`NavigationGraph`, 8-connected walkable grid + A*). Enemies steer
  directly and only fall back to path following once they are demonstrably
  stuck, so open layouts pay nothing for it. Still missing: recoil and
  ammunition.
- Only the 3 highest-priority contacts are individually reported in the
  observation vector. Contacts beyond that are no longer invisible: fields
  66–69 describe them statistically (how many, how many visible, mean and
  minimum distance), so the observation shape is independent of the enemy
  count while still telling the policy it is outnumbered.
- The Control Center cannot run a trained policy in-engine (no neural
  network runtime in Godot); its TRAINING mode is a throughput mode, not a
  trainer. Agent slot 1 still does not exist outside the self-play
  foundation and is reported as unavailable. Field-of-view, line-of-sight,
  sound, memory, obstacles, navigation state and corpses DO exist now and
  are exposed through the seven read-only hooks `PerceptionModel` probes
  for; the Control Center picks them up automatically.
- No Roblox integration exists. `python/sandboxai/contract.py` defines the
  abstract adapter boundary a future implementation would need to satisfy;
  see `docs/ROBLOX_ADAPTER.md` for exactly what is and is not implemented.
- Godot itself must be installed locally; the repository cannot verify live
  Godot behavior on a machine without that executable. This milestone's
  Python-side changes were validated with a scripted fake Godot bridge
  (`python/tests/test_ppo_smoke.py`, `test_godot_env.py`) instead.
- Benchmarking in this milestone was implemented and unit-tested
  (`python/tests/test_benchmark.py`) but not run against a real Godot
  process or GPU — no Godot executable or NVIDIA GPU was available in the
  development environment. Run `sandboxai benchmark` on the target machine
  before relying on its throughput numbers for a training-scale decision.

## Perception and information boundaries

The single most important architectural rule after this milestone: the
policy's observation is produced from `AgentPerception`, not from the
simulation state. `EnvironmentCore._build_observation()` passes the
perception context into `Observation.build()` only when
`CurriculumConfig.perception_enabled()` is true, and the `debug_perception`
flag used by the Control Center deliberately does **not** feed that context
— the debug GUI may look at perception, but it can never change what the
policy sees.

The same rule applies to the opponents: `EnemyBrain` receives the agent's
true position only to run the perception queries, and every decision
downstream reads its own `EnemyMemory` track instead. That is what makes an
enemy genuinely lose you around a corner rather than pretending to.

## Testing workflow

- Python: `python -m pytest -q` from the repository root (`conftest.py`
  puts `python/` on `sys.path`). Optional-dependency tests skip cleanly
  when torch / gymnasium / stable-baselines3 are absent.
- GDScript static analysis and linting need the `gdscript` extra
  (`pip install gdtoolkit`), and run as part of the Python suite via
  `python/tests/test_gdscript_static.py`
  (`sandboxai.gdscript_analysis.analyze()` for parse/resource/symbol/arity
  checks, `lint_all()` for `gdlint`).
- GDScript behaviour tests need a real engine:
  `godot --headless --path . --script res://tests/run_tests.gd`.

The next useful milestone is running the GDScript suite and the benchmark
sweep on a machine with Godot 4.7.2 installed, then training against the
map/lighting condition distribution (`sandboxai.conditions`) and reading
the per-condition generalization report before touching hyperparameters.
