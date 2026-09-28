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

The canonical Godot `Action` has four ternary fields and one binary trigger:

- `move_axis`, `strafe_axis`, `look_yaw_axis`, `look_pitch_axis`: `-1, 0, 1`
- `shoot`: boolean
- `look_delta`: optional continuous mouse delta, retained for human logs

The ML space is `MultiDiscrete([3, 3, 3, 3, 2])`. Each ternary field is
shifted by one (`-1 -> 0`, `0 -> 1`, `1 -> 2`). `Action.from_discrete()`
continues to support Agent 1's ten single-choice actions for compatibility.
Human, stub AI, external PPO and demonstrations all pass through `Action`.

## Observation contract

`Observation.to_array()` always returns 33 float32-compatible values: the
original 17-field single-enemy contract (agent state + primary/nearest-alive
enemy + weapon-ready/in-combat flags), plus 16 additive fields (the primary
enemy's aim bearing, an alive-enemy-count fraction, and two more
individually-tracked enemies). See
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
parallel hardcoded environments:

1. stationary target, fixed spawn directly ahead, no movement/attacks
2. one moving enemy, varied spawn distance/angle
3. moving + attacking enemy, spawn variety, strafing while engaging
4. 3+ enemies, spawn variety, strafing, faster/more aggressive
5. agent-vs-agent hook

See [`docs/CURRICULUM_AND_COMBAT.md`](CURRICULUM_AND_COMBAT.md) for the full
per-level table, the multi-enemy spawn-variety algorithm, and the
deterministic strafing/movement-pattern design.

`SelfPlayEnvironmentCore` has two `AgentState` slots, independent RNG seeds,
mirrored observations, per-agent rewards and per-agent metrics. Python's
`SelfPlayCoordinator` can load a frozen opponent checkpoint. This is a
foundation for population/self-play training, not a population algorithm.

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
  world/      Obstacle, ArenaWorld, WorldGenerator, CharacterMotor
  perception/ PerceptionSystem (FOV/LOS), SoundBus, EnemyMemory,
              ReactionProfile, AgentPerception
  scenario/   ScenarioLibrary (twelve seedable encounters)
  env/        EnvironmentCore and optional EnvironmentView
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
  cli.py          complete command line
```

## Known limits and next milestone

- Only structured observations are trained; RGB and frame stacking are not
  implemented yet.
- Self-play is a two-slot/match foundation. It does not yet implement a
  population scheduler, league or opponent sampling algorithm.
- The arena now has seeded geometry, axis-separated collision response,
  gravity, jumping and standable platforms, but still no recoil, no ammo,
  no navmesh and no path planner. Enemies steer directly toward their
  chosen destination and slide along walls; in a maze-like layout they can
  get stuck against a corner rather than routing around it. Adding a real
  navigation graph is the obvious next structural step.
- Only the 3 nearest alive enemies are individually reported in the
  observation vector even if more exist and fight simultaneously (see
  `docs/OBSERVATION_ACTION_CONTRACT.md`).
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

The next useful milestone is training against the new perception curriculum
(levels 5–8) on the target machine, validating the benchmark sweep to pick a
practical environment count, and then starting the real external Roblox
Player adapter implementation against the contract in
`docs/ROBLOX_ADAPTER.md`.
