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
{"cmd":"step","actions":[[1,1,1,1,0]]}
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

`Observation.to_array()` always returns 17 float32-compatible values:

| Index | Value |
| --- | --- |
| 0–2 | agent position normalized by arena extent |
| 3–5 | agent velocity normalized by move speed |
| 6–8 | agent forward unit vector |
| 9 | agent health / max health |
| 10–12 | primary enemy relative position / max arena distance |
| 13 | primary enemy distance / max arena distance |
| 14 | primary enemy health / max health |
| 15 | weapon-ready flag |
| 16 | in-combat flag |

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

1. stationary large target, no attacks
2. moving target, no attacks
3. moving target with attacks
4. multiple configured enemies
5. agent-vs-agent hook

`SelfPlayEnvironmentCore` has two `AgentState` slots, independent RNG seeds,
mirrored observations, per-agent rewards and per-agent metrics. Python's
`SelfPlayCoordinator` can load a frozen opponent checkpoint. This is a
foundation for population/self-play training, not a population algorithm.

## Files

```text
scripts/
  core/       Action, Observation, config, curriculum, episode, manager
  env/        EnvironmentCore and optional EnvironmentView
  agent/      Agent state/view
  enemy/      Enemy state/view
  weapon/     Weapon state
  reward/     Reward calculation
  rl/         RLAdapter and headless JSON-lines server
  recording/  JSONL recorder and graphical human recording entry point
  self_play/  two-agent match foundation
  input/      human and stub controllers
python/sandboxai/
  config.py       Training/BC/evaluation configuration
  godot_env.py    subprocess, Gymnasium and SB3 adapters
  ppo.py          PPO, evaluation callbacks, checkpoints
  dataset.py      demonstrations and action validation
  bc.py           PyTorch behavior cloning and compatible warm start
  evaluation.py   frozen evaluation and JSON/CSV summaries
  benchmark.py    throughput measurements
  telemetry.py    structured metrics/resource snapshots
  self_play.py    policy slots and frozen-opponent lifecycle
  cli.py          complete command line
```

## Known limits and next milestone

- Only structured observations are trained; RGB and frame stacking are not
  implemented yet.
- Self-play is a two-slot/match foundation. It does not yet implement a
  population scheduler, league or opponent sampling algorithm.
- The analytic arena has no physics collision response, recoil, ammo or
  complex FPS navigation.
- Godot itself must be installed locally; the repository cannot verify live
  Godot behavior on a machine without that executable.

The next useful milestone is a short verified PPO run on the target machine,
followed by improving the structured multi-enemy observation and adding
curriculum-aware self-play evaluation before introducing RGB input.
