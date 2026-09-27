# SandboxAI Architecture (Milestone 1)

## Goals of this milestone

Prove SandboxAI can run a small, deterministic, RL-compatible FPS combat
environment in Godot 4.7.2, controllable by both a human and (eventually)
an RL policy, with multiple independent parallel environments and a
headless/fast-simulation path — without building a real game.

## Layering: simulation vs. presentation

The most important design decision in this codebase is the hard split
between **simulation/RL logic** and **rendering/input**:

| Layer | Extends | Lives in | Depends on scene tree? |
|---|---|---|---|
| `AgentState`, `EnemyState`, `WeaponState`, `Action`, `Observation`, `EpisodeState`, `EnvironmentCore`, `RewardSystem` | `RefCounted` | `scripts/{core,agent,enemy,weapon,reward}` | **No** |
| `AgentView`, `EnemyView`, `EnvironmentView` | `Node3D` | `scripts/{agent,enemy,env}` | Yes (purely to draw) |
| `SimulationManager` | `Node` | `scripts/core` | Optional (`create_visuals` flag) |
| `HumanController`, `AIStubController` | `Node` (via `ControllerBase`) | `scripts/input` | Yes (needs input callbacks) |

`EnvironmentCore` — the RL environment — has **zero** dependency on
`Node`/`Viewport`/rendering. It can be instantiated and stepped with plain
`EnvironmentCore.new()` calls, which is what makes the automated tests fast
and simple, and what makes `SimulationManager.create_visuals = false`
possible: hundreds of environments can be ticked with no Node3D, mesh,
camera or physics-space overhead at all. `EnvironmentView` (and
`AgentView`/`EnemyView`) only *read* state to update a transform/visibility
— deleting them would not change simulation behavior.

## RL interface

`EnvironmentCore` (scripts/env/environment_core.gd) exposes exactly the
interface requested by the milestone:

```gdscript
reset(seed_value: int = -1) -> Observation
step(action: Action, dt: float = SandboxConfig.SIMULATION_DT) -> Dictionary # {observation, reward, done, info}
get_observations() -> Observation
get_rewards() -> float
is_done() -> bool
```

`SimulationManager` provides the batched version across N environments
(`reset_all`, `step_all`, `get_observations`, `get_rewards`, `is_done_all`,
plus `run_headless_steps(n)` for training-speed stepping with no rendering
involved). `RLAdapter` (scripts/rl/rl_adapter.gd) is a thin façade over
`SimulationManager` with the exact Gym-style names
(`reset`/`step`/`get_observations`/`get_rewards`/`is_done`) plus
`action_space_info()`/`observation_space_info()` so a future Python-side
wrapper has a single, small surface to bind to. It intentionally does not
pull in any RL library — that is deliberately deferred to the next
milestone.

## Action space

`Action` (scripts/core/action.gd) is a small structured object with:

- `move_axis` (-1/0/1), `strafe_axis` (-1/0/1)
- `look_yaw_axis` (-1/0/1), `look_pitch_axis` (-1/0/1)
- `shoot` (bool)
- `look_delta: Vector2` — **reserved, unused by default**, for a future
  continuous mouse-aiming mode

`Action.from_discrete(int)` maps the 10 requested single-choice actions
(idle, move forward/backward, strafe left/right, look left/right/up/down,
shoot) onto this struct — this is what `RLAdapter.step()` accepts when
given raw ints. Because `look_delta` already exists on the struct and
`AgentState.apply_action()` already adds it straight into yaw/pitch,
switching to continuous mouse-style aiming later is a matter of *populating
that field* (as `HumanController` already does from real mouse motion) —
no reshaping of the interface is required.

## Observation space

`Observation.build(agent, enemies, arena_half_extent)`
(scripts/core/observation.gd) produces a 17-float, mostly-normalized
vector:

| Index | Field | Normalization |
|---|---|---|
| 0-2 | agent position (x,y,z) | ÷ arena half-extent (÷ wall height for y) |
| 3-5 | agent velocity | ÷ max move speed |
| 6-8 | agent forward vector | unit vector, already in [-1,1] |
| 9 | agent health | ÷ max health |
| 10-12 | enemy relative position | ÷ arena max diagonal distance |
| 13 | enemy distance | ÷ arena max diagonal distance |
| 14 | enemy health | ÷ max health |
| 15 | weapon ready | 0/1 |
| 16 | in combat | 0/1 (enemy alive & within weapon range) |

`Observation.to_dict()` exposes the same data (plus `enemy_relative_direction`
and `enemy_alive`) for debugging/logging. `SandboxConfig.ObservationMode`
(`STRUCTURED` / `HUMAN_INPUT` / `RGB`) exists as an explicit extension
point: milestone 1 only implements `STRUCTURED`. A future RGB/screen mode
would add a *new* build method (e.g. `Observation.build_rgb_from_viewport`)
and a new field on the result, not replace this one — existing structured
observations keep working while a vision model is developed in parallel.

## Reward system

All numbers live in `SandboxConfig` and are combined by the stateless
`RewardSystem.compute(events: Dictionary)`:

| Event | Effect |
|---|---|
| `hit` | `+REWARD_HIT` (1.0) |
| `kill` | `+REWARD_KILL` (10.0) |
| `damage_taken` (HP) | `HP * PENALTY_DAMAGE_TAKEN_PER_HP` (-0.05/HP) |
| `died` | `+PENALTY_DEATH` (-10.0) |
| `useless_shot` (fired on cooldown / no target at all) | `+PENALTY_USELESS_SHOT` (-0.1) |
| `positioning_delta` (meters closed toward the enemy while not already close) | clamped to ±`REWARD_POSITIONING_MAX` (0.05) |
| alive & not died | `+REWARD_SURVIVE_TICK` (0.01) |

Values are deliberately small relative to hit/kill/death so the agent
cannot farm reward by only "surviving" or "positioning" — combat outcomes
dominate.

## Combat

- Weapon: fixed damage (25), fixed range (15m), fixed cooldown (0.5s), no
  recoil/spread/ammo. `WeaponState.try_fire()` gates on cooldown;
  `WeaponState.ray_hits_sphere()` is a deterministic ray-vs-sphere test
  against each alive enemy's "chest" point — this is the "simple
  raycast/hit test" requested, implemented as plain vector math rather
  than a `PhysicsDirectSpaceState3D` query so it works identically whether
  or not the environment has any visual/physics representation at all
  (needed for the headless/many-environments path).
- Enemy AI (`EnemyState.update_ai`): idle (out of detection range) → chase
  (moves toward the agent) → attack (deals fixed damage on a cooldown once
  within attack range). No pathfinding, no perception cones — deliberately
  minimal per the milestone scope.
- Episode ends when the agent dies, all enemies are eliminated, or a step
  timeout (`SandboxConfig.MAX_EPISODE_STEPS`) is reached; each condition is
  independently toggleable via `SandboxConfig.END_EPISODE_ON_*` constants.

## Simulation architecture / multiple environments

`SimulationManager` owns an `Array[EnvironmentCore]`. Each entry is a fully
independent object graph — its own `AgentState`, `EnemyState`s,
`EpisodeState`, and seeded `RandomNumberGenerator` — so stepping or
resetting one environment can never affect another (see
`tests/test_simulation_manager.gd`). When `create_visuals` is true, one
`EnvironmentView` per environment is spawned and placed in a simple grid
purely for on-screen separation; this placement is cosmetic only; each
`EnvironmentCore`'s own coordinate space is always centered at the origin,
so simulation math never has to know about the visual layout.

Two ways to advance simulation time:

1. **Engine-driven** (`SimulationManager._physics_process`): used for
   human play / demos, ticks once per physics frame at
   `SandboxConfig.SIMULATION_DT` (60 Hz).
2. **Headless/batch** (`SimulationManager.run_headless_steps(n)`): steps
   every environment `n` times back-to-back with no relation to real time
   or rendering — the path a future trainer would use for throughput, and
   the path the automated tests use so they run in milliseconds.

Determinism: `EnvironmentCore.reset(seed_value)` seeds a local
`RandomNumberGenerator`; the same seed always reproduces the same initial
enemy spawn layout and therefore the same first observation
(`tests/test_environment_core.gd::test_deterministic_reset_same_seed_gives_identical_observation`).
`SimulationManager.base_seed` seeds environment *i* with `base_seed + i`,
so an entire multi-environment run is reproducible from one integer.

## Human play vs. AI

`HumanController` and `AIStubController` both implement `ControllerBase`
(`get_action(env) -> Action`) — the exact same method `SimulationManager`
calls for every environment every tick, whether a human or an AI is behind
it. `HumanController` reads raw keyboard/mouse state (not the Input Map, to
keep `project.godot` minimal) and already produces continuous
`look_delta` from real mouse motion, so the human is, from day one,
generating the richer/continuous version of the action a discrete AI
policy would only approximate. This matters for the stated next milestone
("record human demonstrations for imitation learning"): the action object
being produced by human play is already exactly the shape any Action
consumer (RL trainer, imitation learner, self-play opponent) will read —
adding a demonstration recorder is a matter of logging `(observation,
action, reward)` tuples around the existing `HumanController.get_action`
call, not redesigning it.

## Known limitations / deliberate scope cuts

- Movement/AI/hit-testing are analytic (no `PhysicsServer3D` stepping) —
  intentional for determinism and headless speed; walls are simple AABB
  clamps rather than real collision response.
- One "primary" enemy drives the structured observation (nearest alive);
  `EnvironmentCore` already supports multiple enemies per environment
  (`enemy_count_per_environment`) for reward/combat purposes, but the
  observation vector would need a fixed-size multi-enemy slot layout
  (or padding/masking) to expose more than one at a time to a policy —
  left for the next milestone once the observation size budget is decided
  by whatever trainer is chosen.
- No recoil, ammo, inventory, multiplayer, or complex enemy AI, per the
  milestone's explicit scope.
- No RGB/vision observation implementation yet — only the extension point
  (`SandboxConfig.ObservationMode`) exists.

## Extension points for later milestones

- **Actual PPO/RL training**: implement a transport (socket or
  GDExtension) that calls `RLAdapter.reset`/`step` from Python; nothing in
  `EnvironmentCore` needs to change.
- **Behavior cloning / human demonstration recording**: wrap
  `HumanController.get_action()` to also append
  `(get_observations(), action, reward)` to a buffer/file.
- **Self-play**: instantiate two `AgentState`/`AgentView` pairs inside one
  `EnvironmentCore` and swap which one the reward/termination logic treats
  as "the agent" per side — the state/action/reward classes do not assume a
  single hardcoded agent identity beyond the current single-agent field, so
  this is a moderate, contained change to `EnvironmentCore`.
- **RGB observations**: add a `SubViewport` + `Camera3D` capture path
  behind `SandboxConfig.ObservationMode.RGB`; this is explicitly designed
  to sit *alongside* `Observation.build()`, not replace it.
- **More complex FPS combat / additional environments (e.g. melee /
  Half-Sword-style)**: `EnvironmentCore`, `AgentState`, `EnemyState` and
  `WeaponState` are already separate, small, replaceable classes —a melee
  variant would swap `WeaponState` for a different combat resolver behind
  the same `EnvironmentCore.step()` contract.
