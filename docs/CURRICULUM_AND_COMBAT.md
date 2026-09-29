# Curriculum and multi-enemy combat

This describes the extended curriculum and enemy behavior added in this
milestone. It **extends** `scripts/core/curriculum_config.gd`
(`CurriculumConfig`) rather than replacing it; each level bundles a richer,
still-deterministic set of behaviors.

The ladder is now 10 combat levels plus the self-play hook. Levels 1–4 are
unchanged from the original implementation (same flags, same spawn
distributions, same analytic melee enemies), so a checkpoint or a
reproduction run targeting them behaves exactly as before. Levels 5–10 are
new and progressively enable the world, perception, sound, memory and
vertical systems. `AGENT_VS_AGENT` kept its enum NAME but moved from 5 to
11; anything that referenced it symbolically is unaffected, anything that
hard-coded the number 5 for self-play must be updated.

## Why: avoiding a trivial policy

Previously every curriculum level spawned (at most) one enemy directly
ahead of the agent, optionally moving straight toward it. A policy could
solve that with a shortcut as simple as "there is always something roughly
in front of me; turn slightly and shoot." That does not generalize to a
real FPS-training goal.

The changes below force the policy to actually:

1. **Locate** the enemy — it is not always directly ahead (spawn angle/
   distance vary).
2. **Orient aim** — the new `enemy_bearing_norm` observation field gives a
   direct "turn this much" signal, and enemies are laterally offset.
3. **Track a moving/strafing target** — enemies weave instead of walking a
   straight line once strafing is enabled.
4. **Handle multiple simultaneous threats** — curriculum level 4+ enemies,
   and the observation contract exposes up to 3 tracked enemies at once.

## Curriculum levels

| Level | Name | Enemies | Movement | Attacks | Spawn variety | Strafing |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `stationary_target` | 1 (configurable) | none | no | no (fixed, directly ahead) | no |
| 2 | `moving_target` | 1 (configurable) | chase | no | yes (±70°, 4–13 m) | no |
| 3 | `enemy_attacks` | 1 (configurable, 2+ supported) | chase | yes | yes (±110°, 4–13 m) | yes |
| 4 | `multiple_enemies` | 3+ (bumped up from any configured value < 3) | chase, faster (1.15x speed) | yes, more often (0.85x cooldown) | yes (±150°, 4–13 m) | yes |
| 5 | `obstacles_cover` | 3+ | tactical (`EnemyBrain`) | ranged | scenario-driven (`cover_fight`) | yes |
| 6 | `fov_los` | 3+ | tactical | ranged | scenario-driven (`corner_fight`) | yes |
| 7 | `sound` | 3+ | tactical | ranged | scenario-driven (`sound_only`) | yes |
| 8 | `memory_lost_targets` | 3+ | tactical + search | ranged | scenario-driven (`target_disappears`) | yes |
| 9 | `vertical_combat` | 3+ | tactical + jumping | ranged | scenario-driven (`vertical_encounter`) | yes |
| 10 | `mixed_randomized` | 3+ | tactical + jumping | ranged | a new random scenario every episode | yes |
| 11 | `agent_vs_agent` | n/a — two-policy self-play (`SelfPlayEnvironmentCore`) | — | — | — | — |

### What each new level turns on

| Level | Geometry | Enemy AI | FOV/LOS gating | Sound | Memory | Jumping | Enemy archetype |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–4 | none | analytic chase | off | off | off | off (agent may still jump) | — |
| 5 | `scattered_cover` / `cover_field` | `EnemyBrain` | off | off | off | off | rookie |
| 6 | `corner` | `EnemyBrain` | **on** | off | off | off | regular |
| 7 | `rooms` | `EnemyBrain` | on | **on** | off | off | regular |
| 8 | `corridor` | `EnemyBrain` + search | on | on | **on** | off | veteran |
| 9 | `vertical` | `EnemyBrain` + jumping | on | on | on | **on** | veteran |
| 10 | randomized | all of the above | on | on | on | on | veteran |

The exact predicates are `obstacles_enabled()`, `tactical_enemies_enabled()`,
`ranged_enemies_enabled()`, `perception_enabled()`, `sound_enabled()`,
`memory_enabled()`, `vertical_enabled()`, `weapon_handling_enabled()`,
`hit_zones_enabled()` and `randomized_scenarios_enabled()` on
`CurriculumConfig`. Every one of them returns `false` for levels 1–4, which
is what keeps the cheap levels cheap.

## Weapon handling (level 5+)

Up to level 4 the weapon is a pure cooldown-gated hitscan: pull the trigger,
and if the cooldown has expired a perfectly straight ray leaves the muzzle.
That is the right model for teaching *aim*, and it is deliberately preserved
bit-for-bit so early levels stay cheap and stay comparable with checkpoints
trained before this layer existed.

From level 5 (`OBSTACLES_COVER`) the same `WeaponState` switches on a
handling layer, gated by `weapon_handling_enabled()`:

| System | Effect | Why it is learnable |
| --- | --- | --- |
| **Fire mode** | `auto` (rifle, SMG), `semi` (pistol), `pump` (shotgun) | A semi-auto refuses a trigger that was never released, so "hold M1" stops being a universal policy |
| **Recoil** | Vertical climb + horizontal drift accumulate per shot, then recover | The pattern is a fixed function of the shot index — no RNG — so it can be compensated |
| **Bloom** | The cone widens per shot and collapses when you stop | Creates a real cost for spraying, and a reason to pause |
| **Movement** | Firing plants you; moving and jumping widen the cone | Makes "stop to shoot" a decision rather than a free action |
| **Magazine / reload** | Finite rounds; an empty magazine auto-reloads on the trigger | Turns sustained fire into a resource to manage |
| **Hit zones** | A smaller head sphere inside the body sphere applies a damage multiplier | Rewards precise aim without adding an observation field |

### It adds no observation and no action fields

The observation stays at exactly 84 floats and the action space stays
`MultiDiscrete([3,3,3,3,2,2])`. Handling is felt through channels that
already exist:

- **recoil** moves the agent's own view, so it shows up in `agent_forward`
  and in the bearing/elevation to the target;
- **reloading and an empty magazine** report through `weapon_ready`
  (observation index 15), which `is_ready()` now returns `false` for. That
  is the policy's only channel for "your trigger currently does nothing",
  so it has to be honest about it;
- **bloom** is not observed directly. It is a consequence of the agent's own
  recent actions, which a recurrent or frame-stacked policy can infer, and
  exposing it would have meant breaking the contract.

There is no reload button, which is why an empty magazine reloads itself:
with no way to ask for a reload, a dry weapon would make the episode
unwinnable.

### Determinism

Nothing here uses RNG. Recoil is a closed-form function of the shot index,
and pellet/bloom offsets come from a fixed golden-angle sunflower pattern.
The same seed and the same actions produce the same fight, which is what
replays and regression tests depend on.

### Trigger discipline is not a wasted shot

A trigger pull the weapon refuses — mid-cycle, mid-reload, empty, or a
semi-auto that was never released — raises a `trigger_discipline` event
carrying a small `PENALTY_TRIGGER_DISCIPLINE` (−0.002), *not* the
`−0.1` `penalty_useless_shot` used for shooting at nothing. The two are
different mistakes: one is bad weapon handling, the other is firing at
empty space. Conflating them would both mis-price the error and hide which
one a policy is actually making. With handling off, the old
`useless_shot` / `shot_result = "cooldown"` contract is preserved exactly.

### Weapon roles

Run `sandboxai weapon-table` to print the live TTK matrix, parsed straight
out of the GDScript so it cannot drift from the engine:

```
profile  mode     rpm  mag  range         role          2m            5m           8m          11m          14m
rifle    auto     120   30  15.0m         long   1.50/1.50s   1.50/1.50s   1.50/1.50s   1.50/1.50s   2.00/2.00s
shotgun  pump      83    6   9.5m  point_blank   0.00/0.00s   0.72/0.72s   1.44/1.44s           --           --
pistol   semi     250   15  12.0m          mid   0.96/0.96s   0.96/0.96s   1.20/1.20s   1.68/1.68s           --
smg      auto     667   30  11.0m          mid   0.63/0.63s   0.63/0.63s   0.81/0.81s   1.35/2.61s           --
```

Cells are *ideal* / *actual* TTK against 100 HP: ideal assumes every round
lands on centre mass, actual walks the trigger shot by shot with bloom,
magazine and reload included. They are identical for the rifle, pistol and
shotgun because those fire slowly enough that the cone fully collapses
between rounds — that is exactly what makes them the precision options.

The SMG is the one profile whose cooldown (0.09 s) is shorter than the
recoil recovery delay (0.12 s), so it can out-run its own bloom. At the edge
of its range that costs it nearly a second (1.35 s → 2.61 s), and a
disciplined 6-round burst recovers a chunk of it (2.15 s). That trade-off is
the intended skill expression, and it is asserted by
`python/tests/test_weapon_balance.py` rather than left to hope.

Only the rifle reaches past 12 m, which is what keeps range selection a real
decision on a map whose longest sight line is about 28 m.

## Automatic progression

`scripts/core/curriculum_controller.gd` (`CurriculumController`) advances the
level from measured performance instead of a fixed step schedule: over a
rolling window of episodes it promotes when the success rate reaches
`promote_threshold` and demotes when it falls to `demote_threshold`, with a
cooldown so a level is never skipped on one lucky episode. A step-count
schedule promotes a policy that is still failing and holds back one that
already generalizes; a performance gate does neither. It is opt-in
(`enabled = false` by default) and leaves the operator's level untouched
while still collecting statistics.

## Scenarios

`scripts/scenario/scenario_library.gd` defines twelve seedable encounters —
`open_arena`, `single_target`, `multiple_targets`, `corner_fight`,
`cover_fight`, `corridor_fight`, `ambush`, `target_disappears`,
`sound_only`, `multi_direction`, `vertical_encounter`, `randomized_arena`.
A scenario is pure data (layout id, enemy count, spawn rule, required
capabilities); `ScenarioLibrary.resolve(id, seed)` is a pure function, so
the same `(id, seed)` pair always produces the same geometry AND the same
spawn points. The occlusion-based spawn rules (`out_of_sight`,
`behind_cover`) sample until the spawn is genuinely not visible from the
agent's eye, so an "ambush" scenario really starts without visual contact
instead of merely hoping for it.

All of the per-level knobs live in `CurriculumConfig` methods
(`spawn_variety_enabled()`, `strafing_enabled()`, `spawn_angle_spread_deg()`,
`spawn_distance_range()`, `effective_enemy_count()`,
`enemy_speed_scale()`, `enemy_cooldown_scale()`) so a new level or a tuning
change is a small, local edit, not a parallel environment implementation.

**Backward-compatibility note:** the previously validated successful
training run used curriculum level 3 with `enemy_count=1`. Level 3 still
supports `enemy_count=1` (it does not force a minimum enemy count) — what
changed is that level 3 now also varies spawn position and enables
strafing. A checkpoint trained on the old level-3 behavior is not expected
to perform identically against the new, harder version of level 3; this is
the intended difficulty increase for this milestone, not a regression.

## Multi-enemy spawn variety

When `spawn_variety_enabled()` is true (level ≥ 2), each enemy's spawn
position is picked as a random point at distance
`[ENEMY_SPAWN_MIN_DISTANCE, ENEMY_SPAWN_MAX_DISTANCE]` (4–13 m) and angle
`[-spread, +spread]` degrees around the agent's spawn-facing direction,
using the environment's seeded `RandomNumberGenerator` — so it is fully
deterministic given the same seed (see
`EnvironmentCore._random_spawn_position()` and
`tests/test_environment_core.gd`'s `test_spawn_variety_is_deterministic_*`
test). Level 1 keeps the exact original fixed layout (enemies lined up
directly ahead with only ±0.5 m jitter) so the originally-validated combat
path is unchanged at that level.

One geometric caveat: the agent's spawn point is offset from the arena
center, so a wide spawn angle can place the *sampled* point outside the
walls. Those spawns are clamped back inside the arena, which shortens the
effective distance below the sampled value (worst case, directly behind the
agent, the arena is only ~3.5 m deep). The 4–13 m range is therefore an
upper envelope for off-axis spawns, not a strict guarantee; every clamped
spawn is still a valid, deterministic, in-arena position.

## Enemy movement patterns

`EnemyState.update_ai(..., enable_strafe)` (default `false`, preserving the
old straight-line chase everywhere it's not explicitly enabled) blends the
direct chase direction with a perpendicular, sinusoidally oscillating
lateral component:

- Far from the agent (still in the CHASE state, i.e. beyond attack range):
  mostly straight-line approach (as before).
- As the enemy closes in on attack range (still while chasing, i.e. distance
  is above `attack_range` but within `attack_range * 4`): up to ~85% of the
  movement weight shifts to the lateral component, producing a weaving,
  circle-strafing-like approach instead of a predictable straight line.
- Once distance drops to `attack_range` or below, the enemy switches to the
  ATTACK state exactly as before this milestone: it stops moving entirely
  and attacks on cooldown from a fixed position. Strafing only affects how
  an enemy *approaches*, not what it does once already in melee/attack
  range — this preserves the original, validated attack-range/damage/
  cooldown behavior byte-for-byte.
- The oscillation (`sin(time_alive * angular_speed + strafe_phase) *
  strafe_direction`) is a pure function of elapsed simulation time — no
  further RNG calls happen during movement — so replaying the same seed
  reproduces byte-identical enemy trajectories.
- `strafe_direction` (±1) and `strafe_phase` (radians) are assigned once per
  enemy per episode from the environment's seeded RNG when
  `strafing_enabled()` is true.

This is deliberately still fully analytic (no physics, no navmesh, no
elevation/verticality) — see "Limitations" below.

## Configuring enemy count and curriculum

Via the Python CLI/training config:

```bash
sandboxai train --curriculum-level 4 --enemy-count 3 --env-count 24
```

`enemy_count` is a *requested* count; `CurriculumConfig.effective_enemy_count()`
can raise it (never lowers it) — e.g. level 4 always uses at least 3 enemies
even if `enemy_count=1` was requested.

Via the headless bridge directly:

```json
{"cmd": "set_curriculum", "level": 4}
```

Via the debug GUI (see `docs/DEBUG_GUI_AND_BENCHMARKING.md`): the
"Enemies -/+" and "Level -/+" buttons.

## Limitations (honest, not implemented)

> **Stale-claim correction.** The first two bullets below described the
> milestone this document was written for. Both have since been implemented:
> curriculum level 9 (`VERTICAL_COMBAT`, `CurriculumConfig.vertical_enabled()`)
> adds jumping, standable boxes and elevation-dependent sight through
> `CharacterMotor`/`ArenaWorld`, and `scripts/world/navigation_graph.gd` +
> `navigation_agent.gd` provide a deterministic grid/A* path around interior
> obstacles. Read those sources, not this paragraph, for current behaviour.

- ~~**No verticality/elevation.**~~ Implemented: `ArenaWorld` carries
  standable geometry and the agent has a jump action; elevation changes what
  is visible. The simulation is still analytic (no engine physics), which is
  the deliberate determinism boundary.
- ~~**No navmesh/obstacle avoidance.**~~ Implemented as a custom
  deterministic navigation graph rather than Godot's NavigationServer, so
  paths are reproducible from a seed.
- **The GDScript half of this milestone was never executed *at the time it
  was written*.** (CI has run the Godot suite on Windows and Linux since.) The Godot
  engine could not be obtained in the environment the handling layer was
  written in, so `tests/run_tests.gd` — including the new
  `tests/test_weapon_handling*.gd` — has not been run. It was validated by
  the real GDScript grammar, gdlint, and the project's static analyzer
  (symbols, call arity, and now calls on typed locals), plus a Python
  mirror of the weapon maths in `python/sandboxai/weapons.py`. That catches
  compile-class errors, not behavioural ones. **Run the Godot suite before
  trusting these numbers in training.**
- **Bloom is not observable.** The widening cone is a consequence of the
  agent's own recent fire, so a memoryless policy cannot read it directly.
  Exposing it would have required a new observation field, which the
  contract forbids. A recurrent or frame-stacked policy can infer it.
- **Enemies do not use the handling layer.** `EnemyBrain` still fires
  through `try_fire()` with `handling_enabled = false`, so enemy weapons
  have no recoil, bloom or magazine. Enemy difficulty is tuned through the
  reaction/archetype profiles instead; arming enemy handling would change
  every existing difficulty curve at once.
- **Only 3 enemies are individually observed** even though more can exist
  and fight simultaneously (curriculum level 4 can be configured with more
  than 3). This is an explicit, documented observation-contract choice (see
  `docs/OBSERVATION_ACTION_CONTRACT.md`), not an oversight.
