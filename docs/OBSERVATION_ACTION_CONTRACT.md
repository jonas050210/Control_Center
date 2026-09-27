# Observation / Action contract

This is the single source of truth for what a policy sees and does. It is
implemented twice, once per language, and both must be kept in sync by hand:

- Godot: `scripts/core/observation.gd` (`Observation`) and
  `scripts/core/action.gd` (`Action`).
- Python: `python/sandboxai/contract.py` (`OBSERVATION_SPEC`, `ACTION_SPEC`),
  used as the documented target for a future external adapter (see
  `docs/ROBLOX_ADAPTER.md`). `python/tests/test_contract.py` checks the
  Python side is internally consistent, and additionally parses the Godot
  sources to fail loudly if `Observation.FIELD_COUNT`, the `to_array()`
  index layout, `Action.MULTI_DISCRETE_NVECS` or the tracked-enemy budget
  drift apart from the Python contract. Semantic drift (a field whose
  *meaning* changes without touching the constants) still cannot be caught
  automatically across the two languages.

The intended data flow (also in `docs/ARCHITECTURE.md`) is:

```
GAME STATE (Godot today; Roblox in the future)
        |
Observation Adapter        <- Observation.build() in Godot
        |
Normalized Observation Vector (33 float32 values, all in [-1, 1])
        |
PPO Policy (MultiDiscrete([3,3,3,3,2]) actions)
        |
Action Vector
        |
Action Adapter              <- Action.from_multidiscrete() in Godot
        |
GAME
```

## Design rules

1. **No privileged information.** Every field is either the agent's own
   state or *relative* enemy information (position, distance, bearing,
   health, aliveness) that a player standing in the agent's position could
   plausibly perceive (e.g. "an enemy is roughly there, at that range, and
   looks hurt"). Nothing about off-screen/undetected enemies, hidden
   cooldown internals beyond "is my weapon ready", or global level layout is
   exposed.
2. **Normalized ranges.** Every field is designed to sit in `[-1, 1]`
   (booleans are emitted as `0.0`/`1.0`). Distances are divided by the
   arena's maximum diagonal distance; positions by the arena half-extent;
   velocities by the agent's move speed.
3. **Stable dimension.** The vector is always exactly 33 floats, regardless
   of curriculum level or configured enemy count. Enemies beyond the 3rd
   nearest alive one still exist and affect the simulation (they can still
   attack/be attacked) but are not individually reported — the policy must
   generalize from the nearest few threats, which is also what is
   practical for a human or a future Roblox client to track.
4. **Additive changes only, unless there is a strong reason otherwise.**
   The original 17-field single-enemy contract (indices 0–16) is unchanged
   in meaning; multi-enemy support was added as new fields (17–32) rather
   than restructuring the existing ones. A checkpoint trained on the old
   17-field contract is **not** input-shape compatible with the new 33-field
   contract — this is an expected, one-time break tied to adding real
   multi-enemy perception (Priority 3 of this milestone), not an accident.

## Observation vector (33 floats)

`Observation.to_array()` / `python/sandboxai/contract.py:OBSERVATION_SPEC`.

| Index | Field | Meaning | Normalization / range |
| --- | --- | --- | --- |
| 0–2 | `agent_position_norm` (x,y,z) | Agent world position | x,z / arena half-extent (10 m); y / wall height (3 m) |
| 3–5 | `agent_velocity_norm` (x,y,z) | Agent velocity | divided by agent move speed |
| 6–8 | `agent_forward` (x,y,z) | Agent look/forward unit vector | already unit-length, in [-1,1] |
| 9 | `agent_health_norm` | Agent health | health / max_health, in [0,1] |
| 10–12 | `enemy_relative_position_norm` (x,y,z) | Primary (nearest alive) enemy position relative to agent | divided by max arena diagonal distance (~28.3 m) |
| 13 | `enemy_distance_norm` | Distance to primary enemy | divided by max arena diagonal distance, [0,1] |
| 14 | `enemy_health_norm` | Primary enemy health | health / max_health, [0,1] |
| 15 | `weapon_ready` | Can the weapon fire this tick | 0/1 |
| 16 | `in_combat` | Primary enemy alive and within weapon range | 0/1 |
| 17 | `enemy_bearing_norm` | Signed horizontal aim offset to the primary enemy | angle / 180°, in [-1,1]; 0 = dead-center, sign matches `look_yaw_axis` (+ = turn right to face it) |
| 18 | `alive_enemy_count_norm` | How many configured enemies are alive right now | alive / total configured enemies, [0,1] |
| 19–21 | `secondary_enemy_relative_position_norm` (x,y,z) | 2nd-nearest alive enemy, relative position | as above; zero vector if absent |
| 22 | `secondary_enemy_distance_norm` | Distance to 2nd-nearest alive enemy | [0,1]; `1.0` (max) if absent |
| 23 | `secondary_enemy_bearing_norm` | Bearing to 2nd-nearest alive enemy | [-1,1]; `0.0` if absent |
| 24 | `secondary_enemy_health_norm` | 2nd-nearest alive enemy health | [0,1]; `0.0` if absent |
| 25 | `secondary_enemy_alive` | Is there a 2nd tracked enemy right now | 0/1 |
| 26–28 | `tertiary_enemy_relative_position_norm` (x,y,z) | 3rd-nearest alive enemy, relative position | as above; zero vector if absent |
| 29 | `tertiary_enemy_distance_norm` | Distance to 3rd-nearest alive enemy | [0,1]; `1.0` (max) if absent |
| 30 | `tertiary_enemy_bearing_norm` | Bearing to 3rd-nearest alive enemy | [-1,1]; `0.0` if absent |
| 31 | `tertiary_enemy_health_norm` | 3rd-nearest alive enemy health | [0,1]; `0.0` if absent |
| 32 | `tertiary_enemy_alive` | Is there a 3rd tracked enemy right now | 0/1 |

If no enemy is alive, the primary slot (indices 10–17) falls back to a fixed
dead-enemy report (`enemy_alive = 0`, `enemy_health_norm = 0`) instead of
being undefined, so a fully cleared episode still returns a stable
observation. Secondary/tertiary slots are simply "not alive" whenever fewer
than 2/3 enemies are alive — this includes curriculum levels 1–3, which
configure only 1 (or optionally 2) enemies, so those slots read as empty
there by design.

## Action vector

`Action.to_multidiscrete()` / `python/sandboxai/contract.py:ACTION_SPEC`.
Gymnasium/SB3 space: `MultiDiscrete([3, 3, 3, 3, 2])`.

| Index | Field | Cardinality | Meaning |
| --- | --- | --- | --- |
| 0 | `move_axis` | 3 | `0` = backward, `1` = idle, `2` = forward |
| 1 | `strafe_axis` | 3 | `0` = left, `1` = idle, `2` = right |
| 2 | `look_yaw_axis` | 3 | `0` = turn left, `1` = idle, `2` = turn right |
| 3 | `look_pitch_axis` | 3 | `0` = look down, `1` = idle, `2` = look up |
| 4 | `shoot` | 2 | `0` = not firing, `1` = firing (subject to weapon cooldown) |

`Action.from_multidiscrete()` shifts each ternary value by `-1` back to the
canonical `{-1, 0, 1}` internal representation. A continuous
`look_delta: Vector2` field exists on the canonical `Action` object purely
for lossless human-demonstration logging; it is not part of the PPO action
space and has no normalized range requirement.

## What changed in this milestone vs. the previous 17-field contract

- Added: primary-enemy bearing (17), alive-enemy-count fraction (18), and
  two more individually-tracked enemies (19–32).
- Unchanged: indices 0–16, action contract, MultiDiscrete shape, `[-1, 1]`
  normalization convention, "no privileged information" rule.
- The Python side (`godot_env.py`) already read the observation dimension
  dynamically from the bridge's `spaces` response before this milestone, so
  no Python code needed to hardcode `33` anywhere except the documentation
  contract in `contract.py`.
