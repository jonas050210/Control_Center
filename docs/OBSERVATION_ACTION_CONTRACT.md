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
Normalized Observation Vector (65 float32 values, all in [-1, 1])
        |
PPO Policy (MultiDiscrete([3,3,3,3,2,2]) actions)
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
3. **Stable dimension.** The vector is always exactly 65 floats, regardless
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

## Perception gating (contract v2)

From curriculum level 6 (`fov_los`) upward the enemy blocks stop being
ground truth. `AgentPerception` decides, per enemy and per tick, whether the
agent can actually see it (inside the FOV cone, unoccluded, within vision
range, and visible for longer than the reaction delay). The observation then
reports:

- a **live sighting** — real position, `*_visible = 1`, `age = 0`,
  `confidence = 1`; or
- a **decaying memory** — the last known position, `*_visible = 0`, a
  growing `info_age_norm` and a shrinking `confidence`; or
- **nothing at all** — the slot is zeroed and `*_alive = 0`, because the
  agent has no information about that enemy.

This is the mechanism that enforces the "no privileged information" rule
now that the arena has geometry. The agent is never handed the coordinates
of an enemy it cannot perceive; when it loses a target around a corner the
observation degrades exactly the way a human's knowledge does.

Below level 6 the gating is disabled and indices 10–32 keep their original
ground-truth meaning byte-for-byte, so curriculum levels 1–4 reproduce the
pre-world dynamics exactly.

## Observation vector (65 floats)

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
| 33 | `agent_on_ground` | Standing on the floor or a box (jump is only possible then) | 0/1 |
| 34 | `agent_vertical_velocity_norm` | Vertical velocity | / jump velocity (6.0 m/s), clamped to [-1,1] |
| 35 | `agent_in_cover` | No living enemy currently has line of sight to the agent | 0/1 |
| 36 | `agent_forward_clearance_norm` | Distance to the first sight-blocking surface straight ahead | / vision range (28 m), [0,1] |
| 37 | `primary_enemy_visible` | Indices 10–17 are a live sighting (`1`) or a memory (`0`) | 0/1 |
| 38 | `primary_enemy_in_fov` | Primary contact lies inside the FOV cone | 0/1 |
| 39 | `primary_enemy_los_clear` | Geometry does not occlude the primary contact | 0/1 |
| 40 | `primary_enemy_elevation_norm` | Signed vertical angle from the eye to the primary contact | angle / 90°, [-1,1] |
| 41 | `primary_enemy_info_age_norm` | Age of the primary contact information | seconds / 12 s, [0,1] |
| 42 | `primary_enemy_confidence` | Confidence in the primary contact position | exponentially decayed, [0,1] |
| 43 | `primary_enemy_source_visual` | The information came from vision | 0/1 |
| 44 | `primary_enemy_source_sound` | The information came from hearing | 0/1 |
| 45 | `secondary_enemy_visible` | Secondary contact is a live sighting | 0/1 |
| 46 | `secondary_enemy_info_age_norm` | Age of the secondary contact information | seconds / 12 s, [0,1] |
| 47 | `secondary_enemy_elevation_norm` | Signed vertical angle to the secondary contact | angle / 90°, [-1,1] |
| 48 | `tertiary_enemy_visible` | Tertiary contact is a live sighting | 0/1 |
| 49 | `tertiary_enemy_info_age_norm` | Age of the tertiary contact information | seconds / 12 s, [0,1] |
| 50 | `tertiary_enemy_elevation_norm` | Signed vertical angle to the tertiary contact | angle / 90°, [-1,1] |
| 51–53 | `last_sound_direction` (x,y,z) | Perceived direction of the loudest audible event | unit vector **with up to ±14° of directional error**; zero if silent |
| 54 | `last_sound_distance_norm` | Perceived distance of that event | / max arena diagonal, [0,1] |
| 55 | `last_sound_bearing_norm` | Signed horizontal offset to that event | angle / 180°, [-1,1] |
| 56 | `last_sound_age_norm` | Age of that event | seconds / 2.0 s lifetime, [0,1] |
| 57 | `last_sound_loudness` | Loudness after distance and per-wall occlusion attenuation | [0,1] |
| 58 | `last_sound_category_norm` | Category ordinal: footstep/jump/land/shot/impact/death/environment | ordinal / 6, [0,1] |
| 59 | `audible_event_count_norm` | How many events are audible this tick | count / 8, clamped [0,1] |
| 60 | `nearest_obstacle_distance_norm` | Distance to the nearest piece of cover/geometry | / max arena diagonal, [0,1] |
| 61 | `nearest_obstacle_bearing_norm` | Signed horizontal offset to that obstacle | angle / 180°, [-1,1] |
| 62 | `visible_enemy_count_norm` | Enemies currently visible | count / 8, clamped [0,1] |
| 63 | `remembered_enemy_count_norm` | Contacts remembered but not visible | count / 8, clamped [0,1] |
| 64 | `corpse_count_norm` | Corpses in the arena (environmental information only) | count / 8, clamped [0,1] |

If no enemy is alive, the primary slot (indices 10–17) falls back to a fixed
dead-enemy report (`enemy_alive = 0`, `enemy_health_norm = 0`) instead of
being undefined, so a fully cleared episode still returns a stable
observation. Secondary/tertiary slots are simply "not alive" whenever fewer
than 2/3 enemies are alive — this includes curriculum levels 1–3, which
configure only 1 (or optionally 2) enemies, so those slots read as empty
there by design.

## Action vector

`Action.to_multidiscrete()` / `python/sandboxai/contract.py:ACTION_SPEC`.
Gymnasium/SB3 space: `MultiDiscrete([3, 3, 3, 3, 2, 2])`.

| Index | Field | Cardinality | Meaning |
| --- | --- | --- | --- |
| 0 | `move_axis` | 3 | `0` = backward, `1` = idle, `2` = forward |
| 1 | `strafe_axis` | 3 | `0` = left, `1` = idle, `2` = right |
| 2 | `look_yaw_axis` | 3 | `0` = turn left, `1` = idle, `2` = turn right |
| 3 | `look_pitch_axis` | 3 | `0` = look down, `1` = idle, `2` = look up |
| 4 | `shoot` | 2 | `0` = not firing, `1` = firing (subject to weapon cooldown) |
| 5 | `jump` | 2 | `0` = grounded, `1` = jump (only takes effect while `agent_on_ground`) |

`Action.from_multidiscrete()` shifts each ternary value by `-1` back to the
canonical `{-1, 0, 1}` internal representation. A continuous
`look_delta: Vector2` field exists on the canonical `Action` object purely
for lossless human-demonstration logging; it is not part of the PPO action
space and has no normalized range requirement.

### Backwards compatibility of the action space

`Action.from_multidiscrete()` still accepts a 5-component v1 action and
simply never jumps, and `RLAdapter` still decodes the 7-value v1 canonical
log array. Recorded human demonstrations from before this milestone remain
loadable: `python/sandboxai/dataset.py:action_to_multidiscrete()` pads them
with `jump = 0`. A v1 **checkpoint**, however, has a 5-head action net and a
33-input observation head, so it cannot be loaded into a v2 policy — that
break is real and intentional.

## What changed in contract v2 (world + perception milestone)

- Observation grew from 33 to 65 floats. **Indices 0–32 are unchanged in
  index and meaning.** Everything new is appended (33–64).
- The enemy blocks (10–32) become belief-based instead of ground-truth from
  curriculum level 6 upward; indices 37/45/48 tell the policy which it is.
- Action grew from `MultiDiscrete([3,3,3,3,2])` to
  `MultiDiscrete([3,3,3,3,2,2])` with `jump` appended at index 5.
- The canonical log array grew from 7 to 8 values: `jump` was inserted at
  index 5, pushing `look_delta.x/y` to indices 6/7.
- `Action.Discrete` gained `JUMP = 10`, so `DISCRETE_COUNT` is now 11.

## What changed in the previous milestone vs. the 17-field contract

- Added: primary-enemy bearing (17), alive-enemy-count fraction (18), and
  two more individually-tracked enemies (19–32).
- Unchanged: indices 0–16, action contract, MultiDiscrete shape, `[-1, 1]`
  normalization convention, "no privileged information" rule.
- The Python side (`godot_env.py`) already read the observation dimension
  dynamically from the bridge's `spaces` response before this milestone, so
  no Python code needed to hardcode `33` anywhere except the documentation
  contract in `contract.py`.
