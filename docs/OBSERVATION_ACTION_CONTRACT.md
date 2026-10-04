# Observation / Action contract

This is the single source of truth for what a policy sees and does. It is
implemented twice, once per language, and both must be kept in sync by hand:

- Godot: `scripts/core/observation.gd` (`Observation`) and
  `scripts/core/action.gd` (`Action`).
- Python: `python/sandboxai/contract.py` (`OBSERVATION_SPEC`, `ACTION_SPEC`).
  `python/tests/test_contract.py` checks the Python side is internally
  consistent, and additionally parses the Godot
  sources to fail loudly if `Observation.FIELD_COUNT`, the `to_array()`
  index layout, `Action.MULTI_DISCRETE_NVECS` or the tracked-enemy budget
  drift apart from the Python contract. Semantic drift (a field whose
  *meaning* changes without touching the constants) still cannot be caught
  automatically across the two languages.

The intended data flow (also in `docs/ARCHITECTURE.md`) is:

```
LOCAL GAME STATE (Godot calibration simulator)
        |
Observation Adapter        <- Observation.build() in Godot
        |
Normalized Observation Vector (126 float32 values, all in [-1, 1])
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
   cooldown internals beyond "is my weapon ready", objects the agent has
   never looked at, or global level layout is exposed: the three object slots
   of contract v4 are filled by the same visibility query the agent's own
   perception uses, never by the map's object list.
2. **Normalized ranges.** Every field is designed to sit in `[-1, 1]`
   (booleans are emitted as `0.0`/`1.0`). Distances are divided by the
   arena's maximum diagonal distance; positions by the arena half-extent;
   velocities by the agent's move speed.
3. **Stable dimension.** The vector is always exactly 126 floats, regardless
   of curriculum level, configured enemy count or how much cover is in the
   map: three enemy slots and three object slots are budgets, and the visible
   object *count* field carries the rest. Enemies beyond the 3rd
   nearest alive one still exist and affect the simulation (they can still
   attack/be attacked) but are not individually reported — the policy must
   generalize from the nearest few threats, which is also what is
   practical for a human player to track.
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

## Observation vector (126 floats)

`Observation.to_array()` / `python/sandboxai/contract.py:OBSERVATION_SPEC`.

The table is split into four runs by the notes that belong to particular rows;
every run has its own header, and the index ranges are contiguous from 0 to
125 with nothing missing between them.

### Indices 0–15: the agent and the primary contact

| Index | Field | Meaning | Normalization / range |
| --- | --- | --- | --- |
| 0–2 | `agent_position_norm` (x,y,z) | Agent world position | x,z / arena half-extent (10 m); y / wall height (3 m) |
| 3–5 | `agent_velocity_norm` (x,y,z) | Agent velocity | divided by agent move speed |
| 6–8 | `agent_forward` (x,y,z) | Agent look/forward unit vector | already unit-length, in [-1,1] |
| 9 | `agent_health_norm` | Agent health | health / max_health, in [0,1] |
| 10–12 | `enemy_relative_position_norm` (x,y,z) | Primary (nearest alive) enemy position relative to agent | divided by max arena diagonal distance (~28.3 m) |
| 13 | `enemy_distance_norm` | Distance to primary enemy | divided by max arena diagonal distance, [0,1] |
| 14 | `enemy_health_norm` | Primary enemy health | health / max_health, [0,1] |
| 15 | `weapon_ready` | Can the weapon fire this tick | 0/1; from curriculum level 5 this also reports `0` while reloading or on an empty magazine — see the note below |

### `weapon_ready` and the weapon handling layer

From curriculum level 5 the weapon gains recoil, bloom, fire modes,
magazines and reloads (see `docs/CURRICULUM_AND_COMBAT.md`). **This adds no
observation fields and no action fields** — the vector is still exactly 126
floats and the action space is still `MultiDiscrete([3,3,3,3,2,2])`.

The one semantic change is to index 15. `WeaponState.is_ready()` now also
returns `false` while a reload is in progress or the magazine is empty, so
`weapon_ready` means "pulling the trigger this tick would actually fire"
rather than "the cooldown has expired". This is the policy's only channel
for a dead trigger; leaving it at `1` during a reload would make the
observation lie.

Everything else is felt indirectly and deliberately so:

- **recoil** displaces the agent's own view, so it appears in
  `agent_forward` and in the bearing/elevation fields;
- **bloom** is not observed at all. It is a function of the agent's own
  recent fire, so a recurrent or frame-stacked policy can infer it, and
  exposing it would have meant breaking the 126-float contract;
- **ammunition count** is not observed either, for the same reason.

Below level 5 the handling layer is inert and index 15 keeps its original
meaning exactly, so checkpoints trained before the layer existed are
unaffected.

### Indices 16–17: combat state and bearing

| Index | Field | Meaning | Normalization / range |
| --- | --- | --- | --- |
| 16 | `in_combat` | Primary enemy alive and within weapon range | 0/1 |
| 17 | `enemy_bearing_norm` | Signed horizontal aim offset to the primary enemy | angle / 180°, in [-1,1]; 0 = dead-center, sign matches `look_yaw_axis` (+ = turn right to face it) |

**One sign convention for every bearing.** All eleven bearing fields
(`primary`/`secondary`/`tertiary_enemy_bearing_norm`, `last`/`second_sound_
bearing_norm`, `nearest_obstacle_bearing_norm`, the two `remembered_*` ones
and the three object slots) are *positive to the agent's right* -
the direction a positive `look_yaw_axis` turns - and negative to the left,
so a contact's bearing can be compared with the cover next to it without a
per-field flip. They are all produced by `VectorMath.signed_bearing_*`
(`scripts/core/vector_math.gd`), which is the single implementation of it;
`python/tests/test_contract.py` fails if a second one appears. It is not
cosmetic: the three enemy fields used a yaw difference while the other eight
(the world, sound and memory queries) took the sign of
`forward.cross(direction).y`, which is the opposite in Godot's right-handed
frame, so the vector (and the Stats page) put a contact and the crate beside
it on opposite sides.

The vertical angle follows the same rule: the `*_elevation_norm` fields are
positive when the contact is **above** the eye, negative below. Yaw itself
(the agent's `yaw_deg`, and the direction a spawn or the stub controller
faces) is `atan2(x, -z)` everywhere, so `yaw = 0` looks along `(0, 0, -1)`
and `yaw = 90` along `+x`. All three formulas live in `VectorMath`.
### Indices 18–105: the multi-enemy, world, sound, map and object blocks

| Index | Field | Meaning | Normalization / range |
| --- | --- | --- | --- |
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
| 65 | `local_illumination` | How bright it is **where the agent stands** | [0,1]; the lighting mode itself is never exposed |
| 66 | `overflow_contact_count_norm` | Contacts beyond the 3 individually tracked slots | count / 8, clamped [0,1] |
| 67 | `overflow_visible_count_norm` | How many of those are visible right now | count / 8, clamped [0,1] |
| 68 | `overflow_mean_distance_norm` | Mean distance of the overflow contacts | / max arena diagonal, [0,1] |
| 69 | `overflow_min_distance_norm` | Distance of the nearest overflow contact | / max arena diagonal, [0,1] |
| 70 | `target_priority_norm` | `TargetSelector` score of the primary contact | score / max score, [0,1] |
| 71 | `target_switch_recent` | The primary slot changed within the last 1.5 s | boolean 0/1 |
| 72 | `second_sound_bearing_norm` | Signed horizontal offset to the second loudest event | angle / 180°, [-1,1] |
| 73 | `second_sound_loudness` | Loudness of that second event | [0,1] |
| 74 | `sound_direction_error_norm` | How imprecisely the loudest event can be placed | degrees / 90, [0,1] |
| 75 | `distinct_sound_source_count_norm` | Separate directions noise is coming from | count / 8, clamped [0,1] |
| 76 | `explored_fraction` | Share of the map the agent has actually looked at | [0,1]; 0 when map tracking is off |
| 77 | `current_area_known` | The agent has observed the cell it stands in | boolean 0/1 |
| 78 | `time_since_area_visited_norm` | Time since it last stood here | seconds / 60 s, [0,1]; 1 if never |
| 79 | `remembered_cover_distance_norm` | Distance to the nearest **remembered** cover | / max arena diagonal, [0,1]; 0 if none |
| 80 | `remembered_cover_bearing_norm` | Bearing to that cover | angle / 180°, [-1,1] |
| 81 | `remembered_danger_distance_norm` | Distance to the nearest place the agent was hurt | / max arena diagonal, [0,1]; 0 if none |
| 82 | `remembered_danger_bearing_norm` | Bearing to that place | angle / 180°, [-1,1] |
| 83 | `contact_uncertainty_norm` | Mean staleness of memory-only contacts | 1 − confidence, averaged, [0,1] |
| 84–86 | `object_1_relative_position_norm` (x,y,z) | Closest surface point of the nearest **visible** object, relative to the agent | x,z / max arena diagonal; y / wall height |
| 87 | `object_1_distance_norm` | Distance to that object | / max arena diagonal, [0,1]; 1 if none visible |
| 88 | `object_1_bearing_norm` | Signed horizontal offset to it | angle / 180°, [-1,1]; 0 if none visible |
| 89 | `object_1_kind_norm` | What kind of object it is (cover box, crate, pillar, platform, wall, ...) | kind ordinal / 6, [0,1] |
| 90 | `object_1_visible` | This slot holds a real sighting | 0/1 |
| 91–93 | `object_2_relative_position_norm` (x,y,z) | Same for the second-nearest visible object | as above |
| 94 | `object_2_distance_norm` | Distance to it | [0,1]; 1 if the slot is empty |
| 95 | `object_2_bearing_norm` | Signed horizontal offset to it | [-1,1]; 0 if empty |
| 96 | `object_2_kind_norm` | Its kind | ordinal / 6, [0,1] |
| 97 | `object_2_visible` | This slot holds a real sighting | 0/1 |
| 98–100 | `object_3_relative_position_norm` (x,y,z) | Same for the third-nearest visible object | as above |
| 101 | `object_3_distance_norm` | Distance to it | [0,1]; 1 if the slot is empty |
| 102 | `object_3_bearing_norm` | Signed horizontal offset to it | [-1,1]; 0 if empty |
| 103 | `object_3_kind_norm` | Its kind | ordinal / 6, [0,1] |
| 104 | `object_3_visible` | This slot holds a real sighting | 0/1 |
| 105 | `visible_object_count_norm` | How many objects are visible right now | count / 8, clamped [0,1] |

### Indices 106–125: the contact's box on the agent's screen (contract v5)

Everything above describes a contact as *numbers about a position*: how far,
which way, how hurt. These last twenty describe it the way a player actually
experiences it — as a shape on a screen. Each of the three slots reports where
the contact's box sits (`screen_x` / `screen_y`, its centre in normalised
device coordinates, so `0, 0` is under the crosshair), how much of the screen
it covers (`screen_half_width` / `screen_half_height`, which grow as it comes
closer), how much of the body is not behind cover (`exposure_fraction`), and
how much light it is standing in (`illumination`). `reticle_on_primary`
answers "is my crosshair inside that box" — the one comparison a policy should
not have to spend capacity on — and `primary_contact_clarity` folds
illumination and distance into one number for "can I make this out at all".

| Index | Field | Meaning | Normalization / range |
| --- | --- | --- | --- |
| 106 | `primary_enemy_screen_x` | Where the primary contact's box sits horizontally | screen x, [-1,1]; 0 when not on screen |
| 107 | `primary_enemy_screen_y` | Where it sits vertically | screen y, [-1,1]; 0 when not on screen |
| 108 | `primary_enemy_screen_half_width` | Half the screen width it covers | [0,1]; grows as it gets closer, 0 when not on screen |
| 109 | `primary_enemy_screen_half_height` | Half the screen height it covers | [0,1]; grows as it gets closer, 0 when not on screen |
| 110 | `primary_enemy_exposure_fraction` | How much of its body is unoccluded | [0,1]; 1 = fully exposed, 0.4 = head and shoulders over cover |
| 111 | `primary_enemy_illumination` | How much light it is standing in | [0,1]; same scale as `local_illumination`, 0 when not on screen |
| 112–117 | `secondary_enemy_screen_*`, `secondary_enemy_exposure_fraction`, `secondary_enemy_illumination` | The same six for the secondary contact | as above |
| 118–123 | `tertiary_enemy_screen_*`, `tertiary_enemy_exposure_fraction`, `tertiary_enemy_illumination` | The same six for the tertiary contact | as above |
| 124 | `reticle_on_primary` | The crosshair is inside the primary contact's box | 0/1 |
| 125 | `primary_contact_clarity` | How well the primary contact can be made out at all | illumination × transmittance over the distance, [0,1] |

The geometry comes from one function,
`PerceptionSystem.target_screen_box()` (`scripts/perception/perception_system.gd`):
the agent's eye, its forward vector, the contact's feet, height and radius,
the field of view (`AGENT_FOV_DEG`, 100° horizontal) and the view aspect
(`AGENT_VIEW_ASPECT`, 16:9). `+x` is the agent's right and `+y` is up, so the
signs match the bearing and elevation fields above, and both are clamped to
`[-1, 1]` — the screen. A target at or behind the eye has **no box at all**:
`in_front` is false and all twenty fields are zero, which reads as "it is not
on my screen" and never as a box pinned to the middle of it.

Two of these numbers are easy to misread, so they are worth stating once:

- `exposure_fraction` is **not** "is it alive" and **not** "can I see it". It
  is the share of the sampled body no geometry is covering: `1.0` is a clear
  silhouette, `0.4` is a head and shoulders over a crate, `0.0` is a contact
  the agent can see the position of but cannot hit. Aliveness has its own
  fields, and a contact with no box is all zeros rather than a zero exposure.
- `illumination` is the light **at the contact**, not at the agent
  (`local_illumination`, index 65). A lit doorway with the agent standing in
  shadow is exactly the case the two together describe.

The block is filled from the tick's beliefs — from the same perception the
visibility flags (38/45/48) describe — so a remembered contact keeps the box
it had where it was last seen and nothing only a live sighting could produce:
no exposure, no clarity, and never a reticle. On the levels without a
perception layer (1–5) there are no beliefs to read, but the agent still has
the enemy on its screen there, so the block is measured directly from ground
truth with the same function (`AgentPerception.vision_reading`) and written by
the same writer (`Observation.apply_vision_reading`). Those levels already
report the enemy's exact position; claiming there that the agent cannot tell
how big it looks would be the one dishonest field in the vector.

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

## What changed in contract v5 (the agent's view of a contact)

- Observation grew from 106 to 126 floats. **Indices 0–105 are unchanged in
  index and meaning.** Everything new is appended (106–125).
- The action space is **unchanged**: `MultiDiscrete([3,3,3,3,2,2])`.
- New: every tracked enemy slot reports its silhouette as a box on the
  agent's own screen — where it sits, how much of the screen it covers, how
  much of the body is unoccluded, and how much light it stands in — plus
  `reticle_on_primary` (124) and `primary_contact_clarity` (125). The section
  above says what each means, and what it deliberately does not mean.
- Before v5 the vector could say "an enemy is 12 m away, 20° left" and
  nothing about how big that enemy looks, whether half of it is behind a
  crate, or whether it is standing in a shadow — so the policy had no way to
  tell a clear shot at a well-lit target from the top of a head in the dark.
  All three are things a player can see, so all three are fair game.
- Still **not** reported: anything the agent could not perceive. An enemy it
  has never seen gets no box, not a box at a guessed position; a lost contact
  keeps only the box it last saw. No pixels, no object identities, no ids.
- The one place the block is *measured* rather than perceived is the levels
  without a perception layer (1–5), which report enemy positions as ground
  truth anyway. Exposure therefore becomes real at level 5
  (`obstacles_cover`), where cover exists before the perception gate does.
- A v4 checkpoint has a 106-input observation head and cannot be loaded into
  a v5 policy; nothing has been trained on v4 yet. Replays stamped
  `contract version 4` stay readable for inspection.

## What changed in contract v4 (visible objects)

- Observation grew from 84 to 106 floats. **Indices 0–83 are unchanged in
  index and meaning.** Everything new is appended (84–105).
- New: the arena's *objects* — cover boxes, crates, pillars, low/high cover,
  platforms — are reported as three ranked slots (nearest visible first)
  plus a visible total. Their bearings follow the one convention above. This is the first time the policy can reason about
  the geometry it is standing next to instead of just "an obstacle is that
  way, 4 m off" (`nearest_obstacle_*`, indices 60–61, which stay exactly as
  they are).
- In the simulation the query runs with the perception system's own cone and
  reach (`AgentPerception.fov_deg` / `vision_range`, passed as the
  `fov_deg`/`vision_range` context keys); a context that names neither falls
  back to `AGENT_FOV_DEG` and `VISION_RANGE`.
- A slot is only ever filled by an object the agent **could actually see**
  from where it stands: inside the FOV cone, within `VISION_RANGE`, and not
  hidden behind other geometry. The arena fence (`Obstacle.Kind.BOUNDARY`)
  is never reported. Empty slots keep the neutral encoding (zero position,
  distance 1.0, bearing 0.0, kind 0.0, `visible = 0`), so "no object
  visible" is not spelled as a real reading at the origin.
- The object kind is normalized by its cardinality
  (`SandboxConfig.OBJECT_KIND_COUNT`), exactly like the sound category —
  the raw enum value never reaches the vector.
- The action space is **unchanged**: `MultiDiscrete([3,3,3,3,2,2])`.
- A v3 checkpoint has an 84-input observation head and cannot be loaded
  into a v4 policy; `BC` datasets that store full observations are likewise
  rejected by the version check, while action-only datasets stay loadable.
  Replay files stamped with `contract version 3` are readable for
  inspection but are not treated as v4-compatible.
- Deliberately *not* exposed: anything about objects the agent has never
  looked at, object identities/IDs, the full geometry list, other agents'
  positions, or the arena layout as data. Three slots is what a player can
  hold in mind at once.

## What changed in contract v3 (maps, lighting, hearing, map knowledge)

- Observation grew from 65 to 84 floats. **Indices 0–64 are unchanged in
  index and meaning.** Everything new is appended (65–83).
- The action space is **unchanged**: `MultiDiscrete([3,3,3,3,2,2])`.
- A v2 checkpoint has a 65-input observation head and cannot be loaded into
  a v3 policy. This is a real, intentional input-shape break; the action
  head is unaffected, and behaviour-cloning datasets stay loadable because
  they store actions, not observations.
- New information sources, and what is deliberately *not* exposed:
  - lighting: only `local_illumination` at the agent's own position. The
    mode (`night`, `fog`, ...) is never observed.
  - maps: nothing at all. No map id, no layout id, no geometry dump, no
    spawn list. A map is an environment, not a label.
  - contacts beyond the tracked slots: statistics only (how many, how far),
    never their individual positions.
  - map knowledge: only what `SpatialMemory` earned by looking. With
    exploration tracking off every one of 76–82 is zero, which honestly
    encodes "I know nothing about this place".
- `last_sound_category_norm` now divides by 6 instead of 5, because the
  environmental-noise category was added to `SoundBus`.

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
