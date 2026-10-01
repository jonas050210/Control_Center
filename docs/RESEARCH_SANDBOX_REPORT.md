# Perception-driven RL sandbox — implementation report

Branch: `arena/01a0e665-sandboxai`, branched from `f5e28f9` on `main`.
Eight commits, 46 files changed, ~7.0k lines added.

This report separates **what was implemented**, **what was actually
executed**, and **what could not be run here**. Nothing in the "measured"
sections is estimated.

---

## 1. Environment constraint that shapes everything below

**There is no Godot executable in this workspace, and one cannot be
obtained.** Every download host for the engine is blocked or absent
(`godotengine.org`, `*.githubusercontent.com`, `downloads.tuxfamily.org`,
`cdn.jsdelivr.net` all fail at the TLS layer); `apt` has no package lists;
the npm mirror of Godot 4.7.2 ships macOS/Windows/WASM binaries only.

Consequences, stated plainly:

* The **GDScript test suite was written but not executed.** 81 new
  behaviour tests across 7 files are in the repository and must be run with
  `godot --headless --path . --script res://tests/run_tests.gd` on a machine
  that has the engine.
* The **benchmark sweep (Phase 20) was not run.** It needs a real Godot
  process; no throughput, CPU or RAM figures are reported, and none are
  invented.
* Confidence in the GDScript half therefore rests on two static gates that
  *were* executed: `sandboxai.gdscript_analysis.analyze()` (gdparse grammar
  parse, resource-path resolution, symbol resolution, call arity) and
  `gdlint` over every `.gd` file. Both are clean.

---

## 2. Implemented systems

### Phase 0 — baseline repair
* `python/tests/test_gdscript_static.py` used a non-existent `Finding.rule`
  attribute (the field is `kind`), so the static-analysis gate could never
  have reported a finding.
* `test_godot_env.py`, `test_cli.py` and `test_simulation_integration.py`
  hard-failed when torch/gymnasium/SB3 were missing instead of skipping.
* `test_cli.py` subprocess tests only passed with `PYTHONPATH=python`
  exported; they now build their own child environment.
* Added a `gdlint` pass and a pytest bootstrap (`conftest.py`).

### Phase 1 — deterministic navigation
* `scripts/world/navigation_graph.gd` — 8-connected walkable grid baked
  from the geometry for a given body radius/height, A* with a reusable
  heap, deterministic tie-breaking (lower node index), connected-component
  labels for O(1) reachability.
* `scripts/world/navigation_agent.gd` — per-character path follower with
  stuck detection (`NAV_STUCK_SPEED`/`NAV_STUCK_TIME`), re-path interval,
  waypoint tolerance, corner recovery, and statuses
  `direct/path/recovery/unreachable`. Path failures are a **metric, never a
  reward**.
* `EnemyBrain` consults it only once direct steering demonstrably fails, so
  an open arena pays two float comparisons per tick. The graph is baked
  lazily, once per episode, for the enemy footprint and shared by all
  enemies.

### Phase 2 — maps and scenarios
* `scripts/world/map_library.gd` — 14 authored maps, each binding layout
  variants, default lighting (plus an optional seeded lighting pool), arena
  size, ambient-noise emitters, tags and prose.
* Three new procedural layouts in `world_generator.gd`: `multi_room`
  (seeded doorways), `ambush` (lane plus blind alcoves), `sound_maze`
  (staggered stubs that break sight lines but carry sound).
* `ScenarioLibrary.resolve_on_world()` so a scenario places characters
  **into** an existing map instead of generating its own geometry.
* Information boundary: `MapLibrary.metadata()` is human-only (label, tags,
  description) and is reachable only through the Control Center hook. No
  map id, layout id, geometry, spawn list or hidden cover reaches the
  observation vector.

### Phase 3 — lighting and visibility
* `scripts/perception/lighting_profile.gd` — six modes (normal, low light,
  night, fog, high contrast, mixed) expressed as *multipliers on the
  existing perception pipeline*: acquisition range, detection delay and
  loss grace. Patchy light uses smoothstepped bilinear value noise over
  integer-hashed patch corners, so it is seeded and replayable with no RNG
  draws at query time.
* Applied symmetrically to `AgentPerception` and `EnemyBrain`: an enemy in
  the dark loses you too.
* The policy never sees the mode — only `local_illumination` at its own
  position (contract field 65).

### Phases 4–5 — Map Analyzer and spatial memory
* `scripts/exploration/spatial_memory.gd` — coarse grid of remembered
  *places*, stored as parallel `Packed*Array` columns (no per-cell
  allocation in the loop). Per cell: observed/visited timestamps, visit
  count, observed brightness, cover score, openness, danger, and a
  confidence that decays with a 45 s half-life to a floor rather than to
  zero (a place you saw stays known but becomes unreliable).
* A cell only becomes known through `observe_cells()`, which runs the same
  FOV + LOS + lighting-limited range test the observation uses. There is no
  bulk import of the world.
* `scripts/exploration/map_analyzer.gd` — drives the memory on a fixed
  0.2 s cadence off the simulation clock, scores exploration progress, and
  exposes a frontier direction as a scripted **baseline** only.
* `EnvironmentCore.set_exploration_mode()` adds a dedicated Map Analyzer
  episode: coverage is the objective, the map may be empty of enemies, and
  the episode ends with `done_reason == "map_explored"` at 85 % coverage.
  Exploration reward is paid **only** in that mode, so combat rewards are
  bit-identical to before.

### Phase 6 — advanced sound
* New `ENVIRONMENT` sound category plus per-map ambient emitters on a fixed
  round-robin schedule: a genuine distractor that means nothing tactically.
* Per-event **bearing uncertainty** that grows with occluding walls and
  with distance relative to the audible radius, reported to the listener as
  `direction_error_deg` and folded into a `confidence`.
* **Masking**: every event is attenuated by the loudest simultaneous other
  event; what falls below the floor is never reported. A footstep under
  covering fire is genuinely unhearable.
* `SoundBus.summarize()` — count, distinct source directions, second
  loudest, mean confidence, masked count. Feeds observation fields 72–75
  and the Control Center.

### Phase 7 — tactical behaviour
* Two new states: `INVESTIGATE` (approach a noise it has never seen,
  cautiously, at 0.7 speed) and `RETREAT` (critically hurt with no cover —
  break away from the *believed* threat position).
* Exposure clock: line of sight is symmetric, so an enemy that can see the
  agent can infer it is standing in the open; after
  `ENEMY_MAX_EXPOSURE_TIME` it repositions to cover. This is what turns a
  static firefight into movement, and it uses no privileged information.
* Still explicitly **baselines**, not personalities: no archetype hard-codes
  a play style beyond reaction latency and aim error.

### Phases 8–9 — multi-enemy scaling and target selection
* `scripts/perception/target_selector.gd` — target choice promoted from a
  private sort into a named-factor abstraction: visibility, recent damage,
  memory confidence, proximity, threat (it can see me), wounded,
  accessibility (navigation reachability), continuity (hysteresis so two
  equal threats do not make the primary slot oscillate). Weights live in
  `SandboxConfig`; `select()` returns the priority, whether a switch
  happened, and a per-factor breakdown plus a human-readable reason that is
  shown **only** in the Control Center.
* `AgentPerception.summarize_contacts()` — contacts beyond the three
  individually tracked slots are described statistically. Eight enemies and
  one enemy produce the same observation shape, and the policy still learns
  that it is outnumbered.

### Observation contract v3 (65 → 84 floats)
Strictly additive; indices 0–64 keep their exact v2 meaning.

| Indices | Group | Content |
| --- | --- | --- |
| 65 | conditions | `local_illumination` at the agent |
| 66–69 | contacts | overflow count / visible / mean distance / min distance |
| 70–71 | target | selection priority, recent switch |
| 72–75 | sound | second bearing, second loudness, bearing error, distinct sources |
| 76–78 | exploration | explored fraction, current area known, time since last here |
| 79–82 | exploration | remembered cover distance/bearing, remembered danger distance/bearing |
| 83 | memory | mean staleness of memory-only contacts |

Updated together, as the contract requires: `Observation.to_array()`,
`Observation.FIELD_SPEC`, `python/sandboxai/contract.py` (+
`OBSERVATION_GROUPS`, which still partitions every field),
`docs/OBSERVATION_ACTION_CONTRACT.md` and `docs/ARCHITECTURE.md`.
`python/tests/test_contract.py` statically diffs the Godot and Python layouts
and passes.

**Break:** a v2 checkpoint has a 65-input head and cannot be loaded into a
v3 policy. The action space is unchanged, and behaviour-cloning datasets
remain loadable because they store actions.

### Phases 10–13, 17 — training infrastructure (Python, executable here)
* `python/sandboxai/league.py` — `CheckpointRegistry` (stable policy ids,
  JSON on disk, snapshots that are genuinely separate weight files so a
  frozen opponent cannot change under an evaluation) and `League`
  (uniform / latest / prioritized opponent sampling, seeded and rewindable;
  win-loss records; head-to-head; deterministic ordered tournaments;
  optional internal Elo documented as research-only).
* `python/sandboxai/conditions.py` — `ConditionSpace` over maps, lighting,
  scenarios, enemy counts and levels, with seeded sampling for training and
  exact grid enumeration for evaluation; `ConditionTracker` with rolling
  per-condition windows; `generalization_report()` that surfaces the
  win-rate **spread** and the worst conditions, because a pooled mean hides
  exactly the failure you are looking for.
* `python/sandboxai/auto_curriculum.py` — rolling-window promotion and
  demotion with separate thresholds (hysteresis), minimum episodes per
  level, cooldown after a change, and a window reset on change so one good
  streak cannot stair-step through several levels. Fully configurable and
  deterministic.

### Structural changes
`EnvironmentCore` was split along a real seam rather than allowed to grow:
`EnvironmentReset` (runs once per episode, may allocate) and
`EnvironmentIntrospection` (read-only Control Center projections). The
simulation loop file stayed under the 1000-line lint limit throughout, and
the separation makes it structurally obvious that the debug UI cannot write
to the simulation.

---

## 3. Tests — executed vs. written

### Executed (this workspace)

| Command | Result |
| --- | --- |
| `python -m pytest -q` (repo root) | **126 passed** in ~28 s |
| `python -m pytest python/tests/test_training_infrastructure.py -q` | **41 passed** |
| `python -m pytest python/tests/test_contract.py -q` | **19 passed** |
| `sandboxai.gdscript_analysis.analyze()` over all `.gd` | **0 findings** |
| `gdlint` over all `.gd` | **Success: no problems found** |

Baseline before this work was 85 passing Python tests; it is now 126. The
41 new ones cover the condition space, the league and the curriculum.

### Written but NOT executed (no engine available)

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_navigation.gd` | 12 | graph generation, determinism, path existence, unreachable goals, corner recovery, obstacle changes |
| `tests/test_maps_and_lighting.gd` | 12 | map resolution, seed determinism, metadata boundary, lighting effects on perception |
| `tests/test_spatial_memory.gd` | 12 | unseen geometry stays unknown, decay, stale replacement, reset clears, replay determinism, darkness reveals less |
| `tests/test_map_analyzer.gd` | 11 | coverage growth, reward only for new ground, completion, seed replay, off-by-default, Control Center payload |
| `tests/test_sound_model.gd` | 14 | attenuation, occlusion, bearing error bound, delay, expiry, masking, ambience, identity never leaked |
| `tests/test_target_and_scaling.gd` | 10 | factor ordering, hysteresis, unreachable penalty, overflow statistics, shape at 1/2/3/5/8 enemies |
| `tests/test_tactical_behavior.gd` | 10 | idle without information, investigate a noise, retreat when critical, exposure reposition, determinism |
| **total new** | **81** | (repository total: 318 GDScript tests across 43 files) |

Run them with:

```bash
godot --headless --path . --script res://tests/run_tests.gd
```

---

## 4. Benchmarks

**Not measured.** Phase 20 requires a live Godot process; none exists here.
`python/sandboxai/benchmark.py` and its unit tests are in place, and the
sweep (1/4/8/16/32/64 envs across levels 1–4, 5–10, Map Analyzer and
perception-heavy combat) should be run on the target Windows machine before
any performance decision. No numbers are reported, because any number
produced here would be fabricated.

Two cost notes that are structural rather than measured, and are worth
verifying in that sweep:

* the navigation bake is lazy and happens at most once per episode;
* the Map Analyzer visibility sweep is the most expensive addition
  (O(cells in range) LOS tests every 0.2 s) and is **off by default**.

---

## 5. Limitations and what needs local Windows/Godot testing

1. Run the 318-test GDScript suite. Expect to fix small issues the static
   analyser cannot see (it checks grammar, resource paths, symbols and call
   arity — not runtime semantics).
2. Run the benchmark sweep and record real steps/s, CPU and RAM.
3. Re-train: the v3 contract changes the input width, so existing
   checkpoints must be retrained, not fine-tuned.
4. The league is not yet wired into `sandboxai train`; matches currently
   have to be driven by the caller.
5. Phases not implemented in this session: 14 (deterministic replay with a
   Control Center timeline), 15 (Control Center 2.0 with 15 named views —
   the data layer gained exploration/conditions/sound-summary hooks and
   panels, but the full view set is not built), 16 (research metric suite),
   18 (teamplay abstractions), 19 (Roblox adapter boundary beyond the
   existing documented contract), 21 (the remaining per-subsystem test
   sweep).
6. The Control Center still cannot run a trained policy in-engine; there is
   no neural-network runtime in Godot.

---

## 6. Recommended next milestone

Run the GDScript suite and the benchmark sweep on the Godot machine, fix
whatever the live run surfaces, then train one policy across the
`ConditionSpace` distribution and read `generalization_report()` before
changing any hyperparameter. The win-rate *spread* across conditions is the
number that says whether the agent learned to fight or learned a map — and
it is now measurable.

After that, Phase 14 (deterministic replay) is the highest-value remaining
item: every subsystem added here is already seed-replayable, so the replay
system is mostly recording and a timeline UI, and it makes every later
debugging session cheaper.
