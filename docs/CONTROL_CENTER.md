# SandboxAI Control Center

The Control Center is the interactive front-end of the simulator: one
window from which you can **operate**, **watch**, **play**, **inspect** and
**evaluate** the exact same simulation the RL trainer uses.

It is a presentation layer, not a second implementation. Every number it
shows is read from `EnvironmentCore` / `Observation` / `EpisodeState`, and
every control it offers calls an existing simulation method. Anything the
simulation cannot actually do is shown as **unavailable** with the reason,
never faked.

```
                      +---------------------------+
                      |   ControlCenterUI (GUI)   |   presentation only
                      +-------------+-------------+
                                    | reads snapshots, calls session methods
                      +-------------v-------------+
                      |   ControlCenterSession    |   orchestration
                      +-------------+-------------+
                                    | step_all(), reset_indices(), set_controller()
                      +-------------v-------------+
                      |     SimulationManager     |   unchanged simulation
                      |   -> EnvironmentCore * N  |
                      +---------------------------+
```

---

## Launching

```bash
# From the repository root, with Godot 4.7.2 on PATH:
godot --path . res://scenes/control_center.tscn

# Or through the Python CLI (identical command, plus settings):
sandboxai control-center --mode watch --env-count 4 --enemy-count 1 \
    --curriculum-level 3 --seed 1234
```

Command-line settings are passed after `--` and parsed by
`ControlCenterMain._apply_command_line`:

| Argument | Values | Meaning |
| --- | --- | --- |
| `--mode=` | `training` \| `watch` \| `human` | start mode |
| `--env-count=` | 1..64 | parallel environments in the process |
| `--enemy-count=` | 1..12 | enemies per environment |
| `--curriculum-level=` | 1..11 | curriculum level (see `CurriculumConfig`) |
| `--seed=` | int | base seed; environment *i* uses `seed + i` |
| `--scenario=` | `target_practice`, `duel`, `three_way`, `overwhelmed`, `cover_fight`, `corner_fight`, `sound_only`, `lost_target`, `vertical`, `randomized` | preset bundle |
| `--force-gui=` | `1` \| `0` | build the GUI even on a headless display server (test escape hatch; off by default) |

`scenes/main.tscn` is still the project's main scene, and headless training
is unchanged: launching the project without arguments, or running
`scripts/rl/rl_server.gd --headless`, never constructs a single Control
Center node.

---

## The three modes

All three drive the **same** `EnvironmentCore`, the same `Action` struct and
the same `Observation` vector. They differ only in who produces actions and
how much presentation work is permitted.

| Mode | Actions from | Rendering | Telemetry | Event log |
| --- | --- | --- | --- | --- |
| **TRAINING** | `AIStubController` (or idle) | off | off | off |
| **WATCH** | `AIStubController` (or idle) | selected environment only | on | on |
| **HUMAN** | `HumanController` (selected env only) | selected environment only | on | on |

* TRAINING runs a frame-budgeted batch loop (`TRAINING_FRAME_BUDGET_MS`,
  default 8 ms/frame) with views hidden, telemetry short-circuited and the
  log buffer disabled, so mode switching is a genuine throughput mode and
  not a "GUI with the panels closed". Real PPO weight updates still belong
  to the Python trainer (`sandboxai train`) — the Settings tab prints the
  exact command for the current settings.
* HUMAN mode rebinds **only** the selected environment to the existing
  `HumanController`. The other environments keep their AI controller, which
  is what makes the HUMAN vs AI comparison meaningful.
* Human input is *armed* separately from the mode (the "human input"
  toggle in the controls row, or clicking the viewport). Leaving HUMAN
  mode always disarms it and hands the mouse back, so the GUI never ends
  up unclickable behind a captured cursor.
* Mode switching never rebuilds environments, never touches
  `Engine.time_scale`, and never creates a second gameplay implementation.

---

## Layout

```
+----------------------------------------------------------------------+
| STATUS BAR: mode | run state | env/agent/policy/camera | sps, fps     |
+-------------+------------------------------------------+-------------+
| AGENT PANEL |            3D VIEW + HUD                  | INSPECTOR   |
| position    |  crosshair, health, weapon, target,       | Perception  |
| health      |  timer, reward, kills, accuracy           | Observation |
| target      |                                           | Results     |
| action      |                                           | Settings    |
| perception  |                                           |             |
| mini-map    |                                           |             |
+-------------+------------------------------------------+-------------+
| CONTROLS: play/pause, step, reset, speed, human input arm             |
| EVENT LOG: ALL | COMBAT | PERCEPTION | SYSTEM | REWARD | ERROR        |
+----------------------------------------------------------------------+
```

Keyboard shortcuts (ignored while human input is armed, so gameplay keys
always win):

| Key | Action |
| --- | --- |
| `F1` / `F2` / `F3` | toggle left dock / right dock / bottom dock |
| `Space` | pause / resume |
| `N` | single simulation step |
| `R` | reset the selected environment |
| `Tab` | next inspector tab |
| `W A S D`, mouse, LMB/`Space`, `Esc` | gameplay (HUMAN mode) |
| right-drag, `W A S D`, `Q`/`E`, `Shift` | free camera (non-HUMAN modes) |

---

## Panels

### Status bar
Mode switch, run state, live steps/second and render FPS, environment and
agent selectors, action-source selector, camera selector, and a warning
whenever settings are pending a rebuild.

### Agent panel (live agent view)
Position, velocity, yaw/pitch, health, weapon state and cooldown, current
target with distance/health/in-range flag, the resolved action for this
tick (move, strafe, look, shoot), episode time, step, reward, kills,
deaths, damage dealt/taken, shots and accuracy, plus a top-down mini-map.

### Perception — "what does the AI see?"
Two explicitly separated sections, produced by `PerceptionModel`:

* **REAL WORLD (ground truth, debug only)** — every enemy with true
  distance, bearing, health and AI state.
* **AI PERCEPTION (decoded from the observation vector)** — the three
  tracked enemy slots, weapon readiness, in-combat flag and alive-enemy
  ratio, all de-normalized from the observation only.

Enemies that exist but are **not** in the observation (beyond the
3-enemy contract budget) are listed in red as `HIDDEN FROM AI` with the
reason. Debug visualization can therefore show hidden world state without
ever leaking it into the policy's input: the AI branch is decoded
exclusively from `Observation.to_array()`.

Perception features the simulation does **not** implement — field-of-view
gating, line-of-sight/occlusion, sound events, target memory/last-known
position, cover, navigation, corpses — are listed as **unavailable** with
an explanation. They are detected with `has_method()` probes, so the moment
the simulation grows such a feature the panel starts reporting it.

The optional 3D overlay (`PerceptionOverlay3D`, 20 Hz) draws cyan lines to
tracked enemies, dashed red lines to hidden enemies, a target ring, the
agent's forward vector and the weapon-range circle.

### Observation inspector
All 84 fields of the observation vector with index, name, group and live
value, plus the action rows (multi-discrete value + canonical value) and
the reward components for the current episode.

Field names come from `Observation.FIELD_SPEC` /
`Observation.field_names()` and the action fields from
`RLAdapter.action_space_info()`. The inspector keeps **no** list of its
own; `python/tests/test_contract.py` fails if it ever tries to.

### Results
Current episode metrics, accumulated averages, the HUMAN vs AI comparison
(per-source aggregates plus human-minus-AI deltas), a selectable history of
finished episodes with per-episode detail, and a JSON export to
`user://control_center_results.json`.

Derived metrics that are only instrumented for the selected environment
(reaction time, useless/missed shots, target switches) report `n/a`
elsewhere instead of a fabricated number.

### Settings
* **Live**: curriculum level, scenario preset.
* **Requires reset** (marked, applied by *Apply & reset*): environment
  count, enemy count, seed. A *Randomize seed* button is provided.
* Panel visibility toggles.
* The exact headless training command for the current settings.

### Controls
Play/pause, single step, step ×10, reset environment (deterministic or with
a fresh seed), reset all, speed presets `0.25x … 8x` plus a free slider up
to 16x, and the human-input arm switch.

Speed is implemented as *how many fixed 1/60 s simulation steps run per
rendered frame* (capped at 32 per frame). `Engine.time_scale` is never
touched, so trajectories are identical at any speed or frame rate.

### Event log
Categorised (`COMBAT`, `PERCEPTION`, `SYSTEM`, `REWARD`, `ERROR`),
filterable, colour-coded, follow-scrolling, with the count of buffered and
dropped entries.

Only **discrete** events are logged (a shot, a hit, a kill, damage taken, a
target change, an episode boundary, a settings change) — never per-tick
state. The log is a bounded ring buffer (400 entries) with per-key
throttling (0.15 s) and a global budget (60 events/second); it is fully
disabled in TRAINING mode.

---

## Cameras

Presentation only — the policy never receives camera data.

| Mode | Behaviour |
| --- | --- |
| First person | the agent's own `Camera3D` from `AgentView` (what a human plays through) |
| Third person | chase camera behind/above the agent |
| Free | right-drag to look, `WASD`/`Q`/`E` to fly, `Shift` to sprint |
| Top-down | overhead view of the whole arena |

Free-flight input is disabled while human input is armed so the camera and
the player never fight over `WASD`.

---

## Performance rules

The Control Center must never cost the trainer anything:

1. **Headless is GUI-free.** `ControlCenterMain` checks
   `DisplayServer.get_name() == "headless"` and returns before creating any
   Control, camera rig or overlay. `SimulationManager.create_visuals` stays
   `false`.
2. **Only the selected environment is visual.** Views are created lazily
   (`SimulationManager.ensure_view`) and only the selected one is visible
   and synced.
3. **Telemetry is opt-in per frame.** Panels refresh at 10 Hz (the 3D
   overlay at 20 Hz) and `build_snapshot()` returns `{}` in TRAINING mode.
   Only the visible inspector tab is refreshed, and the snapshot skips the
   observation/perception sections that nothing is displaying.
4. **Logging is bounded.** See the event-log rules above.
5. **Stepping is explicit.** `auto_tick` is off and the session steps the
   batch itself, so pause/step/speed are exact and the interactive path can
   never run more than 32 steps in one frame.

---

## Testing

GDScript (requires Godot 4.7.2):

```bash
godot --headless --path . --script res://tests/run_tests.gd
```

* `tests/test_control_center_config.gd` — modes, clamping, scenarios,
  rebuild classification, round-tripping.
* `tests/test_control_center_event_log.gd` — filters, throttling, budget,
  capacity, disabled early-out.
* `tests/test_control_center_results.gd` — aggregates, comparison, history,
  export.
* `tests/test_observation_inspector.gd` — inspector rows follow the
  contract exactly.
* `tests/test_perception_model.gd` — REAL WORLD vs AI PERCEPTION
  separation, hidden-enemy reporting, unavailable features.
* `tests/test_control_center_session.gd` — mode switching, human action
  pipeline, pause/step/speed, reset determinism, selection, settings
  propagation, snapshot purity, and an equality check proving the Control
  Center does not change simulation results.
* `tests/test_control_center_scene.gd` — headless scene builds simulation
  only; forced GUI builds every panel and refreshing never steps the
  simulation.

Python (no Godot required):

```bash
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

* `python/tests/test_control_center_isolation.py` — static guards that the
  training path never references the Control Center, that the entry point
  is display-guarded, that no Control Center file touches
  `Engine.time_scale`, and that the GUI does not duplicate simulation logic.
* `python/tests/test_contract.py` — adds drift guards between
  `Observation.FIELD_SPEC` and `python/sandboxai/contract.py`, and asserts
  the inspector has no hard-coded field names.
* `python/tests/test_cli.py` — the `control-center` command shape and
  dispatch.

---

## Known limitations

* **No in-engine trained policy.** Godot has no neural-network runtime, so
  the *Trained policy* action source is listed as unavailable. Trained
  checkpoints run in the Python trainer over the JSON-lines bridge
  (`sandboxai evaluate`).
* **Agent slot 1 is unavailable.** `SimulationManager` builds single-agent
  environments; slot 1 exists for the self-play foundation only and is
  reported as unavailable rather than silently ignored.
* **No FOV/LOS/sound/memory model.** The simulation has none, so the
  perception panel reports those features as unavailable instead of
  inventing them.
* **Charting is textual.** Results are rendered as tables/summaries and a
  JSON export; no plotting widget is included (the dependency-free rule).
* **TRAINING mode inside the window is throughput-only.** It does not train
  weights; use `sandboxai train` for that.

## Perception visualization (world/perception milestone)

`EnvironmentCore` now implements all seven optional hooks `PerceptionModel`
probes for (`get_agent_field_of_view`, `has_line_of_sight`,
`get_sound_events`, `get_target_memory`, `get_obstacles`,
`get_navigation_state`, `get_dead_bodies`), so the Control Center picks
them up automatically — no panel needed changing to light them up.

The perception tab now separates five layers:

- **REAL WORLD** — ground truth. Debug only; the policy never sees it.
- **AI PERCEPTION** — decoded from the observation vector, so it is by
  construction exactly what the policy received.
- **AI MEMORY** — remembered contacts with their source (visual/sound),
  age and decaying confidence, plus the reason the current target was
  selected.
- **SOUND** — audible events this tick with category, approximate bearing,
  loudness after occlusion, age, and how many walls the sound passed
  through.
- **HIDDEN FROM AI** — alive enemies absent from the observation, now with
  a distinguished reason: `beyond_tracked_enemy_budget` (the enemy is
  perceivable but the contract only carries three) versus `not_perceived`
  (outside the FOV cone or occluded).

`PerceptionOverlay3D` draws the same data in-world: the FOV cone clipped to
vision range, cover footprints, violet rings at remembered last-known
positions (radius shrinking with confidence), orange rings for sound
events, and grey crosses on corpses. `EnvironmentView` additionally renders
the solid obstacle boxes, colour-coded by kind (olive = low cover you can
shoot over, blue-grey = standable platform, grey = wall/high cover).

### The isolation guarantee

The session sets `EnvironmentCore.debug_perception = true` only for the
selected environment, only while presentation is enabled, and only outside
TRAINING mode. That flag makes the environment *evaluate* perception so it
can be drawn; it deliberately does **not** feed the perception context into
`Observation.build()`. Watching the AI can never change what the AI sees.
`tests/test_perception_isolation.gd` asserts this by running the same
seeded episode with the flag on and off and comparing every observation
field at every step.
