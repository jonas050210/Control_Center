# SandboxAI Control Center

The Control Center is the interactive front-end of the simulator: one
window from which you can **operate**, **watch**, **play**, **inspect** and
**evaluate** the exact same simulation the RL trainer uses.

It is a presentation layer, not a second implementation. Local simulation
values come from `EnvironmentCore` / `Observation` / `EpisodeState`; managed
training values come from the existing Python PPO/BC telemetry. Every control
calls an existing simulation method or sends a cooperative command to the
Python backend. Anything the backend cannot actually do is shown as
**unavailable** with the reason, never faked.

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

## Dashboard pages

The window is organised as a headless-training dashboard with a persistent
navigation bar. The active page is saved in the preferences
(`active_page`) and restored on the next start.

| Page | Purpose |
| --- | --- |
| **HOME** | PC status strip (GPU / CPU / RAM, see below) + one compact card per launched agent (state, steps, episodes, reward, kills/deaths, accuracy, throughput, Pause/Stop/Details) including recently finished or failed agents. |
| **AGENTS** | Agent management: every agent with lifecycle controls (Pause/Resume/Stop, clear finished), plus "new agent" launch that reuses the current training configuration. |
| **HEADLESS** | The main monitoring page: one panel per managed agent with the full live status published by the Python trainer (steps, episodes, reward, kills, deaths, shots, hits, accuracy, damage, survival, wins/losses, steps/s, ETA when the backend reports one, checkpoint) and a live, bounded log fed from the trainer's `events.jsonl` (timestamps, severity, auto-follow, terminal events preserved). |
| **TRAINING** | The existing PPO / Behavior-Cloning configuration editor, unchanged, hosted as a launch screen. START TRAINING starts the run and jumps to HEADLESS. |
| **SIMULATION** | The classic visual layout described below (3D view, HUD, inspector docks). Unchanged behaviour. |
| **ANALYTICS** | Lightweight reward trends per agent, sampled only when the backend publishes a new progress marker. Real metrics only. |
| **HISTORY** | Previous managed runs read from the persisted `status.json` files: run id, algorithm, final state, duration, steps, episodes, reward, checkpoint, error. |
| **SETTINGS** | The existing settings panel (simulation parameters, scenarios, layout). |

### PC status (HOME)

`ControlCenterSystemMonitor` samples local hardware on a slow timer in a
worker thread, completely outside the simulation/training loop:

* **GPU** — name, utilisation %, VRAM used/total, temperature via
  `nvidia-smi` when present (any NVIDIA model; nothing is hard-coded).
* **CPU** — utilisation from `/proc/stat` (Linux) or WMIC (Windows).
* **RAM** — used/total/percent from `OS.get_memory_info()`.

Every metric that cannot be measured on the current machine is shown as
`N/A` — values are never estimated or invented. There is deliberately no
network or disk monitoring.

### Multi-agent training

`TrainingAgentManager` keeps a registry of `TrainingRunController`s —
Agent 1 wraps the classic single-run controller, additional agents get
their own managed run directory (`user://control_center_runs/<run_id>/`)
and process. Pause/Resume/Stop go through the same cooperative
`command.json` protocol as before; nothing is force-killed beyond the
controller's existing final fallback. All displayed training metrics come
from the backend's `status.json`/`events.jsonl`; the UI only renders them.

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

## Managed training workspace

The **Training** inspector tab configures and launches the repository's real
Python backends. PPO maps to `sandboxai train`/`resume`; Behavior Cloning maps
to `sandboxai bc-train`. Steps/epochs, environment count, curriculum, seed,
device, checkpoint/resume selection and the backend's supported optimizer
parameters are passed through unchanged. The exact command is visible before
launch.

Start, pause, resume and stop use a small file-based control boundary in
`python/sandboxai/run_control.py`. Python acknowledges state only at safe
callback/batch boundaries and atomically publishes status. Stop is graceful:
the normal final checkpoint path runs before the process reports `Finished`.
The GUI never suspends a process behind the trainer's back or infers a state
from button clicks.

* **Visual Mode** keeps the selected in-process simulation and observation
  inspector visible as a deterministic preview using the same requested
  environment settings. PPO itself still uses its separate headless Godot
  bridge, so the preview is not claimed to be the learner's exact live arena.
* **Headless Mode** disables local rendering/telemetry and gives the central
  tile to backend progress, RL/BC metrics, measured CPU/VRAM values and the
  backend event log. GPU utilization remains `n/a` because PyTorch does not
  expose it here; allocator VRAM is shown when CUDA supplies it.
* **Self-Play** is selectable and described, but Start is disabled. The
  repository currently has a real two-slot match bridge and frozen-checkpoint
  league for evaluation, not a self-play optimizer. It is not silently mapped
  to single-agent PPO.

The dashboard docks use split handles, visibility controls and persisted
ordering/preferences (`user://control_center.cfg`). At narrow widths the agent
preview collapses first; the training configuration or headless progress tile
keeps the available space.

---

## The three local simulation modes

All three drive the **same** `EnvironmentCore`, the same `Action` struct and
the same `Observation` vector. They differ only in who produces actions and
how much presentation work is permitted.

| Mode | Actions from | Rendering | Telemetry | Event log |
| --- | --- | --- | --- | --- |
| **TRAINING** | `AIStubController` (or idle) | off | off | off |
| **WATCH** | `AIStubController` (or idle) | selected environment only | on | on |
| **HUMAN** | `HumanController` (selected env only) | selected environment only | on | on |

* TRAINING runs the **local preview session** as a frame-budgeted batch loop
  (`TRAINING_FRAME_BUDGET_MS`, default 8 ms/frame) with views hidden,
  telemetry short-circuited and the local log buffer disabled. Managed PPO
  and BC weight updates remain in the Python trainer launched from the
  Training tab; this local mode is not relabelled as gradient training.
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

## Layout (SIMULATION page)

The inspector dock on the right has six tabs: **Run**, **Perception**,
**Observation**, **Results**, **Metrics**, **Replay**. The settings and
training-configuration editors live on their own pages (SETTINGS /
TRAINING).

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
| `W A S D`, mouse, LMB/`Space`, `Ctrl`, `Esc` | gameplay (HUMAN mode) |
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
* Legacy local-preview settings and panel/tile visibility. The Training tab
  owns the full PPO/BC configuration and exact managed command.

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
* `tests/test_system_monitor.gd` — PC telemetry: payload shape, `N/A` for
  unavailable metrics, `nvidia-smi` / CPU-load / `/proc/stat` / RAM
  parsing, no fabricated values.
* `tests/test_training_agent_manager.gd` — multi-agent registry: identity,
  lifecycle states, pause/resume/stop command propagation, bounded event
  ingestion, terminal agents staying visible.
* `tests/test_training_run_history.gd` — persisted run history parsing
  (only published fields, non-final durations for live runs).
* `tests/test_control_center_dashboard.gd` — dashboard shell: page
  navigation with persistence, agent cards, headless monitors with a
  bounded live log, PC-status rendering (`N/A`), history rows, launch
  navigation.

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
