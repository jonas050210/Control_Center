# Changelog

Human-readable summary of engineering-facing changes. Not a marketing
changelog: every entry here is something a maintainer would want to know
before touching CI, dependencies, or the desktop Control Center. See
`PROJECT.md` for the living technical spec and roadmap; this file only
records what changed and why.

## Unreleased

### Control Center
- **The Stats page.** A new page (the eighth) shows what the trained policy
  actually receives, decoded from a real recording instead of guessed: the
  three tracked contacts (relative position, distance, bearing, elevation,
  health, visibility, in-FOV/LOS, information age, confidence, and whether
  the belief came from vision or hearing), the world objects, cover and
  memory rows around the agent, the hearing summary, the recorded action per
  component, the raw observation vector grouped by contract section, and the
  TTK Testing evidence manifest next to it. Without a recording the page
  still shows the complete contract (`contract.OBSERVATION_SPEC`) with `n/a`
  values rather than zeroes that would read like data, and a light replay is
  labelled as light. The headline line carries the object block too
  (`objects 2 (crate)`), so "is there cover next to me" is answered without
  opening a table. The adapter side (`list_replays`, `replay_stats`) reads
  a replay's header and tick count in one pass, caches both by
  `(mtime_ns, size)` and the decoded episode by the same stamp, and every
  call runs on the background pool, so nothing on the page blocks the Tk
  thread.
- **A replay rescan reads each recording at most once.** `list_replays()`
  read every candidate twice (once for the header line, once to count ticks)
  and kept only the 64 most recent headers, so a folder bigger than that was
  re-read from disk on every Stats poll. Header and tick count now come from
  a single pass and are cached together behind the file's `(mtime_ns, size)`
  stamp; only the header line is JSON-decoded and ticks are recognized by
  prefix, so a corrupt line cannot break the listing. Measured with
  `tools/replay_scan_probe.py` (1200 synthetic recordings, 200 ticks each,
  144.7 MB): a warm rescan went from a 38.8 ms median to 18-21 ms over
  repeated runs (the rest is the directory scan itself), the cold scan
  stayed at ~41-43 ms - it is one pass either way, and what disappeared is
  the re-reading. Two adapter tests pin the behaviour: a
  warm scan does not open an unchanged file, and a rewritten recording is
  noticed.
- **The Stats page keeps the list fresh, and never shows a replay that is
  gone.** Three behaviours the page claimed but did not have: the 20-tick
  rescan timer only ran while *no* replay was selected, so a recording
  written during the session could never appear without pressing Rescan; a
  rescan that found nothing left the last decoded replay on screen as if it
  were still backed by a file; and a failed decode left the previous
  contract warning up. The timer now keeps asking, an unchanged listing is
  recognised by a `(path, size, mtime)` signature and left alone (so the
  operator's selection and scroll position survive it), and both "nothing
  found" and "decode failed" reset the page through one `_clear_decode()`
  path. The evidence manifest is loaded once instead of on every tick. The
  smoke harness pins all three by counting poll submissions and by
  breaking each guard on purpose, which is how the checks were verified.
- **A poll tick stopped reading whole training logs.** The worst offender
  was not a bug in the GUI but in `run_inspection`: every poll of the Runs,
  Evaluations and Dashboard pages counted the lines of every log a run owns,
  and counting lines means reading the whole file. A finished run that kept a
  30 MB `training.jsonl` was read end to end twice a second, forever, and
  `tail_jsonl` decoded and parsed every line of its trailing megabyte to
  return the last five. Line counts are now extended by only the appended
  bytes (remembering `(bytes, lines)` and correcting for a half-written last
  line; a file that shrank is recounted from zero, the same rule
  `IncrementalJsonlTailer` uses), and the tail window is walked from its end,
  stopping as soon as the requested rows are in hand. Measured with the new
  `tools/control_center_poll_probe.py` on a 60-run corpus whose three newest
  runs carry a 64 MB log, warm call before -> after: `dashboard_snapshot`
  68.1 -> 4.3 ms, `discover_checkpoints` 144.8 -> 16.0 ms, `list_runs`
  140.3 -> 14.5 ms. Six new tests pin the counter (append, mid-line
  boundary, shrink, unchanged file, bounded cache) and the tail (a malformed
  final line still cannot fail it, and it no longer parses the whole window).
- **A poll result that has not changed no longer rebuilds the table.** The
  Runs and Evaluations tables were cleared and re-inserted on every 600 ms
  tick. That is wasted work, and in real Tk it also drops the operator's
  row selection - the highlight disappears while the exact same rows come
  back. Both inventories now compare a signature of the values they render
  (`control_center_viewmodel.table_signature`) and leave an unchanged table
  alone; a changed value still rebuilds it. The Evaluations table also
  re-registered its Treeview tag colour once per tick, growing the theme
  bookkeeping list that is replayed entry by entry on every theme switch;
  `Page.tag_style` is idempotent now. The headless smoke harness checks all
  three by counting the fake Treeview's delete/insert calls and the length of
  the tag list, and fails when either guard is removed on purpose.
- **Polling the visible page is now a checked guarantee.** The ticker
  already refreshed only the current page, but nothing stopped a future
  change from walking every page's `refresh()` on each 600 ms tick - eight
  pages submitting background reads (run directories, benchmark history, the
  Stats replay list) for a window nobody is looking at. The smoke harness now
  counts `refresh()` calls per page and fails if a hidden page is polled;
  breaking `_tick` on purpose makes it fail, which is how the check was
  verified.
- **Older recordings stay readable, and are labelled.** Growing the
  observation contract must not make recorded evidence disappear, so
  `SandboxAIAdapter.replay_stats()` reads a replay with
  `strict_contract=False` and reports `contract_match` plus
  `recorded_observation_dim`; the Stats page shows a warning line
  ("recorded under an older contract (84 floats, current 106) - readable,
  but not comparable with a current policy's input") instead of either
  failing or pretending the numbers belong to the current vector. Two adapter
  tests cover the older and the current case, and the replayed values are
  read exactly as recorded - never padded.
- **Every page renders its cards again.** `Page.board()` built the layout
  board and never attached it: Dashboard, Training and Benchmarks rendered
  their heading and then an empty page. The board is packed now, the static
  page-contract test asserts a board always gets a geometry manager (the
  real-Tk suite cannot run on a machine without Tkinter), and the headless
  smoke sweep checks the same thing on every page.
- **Interface rework.** The window is now built from switchable design
  choices instead of one hard-coded look: five themes (Corz, Midnight
  Cyan, Neon Lime, Graphite Mono, Light), three shell layouts (rail,
  topbar, command board), three density presets, four motion levels,
  adjustable corner radius and optional glow/grid effects. All of it
  applies live, persists in `.sandboxai/ui/preferences.json` and never
  drops a font below the 11 px floor.
- **Every widget follows the live theme again.** Eleven construction sites
  (the benchmark phase stepper and live cards, the throughput and telemetry
  charts, the System stat row, several tooltips) built their widget without
  a `ThemeBus`, so those widgets subscribed to the module's default Corz
  palette and never repainted on a theme or accent change - invisible
  without a display. They now pass the shell's bus, a page stamps its bus
  as `_cc_bus` so helper widgets can find it by walking up their parents,
  and both the smoke harness and the real-Tk suite walk every page and fail
  on a foreign bus.
- **Command palette fixes.** `Ctrl+K` opened a second palette every time
  it was pressed, the list could not be navigated with `Up`/`Down` (Return
  always picked the first entry), and neither `Escape` nor `Ctrl+K` closed
  the window from inside it - a second toplevel has its own bindtags, so it
  now carries its own bindings and the page accelerators `Ctrl+1..7` work
  from it too. Arrow keys work from the query field and from the list (the
  list binding wins over Tk's own cursor step, so a press moves one row), the
  highlight no longer snaps back to the first row on the key release that
  follows every arrow press, and the palette takes the keyboard focus once
  it is mapped (a window manager that keeps the focus on the parent window
  would otherwise leave it open but deaf).
- **The wheel scrolls the whole page again.** A page scroll area only bound
  the wheel on its own canvas, so with the pointer over a card's labels (the
  normal place to be) nothing moved and the overlay scrollbar looked like
  the only way down. The area now binds the wheel on the containing toplevel
  and scrolls when the pointer is inside its content, while a log `Text`, a
  table or the scrollbar itself keep the wheel and a wheel outside the area
  leaves it alone. The smoke harness models Tk's bindtags (own bindings,
  class, toplevel) so the routing is checked headlessly, and a real-Tk test
  builds a deliberately overflowing page and drives both cases.
- **One accent colour, free to choose.** Settings -> Appearance offers nine
  curated accents and a hex field; the colour is stored in the preferences
  and applied to *whichever* theme is active (`Theme.with_accent`), with
  the label ink derived from it (white while a bold UI label keeps its 3:1
  contrast, dark otherwise). Picking a theme's own accent is a no-op, so
  the designed pairs stay as authored. Presets carry the accent too.
  Fixed along the way: `Theme.contrast_ratio` unpacked `sorted()` the wrong
  way round and reported the reciprocal of the WCAG ratio (a 3.7:1 pair
  came back as 0.27); it had no caller, which is why it survived, and it
  now has tests.
- **Preset export/import.** A preset can be written to a JSON file and read
  back on another machine or project (`PresetStore.export` /
  `import_preset`, Settings -> Presets). The exported document carries
  layout, appearance, accent and note; an import keeps the name inside the
  file and refuses to replace an existing preset unless that is confirmed.
- **The window remembers being maximized.** `UiPreferences.zoomed` is
  saved with the geometry and re-applied at startup (best effort: a window
  manager that does not support `state("zoomed")` must not break startup).
- **A density rebuild keeps the keyboard focus**, not just the log, the
  selections and the scroll offsets.
- **`tools/desktop_tests.py`.** One command that runs the real-Tk desktop
  suite when the machine can (Tkinter + display, or `xvfb-run`) and falls
  back to the static contracts and the smoke harness otherwise, printing
  the exact package to install. `--strict` fails instead of falling back;
  the `desktop-ui-tests` CI job now goes through that script.
- **Restyles keep the view.** Switching density rebuilds a page's widgets
  (the new paddings have to be laid out, not patched), which used to blank
  the log panel, drop the selected rows and jump the page back to the top.
  The rebuild now captures that transient view state first - log text and
  poll cursors, tree selections, per-area scroll offsets - and re-applies
  it to the fresh widgets.
- **Movable cards and presets.** Dashboard, Training and Benchmarks
  arrange their cards on a layout board. Settings -> Layout studio moves
  a card up/down, changes its 1x/2x/3x span and hides it; the whole
  arrangement (plus theme and density) can be saved as a named preset
  under `.sandboxai/ui/presets/<name>.json`, applied or deleted. A
  preset written by an older build is repaired on load, and re-picking
  the current shell layout no longer rebuilds (and therefore no longer
  resets) every page.
- **Training replaces Agents.** The page is now the training-only
  operational core: launch deck (environment count, workers, device,
  presets 25k/100k/500k steps, resume checkpoint, live verdict),
  **Steps-or-Time budget** where a time-boxed run carries its minutes into
  the training config (`max_train_minutes`) and the trainer stops itself
  at the next safe step boundary, still saving the final checkpoint, plus
  the run table with its lifecycle actions, topology and log. The window's
  own watchdog only steps in after a grace period if a run can no longer
  reach a safe boundary. Benchmark and evaluation processes moved to their own pages.
  The Dashboard gained an *Active runs* strip that lists any live
  non-training process with a scoped Stop.
- **Benchmarks: Auto, Push or Custom.** Auto plans a host-scaled ladder
  from 64 to 128 environment processes and probes worker counts up to 32 -
  including the host's own step, not just the powers of two below the
  conservative `--env-workers auto` recommendation; Push widens the ladder
  to 256 environments to find the saturation point; Custom takes explicit
  environment/worker/step/minute lists. The previous single-button workflow
  capped the sweep at 64 environments and stopped the worker ladder at
  *physical cores - 2*, which is why a 64-env / 4-worker run could sit at
  10-20 % CPU. Invalid plans disable the start button with the reason
  instead of starting a doomed sweep.
- **Chrome and readability.** Cards are rounded and themed; tables keep
  every column reachable with overlay scrollbars and fit-to-width columns
  instead of two permanently pinned native bars; the log panel wraps by
  default (with a wrap toggle) and its overlay bars appear only while
  scrolling; spacing and fonts scale with the display DPI.
- The Benchmarks page runs the full staged pipeline with host-scaled
  defaults in Auto mode, shows a live phase strip (discover → screen →
  devices → validate → pick → apply) and applies the winning
  configuration automatically (persisted recommendation + Training
  launch deck). The budget/grid/finalist form and the custom-configuration
  card are gone from the GUI; the CLI keeps every knob for scripted
  sweeps, and Custom mode passes explicit lists through.
- `tools/control_center_smoke.py` constructs the real application against
  a small fake `tkinter` and drives every page: a development aid for
  machines where the Tk suite skips. It verifies wiring, not pixels.
- Failed launches now surface their real cause: a failing process's
  snapshot appends its last stderr line to the bare
  "process exited with code N", and the agent view prefers the backend's
  published `status.json` error over that process-level symptom.
- The launch slot refuses a doomed launch up front:
  `adapter.validate_runtime_configuration` reports an unresolvable Godot
  executable (including the launch form's explicit override, probed under
  its own cache key) as an error before a subprocess is ever spawned.
- Settings gained a *Godot executable* section backed by
  `adapter.configure_godot_executable`: Verify & save probes the given
  path with the same runtime check the benchmark uses and only persists
  an executable that was actually seen working
  (`.sandboxai/settings.json`), which training, benchmarks and
  evaluations all already resolve.
- The Agents launch form opens with measured defaults: the hardware
  profile's device choice and, once the automatic benchmark has been
  applied, its winning topology. Agent rows are colour-coded by
  lifecycle (failed red, running green, transitional amber).

### TTK Testing scope
- **The bounded live-helper surface is documented, and the statements that
  contradicted it are corrected.** `sandboxai.ttk_testing` had grown the
  helpers the Control Center uses for a manual calibration session (process
  and window probing, reading the client's own log for the live place id,
  launching through the user's shortcut or the public deep link, focusing the
  window, screenshot capture) while `docs/ARCHITECTURE.md`, `PROJECT.md`,
  `README.md` and this file still said "no Roblox integration exists or is
  planned". None of that removed the red lines - no memory reading, no input
  injection, no packet inspection, no client modification, no gameplay
  automation - so the docs now say exactly what the code does:
  `docs/TTK_TESTING_REFERENCE.md` gains the allowed/never table for the
  helper surface, and the Helmetcam row records that the official game
  documents **P = Helmetcam** while the project still excludes it as a
  presentation-only mode.
- `sandboxai ttk-status` plus `sandboxai.ttk_testing`: one source-traceable
  TTK Testing evidence manifest. It separates verified controls and
  wound-painting/bleeding from calibration-required physics/weapon/reload
  behavior, records the normal-first-person/no-Helmetcam project decision,
  and makes manual weapon switching (never automatic empty-magazine
  switching) explicit.
- `docs/TTK_TESTING_REFERENCE.md`: official-source links, the mechanics
  matrix, screenshot-only calibration protocol and exclusions for the
  TTK-focused rebuild.

### Removed
- The invented weapon drills `rifle_lane_drill`, `shotgun_breach_drill`,
  `sidearm_finish_drill` and `smg_tracking_drill`, with a regression test
  preventing their return.
- The unused external-game/Roblox adapter boundary, mock, command and
  documentation. What survives is only the bounded calibration helper
  surface (detect/launch/focus/screenshot, hand-typed values) described in
  the TTK Testing scope entry above - never an automation path.
- The unverified weapon-handling report, whose third-party claims did not
  meet the TTK-only evidence boundary.

### Fixed
- **Three scripts called `VectorMath` without preloading it, and the engine
  refused to compile them.** `environment_reset.gd`, `world_generator.gd` and
  `scenario_library.gd` were given a `VectorMath.yaw_deg_from_direction(...)`
  call but only one of the three also received the matching
  `const VectorMath = preload(...)` line. A `class_name` lives in the editor's
  global class cache; a headless `--script` run on a fresh checkout has no
  such cache, so the bare name is a compile error there. The three files did
  not load, and every call into them failed with "Nonexistent function ...
  in base 'GDScript'" - 38 engine tests went red with messages about
  everything except the actual cause (an environment that never reset, a
  weapon handling level that never armed, maps whose layout dictionary had no
  `map_id`), while `ruff`, `mypy`, `gdlint`, `gdformat` and the Python suite
  stayed green. Fixed by the missing preloads. The blind spot that let it
  through was this analyzer itself: it accepted a bare `class_name`
  reference because it is project-wide *by name*. It now has
  `check_class_cache_references`, which reports exactly that reference when
  the file neither preloads the class nor inherits the const from a
  project-local base, and the new tests pin all three legitimate forms
  (own preload, inherited const, own class name). The engine suite is the
  only place that can prove this class of bug, which is why the CI failure
  comment now carries a filtered error extract instead of only the last
  50 KB of output - the root cause appears at the start of the log, not
  in the summary that used to be the whole comment.
- **A bearing had two opposite sign conventions, and eight of the eleven
  bearing fields used the wrong one.** The observation vector's bearings came
  from two implementations: `Observation` measured the angle as a yaw
  difference
  (positive = the agent's right, which is what the contract documents for
  `enemy_bearing_norm` and what `look_yaw_axis = +1` does), while
  `ArenaWorld` (objects, nearest obstacle), `SoundBus` (hearing) and the
  memory context through `PerceptionSystem.bearing_deg` took the sign of
  `forward.cross(direction).y` - the opposite direction in Godot's
  right-handed frame. Consequences: a contact and the crate next to it were
  reported on opposite sides of the agent, hearing pointed the wrong way
  relative to every other channel, and the Stats page showed an operator the
  mirrored world. The fix is one implementation of the convention
  (`scripts/core/vector_math.gd`, `VectorMath.signed_bearing_*`) that every
  caller now uses; `PerceptionSystem.bearing_deg`'s docstring said "negative
  is left, positive is right" while its body computed the opposite, so the
  comment was wrong too. Verified: the corrected formula agrees with the
  yaw-difference reference to 1e-10 degrees over 200k random geometries (a
  Python port, since no engine is available locally), new GDScript tests put
  an enemy and a crate on the same 45-degree ray and require equal
  normalized bearings from both, the sound test pins its sign, and a new
  drift test in `python/tests/test_contract.py` fails if any GDScript file
  outside `vector_math.gd` takes the sign of `cross().y` again (checked by
  re-introducing one). The same hunt removed the two other angle formulas
  that existed in more than one place: the yaw of a direction
  (`rad_to_deg(atan2(x, -z))`, six copies in the agent, enemies, stub
  controller, scenario/world generators and environment reset) and the
  elevation of a point (`atan2(y, horizontal)`, one copy in the perception
  system and one in `Observation`) are now `VectorMath.yaw_deg_from_direction`
  and `VectorMath.elevation_deg`, with the same drift test covering both
  (also checked by mutation). Both refactors are expression-for-expression
  equivalent, so no behaviour changed. Recorded replays and the wire format
  are unchanged; a policy trained before this fix simply saw some channels
  mirrored.
- **A stale contract-version expectation that only the training extras
  would have caught.** `test_checkpoint_eval.py` asserted that an evaluation
  report records `contract.version == 3`. The v4 object block made that 4,
  but the test skips without torch/SB3, so the local suite stayed green and
  CI would have failed on the first push. Running the full suite with the
  training extras installed (`1123 passed, 46 skipped, 1 failed`) found it;
  the same suite is green after the fix (`1124 passed, 46 skipped, 880
  subtests`). The test now reads `contract.CONTRACT_VERSION` instead of a
  literal, because the
  report has to record the contract the code declares - the deliberate
  freeze of the contract fingerprint lives in `test_manifest.py` and keeps
  its literal on purpose.

### Added
- **Contract v4: the policy can see the objects around it.** Three ranked
  slots for the nearest *visible* world objects - closest surface point,
  distance, bearing, a normalized kind ordinal and a per-slot visibility
  flag - plus the visible total are appended as indices **84-105**; indices
  0-83 are unchanged, and the arena fence (`Obstacle.Kind.BOUNDARY`) is never
  reported. A slot is only filled by geometry the agent could actually see
  from where it stands (FOV cone, `VISION_RANGE`, occlusion probe), which
  keeps the contract's "no privileged information" rule intact: the policy
  learns that cover exists and where it is, not what the map contains.
  `SandboxConfig.OBJECT_KIND_COUNT`/`OBSERVATION_MAX_TRACKED_OBJECTS` mirror
  the budgets, `ArenaWorld.visible_object_infos()` is the single visibility
  query behind it, and the drift tests in `python/tests/test_contract.py`
  compare every new field against `contract.OBSERVATION_SPEC`.
- **Object-visibility regression tests, and the perception's own cone.**
  `tests/test_observation_objects.gd` pins the contract-v4 query behaviour
  that only a live engine can show: nearest visible object first, the fence
  never reported, the FOV cone and vision range honoured, cover hiding what
  stands behind it (while a box that directly faces the agent is still
  reported - the occlusion probe must not count a box as its own blocker),
  boxes that block neither sight nor movement skipped, empty slots neutral,
  and the visible count *not* truncated at the three-slot budget. The
  simulation now forwards its perception system's `fov_deg`/`vision_range`
  into the observation context, so the objects in the vector are exactly the
  ones that system could see, and `contract.OBSERVATION_COUNT_NORMALIZER`
  (with its own Godot-constant drift test) mirrors
  `Observation.COUNT_NORMALIZER` so the Control Center can render "2" or
  "8+" instead of a normalized fraction.
- **Observation-width drift test.** Every living document that restates the
  vector width (`README.md`, `PROJECT.md`, `AGENTS.md`, `CONTRIBUTING.md`,
  `docs/ARCHITECTURE.md`, `docs/CURRICULUM_AND_COMBAT.md`,
  `docs/DEBUG_GUI_AND_BENCHMARKING.md`, the PR template) is now read as text
  and its "<N>-float" / "<N>-field" / "<N> -> 128 -> 128" mentions must equal
  `contract.OBSERVATION_FIELD_COUNT`; the contract document itself is only
  checked on its *first* mention because its version history quotes the old
  widths on purpose. Nine documents still claimed 84 floats when the v4 block
  landed.
- **Trainer-enforced wall-clock budget** (`TrainingConfig.max_train_minutes`,
  `sandboxai train --max-train-minutes`). The budget used to be a
  window-side request: the Control Center watched the elapsed time and
  asked the process to stop, which only worked while that window was
  open. It is now part of the training loop (`ppo._TimeBudget` +
  `_should_continue_step`, checked in the same per-step path that honours
  an operator stop), so the run ends at the first safe step boundary after
  the budget regardless of who started it, and reports why:
  `run_summary.json` gains `stop_reason`, `max_train_minutes` and
  `elapsed_minutes`, and a run stopped by the budget is written to the
  run manifest as `stopped` instead of `completed`. A resumed run gets a
  fresh budget. Unit-tested without stable-baselines3
  (`python/tests/test_ppo_time_budget.py`).

- `sandboxai.hardware_profile`: the single device-comparison
  implementation behind the first-start hardware wizard. It compares CPU,
  Hybrid (GPU updates + CPU inference) and CUDA — offering the GPU
  candidates only when a CUDA device is present — by timing short real PPO
  training slices, never invents a throughput, honours cancellation
  between candidates, persists the selected profile to
  `.sandboxai/hardware_profile.json`, and falls back to CPU defaults when
  nothing can be measured. Exposed through the adapter
  (`hardware_candidates`, `hardware_profile`, `run_hardware_wizard`) and
  the Tk-free view model (`hardware_profile_view`) so no GUI grows a second
  benchmark. Orchestration is unit-tested with injected measurement
  functions; the engine-backed CPU measurement is verified end to end
  against a scripted fake Godot bridge.

- `LICENSE` (MIT). The repository had none, which made every "open
  source" claim in the README legally meaningless: without a license,
  default copyright applies and nobody may fork, vendor or redistribute
  the code.
- Project governance: `CONTRIBUTING.md` (the exact local check sequence),
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, a PR template with
  the checks as a checklist, and issue templates.
- `.pre-commit-config.yaml` running ruff, ruff-format and gdformat, so
  the formatting gate is reachable before CI rather than after it.
  mypy is deliberately *not* a hook - it needs the training extras
  installed and would make `git commit` depend on a 2 GB environment.
- Dependabot for `pip` and `github-actions`, plus a `pip-audit` CI job.
- `mypy` gate. The package has shipped a `py.typed` marker without
  anything verifying the annotations behind it; a `typecheck` job now
  runs mypy over `python/sandboxai` with the training extras installed,
  so the annotations are checked against real torch/SB3/gymnasium types.
  Configured in `pyproject.toml` (invocation is a bare `mypy`) and
  installable as the `typecheck` extra.
- `docs/README.md`: an index of the twelve documents under `docs/`, with
  one line on what each answers and who it is for.
- `python/tests/test_docs_consistency.py`: turns the documentation into
  something CI can fail on. Engine version, package version and the
  Python floor each have exactly one source of truth
  (`contract.py::GODOT_VERSION`, `pyproject.toml::project.version`,
  `requires-python`), and every prose restatement is checked against it.
- `sandboxai --version`.
- `python/tests/test_ppo_helpers.py` (39 tests) and
  `python/tests/test_weapons_constant_parser.py` (18 tests), covering
  code paths that previously only ran inside a full training loop.
- `docs/PYTHON_MODULE_MAP.md`: all 46 modules of the flat `sandboxai`
  package grouped by theme, each described by its own docstring summary.
  This is the deliberate alternative to splitting the package into
  subpackages: every module's import path is public API and appears in
  docs, user scripts and saved run manifests, so a reorganisation would
  rewrite several hundred call sites to buy navigability that a document
  can provide instead. The map is enforced - a new module nobody
  classified, or a description that no longer matches the docstring,
  fails `test_docs_consistency.py`. The same test now also asserts that
  the module-level import graph is acyclic, which is the structural
  invariant that actually matters here, and names the loop when it is
  not. `docs/ARCHITECTURE.md` no longer keeps its own hand-written file
  list, which had gone stale at thirteen of forty-six modules.
- `.gdlintrc`, pinned from `gdlint -d`. `CONTRIBUTING.md` already
  referenced it; without the file gdlint silently followed whatever
  defaults the installed gdtoolkit version shipped, so a toolchain bump
  could change CI's verdict with no GDScript change.
- Python CI workflow (`.github/workflows/python-tests.yml`): ruff lint,
  a numpy-only core-tests job, and a full-tests job (training extras) run
  on a `ubuntu-latest` / `windows-latest` matrix - the project's two real
  target environments (native Windows, and Linux/WSL, which is a genuine
  POSIX host even when the project folder lives on a Windows desktop).
  macOS is explicitly out of scope. The Python suite previously only ran
  on Linux via `godot-tests.yml`.
- `requirements-lock-linux-py311-cpu.txt`: a real, generated-and-verified
  pip-freeze snapshot of the exact dependency versions this repo's Linux/
  CPU CI resolves to. Not a cross-platform lock (torch/CUDA wheels differ
  per OS/accelerator) - see the file header and `PROJECT.md` section 13.
- Desktop Control Center (`sandboxai control-center-desktop`): a Tkinter
  engineering GUI (Dashboard/Training/Agents/Benchmarks/Evaluations/Runs
  and Checkpoints/System and Telemetry/Settings pages) backed by
  `sandboxai.adapter.SandboxAIAdapter` and the Tk-free
  `sandboxai.control_center_viewmodel` presentation layer. See
  `docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md`.

### Changed
- The interactive Godot and Tk desktop Control Centers now use a
  high-contrast cyan/violet glass-and-telemetry visual system: shadowed
  panels, clear hover/pressed states, branded status and navigation zones,
  radar-style local perception rendering, and a compact first-person
  instrumentation overlay. The Godot backdrop is constructed only with the
  GUI, ignores input, and has a persisted reduced-motion preference that
  stops backdrop/page-transition animation without changing simulation,
  telemetry or training. These remain explicitly local calibration
  presentation, not a claim about TTK Testing's player HUD.
- The Godot Control Center now carries the same visual system through dense
  telemetry surfaces too: direct Tree/ItemList/RichText controls, option
  popups, scrollbars, tooltips, progress meters and compact sparklines no
  longer fall back to an engine-default palette. A full-workspace horizontal
  scroll contract keeps the navigation rail, simulation docks and all
  existing controls reachable on narrow desktop windows; the first-person
  reticle's centre point now agrees with its local ready/range state. The Tk
  log reader now resumes its existing auto-follow behavior when an operator
  scrolls back to the newest line instead of silently remaining paused. Its
  Tk background runner now owns exactly one pending pump callback and
  cancels it during close, avoiding an unbounded callback chain during
  manual test/drain calls and late callbacks after window destruction.
- CI (`python-tests.yml`) no longer duplicates the Python suite that
  `godot-tests.yml` was also running; adds pip caching, a
  `concurrency` group that cancels superseded runs, and pins every
  third-party action to a commit SHA rather than a mutable tag. The full
  training matrix has a 30-minute budget and explicitly bounds native math
  threads to one: its tiny torch/SB3 test models were dramatically slower
  when a high-core Windows runner fanned their work out across every CPU.
  Its unbuffered, verbose pytest output now identifies the last test if a
  platform-specific stall ever reappears, and `pytest-timeout` turns a
  wedged individual test into a failure with a stack after three minutes
  rather than an opaque job-level cancellation. Job order: lint, typecheck,
  core-tests, gdscript-checks, desktop-ui-tests, full-tests, coverage,
  audit.
- Ruff now enforces `E,W,F,I,UP,B,SIM,C901` instead of `F` alone, with
  `max-complexity = 15`, and `ruff format` is the formatter of record
  for the Python half. Clearing the new rules touched most of the
  package; the complexity ceiling in particular forced `train_ppo`
  (cyclomatic complexity 103) apart into seven callback factories, a
  `_SelectionState` dataclass and a module-level `_EvaluationDriver`,
  leaving the orchestrator at 15 and the callbacks unit-testable
  without a live environment. The same treatment went to the CLI
  dispatcher, `RuntimeValidator.validate`, the evaluation battery, the
  plan scheduler, dataset validation, replay parsing, run inspection,
  adapter checking, BC training and the GDScript analyzer.
- `scripts/env/environment_core.gd` no longer suppresses gdlint's
  file-length rule. Two cohesive blocks moved out into the same kind of
  stateless static helper the file already used for `EnvironmentReset`
  and `EnvironmentIntrospection`: `EnvironmentCombat` (the agent's shot -
  hitscan, hit zones, the hit / near-miss / useless-shot classification,
  occlusion and the death bookkeeping) and `EnvironmentEnemies` (the
  per-tick opponent update and its motion sounds). Every function body
  moved unchanged, `EnvironmentCore` keeps a wrapper for each entry
  point so `step()` still reads as a sequence of named sub-steps, and
  the file went from 1235 lines to 963. No `.gd` file suppresses
  `max-file-lines` any more, so the limit is a limit again.
- Every `.gd` file is now `gdformat`-clean (123 of 153 files were
  reformatted), and both `gdformat --check` and `gdlint` gate CI.
- `AUDIT_REPORT.md` moved to `docs/AUDIT_REPORT.md`, where the rest of
  the long-form documents live.
- Coverage is measured and gated at 70 % (currently 86 %).

### Fixed
- **Destroyed widgets hand their animations back.** A pulsing status dot
  keeps the shared 16 ms ticker alive, but nothing cancelled its loop when
  the widget died, so a rebuilt page could leave the window animating at
  60 fps for a widget that no longer existed. `StatusDot` now releases its
  loop on `<Destroy>`, `PhaseStepper` cancels the transition it started
  (and replaces an in-flight one instead of stacking a second), and the
  smoke harness destroys a pulsing dot and fails if a loop is left
  registered (verified by removing the cleanup: it fires).
- **The fake Tk now fires `<Destroy>`.** Real Tk runs destroy handlers for
  every widget it tears down; the stub skipped the event, which made those
  handlers look unnecessary - the same blind spot that once hid two
  Tk-only bugs. Children are destroyed first, then the widget's own
  bindings run, exactly like Tk.
- **A destroyed widget can no longer freeze every animation.** The shared
  16 ms `MotionController` called each tween frame unguarded, so when a
  density change rebuilt a page mid-animation its next frame painted a
  destroyed widget, Tk raised `TclError` inside the ticker, and the ticker
  stopped scheduling itself - every remaining animation in the window went
  quiet. Failing frames and loop callbacks are now dropped individually;
  the smoke harness schedules a raising tween next to a healthy one and
  fails if the healthy one stops running (verified by letting the error
  escape again: it fires).
- **Theme listeners of rebuilt pages no longer pile up.** Every widget that
  repaints itself subscribes to the window's `ThemeBus`, and a density
  change destroys and recreates a page's widgets. The subscriptions had no
  owner, so each change left ~190 dead callbacks behind (97 -> 667 after
  three) and every later theme change walked them - invisible headlessly
  because a dead widget's repaint is a suppressed error. `subscribe()` now
  takes `owner=` and the bus drops listeners whose widget no longer exists;
  the smoke harness changes density three times and fails if the settled
  listener count grows (verified by disabling the pruning: it fires).
- The JSON-lines bridge no longer dies on engine output that happens to be
  valid JSON. `GodotProcessTransport.receive` called `.get("ok")` on
  whatever `json.loads` returned, so a single `print(0)` or `print([1, 2])`
  anywhere on Godot's startup path took the transport down with
  `AttributeError: 'int' object has no attribute 'get'` instead of being
  skipped like every other informational line. Only JSON *objects* are now
  treated as protocol frames.
- `EpisodeMetrics` no longer reports a negative `reaction.shot_latency`.
  The first trigger pull was latched whenever it happened, so an agent that
  fired blind before ever seeing the enemy reported a negative "reaction
  time" — neither the -1 "never happened" sentinel nor a latency. The first
  shot is now latched only at or after first contact; the shots themselves
  are still counted in `aim`.
- `MetricsAggregator` no longer turns "never happened" into "happened
  instantly". It correctly excluded the -1 sentinel from latency means, but
  then returned 0.0 for the empty remainder, so a policy that never
  detected, never fired and never reached half map coverage aggregated to
  the fastest possible detection, shot and exploration times. When every
  episode reported the sentinel, the aggregate keeps it.
- The Control Center formatters render non-finite numbers as `n/a`.
  `format_number(float("inf"))` raised `OverflowError` from
  `int(round(inf))` and took down the page rendering it, while
  `format_fraction_as_percent` / `format_bytes` printed `nan%` and
  `inf PB` as if they were measurements. NaN and infinity survive a
  `json.dump`/`json.loads` round trip, so a diverged run's own summaries
  deliver them to the GUI.
- `sandboxai record` builds its Godot command from the executable exactly as
  the CLI resolved it (flag, `GODOT_PATH`, remembered setting, PATH probing)
  instead of re-resolving inside `build_record_command`. On WSL, where the
  default `godot` resolves to a Windows binary, that internal resolution let
  the interop rewrite the `--output` dataset path into `C:\...` form — which
  `pathlib` does not consider absolute, so the resolved location the builder
  computed never reached the command. The builder is now a pure function of
  its arguments; every real invocation produces the same argv as before.
- The checkpoint-evaluation battery perf-regression tests raised
  `AttributeError` on `config.run_id`: the evaluation report gained
  `run_id`/`experiment_id` and the normal-evaluation episode/environment
  counts, but their hand-rolled config mock was not updated alongside the
  producer. The mock now carries the fields the report reads.
- The desktop Control Center no longer tears down its Tk window while a
  background worker is still running. `BackgroundRunner.close()` waited on
  nothing, so a worker in flight when the window closed could drop the last
  reference to a Tk widget and run its finaliser off the Tk thread. On
  Windows that is not an exception but a silent interpreter crash
  (`0x80000003`), which is how CI found it.
- The desktop Control Center no longer lets a background thread collect
  garbage. CPython runs the cyclic collector in whichever thread trips the
  allocation threshold, discarded widget trees are cyclic, and finalising
  one off the Tk thread reaches into Tcl from a thread that does not own
  it. `BackgroundRunner` now suspends automatic collection while it is
  alive and collects from its Tk-thread poll instead.
- The Tk process-log reader no longer steals focus from an operator reading
  older output after they drag its scrollbar or navigate with the keyboard;
  either path now pauses live follow until the viewport returns to the newest
  line. It also exposes the horizontal scrollbar required to inspect long
  commands, file paths and tracebacks while keeping the deliberately
  unwrapped log text readable. Agent-log reads now capture the selected
  process and incremental cursors at submission time, coalesce a slow poll,
  discard a late result for a prior selection (including A → B → A), clear
  output on deselection, and bind Stop/Force Stop to the process that was
  selected when the button was pressed. Evaluation comparisons and run reports
  now use the same selection-generation contract, so a slow disk read cannot
  overwrite a newer selection or leave vanished data presented as current.
  Periodic Dashboard, Training, Benchmark, Evaluation, Runs, Agents and
  System reads are now coalesced per page operation, preserving the bounded
  worker pool and discarding a response that no longer belongs to its active
  run/process selection. An overlap retains exactly one latest-only follow-up
  rather than dropping the freshness request, and changing the output root
  clears guards owned by the retired worker. Every dense Tk inventory table
  now has both native scroll axes, so a narrow desktop never makes its
  right-hand run/checkpoint/error data unreachable.
- Loading a behavior-cloning checkpoint could execute arbitrary code.
  `bc.py` passed `weights_only=False` to `torch.load` at three call
  sites, which unpickles whatever the file contains; a checkpoint is
  therefore a program, not data. Nothing needed it - the checkpoints
  hold tensors, strings, numbers, lists and dicts - so all three go
  through one `_load_checkpoint()` with `weights_only=True`, and a file
  that cannot be read that way is refused with an explanation instead of
  being trusted. The regression test builds a genuinely hostile
  checkpoint and asserts its payload does not run.
- JSON files written on Windows were unreadable. Twenty read sites used
  `encoding="utf-8"`, which hands a byte-order mark straight to
  `json.loads`; PowerShell's `Out-File -Encoding utf8` writes one, so a
  config produced that way failed with `Expecting value: line 1 column
  1`. Reads now use `utf-8-sig`, which is byte-identical for files
  without a BOM. Writes deliberately still use `utf-8` - emitting a BOM
  would be the opposite bug - and a test enforces both directions.
- `ppo._mean` raised `ZeroDivisionError` on an empty buffer. No caller
  could reach it, but the next one would have: it now returns 0.0,
  because "no episodes finished in this interval" is a normal state
  early in a rollout.
- `contract.validate_observation_spec()` consisted entirely of `assert`
  statements, so under `python -O` it silently became a no-op while
  still reading like a guarantee. It raises `ValueError` now.

- `ReplayEpisode.event_tick` was a copy of the recorder's property and
  read `self._tick`, a cursor only the recorder has - so asking a parsed
  episode where to anchor an event raised `AttributeError`. Found by
  mypy.
- The checkpoint evaluation battery called
  `record_step(action, reward, events=events)`, but
  `ReplayRecorder.record_step` takes no `events` parameter, so every
  battery run with replay recording enabled died with `TypeError` on the
  first step. Events are a separate `record_events(...)` call, as
  `TrainingPipeline` has always done it. Found by mypy.
- `weapons.py` parsed GDScript weapon constants with `eval()` on text
  read from disk. Replaced with an AST walker over an explicit operator
  whitelist; a repo-wide old-versus-new comparison caught that the
  whitelist also has to cover `1 << 20`. There is no `eval`/`exec` left
  in `python/sandboxai/`.
- Nine `except Exception` handlers narrowed to the exceptions they can
  actually see, so genuine defects stop being swallowed as "optional
  dependency missing".
- `manifest.py::godot_snapshot()` imported a nonexistent
  `GodotRuntimeValidator` class; a broad `except Exception` silently
  swallowed the resulting `ImportError`, so every run manifest ever
  produced recorded `godot.version: null` even with a working Godot
  install. Now imports the real `runtime_validation.RuntimeValidator`.
- The first Windows CI run surfaced 10 pre-existing test-fixture bugs
  that had never been exercised on native Windows before (missing
  `os.name`/`USERPROFILE` mocks, a resolved-vs-raw project-path
  comparison, a POSIX-only fake Godot executable, a `ProcessManager`
  background-thread race in a force-stop test, and four `is_wsl()`-mocked
  WSL path-translation tests that assumed a POSIX `Path` while running
  under native Windows' `WindowsPath`). None were production bugs; all
  ten are fixed and the Windows leg has run fully green since (see git
  history around the two "Fix ... Windows-only test failures" commits
  for the exact diagnosis of each).

### Removed
- `requirements.txt`. Nothing referenced it - not the README, not CI
  (the pip-audit job builds its own list from `pyproject.toml`), not the
  helper scripts - and it had already drifted, declaring `torch>=2.1`
  where `pyproject.toml` says `torch>=2.1,<3`. Dependencies are declared
  in one place now, and a test keeps the second copy from coming back.
- Dead/unused imports across several `sandboxai` modules found by the
  new ruff CI job.
