# Control Center (headless desktop application)

The Control Center is SandboxAI's operator application: one native
Python/Tkinter window to launch, operate, benchmark and inspect the
headless training stack. It is **headless-only** — it never renders the
game, never opens a `.tscn`, and never plays the agent. Training always
runs `scripts/rl/rl_server.gd` with `--headless` in separate processes;
the GUI exchanges only cooperative command files and read-only status
artifacts with them.

```bash
# from the repository root (no Godot editor needed):
python3 main.py

# equivalent, through the CLI:
sandboxai control-center-desktop
sandboxai control-center-desktop --project-path /path/to/SandboxAI --output-root training
```

On Windows, `tools\windows\start_control_center.bat` launches it from the
project's Python environment (activates `.venv` if present, checks that
`sandboxai` and Tkinter are importable, and reports a clear message
instead of a stack trace if not).

## Requirements

The Control Center itself needs only Python 3.11+ with Tkinter (standard
library). Training and benchmarking additionally need the Godot 4.7.2
executable on this machine and the training extras
(`pip install -e '.[training]'`) — the GUI reports exactly what is missing
on the System page, and can install them from there.

## Architecture in one paragraph

`control_center_desktop.py` builds the window, the shell and the
navigation; `control_center_pages.py` holds the pages;
`control_center_ui.py` provides the themed primitives (rounded cards,
overlay scrollbars, segmented controls, toasts, layout board);
`control_center_widgets.py` provides the reusable infrastructure
(background runner, bounded log panel, tables, charts, tooltips);
`control_center_theme.py` owns the design tokens, the five themes, DPI
scaling and the persisted preferences; `control_center_layout.py` owns
the movable-widget model and the saved presets; `control_center_viewmodel.py`
is the Tk-free presentation logic, tested without a display. Every page reaches
the system exclusively through `sandboxai.adapter.SandboxAIAdapter` —
the same `train`/`resume`/`benchmark`/`evaluate` CLI commands and on-disk
artifacts any shell user would use, never a parallel implementation. The
agent lifecycle (launch/pause/resume/stop/restart bookkeeping) lives in
`sandboxai.agents` on top of the adapter's process manager. See
[ADAPTER_AND_DESKTOP_CONTROL_CENTER.md](ADAPTER_AND_DESKTOP_CONTROL_CENTER.md)
for the adapter contract.

## Appearance, layouts and presets

The window is meant to be *looked at* as well as operated, so it is
built from a small set of switchable choices instead of one hard-coded
look. Everything below is chosen on the **Settings** page and persisted in
`.sandboxai/ui/preferences.json`, so the next start looks the same.

| Choice | Options | Effect |
| --- | --- | --- |
| Theme | Corz, Midnight Cyan, Neon Lime, Graphite Mono, Light | Full colour set incl. text, borders, states and chart colours |
| Density | Comfort, Compact, Ultra | Row heights, paddings and one font step (never below the 11 px floor) |
| Motion | Off, Reduced, Normal, Cinematic | Speed of the small transitions; **Off** renders final states immediately |
| Accent | any `#rrggbb` typed directly | Replaces the accent of whichever theme is active; *Theme accent* restores the designed one |
| Corner radius, glow, grid | slider / toggles | Card rounding and the optional backdrop effects |

The accent is one colour, not a sixth theme: it is stored in the
preferences (`"accent": "#22d3ee"`) and every theme wears it, so an
operator can keep the Corz surfaces with a mint accent. The label colour
*on* the accent is derived (`Theme.with_accent`), never guessed: white
when it keeps the 3:1 contrast a bold UI label needs, dark ink otherwise.
An override that matches the theme's own accent preserves
the designed pair (Cyan with its deep-teal ink) exactly. The header stays
free of theme and layout selectors; those less-frequent choices live on the
Settings page instead of competing with run status in the top-right corner.

A theme change only repaints (ttk styles, canvas colours and the table tag
roles are re-applied). Every widget that draws itself subscribes to the
window's `ThemeBus`; a widget constructed without one would silently fall
back to the module's default palette and keep Corz colours forever, so the
contract is checked twice - the smoke harness and the real-Tk suite each
walk every page and fail if any widget carries a foreign bus. Helper
widgets find the nearest bus by walking up their parents (`_cc_bus`), which
keeps a tooltip or a chart correct even when a call site forgets `bus=`.

Every subscription carries its **owner widget**, so the bus drops a
listener whose widget no longer exists. That matters because a density
change destroys and recreates a page's widgets: without ownership each
change left ~190 dead repaint callbacks behind (97 -> 667 after three
sweeps), and every later theme change walked them. The smoke harness
changes density three times and fails if the settled listener count grows;
the bus unit tests pin the pruning itself (a destroyed owner is neither
called nor kept, a listener without an owner is never pruned).

A **density** change has to rebuild the widgets so
the new paddings fit; that rebuild captures what the operator was looking
at first — the log text and its cursors, the selected rows, the page's
scroll offset — and restores it on the fresh widgets, so the log does not
blank out and the reading position survives.

One 16 ms `MotionController` drives every transition, and it isolates each
callback: a frame that raises `TclError` because its widget was destroyed by
a rebuild is dropped instead of stopping the ticker. Without that, a density
change landing mid-animation froze every other animation in the window.
Widgets also hand their animation back on `<Destroy>` (the status dot
releases its loop, the phase stepper cancels its transition), so the ticker
idles again the moment nothing is animating - and the smoke harness
destroys a pulsing dot and fails if a loop stays registered.

**Scrolling.** Tk delivers the wheel to the widget under the pointer, so a
page scroll area binds it on the *containing toplevel* - the one tag Tk adds
to every descendant's bindtags - and scrolls only when the pointer is inside
its own content. Scrolling therefore works over a card's labels, not just
over the bare canvas background, while a widget that scrolls itself (a log
`Text`, a table, the overlay scrollbar) and anything outside the area keep
the page still.

**Movable cards.** Pages that declare widgets (Dashboard, Training,
Benchmarks, Stats) are arranged by a layout board. On **Settings -> Layout
studio** every card can be moved up/down, given a 1x/2x/3x span and
hidden (cards marked `removable=False`, such as the launch deck, stay
visible). The result applies to the running window immediately.

**Presets.** A preset stores the theme, density, motion and every
card position as one JSON file under `.sandboxai/ui/presets/<name>.json`.
*Save preset* captures the current arrangement, *Apply* switches back to
it, *Delete* removes it, and *Rename to name* moves a preset to the name
typed in the field. *Export…* writes a preset to a JSON file anywhere on
disk and *Import…* reads one back: the file carries the whole document
(layout, appearance, accent, note), an import keeps the name stored inside
it, and a name that already exists is only replaced after a confirmation -
so importing somebody else's arrangement cannot quietly overwrite your
own. A preset from another build is repaired on import like any other:
unknown cards are dropped when it is applied. A rename onto a name that already exists is refused
rather than overwriting the other preset. A preset from an older build is
repaired on load: unknown cards are dropped and new cards appear with
their defaults, so a stale preset can never break the window.

**The window on the screen.** The window opens sized for the display it is
actually on: 1800x980 is the preferred size (a 1920x1080 screen uses its
width for the wide tables and keeps the title bar plus taskbar visible), a
smaller screen shrinks it down to the 1280x800 minimum, and a window with no
remembered position is centred. A *remembered* position is only a suggestion
- the screen it was captured on may have been bigger or gone entirely (a
monitor change, a different WSLg/RDP session), and restoring it blindly is
how a window opens with its title bar off the top of the screen, which reads
as "the GUI shows nothing". Anything that no longer fits is clamped back
onto the display, and **Settings -> Appearance -> Fit window to screen**
does the same on demand (it also un-maximizes first), so recovering a
window never means finding and deleting `preferences.json`. `F11`
maximizes and restores.

**How the window stays still.** Tk re-lays out everything whose requested
size changed, so a window built from cards that measure themselves can feed
its own layout back into Tk forever. That is what used to happen: the
benchmark workflow never settled (396 489 `<Configure>` events in half a
second, and the desktop suite hung inside `update()`, which only returns
when the event queue drains). The loop is pinned shut in four places.

* A **card** (a frame whose content decides its size) paints its rounded
  surface, accent rail and header on a canvas that is *placed* to fill it.
  `place` never contributes to a requested size, so painting cannot resize
  a card, and no card derives its own height from its geometry.
* The **layout board** reacts to a *width* change only, and only when the
  width clears a hysteresis band; the decision is debounced to one idle
  tick, runs under a re-entrancy guard, and returns without touching a
  single widget when the placement it computes is already on screen.
* **Tables** keep their declared column widths and let Tk's own column
  stretching fill spare width; nothing in Python rewrites a column width
  after the table is built (a table that re-fits its columns to the width
  it was just given produces the next Configure by itself).
* **Split cards use grids.** `ttk.Panedwindow` re-arranges its panes
  whenever a child reports a new requested size, and a table or a chart
  reports exactly that; the two never settle. The static page suite fails
  if a panedwindow comes back, and the smoke harness fails if a refresh
  writes a table column.

A table with no rows says why it is empty ("No runs found under the output
root yet — launch a training run and it appears here while it trains"):
headings over nothing read like a broken page. A page whose data is
entirely absent prints one line under its heading telling the operator what
to do next, and no table cell carries a bare "—" placeholder that could be
mistaken for data.

The layout state is also persisted without a preset, so closing and
reopening the window keeps the arrangement. If `tkinter` is missing
entirely, `main.py` still starts, prints the exact fix (`python3-tk`, or
a Python build with Tk) and exits cleanly instead of raising a stack
trace.

## Navigation

The shell is one thing: a fixed-width rail down the left with one button
per page, grouped under OVERVIEW, WORKFLOWS, ARTIFACTS and SYSTEM. It
cannot be switched to another shell and it cannot be collapsed. Those were
both options once, and neither earned its keep — "Command Board" built the
rail anyway (the branch was dead), a collapsed rail replaced the page
titles with two-letter codes (DB, TR, BM, ...) to win back about 160 px
and then needed a shortcut sheet at the bottom to stay readable, and
supporting the switch meant every shell change destroyed and rebuilt all
seven pages. A preferences or preset file written by an older build that
still names a removed shell is repaired to the rail on load.

## Keyboard

`Ctrl+K` opens the command palette, `Ctrl+1..7` jump straight to a page
(the window has seven), `F11` maximizes and restores, `Escape` closes the
palette, and `Up`/`Down` move its selection. The palette
carries those bindings itself in addition to the shell's, because Tk gives a
second toplevel its own bindtags - and pressing `Ctrl+K` twice reuses the
open palette instead of stacking a second window. The arrows are bound on
the query field and on the list itself - the handlers return `break`, which
suppresses Tk's own Listbox cursor step, so one press moves exactly one row
- and they therefore also work after a click into the list. The highlight
only resets to the first row when the filtered page list actually changes (a
key *release* must not undo an arrow press), and the palette forces the
keyboard focus onto the query field once it is mapped.

## Pages

### Dashboard

Live state of the newest run and the agents launched this session:
lifecycle state, run id, progress, steps/s, elapsed/ETA, environment and
worker counts, an agent lifecycle summary (coloured red when anything
failed), device, and reward. Stale status is labelled with its evidence,
exactly as `run_inspection.py` reports it.

The page deliberately does **not** chart run telemetry or list
checkpoints and PPO diagnostics. Runs / Checkpoints already plots those
series — for the run you selected, rather than for whichever run happens to
be newest — and polling them here meant reading telemetry files every tick
for a page nobody was reading.

### Dashboard -> Roblox

The one card that touches the real game. It opens the Roblox client in TTK
Testing, probes which place it is in, and — once a client is running —
focuses its window, captures a screenshot into `.sandboxai/ttk_captures`
and reads the resolution and HUD layout back out of that capture. Nothing
here plays the game: the human plays, the tool opens and photographs, which
is the whole of the bounded helper surface described in
[TTK_TESTING_REFERENCE.md](TTK_TESTING_REFERENCE.md).

Every action answers **in the card**: the line under the status says what it
did - `Screenshot - .sandboxai/ttk_captures/roblox_….png`, `Analyze HUD -
16:9, minimap top right` - and is coloured by whether it worked. The status
bar alone was not enough: the next poll overwrites it, so an action could
succeed and the operator would still be guessing. *Open Captures* opens the
folder the screenshots land in.

A capture is **shown**, not just named: the card draws the frame with a box
around every figure that stands out in it, and *Lift shadows* switches
between the raw capture and the same frame re-lit from its own local
illumination — which is what makes a player standing in an unlit corridor
readable on screen. The caption says what the frame is and how dark it was
(`1920x1080 · 2 boxed · 63% in shadow -> 35%`). The boxes are heuristics,
not recognition: each one carries the score and contrast it was chosen for,
so an operator can judge it instead of trusting it. An annotated copy is
written next to the capture as a PNG, for anyone who would rather open it
in an image viewer.

Focus, Screenshot and Analyze HUD are disabled until a Roblox client is
running, because pressing them without one used to answer with a platform
error nobody could act on. They work under WSL as well as on native
Windows: there is no `ctypes.windll` in a Linux process, so the same Win32
calls are made by the Windows host's PowerShell — the interop the launcher
already uses. A capture writes to the Windows form of the captures
directory so the file appears at the POSIX path this process expects.

### Training

The operational core: the launch deck plus the training-run registry in
one place.

- **Launch form** — five plainly scoped controls (environments,
  Godot workers, training steps, device and difficulty progression) over
  the same `TrainingConfig` the CLI validates. PPO optimizer settings,
  including learning rate, stay on the config's standard defaults in the
  desktop form; the CLI remains available for deliberate advanced tuning.
  The launch slot shows the resolved setup before anything starts:
  environment count, resolved worker count (`0 = auto` becomes the
  concrete number), shard topology, device and step count — or every reason
  the current values are invalid (parse errors, worker/environment
  incompatibilities, a CUDA request on a host without CUDA). The verdict
  comes from `benchmark_pipeline.validate_configuration`, the same check
  the benchmark plan uses, so the launcher and the benchmark can never
  disagree.
- **No hidden quick presets** — the training-step count is entered
  directly, and the launch slot explains that standard PPO settings are
  selected automatically. Benchmark recommendations populate the topology
  when applied; there is no separate sync button or one-click smoke run.
- **Budget: Steps or Time** — both modes are validated before launch.
  *Steps* ends exactly at the configured step count. *Time* is a
  **trainer-enforced** budget: the minutes travel into the run's
  `config.json` as `max_train_minutes`, the training loop stops itself at
  the first PPO step boundary after the budget is spent, and that normal
  shutdown still writes the final checkpoint. It therefore holds for a
  run started from the CLI, and for a run whose window has been closed.
  The run summary records `stop_reason: time_budget` together with the
  elapsed minutes. The window keeps a watchdog for the one case the
  trainer cannot cover — a process that can no longer reach a safe
  boundary — and only steps in after a grace period of
  `max(1 min, 10 %)` past the budget; it uses the trainer's own
  `elapsed_seconds`, so paused time does not burn budget. A budget whose
  checkpoint interval is far longer than the budget, or one above 1440
  minutes, is refused with the reason.
- **Run table** — every training agent launched this session with name,
  lifecycle, PID, environment/worker topology, device, budget, progress,
  steps/s, reward and errors. All values are backend-published facts;
  anything a kind does not publish renders `n/a`, never a guess.
  Benchmark and evaluation processes (which have no training budget)
  live on their own pages and appear in the Dashboard's *Active runs*
  strip while they are alive.
- **Lifecycle** — `AVAILABLE -> LAUNCHING -> RUNNING -> PAUSED ->
  STOPPING -> STOPPED`, plus `FINISHED`, `FAILED` and `RESTARTING`,
  derived from the OS process state, the trainer's `status.json` and the
  operator's requested action (see `sandboxai.agents.derive_lifecycle`).
- **Actions** — Launch, Pause/Resume (training only; other kinds keep the
  button disabled with a tooltip explaining why), Stop (cooperative for
  training, terminate otherwise), Restart, Force stop, Remove, Stop all,
  Clear exited. **Restart** stops the agent and relaunches it from the
  run's newest checkpoint (`latest.zip`, else the highest periodic
  `ppo_*_steps.zip`, else `best_eval.zip`/`final.zip`), continuing the
  same run directory, status file and event log.
- **Topology + log** — the selected agent's Environment -> Worker shard
  plan (the same contiguous plan `sharded_env.plan_shards` builds) and
  its bounded, incrementally polled process log.

### Benchmarks

The GUI offers one automatic benchmark and one idle action: **Start Benchmark**.
It builds the full host-scaled candidate ladder and measures every topology
for a fixed **20-second wall-clock window**, after separate startup/warmup.
There is no duration, mode, step, minute, environment or worker input in the
GUI. Fast topologies cannot finish early at a step target, and an overall
budget does not shorten windows or thin the candidate grid. **Cancel** is
available while running; screening checks it between vector steps.

The plan states its size and minimum screening duration before Start. At the
100-configuration ceiling the measurement windows alone total **33 minutes
20 seconds**, plus startup/warmup, device checks and PPO validation. This is
not the former 30-minute overall budget. Actual elapsed time is reported;
a vector step or PPO update finishes safely if it crosses a window boundary.

The sweep keeps the existing wide environment/worker ladder. Oversubscribed
rows remain measured and labelled, rather than being silently dropped.

**Simulation Steps/s** and **PPO Training Steps/s** have separate live cards
and curves. Simulation screening does not run a policy or optimizer. PPO
validation includes rollout collection and optimizer updates, but excludes
model/bridge startup, periodic evaluation/checkpoints and final saves from
its measured training-loop rate; total wall time remains in the report.
A timed slice without a completed PPO update cannot recommend a training
configuration. Real training can still be slower at harder curriculum levels
or with evaluation/recording enabled. **FPS / env** is simulation ticks per
second per environment, not render FPS; its curve contains simulation rows
only. Leaders and speedups are compared within one measurement path.

The table reports all tested topologies with actual measured duration,
throughput, per-environment rate, latency, stability and failures. Column
fitting uses measured heading-font minima and the available viewport; narrow
windows retain scrolling. Missing Godot is reported as unavailable, never
silently replaced by a synthetic Python benchmark.

The pipeline runs these stages:

1. **Discovery** — probe the Godot executable/version, torch, CUDA, CPU
   count and RAM. No Godot binary means an honest "unavailable" report with
   the command to fix it, never a fabricated number.
2. **Screening** — the real bridge benchmark
   (`benchmark.benchmark_simulation`) across the planned
   `(environments, workers)` pairs, each with the same fixed measurement window:
   separate startup/warmup, throughput, p50/p95 vector-step latency, episodes,
   resources and errors.
3. **Devices** — compare available devices when there is more than one
   candidate and no usable hardware profile is persisted; this reuses
   `hardware_profile`, the single device-comparison implementation.
4. **Validation** — run a separate fixed *real PPO training window* per
   surviving finalist on the selected device, preserving the standard rollout
   geometry and requiring at least one optimizer update.
5. **Recommendation** — automatic mode requires validated PPO throughput,
   within a near-best band of the best measurement, preferring stability
   (low latency jitter, no errors, fewer workers) over an unstable peak.
   The `rationale` and `warnings` quote only measured numbers.

Planning estimates describe overhead, not throughput. Fixed measurement
windows are never resized from an estimated rate; only measured rates are reported.
The full report is persisted under
`training/benchmarks/pipelines/<timestamp>/` (`pipeline.json` plus
benchmark-history-shaped `benchmark.json`), the recommendation is stored
machine-locally in `.sandboxai/recommended_config.json`, and the latest
persisted report is shown on the next start. The recommendation is applied
to the Training launch deck automatically; a not-yet-opened Training page
picks it up when it is built.

Advanced budget and grid options remain available for scripted benchmark
sweeps through the CLI; they are not exposed as GUI setup fields:

```bash
sandboxai benchmark-pipeline              # fixed automatic windows
sandboxai benchmark-pipeline --budget-mode time --minutes 15  # legacy scripted mode
sandboxai benchmark-pipeline --budget-mode steps --steps 2000
sandboxai benchmark-pipeline --show        # print the persisted recommendation
```

### Runs / Checkpoints

A browser over the on-disk run artifacts (state with evidence, progress,
checkpoint and evaluation inventory, log sizes, manifest provenance),
read through `run_inspection.py`.

Evaluations are started from the selected run, not from a page of their
own: **Evaluate latest** / **Evaluate best** run the frozen-weights
battery (episodes, environments, device) on that run's checkpoint, the
card lists every evaluation this run has, and selecting one - or several,
for a comparison - prints its win/loss/timeout outcomes, combat and
accuracy diagnostics and action-head/zero-shot checks. The evaluation is
a property of a run, so it is read next to the run it measures.

### Stats

What the policy actually receives, decoded from a recording - and, since the
complaint was that none of it could be read, what the numbers *mean*. A
**How to read this page** card states the scaling once, from the contract's
own constants: what a tick is (126 values in, 6 out), that "norm" is a
division by a fixed maximum rather than a percentage (a distance of 0.50 is
14 m, because the arena diagonal is 28 m; counts are divided by 8), that
signed values are directions and unsigned ones are amounts, and that a
contact's position is always relative to the agent.

**The agent's own screen** draws the same vector as a picture: the three
contacts where they fall in the agent's field of view, each as the box the
engine reported and lit by how readable the target was - because "inside
the cone with a clear line" and "a readable target" are different facts,
and only the second one is what the policy can act on. The contacts table
carries the same two numbers as columns (*Exposure*, *Clarity*). Centre of
the drawing is under the crosshair, so it is a view and not a map: a
contact behind the agent has no box and is not drawn at all.

The contract
table (`contract.OBSERVATION_SPEC`) is complete without any recording; a
replay is the only thing that can show real values, and only
`--replay-detail detailed` stores the observation vector per tick - a light
replay is reported as such rather than rendered with zeros that would read
like data. A replay recorded under an **older observation contract** still
loads (the same allowance the CLI's `--allow-contract-mismatch` makes) and is
labelled: the values are readable evidence, but they are not comparable with
a current policy's input, and the page says exactly that instead of painting
them into the current contract's table. The page shows the three tracked contacts (relative position,
distance, bearing, elevation, health, visibility, in-FOV/LOS, information
age, confidence and whether the belief came from vision or hearing), the
world objects and memory rows around the agent, the hearing summary, the
recorded action per component, the raw 126-value observation vector, and
the TTK Testing evidence manifest (what is verified about the real game and
what still needs a manual measurement). Listing a large folder is bounded
work: a replay's header and tick count are read in one pass and cached behind
the file's `(mtime_ns, size)` stamp, so a warm rescan does not re-read an
unchanged recording. `tools/replay_scan_probe.py` measures it - synthetic
corpus, not a game benchmark: 1200 recordings / 200 ticks / 144.7 MB, warm
rescan median 18-21 ms after the change (38.8 ms before), cold scan
~41-43 ms.

### System / Telemetry

Real dependency and device status (Python, torch, Godot binary/version,
CPU/RAM) and bounded live telemetry charts. Unavailable metrics are
shown as such, never estimated.

The **Training dependencies** card lists every optional extra with
*installed* or *missing*, read from this interpreter rather than from
memory, and offers one button — *Install training extras* — that runs
`pip install -e '.[training]'` in the project root and shows pip's own
output. Training cannot start without those extras, so the page says which
ones are missing and the Training page refuses a launch with the exact
command instead of failing seconds later inside SB3.

### Settings

**Appearance** (theme, density, motion, radius, glow/grid),
the **Layout studio** (move/span/hide cards, reset one page or
everything), **Presets** (save/apply/delete), then project and output
roots, plus the machine-local **Godot executable**:
*Verify & save* probes the given path with the same runtime check the
benchmark uses and only then remembers it in
`.sandboxai/settings.json`, so training, benchmarks and evaluations all
resolve a binary that was actually seen working. The page also shows
what the current resolution chain (explicit path →
`GODOT_PATH`/`GODOT_EXECUTABLE` → remembered setting → PATH) resolves
to right now.

## Honesty rules

- No measurement is ever invented; every number shown was produced by
  the engine-backed measurers or read from run artifacts.
- Unavailable values render `n/a` (or the reason), never an estimate.
- Actions that a backend cannot honour are disabled with the reason, not
  silently faked.
- The training path never imports the GUI; the GUI is a pure operator
  over the adapter.

## Testing

The Tk-free layers are tested display-free:
`control_center_viewmodel.py` (`test_control_center_viewmodel.py`),
`control_center_theme.py` and `control_center_layout.py`
(`test_control_center_theme_layout.py`), the page/board wiring contract
(`test_control_center_pages_static.py`, source-level so it also runs
without Tkinter), plus `test_agents.py`, `test_adapter.py` and
`test_benchmark_pipeline.py`. The Tk window itself is exercised by
`python/tests/test_control_center_desktop.py` against a real adapter and
throwaway project — those tests skip where Tkinter or a display is
unavailable (environment facts, not regressions); the CI job
`desktop-ui-tests` runs them.

`python3 tools/desktop_tests.py` is the one command that runs whatever the
machine can actually run: the real-Tk suite when Tkinter and a display (or
`xvfb-run`) are present, and otherwise the static contracts plus the smoke
harness below, with the exact package to install for the real thing.
`--strict` fails instead of falling back, which is what CI uses - the
`desktop-ui-tests` job calls that script rather than pytest directly, so
the runner is exercised on every run.

Where neither Tkinter nor a display exists, `python3
tools/control_center_smoke.py` constructs the real application against a
small fake `tkinter` and drives every page, theme, density and motion
level, the layout studio, the presets, and the Training and Benchmark plan
logic. It also fails if the rail grows a collapse state, a shell switcher
or a Settings layout picker again, and it asserts each nav button names
its page instead of a two-letter code for it. It drains the background
queue after every page, so a poll completion callback that raises fails the
run instead of being printed and forgotten. It proves "no name errors, no bad wiring, no
constructor crashes, no failing completion callback" and nothing about
pixels, geometry or event dispatch — the `desktop-ui-tests` CI job runs it
next to the real-Tk pytest file, and it is a development aid, not a
substitute for that suite.

**What the smoke harness cannot see.** It proves the window builds; it
cannot say whether it *looks* right, and "the text is cut off" is a report
only a human can make - badly, over chat. `python3
tools/control_center_ui_report.py` opens the real window instead, walks
every page and writes `.sandboxai/ui_report/report.md` (plus the same
findings as `report.json`) with the things a screenshot does not explain:
labels that need more room than they have, text under the 11 px floor, table
headers wider than their column, buttons that are disabled and why, the
theme-listener count across three density changes, what a `refresh()` costs
per page, and whether any widget still carries a foreign theme bus. Where
Pillow can reach the display it saves one PNG per page; where it cannot, the
report says so instead of inventing an image. It needs Tkinter and a display,
it is a measuring instrument rather than a gate, and it always exits `0`
unless the prerequisites are missing (`2`) or the window failed to build
(`1`).

**Polling.** One timer drives the window (600 ms). Each tick refreshes the
**visible page only** - every page's `refresh()` submits background reads
(run directories, benchmark history, the replay list), so running all seven
would keep reading artifacts for a window nobody is looking at. The headless
smoke harness pins that behaviour by counting `refresh()` calls per page and
fails if a hidden page is polled; the adapter's Godot-runtime probe is cached
(30 s) because it spawns a subprocess.

What a tick costs is measured, not assumed: `tools/control_center_poll_probe.py`
builds a synthetic run tree (including the large logs a real run leaves
behind) and times every call a page's `refresh()` submits. That probe found
the two costs that mattered: line-counting every log on every poll, and
parsing a trailing megabyte of log to return the last few rows. Both are
fixed where they were introduced (`run_inspection`), so every caller
benefits. Measured warm on a 60-run corpus whose three newest runs carry a
64 MB log: `dashboard_snapshot` 68.1 ms -> 4.3 ms, `discover_checkpoints`
144.8 ms -> 16.0 ms, `list_runs` 140.3 ms -> 14.5 ms. Tables fed from a poll
also compare a signature of their rendered values and leave an unchanged
table alone, which keeps the operator's row selection instead of dropping it
every tick.

Three source-level checks in `test_control_center_pages_static.py` are
worth knowing about: every `ttk` style a widget asks for must be one the
theme module configures (an unknown style is painted with the default look
and is otherwise invisible); every `hasattr(self.adapter, "...")` guard
must name a real adapter method (a typo there silently disables a button
on every machine); and a class that inherits from a Tk widget must not
assign an instance attribute that shadows a name Tk already defines —
`self._options = (...)` on a `Canvas` subclass, for example, breaks widget
construction with `TypeError: 'tuple' object is not callable` inside
tkinter, which no headless check notices.
