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
(`pip install -e ".[train]"`) — the GUI reports exactly what is missing
on the System page instead of guessing.

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
| Shell layout | Rail, Topbar, Command Board | Where navigation lives and how much width the content gets |
| Density | Comfort, Compact, Ultra | Row heights, paddings and one font step (never below the 11 px floor) |
| Motion | Off, Reduced, Normal, Cinematic | Speed of the small transitions; **Off** renders final states immediately |
| Accent | nine swatches or any `#rrggbb` | Replaces the accent of whichever theme is active; *Theme accent* restores the designed one |
| Corner radius, glow, grid | slider / toggles | Card rounding and the optional backdrop effects |

The accent is one colour, not a sixth theme: it is stored in the
preferences (`"accent": "#22d3ee"`) and every theme wears it, so an
operator can keep the Corz surfaces with a mint accent. The label colour
*on* the accent is derived (`Theme.with_accent`), never guessed: white
while it keeps the 3:1 contrast a bold UI label needs, dark ink
otherwise, and picking a theme's own accent is a no-op so the designed
pair (Cyan with its deep-teal ink) is preserved exactly.

A theme change only repaints (ttk styles, canvas colours and the table tag
roles are re-applied). Every widget that draws itself subscribes to the
window's `ThemeBus`; a widget constructed without one would silently fall
back to the module's default palette and keep Corz colours forever, so the
contract is checked twice - the smoke harness and the real-Tk suite each
walk every page and fail if any widget carries a foreign bus. Helper
widgets find the nearest bus by walking up their parents (`_cc_bus`), which
keeps a tooltip or a chart correct even when a call site forgets `bus=`. A **density** change has to rebuild the widgets so
the new paddings fit; that rebuild captures what the operator was looking
at first — the log text and its cursors, the selected rows, the page's
scroll offset — and restores it on the fresh widgets, so the log does not
blank out and the reading position survives.

**Movable cards.** Pages that declare widgets (Dashboard, Training,
Benchmarks) are arranged by a layout board. On **Settings -> Layout
studio** every card can be moved up/down, given a 1x/2x/3x span and
hidden (cards marked `removable=False`, such as the launch deck, stay
visible). The result applies to the running window immediately.

**Presets.** A preset stores the theme, shell, density, motion and every
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

The layout state is also persisted without a preset, so closing and
reopening the window keeps the arrangement. If `tkinter` is missing
entirely, `main.py` still starts, prints the exact fix (`python3-tk`, or
a Python build with Tk) and exits cleanly instead of raising a stack
trace.

## Keyboard

`Ctrl+K` opens the command palette, `Ctrl+1..7` jump straight to a page,
`Escape` closes the palette, and `Up`/`Down` move its selection. The palette
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

### Training

The operational core: the launch deck plus the training-run registry in
one place.

- **Launch form** — basic and advanced fields over the same
  `TrainingConfig` the CLI validates. The launch slot under it shows the
  resolved configuration before anything starts: environment count,
  resolved worker count (`0 = auto` becomes the concrete number), the
  shard topology (`12+12+12+12`), device and step count — or every reason
  the current values are invalid (parse errors, worker/environment
  incompatibilities, a CUDA request on a host without CUDA). The verdict
  comes from `benchmark_pipeline.validate_configuration`, the same check
  the benchmark plan uses, so the launcher and the benchmark can never
  disagree.
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

The **automatic benchmark** with three explicit modes:

| Mode | Candidate ladders | When to use it |
| --- | --- | --- |
| **Auto** (default) | Host-scaled ladder from 64 up to 128 environment processes (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128) and worker counts from 1 to the host's own step — it deliberately probes past the conservative `--env-workers auto` recommendation, up to twice the physical-core estimate, capped at 32 | The normal case — one click, plan computed for this machine |
| **Push** | The Auto ladder plus the wide steps (96/128/192/256 envs), used to find where throughput saturates | Once per machine (or after a CPU/RAM change) to raise the ceiling |
| **Custom** | Explicit `Environments`, `Workers`, `Steps / config` or `or minutes` lists — the same lists the CLI accepts | Reproducing a specific sweep or scripting a comparison |

The plan is validated before anything starts: unparseable lists, an
impossible worker/environment pair or an empty candidate set disable
*Start benchmark* and state the reason. Auto distributes its time budget
across the planned configurations by itself; the mode line above the
button always shows the resulting configuration count.

The old grid capped the sweep at 64 environments and stopped the worker
ladder at *physical cores - 2*, which is why a large run could sit at
10-20 % CPU: the plan never asked for more than four workers on a machine
that could feed more. Auto now always reaches the 64 rung and up to 128 on
a large host, and its worker ladder includes the host's own step (up to 32)
instead of only the powers of two below the recommendation. Push and
Custom exist to measure the saturation point directly instead of guessing.

Start runs the complete staged pipeline. The tab shows the live phase strip
(discover → screen → devices → validate → pick → apply), the current
measurement, every tested configuration, and the winning configuration;
when the pipeline completes, the recommendation is persisted and
**applied automatically** to the launch configuration.

1. **Discovery** — probe the Godot executable/version, torch, CUDA, CPU
   count, RAM. No Godot binary means an honest "unavailable" report with
   the command to fix it, never a fabricated number.
2. **Screening** — the real bridge benchmark
   (`benchmark.benchmark_simulation`) across every planned
   `(environments, workers)` pair, each configuration time-capped:
   startup, warmup, throughput, p50/p95 vector-step latency, episodes,
   resources, errors.
3. **Devices** — only when more than one device candidate exists and no
   usable hardware profile is persisted; reuses `hardware_profile`
   (the single device-comparison implementation).
4. **Validation** — a short *real training slice* per surviving
   finalist, on the selected device, so the recommendation reflects the
   training path, not just bridge stepping.
5. **Recommendation** — from validated throughput when available, within
   a near-best band of the best measurement, preferring stability (low
   latency jitter, no errors, fewer workers) over an unstable peak. The
   `rationale` and `warnings` quote only measured numbers.

Auto and Push use the pipeline's own time budget split across stages; in
Custom mode the GUI passes the operator's explicit ladder and step or
minute budget straight through. Planning estimates size the slices; only
measured values are reported. The same budget/grid knobs remain
available on the CLI for scripted sweeps.

**Result:** the *Best configuration* card (config, expected steps/s,
basis, rationale, warnings). After a completed run the tab marks the
recommendation applied and mirrors it into the Training launch deck by
itself; a not-yet-opened Training page picks the applied recommendation
up when it is built. The full report is persisted under
`training/benchmarks/pipelines/<timestamp>/` (`pipeline.json` +
benchmark-history-shaped `benchmark.json`), the recommendation
machine-locally in `.sandboxai/recommended_config.json`, and the latest
persisted report is shown again on the next start.

The same pipeline runs from the shell:

```bash
sandboxai benchmark-pipeline --budget-mode time --minutes 15
sandboxai benchmark-pipeline --budget-mode steps --steps 2000
sandboxai benchmark-pipeline --show        # print the persisted recommendation
```

### Evaluations

Win/loss/timeout outcomes, combat and accuracy diagnostics,
action-head/zero-shot checks, and multi-run comparison over the
evaluations the CLI writes.

### Runs / Checkpoints

A browser over the on-disk run artifacts (state with evidence, progress,
checkpoint and evaluation inventory, log sizes, manifest provenance),
read through `run_inspection.py`.

### System / Telemetry

Real dependency and device status (Python, torch, Godot binary/version,
CPU/RAM) and bounded live telemetry charts. Unavailable metrics are
shown as such, never estimated.

### Settings

**Appearance** (theme, shell layout, density, motion, radius, glow/grid),
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
small fake `tkinter` and drives every page, theme, density, motion level,
shell layout, the layout studio, the presets, and the Training and
Benchmark plan logic. It also drains the background queue after every
page, so a poll completion callback that raises fails the run instead of
being printed and forgotten. It proves "no name errors, no bad wiring, no
constructor crashes, no failing completion callback" and nothing about
pixels, geometry or event dispatch — the `desktop-ui-tests` CI job runs it
next to the real-Tk pytest file, and it is a development aid, not a
substitute for that suite.

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
