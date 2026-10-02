# Adapter and desktop Control Center

The Control Center is the Python desktop application (`python3 main.py` or
`sandboxai control-center-desktop`, documented in
[`CONTROL_CENTER.md`](CONTROL_CENTER.md)); the rendered in-simulator operator
scene was removed with the headless-only focus. `sandboxai.adapter.SandboxAIAdapter`
is the application boundary between the desktop GUI and the rest of the
project: it builds and launches the existing `sandboxai train` /
`sandboxai benchmark` / `sandboxai evaluate` CLI commands, and reads the
run/checkpoint/evaluation/benchmark artifacts those commands already produce.
It does not own RL logic, reimplement PPO/evaluation/benchmark math, or parse
console prose.

```
Desktop Control Center (Tk, control_center_desktop.py)
    -> control_center_viewmodel.py   (pure formatting/validation, no I/O)
    -> SandboxAIAdapter               (adapter.py)
        -> existing CLI / training / evaluation / benchmark infrastructure
            -> Godot workers
```

## Adapter surface (`python/sandboxai/adapter.py`)

* **System status** - `system_status()` reports CPU/RAM/process-RSS
  (`telemetry.resource_snapshot`), Python version, and real Godot detection
  through `runtime_validation.RuntimeValidator` + `config.find_godot_executable`
  (the same resolution order `sandboxai validate-runtime` uses), plus which
  optional dependencies (`torch`, `gymnasium`, `stable_baselines3`,
  `tensorboard`, `psutil`) are actually importable. Nothing here is guessed:
  an unavailable Godot binary reports `godot_available: false` and
  `godot_version: null` rather than a placeholder.
* **Dashboard** - `dashboard_snapshot()` returns the newest run's full
  `run_inspection.inspect_run()` report plus the live process registry, in
  one cheap call (run counting is a directory listing, not a second full
  report build).
* **Live telemetry** - `telemetry_series(run, max_points)` incrementally
  tails `logs/training.jsonl` (`telemetry.IncrementalJsonlTailer`, byte-offset
  based) into bounded `deque`s, one per chartable metric
  (`adapter.SERIES_KEYS`): timesteps, steps/second, ETA, episode/reward/combat
  stats, CPU/RSS/CUDA/GPU fields already in `telemetry.resource_snapshot`, and
  PPO optimizer diagnostics (`n_updates`, `approx_kl`, `clip_fraction`,
  `explained_variance`, `entropy`, `value_loss`, `policy_gradient_loss`,
  `loss`) written by `ppo.PPOStatsCallback`. Repeated polling only costs what
  was appended since the previous call; nothing is ever fully re-read or
  re-parsed.
* **Runs & checkpoints** - `list_runs()` / `inspect_run()` reuse
  `run_inspection.py` verbatim. `discover_checkpoints()` flattens every run's
  checkpoint inventory (`latest`/`best`/`final`) across the whole output root
  for a single-page picker.
* **Evaluations** - `discover_evaluations()` indexes every
  `evaluations/latest.json` and checkpoint-battery `step_*/summary.json`;
  `evaluation_detail()` returns one summary's full structured content
  (win/loss/timeout rate, kills/deaths/damage, accuracy, shots
  fired/hit, `policy_shoot_request_rate`, and the `action_pipeline` block from
  `action_audit.py` used for zero-shot/action-head diagnostics).
* **Benchmarks** - `benchmark_results()` / `benchmark_history()` read
  `benchmark.json` and reuse `benchmark.summarize_scaling()` for the
  best-environment-count / diminishing-returns / worker-speed-up heuristics -
  no scaling math is duplicated in the GUI.
* **Process registry (`ProcessManager`)** - every `train`/`benchmark`/
  `evaluate` subprocess is launched non-blocking, with stdout/stderr drained
  on daemon threads into a bounded, sequence-numbered buffer.
  `log_since(process_id, stdout_after, stderr_after)` returns only new lines
  plus a truncation flag, so a log panel never re-reads or re-renders what it
  already has. `cancel()` is non-blocking: training gets the existing
  cooperative `command.json` stop request; anything else gets a graceful
  terminate. `force_stop()` escalates to killing the process **and its
  discoverable children** (`psutil`, optional - without it only the direct
  child is signalled, and that limitation is not hidden). Finished process
  records are pruned to a bounded retention window.
* **Launching work** - `start_training(TrainingConfig | dict)`,
  `start_benchmark(...)`, `start_evaluation(checkpoint, ...)` validate their
  inputs (`ValueError`/`FileNotFoundError`) *before* spawning a process, and
  attach a small `meta` summary (env/worker counts, device, checkpoint, ...)
  so the Training/Dashboard pages never need to re-parse a command line.

## Hardware wizard (first-start device comparison)

`sandboxai.hardware_profile` is the **single** implementation of the
device-comparison measurement the first-start wizard needs; neither the
Godot operator scene nor the Tk desktop Control Center may grow a second
one. It compares three candidates — **CPU** (updates and inference on the
CPU), **Hybrid** (`device="cuda"`, `inference_device="cpu"`: GPU updates,
CPU inference, no per-step host↔device transfer) and **CUDA** (both on the
GPU) — but only offers Hybrid and CUDA when a CUDA device is actually
present, so a CPU-only host never sees a permanently-unavailable row.

Each candidate is measured by timing a short real PPO training slice
through `ppo.train_ppo` (default ~5,000 steps), so it needs a working
Godot bridge and **never invents a throughput**: a device that cannot be
measured is recorded with an explicit status (`unavailable` / `failed` /
`cancelled`) and no number. Measurement honours cancellation between
candidates (an in-flight slice always finishes), and the selected profile
is persisted to `.sandboxai/hardware_profile.json`. When nothing can be
measured (no engine, or every slice failed) the profile falls back to CPU
defaults and records `fallback: true` with a human-readable note.

The adapter exposes this through `hardware_candidates()`,
`hardware_profile()` and `run_hardware_wizard(...)` (with optional
`cancel`/`on_progress`/`steps` hooks); `control_center_viewmodel`
provides the Tk-free presentation (`hardware_profile_view`,
`hardware_measurement_rows`), which hides accelerator fields the host
cannot fill. `HardwareProfile.config_overrides()` yields the
`{device, inference_device}` pair a training launch should adopt.

The same measurement is reachable from a shell (useful for headless hosts
and CI, and the escape hatch when no GUI is running):

```bash
sandboxai hardware-wizard --godot-executable godot   # measure and persist
sandboxai hardware-wizard --show                     # print the saved profile
sandboxai hardware-wizard --no-save --steps 2000     # a quick, non-persisting run
```

It exits non-zero when it had to fall back to CPU defaults, so a script
can tell a real measurement from a fallback.

## Data ownership (unchanged)

* live state: `status.json`, `events.jsonl`, `logs/training.jsonl`, and
  managed process snapshots
* historical runs: `run_inspection.py` reports under `training/runs`
* checkpoints: each run's `checkpoints/*.zip` inventory
* evaluation: `evaluations/*/summary.json` and `evaluations/latest.json`
* profiling: `logs/training_profile.json`
* static configuration: serialized `TrainingConfig`

## Presentation layer (`python/sandboxai/control_center_viewmodel.py`)

Every dict-shaping, formatting, and form-validation rule the GUI needs lives
in this Tkinter-free module (`dashboard_view`, `runs_table_rows`,
`process_table_rows`, `agent_table_rows`/`agent_action_availability`,
`launch_slot_view`/`topology_rows`,
`evaluation_view`/`evaluation_comparison_rows`, `benchmark_history_rows`,
the benchmark-workflow views (`benchmark_workflow_view`,
`benchmark_pipeline_rows`, `benchmark_recommendation_view`,
`pipeline_progress_view`),
`parse_training_form`, `format_*` helpers,
`downsample_series`). It is unit-tested directly
(`python/tests/test_control_center_viewmodel.py`) without constructing a Tk
window, and the desktop file (`control_center_desktop.py`) only calls into
it - so the same rule (e.g. "a status this old with no tracked process is
stale", or "a blank Godot-executable field means: use the configured
default") cannot silently diverge between pages.

## Desktop GUI (`python/sandboxai/control_center_desktop.py`)

Launch with `python3 main.py` from the repository root, or
`sandboxai control-center-desktop` (optionally
`--project-path`/`--output-root`; see
`tools/windows/start_control_center.bat` for a Windows launcher). It uses
Tkinter from the standard library only (Windows-first, no paid/cloud
dependency) and never renders the game. Pages: **Dashboard**, **Training**,
**Benchmarks**, **Evaluations**, **Runs / Checkpoints**,
**System / Telemetry**, **Settings**. The shell has three layouts (rail,
topbar, command board), five themes and three density presets, all applied
live and persisted in `.sandboxai/ui/preferences.json`; pages that declare
widgets expose a layout board whose card order, span and visibility are
editable in **Settings -> Layout studio** and savable as a preset under
`.sandboxai/ui/presets/`.

* All adapter calls run on a small background thread pool
  (`BackgroundRunner`); results are handed back to the Tk thread through a
  queue drained on a timer, so no adapter call - including subprocess
  launch/cancel and disk reads on a large run directory - blocks the GUI
  thread.
* A page only submits background work and applies the result; it never
  blocks waiting for it. Periodic reads are coalesced per page operation, so
  a slow run-directory/process-status scan cannot fill the shared worker pool
  with duplicate polls. If a refresh overlaps a live operation, only its
  latest request is retained for one follow-up when that operation completes;
  this preserves freshness without creating a queue. Switching output roots
  clears guards from the retired runner. Results that belong to a replaced
  run/process/selection are rejected instead of being rendered as current
  telemetry.
* `LogPanel` bounds the Text widget itself (old lines are deleted past a
  cap) independently of the adapter's own bounded buffer, distinguishes
  stdout from stderr (red), exposes both scroll axes for unwrapped commands
  and tracebacks, and supports pausing auto-scroll without pausing polling.
  Wheel/scrollbar/keyboard reading retains the operator's position; it resumes
  follow only after the newest output is visible.
* Dense inventory tables keep every column reachable with *overlay*
  scrollbars and fit-to-width columns instead of pinning two native
  scrollbars under every table; the log panel wraps by default (the wrap
  toggle restores unwrapped tracebacks with the overlay bar). Training
  captures a process selection and its incremental log cursors per request,
  coalesces a slow poll, and discards a late result for an old selection; a
  scoped Stop/Force Stop request captures the clicked process id before the
  worker starts. Evaluation comparisons and Runs/Checkpoints detail panes use
  the same generation guard, so an older disk read cannot replace the report
  for a newer selection.
* `LineChart` is a small dependency-free Tk Canvas widget; it redraws from
  the adapter's already-bounded series, decimated again to the canvas width
  (`control_center_viewmodel.downsample_series`), so render cost does not
  grow with run length.
* Agent launch validates input through
  `control_center_viewmodel.parse_training_form` (backed by
  `TrainingConfig.validate()`) plus the shared runtime-compatibility check
  (`benchmark_pipeline.validate_configuration`) before ever touching the
  adapter, so an invalid combination never reaches a subprocess.
* Lifecycle actions (Pause/Resume/Stop/Restart/Force stop/Remove/Stop all)
  go through `sandboxai.agents.AgentManager`, which derives operator-facing
  states from the OS process, the trainer's `status.json` and the requested
  action, restarts from the run's newest checkpoint, and refuses actions a
  backend cannot honour with the reason. The Training page only ever acts on
  agents this Control Center itself launched; there is no arbitrary command
  execution surface.
