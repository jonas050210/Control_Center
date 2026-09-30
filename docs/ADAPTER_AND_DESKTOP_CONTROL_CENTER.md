# Adapter and desktop Control Center

The Python Control Center is separate from the existing in-simulator Godot
operator scene (`sandboxai control-center`, documented in
[`CONTROL_CENTER.md`](CONTROL_CENTER.md)). `sandboxai.adapter.SandboxAIAdapter`
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
  so the Agents/Dashboard pages never need to re-parse a command line.

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
`process_table_rows`, `evaluation_view`/`evaluation_comparison_rows`,
`benchmark_history_rows`, `parse_training_form`, `format_*` helpers,
`downsample_series`). It is unit-tested directly
(`python/tests/test_control_center_viewmodel.py`) without constructing a Tk
window, and the desktop file (`control_center_desktop.py`) only calls into
it - so the same rule (e.g. "a status this old with no tracked process is
stale", or "a blank Godot-executable field means: use the configured
default") cannot silently diverge between pages.

## Desktop GUI (`python/sandboxai/control_center_desktop.py`)

Launch with `sandboxai control-center-desktop` (optionally
`--project-path`/`--output-root`; see
`tools/windows/start_control_center.bat` for a Windows launcher). It uses
Tkinter from the standard library only (Windows-first, no paid/cloud
dependency). Pages: **Dashboard**, **Training**, **Agents**, **Benchmarks**,
**Evaluations**, **Runs / Checkpoints**, **System / Telemetry**, **Settings**.

* All adapter calls run on a small background thread pool
  (`BackgroundRunner`); results are handed back to the Tk thread through a
  queue drained on a timer, so no adapter call - including subprocess
  launch/cancel and disk reads on a large run directory - blocks the GUI
  thread.
* A page only submits background work and applies the result; it never
  blocks waiting for it.
* `LogPanel` bounds the Text widget itself (old lines are deleted past a
  cap) independently of the adapter's own bounded buffer, distinguishes
  stdout from stderr (red), and supports pausing auto-scroll without pausing
  polling.
* `LineChart` is a small dependency-free Tk Canvas widget; it redraws from
  the adapter's already-bounded series, decimated again to the canvas width
  (`control_center_viewmodel.downsample_series`), so render cost does not
  grow with run length.
* Training start/stop, benchmark launch, and evaluation launch all validate
  input through `control_center_viewmodel.parse_training_form` (backed by
  `TrainingConfig.validate()`) or explicit range checks before ever touching
  the adapter, so an invalid combination never reaches a subprocess.
* The Agents page only offers scoped cancel/force-stop on processes this
  Control Center itself launched; there is no arbitrary command execution
  surface.
