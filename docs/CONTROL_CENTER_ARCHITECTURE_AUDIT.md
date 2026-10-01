# Control Center architecture audit

Audited against `PROJECT.md`, `python/sandboxai`, the Godot bridge, existing
run-inspection code, and the (since removed) Godot Control Center scene
before and during the desktop Control Center's implementation. The
architecture below was kept unchanged from the first version; only the
depth of what the adapter/GUI expose grew. The desktop application audited
here is now the only Control Center: the rendered operator scene it was
contrasted with no longer exists.

| Concern | Existing owner / contract | GUI exposure |
|---|---|---|
| CLI | `sandboxai.cli`, argparse; train/resume/evaluate/benchmark/inspect-runs | adapter builds the existing `train`/`benchmark`/`evaluate` commands (`--config`/`--env-counts`/`--worker-counts`/`--checkpoint`, ...) |
| TrainingConfig | `config.TrainingConfig`, validated dataclass, `to_dict/save/from_dict/validate` | Training page: basic/advanced field groups (`control_center_viewmodel.TRAINING_FIELDS`) parsed and validated through the real dataclass before launch |
| Training | `ppo.train_ppo` via `sandboxai train`; optional `RunControl` callback | async managed subprocess (`ProcessManager`), live status.json + `logs/training.jsonl` polling |
| PPO diagnostics | `ppo.PPOStatsCallback` relays SB3's own `train/*` logger values (`approx_kl`, `clip_fraction`, `explained_variance`, `entropy`, `value_loss`, `n_updates`, ...) into `logs/training.jsonl` and `status.json` | Dashboard PPO-diagnostics card + telemetry chart; nothing recomputed |
| Stop/pause | `run_control.py`, atomic `command.json`, safe callback boundaries | `ProcessManager.cancel()` writes the stop request non-blocking; `force_stop()` is an explicit, confirmed escalation that kills the process tree (optional `psutil`) |
| Evaluation | `evaluation.evaluate_model`, `action_audit.py`; checkpoint and summary artifacts | async CLI evaluator; structured results incl. win/loss/timeout, combat, accuracy, and `action_pipeline` zero-shot/action-head diagnostics; multi-select comparison table |
| Benchmark | `benchmark.benchmark_simulation`, `benchmark.summarize_scaling`; measured Godot bridge data | async CLI benchmark; result history flattened across sweeps with the same `summarize_scaling` heuristic, never a GUI-side reimplementation |
| Runs | `run_inspection.py`; `config.json`, manifests, summaries, status, logs | read-only reports; Runs/Checkpoints page reuses `inspect_run`/`inspect_runs` directly |
| Checkpoints | run `checkpoints/*.zip`, `latest`, `best_eval`, `final` | `discover_checkpoints()` inventory across every run, never guessed |
| Telemetry | `logs/training.jsonl`, `events.jsonl`, `status.json`; `telemetry.py` resources | `telemetry.IncrementalJsonlTailer`-backed bounded series (`adapter._RunSeries`/`SERIES_KEYS`); machine snapshot via `resource_snapshot()` |
| Godot detection | `runtime_validation.RuntimeValidator`, `config.find_godot_executable` | System/Telemetry page and every run manifest (`manifest.godot_snapshot`) use the same real probe - see "Bug fixed" below |
| Profiling | `logs/training_profile.json`, `TrainingProfiler` | `adapter.profiling()`; explicit separate profile document, not merged into telemetry |
| Worker config | `env_workers` / resolved workers in `TrainingConfig`; sharded bridge | Training page basic field; Dashboard/Agents show resolved worker count from `status.json`/`config.json`, never recomputed in the GUI |
| Process registry | none previously (single "active process" scalar) | `ProcessManager` tracks every launched process (training/benchmark/evaluation) with bounded finished-record retention; Agents page lists all of them with real PID, never arbitrary command execution |
| IPC | Godot JSONL stdio (`GodotProcessTransport`) | not duplicated or parsed by GUI |

## Bug fixed during this pass

`manifest.py::godot_snapshot()` imported a class that did not exist
(`GodotRuntimeValidator`); the broad `except Exception` around it silently
swallowed the resulting `ImportError`, so **every run manifest ever
produced recorded `godot.version: null`**, even on a machine with a working
Godot install. Fixed to import the real `runtime_validation.RuntimeValidator`
class. Covered by a new regression test
(`test_manifest.py::test_godot_snapshot_reports_the_probed_version_when_available`)
that exercises the success path, which no existing test had covered.

## Process model

Training, benchmark, and evaluation subprocesses are asynchronous from the
desktop's point of view. `ProcessManager` drains both output pipes on daemon
threads into a bounded, sequence-numbered buffer (`log_since()` serves only
new lines to a poller), polls process/backend (`status.json`) state, and
closes any still-running processes on application exit
(`ProcessManager.close()`, the one remaining *blocking* method, used only at
shutdown). This avoids orphaned child processes and pipe-fill deadlocks.
`cancel()`/`force_stop()` themselves are non-blocking so a GUI button click
never waits on subprocess I/O; `force_stop()` also reaches a subprocess's own
child processes (e.g. `sandboxai benchmark`'s Godot workers) when `psutil` is
installed, not just the direct child `Popen` handle.

## Presentation layer

`control_center_viewmodel.py` is the GUI's Tkinter-free presentation layer:
every dict-shaping/formatting/validation rule the desktop app needs
(`dashboard_view`, `runs_table_rows`, `process_table_rows`, `evaluation_view`
/`evaluation_comparison_rows`, `benchmark_history_rows`,
`parse_training_form`, chart decimation, number/duration/percent
formatting) lives there and is unit-tested without constructing a Tk window
or an adapter. `control_center_desktop.py` only calls into it, so a page
cannot silently reinterpret a value differently than another page does.

The repository already has a Godot in-simulator operator GUI. It remains
unchanged and is not suitable as the requested desktop application because it
is coupled to the simulator scene lifecycle. The Tk frontend is a local
engineering desktop application; `SandboxAIAdapter` and
`control_center_viewmodel` are independently usable and testable without Tk,
Godot, or a display.

No new telemetry format was introduced. The only new application state is the
in-memory process registry (`ProcessManager`) and its bounded per-run
telemetry-series cache (`adapter._RunSeries`); the existing run-control files
passed to the CLI are unchanged. Historical, live, benchmark, evaluation,
profile, and static config data remain separate, on-disk, and owned by the
same modules as before this pass.
