# Control Center architecture audit

Audited before implementation against `PROJECT.md`, `python/sandboxai`, the
Godot bridge, existing run-inspection code, and the existing Godot Control
Center scene.

| Concern | Existing owner / contract | GUI exposure |
|---|---|---|
| CLI | `sandboxai.cli`, argparse; train/resume/evaluate/benchmark/inspect-runs | adapter builds existing commands |
| TrainingConfig | `config.TrainingConfig`, validated dataclass, `to_dict/save/from_dict` | core fields first; advanced options remain in config |
| Training | `ppo.train_ppo` via `sandboxai train`; optional `RunControl` callback | async managed subprocess |
| Stop/pause | `run_control.py`, atomic `command.json`, safe callback boundaries | adapter writes stop request; no hard kill for training |
| Evaluation | `evaluation.evaluate_model`; checkpoint and summary artifacts | async CLI evaluator |
| Benchmark | `benchmark.benchmark_simulation`; measured Godot bridge data | async CLI benchmark |
| Runs | `run_inspection.py`; `config.json`, manifests, summaries, status, logs | read-only reports |
| Checkpoints | run `checkpoints/*.zip`, `latest`, `best_eval`, `final` | inventory, never guessed |
| Telemetry | `logs/training.jsonl`, `events.jsonl`; `telemetry.py` resources | bounded JSONL tail plus machine snapshot |
| Profiling | `logs/training_profile.json`, `TrainingProfiler` | explicit separate profile document |
| Worker config | `env_workers` / resolved workers in `TrainingConfig`; sharded bridge | editable environment and worker counts |
| IPC | Godot JSONL stdio (`GodotProcessTransport`) | not duplicated or parsed by GUI |

Training and bridge requests are synchronous internally, but the CLI process
is asynchronous from the desktop's point of view. Evaluation and benchmark
are similarly long-running processes. `ProcessManager` drains both output
pipes on daemon threads, polls process/backend state, and closes remaining
processes on application exit. This avoids orphaned child processes and pipe
fill deadlocks.

The repository already has a Godot in-simulator operator GUI. It remains
unchanged and is not suitable as the requested desktop application because it
is coupled to the simulator scene lifecycle. The new Tk frontend is a local
engineering desktop application, while `SandboxAIAdapter` is independently
usable and testable without Tk, Godot, or a display.

No new telemetry format was introduced. The only new application state is the
process record (in memory) and the existing run-control files passed to the
CLI. Historical, live, benchmark, evaluation, profile, and static config data
remain separate.
