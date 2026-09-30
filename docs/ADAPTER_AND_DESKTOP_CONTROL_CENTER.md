# Adapter and desktop Control Center

The Python Control Center is separate from the existing Godot operator scene.
`sandboxai.adapter.SandboxAIAdapter` is the application boundary: it uses
`TrainingConfig`, the existing CLI, run inspection, JSONL telemetry, benchmark
artifacts, and training's cooperative `RunControl` files. It does not own RL
logic or parse console prose.

## Data ownership

* live state: `status.json`, `events.jsonl`, and managed process snapshots
* historical runs: `run_inspection.py` reports under `training/runs`
* checkpoints: each run's `checkpoints/*.zip` inventory
* evaluation: `evaluations/*/summary.json` and `evaluations/latest.json`
* profiling: `logs/training_profile.json`
* static configuration: serialized `TrainingConfig`

Training stop writes `command.json` and uses the existing safe callback
boundary. Benchmark/evaluation cancellation terminates gracefully and only
uses a kill after a bounded timeout. stdout and stderr are drained on daemon
threads, so pipe back-pressure cannot freeze a UI.

Launch with `sandboxai control-center-desktop`. It uses Tkinter from the
standard library (Windows-first, no paid/cloud dependency), polls the adapter
without blocking Tk, and renders only data returned by the adapter.
