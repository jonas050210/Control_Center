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

`control_center_desktop.py` builds the window and navigation;
`control_center_pages.py` holds the pages; `control_center_widgets.py`
provides the reusable infrastructure (background runner, bounded log
panel, tables, charts, tooltips); `control_center_viewmodel.py` is the
Tk-free presentation logic, tested without a display. Every page reaches
the system exclusively through `sandboxai.adapter.SandboxAIAdapter` —
the same `train`/`resume`/`benchmark`/`evaluate` CLI commands and on-disk
artifacts any shell user would use, never a parallel implementation. The
agent lifecycle (launch/pause/resume/stop/restart bookkeeping) lives in
`sandboxai.agents` on top of the adapter's process manager. See
[ADAPTER_AND_DESKTOP_CONTROL_CENTER.md](ADAPTER_AND_DESKTOP_CONTROL_CENTER.md)
for the adapter contract.

## Pages

### Dashboard

Live state of the newest run and the agents launched this session:
lifecycle state, run id, progress, steps/s, elapsed/ETA, environment and
worker counts, an agent lifecycle summary (coloured red when anything
failed), device, and reward. Stale status is labelled with its evidence,
exactly as `run_inspection.py` reports it.

### Agents

The operational core: the launch form plus the agent registry in one
place.

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
- **Agent table** — every training/benchmark/evaluation agent launched
  this session with name, kind, lifecycle, PID, environment/worker
  topology, device, steps, progress, steps/s, reward and errors. All
  values are backend-published facts; anything a kind does not publish
  renders `n/a`, never a guess.
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

The staged **benchmark pipeline** that measures what this machine's
runtime can actually do, then recommends a configuration.

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

**Budgets:** *Steps* (screen until a step count per configuration) or
*Time* (1–60 minutes total, split across stages, candidate grid thinned
to fit). Planning estimates size the slices; only measured values are
reported.

**Result:** the *Recommended Configuration* card (config, expected
steps/s, basis, rationale, warnings) with an **Apply** action that fills
the Agents launch form, and a *Custom configuration* card validated by
the same compatibility check the launcher enforces. The full report is
persisted under `training/benchmarks/pipelines/<timestamp>/`
(`pipeline.json` + benchmark-history-shaped `benchmark.json`), the
recommendation machine-locally in `.sandboxai/recommended_config.json`,
and past runs are listed from the history.

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

Project and output roots and the Godot executable, with the hardware
profile surfaced where one is persisted.

## Honesty rules

- No measurement is ever invented; every number shown was produced by
  the engine-backed measurers or read from run artifacts.
- Unavailable values render `n/a` (or the reason), never an estimate.
- Actions that a backend cannot honour are disabled with the reason, not
  silently faked.
- The training path never imports the GUI; the GUI is a pure operator
  over the adapter.

## Testing

`control_center_viewmodel.py` and the layers below it are tested
display-free (`python/tests/test_control_center_viewmodel.py`,
`test_agents.py`, `test_adapter.py`, `test_benchmark_pipeline.py`,
`test_control_center_isolation.py`). The Tk window itself is exercised by
`python/tests/test_control_center_desktop.py` against a real adapter and
throwaway project — those tests skip where Tkinter or a display is
unavailable (environment facts, not regressions).
