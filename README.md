# SandboxAI

SandboxAI is a local, open-source calibration and reinforcement-learning
platform being narrowed to **verified Roblox TTK Testing** mechanics. Godot
remains the deterministic, headless-capable local test environment; it is not
a Roblox client or connection. The exact evidence boundary, calibration gaps
and explicitly excluded features are in
[docs/TTK_TESTING_REFERENCE.md](docs/TTK_TESTING_REFERENCE.md). Rendering is
not required by the Python trainer.

The repository now contains a real PPO and Behavior Cloning workflow in
addition to the original Godot foundation:

```
HumanController / policy
          |
          v
      RLAdapter  <---- JSON-lines stdio bridge ---->  Python Gymnasium/SB3
          |
          v
  SimulationManager -> EnvironmentCore -> Agent/Enemy/Weapon state
```

Godot owns simulation state and rewards. Python owns neural networks,
checkpoints, evaluation, datasets and telemetry. No paid API, cloud service
or visual-scene dependency is required, and nothing here automates the real
game: the optional TTK calibration helpers only detect, launch, focus and
screenshot the Roblox client for a human to measure, as specified in
[docs/TTK_TESTING_REFERENCE.md](docs/TTK_TESTING_REFERENCE.md).

## Requirements

- Godot **4.7.2** (standard build, GDScript only)
- Python **3.11+**
- Optional Python training dependencies: PyTorch, Gymnasium,
  Stable-Baselines3, TensorBoard and psutil
- Windows 11 or Ubuntu/Linux. CUDA is optional and detected through
  `torch.cuda.is_available()`; no GPU model is hardcoded.

The target RTX 4060 Ti 8 GB is appropriate for the compact MLP and structured
observations, but CPU mode is fully supported.

## Installation

Install Godot separately and make the executable available as `godot`, or
configure its location once with `--godot-executable`. The last verified
executable is remembered in `.sandboxai/settings.json` (gitignored), so
configuring it for one command — for example
`python -m sandboxai validate-runtime --godot-executable "C:\path\to\Godot.exe"`
— makes every later command (`train`, `resume`, `benchmark`, ...) use it
without repeating the flag. Resolution order: an explicit
`--godot-executable`, then the `GODOT_PATH`/`GODOT_EXECUTABLE` environment
variables, then the remembered setting, then `godot` on PATH. Delete
`.sandboxai/settings.json` to return to the PATH default.

Running from **WSL** (Windows Subsystem for Linux) works with either Godot
build. With the Linux Godot build nothing special is needed. With the
**Windows** Godot build — the usual case when the checkout lives on
`/mnt/<drive>` — point `--godot-executable` at the Windows binary, e.g.
`python -m sandboxai validate-runtime --godot-executable "/mnt/c/tools/Godot_v4.7.2-stable_win64_console.exe"`.
SandboxAI then converts the project path (and every other path handed to the
engine) to Windows form (`C:\...`) automatically, and if the direct `.exe`
launch is refused by the OS (`PermissionError`, broken `binfmt` interop
registration, ...) it retries through `cmd.exe /C call` before failing with
an error that lists every attempt. Windows-form executable paths
(`C:\...` — for example a setting remembered by a Windows-side run of the
same checkout) are translated with `wslpath` as well. Do not pass `cmd.exe`
itself as `--godot-executable`: it is rejected with guidance, because the
wrapper is applied automatically when needed. Reliable direct launches
require `[interop] enabled=true` in `/etc/wsl.conf`, the WSLInterop
`systemd-binfmt` registration, and the execute bit on the `.exe` as seen
from WSL.

Then from the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[training,test]"
```

For CUDA, install the PyTorch wheel matching the installed NVIDIA driver from
[pytorch.org](https://pytorch.org/) before installing the remaining extras.
The application detects CUDA at startup and fails clearly if `--device cuda`
is requested but unavailable.

## Run the Godot project

Open the root in Godot 4.7.2 and press **Play**, or run:

```bash
godot --path .
```

Environment 0 is human-controlled:

- `W/A/S/D`: move and strafe
- mouse or arrow keys: aim
- left mouse button or `Space`: shoot
- `Esc`: release/capture the mouse

The remaining default environments use the deterministic stub controller.
Human and AI controllers both produce the same `Action` representation used
by RL and recording.

## Control Center

The Control Center is the interactive front-end: one native desktop window
to launch, operate, benchmark and inspect the headless training stack. It
is a Python/Tkinter application and is **headless-only** — it never renders
the game; training always runs `scripts/rl/rl_server.gd` with `--headless`
in separate processes, and the GUI exchanges only cooperative commands and
read-only status files with them.

```bash
# from the repository root (no Godot editor, no .tscn):
python3 main.py

# ... or through the CLI:
sandboxai control-center-desktop
```

It provides:

- **Dashboard** - live state, progress, steps/s, elapsed/ETA, environment
  and worker counts, agent lifecycle summary, device, reward
- **Training** - the launch deck (validated against the same
  compatibility rules the benchmark uses) with environment count, worker
  count, device and a **Steps or Time** budget: a time-boxed run carries
  its minutes into the training config, so the trainer stops itself at its
  next safe step boundary and still saves the final checkpoint. Below it: every launched training run with
  its lifecycle state (AVAILABLE -> LAUNCHING -> RUNNING ->
  PAUSED/STOPPING -> STOPPED/FINISHED/FAILED), its Environment -> Worker
  topology, backend metrics, and the full action set: Launch,
  Pause/Resume, Stop, Restart (resuming from the run's latest
  checkpoint), Force stop, Stop all
- **Benchmarks** - one **Start Benchmark** action runs the automatic,
  host-scaled measurement pipeline: it checks runtime capabilities, compares
  environment/worker/device candidates, validates the strongest results, and
  applies the fastest stable recommendation to the launch deck. Live telemetry
  reports **Steps/s** (not a duplicate peak-FPS counter), latency, stability
  and progress; the measured configurations and full report remain available
  for review and are persisted. Advanced sweep controls remain available to
  scripted callers, not in the GUI.
- **Evaluations** - win/loss/timeout, combat, accuracy and
  action-head/zero-shot diagnostics plus multi-run comparison
- **Runs / Checkpoints** - a browser over the on-disk run artifacts
- **Stats** - every value the policy receives, decoded from a real
  recording: the three tracked contacts (enemy position, distance,
  bearing, health, visibility/source), the world objects around the agent,
  hearing, memory, the raw observation vector and the recorded action.
  The contract table is complete without a recording; a light replay is
  labelled as such instead of being rendered with zeros that would look
  like data
- **System / Telemetry** - real CPU/RAM/Godot/dependency status and
  bounded live telemetry charts; unavailable metrics are shown as such,
  never estimated
- **Settings** - theme, shell layout (rail/topbar/board), density,
  motion, a free **accent colour** (enter any `#rrggbb`) that any theme can
  wear, accent effects, the **Layout studio** (move, span and
  hide cards), savable **presets** that can also be exported to a file and
  imported back, project/output roots and the Godot executable

The desktop suite runs with `python3 tools/desktop_tests.py`: it uses the
real Tk window when the machine has Tkinter and a display (or `xvfb-run`),
and otherwise runs the static contracts plus the headless smoke harness and
prints the exact package to install. `--strict` fails instead of falling
back, which is what the CI job `desktop-ui-tests` does.

The Control Center is built entirely on
`sandboxai.adapter.SandboxAIAdapter` - the same `train`/`benchmark`/
`evaluate` CLI commands and on-disk artifacts, never a parallel
implementation - and stays **outside** the RL hot path: `scenes/main.tscn`
remains the main scene, and the training modules never import the GUI.

Full documentation: [docs/CONTROL_CENTER.md](docs/CONTROL_CENTER.md) and
[docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md](docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md).

## Automated tests

Godot tests:

```bash
godot --headless --path . --script res://tests/run_tests.gd
```

The runner discovers `tests/test_*.gd`, reports every test, and exits nonzero
on failure. Python tests run from the repository root; `conftest.py` puts
`python/` on `sys.path`, so no `PYTHONPATH` is needed with pytest:

```bash
python -m pytest -q
# or, without pytest:
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

Static lint (pyflakes-equivalent checks only, via `ruff`, install with
`pip install -e ".[lint]"`):

```bash
ruff check .
```

Both GitHub Actions workflows run on every push/PR:
`.github/workflows/godot-tests.yml` (Godot 4.7.2 headless suite on Windows
and Linux) and `.github/workflows/python-tests.yml` (`ruff`, a numpy-only
"core" job, and a "full" job with the training extras on Windows and Linux).

The Python suite also statically analyses and lints the GDScript half of the
repository (`python/tests/test_gdscript_static.py`). It parses every `.gd`
file with the real grammar and fails on the classes of defect a
type-checked language would catch at compile time: unknown `preload`
targets, wrong argument counts, unknown enum members, calls to undeclared
helpers, and calls to methods that do not exist on a typed local
(`var env := EnvironmentCore.new()` followed by `env.typo()`). That last
check matters most on machines without the engine, where it is the only
thing standing between a GDScript typo and a runtime failure. It requires
the optional `gdscript` extra:

```bash
pip install -e ".[test,gdscript]"
```

Without it those tests skip rather than fail, so a machine without
`gdtoolkit` still gets a green — but incomplete — run.

Godot is not bundled in this repository. If the executable is unavailable,
Python tests and static Python compilation can still run, but the Godot test
suite and live PPO bridge cannot be honestly marked as executed.

## CLI workflow

All normal workflows are exposed by one command. `python -m sandboxai` can be
used instead of the installed `sandboxai` executable.

```bash
sandboxai --help
sandboxai install
```

### Record human demonstrations

This opens the graphical Godot scene and records transitions from the real
`HumanController` pipeline. Close the window to save, or provide a duration:

```bash
sandboxai record --output training/datasets/human_demo.jsonl
sandboxai record --output training/datasets/demo.jsonl --duration 300
```

Direct Godot equivalent:

```bash
godot --path . --script res://scripts/recording/record_demo.gd -- \
  --output training/datasets/human_demo.jsonl
```

Each JSONL dataset starts with metadata and then stores compact records with:
`observation`, `action`, `next_observation`, `reward`, `done`, timestamp,
episode ID, environment ID and terminal info. The action keeps the two
continuous mouse-look values for lossless human logs; BC uses the fixed first
five fields.

Inspect and validate a dataset without PyTorch:

```bash
sandboxai inspect-dataset --dataset training/datasets/human_demo.jsonl
sandboxai inspect-dataset --dataset training/datasets/human_demo.jsonl --statistics
```

`--statistics` prints the full data-quality report: episode structure and
group count, per-component action histograms, duplicate-transition fraction,
observation-range violations, episode-boundary problems and the dataset
fingerprint. It is the same report BC writes to `dataset_report.json`.

### Open the Control Center

```bash
python3 main.py                # from the repository root
# or:
sandboxai control-center-desktop
```

This opens the Python/Tkinter desktop application described above. It never
renders the game: it launches the existing `train`/`benchmark`/`evaluate`
CLI commands as separate headless processes when you start one from the
GUI. See [docs/CONTROL_CENTER.md](docs/CONTROL_CENTER.md).

### Behavior Cloning

```bash
sandboxai bc-train \
  --dataset training/datasets/human_demo.jsonl \
  --epochs 25 --batch-size 512 --device auto \
  --output-dir training/bc_runs/human_v1
```

The result contains `latest.pt`, `best.pt`, periodic epoch checkpoints,
`metrics.jsonl`, `loss.csv`, `config.json` and `dataset_report.json`. The model
is a small two-hidden-layer PyTorch MLP with one categorical head per action
field. Validation reports component accuracy and exact six-field action
accuracy.

**The train/validation split is episode-aware.** Splitting demonstrations by
transition leaks: consecutive frames of one episode are nearly identical, so a
transition-level split reports a validation accuracy that mostly measures
memorisation of the training episodes.

```bash
sandboxai bc-train --dataset demo.jsonl --split-strategy episode   # hard requirement
sandboxai bc-train --dataset demo.jsonl --split-strategy auto      # default
sandboxai bc-train --dataset demo.jsonl --split-strategy transition
```

- `auto` (default) groups by `run_id|environment_id|episode_id` when the
  dataset has at least two groups, and otherwise falls back to the old
  transition shuffle while recording `degraded_reason` in the split report.
- `episode` refuses to train on a dataset without usable episode structure
  rather than leaking quietly.
- `transition` is the legacy behaviour and must be asked for explicitly.

`dataset_report.json` records the strategy, the train/validation transition and
group counts, `shared_groups`, a `leakage_free` flag, the seed and the dataset
fingerprint (`blake2b:<hex>`); every checkpoint carries the split and the
fingerprint, so a `.pt` file can always be traced to the exact corpus split
that produced it.

Resume BC training:

```bash
sandboxai bc-train --dataset training/datasets/human_demo.jsonl \
  --output-dir training/bc_runs/human_v1 \
  --resume-checkpoint training/bc_runs/human_v1/latest.pt --epochs 50
```

### PPO training

The trainer starts one Godot headless process containing the requested number
of independent environments and uses a `MultiDiscrete([3,3,3,3,2,2])` action
space. The structured observation is a 106-float `Box` (see
[`docs/OBSERVATION_ACTION_CONTRACT.md`](docs/OBSERVATION_ACTION_CONTRACT.md)
for the full field-by-field table, including the multi-enemy tracking
fields added for curriculum levels with more than one enemy).

```bash
sandboxai train \
  --env-count 8 --env-workers auto --steps 1000000 \
  --rollout-length 0 --batch-size 256 --ppo-epochs 10 \
  --torch-threads 0 --learning-rate 0.0003 --gamma 0.99 --gae-lambda 0.95 \
  --entropy-coefficient 0.01 --clip-range 0.2 \
  --checkpoint-frequency 100000 --evaluation-frequency 50000 \
  --seed 1234 --device auto --curriculum-level 3
```

`--max-train-minutes 30` time-boxes a run: the trainer stops itself at the
first safe step boundary after those minutes and saves the final checkpoint
anyway, so a 30-minute budget over a multi-hour plan is a supported
combination rather than a kill. `0` (the default) makes the step count the
only limit. The expiry is recorded in `run_summary.json` as
`"stop_reason": "time_budget"`.

`--rollout-length 0` is the default auto schedule: it keeps the aggregate
rollout near 16,384 transitions as `--env-count` changes (while capping the
per-environment horizon at 2,048). For example, 48 environments resolve to
352 steps/environment = 16,896 transitions/update instead of silently growing
to 98,304. A positive value requests that exact legacy horizon. Likewise,
`--torch-threads 0` selects a bounded pool after reserving CPUs for Godot
workers; use a positive value only after benchmarking it on the target host.
Both resolved values and the full-rollout timestep schedule are saved in
`config.json` and the optional training profile.

The entropy coefficient is non-zero by default (0.01): with MultiDiscrete
actions, a zero entropy coefficient lets PPO collapse to a degenerate action
(for example never shooting) in the first few updates, after which useful
behavior can no longer be discovered. Omit the flag to use the safe default.

Evaluation-related performance knobs (results are identical either way —
see [`docs/DEBUG_GUI_AND_BENCHMARKING.md`](docs/DEBUG_GUI_AND_BENCHMARKING.md)):

- `--eval-env-count N` and `--checkpoint-eval-env-count N` (both default 8):
  bridge environments used by normal evaluation and the checkpoint battery.
  Both paths explicitly plan every seed and restore report order, so batching
  changes wall time only. During training their separate persistent bridge
  processes run concurrently and are joined before best-model selection or
  early stopping.
- `--inference-device cpu` (default `auto`): run rollout/evaluation policy
  inference on CPU while PPO updates stay on `--device`. On CUDA hardware
  this removes the per-step host<->device round trip that makes GPU training
  *slower* than CPU for the tiny (106 -> 128 -> 128) policy.
- `--env-workers N|auto` (default 1): host the environments in N independent
  headless Godot processes instead of one. Shard *k* owns a contiguous slice
  of the environments and is launched with that slice's base seed, which is
  exactly the seed each environment would have received in a single process —
  so sharding changes wall time, never trajectories. `auto` sizes the pool
  from the host CPU, reserving two cores for the trainer. A shard that dies
  raises a `ShardFailure` naming the shard and closes the others cleanly.
- `--checkpoint-selection-metric KEY`, `--checkpoint-selection-goal max|min`
  and `--checkpoint-selection-min-delta X`: choose what "best checkpoint"
  means. Defaults reproduce the historical rule (strictly higher
  `mean_episode_reward`). Dotted keys reach the mirrored battery sections,
  e.g. `--checkpoint-selection-metric condition_evaluation.mean_win_rate`.
  The active rule is written into `evaluations/best.json` and the run
  manifest; resuming with an incomparable rule restarts selection instead of
  comparing two different quantities.
- `--profile-training`: writes `logs/training_profile.json`, including a
  per-boundary evaluation breakdown (process startup, prediction,
  environment stepping, combined battery execution and role overlap).

A JSON config can replace command-line editing:

```bash
sandboxai train --config training_config.json
```

To warm-start PPO from BC, request an exact-compatible transfer:

```bash
sandboxai train --bc-checkpoint training/bc_runs/human_v1/best.pt \
  --device cuda --steps 500000
```

The transfer copies only matching MLP hidden layers and categorical action
heads into SB3 PPO. If names, dimensions or action heads do not match, it
raises instead of silently pretending that weights transferred.

Resume PPO (SB3 `.zip` checkpoint):

```bash
sandboxai resume \
  --checkpoint training/runs/20260927-120000/checkpoints/latest.zip \
  --steps 500000 --device auto
```

### Integrated research pipeline (curriculum mode `auto`)

By default `sandboxai train` runs the integrated physics-of-learning loop
(`python/sandboxai/pipeline.py`, `python/sandboxai/checkpoint_eval.py`):

- **Every episode is planned.** A `CurriculumDriver` draws a fully seeded
  `EpisodePlan` (seed, map, scenario, lighting, enemy count, level) per
  environment from one authoritative stream (`master_seed +
  env_index + ordinal * 1_000_003`), stages it into Godot at episode
  boundaries only (`set_episode_plans`), and the bridge applies it on the
  consume-at-reset boundary. Resuming from a checkpoint restores the exact
  staged plans and per-environment ordinals
  (`checkpoints/curriculum_state.json`), so a resumed run produces the
  identical episode sequence.
- **Training and evaluation distributions are explicitly split.** The
  per-checkpoint condition evaluation draws from a frozen union-of-ladder
  distribution seeded with `seed + 707_000_017`, so eval seeds are
  provably never training seeds and the eval plan list is identical at
  every checkpoint (comparable across checkpoints). The generalization
  suite builds its train/holdout split from the conditions the run
  actually applied — a trained seed never lands in an unseen bucket, and
  trained (map, seed) pairs only ever count as `known`.
- **Levels 1-4 stay bit-for-bit legacy.** `applied_condition` strips
  map/scenario/lighting below world level 5 and enforces the engine's
  multi-enemy minimum (3) on levels 4-10; plans/tracker/metrics/replays
  always record the *applied* condition, never the sampled one.
- **Research metrics stay separate from rewards.** Per-tick skill metrics
  (8 categories) are measurement only; PPO rewards flow untouched from the
  engine. Aggregates flush per evaluation boundary into
  `evaluations/step_<N>/report.json` with by-environment / by-map /
  by-lighting / by-level groupings.
- **Replay recording is configurable** (`--replay-mode
  off|interesting|every_n|all|evaluation`, `--replay-every-n`,
  `--replay-max-per-run`): existing v1 replay format and header
  fingerprint unchanged, bounded per run, zero recorder allocation after
  the cap.
- **Optional league evaluation** (`--checkpoint-league-eval --league-matches-per-checkpoint 4`)
  freezes each evaluated checkpoint into `league/policies/`, plays
  deterministic two-slot matches on the `--self-play` bridge against older
  frozen snapshots plus the scripted baseline, enforces frozen-opponent
  integrity, and never loads league weights into the training policy.
  Incompatible checkpoints (wrong observation/action contract) fail with
  a named error.
- **Adaptive curriculum is conservative and optional.** With
  `--no-adaptive-curriculum` the same deterministic plans are issued but
  measured results never promote/demote; every decision is logged to
  `logs/curriculum.jsonl` with its evidence window. Adaptive feedback only
  ever changes *what* is trained next, never rewards.
- **A run manifest** (`run_manifest.json`, `sandboxai.run_manifest/v3`)
  records the experiment id, seed, contract fingerprint, curriculum
  config + current level, hyperparameters, parallelism
  (`environment_count`/`env_workers`/resolved workers/torch threads),
  enabled systems, evaluation configuration, the checkpoint-selection
  rule, the host snapshot (Python, OS, CPU count, WSL flag, CUDA device),
  the Godot build that produced the trajectories, and code provenance -
  commit, branch and **whether the working tree was dirty**. Every field
  degrades to `null` rather than failing the run.

Useful combinations:

```bash
# Fully deterministic, non-adaptive curriculum with extra eval replays:
sandboxai train --no-adaptive-curriculum --replay-mode evaluation

# Historic single-arena behavior (pre-pipeline), one fixed level:
sandboxai train --curriculum-mode fixed --curriculum-level 3

# Deeper checkpoint batteries with league matches:
sandboxai train --checkpoint-league-eval --league-matches-per-checkpoint 8 \
  --condition-eval-episodes 48 --generalization-episodes-per-cell 2
```

### Evaluation

Evaluation loads frozen weights and never calls an optimizer or updates model
parameters:

```bash
sandboxai evaluate \
  --checkpoint training/runs/20260927-120000/best_eval.zip \
  --episodes 100 --device auto \
  --output-dir training/evaluations/run_01
```

The evaluation directory includes `summary.json`, `summary.txt` and
`episodes.csv`. It reports reward, kills, deaths, damage dealt/received,
survival time, accuracy, shots fired/hit, win rate and loss rate.

### Inspect local weapon-calibration profiles

```bash
sandboxai weapon-table                          # local diagnostic matrix
sandboxai weapon-table --distances 3,6,9,12 --json
```

The table parses the local values from `scripts/weapon/weapon_state.gd` at
call time, so it cannot drift from the engine. It is **not** a TTK Testing
weapon list, balance sheet or claimed TTK measurement: the bundled profiles
and their recoil, magazine, falloff and reload values predate the current
TTK-only scope. Do not use their values as Roblox facts. The evidence gate and
screenshot/manual-calibration checklist are in
[docs/TTK_TESTING_REFERENCE.md](docs/TTK_TESTING_REFERENCE.md).

`python/tests/test_weapon_balance.py` only guards internal consistency of the
local calibration model. It does not validate real-game balance.

### Human time-to-kill evidence

`sandboxai ttk-report` turns manually annotated human TTK trials into
calibration evidence for the simulator. It reads a JSONL
`sandboxai.ttk_trials` v1 file (optional leading metadata object): consented,
ordinary-play observations with their conditions (weapon, distance, target health, movement state, hit zone,
frame rate, build, tester/session) and the acquisition / first-trigger /
first-damage / lethal timestamps.

```bash
sandboxai ttk-report --trials training/ttk/session_01.jsonl
sandboxai ttk-report --trials training/ttk/session_01.jsonl --by-condition
sandboxai ttk-report --trials training/ttk/session_01.jsonl --compare-simulator
sandboxai ttk-report --trials training/ttk/session_01.jsonl --json
```

It reports kills and censored trials **separately**, median/interquartile
mean with deterministic seeded bootstrap intervals, per-condition breakdowns
(distance bucketed at 3 m) and, with `--compare-simulator`, the difference
between human medians and the ideal/handling-aware TTK computed from
`scripts/weapon/weapon_state.gd`.

Validation is strict and refuses anything that is not ordinary observed play:
required fields, monotonic timestamps, `kill` ⇔ a lethal timestamp,
`shots_hit <= shots_fired`, an explicit `consent: true`, and any field whose
name suggests a private or cheat data source (`memory_`, `process_`,
`packet`, `server_authoritative`, `hidden_`, `injected`, `hook_`, `exploit`,
`aimbot`) is rejected outright. TTK trials are calibration evidence, never a
BC dataset: there is deliberately no array export.

### Inspect training runs

```bash
sandboxai inspect-runs --root training               # one line per run
sandboxai inspect-runs --run training/runs/<run_id>  # full report
sandboxai inspect-runs --root training --json        # stable JSON contract
```

Strictly read-only — safe to point at a run that is currently training. Each
report gives the run state *and the evidence it came from*, progress, the
checkpoint and evaluation inventory, log sizes, manifest provenance (commit +
dirty flag, host, Godot build, selection rule) and two separate lists:
`problems` (files that could not be parsed) and `warnings` (dirty code tree,
observation/action contract mismatch, no checkpoints, no `best_eval.zip`,
unfinished run). A half-written or corrupt run directory still produces a
usable report.

### Benchmark simulation throughput

```bash
sandboxai benchmark --env-counts 1,2,4,8,16,24,32,48,64 \
  --steps 2000 --output-dir training/benchmarks/4060ti

# how many Godot worker processes should host those environments?
sandboxai benchmark --env-counts 4,8,12,16,20 --worker-counts 1,2,4,8 \
  --steps 2000 --output-dir training/benchmarks/workers
```

`--worker-counts` sweeps the `--env-workers` setting for every environment
count and reports measured steps/s per (environments, workers) pair; worker
counts above the environment count are clamped. `tools/bridge_scaling_probe.py`
isolates the transport and process-parallelism scaling **with a synthetic
workload** — it is a probe for the bridge, not a measurement of Godot.

Benchmark output includes environments, total steps, steps/sec,
episodes/sec and best-effort CPU/process/GPU memory snapshots. Each
environment count is time-boxed (`--max-seconds-per-config`, default 20s) so
sweeping the full recommended list does not take an unbounded amount of
time. It is the practical way to choose an environment count before PPO
training — see
[`docs/DEBUG_GUI_AND_BENCHMARKING.md`](docs/DEBUG_GUI_AND_BENCHMARKING.md)
for how to interpret the results and where the bottleneck usually is.

## Checkpoint and output layout

A normal PPO run is structured as:

```
training/
  runs/<run-id>/
    config.json
    warm_start.json
    final.zip
    run_summary.json
    run_manifest.json                  # integrated pipeline (auto mode)
    checkpoints/
      ppo_<timesteps>_steps.zip
      latest.zip
      best_eval.zip
      curriculum_state.json            # resume-exact curriculum/driver state
    logs/
      training.jsonl
      episodes.jsonl                   # one row per episode: plan + metrics
      curriculum.jsonl                 # adaptive decisions (promo/demotions)
      tensorboard/
    evaluations/
      latest.json
      best.json
      step_<timesteps>/
        summary.json, summary.txt       # normal evaluation summary
        normal_episodes.csv             # normal evaluation rows (auto mode)
        episodes.csv                    # generalization rows (auto mode)
        report.json                    # checkpoint battery (auto mode)
        policy.zip                     # frozen policy under evaluation
        generalization.json/.txt       # seen/unseen split results
        replays/                       # when replay-mode all|evaluation
    replays/                           # training replays (interesting/every_n/all)
    league/                            # only when --checkpoint-league-eval
      registry.json, history.json
      policies/<experiment>@<step>.zip
  bc_runs/<run-id>/
    config.json, dataset_report.json   # split/leakage report + fingerprint
    latest.pt, best.pt, epoch_*.pt
    metrics.jsonl, loss.csv
  datasets/<name>.jsonl
  evaluations/<name>/
  benchmarks/<name>/
```

`latest.zip` is written only after a successful training call. `best_eval.zip`
is updated only when the configured checkpoint-selection rule improves
(default: strictly higher mean evaluation reward); the rule itself is recorded
in `evaluations/best.json`. BC checkpoints contain
model/optimizer state, epoch, architecture, action nvec, dataset path and
metrics so training can resume.

TensorBoard can be opened with:

```bash
tensorboard --logdir training/runs/<run-id>/logs/tensorboard
```

Training JSONL telemetry tracks timesteps, progress, steps/sec,
episodes, reward, kills, accuracy, win rate, environment count, checkpoint
location and best-effort resource utilization.

## Environment and extension points

- `EnvironmentCore` is the canonical render-independent FPS test
  environment. `SimulationManager` owns isolated batches.
- `RLAdapter` exposes reset, batched step, observations, rewards, done flags,
  metrics and the fixed action/observation space descriptions.
- `scripts/rl/rl_server.gd` is the local JSON-lines bridge. The Python side
  does not call `localhost`, use a browser, or depend on a visual scene.
- Curriculum levels 1–4 progressively enable stationary targets, moving
  targets with varied spawn positions, strafing/attacking enemies and
  multiple (3+) simultaneously-tracked enemies. Level 5 exposes the
  self-play slot/match foundation. See
  [`docs/CURRICULUM_AND_COMBAT.md`](docs/CURRICULUM_AND_COMBAT.md).
- `SelfPlayEnvironmentCore` provides two controllable `AgentState` slots with
  per-agent rewards/metrics. `SelfPlayCoordinator` supports a learning slot
  against a frozen SB3 checkpoint; population algorithms are intentionally
  not implemented yet.
- `Observation` remains structured-only today. `ObservationMode.RGB` and the
  Python configuration leave room for future RGB observations and temporal
  frame stacking without changing the structured vector contract.
- `scripts/debug/debug_overlay.gd` is an optional, presentation-only debug
  GUI (FPS/telemetry + pause/reset/enemy-count/curriculum controls). It is
  never created by the headless RL bridge. See
  [`docs/DEBUG_GUI_AND_BENCHMARKING.md`](docs/DEBUG_GUI_AND_BENCHMARKING.md).
- The Control Center is the Python/Tkinter desktop application
  (`python/sandboxai/control_center_*.py`, started with `python3 main.py`).
  It operates the headless stack through `sandboxai.adapter`, never renders
  the game, and is not imported by any training module. See
  [`docs/CONTROL_CENTER.md`](docs/CONTROL_CENTER.md).
- `python/sandboxai/contract.py` documents the local Godot/Python
  Observation/Action contract as data. It is not a Roblox integration seam.

Simulation code never imports PyTorch. Python code never depends on Godot
render nodes. The only current external process boundary is the lightweight
JSON-lines bridge, which is portable across Windows and Ubuntu.
