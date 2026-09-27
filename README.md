# SandboxAI

SandboxAI is a local, open-source reinforcement-learning research platform
built around a small Godot 4.7.2 FPS combat simulator. The simulator is the
canonical test environment: it is deterministic, headless-capable, and
rendering is not required by the Python trainer.

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
checkpoints, evaluation, datasets and telemetry. No paid API, cloud service,
Roblox integration or visual-scene dependency is required.

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
pass `--godot-executable` to commands. Then from the repository root:

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

## Automated tests

Godot tests:

```bash
godot --headless --path . --script res://tests/run_tests.gd
```

The runner discovers `tests/test_*.gd`, reports every test, and exits nonzero
on failure. Python tests (after editable installation, or with `PYTHONPATH`)
are:

```bash
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

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
```

### Behavior Cloning

```bash
sandboxai bc-train \
  --dataset training/datasets/human_demo.jsonl \
  --epochs 25 --batch-size 512 --device auto \
  --output-dir training/bc_runs/human_v1
```

The result contains `latest.pt`, `best.pt`, periodic epoch checkpoints,
`metrics.jsonl`, `loss.csv` and `config.json`. The model is a small two-hidden-
layer PyTorch MLP with one categorical head per action field. Validation
reports component accuracy and exact five-field action accuracy.

Resume BC training:

```bash
sandboxai bc-train --dataset training/datasets/human_demo.jsonl \
  --output-dir training/bc_runs/human_v1 \
  --resume-checkpoint training/bc_runs/human_v1/latest.pt --epochs 50
```

### PPO training

The trainer starts one Godot headless process containing the requested number
of independent environments and uses a `MultiDiscrete([3,3,3,3,2])` action
space. The structured observation is a 33-float `Box` (see
[`docs/OBSERVATION_ACTION_CONTRACT.md`](docs/OBSERVATION_ACTION_CONTRACT.md)
for the full field-by-field table, including the multi-enemy tracking
fields added for curriculum levels with more than one enemy).

```bash
sandboxai train \
  --env-count 8 --steps 1000000 \
  --rollout-length 2048 --batch-size 256 \
  --learning-rate 0.0003 --gamma 0.99 --gae-lambda 0.95 \
  --entropy-coefficient 0.01 --clip-range 0.2 \
  --checkpoint-frequency 100000 --evaluation-frequency 50000 \
  --seed 1234 --device auto --curriculum-level 3
```

The entropy coefficient is non-zero by default (0.01): with MultiDiscrete
actions, a zero entropy coefficient lets PPO collapse to a degenerate action
(for example never shooting) in the first few updates, after which useful
behavior can no longer be discovered. Omit the flag to use the safe default.

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

### Benchmark simulation throughput

```bash
sandboxai benchmark --env-counts 1,2,4,8,16,24,32,48,64 \
  --steps 2000 --output-dir training/benchmarks/4060ti
```

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
    checkpoints/
      ppo_<timesteps>_steps.zip
      latest.zip
      best_eval.zip
    logs/
      training.jsonl
      tensorboard/
    evaluations/
      latest.json
      best.json
      step_<timesteps>/summary.json, episodes.csv, summary.txt
  bc_runs/<run-id>/
    config.json, latest.pt, best.pt, epoch_*.pt
    metrics.jsonl, loss.csv
  datasets/<name>.jsonl
  evaluations/<name>/
  benchmarks/<name>/
```

`latest.zip` is written only after a successful training call. `best_eval.zip`
is updated only when mean evaluation reward improves. BC checkpoints contain
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
- `python/sandboxai/contract.py` documents the Observation/Action contract
  as data and defines the abstract `GameAdapter` boundary a future external
  Roblox Player adapter would implement. No Roblox integration exists yet —
  see [`docs/ROBLOX_ADAPTER.md`](docs/ROBLOX_ADAPTER.md).

Simulation code never imports PyTorch. Python code never depends on Godot
render nodes. The only current external process boundary is the lightweight
JSON-lines bridge, which is portable across Windows and Ubuntu.
