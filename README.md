# NEURAL ARENA v2.0

**NEURAL ARENA** is a headless 3D shooter training sandbox built with Gymnasium, Stable-Baselines3 PPO, and a cyberpunk-green Streamlit control center. Simulation and physics run in Python; all arena visuals are rendered in the browser with Plotly. It does not import Pygame, open a game window, or require X11/a display server, so it is suitable for WSL Ubuntu.

## 1. WSL Ubuntu setup

Install Python build tools and create a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential

# Keep Python packages in WSL's Linux home instead of syncing a large .venv through OneDrive
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/control-center
source ~/.venvs/control-center/bin/activate
python -m pip install --upgrade pip wheel
```

Now locate the checkout and `cd` into it. The folder name depends on how you got the code — `Control_Center` after `git clone`, `Control_Center-main` after extracting the GitHub ZIP — so don't guess the path:

```bash
# Find the repository, then cd into the folder it prints
find /mnt/c/Users -maxdepth 7 -type d -iname "Control_Center*" 2>/dev/null
cd "/mnt/c/Users/Jonas/OneDrive/Desktop/Control_Center-main"   # <- paste the real path from above
ls requirements.txt   # must succeed: every command below runs from the repository root
```

This project trains PPO on the CPU to leave the RTX 4060 Ti available for other work. For a smaller CPU-only PyTorch install, install that wheel before the rest of the requirements:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the dashboard from the repository root:

```bash
streamlit run gui/app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://localhost:8501** in the Windows desktop browser. The project is intended to run on the local PC through WSL Ubuntu; it does not open a separate game window or need Pygame/X11. No public hosting or remote game service is required.

## 2. Project layout

```text
.
├── env/
│   ├── shooter_env.py      # Gymnasium 3D duel, MultiDiscrete actions, normalized observations
│   ├── weapons.py          # Six weapon classes and reload/ammo/recoil mechanics
│   ├── maps.py             # Six 3D maps, cover objects, randomization, spawn points
│   ├── map_io.py           # Validated, atomic Custom-map JSON import/export
│   └── physics.py          # Movement, stance, gravity, collision, ray casting
├── training/
│   ├── train.py            # Background-thread PPO, 4-phase curriculum, self-play/checkpoints
│   ├── workers.py          # CPU worker selection, VecEnv setup, 20-second benchmark runner
│   ├── rewards.py          # Auditable reward shaping components
│   └── imitation.py        # Behavior cloning and raw-terminal demonstration recorder
├── gui/
│   ├── app.py              # Streamlit navigation and session setup
│   ├── common.py           # Shared theme, paths, and session helpers
│   ├── games.py            # Aim-drill and dodge-survival state logic
│   ├── visuals.py          # Batched, low-poly Plotly 3D scenes and procedural detail
│   ├── style.css           # Neon-green cyberpunk theme
│   └── tabs/               # Arena, human Playground, training, stats, benchmark, TTK, maps, heatmap
├── data/demos.csv          # Small starter set of valid state/action examples
├── models/                 # PPO/BC checkpoints are written here at runtime
├── logs/                   # Training and episode heatmap CSV logs are written here
├── tests/test_env.py       # Headless Gymnasium and reward contract smoke tests
├── tests/test_games.py     # Aim, dodge, and human-control tests
├── tests/test_playground_ui.py  # Headless Streamlit control smoke test
├── tests/test_visuals.py   # 3D scene detail and WebGL trace-budget tests
└── requirements.txt
```

## 3. Environment contract

`env.shooter_env.ShooterEnv` is a Gymnasium environment compatible with SB3 PPO and Gymnasium's five-value `step()` API. One policy plays Agent 1 against a scripted opponent. `step_duel(action_a, action_b)` is available for browser-side policy-versus-policy matches.

- **Action space:** `MultiDiscrete([3, 3, 3, 3, 2, 2, 3, 3, 2, 2])`. In order: forward/back, strafe, yaw, pitch, shoot, sprint, stand/crouch/prone, lean, jump, reload. Controls can be combined in one action.
- **Observation space:** 31 `float32` values clipped to `[-1, 1]`: own/enemy 3D positions and rotations, distance and aim angles, eight ray distances, both health values, magazine/ammo values, movement/stance/air state, and shot/reload timers.
- **Physics:** 60 Hz, four physics frames per policy action by default, vertical gravity and jumps, walkable ramps/upper floors, AABB cover collision, cylinder hitboxes, spread raycasts, upper-20%-of-hitbox headshots, and distance falloff.
- **Episodes:** end on a fighter's death or truncate at 120 simulated seconds. `reset()` returns `(observation, info)` and `step()` returns `(observation, reward, terminated, truncated, info)`.
- **Maps:** Dust, Warehouse, Highrise, Arena, Sniper Alley, and an empty Custom sandbox. Cover is data-driven and can be randomized or edited in the Maps tab.
- **Weapons:** Pistol, SMG, AK-47, Shotgun (10 pellet rays), Sniper, and LMG.

Quick smoke test after installing dependencies:

```bash
python - <<'PY'
from env.shooter_env import ShooterEnv

env = ShooterEnv(map_name="Dust", seed=7)
obs, info = env.reset()
print("observation:", obs.shape, obs.min(), obs.max(), info)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
print("headless environment OK")
env.close()
PY

python -m unittest discover -s tests -v
```

The unit suite includes map JSON validation, sample-demo checks, reward and worker-lock tests, deterministic mini-game checks, headless Playground AppTests, and 3D scene/trace-budget checks (including 400 props rendered in one mesh). The one-rollout PPO integration test runs automatically when both Stable-Baselines3 and PyTorch are installed; otherwise it is reported as skipped.

## 4. PPO training

Open **TRAINING** in the dashboard to select duration, workers, environments per worker, curriculum/self-play, map, and training method. PPO executes in a daemon worker thread, so Streamlit stays responsive. Stop and pause requests are checked at vector steps; worker environments are closed in the training thread's `finally` path.

The default PPO setup uses CPU, linear learning-rate decay from `3e-4`, `n_steps=2048`, batch size 256, 10 epochs, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, and `ent_coef=0.01`. Observation/reward normalization is applied with `VecNormalize` and saved alongside a model. Worker auto-detection leaves a physical core available; the dashboard caps the environment pool at 24 processes for the target 12-core CPU. A process-wide CPU-job lock prevents a full benchmark and PPO run from saturating the same machine simultaneously.

Curriculum phases progress with training timesteps:

1. Stationary pistol opponent and pistol-only aiming.
2. Slow-moving SMG opponent for tracking.
3. AK-47 opponent returns fire for dodge learning.
4. Full scripted opponent and randomized weapons. If self-play is enabled, the phase-three/best actor is exported as a frozen NumPy MLP and installed in workers without importing a renderer or opening a window.

A rolling win-rate improvement at each 50,000-step check updates `models/best_model.zip` and `models/best_model_vecnormalize.pkl`; the prior best is overwritten rather than retaining unscored snapshots. **Save Checkpoint** creates a timestamped manual checkpoint. Training telemetry is appended to `logs/training_metrics.csv`, and episode events to `logs/heatmap_events.csv`.

Resume a policy with **Resume Checkpoint**. The normalizer sidecar is loaded automatically when it is present next to the `.zip` model.

## 5. Imitation learning and recording

The repository includes a small starter dataset at `data/demos.csv`. Selecting **Imitation → RL** trains a 31-input categorical MLP on the training thread if `models/behavior_clone.pt` does not exist, then copies its actor weights into PPO for fine-tuning.

Train the clone separately:

```bash
python -m training.imitation train --input data/demos.csv --output models/behavior_clone.pt --epochs 80
```

Record additional examples from an interactive WSL terminal (not a display server):

```bash
python -m training.imitation record --map Dust --weapon Pistol --output data/demos.csv
```

Recorder keys: **W/S** forward/back, **A/D** strafe, **J/L** yaw, **I/K** pitch, **Space** fire, **X** sprint, **C** crouch, **P** prone, **B/N** lean, **R** reload, and **Q** quit. The CSV stores one normalized state and discrete action per policy step.

## 6. Dashboard tabs

- **ARENA:** select map, weapons, and optional saved PPO policies; start/pause/reset/step matches; inspect low-poly 3D fighters, detailed cover, HP, facing and grouped bullet trails. Choose Performance, Balanced, or Ultra scene detail.
- **PLAYGROUND:** fight a Gymnasium bot or saved PPO `.zip` policy using desktop controls for movement, aim, fire, reload, stance and sprint; rotate the 3D scene with the mouse. Includes a detailed 3D target gallery, 3D projectile-dodge arena, selectable graphics quality, and downloadable human imitation demonstrations.
- **TRAINING:** background PPO controller, progress, metrics, live worker log, pause/stop, and checkpoint controls.
- **STATS:** reward/win-rate curves, TTK and weapon-use charts, CPU and RAM monitor.
- **BENCHMARK:** measures every valid worker × envs-per-worker pair from the requested matrix, skipping products above 24. Each case has a 20-second stepping window after its worker pool starts. The best throughput can be copied to the Training controls.
- **TTK-TESTER:** tune two weapons, simulate 1,000–100,000 duels, inspect win rate/TTK/accuracy/DPS, and copy a Roblox Lua weapon table.
- **MAPS:** shaded 3D cover preview with a detail preset, map stats, random cover generation, Custom object editing, and validated JSON import/export. The Custom layout autosaves locally to `data/custom_map.json` and is reused by Arena and training.
- **HEATMAP:** filter completed episodes by map, weapon, distance, and episode range; deaths are red, unrecorded/safe areas green, and kill zones yellow.

All Plotly figures explicitly use the `#0a0a0a` / `#141414` dark palette and neon-green typography. The Streamlit session state stores navigation choices, match state, benchmark/training controllers, and browser-side episode events.

## 7. Runtime files

Model binaries and run logs are generated at runtime under `models/` and `logs/`; the editable Custom layout is saved locally as `data/custom_map.json`. These runtime artifacts are git-ignored. The included starter demo CSV is tracked so the imitation path is usable immediately after dependency installation.
