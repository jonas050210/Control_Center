# 3D checkpoint viewer

Watch a trained policy play in a rendered Godot window, on any map, with
spectator cameras and pause, step and speed controls.

```bash
python start.py view                                    # newest run, best checkpoint
python start.py view --checkpoint training/runs/<run>   # a specific run (folder or file)
python start.py view --checkpoint training/bc_runs/latest/best.pt
python start.py view --policy scripted --map compound   # no checkpoint needed
python start.py view --list-maps
```

You can also open it from the Control Center: go to **Runs / Checkpoints**,
select a run and press **Watch in 3D**. The viewer then counts as a normal
job, so you stop it from the active-jobs strip or by closing its window.

`sandboxai view` (and `python -m sandboxai view`) is the same command.

## What you are looking at

The simulation is the canonical `EnvironmentCore`. Every physics tick runs
at the fixed training `dt` and takes exactly one policy decision, the same
as the headless training bridge. The playback speed only changes how many
ticks run per rendered frame, never the physics of a tick. So the agent
sees exactly the observation vector it was trained on, and what you watch is
what the policy experiences.

The window shows:

- the agent and its enemies in the 3D map;
- a HUD with the policy name, map, level, step count and speed;
- the action the policy just chose (move, look and shoot);
- a win/loss/timeout tally of the recent episodes.

Every finished episode is also printed in the terminal (outcome, reward,
kills and survival time). On exit the terminal shows a summary:
`Watched N episode(s) ...` and `Answered N policy decisions.`

## Controls

| Key | Action |
| --- | --- |
| `Space` | pause / resume |
| `N` or `.` | single step (pauses first) |
| `+` / `-` | speed: 0.25x, 0.5x, 1x, 2x, 4x, 8x |
| `C`, or `1` to `4` | camera: chase, orbit, top-down, first person |
| mouse drag / wheel | orbit and zoom (orbit and top-down cameras) |
| `M` / `Shift+M` | next / previous map (includes the generated arena) |
| `R` | restart the episode |
| `H` | hide or show the HUD |
| `Esc` / `Q` | quit |

## Options

| Option | Meaning |
| --- | --- |
| `--checkpoint PATH` | Accepts several forms. A file can be an SB3 PPO `.zip` or a BC `.pt`. A run folder is resolved in this order: `best_eval.zip`, `best.zip`, `final.zip`, `latest.zip`, then the newest `ppo_N_steps.zip`. A BC folder uses `best.pt`, then `latest.pt`. Without this option, the most recently written checkpoint under `training/runs` and `training/bc_runs` is used. |
| `--policy {ppo,bc,scripted}` | Overrides detection. `scripted` uses the built-in baseline. |
| `--map ID` | Start map (`--list-maps`). Without it, the scenario-generated arena is used. |
| `--level 1-10` | Curriculum level. Defaults to the run's own level (from `checkpoints/curriculum_state.json` or `config.json`), otherwise 10. |
| `--enemies`, `--seed`, `--scenario`, `--lighting`, `--weapon` | Episode setup, same meaning as in training. |
| `--speed X`, `--camera MODE` | Initial playback speed and camera. |
| `--stochastic` | Sample actions instead of taking the most likely one. |
| `--headless`, `--max-steps N`, `--max-episodes N` | For automated checks; CI uses these. |
| `--godot-executable PATH` | Normally not needed. `install.py` downloads Godot to `tools/godot/` and remembers it. |

Inference always runs on the CPU. A small MLP needs well under a millisecond
per decision, so the viewer stays smooth at any speed.

## How it works

```
 Godot window (viewer.tscn)                      Python (sandboxai.viewer)
 ViewerMain -> SimulationManager                 loads the checkpoint (SB3 / BC / scripted)
   RemotePolicyController ---- TCP 127.0.0.1 ---> answers one request per tick
     (observation -> action)   newline JSON       prints the episode results
```

`python start.py view` does the following:

1. Loads the policy.
2. Opens a server socket on a free local port.
3. Starts Godot with `--script res://scripts/viewer/viewer_entry.gd -- --policy-port <port> ...`.
4. Answers the viewer's requests until the window closes.

The protocol consists of four messages, each one JSON object per line:

| Message (Godot to Python) | Reply |
| --- | --- |
| `{"type":"hello","observation_size":126,"action_nvec":[...]}` | `{"ok":true,"policy":"<label>"}` or `{"ok":false,"error":...}` |
| `{"type":"act","obs":[126 floats]}` | `{"action":[6 ints]}` |
| `{"type":"episode","summary":{...}}` | `{"ok":true}` |
| `{"type":"bye"}` | none |

The `hello` exchange checks the observation and action contract on both
sides. A checkpoint trained on a different contract fails here with a
clear message, before anything is drawn.

If the connection drops, the agent stands still (idle action) and the HUD
reports the error, instead of a policy acting on stale data.

Without `--policy-port`, the scene runs standalone with the built-in
heuristic controller. This is handy for looking at maps from the Godot
editor:

```bash
godot --path . --script res://scripts/viewer/viewer_entry.gd -- --map compound
```

## Files

| File | Role |
| --- | --- |
| `python/sandboxai/viewer.py` | Checkpoint resolution, policy loading, the socket server and the Godot launch. |
| `scenes/viewer.tscn`, `scripts/viewer/viewer_main.gd` | The scene: episode loop, input, map switching. |
| `scripts/viewer/viewer_entry.gd` | `--script` entry point that opens the scene. |
| `scripts/viewer/remote_policy_client.gd`, `remote_policy_controller.gd` | The TCP client, wrapped as a regular `ControllerBase`. |
| `scripts/viewer/viewer_camera_rig.gd`, `viewer_hud.gd` | Cameras and the HUD. |
| `tests/test_viewer.gd`, `python/tests/test_viewer.py` | Tests for both halves. They include a real TCP round trip and end-to-end runs with PPO, BC and scripted policies. |

CI (`.github/workflows/godot-tests.yml`, job `install-and-viewer`) runs
`python install.py` on clean Linux and Windows runners. It then plays the
viewer headless against the scripted baseline and against a real PPO
checkpoint.
