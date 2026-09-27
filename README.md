# SandboxAI

SandboxAI is a **Godot 4.7.2** based reinforcement-learning training
simulator for a minimal 3D FPS combat environment. Godot is used as the
*simulation/training environment*, not as a shipped game: the arena is
intentionally small, deterministic and free of visual polish so it stays
fast and easy to reason about while the RL/training pipeline around it
matures.

This is **milestone 1**: prove that SandboxAI can run a controllable FPS
environment with RL-compatible observations, actions, rewards, resets, and
multiple independent parallel environments — and that a human can drive the
exact same agent/action pipeline an AI eventually will.

## Requirements

- Godot **4.7.2** (standard build, GDScript only — no C#/Mono needed)
- Nothing else. No plugins, no external engines, no paid services.

Developed against the target spec of Windows 11 + RTX 4060 Ti 8GB +
Python 3.11, but the project itself only needs Godot; a Python RL trainer
is out of scope for this milestone (see "Next milestone" below).

## Running the demo / human-play mode

1. Open the project folder in Godot 4.7.2 (`project.godot` at the repo root).
2. Press **Play** (or F5). `scenes/main.tscn` is the main scene.
3. You control the agent in **environment 0** with:
   - `W`/`A`/`S`/`D` — move / strafe
   - Mouse — look (captured by default; press `Esc` to release, click to
     recapture)
   - Arrow keys — discrete look left/right/up/down (works even without a
     mouse)
   - Left mouse button or `Space` — shoot
4. Every other environment (3 more by default) is driven by a trivial
   deterministic `AIStubController` so you can see multiple independent
   environments running side by side.
5. A small debug overlay in the top-left shows render FPS, simulation
   steps/second, agent health, enemy count, kills, deaths, episode count,
   and current reward.

## Running the automated tests

Tests are plain GDScript (no external test addon) under `tests/`, runnable
headlessly:

```bash
godot4 --headless --path . --script res://tests/run_tests.gd
```

(use whatever the Godot 4.7.2 executable is called on your system, e.g.
`Godot_v4.7.2-stable_win64.exe --headless --path . --script res://tests/run_tests.gd`
on Windows). The runner discovers every `tests/test_*.gd` file, executes
every `test_*` method, prints `PASS`/`FAIL` per test and exits with code
`1` if anything failed (`0` otherwise), so it's CI-friendly.

> **Note on this sandbox:** the environment this milestone was authored in
> has no network access to download the actual Godot editor/export
> binaries (only `git`/PyPI/npm registries are reachable), so the test
> suite could not be executed inside a running Godot process here. Every
> script was instead validated with `gdlint`/`gdformat`
> ([gdtoolkit](https://github.com/Scony/godot-gdscript-toolkit), a real
> GDScript parser) to catch syntax errors, and every algorithm (movement,
> rotation, ray/sphere hit-test, reward math) was independently
> cross-checked with an equivalent Python calculation. Please run the
> command above on your machine and see the final report/PR description
> for details.

## Project layout

```
project.godot                  Godot project file (Godot 4.7.2)
scenes/main.tscn                Entry scene (boots everything from code)
scripts/
  core/
    sandbox_config.gd           Centralized tunable constants (arena, agent,
                                 enemy, weapon, rewards, episode limits)
    action.gd                   Structured/discrete Action type
    observation.gd               Structured Observation builder
    episode_state.gd             Per-environment episode bookkeeping
    simulation_manager.gd       Owns & ticks N independent environments
    main.gd                      Boots lighting, SimulationManager,
                                 controllers, camera, debug overlay
  env/
    environment_core.gd          RL environment: reset/step/observations/
                                 rewards/done (no Node dependency)
    environment_view.gd          Visual-only arena/agent/enemy mirror
  agent/
    agent_state.gd               Agent movement/aim/health logic (pure)
    agent_view.gd                Agent Node3D + FPS camera (visual only)
  enemy/
    enemy_state.gd               Enemy idle/chase/attack AI + health (pure)
    enemy_view.gd                Enemy Node3D mirror (visual only)
  weapon/
    weapon_state.gd               Fire cooldown + ray/sphere hit-test (pure)
  reward/
    reward_system.gd              Centralized reward calculation (pure)
  rl/
    rl_adapter.gd                 Gym-style façade over SimulationManager
  input/
    controller_base.gd            Shared interface for human & AI controllers
    human_controller.gd           WASD + mouse-look + click -> Action
    ai_stub_controller.gd         Deterministic heuristic -> Action
  debug/
    debug_overlay.gd              Minimal on-screen telemetry
tests/
  sandbox_test.gd                Tiny dependency-free assertion helper
  run_tests.gd                    Headless test runner (SceneTree script)
  test_*.gd                       One test file per subsystem
docs/
  ARCHITECTURE.md                Design notes & extension points
```

See `docs/ARCHITECTURE.md` for the full design rationale, the RL
reset/step/observe/reward/done interface, and how vision/RGB observations,
human demonstration recording, self-play and PPO training are meant to
slot in later without reshaping what's here.
