# SandboxAI — technical context, constraints, and research plan

> **Audience:** coding agents and researchers who need repository reality quickly. This is not a user README.
> **Reality check:** audited against source commit `d0d60a2bb7081b6f8acde4b1df8ae01660fbe54c` on 2026-09-29. Source and tests win if this document later drifts.

## Status vocabulary

| Label | Meaning |
| --- | --- |
| **CURRENT** | Implemented in this repository now. |
| **PLANNED** | A concrete next step, not implemented. |
| **RESEARCH** | A hypothesis or architecture to evaluate; do not present it as a commitment. |
| **CONSTRAINT** | An invariant, compatibility boundary, or ethical/technical limit. |

## 1. One-minute orientation

**CURRENT — purpose.** SandboxAI is a controlled, local FPS calibration and machine-learning platform being reduced to verified Roblox TTK Testing mechanics. Godot is the canonical local simulator; Python records demonstrations, trains behavior-cloning (BC) and PPO policies, evaluates checkpoints, and produces research telemetry. Its source-backed TTK scope, calibration gaps and exclusions are recorded in `docs/TTK_TESTING_REFERENCE.md`.

**CONSTRAINT — scope and ethics.** This is **not** a Roblox exploit, cheat, live-game bot, process-memory reader, packet inspector, client modifier or external-game adapter. There is no Roblox connection. Human data must come from the local simulator or from manual, consented observation/annotation of ordinary play; never assume private APIs, server-authoritative internals, hidden positions, or proprietary data. There is no connection to Roblox servers and no gameplay automation: the only client-facing code is the bounded calibration helper surface (detect, launch, focus, screenshot, hand-typed values) specified in `docs/TTK_TESTING_REFERENCE.md`.

**CURRENT — design properties.** The project is local, free/open-source, structured-state-first, reproducibility-oriented, and cloud-independent. Core Python needs only NumPy; training extras are PyTorch, Gymnasium, Stable-Baselines3 (SB3), TensorBoard, and psutil. Godot 4.7.2 is the tested engine target.

### Non-negotiable contracts

- **CONSTRAINT:** every policy observation is **exactly 106 float32-compatible values**, each in `[-1, 1]`.
- **CONSTRAINT:** the PPO action space is **exactly `MultiDiscrete([3,3,3,3,2,2])`**.
- **CONSTRAINT:** the trained path is structured state, **not RGB**, pixels, optical flow, or frame stacking.
- **CONSTRAINT:** do not reorder/reinterpret fields silently. Shape checks cannot detect semantic drift.
- **CONSTRAINT:** Godot's controlled analytic simulation is canonical. Views and debug UI mirror state; they do not drive it.
- **CONSTRAINT:** reproducibility means seeded, fixed-step experiment replay within a recorded software/hardware boundary—not universal bitwise identity across OSes, CPUs, GPUs, PyTorch versions, and Godot builds.
- **CONSTRAINT:** benchmark numbers must be measured on the target machine. The benchmark code deliberately refuses to invent them.

### Intended progression

| Step | Status | What it means here |
| --- | --- | --- |
| Human/TTK evidence | **CURRENT + PLANNED** | Godot can record exact human transitions now. A systematic manual TTK calibration dataset is still to be collected. |
| Behavior cloning | **CURRENT** | Train a six-head categorical policy from JSONL demonstrations; optionally warm-start PPO's actor. |
| Controlled sandbox | **CURRENT** | Seeded analytic movement/combat, maps, visibility, sound, memory, navigation, weapon handling, and metrics. |
| PPO | **CURRENT** | SB3 on-policy PPO over the Godot bridge, with automatic curriculum and checkpoint evaluation. |
| Parallel/headless scale | **CURRENT, limited** | Headless batched environments exist, but one Godot process steps its environments serially. |
| Self-play/complex combat | **CURRENT foundation; PLANNED training** | A two-policy match environment and league registry exist; no end-to-end self-play training command exists, and lethal fire has a documented slot-order bias. |
| RGB learning | **RESEARCH** | No image observation, renderer-to-trainer data plane, CNN, or visual checkpoint exists. |
| TTK calibration | **CURRENT + PLANNED** | Official controls and visible wound/bleeding are catalogued; numerical physics, weapon and reload behavior remain evidence-gated. |

## 2. Architecture and ownership

### Runtime split

| Layer | Status | Owns | Key files |
| --- | --- | --- | --- |
| Godot simulation | **CURRENT** | Episode state, analytic movement/collision, seeded world generation, perception, enemies, weapons, reward, terminal state, optional views | `scripts/env/environment_core.gd`, `scripts/core/simulation_manager.gd` |
| Contract/adaptation | **CURRENT** | Observation and action semantics, batched reset/step, terminal observation | `scripts/core/{observation,action}.gd`, `scripts/rl/rl_adapter.gd` |
| Local IPC | **CURRENT** | One Godot subprocess, request/response JSON Lines over stdin/stdout | `scripts/rl/rl_server.gd`, `python/sandboxai/godot_env.py` |
| ML/orchestration | **CURRENT** | BC, PPO, curriculum plans, evaluation, checkpointing, replay, telemetry, benchmark control | `python/sandboxai/` |
| Visual operator tools | **REMOVED** | The in-simulator operator scene (`scenes/control_center.tscn`, `scripts/control_center/`) was deleted; the Control Center is now the headless-only Python/Tk desktop application (`python3 main.py`). Human demonstration recording remains via `record` | `main.py`, `python/sandboxai/control_center_*.py`, `docs/CONTROL_CENTER.md` |

```text
Python: SB3/PyTorch policy
   | action batch                         ^ obs/reward/done/info batch
   v                                      |
GodotProcessTransport -- newline JSON --> rl_server.gd (one subprocess)
                                              |
                                          RLAdapter
                                              |
                                      SimulationManager
                                              |
                           EnvironmentCore[0], [1], ... [N-1]
                           (currently stepped serially, fixed dt)
```

**CURRENT lifecycle.** Python launches:

```text
godot --headless --path <repo> --script res://scripts/rl/rl_server.gd -- --stdio ...
```

The server blocks on one request, parses JSON, executes it, serializes one response, flushes one line, and repeats. A `step` applies one action per environment and advances one conceptual 60 Hz tick (`dt = 1/60`) as fast as the host can execute; it is not wall-clock throttled. On episode completion, vector mode auto-resets and returns the new observation while preserving the final one as `info.terminal_observation`; `TimeLimit.truncated` is true only for timeout.

**CURRENT bridge commands.** `spaces`, `reset`, `reset_indices`, `step`, `metrics`, `reward_breakdown`, `set_curriculum`, `set_episode_plans`, `episode_conditions`, `health_check`, `profile_snapshot`, `ping`, and `close`; self-play mode exposes a smaller two-slot variant. Input lines are capped at 1 MiB. Python drains both stdout and stderr on background threads and enforces a request timeout, preventing pipe-fill deadlocks.

**CONSTRAINT:** the wire protocol has no explicit negotiated protocol version. The `spaces` handshake and Python's 106-field check catch shape drift, but not every semantic or message-schema change.

## 3. Observation, action, and API contract

### Observation v4: exactly 106 floats

The full field-level authority is `docs/OBSERVATION_ACTION_CONTRACT.md`; the executable mirrors are `Observation.FIELD_SPEC` and `python/sandboxai/contract.py:OBSERVATION_SPEC`. Static tests compare names, widths, indices, total size, tracked-enemy count, and action cardinalities across languages.

| Indices | Width | CURRENT meaning |
| --- | ---: | --- |
| `0–9` | 10 | Agent normalized position/velocity/forward and health |
| `10–18` | 9 | Primary contact relative position, distance, health, weapon readiness, combat flag, bearing, alive-enemy fraction |
| `19–32` | 14 | Secondary and tertiary contact relative position, distance, bearing, health, alive flag |
| `33–36` | 4 | Grounded, vertical velocity, in-cover, forward clearance |
| `37–50` | 14 | Visual/FOV/LOS flags, elevation, source, age, and confidence for tracked contacts |
| `51–59` | 9 | Loudest sound direction/distance/bearing/age/loudness/category and audible-event count |
| `60–64` | 5 | Nearest obstacle, visible/remembered contact counts, corpse count |
| `65` | 1 | Local perceived illumination—not the hidden lighting-mode label |
| `66–71` | 6 | Overflow-contact statistics and target priority/switch recency |
| `72–75` | 4 | Second sound plus hearing error/source-count summary |
| `76–83` | 8 | Perception-earned map coverage, visit age, remembered cover/danger, contact uncertainty |
| `84–105` | 22 | Three ranked **visible** objects: closest-surface position, distance, bearing, kind, per-slot visibility, visible-object count |

Important semantics:

- **CURRENT:** v1 is the stable prefix `0–32`; v2 appended `33–64`; v3 appended `65–83`; v4 appended `84–105`.
- **CURRENT:** booleans are `0/1`; most distances/counts are `[0,1]`; signed directions/angles may use `[-1,1]`.
- **CURRENT:** up to three highest-priority/nearest known contacts are individual; contacts beyond them appear only as aggregate fields `66–69`.
- **CURRENT:** from curriculum level 6, contact positions are live sightings or decaying beliefs produced by FOV/LOS/reaction/memory logic. Unperceived contacts are zeroed. Levels 1–5 deliberately use easier ground-truth contact data.
- **CONSTRAINT:** map ID, layout ID, lighting-mode ID, spawn list, global geometry, and hidden enemy positions are absent. Local illumination and perception-earned memory are allowed.
- **CURRENT:** `weapon_ready` is false during cooldown, reload, or an empty magazine. Ammo count and bloom are not observed.
- **CURRENT:** current PPO is feed-forward, so unobserved action history is not actually remembered by the network even though the simulator's belief fields carry some temporal state.
- **CURRENT:** `ACTIVE_OBSERVATION_MODE` is structured. RGB is disabled and frame stack is zero.

### Action: exactly six categorical components

| Index | Field | Values | Internal meaning |
| ---: | --- | --- | --- |
| 0 | move | `0,1,2` | backward, idle, forward (`-1,0,+1`) |
| 1 | strafe | `0,1,2` | left, idle, right |
| 2 | look yaw | `0,1,2` | left, idle, right |
| 3 | look pitch | `0,1,2` | down, idle, up |
| 4 | shoot | `0,1` | trigger released/held |
| 5 | jump | `0,1` | no jump/jump request |

`Action` also stores continuous `look_delta.x/y` for lossless human logging. Those values are **not** PPO actions. The canonical demonstration log array is eight values: four signed axes, shoot, jump, and two continuous look deltas.

**CURRENT compatibility.** Five-component old actions are padded with `jump=0`; seven-value old log arrays remain readable. This does not make old PPO checkpoints compatible: v1/v2 policies have different input or action-head shapes. Old demonstration files can be loaded and trained at their recorded observation width, but a resulting actor cannot warm-start the 106-input PPO unless dimensions and hidden layers match exactly.

### Engine step boundary

A single environment step is conceptually:

```text
(observation_t, action_t)
  -> update handling/movement/perception/opponents/combat at fixed dt
  -> event dictionary
  -> reward components and episode metrics
  -> (observation_t+1, scalar reward, done, info)
```

**CONSTRAINT:** preserve Gymnasium terminal semantics when changing auto-reset or vectorization. Never replace a terminal observation with the reset observation without retaining `terminal_observation` in `info`.

## 4. Canonical simulation

### Environments and conditions

| Asset | CURRENT inventory | Policy visibility |
| --- | --- | --- |
| Generated layouts | 14: `open_arena`, `scattered_cover`, `corner`, `cover_field`, `corridor`, `rooms`, `pillars`, `vertical`, `multi_room`, `ambush`, `sound_maze`, `combat_complex`, `crossfire_complex`, `randomized` | Geometry is perceived, layout ID is hidden |
| Authored maps | 16: `open_field`, `training_yard`, `blind_corner`, `cover_field`, `long_corridor`, `two_rooms`, `pillar_hall`, `catwalks`, `compound`, `ambush_alley`, `echo_maze`, `combat_complex`, `crossfire_lab`, `night_yard`, `foggy_field`, `random_ops` | Map metadata/ID is hidden |
| Scenarios | 12 generic calibration encounters: single/multi target, corner/cover/corridor fights, ambush, lost contact, sound-only, vertical and randomized arena. Invented weapon drills were removed. | Scenario ID is hidden |
| Lighting | `normal`, `low_light`, `night`, `fog`, `high_contrast`, `mixed` | Only local illumination and resulting perceptual effects |

Maps bind generated geometry variants, arena extent, lighting pools, ambience, and human-readable metadata. `resolve(map_id, seed)` and scenario resolution are seeded. Randomized conditions are episode configuration, not additional policy fields.

### Movement, aiming, navigation, and perception

- **CURRENT:** square analytic arena; normal half-extent 10 m, 3 m wall height, 60 Hz fixed step, 1,200-step/20-second timeout.
- **CURRENT:** agent health 100, speed 4.5 m/s, yaw/pitch turn rate 110°/s, pitch clamp ±80°, gravity −19 m/s², jump velocity 6 m/s, and reduced air control 0.45.
- **CURRENT:** movement is relative to agent forward/right. Diagonal intent is normalized. Firing can temporarily scale speed by weapon profile.
- **CURRENT:** `CharacterMotor` and `ArenaWorld` implement fixed-step collision, floor/platform landing, ceiling checks, gravity, and jumping without `PhysicsServer3D`.
- **CURRENT:** an 8-connected walkable-grid `NavigationGraph` (1.1 m cells) and A* fallback exist. Direct steering is used until an enemy is demonstrably stuck, avoiding pathfinding cost in open space.
- **CURRENT:** tactical levels include obstacles, cover, ranged enemies, FOV/LOS, reaction delay, occluded sound, decaying contact memory, spatial memory, target selection, elevation, and randomized light/fog.
- **CURRENT:** the early analytic enemy path and later `EnemyBrain` path are intentionally different difficulty regimes. At level 5 the tactical brain and ranged combat replace simple chase/melee behavior.
- **CONSTRAINT:** this is a useful controlled approximation, not a full commercial FPS physics/network/rendering model.

### Combat, weapons, and TTK

**CURRENT:** weapons are deterministic hitscan profiles selected by episode/scenario; there is one policy trigger and no weapon-select or reload action. At level 5+, handling adds auto/semi/pump trigger semantics, magazines, automatic empty reload, deterministic recoil, bloom from sustained fire/movement/airborne state, fire movement slowdown, damage falloff, pellet volleys, and head/body zones. Levels 1–4 retain cooldown-only behavior. Scripted enemies do not currently use the full handling model; self-play slots do.

| Profile | Damage | Mode / RPM | Mag / reload | Range | Role |
| --- | ---: | --- | --- | ---: | --- |
| Rifle | 25 | auto / 120 | 30 / 2.2 s | 15 m | controllable long-range baseline |
| Shotgun | 112 total, 8 pellets | pump / 83 | 6 / 3.0 s | 9.5 m | centered point-blank burst |
| Pistol | 20 | semi / 250 | 15 / 1.8 s | 12 m | deliberate precision backup |
| SMG | 14 | auto / 667 | 30 / 2.0 s | 11 m | fast close tracking, sharp falloff/bloom |

For a single-projectile body-shot weapon with no reload, ideal TTK is:

```text
shots_to_kill = ceil(target_health / damage_at_range)
TTK = (shots_to_kill - 1) * cooldown = (shots_to_kill - 1) / (RPM / 60)
```

The first hit occurs at `t=0`, so a one-shot kill is `0.00 s`. `sandboxai weapon-table` parses the Godot profile table rather than duplicating it. Current 100 HP ideal/handling-aware results for a stationary centered target are:

| Range | Rifle | Shotgun | Pistol | SMG |
| ---: | ---: | ---: | ---: | ---: |
| 2 m | 1.50 / 1.50 s | 0.00 / 0.00 s | 0.96 / 0.96 s | 0.63 / 0.63 s |
| 5 m | 1.50 / 1.50 s | 0.72 / 0.72 s | 0.96 / 0.96 s | 0.63 / 0.63 s |
| 8 m | 1.50 / 1.50 s | 1.44 / 1.44 s | 1.20 / 1.20 s | 0.81 / 0.81 s |
| 11 m | 1.50 / 1.50 s | out of range | 1.68 / 1.68 s | 1.35 / 2.61 s |
| 14 m | 2.00 / 2.00 s | out of range | out of range | out of range |

Cells are ideal/handling-aware TTK. These are model calculations, not human performance claims. The handling estimate assumes a stationary target and center aim; actual policy TTK also includes acquisition/reaction time, movement, misses, cover, hit zones, and decision latency.

### Reward

`RewardSystem` turns one tick's events into explicit components. Current constants:

| Component | CURRENT value/gate | Purpose and risk |
| --- | --- | --- |
| Hit | `+1.0 × weapon_damage/25 × pellet_fraction`, capped scale 6 | Normalizes flat hit credit across weapon roles |
| Kill | `+10` | Terminal combat objective |
| Damage dealt | `+0.02 / HP` | Denser aim feedback |
| Damage taken | `−0.05 / HP` | Survival/cover pressure |
| Death | `−10` | Terminal failure |
| Positioning | `0.05 × meters closed`, clamped ±`0.05`/tick | Helps approach while the target is beyond the 2 m enemy-attack threshold; can still distort paths |
| Aiming | `0.10 × alignment gain`, clamped ±`0.02`/tick | Paid only when target is actually hittable |
| Passivity | `−0.012`/tick with a valid target and no meaningful action | Prevents idle survival farming |
| Combat time | `−0.002`/alive combat tick | Prefers efficient completion |
| Useless trigger | `−0.1` | No target, blocked/out of range/bad aim, or legacy dead trigger |
| Near-target miss | `−0.01` | Keeps fine-aim exploration affordable |
| Handling-blocked trigger | `−0.002` | Does not punish correct automatic trigger holding like a real shot |
| Survival | configured `0.0`; only explicit non-combat opt-in | No combat timeout farming |
| Exploration | normalized new-cell gain; completion `+10` | Map Analyzer mode only, absent in combat |

**CURRENT anti-hacking measures.** Aim reward is target-hittable-gated; survival reward is zero in combat; time/passivity costs prevent hiding; hit credit is damage/pellet normalized; timeout is reported separately from loss; diagnostic skill metrics are not reward inputs.

**CURRENT exploit regression suite.** `tests/test_reward_exploits.gd` pins each of those properties as an invariant instead of prose: empty tick pays exactly zero; idling with a live target is negative every tick; closed aim/positioning oscillation cycles are net negative; shaping is clamped per tick and cannot outbid a hit or kill; an even HP trade is a loss; the flat hit bonus is fire-rate neutral per HP removed and clamped; pellet clipping earns only its landed fraction; useless shots cost more than honest near-misses and both stay negative; dying costs the full death penalty and `|death| >= kill`; the survival trickle stays opt-in; the component set is closed and the scalar reward equals its published decomposition, end to end over a real episode. Assertions are signs and orderings derived from `SandboxConfig`, so retuning stays possible but exploitability does not.

**CURRENT limitation:** at perception-gated levels, aiming, passivity, and combat-time gates still use simulator targetability/hittability rather than only the policy's current belief. This privileged training signal is not added to the observation, but strict perception-purity experiments must ablate or redesign it.

**CONSTRAINT:** best-checkpoint selection *defaults* to **mean shaped episode reward** (`checkpoint_selection_metric`), not win rate or generalization. Independent win/loss/timeout, accuracy, encounter duration/steps, condition spread, and skill metrics must therefore be reviewed for proxy exploitation; the separate weapon TTK table is a simulator diagnostic, not a measured policy-TTK metric.

**RESEARCH:** for new dense shaping, prefer task-success outcomes plus carefully tested potential-based shaping `F(s,s') = γΦ(s') − Φ(s)` where applicable. Add an exploit test before tuning coefficients: stationary firing, wall firing, spinning, corner camping, damage farming, deliberate timeout, target-switch churn, and map-coverage loops. Evaluate policies on unshaped success metrics.

## 5. Curriculum and episode distribution

| Level | CURRENT stage | New capability |
| ---: | --- | --- |
| 1 | `movement_and_aim` | Fixed stationary target; basic movement/aim |
| 2 | `moving_targets` | Moving target and seeded spawn variation |
| 3 | `basic_combat` | Attacking/strafe opponent and survival |
| 4 | `multi_enemy_combat` | At least three simultaneous threats and target selection |
| 5 | `cover` | Geometry, collision/navigation, tactical ranged enemies, full weapon handling/hit zones |
| 6 | `fov_and_occlusion` | Observation contact blocks become FOV/LOS-gated beliefs |
| 7 | `sound` | Perceivable footsteps, jumps, shots, impacts, deaths, ambience |
| 8 | `memory` | Decaying lost-contact memory and search |
| 9 | `navigation_and_vertical` | Platforms, jumping, elevation, more complex geometry |
| 10 | `randomized_environments` | Map/lighting/scenario/enemy-count distribution |
| 11 | `self_play` | Separate two-policy match hook; not reached by normal PPO curriculum |

**CURRENT auto curriculum.** `CurriculumDirector` deterministically maps master seed, stage, and stream index to `EpisodePlan`s; parallel slots receive disjoint streams and the next plans are staged before auto-reset. Promotion/demotion uses a rolling success window, minimum episodes, cooldown, and hysteresis. Stage-specific thresholds/minimums live in `curriculum_stages.py`; persisted stream/controller state lives in `checkpoints/curriculum_state.json`. `adaptive_curriculum=false` keeps deterministic planning but freezes the level. Standard training is capped at level 10.

**RESEARCH:** curriculum should target learning progress, not merely monotonically harder enemies. Periodically re-evaluate earlier levels to detect forgetting; mix boundary cases near the current competence threshold; keep an immutable holdout distribution; change only one major capability at a time when diagnosing regressions.

## 6. Human evidence, demonstrations, and behavior cloning

### Ethical manual TTK and demonstration protocol

**CURRENT safe direct demonstration path.** `sandboxai record` opens the local graphical Godot recorder. Human keyboard/mouse control passes through the same `Action`, `EnvironmentCore`, reward, and observation path as policies. It writes JSONL metadata then transitions containing observation, canonical action, next observation, reward, done, timestamps, episode/environment IDs, and info. This is the preferred source for BC because observations and actions are synchronized in the exact current contract.

**CURRENT manual TTK evidence pipeline.** `python/sandboxai/ttk.py` (schema `sandboxai.ttk_trials` v1) + `sandboxai ttk-report` implement the protocol below as data, not prose: `TTKTrial` records weapon, distance, target health, movement state, hit zone, acquisition/first-trigger/first-damage/lethal timestamps, shots fired/hit, outcome (`kill`, `target_escaped`, `tester_died`, `aborted`, `no_damage`), annotator/tester/session/build/frame-rate/start-convention and explicit `consent`. `validate_trial` enforces required fields, monotonic timestamps, `kill <-> lethal_time`, `shots_hit <= shots_fired`, `consent=True`, and **rejects any field whose name suggests a private/cheat source** (`memory_`, `process_`, `packet`, `server_authoritative`, `hidden_`, `injected`, `hook_`, `exploit`, `aimbot`). `TTKDataset` reports censoring separately, summarises per condition (3 m distance buckets) with median/IQM and deterministic seeded bootstrap intervals, and splits a tester/session holdout. `compare_with_simulator()` diffs human medians against `weapons.py` ideal and handling-aware TTK per weapon. There is deliberately no `arrays()` export: TTK trials are calibration evidence, not BC transitions.

The protocol the schema encodes:

1. Record the game/simulator screen and the tester's own input timing with consent, using public UI and ordinary controls only.
2. Freeze weapon/loadout, target health, distance band, movement state, hit zone, patch/version, frame rate, network context, and start convention.
3. Mark acquisition/first-trigger/first-damage/lethal frames from video or visible UI. Report acquisition time, first-shot latency, shots-to-kill, trigger-to-kill TTK, total encounter time, accuracy, and censoring/failed trials separately.
4. Repeat enough trials across people and conditions; preserve raw annotations, uncertainty, and outliers instead of retaining one “best” number.
5. Use aggregate distributions to calibrate reaction delay, spread, damage/falloff, cadence, and scenario difficulty in Godot. Keep a holdout set to test the calibration.

**CONSTRAINT:** manually measured external TTK is calibration evidence, not automatically a BC transition dataset. Direct BC needs each action aligned with the same 106-field SandboxAI observation. Converting video/ordinary inputs from another game would require explicit annotation/state estimation and action quantization; no such importer is implemented. Never fill unavailable fields with private/server data—use neutral “unknown” encodings or do not claim contract compatibility.

### BC implementation

- **CURRENT:** JSON or JSONL `sandboxai.demonstrations` schema; finite-value and action validation; legacy action conversion.
- **CURRENT:** two-layer Tanh MLP, default `106 → 128 → 128`, with one categorical head per action component (sizes `3,3,3,3,2,2`). Loss is the sum of six cross-entropies.
- **CURRENT:** seeded **episode/group-aware** train/validation split (`split_strategy` = `auto`|`episode`|`transition`), Adam, validation loss, component accuracy, exact-six-component accuracy, early stopping, CSV/JSONL metrics. `auto` splits by episode group (`run_id|environment_id|episode_id`) when the dataset has at least two groups and otherwise falls back to the transition shuffle while recording `degraded_reason`; `episode` raises rather than leaking; `transition` must be asked for explicitly.
- **CURRENT:** every BC run writes `dataset_report.json` (split report with `leakage_free`, dataset fingerprint `blake2b:<hex>`, episode structure, action histograms, duplicate fraction, observation-range violations) and stamps the split + fingerprint into each checkpoint. `sandboxai inspect-dataset --statistics` prints the same report without training. `BCConfig.require_contract_observations` and `max_duplicate_fraction` fail a bad corpus before it becomes a model.
- **CURRENT:** resumable trusted PyTorch `sandboxai.bc.v1` `.pt` files containing model and optimizer state; atomic `latest.pt`, `best.pt`, periodic epochs.
- **CURRENT:** compatible hidden layers and categorical heads can initialize SB3 PPO's actor. PPO's value network remains newly initialized. Any shape/layout mismatch raises.
- **CONSTRAINT:** BC is supervised imitation, not offline RL. It does not infer counterfactual returns or improve beyond dataset support by itself.
- **RESEARCH:** plain BC suffers covariate shift and compounding errors. Episode/group holdouts now exist; next, evaluate DAgger-style corrective data collected on learner-visited simulator states. Preserve expert consent and never silently blend evaluation episodes into training.

## 7. PPO and online learning

**CURRENT algorithm.** SB3 PPO alternates fixed on-policy rollout collection with multiple minibatch optimization epochs over a clipped objective. No replay buffer or offline RL algorithm is used.

| Default | Value |
| --- | ---: |
| Environments | 8, in `env_workers` Godot processes (default 1; `--env-workers N|auto` shards them) |
| Total timesteps | 1,000,000 |
| Rollout length | auto (`0`): targets ≈16,384 transitions/update, capped at 2,048/environment; explicit positive values are exact |
| Batch size | 256 |
| Learning rate | `3e-4` |
| Discount / GAE | `0.99` / `0.95` |
| Entropy coefficient | `0.01` |
| PPO clip | `0.2` |
| Network | separate policy/value `[128,128]` Tanh MLPs |
| PPO epochs | 10 (explicitly configured and persisted) |
| PyTorch CPU threads | bounded auto (`0`): CPUs left after bridge workers, capped at 4; explicit positive values are exact |
| Periodic save/evaluation | every 100k / 50k timesteps |
| Normal evaluation | 20 deterministic episodes, up to 8 env slots |
| Curriculum | auto, adaptive, starts level 1 |

**CURRENT rollout/resource schedule.** `rollout_length=0` scales the per-environment horizon down as environment count rises, preserving approximately the historical 8 × 2,048 = 16,384-transition aggregate rollout (and preferring a nearby minibatch-divisible horizon when that changes it by at most 10%). This prevents environment-count scaling from silently reducing a 500k/48-env run to six collect→update cycles; `config.json`, startup telemetry, and the training profile record requested/resolved horizon, expected update count, scheduled full-rollout timesteps, and overshoot. A positive `rollout_length` always preserves the requested legacy geometry. `torch_threads=0` similarly resolves to a bounded host-aware pool after reserving CPUs for Godot workers; a positive value is exact.

**CURRENT device path.** `device=auto` uses CUDA when PyTorch exposes it, otherwise CPU. `inference_device` may explicitly place rollout/evaluation forwards on CPU while PPO updates remain on CUDA; this avoids tiny per-step host/device transfers when they dominate. Changing inference device changes the RNG/device trajectory and is reproducible as a different experiment, not bit-identical to the old one.

**CURRENT resume.** `PPO.load(..., env=..., device=...)` restores model/optimizer/timestep state; `learn(reset_num_timesteps=false)` continues. The integrated pipeline also reloads curriculum state and episode stream.

**RESEARCH priorities.** Before changing algorithms, run multiple seeds and inspect entropy, KL, explained variance, success, timeout, and per-condition results. Tune rollout/batch size only after measuring simulator throughput and update/rollout wall time. For partial observability and hidden ammo/bloom history, compare explicit frame/action history or a recurrent policy against the feed-forward baseline; either requires a versioned model/input decision and must not silently alter the 106-field contract.

### Offline-to-online boundary

- **CURRENT:** demonstrations → BC actor → optional PPO warm start is the only offline-to-online path.
- **PLANNED:** log dataset provenance, policy/source, episode group, condition, contract version, and behavior probabilities where available.
- **RESEARCH:** if offline RL is justified, benchmark BC first and use an algorithm designed for static-data distribution shift (for example CQL/IQL-style conservatism), not ordinary off-policy Q-learning on fixed human data. Validate exclusively in Godot before any external discussion. Human data diversity/support, not dataset size alone, is the limiting factor.

## 8. Evaluation, checkpointing, replay, and model selection

### Evaluation

- **CURRENT:** evaluation freezes weights and uses deterministic policy actions plus explicit episode seeds.
- **CURRENT:** every evaluation counts policy-side action outputs before IPC and reports `policy_shoot_requests`/rate; sparse, RNG-neutral actor-head inspection reports stochastic `P(shoot=1)`. Reports compare those with engine `trigger_pulls` and `shots_fired` to localize a zero-shot result to policy argmax, bridge delivery, or weapon discharge without changing reward/combat behavior.
- **CURRENT:** serial and vector evaluation run the same seed list; vector execution batches plans and restores result order.
- **CURRENT:** normal and checkpoint-battery bridges persist across boundaries. In auto mode the two independent Godot processes simulate concurrently; access to the shared model is locked and both jobs join before selection/early stopping.
- **CURRENT:** checkpoint batteries can run a 24-episode condition sample, held-out generalization cells (one episode/cell default), diagnostic skill summaries, optional replays, and optional league matches.
- **CURRENT:** condition reports expose win/loss/timeout, reward, map/scenario/lighting/enemy count, skill groups, worst conditions, and spread instead of only a global mean.
- **CURRENT:** the checkpoint-selection rule is explicit, configurable and recorded (`python/sandboxai/selection.py`). `CheckpointSelectionRule(metric, goal, min_delta)` is built from `checkpoint_selection_{metric,goal,min_delta}`, validated at config load, written into `evaluations/best.json` next to the score it produced, and mirrored in the run manifest and the `training_start` telemetry event. Dotted metrics reach the mirrored report sections (e.g. `condition_evaluation.mean_win_rate`); `goal=min` selects on quantities like TTK or deaths; `min_delta` suppresses near-noise churn. Defaults reproduce the historical behaviour exactly: strictly higher `mean_episode_reward`.
- **CURRENT:** a resume compares the recorded rule with the configured one. An incomparable rule (different metric or direction) restarts selection and emits `checkpoint_selection_rule_changed` instead of comparing win rate against shaped reward; a metric the run does not produce counts as no improvement and emits `checkpoint_selection_metric_missing`.
- **CONSTRAINT:** battery/generalization evidence is reported and *reachable* by the rule, but the default rule still selects on the normal evaluation's mean shaped reward.

**PLANNED model-selection rule (remaining).** Predeclare a lexicographic or constrained score such as: minimum win rate and maximum timeout rate on core tasks; then worst-condition/generalization success; then median/IQM success or TTK; shaped reward only as a tie-breaker. Keep a final untouched test seed/map split. Report all training seeds, not the luckiest run.

**RESEARCH reporting.** With limited runs, store all per-seed scores and add stratified bootstrap intervals, interquartile mean, probability of improvement, and performance profiles. Never tune on the final holdout.

### Artifacts and compatibility

| Artifact | CURRENT format/boundary |
| --- | --- |
| PPO checkpoints | SB3 `.zip`: periodic `ppo_*`, `latest.zip`, `best_eval.zip`, run `final.zip`, battery `policy.zip` |
| BC checkpoints | PyTorch `.pt`, format tag `sandboxai.bc.v1`, model + optimizer, carrying the split report + dataset fingerprint |
| BC data provenance | `dataset_report.json` (split/leakage report, fingerprint, statistics) next to the BC run's `config.json` |
| Human TTK evidence | JSONL `sandboxai.ttk_trials` v1 trial files (optional leading metadata object); `sandboxai.ttk_simulator_comparison/v1` comparison documents |
| Config/provenance | `config.json`, `run_summary.json`, `warm_start.json`, `run_manifest.json` (`sandboxai.run_manifest/v3`: host/Godot/code-dirty/parallelism/selection-rule provenance plus run status and checkpoint inventory), curriculum state |
| Evaluation | per-evaluation-step `summary.json`/episode CSV; atomic battery `report.json`; rolling `latest.json`/`best.json` |
| Telemetry/profile | JSONL, TensorBoard events, optional `training_profile.json` |
| Replay | JSONL format v1; light stores deterministic setup/actions/reward/done, detailed also stores observations |

**CONSTRAINT:** checkpoint compatibility is bounded by observation/action meaning and shape, network layout, Python/SB3/PyTorch serialization, and code version. Dependencies currently have lower bounds rather than a lockfile; archive a resolved environment for important experiments. SB3 archives and BC `.pt` files must be treated as **trusted local artifacts only** because their loaders may deserialize Python objects. League fingerprint checks prevent a supposedly frozen opponent file from changing, not malicious input.

## 9. Self-play and league status

- **CURRENT:** `SelfPlayEnvironmentCore` accepts both policy actions for one tick, applies both movements before combat, and gives both slots symmetric 106-float observations, handling configuration, independent seeded RNG streams, per-slot reward/metrics, and no privileged opponent fields.
- **CURRENT:** fire is **simultaneous**. Both slots pull their trigger and both volleys are resolved against the pre-tick world (positions, health, alive flags); damage is applied afterwards, clamped to the target's remaining health exactly as `AgentState.take_damage` clamps it. A lethal exchange therefore kills both agents and ends `draw`, neither trigger is cancelled by the other's outcome, and near-miss geometry is sampled before damage lands. Pinned by symmetric-duel tests (mutual kill, slot symmetry, clamped/mirrored damage, non-lethal trade, symmetry under full weapon handling).
- **CURRENT:** Python can load independent frozen checkpoints, register snapshots, verify fingerprints/weight independence, sample opponent pools, schedule deterministic tournaments, and compute reporting-only Elo.
- **CURRENT:** opponent sampling is deterministic by construction. `SelfPlayCoordinator` owns a seeded generator (never the global `random` module) and supports `uniform`, `latest`, `recency_weighted` and `round_robin`; `reset_sampling()` rewinds the stream, `choose_opponent_checkpoint()` selects without loading a model, and `sampling_snapshot()` records strategy/seed/draws/pool for manifests. `SelfPlayConfig.opponent_strategy`/`opponent_seed` put the rule in the config snapshot (`SelfPlayCoordinator.from_config`). `League` was already seeded and is unchanged.
- **CURRENT:** checkpoint-time league evaluation is optional and off by default.
- **CONSTRAINT:** there is no public `self-play` training command, population optimizer, PFSP loop, exploiter role, or automatic promotion of league policies. Level 11 is a match/evaluation hook, not part of standard single-policy PPO.
- **PLANNED:** create an explicit learner-vs-frozen-opponent training VecEnv, snapshot cadence, immutable policy IDs, recent/historical/best pools, and regression gates.
- **RESEARCH:** compare uniform/latest/prioritized sampling with PFSP-style opponent selection. Include historical opponents and exploiters to reduce cycling; report matchup matrices and exploitability indicators, not Elo alone.

## 10. Telemetry, profiling, and operator tools

- **CURRENT telemetry:** asynchronous JSONL writes, episode metrics, resource snapshots, TensorBoard, run summaries, and experiment comparison/summarization.
- **CURRENT diagnostics:** aim, reaction, awareness, positioning, movement, combat, survival, and exploration skill groups. They are measurements, never reward terms.
- **CURRENT profiling:** Python rollout/update/callback/model/bridge timings (including the exact SB3 optimizer call) plus opt-in Godot parse/simulation/encode/write aggregates and request/response byte counters. Top-level phase totals aggregate sharded `workerN` buckets: serial encode/decode CPU phases are summed and overlapping request/wait windows use the slowest-worker critical path.
- **CURRENT benchmark:** environment-count sweeps and five comparable suites—early curriculum, advanced curriculum, perception combat, map analyzer, weapon handling—at `1/4/8/16/32/64` environments. It reports measured throughput, p50/p95 vector-step latency, and resources only. The default wire mode matches PPO's compact non-terminal infos; `benchmark --full-infos` explicitly measures diagnostic serialization instead.
- **REMOVED Control Center scene:** the in-simulator operator UI (watch/human modes, perception and observation inspectors, result tabs) was deleted together with `scenes/control_center.tscn` and `scripts/control_center/`; training is operated through the headless desktop application below.
- **CURRENT run inspection:** `python/sandboxai/run_inspection.py` + `sandboxai inspect-runs` are a strictly read-only backend over the run directory layout (state with the evidence it came from, progress, checkpoint/evaluation inventory, log sizes, manifest provenance, `problems` vs `warnings`). Documents are versioned (`sandboxai.run_report/v1`, `sandboxai.run_index/v1`); the Control Center consumes them instead of re-implementing the layout in GDScript.
- **CURRENT Control Center (headless desktop):** `python3 main.py` (or `sandboxai control-center-desktop`) is the single operator application, a local Tkinter window (`python/sandboxai/control_center_desktop.py`). It is a thin view over `sandboxai.adapter.SandboxAIAdapter`, which only launches the existing `train`/`benchmark`/`evaluate` CLI commands and reads their existing artifacts (`run_inspection.py`, `logs/training.jsonl`, `status.json`, `evaluations/*`, `benchmark.json`); it owns no RL logic and duplicates no CLI/business logic. Dashboard, Training (launch deck with an env/worker/device form, a Steps-or-Time budget and lifecycle actions incl. restart-from-checkpoint), Benchmarks (Auto/Push/Custom staged benchmark pipeline with a measured, persisted recommendation), Evaluations, Runs/Checkpoints, Stats (what the policy receives, decoded
from a recorded replay), System/Telemetry, and Settings pages (theme, shell layout, density, motion, movable cards, savable presets); bounded incremental telemetry/log polling; non-blocking cooperative stop plus a scoped force-stop; unavailable metrics (e.g. no Godot binary, no CUDA) are shown as such, never estimated. It never renders the game and no training module imports it. See [`docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md`](docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md).
- **CURRENT replay:** light deterministic replays for routine capture; detailed observations for debugging contract or nondeterminism. `interesting` mode is default and capped at 64/run.

## 11. CLI, configuration, testing, and reproducibility

### Runtime and public CLI

| Boundary | CURRENT reality |
| --- | --- |
| Python | `>=3.11`; package version `0.3.0` |
| Godot | project feature `4.7`; CI downloads exact `4.7.2-stable` on Windows and Linux |
| Rendering | graphical project uses Mobile renderer; headless bridge creates no views |
| Core dependency | NumPy |
| Optional training | PyTorch, Gymnasium, SB3, TensorBoard, psutil |
| Static GDScript tooling | gdtoolkit/gdlint |
| WSL | Windows-path translation, remembered Godot executable, and Windows process-launch fallback are implemented |

Public subcommands are:

```text
install  train  resume  evaluate  record  control-center-desktop
bc-train  inspect-dataset  inspect-runs  benchmark  benchmark-suites
benchmark-pipeline  replay  curriculum  weapon-table  ttk-report  ttk-status
validate-runtime  compare-experiments  summarize-experiment  smoke-test
hardware-wizard
```

Do not invent `test`, `self-play`, `replay-info`, `replay-play`, or `compare` commands.

`TrainingConfig` is the serializable authority for PPO/run settings. CLI flags override JSON config fields. An explicitly validated Godot executable can be remembered in gitignored `.sandboxai/settings.json`. Run outputs default under gitignored `training/`.

### Determinism model

- **CURRENT:** episode seeds feed local Godot RNGs; episode-plan generation is a pure deterministic function with disjoint streams.
- **CURRENT:** simulation uses explicit `1/60` dt and analytic state/collision; deterministic recoil and bloom consume no RNG.
- **CURRENT:** replay setup stores seed, map/scenario/lighting, enemy count, curriculum, policy/checkpoint identity, actions, and optional observations.
- **CURRENT:** CI executes Godot tests on Windows and Linux (`.github/workflows/godot-tests.yml`), which is important cross-platform evidence.
- **CURRENT:** CI also runs the Python suite (`.github/workflows/python-tests.yml`): a `ruff`/pyflakes-equivalent lint pass, a numpy-only "core" job that guards the deliberately tiny hard dependency set, and a "full" job with the training extras installed on both Windows and Linux.
- **CONSTRAINT:** Godot's general physics engine is officially nondeterministic; this project avoids it for canonical state. Floating-point/compiler/platform differences can still exist.
- **CONSTRAINT:** PyTorch does not promise complete reproducibility across releases/platforms/devices. Record seeds, commit, Godot build, Python package versions, device, CPU thread settings, and hardware. Compare deterministic replay hashes/metrics within a declared boundary.
- **CONSTRAINT:** any future parallel reduction must preserve per-environment RNG ownership and deterministic result ordering. Never share one mutable RNG across workers.

### Validation snapshot and commands

At source commit `d0d60a2` the repository's GitHub checks were green for Python and Godot 4.7.2 on Windows and Linux. The most recent engineering pass (baseline `23115e0`) measured, with CPU PyTorch installed:

- `python -m pytest -q`: 659 passed / 1 skipped at the baseline, 779 passed / 1 skipped / 502 subtests after the pass. No test was removed, weakened or skipped to get there.
- `gdlint scripts tests` and the real-GDScript-grammar parse check (both run through `python/tests/test_gdscript_static.py`): passed, including the new GDScript suites.
- Godot itself could not be installed in that environment, so `tests/run_tests.gd`, live-bridge smoke tests and any Godot throughput number were **not** executed locally; they rest on CI. New GDScript logic was instead checked with `gdscript_analysis`, gdlint, the grammar parser, and by re-deriving every numeric reward claim in Python from the real `SandboxConfig` constants.
- `validate-runtime`: Godot unavailable locally, so the 10 live checks were skipped.
- Headless throughput: still not measured; no Godot performance number is asserted anywhere in this file. `tools/bridge_scaling_probe.py` measures the transport with a synthetic workload only.

Canonical checks:

```bash
python -m pytest python/tests -q
gdlint scripts tests
godot --headless --path . --script res://tests/run_tests.gd
sandboxai smoke-test --device cpu
sandboxai validate-runtime --godot-executable <Godot-4.7.2> --json
sandboxai benchmark-suites --godot-executable <Godot-4.7.2>
```

## 12. Godot 4.7.x scaling: current vs supported vs potential

### Capability matrix

| Topic | CURRENT repository | Godot 4.7.x supports | PLANNED / RESEARCH decision |
| --- | --- | --- | --- |
| Headless | `--headless --script`, Dummy display/audio, no simulation views | `--headless`; `--disable-render-loop`; `--fixed-fps`; exported release binaries | Benchmark editor vs exported release. Do not add flags without measuring; bridge stepping already supplies fixed dt. |
| GDScript | Typed GDScript analytic hot path | Easy iteration; optional static typing; lower peak speed than native code | Profile first. Move only proven kernels to C#/GDExtension; preserve a reference implementation and parity tests. |
| Threads | Simulation environments are serial *within* a shard; requests to all shards are issued before any reply is awaited (`send`/`receive` split on the transport), so N processes simulate concurrently; Python threads drain pipes/telemetry and overlap two eval processes | `Thread`, `WorkerThreadPool`, mutex/semaphore; group tasks for expensive independent work | Prefer process sharding first. Worker tasks may help only if each environment step is heavy enough to beat scheduling/synchronization overhead. |
| Thread safety | No threaded scene-tree simulation | Active scene tree is not generally thread-safe; servers have documented rules; resources/shared containers require care | If trialed, worker code may mutate only its own `EnvironmentCore`; stage immutable inputs, wait, then gather in index order on the main thread. |
| Physics | Analytic `ArenaWorld`/`CharacterMotor`; no PhysicsServer-driven canonical state | Fixed physics callbacks exist, but engine physics is not deterministic | Keep analytic path for research reproducibility. Use engine physics only behind a separately versioned environment. |
| IPC | Strict request/response newline JSON; compact info mode; batched observations/actions; packed observations become generic arrays for `JSON.stringify()` | Packed arrays and Variant binary serialization exist | Profile JSON bytes/encode time. If material, add a versioned length-prefixed binary data plane; keep JSON control/debug path. |
| Isolation | **CURRENT:** `sharded_env.py` splits the environments over `env_workers` independent headless processes (`--env-workers N|auto`, `TrainingConfig.env_workers`); separate persistent normal/battery eval processes | Multiple independent headless processes are ordinary OS isolation | Sharding is result-preserving: shard *k* owning global envs `[off, off+m)` is launched with `--seed base+off`, matching `SimulationManager`'s `base_seed + index` contract, so trajectories are identical and only wall time changes (`test_sharding_is_result_preserving`). A shard failure raises `ShardFailure` naming the shard and closes the rest. Still to measure on target hardware. |
| Navigation | Custom deterministic grid/A* | NavigationServer queries are thread-friendly; shared AStar objects are not | Current custom graph is the reproducibility baseline. Only migrate after parity/performance evidence. |
| High-frequency stepping | One synchronous round trip per 60 Hz conceptual tick | Engine can run without real-time synchronization | Consider action-repeat/internal-step batching only as a new MDP version; accumulate rewards and terminal state correctly. |

**CONSTRAINT:** Godot's Variant binary format is an engine format and its 4.7 documentation carries an outdated-page warning. For a durable Python/Godot wire protocol, a small explicit schema (`magic`, protocol version, message type, payload length, fixed little-endian float32/int fields) is safer than coupling Python to arbitrary Variant decoding. Never deserialize full objects from untrusted bytes.

### Target hardware strategy

Target: **i7-12700F (12 cores/20 threads), RTX 4060 Ti 8 GB, 32 GB RAM, Windows 11 plus Ubuntu/WSL, Godot 4.7.2**.

Confirmed 2026-09-30: this is the project author's actual machine (WSL/Ubuntu with a Linux Godot build, project files under Windows/OneDrive), not a hypothetical target - `tools/wsl/run_full_validation.sh` / `docs/RUN_LOCAL_VALIDATION.md` exist to turn the PLANNED items below into real, measured, timestamped files from that exact machine instead of estimates.

1. **CURRENT expectation:** structured simulation is CPU/IPC-bound; the tiny 106→128→128 MLP often makes per-step CPU inference more sensible than CUDA. The RTX is most useful for PPO update minibatches, BC, and future CNNs—not Godot's headless analytic state.
2. **PLANNED baseline matrix:** measure native Windows Python+Godot, WSL Python+Linux Godot, and (if needed) WSL Python+Windows Godot. WSL interop is supported but must not be assumed free.
3. **PLANNED sweep:** first run existing `1,2,4,8,16,24,32,48,64` single-process benchmarks. Record steps/s, episodes/s, p50/p95 step latency, JSON bytes, CPU/RAM, and profile buckets.
4. **CURRENT tooling, PLANNED measurement:** `sandboxai benchmark --worker-counts 1,2,4,8` sweeps worker processes per environment count and reports measured steps/s per configuration; `recommended_worker_count` (`--env-workers auto`) estimates physical cores and reserves two for the trainer. `tools/bridge_scaling_probe.py` isolates transport/process-parallelism scaling **with a synthetic workload** - it is a transport probe, never a Godot measurement. No target-hardware Godot scaling number is claimed yet.
5. **CONSTRAINT:** control oversubscription. Set/measure `torch_threads`, BLAS/OpenMP threads, Godot process count, and eval process count together. More logical threads can reduce throughput through contention.
6. **PLANNED selection:** choose the knee of throughput vs latency/RAM, not the largest environment count. Re-run advanced perception/weapon suites, not only the cheap level-3 baseline.
7. **RESEARCH transport:** binary framing or shared memory is justified only if JSON encode/copy is a measured bottleneck. Shared memory likely requires native support and a ring-buffer synchronization design; it is not current Godot/GDScript code.

## 13. Known limitations and truth warnings

- **CURRENT limitation:** no checked-in trained policy or target-hardware run establishes learning quality or throughput.
- **CURRENT limitation:** only structured observations are trained; no RGB, CNN, frame stack, recurrent PPO, or visual sim-transfer pipeline.
- **CURRENT limitation:** `env_workers` defaults to 1, so out of the box one bridge process still steps N environments serially; the sharded path exists and is tested but its throughput gain is unmeasured on target hardware.
- **CURRENT limitation:** JSON serialization copies 106 floats/environment/tick plus metadata; compact infos reduce but do not remove this cost.
- **CURRENT limitation:** feed-forward PPO cannot infer long hidden histories; ammo and bloom are consequences of actions but are not explicit observations.
- **CURRENT limitation:** BC splits are episode-aware, but a dataset with no episode structure at all still degrades to a transition shuffle (reported as `degraded_reason`, never silent).
- **CURRENT limitation:** the *default* checkpoint-selection rule is still mean shaped reward, so reward hacking/generalization regressions can win selection unless the rule is configured otherwise.
- **CURRENT limitation:** scripted enemies do not use the same complete handling layer as the agent.
- **CURRENT limitation:** only three contacts have individual state; overflow contacts are aggregates.
- **CURRENT limitation:** normal dependencies are not fully pinned/locked across platforms. `requirements-lock-linux-py311-cpu.txt` is a real, generated-and-installed reference snapshot for Linux/CPU (the one platform this environment could actually produce and verify), but Windows/macOS/CUDA installs legitimately resolve different wheels (torch especially) and have no equivalent lock file yet; repeatable cross-platform experiments still need their own environment snapshot.
- **CURRENT limitation:** bridge and replay formats have weak evolution/negotiation compared with the observation contract.
- **CURRENT limitation:** self-play is match/league/evaluation infrastructure, not end-to-end population training. (Same-tick lethal fire is no longer slot-order-biased: fire resolves simultaneously.)
- **CURRENT limitation:** Godot must be installed separately. Neither this nor the previous audit could run local live validation or a benchmark; the GDScript suite and all live-bridge/throughput claims rest on CI (exact 4.7.2, Windows + Linux). GDScript changes in this pass were checked with `gdscript_analysis`, gdlint and the real GDScript grammar parser, and their numeric claims re-derived in Python against the real `SandboxConfig` constants.
- **CONSTRAINT:** there is no Roblox server connection, private API, Studio plugin, live input automation, memory reader, packet inspector, client modifier, external-game adapter, or transfer claim in this repository. The bounded calibration helpers (detect/launch/focus/screenshot and the hand-typed calibration file) are the only code that touches the real client; their exact limits are tabulated in `docs/TTK_TESTING_REFERENCE.md`.
- **TRUTH WARNING:** older milestone prose can be stale. The current source still contains a controlled calibration simulator with vertical worlds, navigation and handling models; those numbers are not claims about Roblox TTK Testing. The source-backed boundary, calibration gaps and excluded mechanics are maintained in `docs/TTK_TESTING_REFERENCE.md`; re-check that document and the current source before repeating gameplay claims.

## 14. Research directions that fit this system

| Area | Practical next experiment | Guardrail / failure to watch |
| --- | --- | --- |
| PPO | Multi-seed baseline; ablate BC initialization, entropy, rollout size, and recurrence one at a time | Seed lottery; update time hidden by simulator time; reward-only conclusions |
| BC/imitation | Episode-group split, condition holdout, learner-rollout error analysis, then DAgger corrections | Leakage and covariate shift; over-represented idle actions |
| Curriculum | Learning-progress/boundary sampling plus periodic earlier-stage regression tests | Premature promotion, oscillation, forgetting, curriculum overfitting |
| Self-play | Frozen historical pool + recent/best + PFSP-style sampling; immutable snapshots | Latest-vs-latest cycling, shared weights, Elo as false objective |
| Vector scale | Multi-process Godot shards with centralized batched inference | CPU/thread oversubscription, nondeterministic gather, process leaks |
| Rewards | Adversarial behavior suite and independent task metrics; test potential shaping | Camping, damage farming, trigger spam, timeouts, shaping cycles |
| Exploration | Use existing spatial-memory coverage as a baseline; compare count/hash or RND bonus only in sparse exploration tasks | “Noisy TV”/stochastic novelty, bonus dominating combat objective |
| Model selection | Core success constraints + worst-condition/generalization + multi-seed intervals | Tuning on holdout, choosing by shaped return, hiding failed seeds |
| Offline/online | BC baseline → conservative offline method only if dataset support warrants → Godot online PPO | OOD action overestimation and unsupported counterfactual claims |
| Determinism | Replay hashes across repeated runs, processes, Windows/Linux, CPU/GPU; record environment lock | Confusing seeded with universally bitwise deterministic |
| Sim transfer | Calibrate parameter distributions from manual human measurements; validate held-out sim variants first | Simulator exploitation, unit/latency mismatch, private information |
| RGB | Versioned visual sensor, offscreen render benchmark, binary transport, encoder pretraining, multimodal ablations | JSON explosion, render bottleneck, shortcut textures/labels, 8 GB VRAM |

### RGB progression (not implemented)

1. **RESEARCH:** define camera, resolution, color space, latency, frame rate, action repeat, and whether structured state remains as privileged training-only input. Version this separately from observation v3.
2. **RESEARCH:** prove Godot 4.7.2's chosen offscreen rendering path on Windows and Linux; `--headless` disables ordinary display/rendering behavior, so do not assume current launch flags produce pixels.
3. **PLANNED only after proof:** replace JSON pixel transport with binary/shared memory; benchmark end-to-end capture, transfer, augmentation, inference, and update.
4. **RESEARCH:** pretrain or jointly train a CNN/ViT encoder, compare pixels-only vs structured-only vs multimodal, and test texture/light/map holdouts for shortcuts.
5. **CONSTRAINT:** keep the existing 106-float model family and replays versioned and usable. RGB is a new modality, not a silent contract mutation.

## 15. Roadmap with exit gates

| Priority | Status | Deliverable | Exit gate |
| ---: | --- | --- | --- |
| P0 | **PLANNED** | Reproduce full Python/Godot suite and live runtime validation on target machine | Exact versions archived; all checks green; repeated replay agrees |
| P0 | **PLANNED** | Measure five benchmark suites on all three relevant Windows/WSL runtime combinations | Raw JSON/CSV committed to experiment storage; no estimated numbers |
| P1 | **CURRENT (tooling) / PLANNED (corpus)** | Human TTK protocol + local Godot demonstration corpus | `sandboxai.ttk` + `ttk-report` enforce consent/provenance/condition metadata, tester/session holdout and bootstrap uncertainty; no corpus is collected yet |
| P1 | **CURRENT (split) / PLANNED (baseline)** | Episode/group-aware BC split and contract/provenance validation | No episode overlap (`leakage_free` in `dataset_report.json`); old-format behavior tested; BC baseline still unreported |
| P2 | **PLANNED** | Multi-seed BC→PPO baseline through levels 1–10 | Learning curves, all seeds, win/loss/timeout, TTK, worst conditions, generalization |
| P2 | **CURRENT (mechanism) / PLANNED (rule choice)** | Success/generalization-aware checkpoint selection | Adversarial reward tests exist and pass; the rule is predeclared, validated, recorded in `best.json`/manifest and tested - choosing a non-default rule still needs multi-seed evidence |
| P3 | **CURRENT (implementation) / PLANNED (measurement)** | Multi-process sharded VecEnv | Deterministic seed mapping and crash cleanup done and tested (`sharded_env.py`, `--env-workers`); measured throughput gain on target hardware still outstanding |
| P3 | **RESEARCH** | Binary bridge only if serialization is material | Protocol version/parity/fuzz tests; JSON debug fallback; measured end-to-end gain |
| P4 | **RESEARCH** | Recurrence/history and DAgger | Beats feed-forward/BC baselines on lost-contact and handling holdouts across seeds |
| P5 | **PLANNED** | End-to-end self-play trainer using frozen league pools | Immutable snapshots, matchup matrix, historical regression and anti-cycling evidence |
| P6 | **RESEARCH** | Versioned RGB simulator path | Stable capture + transport budget on 8 GB GPU; visual holdout generalization |

## 16. Agent working rules and source map

### Before changing code

1. Read this file, then inspect the named authority—not just prose docs.
2. Run `git status`; preserve the session branch and user changes.
3. If touching observations/actions, update both languages, the canonical table, observation groups, recorder/dataset compatibility, replays, and drift tests together.
4. If touching episode/reset/vector logic, test terminal observations, partial resets, staged-plan consumption, and seed/order invariance.
5. If touching reward, add a case to `tests/test_reward_exploits.gd` and inspect independent success metrics. Adding a reward component fails `test_reward_component_set_is_closed` until it is documented there.
6. If touching performance, profile and benchmark the exact workload before and after; never report estimates as measurements.
7. If touching serialization/checkpoints, state the trust and version boundary and test old/new behavior explicitly.

### Fast source index

| Question | Authority |
| --- | --- |
| Observation/action semantics | `scripts/core/{observation,action}.gd`, `python/sandboxai/contract.py`, `docs/OBSERVATION_ACTION_CONTRACT.md` |
| Episode order and reward events | `scripts/env/environment_core.gd` |
| Batch/reset/auto-reset | `scripts/core/simulation_manager.gd`, `scripts/rl/rl_adapter.gd` |
| Weapon and TTK truth | `scripts/weapon/weapon_state.gd`, `python/sandboxai/weapons.py` |
| Reward constants/combination | `scripts/core/sandbox_config.gd`, `scripts/reward/reward_system.gd`, exploit invariants in `tests/test_reward_exploits.gd` |
| Curriculum semantics/plans | `scripts/core/curriculum_config.gd`, `python/sandboxai/{curriculum_stages,auto_curriculum,randomization,pipeline}.py` |
| Bridge/process behavior | `scripts/rl/rl_server.gd`, `python/sandboxai/godot_env.py` |
| PPO/config/checkpoints | `python/sandboxai/{config,ppo,checkpoint_eval}.py` |
| Checkpoint-selection rule | `python/sandboxai/selection.py` |
| Multi-process env sharding | `python/sandboxai/{sharded_env,godot_env}.py`, `tools/bridge_scaling_probe.py` |
| Run provenance/manifest | `python/sandboxai/manifest.py` |
| Read-only run inspection | `python/sandboxai/run_inspection.py` |
| Demonstrations/BC | `scripts/recording/`, `python/sandboxai/{dataset,bc}.py` |
| Human TTK evidence | `python/sandboxai/ttk.py` |
| Evaluation/generalization | `python/sandboxai/{evaluation,checkpoint_eval,conditions,generalization}.py` |
| Self-play/league | `scripts/self_play/`, `python/sandboxai/{self_play,league,policies}.py` |
| Replay/metrics/profile | `python/sandboxai/{replay,metrics,telemetry,training_profile}.py` |
| Public commands | `python/sandboxai/cli.py` |
| Real behavior tests | `python/tests/`, `tests/`, `.github/workflows/{godot-tests,python-tests}.yml` |

## 17. Research evidence translated into engineering decisions

This is intentionally a decision list, not a literature dump.

- [PPO](https://arxiv.org/abs/1707.06347) and [SB3 PPO guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html): retain clipped on-policy updates; benchmark CPU MLP inference and subprocess vectorization instead of assuming GPU is faster.
- [SB3 evaluation/reproducibility guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html), [Deep RL That Matters](https://mlanthology.org/aaai/2018/henderson2018aaai-deep/), and [Rliable](https://github.com/google-research/rliable): separate train/eval environments, use multiple seeds, publish uncertainty/performance profiles, and do not select lucky runs.
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html): plain BC's state-distribution shift motivates learner-rollout corrective demonstrations after the dataset split is fixed.
- [Automatic curriculum survey](https://arxiv.org/abs/2003.04664): sample near competence/learning progress, maintain holdouts, and guard promotion with enough evidence.
- [AlphaStar league training](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf): historical pools, PFSP, and exploiters are more robust than latest-vs-latest self-play.
- [Reward tampering/specification gaming](https://arxiv.org/abs/1606.06565) and [policy-invariant reward shaping](https://cseweb.ucsd.edu/~ewiewior/03potential.pdf): audit proxy exploits with independent outcomes; prefer potential-form shaping when its assumptions fit.
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779): static datasets create OOD-action value errors; do not relabel current BC as offline RL or apply naive off-policy learning.
- [Random Network Distillation](https://arxiv.org/abs/1810.12894): intrinsic novelty is an optional sparse-exploration experiment, not a default combat reward, and needs stochastic-noise failure tests.
- [Domain randomization](https://arxiv.org/abs/1703.06907) and [automatic domain randomization](https://arxiv.org/abs/1910.07113): calibrate broad simulator parameter distributions and holdouts before discussing transfer; shape compatibility alone is not transfer evidence.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html): archive versions/device/thread settings and avoid claiming cross-platform bitwise identity.
- Godot 4.7 official guidance: [command-line/headless options](https://docs.godotengine.org/en/4.7/tutorials/editor/command_line_tutorial.html), [CPU profiling](https://docs.godotengine.org/en/4.7/tutorials/performance/cpu_optimization.html), [thread-safe APIs](https://docs.godotengine.org/en/4.7/tutorials/performance/thread_safe_apis.html), [WorkerThreadPool](https://docs.godotengine.org/en/4.7/classes/class_workerthreadpool.html), [physics nondeterminism](https://docs.godotengine.org/en/4.7/tutorials/physics/physics_introduction.html), and [packed float arrays](https://docs.godotengine.org/en/4.7/classes/class_packedfloat32array.html): profile first, isolate mutable state, keep the analytic deterministic boundary, and consider compact/binary transport only after measurement.
