# Replay, research metrics, generalization and curriculum

This document covers the subsystems added on top of the original
Godot + PPO sandbox. Everything here is either deterministic bookkeeping or
diagnostic measurement; none of it changes what a policy observes.

## 1. Deterministic replay

**Format version 1.** Implemented twice, byte-compatible on purpose:

| Side | Files |
| --- | --- |
| Python | `python/sandboxai/replay.py` |
| Godot | `scripts/replay/replay_format.gd`, `replay_recorder.gd`, `replay_player.gd` |

JSON Lines, one object per line:

```
{"header": {...}}                                     # line 1
{"tick": {"t":0,"a":[1,1,1,1,0,0],"r":0.0,"d":true?,"o":[...]?,"i":{...}?}}
{"event": {"t":42,"e":"combat","l":"shot","v":{...}}}
{"result": {...}}                                     # last line
```

**Header** (reproduction inputs + provenance + contract fingerprint):
`magic` (`sandboxai.replay`), `version`, `detail`, `seed`, `map_id`,
`scenario`, `lighting`, `enemy_count`, `curriculum_level`, `policy_id`,
`checkpoint`, `environment_index`, `agent_slot`, `team_id`,
`observation_dim`, `action_nvec`, `simulation_dt`, `notes`.

**Detail levels.** `light` (default) stores only actions, rewards and
done flags per tick: an episode is reproducible from
`(seed, map, scenario, conditions, actions)`, so storing the observation
stream as well would be redundant. `detailed` additionally stores the
observation vector, for analysis.

**Event kinds.** `episode_start`, `episode_end`, `target_change`,
`perception`, `memory`, `sound`, `combat`, `death`, `curriculum`, `note`.
The timeline defaults to the "important" subset (`episode_start`,
`episode_end`, `target_change`, `combat`, `death`).

**Compatibility.** Loading rejects: a wrong magic, an unreadable version,
an unknown detail level, non-contiguous tick indices, an action width that
is not `len(ACTION_NVEC)`, an unknown record type, and (unless the caller
opts out) an `observation_dim` / `action_nvec` that disagrees with this
build's contract. Opting out keeps an old recording *inspectable* without
pretending it is playable.

**Playback.** `ReplayPlayer` (both languages) owns a cursor: play, pause,
single step, speed, seek by tick or by time, jump to the next/previous
event. It mutates nothing but itself.

CLI: `sandboxai replay --path run.jsonl --timeline`.

## 2. Research metrics

`python/sandboxai/metrics.py` (per-episode + aggregator, JSON and CSV
export) and `scripts/metrics/skill_metrics.gd` (the live Control Center
view) share eight categories: **AIM, REACTION, AWARENESS, POSITIONING,
MOVEMENT, COMBAT, SURVIVAL, EXPLORATION**.

Two hard rules:

1. **Diagnostic, not reward.** No metric is wired into the reward system.
2. **Ground truth is separate.** `SkillMetrics.ai_available()` returns only
   what the agent could have derived from its own observation;
   `SkillMetrics.ground_truth()` returns simulator state in a separate
   dictionary, rendered under a red "DEBUG ONLY" heading in the METRICS
   tab. `GROUND_TRUTH_KEYS` names the second set so the two can never be
   merged by accident.

Aggregation keys: `policy_id`, `map_id`, `scenario`, `lighting`,
`enemy_count`, `curriculum_level`, `seed`, `episode_id`.

## 3. Conditions, randomization and generalization

- `conditions.py` — the condition space (map × lighting × scenario ×
  enemy count × level), per-condition rolling statistics and
  `generalization_report`, which reports the **win-rate spread** across
  conditions. A large spread means the policy memorized rather than
  generalized.
- `randomization.py` — deterministic episode plans. `_derive` is a
  blake2b function of `(master_seed, index, salt)` with no shared RNG
  state, so `episode_plan(i)` is identical across processes, machines,
  resumes and replays, and parallel environments simply use disjoint
  index strides. `EpisodePlan.environment_commands()` returns the ordered
  engine calls (`set_curriculum_level`, `set_map`, `set_lighting_mode`,
  `set_scenario`, `reset`).
- `generalization.py` — known maps vs unseen seeds/variants/maps, the
  lighting sweep, the enemy-count sweep and the scenario families, with
  structured per-condition reports.

No map identifier ever enters the observation:
`randomization.assert_no_environment_labels()` scans `OBSERVATION_SPEC`
for forbidden tokens and a test proves it actually fires.

## 4. Curriculum integration

`python/sandboxai/curriculum_stages.py` defines the ladder, mirroring
`CurriculumConfig.Level` in GDScript (a test asserts the levels match):

| Level | Stage | Focus |
| --- | --- | --- |
| 1 | movement_and_aim | move, point at a stationary target |
| 2 | moving_targets | track a moving target |
| 3 | basic_combat | an opponent that shoots back |
| 4 | multi_enemy_combat | target selection |
| 5 | cover | use geometry |
| 6 | fov_and_occlusion | observation stops being ground truth |
| 7 | sound | act on what you only hear |
| 8 | memory | track a contact you cannot see |
| 9 | navigation_and_vertical | complex geometry, height |
| 10 | randomized_environments | generalize across conditions |
| 11 | self_play | fight another learned policy |

`CurriculumDirector` owns an `AutoCurriculum`, turns the current stage
into a `TrainingDistribution`, and hands out reproducible episode plans.
Promotion is performance-gated: a rolling window, a per-stage minimum
episode count (20 at level 1 rising to 100 at level 11), a cooldown, and a
window that is cleared on every change. **A single lucky episode can never
promote**, and the level can only move one step at a time — both asserted
by tests. Demotion uses a separate, lower threshold and level 1 cannot be
demoted.

An off-ladder `EXPLORATION_STAGE` (Map Analyzer) is gated on coverage
rather than win rate and never promotes or demotes.

CLI: `sandboxai curriculum [--json]`.

## 5. Map Analyzer 2.0

`scripts/exploration/exploration_report.gd` is the analysis layer over
`SpatialMemory`. It is constructed with a `SpatialMemory` and nothing
else, so there is no handle through which hidden geometry could arrive.

Heatmap layers (`coverage`, `recency`, `visits`, `illumination`, `cover`,
`danger`, `confidence`) report **-1.0 for unknown cells**, so "unknown"
renders differently from "known and zero". Also provides region counts,
the walked route and its length, exploration efficiency
(cells learned per meter walked), remembered sightlines and cover, the
coverage-over-time curve, and a diagnostic metric set.

## 6. Benchmarks

`python/sandboxai/benchmark_suites.py` defines four comparable suites —
`curriculum_1_4`, `curriculum_5_10`, `perception_combat`, `map_analyzer` —
each measured at **1, 4, 8, 16, 32 and 64** environments. The difference
between two suites at the same environment count is the overhead of what
was switched on.

The module measures; it never estimates. Without a real Godot executable
`run_suites()` raises `BenchmarkUnavailableError` rather than returning a
plausible table, and `describe_plan()` is explicitly marked
`"measured": false`.

CLI: `sandboxai benchmark-suites [--plan-only]`.
