# SandboxAI Full-Repository Bug Hunt — Final Report

Scope: every GDScript file under `scripts/` and `tests/`, every Python module under `python/sandboxai/`, every Python test, the Godot↔Python bridge, replay/metrics/curriculum/conditions/randomization/self-play/teamplay/generalization/external-adapter subsystems, the Control Center (session + UI + introspection), docs, and the README. Single-commit baseline `c9d3b0c`, clean tree, 421 pytest green before changes.

Method: full read of ~43k lines, dependency modelling of the producer/consumer seams (observation/action contract, `_brain_context`, `heard` sound events, `step_all` results, replay/demonstration formats), then root-cause fixes with regression tests, then duplicate-pattern searches, then a whole-project re-audit.

> **Later-pass note (headless-only Control Center).** The in-simulator
> operator scene audited here (`scenes/control_center.tscn`,
> `scripts/control_center/`, and the `tests/test_control_center_*.gd`
> regression files) has since been removed; the Control Center is now the
> headless-only Python/Tk desktop application. Findings B4/B5/B6 and the
> session-related tests below are therefore historical records of that
> deleted layer; the fixes they describe went with it.

---

## 1. Bugs found and fixed

### B1 — Missed point-blank enemy shot silently converts into melee damage (Medium-High, simulation/combat)
- **File:** `scripts/enemy/enemy_brain.gd` `_try_attack()`
- **Root cause:** the ranged branch `return`s on a hit, on a weapon-cooldown failure, out of fire range and on blocked LOS — but a **miss** fell through to the melee check. At ≤2 m a missed roll therefore dealt `ENEMY_ATTACK_DAMAGE` (10.0) *in addition to* the fired shot, while a hit dealt only `ENEMY_FIRE_DAMAGE` (9.0) and a cooling-down weapon dealt nothing. Damage at point-blank was random-roll dependent, and a miss was deadlier than a hit.
- **Fix:** the ranged attempt (hit or miss) ends the attack for that tick; the melee block is now reachable only when `allow_ranged` is false, matching the curriculum contract "enemies shoot instead of meleeing" (`curriculum_config.gd`).
- **Regression test:** `tests/test_bug_regressions.gd::test_missed_point_blank_shot_does_not_also_melee` (rigged-miss RNG stub; also asserts melee still works with `allow_ranged=false`).

### B2 — Python `SPAWN_RULES` names three rules the engine doesn't have and misses four it does (Medium, training distribution)
- **File:** `python/sandboxai/randomization.py`
- **Root cause:** the tuple claimed to mirror `ScenarioLibrary` but contained invented ids `spread`, `flank`, `cover` and omitted `out_of_sight`, `behind_cover`, `surround`, `elevated`. Any consumer applying `EpisodePlan.spawn.rule` through `ScenarioLibrary` would silently degrade those plans to default placement, and the real rules could never be drawn.
- **Fix:** aligned the tuple to the GDScript `SPAWN_*` constants.
- **Regression test:** `python/tests/test_conditions.py::SourceDriftTests::test_spawn_rules_match_scenario_library` — parses `scenario_library.gd` and fails on drift (same pattern as the existing MAP_IDS/LIGHTING_IDS guards), plus uniqueness assertion.

### B3 — Map Analyzer's heard-sound danger marking was dead code (Medium, exploration/Map Analyzer)
- **File:** `scripts/exploration/map_analyzer.gd` `update()`
- **Root cause:** the loop read `sound["position"]`, but heard events (from `AgentPerception.heard` → `SoundBus.sample`) never carry a `position` — hearing is deliberately directional. The documented behaviour "hearing a shot raises a danger belief" never executed.
- **Fix:** estimate the source position from the heard event's own `direction`/`distance` (the same perception-honest estimate `AgentPerception` builds for its memory track) and mark danger there; non-combat categories still excluded.
- **Regression test:** `tests/test_map_analyzer.gd::test_heard_combat_sound_marks_danger_at_the_estimated_location` (SHOT marks the estimated cell, FOOTSTEP does not).

### B4 — Control Center recorded every episode with the next episode's number (Low-Medium, Control Center)
- **File:** `scripts/control_center/control_center_session.gd` `_record_episode()`
- **Root cause:** `SimulationManager.step_all()` auto-resets a finished environment *before* the result is returned, so `episode.episode_count` read during recording already pointed at the next episode. The first finished episode was recorded as episode 2, and every HUMAN-vs-AI comparison row was shifted by one.
- **Fix:** read the step result's `auto_reset` flag and record `episode_count - 1` for the finished episode.
- **Regression test:** extended `tests/test_control_center_session.gd::test_episode_results_are_recorded_with_their_action_source` to assert the first record is episode 1.

### B5 — `ControlCenterSession.advance(0)` advanced the simulation one step (Low, Control Center)
- **File:** `scripts/control_center/control_center_session.gd` `advance()`
- **Root cause:** `for _index in range(maxi(1, steps))` treated a zero/negative request as one. Public contract says "advances by `steps` ticks".
- **Fix:** early-return for `steps <= 0`, loop `range(steps)`.
- **Regression test:** `tests/test_control_center_session.gd::test_advance_zero_advances_nothing`.

### B6 — Control Center clamped `--curriculum-level` to 1..5 while everything else accepts 1..11 (Low-Medium, Control Center)
- **File:** `scripts/control_center/control_center_main.gd`
- **Root cause:** a stale limit from before the world/perception milestone: `@export_range(1, 5)` and `clampi(int(value), 1, 5)` silently downgraded `--curriculum-level=8` even though the session, `config.sanitize()`, the settings-panel picker (offers 1..11) and the scenario presets (up to level 10) all use the full range.
- **Fix:** clamp to `CurriculumConfig.Level.STATIONARY_TARGET..AGENT_VS_AGENT`; doc table updated (`docs/CONTROL_CENTER.md`), and the Python CLI `--scenario` help now lists all ten presets instead of four.
- **Regression coverage:** static suite validates the new code; behaviour is now consistent with the in-UI picker.

### B7 — Threat LOS test hardcoded the agent's height (Low, perception)
- **File:** `scripts/perception/agent_perception.gd` `_evaluate_enemy()`
- **Root cause:** `has_line_of_sight(world, enemy.get_eye_position(), agent.position, 1.8)` used a literal instead of the agent's actual height (`agent.height`, which `EnemyBrain` uses on the other side of the same test). Bit-identical today (`AGENT_HEIGHT == 1.8`) but breaks silently if the constant or per-agent height ever changes.
- **Fix:** use `agent.height` (in scope).

### B8 — Demonstration datasets documented the wrong action encoding (Low, demonstrations/BC)
- **Files:** `scripts/recording/demonstration_recorder.gd`, `python/sandboxai/dataset.py`
- **Root cause:** the `action_encoding` metadata string still described the contract-v1 7-value layout `[move, strafe, yaw, pitch, shoot, look_delta_x, look_delta_y]`, while both recorders write the v2 8-value log array with `jump` at index 5. Any consumer decoding from the header would read `jump` as `look_delta_x`. (Parsing was already correct — `action_to_multidiscrete` handles v2 — so this was silent misinterpretation risk, not data corruption.)
- **Fix:** both headers now describe the v2 layout. No test pinned the stale string.

### B9 — Stale contract dimensions in user-facing docs (Low, docs)
- **Files:** `README.md`, `docs/CONTROL_CENTER.md`, `docs/DEBUG_GUI_AND_BENCHMARKING.md`, `scripts/core/simulation_manager.gd` (comment)
- **Root cause:** pre-v2/v3 numbers survived the contract growth: "33-float" observation (now 84, in three doc places plus a code comment), `MultiDiscrete([3,3,3,3,2])` (now `[…,2,2]`), "all 33 contract fields" (now 84), "exact five-field action accuracy" (BC now trains/evaluates six fields). The README is the authoritative quickstart, so these actively mislead (e.g. someone building an adapter to the documented 33-float shape).
- **Fix:** corrected all four locations to the v3 contract; historical changelog entries in `docs/OBSERVATION_ACTION_CONTRACT.md` intentionally kept.

### B10 — `stream_for_environment` docstring promised an interleaving it doesn't implement (Low, training distribution)
- **File:** `python/sandboxai/randomization.py`
- **Root cause:** the docstring promised "environment k plays episodes k, k+N, k+2N…", but the code (which has no N parameter) uses a fixed prime stride of 1000003. The code's design is actually better (per-env streams are stable when the environment count changes); the documented contract was wrong.
- **Fix:** docstring now describes the real residue-stride scheme and its rationale.

---

## 2. Suspicions investigated and cleared (no bug)

- **`EnemyState.reset()` doesn't reset the reaction profile** — not a bug: `EnvironmentReset.reset_enemy()` calls `_apply_enemy_difficulty()` on every reset, which re-applies the archetype; `set_curriculum_level()` re-applies it mid-episode (guarded by `test_bug_regressions.gd::test_curriculum_change_applies_every_enemy_parameter`).
- **`SOUND_UNKNOWN_SOURCE_ID` (-999) colliding with enemy memory keys** — no collision: enemy ids are 0..N-1, the agent is -1, unknown is -999; the agent never hears its own sounds (`ignore_source_id = -1`).
- **`scenario["enemy_archetype"]` always REGULAR** — dead-but-harmless data: the live source is `curriculum.enemy_archetype()` via `_apply_enemy_difficulty`; the scenario field is unused (left as-is, see remaining issues).
- **Null-RNG default in the enemy hit roll** — `_try_attack` resolves `rng == null` as a guaranteed hit (roll 0.0). Production always passes the environment RNG; the tactical tests depend on the deterministic default. Documented as a hazard, behaviour preserved (see remaining issues).
- **Reset RNG draw order** (`environment_reset.gd`), reset seeding (`seed_base + index` both sides of the bridge), replay format fingerprints, BC/PPO shape compatibility, Elo zero-sum updates, `verify_determinism` design, evaluation seeding, parallel-env streams, observation/action contract (84 floats, nvec [3,3,3,3,2,2]) — all verified consistent across the GDScript/Python boundary (guards exist in `godot_env.py`, `replay.py`, `cli.py`, plus `test_contract.py` source-drift tests).

## 3. Files changed, by subsystem

- **Godot simulation:** `scripts/enemy/enemy_brain.gd` (B1), `scripts/perception/agent_perception.gd` (B7), `scripts/exploration/map_analyzer.gd` (B3)
- **Godot Control Center:** `scripts/control_center/control_center_session.gd` (B4, B5), `scripts/control_center/control_center_main.gd` (B6)
- **Godot recording/docs-in-code:** `scripts/recording/demonstration_recorder.gd` (B8), `scripts/core/simulation_manager.gd` (B9 comment)
- **Python training stack:** `python/sandboxai/randomization.py` (B2, B10), `python/sandboxai/dataset.py` (B8), `python/sandboxai/cli.py` (B6 help text)
- **Docs:** `README.md`, `docs/CONTROL_CENTER.md`, `docs/DEBUG_GUI_AND_BENCHMARKING.md` (B6, B9)
- **Tests (regression coverage):** `tests/test_bug_regressions.gd` (+1 test, +MissRng stub), `tests/test_map_analyzer.gd` (+1 test), `tests/test_control_center_session.gd` (+1 test, 1 extended), `python/tests/test_conditions.py` (+1 source-drift test)

No public contract changed: the observation/action spaces, replay format, JSONL schemas (apart from the corrected human-readable `action_encoding` description string), and all APIs keep their shapes.

## 4. Tests

**Executed here (no Godot binary available in this environment):**
- `cd /home/user/SandboxAI && /tmp/venv/bin/python -m pytest -q` → **422 passed, 12 subtests passed** (baseline 421 + 1 new drift test), ~22 s, full training stack installed (numpy, torch 2.14.0, gymnasium 1.3.0, stable-baselines3 2.9.0, psutil, gdtoolkit 4.x, tensorboard).
- `cd /home/user/SandboxAI && /tmp/venv/bin/python -m pytest python/tests/test_gdscript_static.py -v` → **4 passed** (real gdtoolkit grammar parse of every `.gd` in `scripts/` and `tests/`, gdlint clean, analyzer coverage checks).
- `python -c "from sandboxai.gdscript_analysis import analyze; analyze('.')"` → **no findings** (syntax + resource paths + symbol resolution + call arity over all 60+ scripts, includes every edited/new file).
- `lint_all('.')` (gdlint rules) → **no findings**.

**Unable to execute here (requires a Godot 4.7.2 binary) — must be run locally:**
- `godot --headless --path . --script res://tests/run_tests.gd` — the full Godot suite; it auto-discovers the three new GDScript regression tests (`missed_point_blank_shot_does_not_also_melee`, `heard_combat_sound_marks_danger_at_the_estimated_location`, `session_advance_zero_is_a_noop`) and the extended episode-numbering assertion.
- Live bridge smoke: `sandboxai train --env-count 2 --steps 2048 …` and `sandboxai benchmark-suites` (need a real engine process).

**Not executed (by design):** no benchmarks were run and no performance numbers are claimed — Godot is unavailable here and the benchmark modules refuse to fabricate numbers (`BenchmarkUnavailableError`).

## 5. Remaining issues (reviewed, deliberately not changed)

1. **Null-RNG hit-roll default** (`enemy_brain.gd`): a caller omitting `rng` from the brain context gets an always-hit enemy. Production and tests are consistent today; if a new caller appears, prefer making the roll fail closed or require the rng.
2. ~~**`SelfPlayCoordinator.sample_opponent(rng=None)`** falls back to the unseeded global `random` module.~~ **Fixed in a later pass:** the coordinator owns a seeded generator, gains explicit strategies (`uniform`/`latest`/`recency_weighted`/`round_robin`), `reset_sampling()` and `sampling_snapshot()`, and the rule lives in `SelfPlayConfig.opponent_strategy`/`opponent_seed`. It never touches the global `random` module.
3. **Seed labelling of continuation episodes**: Control Center records and `evaluation.py` tag each finished episode with `seed + env_index`, which is strictly true only for the first episode after an explicit seeded reset (later episodes continue the RNG stream). Identifier-only; does not affect training or rewards.
4. **Vestigial data**: `scenario["enemy_archetype"]` (always REGULAR, unread) and `_brain_context["agent_eye"]` (produced, never consumed). Harmless; removing would churn dict surfaces for no behavioural gain.
5. **`EpisodePlan.spawn`/`layout_seed`** are reproducibility data not yet applied by the bridge (`rl_server.gd` has no `set_map`/`set_scenario`/`set_lighting_mode` commands); `environment_commands()` targets `EnvironmentCore` directly. Foundation state, documented as such — the SPAWN_RULES fix (B2) makes this data correct for when it is wired.
6. **HumanController has no jump binding** (SPACE shoots), so human demonstrations cannot contain jumps — a feature gap in the demo-recording workflow, not a correctness bug.

> **Later-pass note.** Item 2 is fixed (see above). The slot-order fire bias
> called out elsewhere in this report is also fixed: `SelfPlayEnvironmentCore`
> now resolves both slots' fire against the pre-tick world and applies damage
> afterwards, so a mutual lethal exchange kills both agents and ends `draw`.

## 6. Regression assessment

- The only simulation-behaviour change is B1: at <2 m, missed enemy shots no longer deal melee damage (average point-blank damage drops slightly and stops depending on the roll). No existing test pins the old behaviour (verified by searching all `tests/*.gd` for tactical-level damage assertions); melee-only enemies (levels 1–4, `update_ai`) are untouched.
- B3 changes Map Analyzer behaviour only in modes where combat sounds occur; solo Map Analyzer mode (the rewarded one) has no combat sounds, so exploration rewards and coverage tests are unaffected. Tracking mode gains danger beliefs, which is the documented intent.
- B7 is bit-for-bit identical today (`agent.height == 1.8`).
- B4/B5/B6 affect only the Control Center operator layer; B2/B8/B9/B10 are data/doc corrections with no behavioural surface beyond the corrected metadata.
- Whole-project re-audit after the fixes: full pytest green, GDScript parse/lint/analyze green, no duplicate instances of any fixed pattern remain (searched: fall-through-after-miss patterns, `has("position")`-style key mismatches on heard events, post-auto-reset `episode_count` reads, hardcoded height literals in perception, stale 33/65/five-field action references).

## 7. Recommended next step

Run the Godot suite locally (`godot --headless --path . --script res://tests/run_tests.gd`) — it validates the three new GDScript regression tests and the combat-behaviour change end-to-end — then retrain/evaluate a level-5+ checkpoint briefly to confirm the (intended) small drop in point-blank enemy damage, and checkpoint-compare win rates before/after B1 if curve continuity matters.
