# Weapon handling and combat realism — engineering report

Branch `arena/01a0eba1-sandboxai`, commit `2a48630`.

## Scope note, up front

The brief asked for a repo-wide pass across roughly fifteen areas. This
pass went deep on one vertical — **combat / weapon simulation / balance**,
plus the **testing and developer tooling** needed to keep it honest — and
touched curriculum, rewards, metrics, telemetry, self-play and docs where
that vertical reached into them. The remaining areas (PPO/BC pipeline
internals, parallel-env performance, opponent diversity, replay format,
Control Center UI) were read and audited but not modified. Section (f)
lists them in priority order rather than pretending they were done.

---

## (a) Research findings that influenced the implementation

Research was technical reference only; nothing here interacts with a live
Roblox game.

| Finding | Source | What it changed here |
| --- | --- | --- |
| Realism-first FPS design makes **movement while shooting slow you**, and ADS/stillness tighten the cone | TTK Testing wiki + controls pages | `move_speed_scale_firing` per profile; `spread_move_deg` scales with actual resolved speed |
| Recoil has **three** components: vertical climb, horizontal drift that grows under full auto, and a first-shot kick | TTK Testing gunsmith/update notes | `recoil_vertical_deg` saturating toward `RECOIL_SUSTAIN_SCALE`, `recoil_horizontal_deg` on a golden-angle sine, full kick on shot 0 |
| Recommended play is **burst fire with a ~0.4 s reset pause** | TTK Testing build guides | `RECOIL_RECOVERY_DELAY` (0.12 s) + per-profile `spread_recovery_deg_per_s`, tuned so a 6-round SMG burst beats spraying at range |
| Each class owns a **distinct effective range band**; no single best weapon | TTK Testing class list; BO7 TTK analysis | Verified as a test, not a hope: `test_no_profile_wins_every_band`, `test_every_profile_is_the_best_choice_somewhere` |
| Damage falloff is **two-tier**: flat near band → linear transition → flat far band | Black Ops damage analysis | Already the shape of `damage_scale_at_distance`; now asserted monotonic and correctly anchored at both ends |
| `TTK = (shots_to_kill − 1) / (RPM/60)` | Apex/wardogs TTK tools | Matches the existing `ideal_ttk_seconds`; reused verbatim in the Python mirror |
| Tuning direction over versions was to **lengthen** TTK and make recoil more pronounced and more readable — "longer fights, more skill expressed" | TTK Testing V0.04→V0.09 notes | Base damage/range/cooldown left **unchanged**; all new difficulty comes from handling, which is gated by curriculum |
| Practical TTK should assume a headshot rate, not pure body-shot ideal | BattleBit balance discussion | Headshot multipliers per profile, plus `headshots` / `headshot_rate` metrics |

---

## (b) Major changes

### Engine (`scripts/`)

- **`weapon/weapon_state.gd`** — fire modes (`auto`/`semi`/`pump`),
  magazines, reloads, deterministic recoil, bloom, hit zones,
  `sustained_ttk_seconds`. ~510 lines added.
- **`core/curriculum_config.gd`** — `weapon_handling_enabled()` and
  `hit_zones_enabled()`, both gated at level 5 (`OBSTACLES_COVER`).
- **`agent/agent_state.gd`** — recoil moves the agent's *real* aim; firing
  slows movement; `speed_fraction` feeds bloom.
- **`reward/reward_system.gd`** — `hit_reward_scale()` normalizes the flat
  hit bonus by weapon damage and prorates by pellets landed;
  `penalty_trigger_discipline` (−0.002) replaces a −0.1 useless shot for
  a *refused* trigger pull.
- **`env/environment_core.gd`**, `core/episode_state.gd`,
  `metrics/skill_metrics.gd`, `control_center/control_center_telemetry.gd`,
  `env/environment_introspection.gd`, `self_play/self_play_environment.gd`
  — events, counters, telemetry and introspection for all of the above;
  self-play arms both slots symmetrically.

### Two reward-hacking holes closed

1. **Trigger-holding was punished as if it were shooting at nothing.**
   Holding the trigger on an automatic weapon is *correct play*; it cost
   −0.1 per tick. Refused pulls now cost −0.002 and are counted
   separately, because "bad weapon handling" and "firing at empty space"
   are different mistakes and one number cannot tell you which is
   happening.
2. **The flat +1 hit bonus scaled with fire rate.** A 14 HP SMG needs 8
   connecting pulls to kill and collected 8× the flat bonus of one 112 HP
   shotgun shell, so episode return depended on which weapon the scenario
   handed the policy. Now normalized against the reference rifle — which
   scores **exactly** 1.0, keeping legacy runs bit-for-bit.

### Contract preserved

Observation is still **exactly 84 floats**; action is still
**`MultiDiscrete([3,3,3,3,2,2])`**; no RGB; no RNG anywhere in the new
code. The only semantic change is that `weapon_ready` (index 15) now
reports `false` while reloading or empty — the policy's only channel for
"your trigger currently does nothing". Levels 1–4 are unchanged
bit-for-bit, so pre-existing checkpoints and replays still apply.

### Python (`python/sandboxai/`)

- **`weapons.py` (new)** — *parses* the GDScript weapon tables rather than
  duplicating them, and mirrors the TTK/falloff/recoil/bloom maths. A
  retune in the engine moves these numbers automatically.
- **`gdscript_analysis.py`** — new `check_typed_local_calls`. See (c).
- **`metrics.py`, `curriculum_stages.py`, `benchmark_suites.py`,
  `cli.py`** — handling metrics on both sides of the language boundary, a
  `HANDLING_MIN_LEVEL` mirror with a drift test, a `weapon_handling`
  benchmark suite, and a `sandboxai weapon-table` command.

---

## (c) Tests and exact results

### Python suite

| Run | Result |
| --- | --- |
| Baseline (before this work) | **10 failed**, 502 passed, 25 skipped |
| After, without optional extras | 549 passed, 33 skipped, **0 failed** |
| **After, with all extras installed** | **582 passed, 0 skipped, 0 failed**, 499 subtests, 53 s |

The 10 baseline failures were a pre-existing defect: `test_checkpoint_eval.py`
imported `stable_baselines3` without the `optional_deps.HAS_SB3` guard the
rest of the suite uses, so a missing optional extra hard-failed instead of
skipping. Fixed.

### New tests

- **`python/tests/test_weapon_balance.py`** — 35 tests / 472 subtests.
  Asserts *design intent*, not literal numbers: role separation, falloff
  monotonicity, "no weapon wins every band", "every weapon beats every
  other somewhere", headshots pay off at range but never one-shot,
  effective TTK never beats ideal, moving fire never beats planted fire,
  and burst discipline beats spraying at the SMG's range edge.
- **`tests/test_weapon_handling.gd`** (15) and
  **`tests/test_weapon_handling_effects.gd`** (12) — authored, parse- and
  lint-clean, **but not executed** (see (e)). Cover legacy equivalence,
  fire modes, magazine/reload cycle, recoil accumulation/recovery/clamping,
  bloom, hit zones, reward scaling and environment integration.

### The GDScript analyzer gap, found and closed

Mid-pass I checked whether the static gate was actually load-bearing by
introducing a deliberate typo, `env.get_weapon_state_typo()`. **It passed.**
`check_symbols` only inspected `ClassName.member`, so calls on typed locals
— `var env := EnvironmentCore.new()` then `env.anything()`, the dominant
shape in this repo — were entirely unchecked. On a machine without the
engine that was the difference between catching a GDScript typo and
shipping it.

`check_typed_local_calls` now covers it: 0 findings on the clean tree, and
it catches both probes. It has 6 unit tests of its own, including the
subtle case that broke the first implementation (a declaration whose
right-hand side is itself a call, `var d: Dictionary = env.get_metrics()`).

### Other checks

- `sandboxai smoke-test` — full BC + SB3 weight-transfer path: **all
  passed**, `observation_dim: 84`.
- `sandboxai weapon-table` — renders correctly; numbers below.
- `sandboxai benchmark-suites` — correctly **refuses** to emit numbers
  without a Godot binary rather than estimating.

---

## (d) Performance results

**Headless Godot throughput could not be measured** — no engine binary was
obtainable in this environment, and the benchmark module raises
`BenchmarkUnavailableError` rather than fabricating a table. What was
measurable:

| Measurement | Result |
| --- | --- |
| Metrics tick, legacy events | 178,073 ticks/s |
| Metrics tick, full handling events | 176,867 ticks/s |
| **Overhead of the new telemetry** | **0.7 %** — negligible next to a physics step |
| `effective_ttk()` before caching | 889 calls/s |
| `effective_ttk()` after caching | **11,460 calls/s** (12.9×) |
| Balance suite runtime | 0.88 s → **0.15 s** |

The caching fix was a bug in my own new code: `effective_ttk` re-parsed the
GDScript source (1.6 ms) on every call. Now cached per resolved root, with
copies handed out so the cache cannot be mutated.

### Balance outcome

```
profile  mode     rpm  mag  range         role          2m           5m           8m          11m          14m
rifle    auto     120   30  15.0m         long   1.50/1.50s  1.50/1.50s   1.50/1.50s   1.50/1.50s   2.00/2.00s
shotgun  pump      83    6   9.5m  point_blank   0.00/0.00s  0.72/0.72s   1.44/1.44s           --           --
pistol   semi     250   15  12.0m          mid   0.96/0.96s  0.96/0.96s   1.20/1.20s   1.68/1.68s           --
smg      auto     667   30  11.0m          mid   0.63/0.63s  0.63/0.63s   0.81/0.81s   1.35/2.61s           --
```

Cells are *ideal* / *actual* TTK. They diverge only for the SMG, because it
is the one profile whose cooldown (0.09 s) is shorter than the recoil
recovery delay (0.12 s) — so it can out-run its own bloom. At its range
edge that costs it **1.35 s → 2.61 s**, and a disciplined 6-round burst
recovers it to **2.15 s**. The rifle, pistol and shotgun fire slowly enough
that their cone fully collapses between rounds, which is exactly what makes
them the precision options. Only the rifle reaches past 12 m.

---

## (e) Remaining limitations

1. **The GDScript was never executed.** This is the big one. No Godot
   binary could be obtained (GitHub releases, `godot-builds`, tuxfamily,
   flathub, npm mirrors — all failed). The engine-side changes are
   validated by the real GDScript grammar, gdlint, the project's static
   analyzer (symbols, arity, resource paths, and now typed-local calls),
   and a Python mirror of the maths. **That catches compile-class errors,
   not behavioural ones.** Run `godot --headless --path . --script
   res://tests/run_tests.gd` before trusting any of this in training.
2. **Bloom and ammo count are not observable.** Both are consequences of
   the agent's own recent actions, inferable by a recurrent or
   frame-stacked policy. Exposing them would have broken the 84-float
   contract.
3. **Enemies do not use the handling layer.** `EnemyBrain` still fires via
   `try_fire()` with handling off. Arming it would shift every existing
   difficulty curve at once.
4. **`effective_ttk` is an approximation** — uniform-disc cone model,
   stationary target, perfect aim at centre. It is an upper bound on
   accuracy, and it is documented as such.
5. **Pistol and SMG both land in the `mid` band.** Genuine overlap; the
   pistol's extra 1 m of range is thin differentiation.
6. **No training run was performed.** No claim is made about whether a
   policy actually learns burst discipline — only that the incentive
   exists and is measurable.

---

## (f) Highest-priority next steps

1. **Run the Godot suite.** Everything else is downstream of this. Expect
   to fix small things in the 27 new GDScript tests.
2. **Train a level-5 policy and read the new metrics.** The specific
   question: does `trigger_discipline_rate` fall over training, and does
   `headshot_rate` rise? If neither moves, the incentives are too weak to
   matter and the constants need retuning — not the architecture.
3. **Benchmark the handling layer's real cost.** The `weapon_handling`
   suite exists for exactly this; compare it against `curriculum_5_10` at
   matched environment counts.
4. **Widen the pistol/SMG separation** — give the pistol meaningfully more
   range or the SMG less, so the `mid` band has one clear owner.
5. **Then resume the wider audit**, in this order: PPO/BC pipeline and
   parallel-env throughput → opponent diversity in the league → replay
   format coverage for the new events → Control Center diagnostics for
   handling state.
