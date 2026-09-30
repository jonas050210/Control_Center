# AGENTS.md

Context for AI coding agents working in this repository. Read this before
touching anything; it is the accumulated knowledge of previous sessions,
including the mistakes. `CONTRIBUTING.md` is the human-facing version and
the two must not contradict each other.

---

## 1. Ground rules

- **Never invent numbers.** This repository is a measurement tool. If you
  cannot run something, say so. Several modules (`benchmark.py`,
  `tools/bridge_scaling_probe.py`) refuse to produce numbers they did not
  measure; do not undo that.
- **Every claim in prose must be checkable.** Version numbers, test
  counts and file lists drift. If you write one down, add it to a drift
  test (see section 5) or do not write it down.
- **Prefer honesty over green.** A suppressed lint rule is not a passing
  lint rule. If a limit is wrong, change the limit and say why; do not
  add a blanket `# noqa` or `gdlint:disable`.

## 2. The environment you will probably get

| Fact | Consequence |
| --- | --- |
| **No Godot binary**, and the release download is network-blocked | You cannot run the GDScript suite. Use `gdlint`, `gdformat --check` and `sandboxai.gdscript_analysis.analyze('.')`. The `godot-tests.yml` CI job is the real check. |
| System Python is **PEP 668 managed** | `pip install -e .` fails. Create a venv: `python3 -m venv /tmp/venv`. |
| `download.pytorch.org` is **SSL-blocked** locally | Install torch from PyPI locally. CI uses the CPU index, where it works. |
| `python3-tk` / `xvfb` **cannot be apt-installed** | `test_control_center_desktop.py` is skipped locally (one of 9 skips). CI job `desktop-ui-tests` covers it. |

Setup that works:

```bash
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install torch gymnasium stable-baselines3 tensorboard \
    psutil coverage ruff mypy gdtoolkit pyyaml
cd /home/user/SandboxAI
PYTHONPATH=python /tmp/venv/bin/python -m pytest -q
```

## 3. The gates — all of them must pass

```bash
ruff check .                 # All checks passed!
ruff format --check .        # 126 files already formatted
mypy                         # Success: no issues found in 48 source files
gdlint scripts tests         # Success: no problems found
gdformat --check scripts tests
PYTHONPATH=python python -m pytest -q   # 945 passed, 17 skipped, 711 subtests
```

`mypy` takes **no arguments** — its configuration lives in
`pyproject.toml`. It is deliberately **not** `--strict`: the torch/SB3
boundary is `Any` by necessity and `disallow_untyped_defs` would produce
hundreds of meaningless findings. It still earns its keep; it found two
real crashes (see section 6).

`warn_unused_ignores` is **off on purpose**. Turning it on reports 33
errors, almost all on `import torch  # type: ignore` shims, and whether
such an ignore is "unused" depends on which extras are installed — the
gate would disagree between your machine and CI. This was tried and
reverted; do not re-enable it.

## 4. Repository shape

- `python/sandboxai/` — 46 modules, **flat on purpose**. See
  `docs/PYTHON_MODULE_MAP.md`; it is generated from the module docstrings
  and enforced by a test. Do not reorganise into subpackages: every
  import path is public API and appears in docs, user scripts and saved
  run manifests.
- `python/tests/` — 46 pytest files. Note the real names:
  `test_ppo_smoke.py`, `test_ppo_helpers.py`, `test_training_infrastructure.py`,
  `test_weapon_profiles_static.py`, `test_weapons_constant_parser.py`.
  There is **no** `test_ppo.py` and **no** `test_weapons.py`.
- `scripts/` — ~130 GDScript files. `scripts/env/environment_core.gd` is
  the simulation loop; it delegates to `EnvironmentReset`,
  `EnvironmentCombat`, `EnvironmentEnemies` and `EnvironmentIntrospection`,
  all stateless static helpers taking the environment as an untyped first
  argument (typing it would need a preload cycle).
- `tests/` — ~50 GDScript tests, runner `tests/run_tests.gd`.

## 5. Single sources of truth (all enforced by `python/tests/test_docs_consistency.py`)

| Fact | Lives in |
| --- | --- |
| Engine version | `python/sandboxai/contract.py::GODOT_VERSION` |
| Package version | `pyproject.toml::project.version` |
| Python floor | `pyproject.toml::requires-python` |
| Module inventory | module docstrings → `docs/PYTHON_MODULE_MAP.md` |
| Document index | `docs/README.md` |

If you restate one of these in prose, add the file to the relevant tuple
in `test_docs_consistency.py`. That test also asserts the module-level
import graph is **acyclic** — deferred imports inside functions are
excluded, because that is exactly how `config` reaches `sharded_env`,
`self_play` and `selection` without creating a cycle.

## 6. Hard-won facts — do not rediscover these

**Correctness**

- `replay.parse_replay(lines, ...)` takes an **iterable of lines**, not a
  `Path`.
- Replay recording is **two calls**: `record_step(...)` and
  `record_events(...)`. The canonical pattern is `pipeline.py:465-467`.
  Passing `events=` to `record_step` is a `TypeError` — that was a real
  shipped bug.
- `TrainingConfig.reward_breakdown_logging` defaults to **`True`**.
- `PlannedEpisode.record_replay` defaults to **`False`**.
- `EpisodePlan.spawn` is `EnemySpawnPlan | None`, default `None`.
- The BC policy has **exactly two** hidden layers; `load_bc_checkpoint`
  validates this rather than indexing `[0]`/`[1]` and hoping.
- Never write a new test against an assumed default. Look it up.

**Tooling traps**

- Batch edits via a Python heredoc: one missing `old` string aborts the
  loop midway. Assert **all** replacements before writing **any** file.
- In `bash <<'PYEOF'`, `"\\n"` inside Python source is backslash-plus-n.
  Windows paths need four backslashes.
- `sed -i` with multi-line replacement text is dangerous — it once put an
  `if TYPE_CHECKING:` block *above* `import contextlib`. Always re-read
  the file head afterwards.
- Regex-substituting a bare name to `self.<name>` also hits comments,
  docstrings and string literals. Mask strings and comments first, then
  grep to verify.
- `C901` on a factory function sums every nested class body. Only lifting
  the class to module level actually reduces it.
- Ruff rule `BLE` is **not enabled** here — `# noqa: BLE001` is purely
  documentary.
- mypy findings are neither all noise nor all bugs. `bc.py:457`
  (`int + Module`) is a torch-stub artefact; `replay.py:259` and
  `checkpoint_eval.py:359` were real crashes.
- TypedDict and `dict[str, Any]` are mutually incompatible in mypy
  (invariance). Fix the signature (`-> ProcessSnapshot`), do not `cast`
  at every call site.
- The same loop variable name in two loops of one function produces an
  `[assignment]` error on the *second* loop. Rename it.
- mypy does not narrow `self.x` into a closure. Pull it into a local and
  `assert ... is not None` once, right before the closure.

## 7. Conventions

**Commits.** Imperative subject line, no prefix, no ticket number. Then
prose paragraphs explaining *why* — not bullet lists, not a diff summary.
Look at `git log` before writing one.

**Refactoring for complexity.** The patterns that worked here, in short:
validation chains become an ordered `list[tuple[bool, str]]` plus a
helper (preserve the message text verbatim); dispatch becomes a handler
per case plus a module-level dict with heavy imports inside the handlers;
nested callback classes become module-level factories `_make_x(Base, *, …)`
with byte-identical bodies and shared state in a dataclass; long
orchestrators get their preparation phases lifted into module functions
with keyword-only signatures and an early return.

**Refactoring GDScript without an engine.** Extract into a
`class_name X extends RefCounted` file of `static func name(env, …)`, keep
a thin wrapper in the original. Then **prove the bodies are unchanged**:
transform them back mechanically and diff statement by statement.
`gdformat` will add wrapping parentheses to long lines — that is the only
difference you should see. Finish with `gdlint`, `gdformat --check` and
`sandboxai.gdscript_analysis.analyze('.')` (syntax, resource paths,
symbol resolution, call arity, undefined local calls).

The eight extra skips versus a Tk-capable machine are the desktop tests;
`desktop-ui-tests` in CI runs them under Xvfb, and the Windows leg runs
them natively. A skip count of 9 there is expected, not a regression.

## 8. Findings — all six closed

The audit that produced this list is done and every item is fixed. The
table is kept because the *reasoning* is worth more than the diff.

| ID | Was | Resolution |
| --- | --- | --- |
| **S1** | `torch.load(..., weights_only=False)` at three call sites: opening a checkpoint executed whatever its author pickled into it. | One `_load_checkpoint()` with `weights_only=True`. `test_bc.py::CheckpointLoadingIsSandboxedTests` builds an actually hostile checkpoint and asserts the payload does not run. |
| **D1** | `requirements.txt` and `pyproject.toml` had diverged (missing upper bounds). | Deleted. Nothing referenced it — CI builds its own list from pyproject. A test keeps it deleted. |
| **W1** | 20 JSON reads used `encoding="utf-8"`; a Windows BOM made them fail with `Expecting value: line 1 column 1`. | All reads use `utf-8-sig` (identical for BOM-less files). `test_bom_tolerance.py` pins the behaviour *and* fails on a new plain-`utf-8` JSON read. |
| **R1** | `ppo._mean` raised `ZeroDivisionError` on an empty buffer. | Returns `0.0`. "No episodes finished this interval" is a normal state, not an error. |
| **R2** | `contract.validate_observation_spec()` was built from `assert` and became a no-op under `python -O`. | Raises `ValueError`. Verified under `-O`. |
| **R3** | `godot_env._stdout_lines` is an unbounded queue. | **Left as is, deliberately.** A `maxsize` would apply backpressure to the pump thread, fill the OS pipe buffer and block Godot's next write — recreating the exact deadlock the surrounding code prevents. Now documented in place. Do not "fix" this. |

**Checked and found clean** (do not re-audit without reason): division by
zero in GDScript (all four candidates guarded), `DemonstrationDataset.statistics`
(`max(..., 1)`), fuzzing of `parse_replay` / `read_json` / `tail_jsonl`,
path traversal in `artifact_repository.cleanup` (`root in path.parents`,
`path != root`, `confirm` flag), pipe deadlocks in the Godot bridge
(solved, including `BaseException` during the handshake), no `shell=True`,
no `eval`/`exec`, no mutable default arguments, no `utcnow`, no committed
artefacts.

Weakest test coverage: `benchmark.py` 43 %, `cli.py` 61 %, `adapter.py`
72 %. Total 86 %, gate 70 %. `benchmark.py` is the one worth raising —
it is the module the maintainer has to trust when sizing `--env-workers`.

## 9. The maintainer's setup — read before optimising anything

Ubuntu under Windows 11 (WSL), i7-12700F (8P + 4E cores), RTX 4060 Ti,
32 GB RAM, Godot 4.7.2.

**A GPU does not help this workload, and that is expected.** The policy is
84 → 128 → 128. Per-step kernel launches and host↔device transfers cost
more than the arithmetic saves; the maintainer measured CPU at roughly
twice the throughput and that matches `README.md:403-405`, which offers
`--inference-device cpu` for exactly this reason. Do not "fix" this and
do not propose CUDA work. A GPU becomes relevant only once image
observations (CNN) exist, and those are explicitly not implemented.

**The bottleneck is Godot.** One bridge process steps its environments
serially on one core. The lever is `--env-workers N`: independent Godot
processes, each owning a contiguous slice of environments with the seed
that slice would have received anyway — so sharding changes wall time,
never trajectories. On a 12-core machine the default of 1 leaves most of
the CPU idle. Measure, do not guess:

```bash
sandboxai benchmark --worker-counts 1,2,4,6,8 --env-counts 4,8,16
```

Single-user project. Checkpoints are never loaded from third parties.
There is no release process and none is wanted — do not propose PyPI
publishing, semantic-version ceremony or a support policy.

## 10. Session/branch hygiene

Arena sessions are pinned to one branch (`arena/<id>-sandboxai`). Commit
and push only there, and open PRs from there. Do not create or switch to
other branches even if asked — the work would stop being associated with
the session.
