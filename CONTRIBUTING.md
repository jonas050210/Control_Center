# Contributing to SandboxAI

SandboxAI is a local, offline-first reinforcement-learning research platform:
a Godot 4.7.2 FPS simulator (canonical state) plus a Python training stack
(PPO, behavior cloning, evaluation, telemetry). Both halves are tested, and
both halves must stay green.

Read `PROJECT.md` first — it is the living technical spec and tells you what
is **CURRENT**, **PLANNED**, **RESEARCH** and **CONSTRAINT**. `docs/README.md`
indexes the rest of the documentation.

## Scope and ethics

This project is **not** a game cheat, exploit, live-game bot, memory reader,
packet inspector or client modifier. Contributions that move it in that
direction will be rejected regardless of code quality. See the "scope and
ethics" constraint in `PROJECT.md` §1.

## Non-negotiable contracts

Changes that touch these need an explicit, documented migration — never a
silent edit:

| Contract | Value | Source of truth |
| --- | --- | --- |
| Observation vector | exactly **84** floats in `[-1, 1]` | `python/sandboxai/contract.py`, `scripts/core/observation.gd` |
| Action space | exactly `MultiDiscrete([3,3,3,3,2,2])` | `python/sandboxai/contract.py`, `scripts/core/action.gd` |
| Engine target | Godot **4.7.2** | `python/sandboxai/contract.py::GODOT_VERSION` |

`python/tests/test_contract.py` and `python/tests/test_docs_consistency.py`
fail if code or documentation drifts from these values.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"      # pytest + ruff + gdtoolkit
pre-commit install           # optional but recommended
```

Install Godot 4.7.2 separately and either put it on `PATH` as `godot` or
register it once:

```bash
python -m sandboxai validate-runtime --godot-executable /path/to/godot
```

> Working with an AI coding agent? Point it at
> [`AGENTS.md`](AGENTS.md) first. It carries the same rules as this file
> plus the sandbox constraints, the traps previous sessions fell into and
> the list of verified-but-unfixed findings.

## Running the checks

Everything CI runs, in the order CI runs it:

```bash
ruff check .                                  # lint
ruff format --check .                         # Python formatting
mypy                                          # types (config in pyproject)
gdformat --check scripts tests                # GDScript formatting
gdlint scripts tests                          # GDScript style
python -m pytest -q                           # Python suite (900+ tests)
godot --headless --path . --script res://tests/run_tests.gd   # Godot suite
```

`python/tests/test_gdscript_static.py` parses every `.gd` file with the real
grammar and catches unknown `preload` targets, wrong argument counts, unknown
enum members and calls to methods that do not exist on a typed local. On a
machine without Godot it is the main safety net for the GDScript half, so
install the `gdscript` extra rather than letting those tests skip.

## Code standards

**Python**

- Formatted with `ruff format`, line length 100, target `py311`.
- Ruff lint rules: `E`, `W`, `F`, `I`, `UP`, `B`, `SIM`, `C901`
  (`max-complexity = 15`). New code must not need a `noqa`.
- Fully type-annotated; the package ships `py.typed`, and `mypy` (config
  in `pyproject.toml`) has to stay clean. It is not `--strict`: the torch /
  stable-baselines3 boundary is `Any` by necessity. It still earns its keep
  - it found `ReplayEpisode.event_tick` reading a cursor that object never
  had, and the checkpoint battery passing an argument `record_step` does
  not take.
- Prefer narrow exception types. A bare `except Exception` needs a comment
  explaining why swallowing is correct — a silently swallowed `ImportError`
  once made every run manifest record `godot.version: null`.

**GDScript**

- Formatted with `gdformat`, linted with `gdlint` (see `.gdlintrc`).
- Statically typed: every function argument and return value gets a type.
- `preload` over `load` unless the path is genuinely dynamic.
- Views and debug UI mirror simulation state; they never drive it.

**Comments** explain *why*, not *what*. The existing codebase is unusually
good at this — match it.

## Tests

- Every bug fix gets a regression test that fails before the fix.
- Python tests live in `python/tests/test_<module>.py`, Godot tests in
  `tests/test_<subject>.gd` (auto-discovered by `tests/run_tests.gd`).
- Godot tests own their objects: anything that is not `RefCounted` must be
  freed by the test that created it, or the runner reports leaks at exit.
- Optional-dependency tests must **skip**, never fail, when the extra is
  missing (`python/tests/optional_deps.py`).

## Pull requests

1. Work on a feature branch, keep the diff focused.
2. Make the full check list above pass locally.
3. Describe *why*, list the contracts you touched (if any), and note which
   suites you ran — especially whether you ran the Godot suite, since CI is
   the only other place it runs.
4. Update `CHANGELOG.md` under `## Unreleased` for anything a maintainer
   would want to know before touching CI, dependencies or the Control Center.

## Reporting bugs

Use the issue templates. For simulation bugs include the seed, the curriculum
level and the exact command line — the simulator is deterministic, so a seed
plus a command is a complete reproduction.
