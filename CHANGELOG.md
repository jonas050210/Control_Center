# Changelog

Human-readable summary of engineering-facing changes. Not a marketing
changelog: every entry here is something a maintainer would want to know
before touching CI, dependencies, or the desktop Control Center. See
`PROJECT.md` for the living technical spec and roadmap; this file only
records what changed and why.

## Unreleased

### Added
- `LICENSE` (MIT). The repository had none, which made every "open
  source" claim in the README legally meaningless: without a license,
  default copyright applies and nobody may fork, vendor or redistribute
  the code.
- Project governance: `CONTRIBUTING.md` (the exact local check sequence),
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, a PR template with
  the checks as a checklist, and issue templates.
- `.pre-commit-config.yaml` running ruff, ruff-format and gdformat, so
  the formatting gate is reachable before CI rather than after it.
  mypy is deliberately *not* a hook - it needs the training extras
  installed and would make `git commit` depend on a 2 GB environment.
- Dependabot for `pip` and `github-actions`, plus a `pip-audit` CI job.
- `mypy` gate. The package has shipped a `py.typed` marker without
  anything verifying the annotations behind it; a `typecheck` job now
  runs mypy over `python/sandboxai` with the training extras installed,
  so the annotations are checked against real torch/SB3/gymnasium types.
  Configured in `pyproject.toml` (invocation is a bare `mypy`) and
  installable as the `typecheck` extra.
- `docs/README.md`: an index of the twelve documents under `docs/`, with
  one line on what each answers and who it is for.
- `python/tests/test_docs_consistency.py`: turns the documentation into
  something CI can fail on. Engine version, package version and the
  Python floor each have exactly one source of truth
  (`contract.py::GODOT_VERSION`, `pyproject.toml::project.version`,
  `requires-python`), and every prose restatement is checked against it.
- `sandboxai --version`.
- `python/tests/test_ppo_helpers.py` (39 tests) and
  `python/tests/test_weapons_constant_parser.py` (18 tests), covering
  code paths that previously only ran inside a full training loop.
- `docs/PYTHON_MODULE_MAP.md`: all 46 modules of the flat `sandboxai`
  package grouped by theme, each described by its own docstring summary.
  This is the deliberate alternative to splitting the package into
  subpackages: every module's import path is public API and appears in
  docs, user scripts and saved run manifests, so a reorganisation would
  rewrite several hundred call sites to buy navigability that a document
  can provide instead. The map is enforced - a new module nobody
  classified, or a description that no longer matches the docstring,
  fails `test_docs_consistency.py`. The same test now also asserts that
  the module-level import graph is acyclic, which is the structural
  invariant that actually matters here, and names the loop when it is
  not. `docs/ARCHITECTURE.md` no longer keeps its own hand-written file
  list, which had gone stale at thirteen of forty-six modules.
- `.gdlintrc`, pinned from `gdlint -d`. `CONTRIBUTING.md` already
  referenced it; without the file gdlint silently followed whatever
  defaults the installed gdtoolkit version shipped, so a toolchain bump
  could change CI's verdict with no GDScript change.
- Python CI workflow (`.github/workflows/python-tests.yml`): ruff lint,
  a numpy-only core-tests job, and a full-tests job (training extras) run
  on a `ubuntu-latest` / `windows-latest` matrix - the project's two real
  target environments (native Windows, and Linux/WSL, which is a genuine
  POSIX host even when the project folder lives on a Windows desktop).
  macOS is explicitly out of scope. The Python suite previously only ran
  on Linux via `godot-tests.yml`.
- `requirements-lock-linux-py311-cpu.txt`: a real, generated-and-verified
  pip-freeze snapshot of the exact dependency versions this repo's Linux/
  CPU CI resolves to. Not a cross-platform lock (torch/CUDA wheels differ
  per OS/accelerator) - see the file header and `PROJECT.md` section 13.
- Desktop Control Center (`sandboxai control-center-desktop`): a Tkinter
  engineering GUI (Dashboard/Training/Agents/Benchmarks/Evaluations/Runs
  and Checkpoints/System and Telemetry/Settings pages) backed by
  `sandboxai.adapter.SandboxAIAdapter` and the Tk-free
  `sandboxai.control_center_viewmodel` presentation layer. See
  `docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md`.

### Changed
- CI (`python-tests.yml`) no longer duplicates the Python suite that
  `godot-tests.yml` was also running; adds pip caching, a
  `concurrency` group that cancels superseded runs, and pins every
  third-party action to a commit SHA rather than a mutable tag.
  Job order: lint, typecheck, core-tests, gdscript-checks,
  desktop-ui-tests, full-tests, coverage, audit.
- Ruff now enforces `E,W,F,I,UP,B,SIM,C901` instead of `F` alone, with
  `max-complexity = 15`, and `ruff format` is the formatter of record
  for the Python half. Clearing the new rules touched most of the
  package; the complexity ceiling in particular forced `train_ppo`
  (cyclomatic complexity 103) apart into seven callback factories, a
  `_SelectionState` dataclass and a module-level `_EvaluationDriver`,
  leaving the orchestrator at 15 and the callbacks unit-testable
  without a live environment. The same treatment went to the CLI
  dispatcher, `RuntimeValidator.validate`, the evaluation battery, the
  plan scheduler, dataset validation, replay parsing, run inspection,
  adapter checking, BC training and the GDScript analyzer.
- `scripts/env/environment_core.gd` no longer suppresses gdlint's
  file-length rule. Two cohesive blocks moved out into the same kind of
  stateless static helper the file already used for `EnvironmentReset`
  and `EnvironmentIntrospection`: `EnvironmentCombat` (the agent's shot -
  hitscan, hit zones, the hit / near-miss / useless-shot classification,
  occlusion and the death bookkeeping) and `EnvironmentEnemies` (the
  per-tick opponent update and its motion sounds). Every function body
  moved unchanged, `EnvironmentCore` keeps a wrapper for each entry
  point so `step()` still reads as a sequence of named sub-steps, and
  the file went from 1235 lines to 963. No `.gd` file suppresses
  `max-file-lines` any more, so the limit is a limit again.
- Every `.gd` file is now `gdformat`-clean (123 of 153 files were
  reformatted), and both `gdformat --check` and `gdlint` gate CI.
- `AUDIT_REPORT.md` moved to `docs/AUDIT_REPORT.md`, where the rest of
  the long-form documents live.
- Coverage is measured and gated at 70 % (currently 86 %).

### Fixed
- Loading a behavior-cloning checkpoint could execute arbitrary code.
  `bc.py` passed `weights_only=False` to `torch.load` at three call
  sites, which unpickles whatever the file contains; a checkpoint is
  therefore a program, not data. Nothing needed it - the checkpoints
  hold tensors, strings, numbers, lists and dicts - so all three go
  through one `_load_checkpoint()` with `weights_only=True`, and a file
  that cannot be read that way is refused with an explanation instead of
  being trusted. The regression test builds a genuinely hostile
  checkpoint and asserts its payload does not run.
- JSON files written on Windows were unreadable. Twenty read sites used
  `encoding="utf-8"`, which hands a byte-order mark straight to
  `json.loads`; PowerShell's `Out-File -Encoding utf8` writes one, so a
  config produced that way failed with `Expecting value: line 1 column
  1`. Reads now use `utf-8-sig`, which is byte-identical for files
  without a BOM. Writes deliberately still use `utf-8` - emitting a BOM
  would be the opposite bug - and a test enforces both directions.
- `ppo._mean` raised `ZeroDivisionError` on an empty buffer. No caller
  could reach it, but the next one would have: it now returns 0.0,
  because "no episodes finished in this interval" is a normal state
  early in a rollout.
- `contract.validate_observation_spec()` consisted entirely of `assert`
  statements, so under `python -O` it silently became a no-op while
  still reading like a guarantee. It raises `ValueError` now.

- `ReplayEpisode.event_tick` was a copy of the recorder's property and
  read `self._tick`, a cursor only the recorder has - so asking a parsed
  episode where to anchor an event raised `AttributeError`. Found by
  mypy.
- The checkpoint evaluation battery called
  `record_step(action, reward, events=events)`, but
  `ReplayRecorder.record_step` takes no `events` parameter, so every
  battery run with replay recording enabled died with `TypeError` on the
  first step. Events are a separate `record_events(...)` call, as
  `TrainingPipeline` has always done it. Found by mypy.
- `weapons.py` parsed GDScript weapon constants with `eval()` on text
  read from disk. Replaced with an AST walker over an explicit operator
  whitelist; a repo-wide old-versus-new comparison caught that the
  whitelist also has to cover `1 << 20`. There is no `eval`/`exec` left
  in `python/sandboxai/`.
- Nine `except Exception` handlers narrowed to the exceptions they can
  actually see, so genuine defects stop being swallowed as "optional
  dependency missing".
- `manifest.py::godot_snapshot()` imported a nonexistent
  `GodotRuntimeValidator` class; a broad `except Exception` silently
  swallowed the resulting `ImportError`, so every run manifest ever
  produced recorded `godot.version: null` even with a working Godot
  install. Now imports the real `runtime_validation.RuntimeValidator`.
- The first Windows CI run surfaced 10 pre-existing test-fixture bugs
  that had never been exercised on native Windows before (missing
  `os.name`/`USERPROFILE` mocks, a resolved-vs-raw project-path
  comparison, a POSIX-only fake Godot executable, a `ProcessManager`
  background-thread race in a force-stop test, and four `is_wsl()`-mocked
  WSL path-translation tests that assumed a POSIX `Path` while running
  under native Windows' `WindowsPath`). None were production bugs; all
  ten are fixed and the Windows leg has run fully green since (see git
  history around the two "Fix ... Windows-only test failures" commits
  for the exact diagnosis of each).

### Removed
- `requirements.txt`. Nothing referenced it - not the README, not CI
  (the pip-audit job builds its own list from `pyproject.toml`), not the
  helper scripts - and it had already drifted, declaring `torch>=2.1`
  where `pyproject.toml` says `torch>=2.1,<3`. Dependencies are declared
  in one place now, and a test keeps the second copy from coming back.
- Dead/unused imports across several `sandboxai` modules found by the
  new ruff CI job.
