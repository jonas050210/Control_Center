# Changelog

Human-readable summary of engineering-facing changes. Not a marketing
changelog: every entry here is something a maintainer would want to know
before touching CI, dependencies, or the desktop Control Center. See
`PROJECT.md` for the living technical spec and roadmap; this file only
records what changed and why.

## Unreleased

### Added
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

### Fixed
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
- Dead/unused imports across several `sandboxai` modules found by the
  new ruff CI job.
