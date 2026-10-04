<!--
Thanks for contributing. Keep the diff focused; see CONTRIBUTING.md.
-->

## What and why

<!-- What changes, and what problem it solves. The "why" matters more than
     the "what" - the diff already shows the what. -->

## Contracts touched

<!-- Tick only what this PR actually changes. Any tick needs a documented
     migration, not a silent edit (CONTRIBUTING.md -> "Non-negotiable
     contracts"). -->

- [ ] Observation vector (126 floats) — `contract.py` / `observation.gd`
- [ ] Action space (`MultiDiscrete([3,3,3,3,2,2])`) — `contract.py` / `action.gd`
- [ ] JSON-lines bridge commands — `rl_server.gd` / `godot_env.py`
- [ ] Replay / demonstration / dataset on-disk format
- [ ] Godot engine version target
- [ ] None of the above

## Checks run locally

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy`
- [ ] `python -m pytest -q`
- [ ] `gdlint scripts tests` and `gdformat --check scripts tests`
- [ ] `godot --headless --path . --script res://tests/run_tests.gd`
      <!-- If you could not run this, say so - CI is the only other place
           the engine suite runs. -->

## Notes

- [ ] `CHANGELOG.md` updated under `## Unreleased` (or: not maintainer-relevant)
- [ ] Bug fix includes a regression test that fails without the fix
