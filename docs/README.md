# Documentation index

Every Markdown file under `docs/` is listed here. This index is enforced:
`python/tests/test_docs_consistency.py` fails if a document exists without
an entry, or an entry points at a document that does not exist. A document
nobody links to is a document nobody maintains.

Start with the root [`README.md`](../README.md) for installation and the
first training run, and [`PROJECT.md`](../PROJECT.md) for the living
technical spec and roadmap. [`CONTRIBUTING.md`](../CONTRIBUTING.md)
describes the workflow, the lint/test gates and the single sources of
truth for versions and the engine contract.

## Contracts and architecture

| Document | What it is for |
| --- | --- |
| [OBSERVATION_ACTION_CONTRACT.md](OBSERVATION_ACTION_CONTRACT.md) | The single source of truth for what a policy sees and does. Read this before touching the bridge, the observation builder or the action space. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The Godot/Python runtime split, the module map, and the testing rules the suite is held to. |
| [PYTHON_MODULE_MAP.md](PYTHON_MODULE_MAP.md) | Where to find things in `python/sandboxai/`: all 47 modules grouped by theme, with each description generated from the module's own docstring. |
| [CURRICULUM_AND_COMBAT.md](CURRICULUM_AND_COMBAT.md) | Curriculum levels, enemy behaviour and the multi-enemy combat model. |
| [REPLAY_AND_METRICS.md](REPLAY_AND_METRICS.md) | Replay format, research metrics, generalization conditions and curriculum telemetry. |
| [TTK_TESTING_REFERENCE.md](TTK_TESTING_REFERENCE.md) | Official TTK Testing evidence, calibration gaps, screenshot protocol and excluded mechanics. |

## Running the simulator and the tooling

| Document | What it is for |
| --- | --- |
| [CONTROL_CENTER.md](CONTROL_CENTER.md) | The headless desktop Control Center (`python3 main.py`): pages, lifecycle actions, benchmark pipeline and honesty rules. |
| [ADAPTER_AND_DESKTOP_CONTROL_CENTER.md](ADAPTER_AND_DESKTOP_CONTROL_CENTER.md) | The Python adapter and the Tkinter desktop Control Center built on top of it. |
| [DEBUG_GUI_AND_BENCHMARKING.md](DEBUG_GUI_AND_BENCHMARKING.md) | The in-scene debug overlay and how to run and read the benchmark suites. |
| [RUN_LOCAL_VALIDATION.md](RUN_LOCAL_VALIDATION.md) | Step-by-step full validation on a machine that actually has Godot installed, including the Windows/WSL paths. |

## Reports and audits

These are point-in-time records. They are kept because they explain *why*
parts of the system look the way they do; they are not specifications and
may describe state that has since changed.

| Document | What it is for |
| --- | --- |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Full-repository bug hunt: the bugs found, their root causes and the regression tests added for each. |
| [CONTROL_CENTER_ARCHITECTURE_AUDIT.md](CONTROL_CENTER_ARCHITECTURE_AUDIT.md) | Architecture audit of the Control Center against `PROJECT.md` and the bridge. |
| [RESEARCH_SANDBOX_REPORT.md](RESEARCH_SANDBOX_REPORT.md) | Implementation report for the perception-driven RL sandbox work. |
