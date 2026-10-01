# Python module map

`sandboxai` is a flat package: all 49 modules sit directly under
`python/sandboxai/`. That is deliberate. Every module's import path is
public API - it appears in the docs, in user scripts and in saved run
manifests - so rearranging the files into subpackages would rewrite
several hundred call sites and break every downstream import to buy
something no test can check. What subpackages would have bought is
navigability, and this file buys that instead.

The grouping below is thematic, not a dependency order. Imports do not
flow strictly downwards through it: `pipeline` composes curriculum,
metrics and replay, and `config` reaches `sharded_env`, `self_play` and
`selection` from inside functions precisely so the module-level graph
stays acyclic. That acyclicity is the invariant worth having, and it is
tested.

This map is enforced too. `python/tests/test_docs_consistency.py` fails
if a module is missing here, listed twice, listed but absent from the
package, or described with a line that is not the module's own docstring
summary. To change a description, change the module docstring.

## Contract and configuration

What a policy sees, what it may do, and how a run is described. Read
these first; everything else restates or consumes them.

| Module | Summary |
| --- | --- |
| [`contract`](../python/sandboxai/contract.py) | Local Godot/Python observation and action contract description. |
| [`config`](../python/sandboxai/config.py) | Central, serialisable configuration for SandboxAI experiments. |
| [`control_center_schema`](../python/sandboxai/control_center_schema.py) | Versioned data contract shared by Control Center producers and consumers. |
| [`weapons`](../python/sandboxai/weapons.py) | Local weapon-calibration profiles — the Python mirror of the Godot tables. |

## The Godot bridge

Everything that speaks the JSON-lines protocol to a Godot process, or
measures how fast it does so.

| Module | Summary |
| --- | --- |
| [`godot_env`](../python/sandboxai/godot_env.py) | Gymnasium/SB3 adapters for the headless Godot JSON-lines bridge. |
| [`sharded_env`](../python/sandboxai/sharded_env.py) | Multi-process (sharded) Godot environment execution. |
| [`wsl`](../python/sandboxai/wsl.py) | WSL -> Windows interop for launching Windows Godot binaries from WSL. |
| [`runtime_validation`](../python/sandboxai/runtime_validation.py) | Automated headless runtime validation harness for the real Godot bridge. |
| [`benchmark`](../python/sandboxai/benchmark.py) | Headless Godot simulation throughput benchmark. |
| [`benchmark_suites`](../python/sandboxai/benchmark_suites.py) | Benchmark suites (Phase 13). |
| [`benchmark_pipeline`](../python/sandboxai/benchmark_pipeline.py) | Staged, budget-aware runtime benchmark that recommends a configuration. |
| [`training_profile`](../python/sandboxai/training_profile.py) | Low-overhead wall-clock profiling for PPO and the Godot bridge. |
| [`hardware_profile`](../python/sandboxai/hardware_profile.py) | Single source of truth for hardware device-comparison measurement. |

## Training

Turning environment interaction into policy weights.

| Module | Summary |
| --- | --- |
| [`ppo`](../python/sandboxai/ppo.py) | Stable-Baselines3 PPO orchestration for the Godot environment. |
| [`bc`](../python/sandboxai/bc.py) | Small multi-head PyTorch behavior-cloning policy. |
| [`schedule`](../python/sandboxai/schedule.py) | Pure PPO rollout-schedule helpers. |
| [`policies`](../python/sandboxai/policies.py) | Multi-policy ("multi-brain") architecture: identity, slots and matchups. |
| [`pipeline`](../python/sandboxai/pipeline.py) | Integrated training pipeline: curriculum, conditions, metrics, replay. |
| [`dataset`](../python/sandboxai/dataset.py) | Human demonstration storage, validation, statistics and splitting. |
| [`randomization`](../python/sandboxai/randomization.py) | Deterministic training distribution over maps, conditions and spawns. |

## Curriculum, opponents and multi-agent

What the agent is made to face, and who it faces.

| Module | Summary |
| --- | --- |
| [`auto_curriculum`](../python/sandboxai/auto_curriculum.py) | Automatic curriculum: promotion, demotion and the guards around them. |
| [`curriculum_stages`](../python/sandboxai/curriculum_stages.py) | Curriculum integration: stages, gating and the episode stream. |
| [`conditions`](../python/sandboxai/conditions.py) | Training/evaluation condition space and per-condition performance tracking. |
| [`league`](../python/sandboxai/league.py) | Checkpoint registry, opponent sampling and a self-play league. |
| [`self_play`](../python/sandboxai/self_play.py) | Two-policy/frozen-opponent foundation for multi-agent training and evaluation. |
| [`teamplay`](../python/sandboxai/teamplay.py) | Teamplay foundation: teams, teammate identity, comms and reward hooks. |

## Evaluation and metrics

Answering whether a checkpoint is actually better, with the weights
frozen.

| Module | Summary |
| --- | --- |
| [`evaluation`](../python/sandboxai/evaluation.py) | Weight-frozen evaluation and machine-readable summaries. |
| [`checkpoint_eval`](../python/sandboxai/checkpoint_eval.py) | Checkpoint-time evaluation: conditions, generalization, league, metrics. |
| [`generalization`](../python/sandboxai/generalization.py) | Generalization evaluation: does the policy fight, or did it memorize? |
| [`metrics`](../python/sandboxai/metrics.py) | Research / skill metrics: what the policy actually did, per episode. |
| [`selection`](../python/sandboxai/selection.py) | Configurable, recorded checkpoint-selection rule. |
| [`experiment`](../python/sandboxai/experiment.py) | Multi-seed experiment management, statistical aggregation and regression testing. |
| [`action_audit`](../python/sandboxai/action_audit.py) | Low-overhead policy-action diagnostics for training and evaluation. |
| [`ttk`](../python/sandboxai/ttk.py) | Manually measured time-to-kill (TTK) trials: schema, validation, statistics. |
| [`ttk_testing`](../python/sandboxai/ttk_testing.py) | Verified TTK Testing mechanics, evidence sources and calibration gates. |

## Runs, artifacts and telemetry

What a result was produced by, and how to find it again afterwards.

| Module | Summary |
| --- | --- |
| [`manifest`](../python/sandboxai/manifest.py) | Run manifest: what a result was produced BY. |
| [`replay`](../python/sandboxai/replay.py) | Deterministic episode replay: recording, storage, validation and playback. |
| [`telemetry`](../python/sandboxai/telemetry.py) | Structured training telemetry with optional CPU/GPU gauges. |
| [`run_control`](../python/sandboxai/run_control.py) | Cooperative process control for training launched by the Control Center. |
| [`run_inspection`](../python/sandboxai/run_inspection.py) | Read-only inspection of training runs on disk. |
| [`artifact_repository`](../python/sandboxai/artifact_repository.py) | Single read-only gateway for persisted runs, checkpoints and evaluations. |

## Desktop Control Center

The Tkinter engineering GUI. It reaches the rest of the system only
through `adapter`, and the Tk-free `control_center_viewmodel` holds the
presentation logic so it can be tested without a display.

| Module | Summary |
| --- | --- |
| [`adapter`](../python/sandboxai/adapter.py) | Stable application boundary for the SandboxAI Control Center. |
| [`agents`](../python/sandboxai/agents.py) | Agent lifecycle management for the headless Control Center. |
| [`control_center_desktop`](../python/sandboxai/control_center_desktop.py) | Tk desktop Control Center backed exclusively by :mod:`sandboxai.adapter`. |
| [`control_center_pages`](../python/sandboxai/control_center_pages.py) | Desktop Control Center pages. |
| [`control_center_viewmodel`](../python/sandboxai/control_center_viewmodel.py) | Pure presentation logic for the desktop Control Center. |
| [`control_center_widgets`](../python/sandboxai/control_center_widgets.py) | Reusable Tk infrastructure and presentation widgets for the desktop Control Center. |

## Entry points and developer tooling

Reached from a shell rather than from Python.

| Module | Summary |
| --- | --- |
| [`cli`](../python/sandboxai/cli.py) | Command-line workflow for recording, BC, PPO, evaluation, benchmarks and smoke tests. |
| [`gdscript_analysis`](../python/sandboxai/gdscript_analysis.py) | Static analysis for the Godot/GDScript half of SandboxAI. |
