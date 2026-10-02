"""Unit tests for the Tkinter-free Control Center presentation layer.

These exercise `control_center_viewmodel` directly against the shapes
`SandboxAIAdapter` actually returns (see test_adapter.py for the adapter
side), without ever constructing a Tk widget - this module has no GUI
dependency, so it must be importable and testable everywhere.
"""

import math

import pytest

from sandboxai import control_center_viewmodel as vm
from sandboxai.config import TrainingConfig

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_number_handles_missing_and_present_values():
    assert vm.format_number(None) == "n/a"
    assert vm.format_number(float("nan")) == "n/a"
    assert vm.format_number(1234) == "1,234"
    assert vm.format_number(1234.5, decimals=1) == "1,234.5"


def test_format_fraction_as_percent():
    assert vm.format_fraction_as_percent(None) == "n/a"
    assert vm.format_fraction_as_percent(0.256) == "25.6%"


def test_format_duration_handles_missing_and_ranges():
    assert vm.format_duration(None) == "n/a"
    assert vm.format_duration(-1) == "n/a"
    assert vm.format_duration(5) == "5s"
    assert vm.format_duration(65) == "1m 05s"
    assert vm.format_duration(3725) == "1h 02m 05s"
    assert vm.format_duration(math.inf) == "n/a"


def test_format_bytes():
    assert vm.format_bytes(None) == "n/a"
    assert vm.format_bytes(512) == "512.0 B"
    assert vm.format_bytes(2048) == "2.0 KB"
    assert vm.format_bytes(5 * 1024 * 1024) == "5.0 MB"


@pytest.mark.parametrize("value", [math.inf, -math.inf, float("nan")])
def test_formatters_render_non_finite_values_as_na(value):
    """A diverged run's NaN/inf must never reach the screen as a number.

    `json` round-trips NaN and Infinity (a training summary written with
    `json.dump` reads straight back as a float), so these values genuinely
    arrive here. `format_number` used to raise OverflowError on an
    infinity - `int(round(inf))` - taking down the page that rendered it,
    while the other formatters printed "nan%" or "inf PB" as if they were
    measurements.
    """
    assert vm.format_number(value) == "n/a"
    assert vm.format_number(value, decimals=2) == "n/a"
    assert vm.format_fraction_as_percent(value) == "n/a"
    assert vm.format_duration(value) == "n/a"
    assert vm.format_bytes(value) == "n/a"
    assert vm.format_timestamp(value) == "n/a"


def test_format_timestamp_rejects_non_numeric():
    assert vm.format_timestamp(None) == "n/a"
    assert vm.format_timestamp("not-a-number") == "n/a"
    assert vm.format_timestamp(0) != "n/a"


# ---------------------------------------------------------------------------
# Chart downsampling
# ---------------------------------------------------------------------------


def test_downsample_series_keeps_bound_and_last_point():
    points = [(float(i), float(i)) for i in range(1000)]
    reduced = vm.downsample_series(points, max_points=100)
    assert len(reduced) <= 101
    assert reduced[-1] == points[-1]


def test_downsample_series_noop_under_the_limit():
    points = [(0.0, 0.0), (1.0, 1.0)]
    assert vm.downsample_series(points, max_points=100) == points


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_view_with_no_runs_yet():
    view = vm.dashboard_view({"latest_run": None, "active_processes": []})
    assert view["has_run"] is False
    assert view["state"] is None
    assert view["warnings"] == []


def test_dashboard_view_prefers_fresher_telemetry_row_when_available():
    snapshot = {
        "latest_run": {
            "run_id": "run-a",
            "run_dir": "/tmp/run-a",
            "control": {
                "state": "Running",
                "timesteps": 100,
                "total_training_steps": 1000,
                "updated_at": 10_000.0,
            },
            "config": {},
            "checkpoints": {},
            "evaluation": {},
            "warnings": [],
            "problems": [],
        },
        "active_processes": [{"run_dir": "/tmp/run-a"}],
    }
    telemetry = {"available": True, "latest": {"timesteps": 150, "mean_episode_reward": 2.5}}
    view = vm.dashboard_view(snapshot, telemetry)
    assert view["timesteps"] == 150
    assert view["reward"] == 2.5
    assert view["is_active_process"] is True
    assert view["stale"] is False


def test_dashboard_view_marks_stale_when_no_process_and_old_update():
    snapshot = {
        "latest_run": {
            "run_id": "run-a",
            "run_dir": "/tmp/run-a",
            "control": {"state": "Running", "updated_at": 1000.0},
            "config": {},
            "checkpoints": {},
            "evaluation": {},
            "warnings": [],
            "problems": [],
        },
        "active_processes": [],
    }
    view = vm.dashboard_view(snapshot, now=1000.0 + vm.STALE_STATUS_SECONDS + 1)
    assert view["stale"] is True
    assert len(view["warnings"]) == 1


def test_dashboard_view_never_marks_finished_runs_stale():
    snapshot = {
        "latest_run": {
            "run_id": "run-a",
            "run_dir": "/tmp/run-a",
            "control": {"state": "Finished", "updated_at": 1000.0},
            "config": {},
            "checkpoints": {},
            "evaluation": {},
            "warnings": [],
            "problems": [],
        },
        "active_processes": [],
    }
    view = vm.dashboard_view(snapshot, now=1000.0 + 10_000)
    assert view["stale"] is False


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_runs_table_rows_extracts_expected_columns():
    list_runs_result = {
        "runs": [
            {
                "run_id": "run-a",
                "status": {"state": "Running"},
                "progress": {"timesteps": 50, "target_timesteps": 100, "fraction": 0.5},
                "config": {"device": "cpu", "environment_count": 4, "env_workers": 2},
                "checkpoints": {"count": 2, "has_best": True},
                "evaluation": {"latest": {"mean_episode_reward": 1.1, "win_rate": 0.4}},
                "modified_utc": "2024-01-01T00:00:00Z",
                "warnings": ["x"],
                "run_dir": "/tmp/run-a",
            }
        ]
    }
    rows = vm.runs_table_rows(list_runs_result)
    assert rows[0]["run_id"] == "run-a"
    assert rows[0]["progress_percent"] == 50.0
    assert rows[0]["warning_count"] == 1


def test_process_table_rows_extracts_pid_and_progress():
    processes = [
        {
            "id": "p1",
            "kind": "training",
            "pid": 4242,
            "started_at": 10.0,
            "run_dir": "/tmp/run-a",
            "meta": {"run_id": "run-a", "env_workers": 2, "environment_count": 4},
            "backend": {
                "state": "Running",
                "timesteps": 250,
                "total_training_steps": 1000,
                "updated_at": 20.0,
            },
        }
    ]
    rows = vm.process_table_rows(processes)
    assert rows[0]["pid"] == 4242
    assert rows[0]["progress_percent"] == 25.0
    assert rows[0]["status"] == "Running"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluation_view_reports_unavailable_without_crashing():
    assert vm.evaluation_view(None)["available"] is False
    assert vm.evaluation_view({"available": False, "error": "boom"})["available"] is False


def test_evaluation_view_structures_action_head_diagnostics():
    summary = {
        "path": "/tmp/eval.json",
        "episodes": 20,
        "timesteps": 1000,
        "mean_episode_reward": 2.0,
        "win_rate": 0.6,
        "loss_rate": 0.3,
        "timeout_rate": 0.1,
        "mean_kills": 1.5,
        "mean_accuracy": 0.4,
        "policy_shoot_request_rate": 0.7,
        "action_pipeline": {"fire_conversion_rate": 0.9, "localization": "matched"},
    }
    view = vm.evaluation_view(summary)
    assert view["outcomes"]["win_rate"] == 0.6
    assert view["action_head_diagnostics"]["discharge_rate"] == 0.9
    assert view["action_head_diagnostics"]["localization"] == "matched"


def test_evaluation_comparison_rows_skips_unavailable_entries():
    summaries = [
        {"path": "a.json", "win_rate": 0.5},
        {"available": False, "error": "missing"},
    ]
    rows = vm.evaluation_comparison_rows(summaries)
    assert len(rows) == 1
    assert rows[0]["path"] == "a.json"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_history_rows_labels_each_row_with_its_source():
    history = [
        {
            "directory": "/tmp/sweep-1",
            "results": [{"environments": 4, "workers": 1, "steps_per_second": 100.0}],
        },
        {
            "directory": "/tmp/sweep-2",
            "results": [{"environments": 8, "workers": 1, "steps_per_second": 180.0}],
        },
    ]
    rows = vm.benchmark_history_rows(history)
    assert {row["source"] for row in rows} == {"/tmp/sweep-1", "/tmp/sweep-2"}


# ---------------------------------------------------------------------------
# Training form
# ---------------------------------------------------------------------------


def test_default_training_values_match_training_config_defaults():
    defaults = vm.default_training_values()
    assert defaults["environment_count"] == str(TrainingConfig().environment_count)
    assert defaults["device"] == TrainingConfig().device


def test_parse_training_form_builds_a_validated_config():
    values = vm.default_training_values()
    values.update({"run_id": "unit-test", "environment_count": "4", "total_training_steps": "1000"})
    config = vm.parse_training_form(values)
    assert isinstance(config, TrainingConfig)
    assert config.run_id == "unit-test"
    assert config.environment_count == 4
    assert config.total_training_steps == 1000


def test_parse_training_form_rejects_bad_types_with_a_clear_message():
    values = vm.default_training_values()
    values["environment_count"] = "not-a-number"
    with pytest.raises(ValueError, match="Environments"):
        vm.parse_training_form(values)


def test_parse_training_form_rejects_invalid_choice():
    values = vm.default_training_values()
    values["device"] = "quantum"
    with pytest.raises(ValueError, match="Device"):
        vm.parse_training_form(values)


def test_parse_training_form_rejects_unknown_fields():
    values = vm.default_training_values()
    values["totally_made_up_field"] = "1"
    with pytest.raises(ValueError, match="unknown training fields"):
        vm.parse_training_form(values)


def test_parse_training_form_surfaces_domain_validation_errors():
    values = vm.default_training_values()
    values["total_training_steps"] = "-5"
    with pytest.raises(ValueError):
        vm.parse_training_form(values)


def test_parse_training_form_blank_string_field_falls_back_to_the_config_default():
    values = vm.default_training_values()
    values["godot_executable"] = ""  # cleared by the user
    config = vm.parse_training_form(values)
    assert config.godot_executable == TrainingConfig().godot_executable == "godot"


def test_training_field_groups_partition_basic_and_advanced():
    groups = vm.training_field_groups()
    basic_names = {spec.name for spec in groups["basic"]}
    advanced_names = {spec.name for spec in groups["advanced"]}
    assert "environment_count" in basic_names
    assert "learning_rate" in advanced_names
    assert basic_names.isdisjoint(advanced_names)


def test_hardware_profile_view_reports_not_run_when_absent():
    view = vm.hardware_profile_view(None)
    assert view["available"] is False
    assert "wizard" in view["note"].lower()


def test_hardware_profile_view_shows_only_measured_throughput():
    profile = {
        "selected_device": "cpu",
        "device": "cpu",
        "inference_device": "cpu",
        "measurement_steps": 5000,
        "fallback": False,
        "note": "Selected cpu",
        "godot_available": True,
        "host": {"cuda_available": False},
        "measurements": [
            {
                "label": "cpu",
                "status": "measured",
                "steps_per_second": 123.4,
                "wall_seconds": 40.5,
                "error": None,
            },
            {"label": "cuda", "status": "unavailable", "steps_per_second": None, "error": None},
        ],
    }
    view = vm.hardware_profile_view(profile)
    assert view["available"] is True
    assert view["selected_device"] == "cpu"
    # No CUDA on this host -> no cuda_device line surfaced.
    assert "cuda_device" not in view
    rows = view["measurements"]
    assert rows[0]["steps_per_second"] == "123.4"
    assert rows[1]["steps_per_second"] == "n/a"


def test_hardware_profile_view_surfaces_cuda_device_when_present():
    profile = {
        "selected_device": "hybrid",
        "device": "cuda",
        "inference_device": "cpu",
        "measurement_steps": 5000,
        "fallback": False,
        "note": "",
        "godot_available": True,
        "host": {"cuda_available": True, "cuda_device": "RTX 4060 Ti"},
        "measurements": [],
    }
    view = vm.hardware_profile_view(profile)
    assert view["cuda_device"] == "RTX 4060 Ti"


def test_training_values_from_profile_applies_measured_device():
    profile = {
        "selected_device": "hybrid",
        "device": "cuda",
        "inference_device": "cpu",
        "fallback": False,
    }
    values = vm.training_values_from_profile(profile)
    assert values["device"] == "cuda"
    assert values["inference_device"] == "cpu"


def test_training_values_from_profile_ignores_fallback():
    profile = {"device": "cpu", "inference_device": "cpu", "fallback": True}
    baseline = vm.default_training_values()
    values = vm.training_values_from_profile(profile)
    assert values["device"] == baseline["device"]


def test_training_values_from_profile_without_profile_is_defaults():
    assert vm.training_values_from_profile(None) == vm.default_training_values()


# ---------------------------------------------------------------------------
# Agents (lifecycle registry)
# ---------------------------------------------------------------------------


def test_lifecycle_colors_cover_every_derived_state():
    from sandboxai import agents

    for constant in (
        agents.LIFECYCLE_AVAILABLE,
        agents.LIFECYCLE_LAUNCHING,
        agents.LIFECYCLE_RUNNING,
        agents.LIFECYCLE_PAUSED,
        agents.LIFECYCLE_STOPPING,
        agents.LIFECYCLE_STOPPED,
        agents.LIFECYCLE_FINISHED,
        agents.LIFECYCLE_FAILED,
        agents.LIFECYCLE_RESTARTING,
    ):
        assert constant in vm.LIFECYCLE_COLORS, constant


def test_agent_table_rows_relay_published_facts_only():
    rows = vm.agent_table_rows(
        [
            {
                "agent_id": "a1",
                "kind": "training",
                "lifecycle": "RUNNING",
                "pid": 4242,
                "backend": {
                    "environment_count": 12,
                    "timesteps": 3_000,
                    "total_training_steps": 10_000,
                    "steps_per_second": 91.5,
                    "mean_episode_reward": 0.42,
                },
                "meta": {"env_workers": 4, "device": "cpu"},
                "name": "run-1",
                "created_at": 1000.0,
            },
            {
                # A benchmark agent publishes no backend metrics; every
                # missing fact must stay None, never be guessed.
                "agent_id": "a2",
                "kind": "benchmark",
                "lifecycle": "FINISHED",
                "meta": {"environment_counts": [4, 8]},
                "created_at": 1001.0,
            },
        ]
    )
    assert rows[0]["environment_count"] == 12
    assert rows[0]["env_workers"] == 4
    assert rows[0]["device"] == "cpu"
    assert rows[0]["timesteps"] == 3_000
    assert rows[0]["name"] == "run-1"
    assert rows[1]["name"] == "a2"
    assert rows[1]["timesteps"] is None
    assert rows[1]["environment_count"] is None
    assert rows[1]["error"] is None


def test_agent_progress_percent_is_bounded_and_honest():
    row = {"timesteps": 2_500, "total_steps": 10_000}
    assert vm.agent_progress_percent(row) == 25.0
    # Never above 100 even if the backend overshoots its total.
    assert vm.agent_progress_percent({"timesteps": 99, "total_steps": 10}) == 100.0
    assert vm.agent_progress_percent({"timesteps": None, "total_steps": 10}) is None
    assert vm.agent_progress_percent({"timesteps": 5, "total_steps": 0}) is None


def test_agent_action_availability_matrix():
    # Training agent: the full cooperative set.
    running_training = vm.agent_action_availability("RUNNING", "training")
    assert running_training["pause"] and running_training["stop"]
    assert not running_training["resume"]
    assert not running_training["remove"]
    assert running_training["pause_unsupported_reason"] is None

    paused_training = vm.agent_action_availability("PAUSED", "training")
    assert paused_training["resume"] and not paused_training["pause"]

    # Benchmark/evaluation agents have no pause protocol: the reason says so.
    running_benchmark = vm.agent_action_availability("RUNNING", "benchmark")
    assert not running_benchmark["pause"]
    assert "no pause protocol" in running_benchmark["pause_unsupported_reason"]
    assert running_benchmark["stop"]

    # A stop already in progress disables pressing stop again.
    stopping = vm.agent_action_availability("STOPPING", "training")
    assert not stopping["stop"] and not stopping["pause"]
    assert stopping["force_stop"]

    # Terminal agents can be restarted or removed, not stopped.
    stopped = vm.agent_action_availability("STOPPED", "training")
    assert stopped["restart"] and stopped["remove"]
    assert not stopped["stop"] and not stopped["force_stop"]

    failed = vm.agent_action_availability("FAILED", "training")
    assert failed["restart"] and failed["remove"]


def test_launch_slot_view_resolves_auto_workers_and_reports_invalid_values():
    slot = vm.launch_slot_view(vm.default_training_values())
    assert slot["state"] == "AVAILABLE", slot["errors"]
    assert slot["warnings"] == []
    summary = slot["summary"]
    expected = vm.parse_training_form(vm.default_training_values())
    assert summary["env_workers"] == expected.resolved_env_workers()
    assert summary["environment_count"] == expected.environment_count

    invalid = vm.launch_slot_view({**vm.default_training_values(), "environment_count": "zero"})
    assert invalid["state"] == "INVALID"
    assert invalid["summary"] is None
    assert any("Environments" in error for error in invalid["errors"])

    # Runtime incompatibilities (e.g. CUDA requested without CUDA) merge in.
    with_cuda = vm.launch_slot_view(
        {**vm.default_training_values(), "device": "cuda"},
        compatibility={"valid": False, "errors": ["no CUDA runtime"], "warnings": []},
    )
    assert with_cuda["state"] == "INVALID"
    assert "no CUDA runtime" in with_cuda["errors"]


def test_topology_rows_mirror_the_sharded_bridge_plan():
    rows = vm.topology_rows(10, 3)
    assert [(row["worker"], row["environments"]) for row in rows] == [(0, 4), (1, 3), (2, 3)]
    assert rows[0]["first_environment"] == 0 and rows[0]["last_environment"] == 3
    assert rows[1]["first_environment"] == 4
    assert vm.topology_rows(4, 1) == [
        {"worker": 0, "first_environment": 0, "last_environment": 3, "environments": 4}
    ]
    # Invalid or missing input renders nothing, never a guessed plan.
    assert vm.topology_rows(None, 4) == []
    assert vm.topology_rows(8, 0) == []
    assert vm.topology_rows("eight", 2) == []


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


def test_benchmark_workflow_view_idle_shows_every_phase_pending():
    view = vm.benchmark_workflow_view(running=False, event=None, report=None)
    assert [phase["status"] for phase in view["phases"]] == ["pending"] * 6
    assert "idle" in view["detail"]
    assert [phase["key"] for phase in view["phases"]] == [
        "discovery",
        "screening",
        "devices",
        "validation",
        "recommendation",
        "apply",
    ]


def test_benchmark_workflow_view_marks_the_active_stage_while_running():
    view = vm.benchmark_workflow_view(
        running=True,
        event={"stage": "screening", "status": "started", "index": 1, "total": 4},
        report=None,
    )
    states = {phase["key"]: phase["status"] for phase in view["phases"]}
    assert states["screening"] == "active"
    assert states["validation"] == "pending"
    assert "screening" in view["detail"]
    assert "2/4" in view["detail"]


def test_benchmark_workflow_view_reads_real_stage_outcomes_from_the_report():
    report = {
        "status": "completed",
        "elapsed_seconds": 61.0,
        "stages": [
            {"name": "discovery", "status": "completed"},
            {"name": "screening", "status": "completed"},
            {"name": "devices", "status": "skipped"},
            {"name": "validation", "status": "completed"},
        ],
        "recommendation": {"environment_count": 8, "env_workers": 2},
    }
    view = vm.benchmark_workflow_view(running=False, event=None, report=report, applied=True)
    states = {phase["key"]: phase["status"] for phase in view["phases"]}
    assert states == {
        "discovery": "done",
        "screening": "done",
        "devices": "skipped",
        "validation": "done",
        "recommendation": "done",
        "apply": "done",
    }
    assert "completed" in view["detail"]


def test_benchmark_workflow_view_keeps_failures_visible():
    report = {
        "status": "unavailable",
        "elapsed_seconds": 0.2,
        "stages": [{"name": "discovery", "status": "failed", "reason": "no godot"}],
        "recommendation": None,
    }
    view = vm.benchmark_workflow_view(running=False, event=None, report=report, applied=False)
    states = {phase["key"]: phase["status"] for phase in view["phases"]}
    assert states["discovery"] == "failed"
    assert states["recommendation"] == "failed"
    assert states["apply"] == "skipped"


def test_benchmark_pipeline_rows_label_stages_and_keep_failures():
    rows = vm.benchmark_pipeline_rows(
        {
            "stages": [
                {
                    "name": "screening",
                    "configurations": [
                        {
                            "environments": 8,
                            "workers": 2,
                            "status": "ok",
                            "steps_per_second": 120.0,
                            "vector_step_latency_p50_ms": 1.0,
                            "vector_step_latency_p95_ms": 2.5,
                            "latency_jitter": 2.5,
                            "startup_seconds": 0.4,
                        },
                        {
                            "environments": 8,
                            "workers": 4,
                            "status": "failed",
                            "error": "worker refused to start",
                        },
                    ],
                },
                {
                    "name": "validation",
                    "configurations": [
                        {
                            "environments": 8,
                            "workers": 2,
                            "status": "measured",
                            "device": "cpu",
                            "steps": 3000,
                            "steps_per_second": 41.0,
                        }
                    ],
                },
            ]
        }
    )
    assert [row["stage"] for row in rows] == ["screening", "screening", "validation"]
    assert rows[0]["steps_per_second"] == 120.0
    assert rows[1]["status"] == "failed" and rows[1]["error"]
    assert rows[2]["device"] == "cpu"
    assert vm.benchmark_pipeline_rows(None) == []


def test_benchmark_recommendation_view_requires_a_real_recommendation():
    empty = vm.benchmark_recommendation_view(None)
    assert not empty["available"] and empty["reason"]
    broken = vm.benchmark_recommendation_view({"environment_count": "many"})
    assert not broken["available"]

    view = vm.benchmark_recommendation_view(
        {
            "environment_count": 24,
            "env_workers": 4,
            "device": "cpu",
            "inference_device": "cpu",
            "expected_steps_per_second": 380.2,
            "basis": "validated_training_slice",
            "rationale": ["measured 380.2 steps/s"],
            "warnings": [],
        }
    )
    assert view["available"]
    assert view["summary"].startswith("24 environments / 4 workers")
    assert view["basis"] == "validated_training_slice"


def test_pipeline_progress_view_formats_counts_and_messages():
    counted = vm.pipeline_progress_view(
        {
            "stage": "screening",
            "status": "completed",
            "index": 2,
            "total": 6,
            "configuration": {"environments": 8, "workers": 2},
        }
    )
    assert counted["text"] == "screening: 3/6 — 8 envs / 2 workers (completed)"
    assert counted["fraction"] == 0.5

    messaged = vm.pipeline_progress_view(
        {"stage": "devices", "status": "skipped", "message": "reusing profile"}
    )
    assert messaged["text"] == "devices: reusing profile"
    assert messaged["fraction"] is None
    assert vm.pipeline_progress_view(None) == {"text": "idle", "stage": None, "fraction": None}


def test_launch_field_specs_exclude_run_id_and_experiment_id():
    names = [spec.name for spec in vm.launch_field_specs()]
    assert names == [
        "environment_count",
        "env_workers",
        "total_training_steps",
        "device",
        "curriculum_mode",
    ]


def test_benchmark_live_telemetry_view_tracks_live_step_events():
    live_event = {
        "stage": "screening",
        "status": "running",
        "index": 1,
        "total": 4,
        "configuration": {"environments": 16, "workers": 4},
        "live": {
            "phase": "stepping",
            "completed_steps": 120,
            "total_steps": 1920,
            "target_steps": 600,
            "steps_per_second": 960.0,
            "vector_step_latency_p50_ms": 1.4,
            "vector_step_latency_p95_ms": 2.8,
            "latency_jitter": 2.0,
            "elapsed_seconds": 2.0,
            "resources": {"cpu_percent": 55.0},
        },
        "completed_rows": [
            {
                "stage": "screening",
                "status": "ok",
                "environments": 1,
                "workers": 1,
                "steps": 600,
                "steps_per_second": 300.0,
                "vector_step_latency_p50_ms": 3.0,
                "vector_step_latency_p95_ms": 4.5,
                "latency_jitter": 1.5,
            }
        ],
    }
    view = vm.benchmark_live_telemetry_view(running=True, event=live_event, report=None)
    assert view["live_fps"] == 960.0
    assert view["peak_fps"] == 960.0
    assert view["live_steps"] == 1920.0
    assert view["steps_per_env"] == 120.0
    assert view["p50_ms"] == 1.4
    assert view["p95_ms"] == 2.8
    assert view["jitter"] == 2.0
    assert view["cpu_percent"] == 55.0
    assert len(view["rows"]) == 1
    assert len(view["chart_points"]) == 2


def test_ttk_testing_view_and_helpers():
    assert vm.format_ascii_bar(0.5, 4) == "[██░░]"
    assert vm.estimate_training_duration(6000, 100.0) == "1m 00s"
    health = vm.ppo_health_view(
        {"approx_kl": 0.01, "clip_fraction": 0.1, "explained_variance": 0.8, "entropy": -1.2}
    )
    assert health["status"] == "OPTIMAL" and health["healthy"] is True
    tview = vm.ttk_testing_view(
        {
            "live_session": {
                "running": True,
                "pid": 4242,
                "in_ttk_testing": True,
                "ttk_session_active": True,
                "launcher_found": True,
                "resolved_launcher": r"C:\Users\jonas\OneDrive\Desktop\Roblox Player.lnk",
                "window_found": True,
                "window_width": 1920,
                "window_height": 1080,
                "window_focused": True,
                "detected_place_id": "120189115846709",
            },
            "calibration": {
                "recoil_values_and_pattern": {
                    "measured_value": "vertical_kick=1.4deg",
                    "notes": "measured from screenshot",
                }
            },
            "mechanics": {
                "verified": [
                    {"mechanic": "fire", "implementation_rule": "M1", "source_label": "s"}
                ],
                "calibration_required": [
                    {
                        "mechanic": "recoil_values_and_pattern",
                        "implementation_rule": "measure",
                        "source_label": "s",
                    }
                ],
                "excluded": [],
            },
        }
    )
    assert tview["connected"] is True
    assert "120189115846709" in tview["place_text"]
    assert tview["calibrated_count"] == 1
    assert tview["calibration_progress_pct"] == 100.0


def test_ubuntu_cpu_turbo_convergence_and_tactical_lab_views():
    uview = vm.ubuntu_cpu_turbo_view(
        {
            "os_name": "Ubuntu 24.04 LTS",
            "cpu_model": "AMD Ryzen",
            "logical_cores": 16,
            "physical_cores_est": 8,
            "cpu_governor": "performance",
            "recommended_envs": 28,
            "recommended_workers": 7,
            "recommended_trainer_threads": 4,
            "anti_thrash_active": True,
        }
    )
    assert uview["available"] is True
    assert uview["anti_thrash_active"] is True
    assert "28e/7w" in uview["badge"]

    conv_up = vm.training_convergence_view([(i * 1000, float(i * 10)) for i in range(1, 13)])
    assert conv_up["state"] == "IMPROVING"
    conv_flat = vm.training_convergence_view([(i * 1000, 50.0) for i in range(1, 15)])
    assert conv_flat["state"] == "PLATEAU"

    tactical = vm.tactical_combat_profile_view(
        {
            "available": True,
            "win_rate": 0.75,
            "loss_rate": 0.20,
            "mean_kills": 1.5,
            "mean_deaths": 0.5,
            "mean_damage_dealt": 150.0,
            "mean_damage_taken": 60.0,
            "mean_episode_length": 140.0,
        }
    )
    assert tactical["available"] is True
    assert tactical["kd_ratio"] == "3.00"
    assert tactical["archetype"] == "AGGRESSIVE ENTRY FRAGGER"


# ---------------------------------------------------------------------------
# Training budget (Steps | Time) and the benchmark mode plan
# ---------------------------------------------------------------------------


def test_budget_view_steps_and_time_are_both_valid():
    steps = vm.budget_view("steps", steps_raw="250000", minutes_raw="")
    assert steps["errors"] == []
    assert steps["steps"] == 250000
    assert "250,000" in steps["budget_line"]

    time = vm.budget_view("time", steps_raw="", minutes_raw="45")
    assert time["errors"] == []
    assert time["minutes"] == 45.0
    # A time budget is a cooperative stop, and the wording has to say so:
    # the trainer still finishes a boundary and saves its final checkpoint.
    assert "cooperative stop" in time["budget_line"]


def test_budget_view_rejects_unusable_values():
    empty = vm.budget_view("steps", steps_raw="", minutes_raw="")
    assert empty["errors"]
    negative = vm.budget_view("steps", steps_raw="-5", minutes_raw="")
    assert negative["errors"]
    # A budget beyond a day is a typo, not a plan.
    marathon = vm.budget_view("time", steps_raw="", minutes_raw="2000")
    assert marathon["errors"]


def test_auto_benchmark_plan_reaches_the_wide_ladder():
    view = vm.benchmark_mode_view("auto", minutes_raw="15", cpu_count=32)
    assert view["errors"] == []
    assert view["environments"][-1] == 128
    assert view["environments"][-3:] == [64, 96, 128]
    # The worker ladder must probe past the conservative auto recommendation:
    # that recommendation is what produced the 64-env / 4-worker runs that
    # left the CPU at 10-20 %.
    assert 32 in view["workers"]
    assert view["expected_configurations"] > 20
    assert "environments_note" in view


def test_auto_benchmark_plan_is_the_pipeline_ladder():
    from sandboxai.benchmark_pipeline import default_environment_counts, default_worker_counts

    view = vm.benchmark_mode_view("auto", minutes_raw="15", cpu_count=32)
    assert view["environments"] == list(default_environment_counts(32))
    assert view["workers"] == list(default_worker_counts(128, 32))


def test_push_benchmark_plan_goes_wider_than_auto():
    auto = vm.benchmark_mode_view("auto", minutes_raw="15", cpu_count=32)
    push = vm.benchmark_mode_view("push", minutes_raw="15", cpu_count=32)
    assert max(push["environments"]) > max(auto["environments"])
    assert push["warnings"], "push mode must say that it oversubscribes"


def test_custom_benchmark_plan_skips_impossible_pairs():
    view = vm.benchmark_mode_view(
        "custom",
        environment_text="16,32",
        worker_text="4,64",
        steps_raw="",
        minutes_raw="5",
    )
    assert view["errors"] == []
    # 64 workers cannot be fed by 16 or 32 environments, so those pairs are
    # skipped rather than clamped into a duplicate of the 4-worker rows.
    assert view["expected_configurations"] == 2
    assert view["warnings"], "dropping half the requested pairs must be said out loud"
    assert view["budget_mode"] == "time"

    broken = vm.benchmark_mode_view(
        "custom", environment_text="", worker_text="", steps_raw="", minutes_raw="5"
    )
    assert broken["errors"]


def test_training_form_carries_the_trainers_own_time_budget():
    """Time mode must reach the trainer, not just the window's watchdog.

    The Control Center used to enforce the budget itself by asking the
    process to stop. The form now carries the same minutes into
    ``TrainingConfig.max_train_minutes``, so the run stops itself at a safe
    boundary even if nobody is watching the window.
    """
    config = vm.parse_training_form({"max_train_minutes": "45"})
    assert config.max_train_minutes == 45.0
    # Steps mode is the default and means "no wall-clock budget".
    assert vm.default_training_values()["max_train_minutes"] == "0.0"

    with pytest.raises(ValueError):
        vm.parse_training_form({"max_train_minutes": "soon"})
    with pytest.raises(ValueError):
        vm.parse_training_form({"max_train_minutes": "-5"})
