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
