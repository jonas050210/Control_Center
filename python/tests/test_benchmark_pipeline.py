"""Tests for :mod:`sandboxai.benchmark_pipeline`.

Three layers:

* pure planning/selection logic (budgets, candidate grids, validation,
  recommendation) — no engine, no torch;
* orchestration with the stage functions injected — proves the stage
  order, budget accounting, cancellation, progress events and persistence
  without fabricating a single measurement;
* one end-to-end run against the simulated bridge so the real
  ``benchmark_simulation`` screening path is exercised too (POSIX only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandboxai.benchmark_pipeline import (
    DEFAULT_TIME_BUDGET_MINUTES,
    JITTER_THRESHOLD,
    MAX_TIME_BUDGET_MINUTES,
    MIN_TIME_BUDGET_MINUTES,
    NEAR_BEST_FRACTION,
    PIPELINE_FORMAT,
    RECOMMENDATION_FORMAT,
    PipelineBudget,
    allocate_time_budget,
    default_environment_counts,
    default_worker_counts,
    discover_reports,
    estimate_validation_steps,
    fit_candidates_to_budget,
    load_recommendation,
    mark_recommendation_applied,
    plan_candidates,
    recommend,
    recommendation_path,
    run_benchmark_pipeline,
    save_recommendation,
    select_finalists,
    validate_configuration,
    write_report,
)


def screen_row(environments: int, workers: int, sps: float, **extra):
    return {
        "environments": environments,
        "workers": workers,
        "status": "ok",
        "steps_per_second": sps,
        "vector_step_latency_p50_ms": 1.0,
        "vector_step_latency_p95_ms": extra.pop("p95", 2.0),
        "startup_seconds": 0.5,
        **extra,
    }


class TestPipelineBudget:
    def test_steps_mode_defaults_and_validation(self):
        budget = PipelineBudget.for_steps(500)
        assert budget.mode == "steps"
        assert budget.steps == 500
        # Slice sizes can never exceed the screening budget.
        assert budget.device_steps <= 500
        assert budget.validation_steps <= 500
        with pytest.raises(ValueError):
            PipelineBudget.for_steps(0)

    def test_time_mode_range_is_enforced(self):
        assert PipelineBudget.for_time(10).minutes == 10
        assert PipelineBudget.for_time(MIN_TIME_BUDGET_MINUTES).mode == "time"
        assert PipelineBudget.for_time(MAX_TIME_BUDGET_MINUTES).mode == "time"
        with pytest.raises(ValueError):
            PipelineBudget.for_time(MIN_TIME_BUDGET_MINUTES - 1)
        with pytest.raises(ValueError):
            PipelineBudget.for_time(MAX_TIME_BUDGET_MINUTES + 1)

    def test_default_budget_is_the_middle_of_the_offered_range(self):
        assert DEFAULT_TIME_BUDGET_MINUTES == 15.0

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError):
            PipelineBudget(mode="energy")


class TestCandidatePlanning:
    def test_environment_ladder_is_host_scaled_not_hardcoded(self):
        # A small host must not be planned a 64-environment sweep.
        assert default_environment_counts(2) == (1, 2, 4, 8)
        # A large host reaches the ladder cap.
        assert default_environment_counts(32)[-1] == 64

    def test_worker_ladder_is_bounded_by_environment_count(self):
        assert default_worker_counts(2, cpu_count=32) == (1, 2)
        assert default_worker_counts(16, cpu_count=32) == (1, 2, 4, 8)

    def test_plan_candidates_deduplicates_and_skips_invalid_pairs(self):
        # 4 workers with 2 environments would be clamped to 2 -> duplicate.
        plan = plan_candidates([2, 4], [1, 4])
        assert [(row["environments"], row["workers"]) for row in plan] == [
            (2, 1),
            (2, 2),
            (4, 1),
            (4, 4),
        ]
        # An explicit worker count above the environment count is clamped,
        # never measured as-is (the launcher would refuse it).
        assert plan[1]["workers"] == 2

    def test_plan_candidates_matches_launcher_compatibility(self):
        plan = plan_candidates(None, None, cpu_count=8)
        assert plan
        for row in plan:
            verdict = validate_configuration(
                row["environments"], row["workers"], "cpu", cpu_count=8
            )
            assert verdict["valid"], verdict["errors"]

    def test_budget_thinning_keeps_the_sweep_spread(self):
        candidates = plan_candidates([1, 2, 4, 8, 16], [1, 2, 4, 8], cpu_count=16)
        thinned, cap = fit_candidates_to_budget(candidates, 60.0)
        assert len(thinned) < len(candidates)
        assert 3.0 <= cap <= 30.0
        # Endpoints of the environment ladder survive thinning.
        environments = {row["environments"] for row in thinned}
        assert 1 in environments and 16 in environments


class TestAllocateTimeBudget:
    def test_shares_sum_to_the_budget(self):
        allocation = allocate_time_budget(
            10,
            candidate_count=8,
            finalist_count=4,
            device_candidate_count=1,
            has_training_runtime=True,
        )
        assert allocation["screen_seconds"] + allocation["device_seconds"] + (
            allocation["validation_seconds"]
        ) == pytest.approx(allocation["total_seconds"])
        assert allocation["screen_seconds"] > allocation["validation_seconds"]

    def test_no_training_runtime_folds_everything_into_screening(self):
        allocation = allocate_time_budget(
            10,
            candidate_count=8,
            finalist_count=4,
            device_candidate_count=1,
            has_training_runtime=False,
        )
        assert allocation["validation_seconds"] == 0.0
        assert allocation["device_seconds"] == 0.0
        assert allocation["screen_seconds"] == allocation["total_seconds"]

    def test_validation_split_is_even_across_finalists(self):
        allocation = allocate_time_budget(
            10,
            candidate_count=4,
            finalist_count=4,
            device_candidate_count=1,
            has_training_runtime=True,
        )
        per_finalist = allocation["validation_seconds_per_finalist"]
        assert per_finalist * 4 == pytest.approx(allocation["validation_seconds"])


class TestValidateConfiguration:
    def test_valid_configuration_reports_shards(self):
        verdict = validate_configuration(12, 4, "cpu")
        assert verdict["valid"]
        assert verdict["env_workers"] == 4
        assert [shard["count"] for shard in verdict["shards"]] == [3, 3, 3, 3]

    def test_more_workers_than_environments_is_an_error(self):
        verdict = validate_configuration(4, 8, "cpu")
        assert not verdict["valid"]
        assert verdict["errors"]

    def test_cuda_without_runtime_is_an_error_not_a_fallback(self):
        verdict = validate_configuration(8, 2, "cuda", runtime={"cuda_available": False})
        assert not verdict["valid"]
        assert "no CUDA runtime" in " ".join(verdict["errors"])

    def test_unknown_device_is_rejected(self):
        assert not validate_configuration(8, 2, "tpu")["valid"]

    def test_uneven_shards_warn_but_stay_valid(self):
        verdict = validate_configuration(10, 3, "cpu")
        assert verdict["valid"]
        assert verdict["warnings"]
        assert [shard["count"] for shard in verdict["shards"]] == [4, 3, 3]


class TestSelectionAndRecommendation:
    def test_finalists_are_ranked_by_throughput_then_stability(self):
        rows = [
            screen_row(8, 1, 100.0),
            screen_row(16, 4, 300.0),
            screen_row(16, 2, 200.0),
            screen_row(32, 8, 250.0, p95=100.0),  # fast but jittery
            screen_row(4, 1, 50.0),
            {"environments": 2, "workers": 1, "status": "failed", "error": "x"},
        ]
        finalists = select_finalists(rows, 3)
        assert [row["steps_per_second"] for row in finalists] == [300.0, 250.0, 200.0]

    def test_unmeasured_rows_are_never_selected(self):
        rows = [
            {"environments": 4, "workers": 1, "status": "skipped"},
            {"environments": 8, "workers": 1, "status": "failed"},
        ]
        assert select_finalists(rows, 3) == []

    def test_recommendation_requires_measurements(self):
        assert recommend([]) is None
        assert recommend([{"environments": 4, "workers": 1, "status": "failed"}]) is None

    def test_recommendation_prefers_stability_over_unstable_peak(self):
        rows = [
            screen_row(48, 8, 400.0, p95=100.0),  # peak, but unstable
            screen_row(24, 4, 380.0, p95=2.2),  # near-best and stable
            screen_row(12, 2, 390.0, p95=2.0),  # near-best, stable, fewer workers
        ]
        recommendation = recommend(rows)
        assert recommendation is not None
        assert recommendation["environment_count"] == 12
        assert recommendation["env_workers"] == 2
        assert recommendation["basis"] == "screening_only"
        assert any("unstable" in warning for warning in recommendation["warnings"])

    def test_validated_throughput_outranks_screening_throughput(self):
        rows = [
            screen_row(8, 1, 900.0),  # best bridge throughput
            screen_row(16, 4, 500.0),
        ]
        validation = [
            {"environments": 16, "workers": 4, "status": "ok", "steps_per_second": 120.0},
            {"environments": 8, "workers": 1, "status": "measured", "steps_per_second": 80.0},
        ]
        recommendation = recommend(rows, validation)
        assert recommendation is not None
        assert recommendation["environment_count"] == 16
        assert recommendation["basis"] == "validated_training_slice"
        assert recommendation["validated_steps_per_second"] == 120.0

    def test_unvalidated_configurations_are_excluded(self):
        rows = [screen_row(8, 1, 900.0), screen_row(16, 4, 500.0)]
        validation = [{"environments": 16, "workers": 4, "status": "ok", "steps_per_second": 120.0}]
        recommendation = recommend(rows, validation)
        assert recommendation["environment_count"] == 16

    def test_rationale_quotes_only_measured_numbers(self):
        rows = [screen_row(24, 4, 380.0, p95=2.2), screen_row(8, 1, 100.0)]
        recommendation = recommend(rows)
        assert recommendation is not None
        assert "380.0 steps/s" in recommendation["rationale"][0]
        assert f"within {NEAR_BEST_FRACTION:.0%}" in recommendation["rationale"][1]
        assert recommendation["expected_steps_per_second"] == 380.0

    def test_constants_pin_the_stability_policy(self):
        assert NEAR_BEST_FRACTION == 0.10
        assert JITTER_THRESHOLD == 4.0


class TestPersistence:
    def test_report_and_recommendation_round_trip(self, tmp_path):
        report = {
            "format": PIPELINE_FORMAT,
            "schema_version": 1,
            "created_utc": "2026-01-01T00:00:00Z",
            "status": "completed",
            "stages": [
                {
                    "name": "screening",
                    "status": "completed",
                    "configurations": [
                        screen_row(8, 2, 123.0),
                        {"environments": 4, "workers": 1, "status": "failed", "error": "x"},
                    ],
                }
            ],
            "recommendation": {"environment_count": 8, "env_workers": 2},
        }
        path = write_report(report, tmp_path / "run")
        assert path.name == "pipeline.json"
        # Only measured rows land in the benchmark-history-shaped file.
        measured = json.loads((tmp_path / "run" / "benchmark.json").read_text())
        assert len(measured) == 1 and measured[0]["steps_per_second"] == 123.0

        recommendation = {
            "format": RECOMMENDATION_FORMAT,
            "environment_count": 8,
            "env_workers": 2,
            "device": "cpu",
        }
        save_recommendation(recommendation, tmp_path, source_report=path)
        assert load_recommendation(tmp_path)["environment_count"] == 8
        assert load_recommendation(tmp_path)["applied_utc"] is None

        applied = mark_recommendation_applied(tmp_path)
        assert applied is not None and applied["applied_utc"]
        assert load_recommendation(tmp_path)["applied_utc"] is not None

        entries = discover_reports(tmp_path)
        assert len(entries) == 1
        assert entries[0]["status"] == "completed"
        assert entries[0]["recommendation"]["environment_count"] == 8

    def test_load_recommendation_tolerates_garbage(self, tmp_path):
        assert load_recommendation(tmp_path) is None
        path = recommendation_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_recommendation(tmp_path) is None
        path.write_text(json.dumps({"format": "something.else/v9"}), encoding="utf-8")
        assert load_recommendation(tmp_path) is None
        assert mark_recommendation_applied(tmp_path) is None

    def test_recommendation_lives_in_the_gitignored_dot_directory(self, tmp_path):
        assert recommendation_path(tmp_path).parts[-2] == ".sandboxai"


class TestOrchestration:
    """Stage order, budgeting, cancellation and honesty with injected stages."""

    @staticmethod
    def runtime(**overrides):
        runtime = {
            "godot_available": True,
            "godot_resolved_executable": "/usr/bin/godot",
            "godot_version": "4.7.2",
            "cpu_count": 8,
            "cpu_count_available_to_process": 8,
            "torch_available": True,
            "cuda_available": False,
        }
        runtime.update(overrides)
        return runtime

    def test_completed_pipeline_with_validation(self, tmp_path):
        events = []

        def screen(environments, workers):
            # Distinct measured throughputs: 8/2 wins on stability policy.
            table = {
                (4, 1): 90.0,
                (4, 2): 95.0,
                (4, 4): 100.0,
                (8, 1): 120.0,
                (8, 2): 150.0,
                (8, 4): 160.0,
            }
            return [screen_row(environments, workers, table[(environments, workers)])]

        def validate_training(environments, workers, device, steps):
            class Measurement:
                ok = True
                status = "measured"
                error = None

                def to_dict(self):
                    # (8, 2) is within the near-best band of (8, 4) and wins
                    # on stability (fewer workers at equal jitter).
                    return {"steps_per_second": 43.0 if (environments, workers) == (8, 2) else 45.0}

            return Measurement()

        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_steps(100),
            environment_counts=[4, 8],
            worker_counts=[1, 2, 4],
            finalists=2,
            runtime=self.runtime(),
            screen=screen,
            validate_training=validate_training,
            recommendation_project_root=tmp_path,
            on_progress=events.append,
        )
        assert report["status"] == "completed"
        stages = [stage["name"] for stage in report["stages"]]
        assert stages == ["discovery", "screening", "devices", "validation"]

        screening = report["stages"][1]
        assert screening["status"] == "completed"
        # [4, 8] environments x [1, 2, 4] workers = 6 valid pairs.
        assert len(screening["configurations"]) == 6
        assert all(
            row["status"] == "ok" and isinstance(row["steps_per_second"], float)
            for row in screening["configurations"]
        )

        # Device stage: single CPU candidate, no comparison needed.
        assert report["stages"][2]["status"] == "skipped"

        validation = report["stages"][3]
        assert validation["status"] == "completed"
        assert len(validation["configurations"]) == 2

        recommendation = report["recommendation"]
        assert recommendation["basis"] == "validated_training_slice"
        assert (recommendation["environment_count"], recommendation["env_workers"]) == (8, 2)

        # Progress events cover every stage and every configuration.
        started = [e for e in events if e["status"] == "started"]
        assert {e["stage"] for e in started} >= {"discovery", "screening", "validation"}
        screen_events = [e for e in events if e["stage"] == "screening"]
        assert any(e.get("index") == 0 and e.get("total") == 6 for e in screen_events)

        # Report and recommendation were persisted next to the project.
        assert Path(report["report_path"]).is_file()
        persisted = load_recommendation(tmp_path)
        assert persisted["environment_count"] == 8
        assert persisted["source_report"] == report["report_path"]

    def test_failed_screening_rows_are_data_not_crashes(self, tmp_path):
        def screen(environments, workers):
            if workers > 1:
                raise RuntimeError("worker refused to start")
            return [screen_row(environments, workers, 100.0)]

        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_steps(50),
            environment_counts=[4],
            worker_counts=[1, 2],
            runtime=self.runtime(),
            screen=screen,
            save=False,
        )
        assert report["status"] == "completed"
        rows = report["stages"][1]["configurations"]
        assert rows[1]["status"] == "failed"
        assert "worker refused" in rows[1]["error"]
        # The recommendation warns about the failed configuration.
        assert any("failed screening" in w for w in report["recommendation"]["warnings"])

    def test_no_godot_means_an_honest_unavailable_report(self, tmp_path):
        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_steps(50),
            runtime=self.runtime(godot_available=False),
            save=False,
        )
        assert report["status"] == "unavailable"
        assert report["recommendation"] is None
        assert "no usable Godot runtime" in report["recommendation_reason"]
        assert load_recommendation(tmp_path) is None

    def test_cancellation_between_configurations(self, tmp_path):
        calls = {"n": 0}

        def screen(environments, workers):
            calls["n"] += 1
            return [screen_row(environments, workers, 100.0)]

        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_steps(50),
            environment_counts=[4, 8],
            worker_counts=[1, 2],
            runtime=self.runtime(torch_available=False),
            screen=screen,
            cancel=lambda: calls["n"] >= 1,
            save=False,
        )
        assert report["status"] == "cancelled"
        assert calls["n"] == 1
        assert report["stages"][1]["status"] == "cancelled"
        assert report["recommendation"] is None
        assert report["recommendation_reason"]

    def test_missing_torch_skips_validation_honestly(self, tmp_path):
        def screen(environments, workers):
            return [screen_row(environments, workers, 100.0)]

        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_steps(50),
            environment_counts=[4],
            worker_counts=[1],
            runtime=self.runtime(torch_available=False),
            screen=screen,
            save=False,
        )
        validation = report["stages"][3]
        assert validation["status"] == "skipped"
        assert "torch" in validation["reason"]
        assert report["recommendation"]["basis"] == "screening_only"

    def test_time_mode_thins_the_grid_and_sizes_slices(self, tmp_path):
        def screen(environments, workers):
            return [screen_row(environments, workers, 100.0)]

        report = run_benchmark_pipeline(
            project_path=tmp_path,
            budget=PipelineBudget.for_time(1.0),
            environment_counts=[1, 2, 4, 8, 16],
            worker_counts=[1, 2, 4],
            runtime=self.runtime(torch_available=False),
            screen=screen,
            save=False,
        )
        assert report["status"] == "completed"
        plan = report["plan"]
        assert len(plan["candidates"]) < 15  # the full grid would be 15
        assert "screen_cap_seconds_per_config" in report["budget"]

    def test_estimate_validation_steps_sizes_from_measured_throughput(self):
        row = screen_row(8, 4, 1000.0)
        fast = estimate_validation_steps(row, 10.0)
        slow = estimate_validation_steps(screen_row(8, 4, 1.0), 10.0)
        assert fast > slow
        assert 500 <= fast <= 20_000
        # No throughput measured -> the default slice.
        assert estimate_validation_steps({"steps_per_second": 0.0}, 10.0) == 3000


class TestSimulatedBridgeEndToEnd:
    """The real screening measurer against the fake JSON-lines bridge."""

    def test_steps_budget_pipeline_measures_and_persists(self, tmp_path):
        from simulated_bridge import SimulatedBridgeExecutable, supported

        if not supported():  # pragma: no cover - POSIX only
            pytest.skip("simulated bridge requires POSIX shebang support")
        bridge = SimulatedBridgeExecutable()
        try:
            report = run_benchmark_pipeline(
                project_path=Path(__file__).resolve().parents[2],
                godot_executable=bridge.path,
                budget=PipelineBudget.for_steps(30, device_steps=10, validation_steps=10),
                environment_counts=[2],
                worker_counts=[1, 2],
                finalists=1,
                use_saved_profile=False,
                recommendation_project_root=tmp_path,
                output_dir=tmp_path / "pipelines" / "run",
            )
        finally:
            bridge.cleanup()
        assert report["status"] == "completed"
        rows = report["stages"][1]["configurations"]
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "ok"
            # Real measurement, not a placeholder.
            assert row["steps_per_second"] > 0.0
            assert row["startup_seconds"] >= 0.0
        recommendation = report["recommendation"]
        assert recommendation is not None
        assert recommendation["environment_count"] == 2
        assert recommendation["expected_steps_per_second"] > 0.0
        assert (tmp_path / "pipelines" / "run" / "pipeline.json").is_file()
        assert (tmp_path / "pipelines" / "run" / "benchmark.json").is_file()
        assert load_recommendation(tmp_path) is not None
