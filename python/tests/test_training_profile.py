"""Regression tests for aggregate training-profile accounting."""
from sandboxai.training_profile import TrainingProfiler


def test_sharded_bridge_phases_are_not_reported_as_zero():
    profiler = TrainingProfiler()
    profiler.record("worker0.bridge.step.wait_response", 1.0)
    profiler.record("worker1.bridge.step.wait_response", 2.0)
    profiler.record("worker0.bridge.step.json_encode", 0.1)
    profiler.record("worker1.bridge.step.json_encode", 0.2)

    phases = profiler.report()["phase_totals_seconds"]
    # Requests overlap, so the slowest worker is the critical path.
    assert phases["bridge.step.wait_response"] == 2.0
    # The facade encodes each request serially before dispatch completes.
    assert abs(phases["bridge.step.json_encode"] - 0.3) < 1e-12


def test_eval_prefix_is_preserved_while_worker_component_is_removed():
    profiler = TrainingProfiler()
    profiler.record("eval.normal.worker0.bridge.step.total", 0.4)
    profiler.record("eval.normal.worker1.bridge.step.total", 0.6)
    phases = profiler.report()["phase_totals_seconds"]
    assert phases["eval.normal.bridge.step.total"] == 0.6


def test_facade_wall_measurement_wins_over_worker_breakdown():
    profiler = TrainingProfiler()
    profiler.record("bridge.step.wait_response", 0.75)
    profiler.record("worker0.bridge.step.wait_response", 0.5)
    profiler.record("worker1.bridge.step.wait_response", 0.7)
    assert profiler.report()["phase_totals_seconds"]["bridge.step.wait_response"] == 0.75
