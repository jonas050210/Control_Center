"""Integration/regression coverage for the production platform additions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.actions import MouseBinner
from data_pipeline.catalog import build_catalog
from data_pipeline.input_listener import RawInputEvent
from data_pipeline.inspect import generate_inspection_report
from data_pipeline.recorder import SessionRecorder
from data_pipeline.sync import ActionSynchronizer
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry
from sandbox.actions import ACTION_DIMS, SandboxAction, coerce_sandbox_action
from sandbox.env import TacticalArenaEnv
from sandboxai.workflow import run_end_to_end


def test_full_action_contract_and_legacy_adapter() -> None:
    action = SandboxAction(
        move_x=-1,
        move_y=1,
        jump=1,
        crouch=1,
        sprint=1,
        reload=1,
        fire=1,
        ads=1,
        mouse_dx_bin=3,
        mouse_dy_bin=18,
    )
    encoded = action.to_array()
    assert tuple(encoded.shape) == (len(ACTION_DIMS),)
    assert coerce_sandbox_action(encoded) == action
    legacy = coerce_sandbox_action([0, 2, 1, 3, 18])
    assert legacy.move_x == -1 and legacy.move_y == 1 and legacy.fire == 1


def test_tactical_arena_is_deterministic_for_seed_and_actions() -> None:
    first = TacticalArenaEnv(seed=7, max_steps=50)
    second = TacticalArenaEnv(seed=9, max_steps=50)
    obs_a, info_a = first.reset(seed=1234)
    obs_b, info_b = second.reset(seed=1234)
    assert np.array_equal(obs_a, obs_b)
    actions = [
        SandboxAction(move_y=1, sprint=1, mouse_dx_bin=12).to_array(),
        SandboxAction(move_x=-1, ads=1, mouse_dy_bin=9).to_array(),
        SandboxAction(fire=1, ads=1).to_array(),
    ] * 5
    for action in actions:
        result_a = first.step(action)
        result_b = second.step(action)
        assert np.array_equal(result_a[0], result_b[0])
        assert result_a[1:] == result_b[1:]
    assert first.state_snapshot() == second.state_snapshot()


def test_weapon_reload_and_rich_info() -> None:
    env = TacticalArenaEnv(seed=2, max_steps=100)
    env.reset(seed=2)
    env.ammo = 1
    env.reserve_ammo = 10
    _, _, _, _, info = env.step(SandboxAction(reload=1).to_array())
    assert info["reloading"] is True
    for _ in range(env.RELOAD_STEPS):
        _, _, _, _, info = env.step(SandboxAction().to_array())
    assert info["reloading"] is False
    assert info["ammo"] == 11
    assert info["reserve_ammo"] == 0
    assert "reward_terms" in info and "action" in info


def test_mouse_bin_metadata_rejects_non_finite_or_unknown_configuration() -> None:
    valid_edges = MouseBinner().edges
    with pytest.raises(ValueError, match="finite"):
        MouseBinner(custom_edges=[float("-inf"), *valid_edges[1:]])
    with pytest.raises(ValueError, match="strategy"):
        MouseBinner(strategy="misspelled")


def test_synchronizer_preserves_event_arriving_ahead_of_frame() -> None:
    synchronizer = ActionSynchronizer()
    future = RawInputEvent(2.0, "key_down", {"key": "w"})
    first = synchronizer.create_sample(0, 1.0, 0.0, "0.jpg", [future])
    second = synchronizer.create_sample(1, 2.1, 1.0, "1.jpg", [])
    assert first.actions.move_y == 0
    assert second.actions.move_y == 1


def test_end_to_end_requires_explicit_synthetic_data_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Record manual gameplay"):
        run_end_to_end(
            data_dir=tmp_path / "datasets",
            checkpoint_dir=tmp_path / "checkpoints",
            logs_dir=tmp_path / "logs",
            quick=True,
        )


def test_catalog_visual_report_and_experiment_manifest(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    recorder = SessionRecorder(
        datasets,
        session_id="inspection",
        target_fps=20,
        frame_width=84,
        frame_height=84,
        is_mock=True,
    )
    recorder.record(max_duration=0.2)
    metadata_text = recorder.metadata_file.read_text(encoding="utf-8")
    assert "Infinity" not in metadata_text and "NaN" not in metadata_text
    catalog_file = tmp_path / "catalog.json"
    catalog = build_catalog(datasets, catalog_file)
    assert catalog["summary"]["valid_sessions"] == 1
    assert catalog_file.is_file()
    html_file = generate_inspection_report(recorder.session_dir, tmp_path / "inspection.html", 2)
    assert "Validation: PASSED" in html_file.read_text(encoding="utf-8")

    tracker = ExperimentTracker("test", {"seed": 1}, tmp_path / "experiments")
    tracker.log_metrics(1, {"reward": 2.5})
    artifact = tracker.add_artifact(catalog_file, "catalog")
    tracker.finish({"ok": True})
    manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert artifact["sha256"] and manifest["artifacts"][0]["role"] == "catalog"


def test_telemetry_instances_merge_component_updates(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    orchestrator = SystemTelemetry(state_file)
    trainer = SystemTelemetry(state_file)
    trainer.update_rl_status("ppo.zip", 512, 123.0)
    orchestrator.update_stage("IDLE")
    merged = SystemTelemetry(state_file).state
    assert merged["rl"]["timesteps"] == 512
    assert merged["pipeline_stage"] == "IDLE"


def test_tactical_arena_state_snapshot_restores_rng_and_entities():
    env = TacticalArenaEnv(seed=17, max_steps=20)
    env.reset(seed=17)
    for _ in range(3):
        env.step(SandboxAction(move_y=1, fire=1).to_array())
    snapshot = env.state_snapshot()
    first = env.step(SandboxAction(move_x=1, mouse_dx_bin=12).to_array())
    env.restore_state(snapshot)
    second = env.step(SandboxAction(move_x=1, mouse_dx_bin=12).to_array())
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]
