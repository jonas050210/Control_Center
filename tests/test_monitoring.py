"""Tests for System Monitoring and Telemetry (Phase 7)."""

from pathlib import Path

import pytest

from monitoring.state import SystemTelemetry
from monitoring.status import display_dashboard


def test_system_telemetry_lifecycle(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    telemetry = SystemTelemetry(state_file=state_file)

    assert telemetry.state["pipeline_stage"] == "IDLE"
    assert "cpu_count" in telemetry.state["hardware"]

    # Update stage
    telemetry.update_stage("BC_TRAINING")
    assert telemetry.state["pipeline_stage"] == "BC_TRAINING"

    # Update BC status
    telemetry.update_bc_status(
        checkpoint="checkpoints/bc_best.pt",
        best_val_loss=1.234,
        metrics={"acc": 85.0},
    )
    assert telemetry.state["bc"]["best_val_loss"] == 1.234

    # Update RL status
    telemetry.update_rl_status(
        checkpoint="checkpoints/ppo.zip",
        timesteps=10000,
        steps_per_sec=450.0,
    )
    assert telemetry.state["rl"]["timesteps"] == 10000

    # Ensure file exists and can be reloaded
    telemetry.save()
    assert state_file.exists()

    reloaded = SystemTelemetry(state_file=state_file)
    assert reloaded.state["pipeline_stage"] == "BC_TRAINING"
    assert reloaded.state["bc"]["best_val_loss"] == 1.234


def test_dashboard_display_smoke(tmp_path: Path) -> None:
    state_file = tmp_path / "state_dash.json"
    display_dashboard(state_file=str(state_file))
