"""The trainer's own wall-clock budget, tested without stable-baselines3.

The budget used to be a window-side request: the Control Center watched the
elapsed time and asked the process to stop. It is now part of the training
loop (``TrainingConfig.max_train_minutes`` -> ``ppo._should_continue_step``),
so a run stays time-boxed even when no window is watching. The policy is
plain Python - no torch, no gymnasium, no engine - which is exactly why it
can be pinned here.
"""

from __future__ import annotations

import unittest
from typing import Any

from sandboxai import ppo
from sandboxai.config import TrainingConfig


class RecordingTelemetry:
    """Minimal stand-in for JsonlTelemetry that keeps the rows in memory."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, payload: dict[str, Any]) -> None:
        self.rows.append(payload)

    def events(self) -> list[str]:
        return [str(row.get("event")) for row in self.rows]


class RecordingRunControl:
    """Stand-in for RunControl: records instead of writing status files."""

    def __init__(self, *, stopping: bool = False) -> None:
        self.stopping = stopping
        self.checked = 0
        self.updates: list[dict[str, Any]] = []
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def checkpoint(self) -> bool:
        self.checked += 1
        return not self.stopping

    def update(self, **values: Any) -> None:
        self.updates.append(values)

    def event(self, kind: str, message: str, values: dict[str, Any] | None = None) -> None:
        self.events.append((kind, message, dict(values or {})))


def _state() -> ppo._SelectionState:
    return ppo._SelectionState(best_score=0.0)


class TimeBudgetTests(unittest.TestCase):
    def test_zero_minutes_disables_the_budget(self) -> None:
        budget = ppo._TimeBudget(TrainingConfig().max_train_minutes)
        self.assertFalse(budget.enabled)
        self.assertFalse(budget.reached())

    def test_a_spent_budget_is_reached_and_restart_resets_it(self) -> None:
        budget = ppo._TimeBudget(10)
        self.assertEqual(budget.seconds, 600.0)
        budget.started -= 601.0
        self.assertTrue(budget.reached())
        # A resumed run gets a fresh budget; the clock starts with its loop.
        budget.restart()
        self.assertFalse(budget.reached())

    def test_a_spent_budget_stops_the_run_and_says_why(self) -> None:
        budget = ppo._TimeBudget(5)
        budget.started -= 301.0
        state = _state()
        telemetry = RecordingTelemetry()
        control = RecordingRunControl()

        keep_going = ppo._should_continue_step(
            budget, state, run_control=control, telemetry=telemetry, timesteps=123
        )

        self.assertFalse(keep_going)
        self.assertTrue(state.stop_training)
        self.assertEqual(state.stop_reason, "time_budget")
        self.assertEqual(telemetry.events(), ["time_budget_reached"])
        self.assertEqual(telemetry.rows[0]["timesteps"], 123)
        self.assertEqual(telemetry.rows[0]["budget_minutes"], 5.0)
        # The operator-facing status says "Stopping" while the final
        # checkpoint is written, exactly like an operator stop.
        self.assertEqual(control.updates[-1]["state"], "Stopping")
        self.assertIn("budget", control.events[-1][1])

    def test_an_unspent_budget_keeps_training(self) -> None:
        budget = ppo._TimeBudget(60)
        state = _state()
        telemetry = RecordingTelemetry()

        keep_going = ppo._should_continue_step(
            budget, state, run_control=None, telemetry=telemetry, timesteps=0
        )

        self.assertTrue(keep_going)
        self.assertFalse(state.stop_training)
        self.assertEqual(state.stop_reason, "")
        self.assertEqual(telemetry.rows, [])

    def test_an_operator_stop_wins_and_is_named(self) -> None:
        # The operator asked to stop; the wall-clock budget is irrelevant.
        budget = ppo._TimeBudget(0)
        control = RecordingRunControl(stopping=True)
        state = _state()

        keep_going = ppo._should_continue_step(
            budget, state, run_control=control, telemetry=RecordingTelemetry(), timesteps=0
        )

        self.assertFalse(keep_going)
        self.assertTrue(state.stop_training)
        self.assertEqual(state.stop_reason, "operator")
        self.assertEqual(control.checked, 1)

    def test_the_budget_never_renames_an_existing_stop(self) -> None:
        budget = ppo._TimeBudget(1)
        budget.started -= 61.0
        state = _state()
        state.stop_training = True
        state.stop_reason = "early_stopping"
        control = RecordingRunControl()

        keep_going = ppo._should_continue_step(
            budget, state, run_control=control, telemetry=RecordingTelemetry(), timesteps=0
        )

        # The run is already ending for a better-documented reason; the
        # budget must not overwrite it or announce a second stop.
        self.assertFalse(keep_going)
        self.assertTrue(state.stop_training)
        self.assertEqual(state.stop_reason, "early_stopping")
        self.assertEqual(control.events, [])


if __name__ == "__main__":
    unittest.main()
