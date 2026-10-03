"""Tests for :mod:`sandboxai.agents` — the agent lifecycle layer.

The lifecycle actions are exercised against *real* subprocesses driven
through the same ProcessManager the Control Center uses, with a small
fake trainer that speaks the real cooperative protocol (``command.json``
+ ``status.json``). No Tk, no Godot, no torch.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from sandboxai.adapter import ProcessManager
from sandboxai.agents import (
    LIFECYCLE_AVAILABLE,
    LIFECYCLE_FAILED,
    LIFECYCLE_FINISHED,
    LIFECYCLE_LAUNCHING,
    LIFECYCLE_PAUSED,
    LIFECYCLE_RESTARTING,
    LIFECYCLE_RUNNING,
    LIFECYCLE_STOPPED,
    LIFECYCLE_STOPPING,
    AgentManager,
    AgentRecord,
    derive_lifecycle,
)

FAKE_TRAINER = """
import json, sys, time
from pathlib import Path

run_dir = Path(sys.argv[1])
status_path = run_dir / "status.json"
command_path = run_dir / "command.json"

def publish(state):
    status_path.write_text(json.dumps({"kind": "training", "state": state}), encoding="utf-8")

publish("Starting")
publish("Running")
while True:
    try:
        command = json.loads(command_path.read_text(encoding="utf-8")).get("command")
    except (OSError, ValueError):
        command = None
    if command == "pause":
        publish("Paused")
    elif command == "resume":
        publish("Running")
    elif command == "stop":
        publish("Stopping")
        publish("Finished")
        sys.exit(0)
    time.sleep(0.01)
"""


class StubLauncher:
    """Speaks the AgentLauncher protocol with fake trainer processes."""

    def __init__(self, processes: ProcessManager, root: Path) -> None:
        self.processes = processes
        self.root = root
        self.training_calls: list[dict[str, object]] = []
        self.benchmark_calls: list[dict[str, object]] = []

    def start_training(self, config, *, checkpoint=None) -> dict[str, object]:
        run_dir = self.root / f"run{len(self.training_calls) + 1}"
        self.training_calls.append(
            {"config": config, "checkpoint": checkpoint, "run_dir": str(run_dir)}
        )
        record = self.processes.start(
            "training", [sys.executable, "-c", FAKE_TRAINER, str(run_dir)], run_dir, self.root
        )
        return {"process_id": record.id, "run_dir": str(run_dir)}

    def start_benchmark(self, **spec) -> dict[str, object]:
        self.benchmark_calls.append(spec)
        record = self.processes.start(
            "benchmark",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            self.root / "bench",
            self.root,
        )
        return {"process_id": record.id, "run_dir": str(self.root / "bench")}


@pytest.fixture()
def stack(tmp_path):
    processes = ProcessManager()
    launcher = StubLauncher(processes, tmp_path)
    manager = AgentManager(processes, launcher, restart_stop_timeout=5.0)
    yield processes, launcher, manager
    processes.close()


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not reached within {timeout}s")


def view_of(manager: AgentManager, agent_id: str) -> dict:
    views = {view["agent_id"]: view for view in manager.views()}
    return views[agent_id]


class TestDeriveLifecycle:
    RECORD = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")

    def test_unlaunched_states(self):
        assert derive_lifecycle(self.RECORD, None) == LIFECYCLE_AVAILABLE
        failed = AgentRecord(agent_id="a", kind="training", spec={}, error="boom")
        assert derive_lifecycle(failed, None) == LIFECYCLE_FAILED
        failed_launched = AgentRecord(
            agent_id="a", kind="training", spec={}, process_id="p", error="spawn failed"
        )
        assert derive_lifecycle(failed_launched, None) == LIFECYCLE_FAILED

    def test_launching_then_running(self):
        assert (
            derive_lifecycle(self.RECORD, {"state": "running", "backend": {}})
            == LIFECYCLE_LAUNCHING
        )
        assert (
            derive_lifecycle(self.RECORD, {"state": "running", "backend": {"state": "Starting"}})
            == LIFECYCLE_LAUNCHING
        )
        assert (
            derive_lifecycle(self.RECORD, {"state": "running", "backend": {"state": "Running"}})
            == LIFECYCLE_RUNNING
        )
        # A benchmark/evaluation process *is* its work: alive means running.
        benchmark = AgentRecord(agent_id="a", kind="benchmark", spec={})
        benchmark.process_id = "p"
        assert derive_lifecycle(benchmark, {"state": "running", "backend": {}}) == LIFECYCLE_RUNNING

    def test_pause_stop_and_restart_transitions(self):
        paused = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        assert (
            derive_lifecycle(paused, {"state": "running", "backend": {"state": "Paused"}})
            == LIFECYCLE_PAUSED
        )
        stopping = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        stopping.desired = "stop"
        assert (
            derive_lifecycle(stopping, {"state": "running", "backend": {"state": "Running"}})
            == LIFECYCLE_STOPPING
        )
        restarting = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        restarting.desired = "restart"
        assert (
            derive_lifecycle(restarting, {"state": "running", "backend": {"state": "Running"}})
            == LIFECYCLE_RESTARTING
        )

    def test_terminal_states(self):
        stopped = AgentRecord(
            agent_id="a", kind="training", spec={}, process_id="p", stop_requested_at=1.0
        )
        assert derive_lifecycle(stopped, {"state": "finished"}) == LIFECYCLE_STOPPED
        finished = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        assert derive_lifecycle(finished, {"state": "finished"}) == LIFECYCLE_FINISHED
        errored = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        assert (
            derive_lifecycle(errored, {"state": "finished", "backend": {"state": "Error"}})
            == LIFECYCLE_FAILED
        )
        assert derive_lifecycle(finished, {"state": "failed"}) == LIFECYCLE_FAILED


class TestLifecycleActions:
    def test_training_agent_runs_the_full_lifecycle(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10})
        agent_id = view["agent_id"]
        # The record keeps the spec the restart will need.
        record = manager.get(agent_id)
        assert record is not None
        assert record.spec["config"]["total_training_steps"] == 10
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)

        # Pause: cooperative command, backend confirms.
        result = manager.pause(agent_id)
        assert result["ok"], result
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_PAUSED)

        # Resume.
        result = manager.resume(agent_id)
        assert result["ok"], result
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)

        # Stop: requested while the trainer exits, then STOPPED.
        result = manager.stop(agent_id)
        assert result["ok"], result
        assert view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_STOPPING
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_STOPPED)

    def test_pause_is_refused_for_unsupported_kinds(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_benchmark(environment_counts=[4], steps=10)
        agent_id = view["agent_id"]
        result = manager.pause(agent_id)
        assert not result["ok"]
        assert "not supported" in result["error"]
        assert "training" in result["error"]
        result = manager.resume(agent_id)
        assert not result["ok"]
        manager.stop(agent_id)
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] != LIFECYCLE_RUNNING)

    def test_actions_on_missing_agent_fail_cleanly(self, stack):
        processes, launcher, manager = stack
        for action in (manager.pause, manager.resume, manager.stop, manager.remove):
            result = action("nope")
            assert not result["ok"]
            assert result["error"] == "agent not found"

    def test_failed_launch_records_the_error(self, stack):
        processes, launcher, manager = stack

        class BrokenLauncher(StubLauncher):
            def start_training(self, config, *, checkpoint=None):
                raise RuntimeError("no engine")

        manager_broken = AgentManager(
            processes, BrokenLauncher(processes, Path("/nonexistent")), restart_stop_timeout=5.0
        )
        view = manager_broken.launch_training({"total_training_steps": 10})
        assert view["lifecycle"] == LIFECYCLE_FAILED
        assert "no engine" in view["error"]
        # A failed launch is not a live process: it can be removed again.
        assert manager_broken.remove(view["agent_id"])["ok"]

    def test_backend_error_outranks_the_generic_process_exit_error(self, stack):
        """The view must surface the backend's published root cause, not the
        process layer's "exited with code N" symptom of the same failure."""
        processes, launcher, manager = stack
        record = AgentRecord(agent_id="a", kind="training", spec={}, process_id="p")
        view = manager._view(
            record,
            {
                "state": "failed",
                "error": "process exited with code 1",
                "backend": {"state": "Error", "error": "Could not launch Godot executable"},
            },
        )
        assert view["lifecycle"] == LIFECYCLE_FAILED
        assert "Could not launch Godot executable" in view["error"]
        # Without a backend error the process layer's account still shows.
        no_backend = manager._view(
            record, {"state": "failed", "error": "process exited with code 1", "backend": {}}
        )
        assert no_backend["error"] == "process exited with code 1"

    def test_restart_resumes_from_the_latest_checkpoint(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10})
        agent_id = view["agent_id"]
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)

        # Simulate the periodic mid-run checkpoints a stopped trainer leaves
        # behind (ppo.py only writes latest.zip on natural completion).
        run_dir = Path(str(launcher.training_calls[0]["run_dir"]))
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "ppo_1000_steps.zip").write_bytes(b"old")
        (checkpoints / "ppo_9000_steps.zip").write_bytes(b"new")
        (checkpoints / "ppo_2000_steps.zip").write_bytes(b"middle")

        result = manager.restart(agent_id)
        assert result["ok"], result
        assert len(launcher.training_calls) == 2
        resumed_from = launcher.training_calls[1]["checkpoint"]
        assert str(resumed_from).endswith("ppo_9000_steps.zip")

        # The same agent id, a brand-new process, running again.
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)
        assert view_of(manager, agent_id)["process_id"] != view["process_id"]
        manager.stop(agent_id)
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_STOPPED)

    def test_restart_without_checkpoint_starts_fresh(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10})
        agent_id = view["agent_id"]
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)
        result = manager.restart(agent_id)
        assert result["ok"], result
        assert launcher.training_calls[1]["checkpoint"] is None

    def test_stop_all_only_touches_live_agents(self, stack):
        processes, launcher, manager = stack
        first = manager.launch_training({"total_training_steps": 10})
        second = manager.launch_training({"total_training_steps": 10})
        wait_for(lambda: view_of(manager, first["agent_id"])["lifecycle"] == LIFECYCLE_RUNNING)
        wait_for(lambda: view_of(manager, second["agent_id"])["lifecycle"] == LIFECYCLE_RUNNING)
        # A finished agent must not produce a stop result.
        finished_id = manager.launch_benchmark(environment_counts=[1], steps=1)["agent_id"]
        # Use force_stop: benchmark processes have no cooperative protocol.
        manager.force_stop(finished_id)
        wait_for(lambda: view_of(manager, finished_id)["lifecycle"] != LIFECYCLE_RUNNING)

        results = manager.stop_all()
        assert {result["agent_id"] for result in results} == {
            first["agent_id"],
            second["agent_id"],
        }
        assert all(result["ok"] for result in results)
        wait_for(lambda: view_of(manager, first["agent_id"])["lifecycle"] == LIFECYCLE_STOPPED)

    def test_remove_and_clear_finished(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10})
        agent_id = view["agent_id"]
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_RUNNING)
        # Refused while live.
        assert not manager.remove(agent_id)["ok"]
        manager.stop(agent_id)
        wait_for(lambda: view_of(manager, agent_id)["lifecycle"] == LIFECYCLE_STOPPED)
        assert manager.remove(agent_id)["ok"]
        assert manager.get(agent_id) is None

        # clear_finished sweeps every exited agent.
        other = manager.launch_training({"total_training_steps": 10})
        wait_for(lambda: view_of(manager, other["agent_id"])["lifecycle"] == LIFECYCLE_RUNNING)
        manager.stop(other["agent_id"])
        wait_for(lambda: view_of(manager, other["agent_id"])["lifecycle"] == LIFECYCLE_STOPPED)
        assert manager.clear_finished() == 1
        assert manager.get(other["agent_id"]) is None

        # clear_finished also sweeps launch-failed records (process_id is None and error is set).
        failed = manager.launch_evaluation("/nonexistent/checkpoint.zip")
        assert failed["process_id"] is None
        assert failed["error"] is not None
        assert manager.clear_finished() == 1
        assert manager.get(failed["agent_id"]) is None

    def test_summary_counts_lifecycles(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10})
        wait_for(lambda: view_of(manager, view["agent_id"])["lifecycle"] == LIFECYCLE_RUNNING)
        assert manager.summary()[LIFECYCLE_RUNNING] == 1
        manager.stop(view["agent_id"])
        wait_for(lambda: manager.summary().get(LIFECYCLE_STOPPED) == 1)

    def test_views_expose_backend_facts_only(self, stack):
        processes, launcher, manager = stack
        view = manager.launch_training({"total_training_steps": 10}, name="unit")
        wait_for(lambda: view_of(manager, view["agent_id"])["lifecycle"] == LIFECYCLE_RUNNING)
        row = view_of(manager, view["agent_id"])
        assert row["name"] == "unit"
        assert row["kind"] == "training"
        assert isinstance(row["pid"], int)
        assert row["backend"]["state"] == "Running"
        assert row["spec"]["config"]["total_training_steps"] == 10


class TestLatestCheckpoint:
    def test_prefers_latest_zip(self, tmp_path):
        run = tmp_path / "run"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "latest.zip").write_bytes(b"x")
        (checkpoints / "ppo_9000_steps.zip").write_bytes(b"x")
        record = AgentRecord(agent_id="a", kind="training", spec={}, run_dir=str(run))
        assert AgentManager._latest_checkpoint(record).endswith("latest.zip")

    def test_picks_the_highest_periodic_checkpoint(self, tmp_path):
        run = tmp_path / "run"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        for name in ("ppo_1000_steps.zip", "ppo_9000_steps.zip", "ppo_2000_steps.zip"):
            (checkpoints / name).write_bytes(b"x")
        record = AgentRecord(agent_id="a", kind="training", spec={}, run_dir=str(run))
        assert AgentManager._latest_checkpoint(record).endswith("ppo_9000_steps.zip")

    def test_falls_back_to_best_eval_and_final(self, tmp_path):
        run = tmp_path / "run"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "best_eval.zip").write_bytes(b"x")
        record = AgentRecord(agent_id="a", kind="training", spec={}, run_dir=str(run))
        assert AgentManager._latest_checkpoint(record).endswith("best_eval.zip")
        (run / "final.zip").write_bytes(b"x")
        (checkpoints / "best_eval.zip").unlink()
        assert AgentManager._latest_checkpoint(record).endswith("final.zip")

    def test_no_checkpoint_at_all(self, tmp_path):
        record = AgentRecord(agent_id="a", kind="training", spec={}, run_dir=str(tmp_path))
        assert AgentManager._latest_checkpoint(record) is None
        assert AgentManager._latest_checkpoint(AgentRecord("a", "training", {})) is None


def test_fake_trainer_status_file_matches_run_control_states(stack):
    """The fake trainer publishes the exact states run_control.py defines."""
    processes, launcher, manager = stack
    view = manager.launch_training({"total_training_steps": 10})
    run_dir = Path(str(launcher.training_calls[0]["run_dir"]))
    wait_for(lambda: (run_dir / "status.json").exists())
    published = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert published["state"] in {"Starting", "Running", "Paused", "Stopping", "Finished"}
    manager.stop(view["agent_id"])
    wait_for(lambda: view_of(manager, view["agent_id"])["lifecycle"] == LIFECYCLE_STOPPED)
