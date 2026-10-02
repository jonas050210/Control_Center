import json
import sys
import time
from pathlib import Path

from sandboxai.adapter import ProcessManager, SandboxAIAdapter
from sandboxai.config import TrainingConfig
from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.telemetry import JsonlTelemetry


def test_adapter_discovers_real_run_and_missing_artifacts(tmp_path):
    run = tmp_path / "training" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"total_training_steps": 10}))
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    report = adapter.list_runs()
    assert report["format"] == "sandboxai.run_index/v1"
    assert report["runs"][0]["status"]["state"] == "incomplete"


def test_config_translation_is_domain_config(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    cfg = TrainingConfig(
        total_training_steps=12, output_root=str(tmp_path / "training"), run_id="unit"
    )
    run, command = adapter._managed_training(cfg)
    assert json.loads((run / "config.json").read_text())["total_training_steps"] == 12
    assert "--control-file" in command


def test_process_manager_captures(tmp_path):
    manager = ProcessManager()
    record = manager.start("test", [sys.executable, "-c", "print('hello')"], tmp_path, tmp_path)
    record.process.wait(timeout=3)
    time.sleep(0.05)
    assert manager.snapshot(record.id)["state"] == "finished"
    assert "hello" in manager.snapshot(record.id)["stdout"]
    assert isinstance(manager.snapshot(record.id)["pid"], int)


def test_failed_process_error_includes_the_last_stderr_line(tmp_path):
    """ "process exited with code 1" alone is useless; the snapshot appends
    the process's own last words so the Agents table can show the cause."""
    manager = ProcessManager()
    record = manager.start(
        "test",
        [
            sys.executable,
            "-c",
            "import sys; print('boom: engine missing', file=sys.stderr); sys.exit(1)",
        ],
        tmp_path,
        tmp_path,
    )
    record.process.wait(timeout=5)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(record.id)
        if snapshot.get("returncode") is not None and snapshot.get("stderr"):
            break
        time.sleep(0.02)
    snapshot = manager.snapshot(record.id)
    assert snapshot["state"] == "failed"
    assert snapshot["error"].startswith("process exited with code 1")
    assert "boom: engine missing" in snapshot["error"]
    manager.close()


def test_process_manager_log_since_is_incremental(tmp_path):
    manager = ProcessManager()
    record = manager.start(
        "test",
        [sys.executable, "-c", "print('one'); print('two'); print('three')"],
        tmp_path,
        tmp_path,
    )
    record.process.wait(timeout=3)
    time.sleep(0.1)
    first = manager.log_since(record.id)
    assert first["stdout"] == ["one", "two", "three"]
    assert not first["stdout_truncated"]
    # Polling again with the returned cursor must not re-deliver old lines.
    second = manager.log_since(record.id, stdout_after=first["stdout_cursor"])
    assert second["stdout"] == []


def test_process_manager_cancel_is_non_blocking_for_training(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manager = ProcessManager()
    # A long-lived placeholder process; cancel() for kind="training" must
    # only write command.json and return immediately, never touch the OS
    # process directly (that is the trainer's own cooperative shutdown).
    record = manager.start(
        "training", [sys.executable, "-c", "import time; time.sleep(5)"], run_dir, tmp_path
    )
    started = time.monotonic()
    result = manager.cancel(record.id)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert json.loads((run_dir / "command.json").read_text())["command"] == "stop"
    assert result["state"] == "running"
    manager.force_stop(record.id)
    record.process.wait(timeout=3)


def test_process_manager_force_stop_kills_non_training_process(tmp_path):
    manager = ProcessManager()
    record = manager.start(
        "benchmark", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path
    )
    manager.force_stop(record.id)
    record.process.wait(timeout=3)
    # `record.process.wait()` above only confirms the OS process exited; the
    # manager's own state field is set by its background reaper thread
    # (ProcessManager._wait), which is scheduled independently and can lag
    # behind by more than a few milliseconds under load (observed on
    # Windows CI runners). Poll briefly for that eventual update instead of
    # racing it - this changes nothing about what is asserted, only how
    # long the test is willing to wait for an already-true fact to become
    # visible.
    deadline = time.monotonic() + 3.0
    state = manager.snapshot(record.id)["state"]
    while state == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        state = manager.snapshot(record.id)["state"]
    assert state == "failed"


def test_finished_process_retention_is_bounded(tmp_path):
    manager = ProcessManager(finished_retention=2)
    for _ in range(5):
        record = manager.start("test", [sys.executable, "-c", "pass"], tmp_path, tmp_path)
        record.process.wait(timeout=3)
    time.sleep(0.1)
    with manager._lock:
        manager._prune_finished_locked()
        assert len(manager._records) <= 2


def _write_run(
    base,
    run_id,
    *,
    state="Running",
    timesteps=1000,
    target=10000,
    extra_status=None,
    extra_config=None,
    checkpoints=(),
    best_eval=None,
):
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True)
    config = TrainingConfig(
        total_training_steps=target,
        environment_count=4,
        env_workers=2,
        output_root=str(base),
        run_id=run_id,
    )
    config.save(run_dir / "config.json")
    status = {
        "state": state,
        "pid": 4242,
        "updated_at": time.time(),
        "timesteps": timesteps,
        "total_training_steps": target,
        "steps_per_second": 512.0,
        "eta_seconds": 30.0,
        "episodes": 12,
        "mean_episode_reward": 1.5,
        "win_rate": 0.6,
        "loss_rate": 0.3,
        "ppo_n_updates": 3,
        "ppo_approx_kl": 0.01,
        "device": "cpu",
    }
    if extra_status:
        status.update(extra_status)
    (run_dir / "status.json").write_text(json.dumps(status))
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    for name in checkpoints:
        (checkpoints_dir / name).write_bytes(b"0")
    if best_eval is not None:
        evaluations_dir = run_dir / "evaluations"
        evaluations_dir.mkdir(exist_ok=True)
        (evaluations_dir / "latest.json").write_text(json.dumps(best_eval))
    return run_dir


def test_dashboard_snapshot_reports_latest_run_and_active_process(tmp_path):
    output_root = tmp_path / "training"
    _write_run(output_root, "run-a", timesteps=100, target=1000)
    _write_run(output_root, "run-b", timesteps=500, target=1000, checkpoints=["latest.zip"])
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=output_root)
    snapshot = adapter.dashboard_snapshot()
    assert snapshot["run_count"] == 2
    assert snapshot["latest_run"]["run_id"] == "run-b"
    assert snapshot["latest_run"]["checkpoints"]["has_latest"] is True
    assert snapshot["active_processes"] == []

    from sandboxai.control_center_viewmodel import dashboard_view

    view = dashboard_view(snapshot)
    assert view["run_id"] == "run-b"
    assert view["timesteps"] == 500
    assert view["progress_percent"] == 50.0
    assert view["current_checkpoint"] == "checkpoints/latest.zip"
    # No managed process is tracking this run: a "Running" status this old
    # is not stale by default (updated_at is fresh in this fixture).
    assert view["stale"] is False


def test_dashboard_view_flags_stale_status(tmp_path):
    output_root = tmp_path / "training"
    _write_run(output_root, "run-a", extra_status={"updated_at": time.time() - 120})
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=output_root)
    snapshot = adapter.dashboard_snapshot()

    from sandboxai.control_center_viewmodel import dashboard_view

    view = dashboard_view(snapshot)
    assert view["stale"] is True
    assert any("crashed" in warning or "outside" in warning for warning in view["warnings"])


def test_discover_checkpoints_across_runs(tmp_path):
    output_root = tmp_path / "training"
    _write_run(output_root, "run-a", checkpoints=["latest.zip", "best_eval.zip"])
    _write_run(output_root, "run-b", checkpoints=["latest.zip"])
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=output_root)
    checkpoints = adapter.discover_checkpoints()
    kinds = {(entry["run_id"], entry["kind"]) for entry in checkpoints}
    assert ("run-a", "latest") in kinds
    assert ("run-a", "best") in kinds
    assert ("run-b", "latest") in kinds


def test_discover_and_detail_evaluations(tmp_path):
    output_root = tmp_path / "training"
    _write_run(
        output_root,
        "run-a",
        best_eval={
            "timesteps": 1000,
            "episodes": 20,
            "mean_episode_reward": 3.2,
            "win_rate": 0.7,
            "loss_rate": 0.2,
            "mean_kills": 1.1,
            "policy_shoot_request_rate": 0.4,
            "action_pipeline": {"fire_conversion_rate": 0.8, "localization": "ok"},
        },
    )
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=output_root)
    entries = adapter.discover_evaluations()
    assert len(entries) == 1
    assert entries[0]["win_rate"] == 0.7
    detail = adapter.evaluation_detail(entries[0]["path"])
    assert detail["available"] is True
    assert detail["mean_episode_reward"] == 3.2

    from sandboxai.control_center_viewmodel import evaluation_view

    view = evaluation_view(detail)
    assert view["outcomes"]["win_rate"] == 0.7
    assert view["action_head_diagnostics"]["discharge_rate"] == 0.8


def test_evaluation_detail_reports_missing_file_without_raising(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    detail = adapter.evaluation_detail(tmp_path / "nope.json")
    assert detail["available"] is False
    assert "error" in detail


def test_evaluation_detail_reports_corrupt_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    detail = adapter.evaluation_detail(bad)
    assert detail["available"] is False


def test_benchmark_history_and_results(tmp_path):
    directory = tmp_path / "training" / "benchmarks" / "sweep-1"
    directory.mkdir(parents=True)
    rows = [
        {
            "environments": 4,
            "workers": 1,
            "steps_per_second": 100.0,
            "total_steps": 2000,
            "episodes_per_second": 1.0,
            "vector_step_latency_p50_ms": 2.0,
            "vector_step_latency_p95_ms": 4.0,
            "elapsed_seconds": 20.0,
        },
        {
            "environments": 8,
            "workers": 1,
            "steps_per_second": 180.0,
            "total_steps": 2000,
            "episodes_per_second": 1.8,
            "vector_step_latency_p50_ms": 1.5,
            "vector_step_latency_p95_ms": 3.0,
            "elapsed_seconds": 11.0,
        },
    ]
    (directory / "benchmark.json").write_text(json.dumps(rows))
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    result = adapter.benchmark_results(directory)
    assert result["scaling"]["best_environment_count"] == 8
    history = adapter.benchmark_history()
    assert len(history) == 1
    assert history[0]["directory"] == str(directory)

    from sandboxai.control_center_viewmodel import benchmark_history_rows

    rows_view = benchmark_history_rows(history)
    assert len(rows_view) == 2
    assert {row["environments"] for row in rows_view} == {4, 8}


def test_benchmark_results_reports_missing_directory_without_raising(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    result = adapter.benchmark_results(tmp_path / "training" / "benchmarks" / "missing")
    assert result["results"] == []
    assert "error" in result


def test_telemetry_series_is_incremental_and_bounded(tmp_path):
    output_root = tmp_path / "training"
    run_dir = _write_run(output_root, "run-a")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=output_root)
    telemetry = JsonlTelemetry(run_dir / "logs" / "training.jsonl")
    telemetry.write({"timesteps": 100, "steps_per_second": 500.0, "mean_episode_reward": 1.0})
    telemetry.write({"timesteps": 200, "steps_per_second": 510.0, "mean_episode_reward": 1.2})
    telemetry.close()

    first = adapter.telemetry_series(run_dir, max_points=10)
    assert first["available"] is True
    assert first["series"]["timesteps"] == [(100.0, 100.0), (200.0, 200.0)]
    assert first["latest"]["timesteps"] == 200

    with JsonlTelemetry(run_dir / "logs" / "training.jsonl") as telemetry:
        telemetry.write({"timesteps": 300, "steps_per_second": 520.0, "mean_episode_reward": 1.4})
    second = adapter.telemetry_series(run_dir, max_points=10)
    assert [point[0] for point in second["series"]["timesteps"]] == [100.0, 200.0, 300.0]

    # Bounded: only the most recent max_points survive.
    bounded = adapter.telemetry_series(run_dir, max_points=2)
    assert len(bounded["series"]["timesteps"]) <= 2


def test_telemetry_series_missing_run_reports_unavailable(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    result = adapter.telemetry_series(tmp_path / "training" / "runs" / "missing")
    assert result["available"] is False


def test_system_status_reports_real_dependency_and_godot_fields(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    status = adapter.system_status()
    assert status["python"] == sys.executable
    assert "godot_available" in status
    assert "godot_version" in status
    assert status["dependencies"]["numpy"] is True


def test_start_training_rejects_invalid_config(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    try:
        adapter.start_training({"total_training_steps": -1})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid TrainingConfig")


def test_start_benchmark_rejects_invalid_arguments(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    try:
        adapter.start_benchmark(environment_counts=[])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty environment_counts")


def test_start_evaluation_rejects_missing_checkpoint(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    try:
        adapter.start_evaluation(tmp_path / "does_not_exist.zip")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing checkpoint")


def test_list_processes_and_process_log_round_trip(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    record = adapter.processes.start(
        "test", [sys.executable, "-c", "print('a-line')"], tmp_path, tmp_path
    )
    record.process.wait(timeout=3)
    time.sleep(0.1)
    processes = adapter.list_processes()
    assert any(item["id"] == record.id for item in processes)
    log = adapter.process_log(record.id)
    assert "a-line" in log["stdout"]

    from sandboxai.control_center_viewmodel import process_table_rows

    rows = process_table_rows(processes)
    assert any(row["id"] == record.id and isinstance(row["pid"], int) for row in rows)


def test_hardware_candidates_and_profile_passthrough(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    candidates = adapter.hardware_candidates()
    assert candidates, "at least the CPU candidate must always be offered"
    assert all("label" in c and "description" in c for c in candidates)
    # No wizard has run in this fresh checkout view, so nothing is persisted.
    assert adapter.hardware_profile() is None


def test_run_hardware_wizard_falls_back_without_engine(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    progress: list[str] = []
    result = adapter.run_hardware_wizard(
        godot_executable="definitely-not-a-real-godot-binary",
        steps=10,
        on_progress=lambda m: progress.append(m["status"]),
        save=False,
    )
    assert result["fallback"] is True
    assert result["selected_device"] == "cpu"
    assert result["godot_available"] is False


def test_send_command_writes_the_cooperative_command_file(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = adapter.processes.start(
        "training", [sys.executable, "-c", "import time; time.sleep(5)"], run_dir, tmp_path
    )
    try:
        snapshot = adapter.processes.send_command(record.id, "pause")
        assert snapshot["state"] == "running"
        command = json.loads((run_dir / "command.json").read_text())
        assert command["command"] == "pause"
    finally:
        adapter.processes.force_stop(record.id)
        record.process.wait(timeout=3)


def test_send_command_guards(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    # Unknown command ids are rejected outright.
    try:
        adapter.processes.send_command("whatever", "explode")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unsupported command")

    # Unknown process ids report, not raise.
    missing = adapter.processes.send_command("nope", "pause")
    assert missing["error_code"] == "process_not_found"

    # Non-training kinds never speak the protocol.
    other = tmp_path / "bench"
    other.mkdir()
    benchmark = adapter.processes.start(
        "benchmark", [sys.executable, "-c", "import time; time.sleep(5)"], other, tmp_path
    )
    try:
        refused = adapter.processes.send_command(benchmark.id, "pause")
        assert refused["error_code"] == "unsupported_command"
        assert not (other / "command.json").exists()
    finally:
        adapter.processes.force_stop(benchmark.id)
        benchmark.process.wait(timeout=3)

    # Exited training processes cannot take commands.
    finished = adapter.processes.start(
        "training", [sys.executable, "-c", "print('done')"], tmp_path / "r2", tmp_path
    )
    finished.process.wait(timeout=3)
    time.sleep(0.05)
    exited = adapter.processes.send_command(finished.id, "stop")
    assert exited["error_code"] == "process_not_running"


def test_start_training_resume_reuses_the_checkpoint_run_directory(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    run_dir = tmp_path / "training" / "runs" / "original"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "ppo_4000_steps.zip"
    checkpoint.write_bytes(b"weights")

    config = TrainingConfig(
        total_training_steps=100, output_root=str(tmp_path / "training"), run_id="original"
    )
    snapshot = adapter.start_training(config, checkpoint=checkpoint)
    try:
        view = next(
            item for item in adapter.list_processes() if item["id"] == snapshot["process_id"]
        )
        assert view["run_dir"] == str(run_dir)
        assert "resume" in view["command"]
        assert str(checkpoint) in view["command"]
        assert view["meta"]["resumed_from"] == str(checkpoint)
        # The config the resume run uses is persisted in that same run dir.
        persisted = json.loads((run_dir / "config.json").read_text())
        assert persisted["total_training_steps"] == 100
    finally:
        adapter.close()


def test_adapter_wires_the_agent_manager_to_itself(tmp_path):
    from sandboxai.agents import AgentManager

    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    assert isinstance(adapter.agents, AgentManager)
    # The registry drives the adapter's own processes and launch surface.
    assert adapter.agents._processes is adapter.processes
    assert adapter.agents._launcher is adapter


def test_validate_runtime_configuration_shares_the_pipeline_verdict(tmp_path):
    import time as _time

    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    # Deterministic runtime facts: godot present, no CUDA, 8 usable cores.
    runtime = {"godot_available": True, "cuda_available": False, "cpu_count": 8}
    adapter._runtime_status = {None: (_time.monotonic(), runtime)}

    valid = adapter.validate_runtime_configuration(12, 4, "cpu")
    assert valid["valid"], valid["errors"]
    assert [shard["count"] for shard in valid["shards"]] == [3, 3, 3, 3]

    too_many_workers = adapter.validate_runtime_configuration(4, 8, "cpu")
    assert not too_many_workers["valid"]

    no_cuda = adapter.validate_runtime_configuration(8, 2, "cuda")
    assert not no_cuda["valid"]
    assert any("CUDA" in error for error in no_cuda["errors"])


def test_validate_runtime_configuration_reports_a_missing_godot_before_launch(tmp_path):
    import time as _time

    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    runtime = {
        "godot_available": False,
        "godot_executable": "godot",
        "cuda_available": False,
        "cpu_count": 8,
    }
    adapter._runtime_status = {None: (_time.monotonic(), runtime)}

    verdict = adapter.validate_runtime_configuration(8, 2, "cpu")
    assert not verdict["valid"]
    assert any("Godot executable" in error for error in verdict["errors"])

    # A form override is probed under its own cache key, not the default's.
    adapter._runtime_status["/custom/godot"] = (
        _time.monotonic(),
        {"godot_available": True, "cuda_available": False, "cpu_count": 8},
    )
    override = adapter.validate_runtime_configuration(8, 2, "cpu", godot_executable="/custom/godot")
    assert override["valid"], override["errors"]


def test_configure_godot_executable_only_persists_a_verified_executable(tmp_path, monkeypatch):
    import os
    import stat

    settings_path = tmp_path / ".sandboxai" / "settings.json"
    (tmp_path / "project.godot").write_text("", encoding="utf-8")
    monkeypatch.setattr("sandboxai.config._settings_path", lambda: settings_path)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    # Nothing given / nothing resolvable: refused, nothing written.
    assert not adapter.configure_godot_executable("")["ok"]
    missing = adapter.configure_godot_executable(str(tmp_path / "missing-binary"))
    assert not missing["ok"]
    assert not settings_path.exists()

    # A real executable that answers --version is verified and remembered.
    fake = tmp_path / "fake_godot.sh"
    fake.write_text("#!/bin/sh\necho 4.7.2.fake\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    if os.name == "nt":  # pragma: no cover - POSIX shell script
        import pytest

        pytest.skip("POSIX shell executable")
    result = adapter.configure_godot_executable(str(fake))
    assert result["ok"], result
    assert result["version"] == "4.7.2.fake"
    assert settings_path.is_file()
    assert adapter.godot_executable_setting() == str(fake)
    adapter.close()


# ---------------------------------------------------------------------------
# Stats page source: real replays on disk
# ---------------------------------------------------------------------------


def _write_replay(root, name="episode_0001.jsonl", *, detail="detailed", ticks=3, seed=4242):
    """Write a real (recorder-produced) replay under ``<root>/<run>/replays``."""
    from sandboxai.replay import ReplayHeader, ReplayRecorder

    run_dir = root / "runs" / "run-a" / "replays"
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = ReplayRecorder(
        header=ReplayHeader(
            seed=seed,
            map_id="blind_corner",
            scenario="corner_fight",
            lighting="low_light",
            enemy_count=2,
            curriculum_level=6,
            policy_id="unit-brain",
        ),
        detail=detail,
    )
    recorder.start(seed=seed)
    for index in range(ticks):
        observation = [0.0] * OBSERVATION_FIELD_COUNT
        observation[16] = 1.0  # in_combat
        observation[9] = 0.5  # agent_health_norm
        recorder.record_step(
            [index % 3, 1, 1, 1, index % 2, 0],
            0.25,
            observation if detail == "detailed" else None,
            done=index == ticks - 1,
        )
    recorder.save(run_dir / name)
    return run_dir / name


def test_list_replays_reads_headers_without_parsing_the_whole_file(tmp_path):
    path = _write_replay(tmp_path / "training")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    replays = adapter.list_replays()
    assert len(replays) == 1
    entry = replays[0]
    assert entry["path"] == str(path)
    assert entry["run"] == "run-a"
    assert entry["header"]["seed"] == 4242
    assert entry["header"]["detail"] == "detailed"
    assert entry["header"]["curriculum_level"] == 6
    assert entry["ticks"] == 3
    assert entry["size_bytes"] > 0
    assert "error" not in entry


def test_list_replays_reports_a_broken_header_instead_of_hiding_the_file(tmp_path):
    run_dir = tmp_path / "training" / "runs" / "run-b" / "replays"
    run_dir.mkdir(parents=True)
    (run_dir / "broken.jsonl").write_text("{not json}\n", encoding="utf-8")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    entries = {entry["name"]: entry for entry in adapter.list_replays()}
    assert "broken.jsonl" in entries
    assert entries["broken.jsonl"]["error"]
    assert "header" not in entries["broken.jsonl"]


def test_list_replays_without_an_output_root_is_empty_not_an_error(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "missing")
    assert adapter.list_replays() == []


def test_list_replays_does_not_re_read_an_unchanged_replay(tmp_path, monkeypatch):
    """A warm rescan must not open replay files again.

    The Stats page lists replays on a timer; opening every (or even the most
    recent 64) recording on every tick is the difference between a window
    that idles and one that keeps re-reading the disk. The cache is keyed by
    ``(mtime_ns, size)``, so an unchanged file is served from memory."""
    path = _write_replay(tmp_path / "training")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    assert adapter.list_replays()[0]["ticks"] == 3

    opened: list[Path] = []
    original_open = Path.open

    def counting_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    entries = adapter.list_replays()
    assert entries[0]["ticks"] == 3
    assert path not in opened, "a warm rescan re-read a replay that cannot have changed"


def test_list_replays_notices_a_replay_that_changed(tmp_path):
    """The stamp is (mtime_ns, size): an edited recording must not stay cached."""
    _write_replay(tmp_path / "training", ticks=3)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    assert adapter.list_replays()[0]["ticks"] == 3

    _write_replay(tmp_path / "training", ticks=5)
    assert adapter.list_replays()[0]["ticks"] == 5


def _rewrite_as_older_contract(path, observation_dim=84):
    """Turn a freshly recorded replay into one recorded under an older contract.

    The file format is unchanged - only the header's ``observation_dim`` and
    the stored observations shrink, exactly like a replay written before the
    vector grew. This is what ``--allow-contract-mismatch`` exists for.
    """
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(raw)
        if "header" in payload:
            payload["header"]["observation_dim"] = observation_dim
        if "tick" in payload:
            observation = payload["tick"].get("o")
            if observation is not None:
                payload["tick"]["o"] = observation[:observation_dim]
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_replay_stats_still_reads_an_older_contract_replay(tmp_path):
    """A recording is evidence: growing the vector must not hide old files."""
    path = _write_replay(tmp_path / "training", ticks=2)
    _rewrite_as_older_contract(path)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    result = adapter.replay_stats(path, tick=1)
    assert result["contract_match"] is False
    assert result["recorded_observation_dim"] == 84
    assert result["current_observation_dim"] == OBSERVATION_FIELD_COUNT
    assert len(result["observation"]) == 84, "the values are read as recorded, not padded"
    assert result["tick_index"] == 1
    # It stays a *readable* file, not a broken one.
    assert adapter.list_replays()[0].get("error") is None


def test_replay_stats_reports_a_current_contract_replay_as_matching(tmp_path):
    path = _write_replay(tmp_path / "training", ticks=2)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    result = adapter.replay_stats(path)
    assert result["contract_match"] is True
    assert result["recorded_observation_dim"] == OBSERVATION_FIELD_COUNT


def test_replay_stats_decodes_a_detailed_tick(tmp_path):
    from sandboxai.contract import OBSERVATION_FIELD_COUNT

    path = _write_replay(tmp_path / "training", ticks=4)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    result = adapter.replay_stats(path, tick=2)
    assert result["tick_count"] == 4
    assert result["detailed"] is True
    assert result["tick_index"] == 2
    assert result["action"] == [2, 1, 1, 1, 0, 0]
    assert result["reward"] == 0.25
    assert len(result["observation"]) == OBSERVATION_FIELD_COUNT
    assert result["observation"][9] == 0.5
    assert result["header"]["map_id"] == "blind_corner"
    # A tick beyond the end is clamped to the last one, never an exception.
    assert adapter.replay_stats(path, tick=99)["tick_index"] == 3
    assert adapter.replay_stats(path)["tick_index"] == 0


def test_replay_stats_reports_a_light_replay_without_inventing_a_vector(tmp_path):
    path = _write_replay(tmp_path / "training", detail="light")
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")

    result = adapter.replay_stats(path)
    assert result["detailed"] is False
    assert result["observation"] is None
    assert result["action"] == [0, 1, 1, 1, 0, 0]


def test_replay_stats_reparses_when_the_file_changes(tmp_path):
    """The decode cache is keyed by (size, mtime), so a rewrite is seen."""
    root = tmp_path / "training"
    path = _write_replay(root, ticks=2)
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=root)
    assert adapter.replay_stats(path)["tick_count"] == 2

    _write_replay(root, ticks=5)
    assert adapter.replay_stats(path)["tick_count"] == 5


def test_replay_stats_rejects_a_missing_file(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    try:
        adapter.replay_stats(tmp_path / "nope.jsonl")
    except FileNotFoundError:
        return
    raise AssertionError("a missing replay must raise FileNotFoundError")


def test_ttk_evidence_is_the_static_manifest(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    summary = adapter.ttk_evidence()
    assert summary["target"] == "Roblox TTK Testing"
    assert summary["mechanics"]["verified"]
    # No Roblox probing happens for the Stats page: an empty project has no
    # calibration file and the call still succeeds.
    assert adapter.ttk_calibration_notes() == {}
