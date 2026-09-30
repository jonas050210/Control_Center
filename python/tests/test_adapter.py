import json
from sandboxai.adapter import SandboxAIAdapter, ProcessManager
from sandboxai.config import TrainingConfig

def test_adapter_discovers_real_run_and_missing_artifacts(tmp_path):
    run = tmp_path / "runs" / "r1"; run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"total_training_steps": 10}))
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    report = adapter.list_runs()
    assert report["format"] == "sandboxai.run_index/v1"
    assert report["runs"][0]["status"]["state"] == "incomplete"

def test_config_translation_is_domain_config(tmp_path):
    adapter = SandboxAIAdapter(project_root=tmp_path, output_root=tmp_path / "training")
    cfg = TrainingConfig(total_training_steps=12, output_root=str(tmp_path / "training"), run_id="unit")
    run, command = adapter._managed_training(cfg)
    assert json.loads((run / "config.json").read_text())["total_training_steps"] == 12
    assert "--control-file" in command

def test_process_manager_captures(tmp_path):
    import sys
    manager = ProcessManager(); record = manager.start("test", [sys.executable, "-c", "print('hello')"], tmp_path, tmp_path)
    record.process.wait(timeout=3)
    import time; time.sleep(.05)
    assert manager.snapshot(record.id)["state"] == "finished"
    assert "hello" in manager.snapshot(record.id)["stdout"]
