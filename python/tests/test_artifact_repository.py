import json
from pathlib import Path

import pytest

from sandboxai.artifact_repository import ArtifactRepository, RetentionPolicy


def _run(root: Path, name: str, state: str = "Finished") -> Path:
    path = root / "runs" / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    return path


def test_retention_policy_rejects_destructive_zero_values():
    with pytest.raises(ValueError):
        RetentionPolicy(keep_newest_runs=0)


def test_cleanup_is_preview_only_until_explicitly_confirmed(tmp_path):
    old = _run(tmp_path, "2024-old")
    newest = _run(tmp_path, "2025-new")
    checkpoints = newest / "checkpoints"
    checkpoints.mkdir()
    for index in range(3):
        path = checkpoints / f"step_{index}.zip"
        path.write_bytes(b"checkpoint")
        path.touch()
    (checkpoints / "latest.zip").write_bytes(b"protected")

    repository = ArtifactRepository(tmp_path)
    policy = RetentionPolicy(keep_newest_runs=1, keep_checkpoints_per_run=1)
    preview = repository.cleanup(policy)
    assert str(old.resolve()) in preview["runs"]
    assert len(preview["checkpoints"]) == 2
    assert old.exists()
    assert (checkpoints / "latest.zip").exists()

    repository.cleanup(policy, confirm=True)
    assert not old.exists()
    assert newest.exists()
    assert (checkpoints / "latest.zip").exists()
    assert len(list(checkpoints.glob("step_*.zip"))) == 1


def test_active_and_incomplete_runs_are_never_cleanup_candidates(tmp_path):
    active = _run(tmp_path, "2023-active", "Running")
    incomplete = _run(tmp_path, "2024-incomplete", "Idle")
    _run(tmp_path, "2025-new")
    plan = ArtifactRepository(tmp_path).cleanup_plan(RetentionPolicy(keep_newest_runs=1))
    assert str(active.resolve()) not in plan["runs"]
    assert str(incomplete.resolve()) not in plan["runs"]
