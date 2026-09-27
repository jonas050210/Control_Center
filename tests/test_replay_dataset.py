import json
from pathlib import Path
from data_pipeline.replay_dataset import convert_replay_file, export_bc_session


def test_replay_converts_to_bc_session(tmp_path: Path):
    source = tmp_path / "episode.json"
    source.write_text(json.dumps({"events": [{"type": "action", "frame": 1, "action": {"fire": 1}}]}))
    out = tmp_path / "session"
    assert export_bc_session(source, out) == 1
    assert (out / "metadata.json").exists()
    assert '"fire":1' in (out / "samples.jsonl").read_text()


def test_replay_jsonl_conversion_ignores_non_training_events(tmp_path: Path):
    source = tmp_path / "episode.json"
    source.write_text(json.dumps({"events": [{"type": "perception"}, {"type": "step", "action": [1]}]}))
    destination = tmp_path / "out.jsonl"
    assert convert_replay_file(source, destination) == 1
