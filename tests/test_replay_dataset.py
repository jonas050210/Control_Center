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


def test_replay_associates_action_with_same_frame_observation(tmp_path: Path):
    from data_pipeline.replay_dataset import replay_to_records
    import base64
    from io import BytesIO
    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (4, 4), color=(20, 30, 40)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    records = replay_to_records(
        {
            "events": [
                {"type": "action", "frame": 3, "action": [1, 1, 0, 0, 0, 0, 1, 0, 10, 10]},
                {"type": "observation", "frame": 3, "observation_png": encoded},
            ]
        },
        observation_dir=tmp_path / "frames",
    )
    assert len(records) == 1
    assert records[0]["frame"] == 3
    assert records[0]["action"]["fire"] == 1
    assert Path(tmp_path / "frames" / records[0]["observation_path"]).is_file()
