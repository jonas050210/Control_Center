import pytest

from rocketai.config import OBS_SIZE, PRESETS, TrainConfig, preset_config, run_paths


def test_obs_size_matches_padding():
    assert OBS_SIZE == 172


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_are_valid(preset):
    preset_config(preset, name="x").validate()


def test_unknown_preset():
    with pytest.raises(ValueError):
        preset_config("nope")


def test_roundtrip_and_unknown_keys(tmp_path):
    config = TrainConfig(name="abc", team_size=2)
    config.save(tmp_path / "c.json")
    loaded = TrainConfig.load(tmp_path / "c.json")
    assert loaded == config
    assert TrainConfig.from_dict({"name": "abc", "future_option": 1}).name == "abc"


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "bad name!"},
        {"team_size": 4},
        {"reward_stage": 0},
        {"learning_rate": 2},
        {"hidden_sizes": []},
    ],
)
def test_validation_rejects(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes).validate()


def test_run_paths_reject_traversal():
    with pytest.raises(ValueError):
        run_paths("../evil")
    assert run_paths("ok-name").root.name == "ok-name"
