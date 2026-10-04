import pytest

from rocketai.config import (
    OBS_BASE_SIZE,
    OBS_EXTRA_SIZE,
    OBS_SIZE,
    PRESETS,
    TrainConfig,
    preset_config,
    run_paths,
)
from rocketai.obs import obs_size


def test_obs_size_matches_padding():
    assert OBS_BASE_SIZE == 172  # RLGym DefaultObs, 3 Autos pro Team
    assert OBS_SIZE == OBS_BASE_SIZE + OBS_EXTRA_SIZE
    assert obs_size(extras=True) == OBS_SIZE
    assert obs_size(extras=False) == OBS_BASE_SIZE
    assert TrainConfig().obs_size == OBS_SIZE
    assert TrainConfig(obs_extras=False).obs_size == OBS_BASE_SIZE
    assert TrainConfig(obs_extras=True).teacher_labels is False  # Lehrer ist standardmäßig aus
    assert TrainConfig(teacher_weight=1.0).teacher_labels is True


def test_teacher_without_start_weight_is_rejected():
    with pytest.raises(ValueError):
        TrainConfig(teacher_weight=0.0, teacher_final_weight=0.1).validate()


def test_hints_are_not_errors():
    config = TrainConfig(n_workers=9999, envs_per_worker=32)
    config.validate()  # darf nicht werfen
    assert config.hints()


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
