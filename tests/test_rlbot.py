"""The real-game bridge: packet conversion must reproduce the training observation."""

import random
import tomllib

import numpy as np
import pytest

flat = pytest.importorskip("rlbot.flat")

from rlgym.rocket_league.common_values import BOOST_LOCATIONS  # noqa: E402
from rlgym.rocket_league.obs_builders import DefaultObs  # noqa: E402

from rocketai.config import OBS_PADDING  # noqa: E402
from rocketai.env import lookup_table, make_env  # noqa: E402
from rocketai.model import ActorCritic, save_checkpoint  # noqa: E402
from rocketai.play import PlaySettings, bot_toml, match_toml  # noqa: E402
from rocketai.rlbot_convert import PacketConverter, controls_from_action  # noqa: E402


def v3(a):
    return flat.Vector3(float(a[0]), float(a[1]), float(a[2]))


def physics(obj):
    pitch, yaw, roll = (float(x) for x in obj.euler_angles)
    return flat.Physics(
        location=v3(obj.position),
        rotation=flat.Rotator(pitch, yaw, roll),
        velocity=v3(obj.linear_velocity),
        angular_velocity=v3(obj.angular_velocity),
    )


def packet_from_state(state, pad_order, active=None):
    players = []
    for car in state.cars.values():
        players.append(
            flat.PlayerInfo(
                physics=physics(car.physics),
                air_state=flat.AirState.OnGround if car.on_ground else flat.AirState.InAir,
                dodge_timeout=-1.0,
                demolished_timeout=-1.0,
                is_supersonic=car.is_supersonic,
                team=car.team_num,
                boost=float(car.boost_amount),
                last_input=flat.ControllerState(),
                has_jumped=car.has_jumped,
                has_double_jumped=car.has_double_jumped,
                has_dodged=car.has_flipped,
            )
        )
    active = active or {}
    pads = [
        flat.BoostPadState(
            is_active=active.get(i, (True, 0.0))[0], timer=active.get(i, (True, 0.0))[1]
        )
        for i in pad_order
    ]
    return flat.GamePacket(
        players=players,
        balls=[flat.BallInfo(physics=physics(state.ball))],
        boost_pads=pads,
        match_info=flat.MatchInfo(frame_num=state.tick_count),
    )


def field_pads(order):
    return [
        flat.BoostPad(location=v3(BOOST_LOCATIONS[i]), is_full_boost=BOOST_LOCATIONS[i][2] > 72)
        for i in order
    ]


@pytest.mark.parametrize("team_size", [1, 2])
def test_kickoff_observation_is_identical(team_size):
    env = make_env(team_size=team_size, kickoff_probability=1.0, seed=4)
    env.reset()
    state = env.state
    order = list(range(len(BOOST_LOCATIONS)))
    random.Random(0).shuffle(order)  # RLBot's pad order differs from RLGym's
    converter = PacketConverter(field_pads(order))
    converted = converter.convert(packet_from_state(state, order))
    builder = DefaultObs(zero_padding=OBS_PADDING)
    for index, agent in enumerate(state.cars):
        expected = builder.build_obs([agent], state, {})[agent]
        got = builder.build_obs([converter.agent_id(index)], converted, {})[
            converter.agent_id(index)
        ]
        assert np.allclose(expected, got, atol=1e-4), np.abs(expected - got).max()


def test_midgame_physics_and_rotation_match():
    env = make_env(seed=5)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step({a: np.array([rng.integers(90)]) for a in env.agents})
    state = env.state
    order = list(range(len(BOOST_LOCATIONS)))
    converted = PacketConverter(field_pads(order)).convert(packet_from_state(state, order))
    for sim_car, car in zip(state.cars.values(), converted.cars.values(), strict=True):
        assert np.allclose(sim_car.physics.rotation_mtx, car.physics.rotation_mtx, atol=1e-4)
        assert np.allclose(sim_car.physics.position, car.physics.position)
        assert car.on_ground == sim_car.on_ground
    assert np.allclose(state.ball.position, converted.ball.position)


def test_boost_pad_timers_follow_location_mapping():
    order = list(range(len(BOOST_LOCATIONS)))[::-1]
    converter = PacketConverter(field_pads(order))
    big = next(i for i, loc in enumerate(BOOST_LOCATIONS) if loc[2] > 72)
    small = next(i for i, loc in enumerate(BOOST_LOCATIONS) if loc[2] < 72)
    timers = converter.boost_timers(
        [
            flat.BoostPadState(
                is_active=i not in (big, small), timer=3.0 if i in (big, small) else 0.0
            )
            for i in order
        ]
    )
    assert timers[big] == pytest.approx(7.0)  # 10 s cooldown, picked up 3 s ago
    assert timers[small] == pytest.approx(1.0)  # 4 s cooldown
    assert np.count_nonzero(timers) == 2


def test_controls_from_every_action():
    for row in lookup_table():
        controls = flat.ControllerState(**controls_from_action(row))
        assert -1 <= controls.throttle <= 1 and isinstance(controls.jump, bool)


@pytest.mark.parametrize("mode", ["psyonix", "human", "bot", "self"])
def test_match_toml_is_accepted_by_rlbot(tmp_path, mode):
    from rlbot.config import load_match_config

    checkpoint = tmp_path / "c.pt"
    save_checkpoint(checkpoint, ActorCritic(hidden_sizes=[8]), steps=0, config={})
    other = tmp_path / "other" / "bot.toml"
    other.parent.mkdir()
    other.write_text(
        '[settings]\nname = "Other"\nagent_id = "x/other"\nrun_command = "other.exe"\n'
    )
    settings = PlaySettings(
        checkpoint=str(checkpoint), mode=mode, team_size=2, opponent_bot=str(other)
    )
    settings.validate()
    (tmp_path / "bot.toml").write_text(bot_toml(settings))
    (tmp_path / "match.toml").write_text(match_toml(settings))
    tomllib.loads(match_toml(settings))
    config = load_match_config(tmp_path / "match.toml")
    assert len(config.player_configurations) == 4
    ours = config.player_configurations[0].variety
    assert "rocketai" in ours.run_command and str(checkpoint) in ours.run_command
