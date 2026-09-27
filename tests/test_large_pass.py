from pathlib import Path

ROOT = Path(__file__).parents[1]
GODOT = ROOT / "sandbox" / "godot_project"


def test_generated_geometry_is_collision_backed_and_navigation_enabled():
    source = (GODOT / "scripts" / "arena_environment.gd").read_text()
    assert "StaticBody3D" in source
    assert "CollisionShape3D" in source
    assert "NavigationRegion3D" in source
    assert "is_valid_spawn" in source


def test_bridge_supports_requested_episode_configuration_and_replay():
    bridge = (GODOT / "scripts" / "rl_bridge.gd").read_text()
    python = (ROOT / "sandbox" / "godot_env.py").read_text()
    assert 'command == "replay"' in bridge
    assert 'request.get("map"' in bridge
    assert 'request.get("scenario"' in bridge
    assert 'def replay' in python
    assert 'options["scenario"]' in python


def test_episode_logger_is_bounded_and_structured():
    source = (GODOT / "scripts" / "episode_logger.gd").read_text()
    assert 'events.size() > 256' in source
    assert '"frame"' in source
    assert '"time_ms"' in source
