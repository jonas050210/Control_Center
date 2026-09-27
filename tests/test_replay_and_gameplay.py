from pathlib import Path

ROOT = Path(__file__).parents[1]
GODOT = ROOT / "sandbox" / "godot_project"


def test_replay_is_persisted_and_bridge_exposes_save_command():
    logger = (GODOT / "scripts" / "episode_logger.gd").read_text()
    bridge = (GODOT / "scripts" / "rl_bridge.gd").read_text()
    py = (ROOT / "sandbox" / "godot_env.py").read_text()
    assert "FileAccess.open" in logger
    assert 'command == "save_replay"' in bridge
    assert "def save_replay" in py


def test_flank_behavior_and_valid_spawn_are_wired():
    enemy = (GODOT / "scripts" / "enemy_behavior.gd").read_text()
    scenario = (GODOT / "scripts" / "scenario_director.gd").read_text()
    controller = (GODOT / "scripts" / "ai_controller.gd").read_text()
    assert '"flank"' in enemy
    assert 'enemy.behavior = "flank"' in scenario
    assert "is_valid_spawn" in controller


def test_damage_feedback_exists():
    player = (GODOT / "scripts" / "player.gd").read_text()
    assert "damage_feedback_timer" in player
    assert "Color(1.0, 0.35, 0.35)" in player
