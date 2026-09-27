from pathlib import Path

ROOT = Path(__file__).parents[1] / "sandbox" / "godot_project"


def test_enemy_archetypes_and_scenarios_are_present():
    enemy = (ROOT / "scripts" / "enemy_behavior.gd").read_text()
    scenarios = (ROOT / "scripts" / "scenario_director.gd").read_text()
    for archetype in ("stationary", "patrol", "aggressive", "cover"):
        assert archetype in enemy
    for scenario in ("doorway_ambush", "crossing_target", "cover_transition", "flank"):
        assert scenario in scenarios


def test_player_scene_has_collision_and_first_person_viewmodel():
    scene = (ROOT / "scenes" / "Player.tscn").read_text()
    assert 'type="CollisionShape3D"' in scene
    assert 'name="ViewModel"' in scene
    assert 'type="Camera3D"' in scene


def test_scenario_reset_is_seeded():
    source = (ROOT / "scripts" / "scenario_director.gd").read_text()
    assert "rng.seed = seed" in source
    assert "func configure" in source
