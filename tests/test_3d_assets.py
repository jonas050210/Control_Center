from pathlib import Path

ROOT = Path(__file__).parents[1] / "sandbox" / "godot_project"


def test_enemy_builds_real_3d_character_visuals():
    source = (ROOT / "scripts" / "enemy_behavior.gd").read_text()
    assert "CapsuleMesh" in source
    assert "SphereMesh" in source
    assert "StandardMaterial3D" in source
    assert "_build_enemy_visual" in source


def test_weapons_have_distinct_3d_viewmodel_geometry():
    source = (ROOT / "scripts" / "player.gd").read_text()
    assert "_configure_weapon_model" in source
    for weapon in ("pistol", "smg", "rifle", "shotgun", "marksman"):
        assert f'"{weapon}"' in source
    assert "BoxMesh" in source


def test_maps_use_meshes_and_collision_bodies():
    source = (ROOT / "scripts" / "arena_environment.gd").read_text()
    assert "MeshInstance3D" in source
    assert "StaticBody3D" in source
    assert "CollisionShape3D" in source
    assert "NavigationRegion3D" in source
