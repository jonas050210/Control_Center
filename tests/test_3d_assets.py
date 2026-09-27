from pathlib import Path

ROOT = Path(__file__).parents[1] / "sandbox" / "godot_project"


def test_enemy_builds_real_3d_character_visuals():
    source = (ROOT / "scripts" / "enemy_behavior.gd").read_text()
    factory = (ROOT / "scripts" / "model_factory.gd").read_text()
    assert "capsule_mesh" in factory and "add_capsule" in source
    assert "sphere_mesh" in factory and "add_sphere" in source
    assert "StandardMaterial3D" in source and "material" in factory
    assert "_build_enemy_visual" in source
    assert "ChestPlate" in source and "Backpack" in source


def test_weapons_have_distinct_3d_viewmodel_geometry():
    source = (ROOT / "scripts" / "player.gd").read_text()
    factory = (ROOT / "scripts" / "model_factory.gd").read_text()
    assert "_configure_weapon_model" in source
    for weapon in ("pistol", "smg", "rifle", "shotgun", "marksman"):
        assert f'"{weapon}"' in source
    assert "box_mesh" in factory and "MODEL_FACTORY" in source
    assert "Barrel" in source and "Magazine" in source and "Sight" in source
    assert "add_cylinder" in factory and "add_box" in factory


def test_maps_use_meshes_and_collision_bodies():
    source = (ROOT / "scripts" / "arena_environment.gd").read_text()
    factory = (ROOT / "scripts" / "model_factory.gd").read_text()
    assert "MeshInstance3D" in factory and "add_box" in source
    assert "StaticBody3D" in source
    assert "CollisionShape3D" in source
    assert "NavigationRegion3D" in source
    assert "TopTrim" in source and "FrontPanel" in source and "SupportLeft" in source


def test_target_dummy_has_a_readable_training_target_model():
    source = (ROOT / "scripts" / "target_dummy.gd").read_text()
    assert "_build_visual" in source
    assert "TargetPlate" in source and "TargetSensor" in source
    assert "MODEL_FACTORY" in source
