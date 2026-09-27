"""Static regression coverage for the custom Godot tactical sandbox."""
from pathlib import Path

ROOT = Path(__file__).parents[1] / "sandbox" / "godot_project"


def test_environment_has_original_map_profiles_and_time_of_day():
    source = (ROOT / "scripts" / "arena_environment.gd").read_text()
    for name in ("facility", "research_complex", "compound"):
        assert name in source
    for name in ("day", "evening", "night"):
        assert name in source
    assert "ambient_light_energy" in source


def test_weapon_catalog_has_distinct_tactical_roles():
    source = (ROOT / "scripts" / "weapon_config.gd").read_text()
    for weapon in ("pistol", "smg", "rifle", "shotgun", "marksman"):
        assert f'"{weapon}"' in source
    assert "pellets" in source
    assert "effective_range" in source


def test_perception_is_camera_derived_and_not_coordinate_only():
    source = (ROOT / "scripts" / "ai_controller.gd").read_text()
    assert "is_position_in_frustum" in source
    assert "intersect_ray" in source
    for field in ("detected", "confidence", "screen_position", "screen_rect", "moving", "tracked"):
        assert f'"{field}"' in source
