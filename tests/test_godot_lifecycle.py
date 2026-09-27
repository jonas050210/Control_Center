"""Regression checks for the tracked overlays applied to the FPS benchmark."""
from pathlib import Path

from feasibility import setup_examples


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_FPS = ROOT / "feasibility" / "godot_overlays" / "examples" / "FPS"


def test_projectile_teardown_is_deferred_and_idempotent():
    script = (OVERLAY_FPS / "projectile.gd").read_text(encoding="utf-8")

    assert "if _destroying:" in script
    assert 'set_deferred("monitoring", false)' in script
    assert 'set_deferred("monitorable", false)' in script
    assert 'call_deferred("_finish_destroy")' in script
    assert "func _on_area_shape_entered" not in script
    # No collision callback may directly tear an Area3D out of the tree.
    callbacks, _, teardown = script.partition("func _destroy()")
    assert "queue_free()" not in callbacks
    assert "queue_free()" in teardown


def test_projectile_teardown_always_frees_the_node():
    """Regression: the teardown must not depend on is_inside_tree().

    A projectile whose shooter was freed/respawned in the same frame is no
    longer inside the tree; the previous `if is_inside_tree(): queue_free()`
    guard skipped the free entirely and the orphaned Area3D survived until
    engine shutdown ("ObjectDB instances leaked at exit").
    """
    script = (OVERLAY_FPS / "projectile.gd").read_text(encoding="utf-8")
    _, _, finish = script.partition("func _finish_destroy()")

    assert finish, "_finish_destroy() must exist"
    assert "queue_free()" in finish
    assert "is_inside_tree()" not in finish
    assert "is_queued_for_deletion()" in finish


def test_projectile_script_base_class_matches_scene_root_type():
    """`monitoring`/`monitorable` only exist on Area3D.

    The script used to `extends Node3D`, so a set_deferred() on those
    properties was not statically checked and would silently become a no-op
    if the scene root type ever changed.
    """
    script = (OVERLAY_FPS / "projectile.gd").read_text(encoding="utf-8")
    scene = (OVERLAY_FPS / "projectile.tscn").read_text(encoding="utf-8")

    assert script.splitlines()[0].strip() == "extends Area3D"
    assert '[node name="Projectile" type="Area3D"]' in scene


def test_projectile_scene_has_one_handler_per_overlap_kind():
    scene = (OVERLAY_FPS / "projectile.tscn").read_text(encoding="utf-8")

    assert scene.count('signal="area_entered"') == 1
    assert scene.count('signal="body_entered"') == 1
    assert 'signal="area_shape_entered"' not in scene


def test_setup_applies_lifecycle_overlay(tmp_path, monkeypatch):
    destination = tmp_path / "examples"
    monkeypatch.setattr(setup_examples, "DEST", destination)

    setup_examples.apply_overlay()

    installed_script = destination / "examples" / "FPS" / "projectile.gd"
    installed_scene = destination / "examples" / "FPS" / "projectile.tscn"
    assert installed_script.read_bytes() == (OVERLAY_FPS / "projectile.gd").read_bytes()
    assert installed_scene.read_bytes() == (OVERLAY_FPS / "projectile.tscn").read_bytes()
