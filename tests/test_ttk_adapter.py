"""Tests for TTK Adapter and Game Window Isolation (M1)."""

from data_pipeline.ttk_adapter import GameSourceConfig, find_game_window_rect


def test_game_source_config_presets() -> None:
    presets = GameSourceConfig.PRESETS
    assert "ttk_testing" in presets
    assert "godot_sandbox" in presets
    assert "desktop" in presets

    assert "Roblox" in presets["ttk_testing"]["window_title_patterns"]
    assert "Godot" in presets["godot_sandbox"]["window_title_patterns"]


def test_find_game_window_rect_safe_call() -> None:
    # On Linux or CI environments, should safely return None without throwing errors
    rect = find_game_window_rect(["NonExistentGameWindow123"])
    assert rect is None or (isinstance(rect, tuple) and len(rect) == 4)
