## DebugOverlay
##
## Minimal on-screen telemetry: render FPS, simulation steps/second, active
## environment count, and the focused environment's health / enemy count /
## kills / deaths / episode count / current reward / simulation speed.
## Intentionally just a single Label — this is a debugging aid, not product
## UI.
class_name DebugOverlay
extends CanvasLayer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")


var simulation_manager: SimulationManager
var focused_env_index: int = 0
var _label: Label


func setup(p_simulation_manager: SimulationManager, p_focused_env_index: int = 0) -> void:
	simulation_manager = p_simulation_manager
	focused_env_index = p_focused_env_index

	_label = Label.new()
	_label.position = Vector2(12.0, 12.0)
	_label.add_theme_font_size_override("font_size", 16)
	_label.add_theme_color_override("font_color", Color(1.0, 1.0, 1.0))
	_label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.9))
	_label.add_theme_constant_override("shadow_offset_x", 1)
	_label.add_theme_constant_override("shadow_offset_y", 1)
	add_child(_label)


func _process(_delta: float) -> void:
	if simulation_manager == null or _label == null:
		return
	if simulation_manager.environments.is_empty():
		return

	var idx: int = clampi(focused_env_index, 0, simulation_manager.environments.size() - 1)
	var env: EnvironmentCore = simulation_manager.environments[idx]

	var lines: PackedStringArray = PackedStringArray()
	lines.append("SandboxAI — debug overlay")
	lines.append("render fps: %d" % Engine.get_frames_per_second())
	lines.append("sim steps/sec: %.1f" % simulation_manager.steps_per_second)
	lines.append("simulation speed (time_scale): %.2fx" % Engine.time_scale)
	lines.append("active environments: %d" % simulation_manager.environments.size())
	lines.append("")
	lines.append("focused env: %d" % idx)
	lines.append("agent health: %.0f / %.0f" % [env.agent.health, env.agent.max_health])
	lines.append("enemies alive: %d / %d" % [env.get_alive_enemy_count(), env.enemies.size()])
	lines.append("kills: %d   deaths: %d" % [env.episode.total_kills, env.episode.total_deaths])
	lines.append("episode: %d   step: %d" % [env.episode.episode_count, env.episode.step_count])
	lines.append(
		(
			"current reward: %.3f   cumulative: %.2f"
			% [env.episode.last_reward, env.episode.cumulative_reward]
		)
	)

	_label.text = "\n".join(lines)
