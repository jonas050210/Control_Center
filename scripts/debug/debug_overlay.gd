## DebugOverlay
##
## Lightweight, presentation-only debug/testing interface for understanding
## what the AI is doing. Reads state from SimulationManager/EnvironmentCore
## and displays it; it never drives simulation logic itself (the few control
## buttons call existing public SimulationManager/EnvironmentCore methods,
## the same ones Python or any other caller could use).
##
## `build_telemetry_dict()` and `format_lines()` are pure functions (no Node
## / UI dependency) so the telemetry content itself is unit-testable without
## a live scene tree — see tests/test_debug_overlay.gd. This overlay is only
## ever instantiated from scripts/core/main.gd; the headless RL bridge
## (scripts/rl/rl_server.gd) never creates it, so it has zero effect on
## headless training.
class_name DebugOverlay
extends CanvasLayer

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

var simulation_manager: SimulationManager
var focused_env_index: int = 0
var _label: Label
var _paused: bool = false
var _pause_button: Button
var _crosshair_nodes: Array = []


func setup(p_simulation_manager: SimulationManager, p_focused_env_index: int = 0) -> void:
	simulation_manager = p_simulation_manager
	focused_env_index = p_focused_env_index
	_build_label()
	_build_controls()
	_build_crosshair()


func _build_label() -> void:
	_label = Label.new()
	_label.position = Vector2(12.0, 12.0)
	_label.add_theme_font_size_override("font_size", 15)
	_label.add_theme_color_override("font_color", Color(1.0, 1.0, 1.0))
	_label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.9))
	_label.add_theme_constant_override("shadow_offset_x", 1)
	_label.add_theme_constant_override("shadow_offset_y", 1)
	add_child(_label)


func _build_controls() -> void:
	var panel := VBoxContainer.new()
	panel.name = "DebugControls"
	panel.position = Vector2(12.0, 420.0)
	add_child(panel)

	var row1 := HBoxContainer.new()
	panel.add_child(row1)
	_pause_button = Button.new()
	_pause_button.text = "Pause"
	_pause_button.pressed.connect(_on_pause_pressed)
	row1.add_child(_pause_button)

	var reset_button := Button.new()
	reset_button.text = "Reset Episode"
	reset_button.pressed.connect(_on_reset_pressed)
	row1.add_child(reset_button)

	var row2 := HBoxContainer.new()
	panel.add_child(row2)
	row2.add_child(_make_button("Enemies -", _on_enemy_count_delta.bind(-1)))
	row2.add_child(_make_button("Enemies +", _on_enemy_count_delta.bind(1)))
	row2.add_child(_make_button("Level -", _on_curriculum_delta.bind(-1)))
	row2.add_child(_make_button("Level +", _on_curriculum_delta.bind(1)))

	var row3 := HBoxContainer.new()
	panel.add_child(row3)
	row3.add_child(_make_button("< Env", _on_focus_delta.bind(-1)))
	row3.add_child(_make_button("Env >", _on_focus_delta.bind(1)))


## Minimal true-aim reticle for the normal playable scene. The human's
## weapon fires from the first-person camera centre, so these tiny red
## rectangles mark the exact hitscan ray without adding a busy HUD.
func _build_crosshair() -> void:
	var definitions: Array = [
		{"pos": Vector2(-1.5, -1.5), "size": Vector2(3.0, 3.0), "alpha": 0.95},
		{"pos": Vector2(-10.0, -0.5), "size": Vector2(6.0, 1.0), "alpha": 0.55},
		{"pos": Vector2(4.0, -0.5), "size": Vector2(6.0, 1.0), "alpha": 0.55},
		{"pos": Vector2(-0.5, -10.0), "size": Vector2(1.0, 6.0), "alpha": 0.55},
		{"pos": Vector2(-0.5, 4.0), "size": Vector2(1.0, 6.0), "alpha": 0.55},
	]
	for definition_value in definitions:
		var definition: Dictionary = definition_value
		var rect := ColorRect.new()
		rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
		rect.anchor_left = 0.5
		rect.anchor_right = 0.5
		rect.anchor_top = 0.5
		rect.anchor_bottom = 0.5
		var offset: Vector2 = definition["pos"]
		var rect_size: Vector2 = definition["size"]
		rect.offset_left = offset.x
		rect.offset_top = offset.y
		rect.offset_right = rect.offset_left + rect_size.x
		rect.offset_bottom = rect.offset_top + rect_size.y
		rect.color = Color(1.0, 0.03, 0.02, float(definition["alpha"]))
		add_child(rect)
		_crosshair_nodes.append(rect)


func _make_button(text: String, callback: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.pressed.connect(callback)
	return button


func _on_pause_pressed() -> void:
	_paused = not _paused
	Engine.time_scale = 0.0 if _paused else 1.0
	if _pause_button != null:
		_pause_button.text = "Resume" if _paused else "Pause"


func _on_reset_pressed() -> void:
	if simulation_manager == null:
		return
	var idx: int = clampi(focused_env_index, 0, simulation_manager.environments.size() - 1)
	if idx >= 0 and idx < simulation_manager.environments.size():
		(simulation_manager.environments[idx] as EnvironmentCore).reset(-1)


func _on_enemy_count_delta(delta: int) -> void:
	if simulation_manager == null:
		return
	var new_count: int = maxi(1, simulation_manager.enemy_count_per_environment + delta)
	simulation_manager.build(simulation_manager.environment_count, new_count)


func _on_curriculum_delta(delta: int) -> void:
	if simulation_manager == null:
		return
	var new_level: int = clampi(
		simulation_manager.curriculum_level + delta,
		CurriculumConfig.Level.STATIONARY_TARGET,
		CurriculumConfig.Level.AGENT_VS_AGENT
	)
	simulation_manager.set_curriculum_level(new_level)


func _on_focus_delta(delta: int) -> void:
	if simulation_manager == null or simulation_manager.environments.is_empty():
		return
	var count: int = simulation_manager.environments.size()
	focused_env_index = ((focused_env_index + delta) % count + count) % count


func _process(_delta: float) -> void:
	if simulation_manager == null or _label == null:
		return
	if simulation_manager.environments.is_empty():
		return
	_label.text = "\n".join(
		format_lines(build_telemetry_dict(simulation_manager, focused_env_index))
	)


## Pure data-collection function: reads current state into a plain
## Dictionary. Contains zero UI/Node dependencies (besides Engine's static
## frame-rate query) so it can be unit tested and reused by other tooling
## (e.g. a future in-editor inspector or a JSON export) without touching
## Control nodes.
static func build_telemetry_dict(
	p_simulation_manager: SimulationManager, env_index: int
) -> Dictionary:
	var telemetry: Dictionary = {}
	if p_simulation_manager == null or p_simulation_manager.environments.is_empty():
		return telemetry

	var idx: int = clampi(env_index, 0, p_simulation_manager.environments.size() - 1)
	var env: EnvironmentCore = p_simulation_manager.environments[idx]

	telemetry["render_fps"] = Engine.get_frames_per_second()
	telemetry["sim_steps_per_second"] = p_simulation_manager.steps_per_second
	telemetry["time_scale"] = Engine.time_scale
	telemetry["active_environments"] = p_simulation_manager.environments.size()
	telemetry["focused_env_index"] = idx
	telemetry["seed"] = p_simulation_manager.base_seed
	telemetry["curriculum_level"] = p_simulation_manager.curriculum_level
	telemetry["curriculum_name"] = CurriculumConfig.level_name(
		p_simulation_manager.curriculum_level
	)

	telemetry["episode"] = env.episode.episode_count
	telemetry["timestep"] = env.episode.step_count

	telemetry["agent_health"] = env.agent.health
	telemetry["agent_max_health"] = env.agent.max_health
	telemetry["agent_position"] = env.agent.position
	telemetry["agent_yaw_deg"] = env.agent.yaw_deg
	telemetry["agent_pitch_deg"] = env.agent.pitch_deg
	telemetry["agent_forward"] = env.agent.get_forward_vector()
	telemetry["weapon_ready"] = env.agent.weapon.is_ready()

	var enemy_reports: Array = []
	for i in range(env.enemies.size()):
		var enemy: EnemyState = env.enemies[i]
		(
			enemy_reports
			. append(
				{
					"index": i,
					"position": enemy.position,
					"health": enemy.health,
					"max_health": enemy.max_health,
					"alive": enemy.alive,
					"ai_state": EnemyState.ai_state_name(enemy.ai_state),
				}
			)
		)
	telemetry["enemy_count"] = env.enemies.size()
	telemetry["enemies_alive"] = env.get_alive_enemy_count()
	telemetry["enemies"] = enemy_reports

	telemetry["shots_fired"] = env.episode.shots_fired
	telemetry["shots_hit"] = env.episode.shots_hit
	telemetry["accuracy"] = (
		float(env.episode.shots_hit) / float(env.episode.shots_fired)
		if env.episode.shots_fired > 0
		else 0.0
	)
	telemetry["kills"] = env.episode.kills
	telemetry["total_kills"] = env.episode.total_kills
	telemetry["deaths"] = env.episode.deaths
	telemetry["total_deaths"] = env.episode.total_deaths
	telemetry["episode_reward"] = env.episode.cumulative_reward
	telemetry["last_reward"] = env.episode.last_reward
	telemetry["reward_breakdown"] = env.episode.get_reward_breakdown()
	telemetry["survival_time_seconds"] = (
		float(env.episode.survival_steps) * SandboxConfig.SIMULATION_DT
	)

	var last_actions: Array = p_simulation_manager.get_last_actions()
	if idx < last_actions.size() and last_actions[idx] != null:
		telemetry["current_action"] = last_actions[idx].to_array()

	var obs = env.get_observations()
	if obs != null:
		telemetry["observation_summary"] = {
			"agent_health_norm": obs.agent_health_norm,
			"primary_enemy_distance_norm": obs.enemy_distance_norm,
			"primary_enemy_bearing_norm": obs.enemy_bearing_norm,
			"primary_enemy_alive": obs.enemy_alive,
			"alive_enemy_count_norm": obs.alive_enemy_count_norm,
			"in_combat": obs.in_combat,
		}

	return telemetry


## Turns a telemetry Dictionary (as returned by build_telemetry_dict) into
## printable lines for the on-screen label. Pure formatting, no side effects.
static func format_lines(telemetry: Dictionary) -> PackedStringArray:
	var lines: PackedStringArray = PackedStringArray()
	if telemetry.is_empty():
		lines.append("SandboxAI — debug overlay (no environments)")
		return lines

	lines.append("SandboxAI — debug overlay")
	lines.append(
		(
			"render fps: %d   sim steps/sec: %.1f   time scale: %.2fx"
			% [telemetry.render_fps, telemetry.sim_steps_per_second, telemetry.time_scale]
		)
	)
	lines.append(
		(
			"environments: %d   focused: %d   seed: %d"
			% [telemetry.active_environments, telemetry.focused_env_index, telemetry.seed]
		)
	)
	lines.append("curriculum: %d (%s)" % [telemetry.curriculum_level, telemetry.curriculum_name])
	lines.append("episode: %d   timestep: %d" % [telemetry.episode, telemetry.timestep])
	lines.append("")
	lines.append(
		(
			"agent hp: %.0f / %.0f   pos: %s"
			% [telemetry.agent_health, telemetry.agent_max_health, str(telemetry.agent_position)]
		)
	)
	lines.append(
		(
			"aim yaw/pitch: %.1f / %.1f deg   weapon ready: %s"
			% [telemetry.agent_yaw_deg, telemetry.agent_pitch_deg, str(telemetry.weapon_ready)]
		)
	)
	if telemetry.has("current_action"):
		lines.append(
			"current action [move,strafe,yaw,pitch,shoot]: %s" % str(telemetry.current_action)
		)
	lines.append("")
	lines.append("enemies alive: %d / %d" % [telemetry.enemies_alive, telemetry.enemy_count])
	for enemy_info in telemetry.enemies:
		lines.append(
			(
				"  #%d [%s] hp %.0f/%.0f pos %s"
				% [
					enemy_info.index,
					enemy_info.ai_state,
					enemy_info.health,
					enemy_info.max_health,
					str(enemy_info.position)
				]
			)
		)
	lines.append("")
	lines.append(
		(
			"shots %d   hits %d   accuracy %.1f%%"
			% [telemetry.shots_fired, telemetry.shots_hit, telemetry.accuracy * 100.0]
		)
	)
	lines.append(
		(
			"kills %d (total %d)   deaths %d (total %d)"
			% [telemetry.kills, telemetry.total_kills, telemetry.deaths, telemetry.total_deaths]
		)
	)
	lines.append(
		(
			"episode reward: %.2f   last reward: %.3f   survival: %.2fs"
			% [telemetry.episode_reward, telemetry.last_reward, telemetry.survival_time_seconds]
		)
	)
	return lines
