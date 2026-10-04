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
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")

## Contact boxes: orange is "in view and clear" (the agent can shoot what it
## is looking at), green is "in view but behind cover" (it sees the contact,
## it just cannot hit all of it). The stroke is thin on purpose - the box
## frames the enemy, it does not paint over it.
const COL_IN_VIEW: Color = Color(1.0, 0.62, 0.10)
const COL_BLOCKED: Color = Color(0.60, 0.95, 0.20)
## A contact this small in pixels is a speck, not a target; keep it visible.
const CONTACT_BOX_MIN_SIZE: float = 6.0
## Building nodes mid-fight is wasted work, so a small pool is warmed up
## front; `_contact_box_node()` still grows it when the arena holds more.
const CONTACT_BOX_POOL: int = 4

var simulation_manager: SimulationManager
var focused_env_index: int = 0
var _label: Label
var _paused: bool = false
var _pause_button: Button
var _crosshair_nodes: Array = []
var _contact_box_nodes: Array = []


## A hollow rectangle drawn around one enemy. `ColorRect` can only fill, and a
## filled rectangle would hide the very thing the box is meant to point at, so
## this is a stroke-only Control that is moved and coloured every frame.
class ContactBox:
	extends Control
	## Lives on the class, not the overlay: a GDScript inner class is compiled
	## on its own and must not depend on the enclosing script's constants.
	const STROKE_WIDTH: float = 2.0

	var box_color: Color = Color(1.0, 0.62, 0.1, 0.9)

	func show_at(p_rect: Rect2, p_color: Color) -> void:
		box_color = p_color
		position = p_rect.position
		size = p_rect.size
		visible = true
		queue_redraw()

	func _draw() -> void:
		var inset: float = STROKE_WIDTH * 0.5
		var rect := Rect2(Vector2(inset, inset), size - Vector2(STROKE_WIDTH, STROKE_WIDTH))
		if rect.size.x <= 0.0 or rect.size.y <= 0.0:
			return
		draw_rect(rect, box_color, false, STROKE_WIDTH)


func setup(p_simulation_manager: SimulationManager, p_focused_env_index: int = 0) -> void:
	simulation_manager = p_simulation_manager
	focused_env_index = p_focused_env_index
	_build_label()
	_build_controls()
	_build_crosshair()
	# Warm the pool so the first frame that sees an enemy does not also have
	# to build nodes for it.
	for index in range(CONTACT_BOX_POOL):
		_contact_box_node(index)
	_show_focused_environment()


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
	# `build()` throws the old views away, so the camera that was on screen is
	# gone with them: re-point it at the focused environment.
	_show_focused_environment()


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
	_show_focused_environment()


## Puts the focused environment's own camera on screen. The text panel, the
## contact boxes and the picture have to be the same environment: the focus
## buttons used to move the numbers only, which was merely confusing before
## and became plainly wrong once rectangles were drawn over the view - the
## boxes were computed for one environment and painted onto another's image.
##
## Guarded, because only the graphical scene has views: the headless bridge
## never creates this overlay, and a manager built with `create_visuals =
## false` leaves `null` in the slot.
func _show_focused_environment() -> void:
	var camera: Camera3D = _focused_camera()
	if camera != null and not camera.is_current():
		camera.make_current()


## The camera of the focused environment, or `null` when it has none.
func _focused_camera() -> Camera3D:
	if simulation_manager == null:
		return null
	var views: Array = simulation_manager.views
	if views.is_empty() or focused_env_index >= views.size():
		return null
	var view = views[focused_env_index]
	if view == null or not is_instance_valid(view):
		return null
	return view.get_camera()


## One screen-space box per enemy the agent can actually see right now, in
## the engine's own normalised device coordinates (-1 left .. +1 right, -1
## bottom .. +1 top).
##
## Two ways in, one rule: an enemy is drawn when the agent could see it, never
## because it happens to be alive.
##
## 1. The perception levels (6-10) build a belief per enemy every tick, and
##    the observation vector is built from those beliefs - so this reads the
##    very same `screen_box` the policy reads. A box here and a box in the
##    Stats page therefore cannot disagree.
## 2. The legacy levels (1-4) run without a perception layer at all: the
##    observation there is ground truth and carries no boxes, so asking the
##    beliefs would show nothing at all. For those the geometry is evaluated
##    directly through the agent's own cone and reach.
##
## Either way "in view" is the vision system's answer - in the field of view,
## within reach, with a clear line - not a distance or an angle guessed here,
## and a contact that is merely remembered (seen a second ago, behind cover
## now) gets no box. That is the whole point of the overlay: it shows what
## the agent can act on, which is not the same as what is standing in the
## arena.
##
## Pure: no node or viewport access, so it can be unit tested headless (see
## tests/test_debug_overlay.gd) and reused by any other caller.
static func contact_boxes(p_env) -> Array:
	var boxes: Array = []
	if p_env == null or p_env.agent == null or not p_env.agent.alive:
		return boxes
	var agent = p_env.agent
	var eye: Vector3 = agent.position + Vector3(0.0, agent.eye_height, 0.0)
	var forward: Vector3 = agent.get_forward_vector()
	var beliefs: Array = p_env.get_beliefs()
	if beliefs.is_empty():
		return _geometric_contact_boxes(p_env, eye, forward)
	var slot_of_enemy: Dictionary = {}
	for index in range(p_env.enemies.size()):
		slot_of_enemy[int(p_env.enemies[index].enemy_id)] = index
	for belief_value in beliefs:
		var belief: Dictionary = belief_value
		if not bool(belief.get("visible", false)):
			continue
		var box: Dictionary = belief.get("screen_box", {})
		if not bool(box.get("in_front", false)):
			continue
		(
			boxes
			. append(
				{
					"index": int(slot_of_enemy.get(int(belief.get("id", -1)), -1)),
					"center_x": float(box.get("center_x", 0.0)),
					"center_y": float(box.get("center_y", 0.0)),
					"half_width": float(box.get("half_width", 0.0)),
					"half_height": float(box.get("half_height", 0.0)),
					"depth": float(box.get("depth", 0.0)),
					"exposure": clampf(float(belief.get("exposure_fraction", 1.0)), 0.0, 1.0),
				}
			)
		)
	return boxes


## The legacy-levels path: no beliefs exist, so sight is evaluated from the
## agent's own cone and reach through the same `PerceptionSystem` the
## perception layer itself calls. Same filters (field of view, line of sight,
## reach) so the two paths mean the same thing.
static func _geometric_contact_boxes(p_env, eye: Vector3, forward: Vector3) -> Array:
	var boxes: Array = []
	var fov_deg: float = SandboxConfig.AGENT_FOV_DEG
	var vision_range: float = SandboxConfig.VISION_RANGE
	if p_env.perception != null:
		fov_deg = float(p_env.perception.fov_deg)
		vision_range = float(p_env.perception.vision_range)
	for index in range(p_env.enemies.size()):
		var enemy = p_env.enemies[index]
		if enemy == null or not enemy.alive:
			continue
		var evaluation: Dictionary = PerceptionSystem.evaluate_target(
			p_env.world,
			eye,
			p_env.agent.position,
			forward,
			enemy.position,
			enemy.height,
			fov_deg,
			vision_range
		)
		if not bool(evaluation.get("visible", false)):
			continue
		var box: Dictionary = PerceptionSystem.target_screen_box(
			eye, forward, enemy.position, enemy.height, enemy.radius, fov_deg
		)
		if not bool(box.get("in_front", false)):
			continue
		(
			boxes
			. append(
				{
					"index": index,
					"center_x": float(box.get("center_x", 0.0)),
					"center_y": float(box.get("center_y", 0.0)),
					"half_width": float(box.get("half_width", 0.0)),
					"half_height": float(box.get("half_height", 0.0)),
					"depth": float(box.get("depth", 0.0)),
					"exposure":
					PerceptionSystem.exposure_fraction(
						p_env.world, eye, enemy.position, enemy.height
					),
				}
			)
		)
	return boxes


func _process(_delta: float) -> void:
	if simulation_manager == null or _label == null:
		return
	if simulation_manager.environments.is_empty():
		return
	_label.text = "\n".join(
		format_lines(build_telemetry_dict(simulation_manager, focused_env_index))
	)
	_update_contact_boxes()


## Repositions the contact-box pool over the enemies the agent can see right
## now. Boxes are pure decoration: they read `contact_boxes()`, they never
## feed back into the simulation.
func _update_contact_boxes() -> void:
	var boxes: Array = []
	var camera: Camera3D = _focused_camera()
	# Boxes are screen-space: they only mean something over the picture the
	# focused environment's camera is drawing. If another camera is on screen
	# (or the focus has no view at all), drawing them would put one
	# environment's contacts on top of another's image - so they stay hidden.
	if camera != null and camera.is_current() and not simulation_manager.environments.is_empty():
		var idx: int = clampi(focused_env_index, 0, simulation_manager.environments.size() - 1)
		boxes = contact_boxes(simulation_manager.environments[idx])
	var viewport: Vector2 = get_viewport().get_visible_rect().size
	for index in range(boxes.size()):
		var box: Dictionary = boxes[index]
		var half := Vector2(
			maxf(float(box.get("half_width", 0.0)) * 0.5 * viewport.x, CONTACT_BOX_MIN_SIZE * 0.5),
			maxf(float(box.get("half_height", 0.0)) * 0.5 * viewport.y, CONTACT_BOX_MIN_SIZE * 0.5)
		)
		var center := Vector2(
			(float(box.get("center_x", 0.0)) * 0.5 + 0.5) * viewport.x,
			(0.5 - float(box.get("center_y", 0.0)) * 0.5) * viewport.y
		)
		var exposure: float = clampf(float(box.get("exposure", 1.0)), 0.0, 1.0)
		_contact_box_node(index).show_at(
			Rect2(center - half, half * 2.0), COL_IN_VIEW.lerp(COL_BLOCKED, 1.0 - exposure)
		)
	for index in range(boxes.size(), _contact_box_nodes.size()):
		_contact_box_nodes[index].visible = false


## Grows the contact-box pool on demand; the enemy count is user-controlled
## and has no fixed cap, so the pool is created lazily instead of sized to a
## guess.
func _contact_box_node(index: int) -> ContactBox:
	while _contact_box_nodes.size() <= index:
		var box := ContactBox.new()
		box.name = "ContactBox%d" % _contact_box_nodes.size()
		box.mouse_filter = Control.MOUSE_FILTER_IGNORE
		box.visible = false
		add_child(box)
		_contact_box_nodes.append(box)
	return _contact_box_nodes[index]


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
	# The same boxes the overlay draws and the observation vector carries, so
	# the text and the rectangles can never tell two different stories.
	telemetry["contact_boxes"] = contact_boxes(env)

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
	var contacts: Array = telemetry.get("contact_boxes", [])
	var contact_note: String = "in view: %d" % contacts.size()
	if contacts.size() > 0:
		var exposure_sum: float = 0.0
		for contact_value in contacts:
			var contact: Dictionary = contact_value
			exposure_sum += float(contact.get("exposure", 1.0))
		contact_note += "   exposure: %.2f" % (exposure_sum / float(contacts.size()))
	lines.append(contact_note)
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
