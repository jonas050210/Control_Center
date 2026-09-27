extends Node3D
class_name SandboxAIController

const PLAYER_SCRIPT: Script = preload("res://scripts/player.gd")
const TARGET_SCRIPT: Script = preload("res://scripts/target_dummy.gd")
@export var player_path: NodePath
@export var obs_viewport_path: NodePath
@export var max_steps: int = 500
@export var perception_fov_margin: float = 1.15

@onready var player: SandboxPlayer = get_node(player_path) as SandboxPlayer
@onready var obs_viewport: SubViewport = get_node(obs_viewport_path) as SubViewport
var reward: float = 0.0
var done: bool = false
var terminated: bool = false
var truncated: bool = false
var step_count: int = 0
var episode_seed: int = 42
var rng: RandomNumberGenerator = RandomNumberGenerator.new()
var shots_fired: int = 0
var hits: int = 0
var kills: int = 0
var damage_taken: float = 0.0
var tracked_target_id: int = 0
var tracking_confidence: float = 0.0
var tracking_memory_seconds: float = 0.0
var previous_detection: Dictionary = {}
@onready var scenario_director: SandboxScenarioDirector = get_tree().current_scene.get_node_or_null("ScenarioDirector") as SandboxScenarioDirector
@onready var episode_logger: SandboxEpisodeLogger = get_node("../EpisodeLogger") as SandboxEpisodeLogger
@onready var arena: SandboxArenaEnvironment = get_tree().current_scene as SandboxArenaEnvironment

func _ready() -> void:
	if player: player.shot_fired.connect(_on_player_shot_fired)

func _exit_tree() -> void:
	if player and player.shot_fired.is_connected(_on_player_shot_fired): player.shot_fired.disconnect(_on_player_shot_fired)

func _on_player_shot_fired(hit: bool, killed: bool) -> void:
	shots_fired += 1
	if episode_logger: episode_logger.record("shot", {"hit": hit, "killed": killed, "weapon": player.current_weapon_id})
	if hit: hits += 1; reward += 0.45
	else: reward -= 0.025
	if killed: kills += 1; reward += 2.0

func set_agent_mode(enabled: bool) -> void:
	player.set_agent_controlled(enabled)
	for node: Node in get_tree().get_nodes_in_group("targets"):
		var target: SandboxEnemy = node as SandboxEnemy
		if target: target.agent_controlled = enabled

func get_observation_image() -> Image:
	player.obs_camera.global_transform = player.camera.global_transform
	player.obs_camera.fov = player.camera.fov
	RenderingServer.force_draw(false)
	var image: Image = obs_viewport.get_texture().get_image()
	if episode_logger: episode_logger.record_observation(image)
	return image

func _visible_perception() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	var viewport_size: Vector2 = Vector2(obs_viewport.size)
	for node: Node in get_tree().get_nodes_in_group("targets"):
		var target: SandboxEnemy = node as SandboxEnemy
		if target == null or target.health <= 0.0: continue
		var world_pos: Vector3 = target.global_position + Vector3.UP * 0.8
		var screen: Vector2 = player.camera.unproject_position(world_pos)
		var in_frame: bool = player.camera.is_position_in_frustum(world_pos)
		var query: PhysicsRayQueryParameters3D = PhysicsRayQueryParameters3D.create(player.camera.global_position, world_pos)
		query.collide_with_areas = true
		query.exclude = [player]
		var hit: Dictionary = player.get_world_3d().direct_space_state.intersect_ray(query)
		var visible: bool = in_frame and (hit.is_empty() or hit.get("collider") == target)
		var distance: float = player.global_position.distance_to(target.global_position)
		var projected: Vector2 = Vector2(clampf(screen.x / 1280.0, 0.0, 1.0), clampf(screen.y / 720.0, 0.0, 1.0))
		var confidence: float = 0.0 if not visible else clampf(1.0 - distance / 35.0, 0.25, 0.99)
		var target_id: int = target.get_instance_id()
		if visible:
			tracked_target_id = target_id; tracking_confidence = confidence; tracking_memory_seconds = 0.75
		elif tracking_memory_seconds > 0.0 and target_id == tracked_target_id:
			tracking_memory_seconds = maxf(0.0, tracking_memory_seconds - 0.05); confidence = tracking_confidence * (tracking_memory_seconds / 0.75)
		if previous_detection.get(target_id, false) != visible and episode_logger:
			episode_logger.record("perception", {"target_id": target_id, "detected": visible, "confidence": confidence})
		previous_detection[target_id] = visible
		result.append({"id": target_id, "detected": visible, "confidence": confidence, "screen_position": [projected.x, projected.y], "screen_rect": [clampf(projected.x - 0.035, 0.0, 1.0), clampf(projected.y - 0.12, 0.0, 1.0), 0.07, 0.24], "distance": distance, "relative_position": [target.global_position.x - player.global_position.x, target.global_position.y - player.global_position.y, target.global_position.z - player.global_position.z], "moving": target.is_moving, "tracked": target_id == tracked_target_id and visible})
	return result

func get_replay() -> Array[Dictionary]:
	return episode_logger.snapshot() if episode_logger else []

func save_replay(path: String = "user://replays") -> String:
	return episode_logger.save_json(path) if episode_logger else ""

func get_state_info() -> Dictionary:
	return {"episode_seed": episode_seed, "step": step_count, "targets_hit": hits, "kills": kills, "shots_fired": shots_fired, "accuracy": float(hits) / float(maxi(1, shots_fired)), "last_shot_hit": player.last_shot_hit, "health": player.health, "ammo": player.ammo, "reserve_ammo": player.reserve_ammo, "reloading": player.reload_timer > 0.0, "damage_taken": damage_taken, "player_pos": [player.global_position.x, player.global_position.z], "perception": _visible_perception(), "tracked_target_id": tracked_target_id, "tracking_confidence": tracking_confidence, "scenario": scenario_director.state() if scenario_director else {}, "telemetry": {"fps": Engine.get_frames_per_second(), "frame_time_ms": 1000.0 / maxf(1.0, Engine.get_frames_per_second())}, "events": episode_logger.snapshot() if episode_logger else []}

func set_action(action: Array) -> void:
	if action.size() < 10: push_error("Sandbox action must contain 10 values"); return
	step_count += 1
	if episode_logger:
		episode_logger.tick()
		episode_logger.record("action", {"action": action.duplicate(), "health": player.health, "ammo": player.ammo})
	player.advance_simulation_timers(0.05)
	var health_before: float = player.health
	for node: Node in get_tree().get_nodes_in_group("targets"):
		var target: SandboxEnemy = node as SandboxEnemy
		if target: target.advance_simulation(0.05)
	var lost_health: float = maxf(0.0, health_before - player.health)
	if lost_health > 0.0: damage_taken += lost_health; reward -= lost_health * 0.018
	var move_x: int = clampi(int(action[0]), 0, 2) - 1
	var move_y: int = clampi(int(action[1]), 0, 2) - 1
	var do_ads: bool = bool(action[7])
	player.set_ads(do_ads)
	player.apply_rotation_input((clampi(int(action[8]), 0, 20) - 10) * 0.035 * (0.55 if do_ads else 1.0), (clampi(int(action[9]), 0, 20) - 10) * 0.018 * (0.55 if do_ads else 1.0))
	player.apply_movement_input(Vector2(move_x, move_y), bool(action[4]) and not do_ads, bool(action[2]), bool(action[3]))
	if bool(action[5]): player.start_reload()
	if bool(action[6]): player.shoot()
	terminated = player.health <= 0.0; truncated = step_count >= max_steps and not terminated; done = terminated or truncated

func reset(seed_value: int = 42, requested_map: String = "", requested_scenario: String = "") -> void:
	episode_seed = seed_value; rng.seed = seed_value; step_count = 0; done = false; terminated = false; truncated = false; reward = 0.0; shots_fired = 0; hits = 0; kills = 0; damage_taken = 0.0; tracked_target_id = 0; tracking_confidence = 0.0; tracking_memory_seconds = 0.0
	if arena: arena.configure(seed_value, requested_map)
	if scenario_director: scenario_director.configure(seed_value, requested_scenario)
	if episode_logger: episode_logger.begin(seed_value, "randomized", scenario_director.active_scenario if scenario_director else "none")
	var spawn_candidates: Array[Vector3] = [Vector3(-3, 0, 0), Vector3(3, 0, 0), Vector3(0, 0, 3), Vector3(0, 0, -1.5)]
	var valid_spawns: Array[Vector3] = []
	for candidate: Vector3 in spawn_candidates:
		if arena == null or arena.is_valid_spawn(candidate): valid_spawns.append(candidate)
	if valid_spawns.is_empty(): valid_spawns.append(Vector3.ZERO)
	player.reset_player(valid_spawns[rng.randi_range(0, valid_spawns.size() - 1)])
	var targets: Array[Node] = get_tree().get_nodes_in_group("targets")
	for index: int in range(targets.size()):
		var target: SandboxEnemy = targets[index] as SandboxEnemy
		if target: target.configure_seed(seed_value + index * 1009); target.respawn()
