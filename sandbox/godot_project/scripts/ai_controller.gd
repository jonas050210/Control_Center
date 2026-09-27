extends Node3D
class_name SandboxAIController

@export var player_path: NodePath
@export var obs_viewport_path: NodePath
@export var max_steps: int = 500

@onready var player: SandboxPlayer = get_node(player_path)
@onready var obs_viewport: SubViewport = get_node(obs_viewport_path)

var reward: float = 0.0
var done: bool = false
var terminated: bool = false
var truncated: bool = false
var step_count: int = 0
var episode_seed: int = 42
var rng := RandomNumberGenerator.new()
var shots_fired: int = 0
var hits: int = 0
var kills: int = 0
var damage_taken: float = 0.0

func _ready() -> void:
	if player:
		player.shot_fired.connect(_on_player_shot_fired)

func _exit_tree() -> void:
	if player and player.shot_fired.is_connected(_on_player_shot_fired):
		player.shot_fired.disconnect(_on_player_shot_fired)

func _on_player_shot_fired(hit: bool, killed: bool) -> void:
	shots_fired += 1
	if hit:
		hits += 1
		reward += 0.45
	else:
		reward -= 0.025
	if killed:
		kills += 1
		reward += 2.0

func set_agent_mode(enabled: bool) -> void:
	player.set_agent_controlled(enabled)
	for node in get_tree().get_nodes_in_group("targets"):
		var target := node as TargetDummy
		if target:
			target.agent_controlled = enabled

func get_observation_image() -> Image:
	player.obs_camera.global_transform = player.camera.global_transform
	RenderingServer.force_draw(false)
	return obs_viewport.get_texture().get_image()

func consume_reward() -> float:
	var result := reward - 0.002
	reward = 0.0
	return result

func get_state_info() -> Dictionary:
	return {
		"episode_seed": episode_seed,
		"step": step_count,
		"targets_hit": hits,
		"kills": kills,
		"shots_fired": shots_fired,
		"accuracy": float(hits) / float(maxi(1, shots_fired)),
		"last_shot_hit": player.last_shot_hit,
		"health": player.health,
		"ammo": player.ammo,
		"reserve_ammo": player.reserve_ammo,
		"reloading": player.reload_timer > 0.0,
		"damage_taken": damage_taken,
		"player_pos": [player.global_position.x, player.global_position.z],
	}

func set_action(action: Array) -> void:
	if action.size() < 10:
		push_error("Sandbox action must contain 10 values")
		return
	step_count += 1
	player.advance_simulation_timers(0.05)
	var health_before := player.health
	for node in get_tree().get_nodes_in_group("targets"):
		var target := node as TargetDummy
		if target:
			target.advance_simulation(0.05)
	var damage_taken := maxf(0.0, health_before - player.health)
	if damage_taken > 0.0:
		self.damage_taken += damage_taken
		reward -= damage_taken * 0.018
	var move_x := clampi(int(action[0]), 0, 2) - 1
	var move_y := clampi(int(action[1]), 0, 2) - 1
	var do_jump := bool(action[2])
	var do_crouch := bool(action[3])
	var do_sprint := bool(action[4])
	var do_reload := bool(action[5])
	var do_fire := bool(action[6])
	var do_ads := bool(action[7])
	var yaw_bin := clampi(int(action[8]), 0, 20)
	var pitch_bin := clampi(int(action[9]), 0, 20)
	var look_scale := 0.55 if do_ads else 1.0
	player.set_ads(do_ads)
	player.apply_rotation_input((yaw_bin - 10) * 0.035 * look_scale, (pitch_bin - 10) * 0.018 * look_scale)
	player.apply_movement_input(Vector2(move_x, move_y), do_sprint and not do_ads, do_jump, do_crouch)
	if do_reload:
		player.start_reload()
	if do_fire:
		player.shoot()
	terminated = player.health <= 0.0
	truncated = step_count >= max_steps and not terminated
	done = terminated or truncated

func reset(seed_value: int = 42) -> void:
	episode_seed = seed_value
	rng.seed = seed_value
	step_count = 0
	done = false
	terminated = false
	truncated = false
	reward = 0.0
	shots_fired = 0
	hits = 0
	kills = 0
	damage_taken = 0.0
	var spawn_points := [Vector3(-3, 0, 0), Vector3(3, 0, 0), Vector3(0, 0, 3), Vector3(0, 0, -1.5)]
	player.reset_player(spawn_points[rng.randi_range(0, spawn_points.size() - 1)])
	var targets := get_tree().get_nodes_in_group("targets")
	for index in range(targets.size()):
		var target := targets[index] as TargetDummy
		if target:
			target.configure_seed(seed_value + index * 1009)
			target.respawn()
