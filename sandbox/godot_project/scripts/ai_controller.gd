extends Node3D
class_name SandboxAIController

@export var player_path: NodePath
@export var obs_viewport_path: NodePath

@onready var player: SandboxPlayer = get_node(player_path)
@onready var obs_viewport: SubViewport = get_node(obs_viewport_path)

var reward: float = 0.0
var done: bool = false
var step_count: int = 0
@export var max_steps: int = 500

var rng: RandomNumberGenerator = RandomNumberGenerator.new()

func _ready() -> void:
	rng.randomize()
	if player:
		player.connect("shot_fired", Callable(self, "_on_player_shot_fired"))

func _on_player_shot_fired(hit: bool) -> void:
	if hit:
		reward += 1.0  # +1 reward on target hit
	else:
		reward -= 0.02  # small penalty on missed shot

func get_obs() -> Dictionary:
	# Returns observation dictionary for godot_rl_agents
	var obs := {}
	# Visual observation from SubViewport (84x84 or 160x120 RGB)
	obs["camera_2d"] = obs_viewport.get_texture().get_image()
	return obs

func get_reward() -> float:
	var r = reward
	reward = 0.0
	# Minor step penalty to encourage speed
	r -= 0.005
	return r

func get_action_space() -> Dictionary:
	return {
		"move_x": {"size": 3, "action_type": "discrete"},  # 0=Left, 1=None, 2=Right
		"move_y": {"size": 3, "action_type": "discrete"},  # 0=Backward, 1=None, 2=Forward
		"fire": {"size": 2, "action_type": "discrete"},    # 0=None, 1=Shoot
		"turn_yaw": {"size": 21, "action_type": "discrete"}, # Discretized horizontal look
		"turn_pitch": {"size": 21, "action_type": "discrete"}, # Discretized vertical look
	}

func set_action(action: Variant) -> void:
	step_count += 1
	if step_count >= max_steps:
		done = true

	# Action parsing
	var move_x_idx: int = 1
	var move_y_idx: int = 1
	var do_fire: int = 0
	var yaw_bin: int = 10
	var pitch_bin: int = 10

	if action is Array and action.size() >= 5:
		move_x_idx = int(action[0])
		move_y_idx = int(action[1])
		do_fire = int(action[2])
		yaw_bin = int(action[3])
		pitch_bin = int(action[4])
	elif action is Dictionary:
		move_x_idx = int(action.get("move_x", 1))
		move_y_idx = int(action.get("move_y", 1))
		do_fire = int(action.get("fire", 0))
		yaw_bin = int(action.get("turn_yaw", 10))
		pitch_bin = int(action.get("turn_pitch", 10))

	var move_x = float(move_x_idx - 1)  # 0, 1, 2 -> -1, 0, 1
	var move_y = float(move_y_idx - 1)  # 0, 1, 2 -> -1, 0, 1

	# Dequantize yaw/pitch bin (center bin 10 is 0.0 rad)
	var yaw_delta = float(yaw_bin - 10) * 0.03
	var pitch_delta = float(pitch_bin - 10) * 0.02

	player.apply_rotation_input(yaw_delta, pitch_delta)
	player.apply_movement_input(Vector2(move_x, move_y), false, false)

	if do_fire == 1:
		player.shoot()

func reset() -> void:
	step_count = 0
	done = false
	reward = 0.0
	var angle = rng.randf_range(0.0, TAU)
	var spawn = Vector3(cos(angle) * 3.0, 1.0, sin(angle) * 3.0)
	player.reset_player(spawn)

	# Reset targets
	var targets = get_tree().get_nodes_in_group("targets")
	for target in targets:
		if target is TargetDummy:
			target.respawn()
