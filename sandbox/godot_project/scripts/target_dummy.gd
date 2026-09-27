extends Area3D
class_name TargetDummy

signal target_hit(dummy: TargetDummy)

@export var max_health: float = 100.0
@export var respawn_radius: float = 12.0
var health: float = 100.0
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

func _ready() -> void:
	rng.randomize()
	health = max_health
	add_to_group("targets")

func take_damage(amount: float) -> bool:
	health -= amount
	emit_signal("target_hit", self)
	if health <= 0.0:
		respawn()
		return true
	return false

func respawn() -> void:
	health = max_health
	# Pick random position inside arena
	var angle = rng.randf_range(0.0, TAU)
	var dist = rng.randf_range(4.0, respawn_radius)
	global_position = Vector3(cos(angle) * dist, 1.0, sin(angle) * dist)
