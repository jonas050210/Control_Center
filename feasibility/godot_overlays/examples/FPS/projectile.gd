extends Area3D
class_name Projectile

@export var speed = 40
@export var damage = 10
var velocity = Vector3.ZERO
var shooter = null
var _destroying := false

var ProjectileImpact = preload("res://projectile_impact.tscn")

var team_materials = {
	0: preload("res://projectile_mat_green_team.tres"),
	1: preload("res://projectile_mat_red_team.tres"),
}

func set_team(value):
	if value != -1:
		$MeshInstance3d.set_surface_override_material(0, team_materials[value])

func _physics_process(delta):
	#look_at(transform.origin + velocity.normalized(), Vector3.UP)
	transform.origin += velocity * delta

func _on_timer_timeout():
	_destroy()

func _on_projectile_area_entered(area):
	if area is PlayerHitBox and shooter != null:
		shooter.hit_player(area._player)

	# create a cool animation
	#explode()
	_destroy()

func _on_body_entered(_body):
	# Do not free an Area3D from inside a physics overlap callback. Godot is
	# still emitting overlap notifications and maintaining its internal
	# tree_entered/tree_exited connections at this point.
	_destroy()

func explode():
	print("explode")
	var proj_impact = ProjectileImpact.instantiate()
	shooter.add_child(proj_impact)
	proj_impact.global_position = global_position
	proj_impact.set_as_top_level(true)


func _destroy() -> void:
	# More than one signal can be emitted for the same impact. Begin teardown
	# once, stop new overlap notifications after the physics flush, and only
	# then remove the projectile from the tree.
	if _destroying:
		return
	_destroying = true
	set_deferred("monitoring", false)
	set_deferred("monitorable", false)
	call_deferred("_finish_destroy")

func _finish_destroy() -> void:
	# queue_free() is valid for a node that already left the tree (e.g. its
	# shooter was freed/respawned in the same frame).  Skipping the free for
	# such a projectile silently leaked it: nothing else owns it, so it
	# survived until engine shutdown and was reported as
	# "ObjectDB instances leaked at exit".
	if not is_queued_for_deletion():
		queue_free()
