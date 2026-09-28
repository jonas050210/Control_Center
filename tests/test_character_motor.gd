## Tests for CharacterMotor: gravity, jumping, landing, air control and
## obstacle-aware movement. These are the physics both the agent and the
## enemies run on, so a regression here silently changes every level.
class_name TestCharacterMotor
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const CharacterMotor = preload("res://scripts/world/character_motor.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")

const DT: float = 1.0 / 60.0


func _step(
	world,
	position: Vector3,
	velocity: Vector3,
	wish: Vector3,
	jump: bool,
	on_ground: bool
) -> Dictionary:
	return CharacterMotor.step(
		world, position, velocity, wish, 4.5, DT, 0.4, 1.8, jump, on_ground, 10.0
	)


func test_grounded_character_does_not_fall_through_the_floor() -> SandboxTest:
	var t := SandboxTest.new("grounded_character_does_not_fall_through_the_floor")
	var state: Dictionary = {
		"position": Vector3.ZERO, "velocity": Vector3.ZERO, "on_ground": true
	}
	for _i in range(120):
		var result: Dictionary = _step(
			null, state["position"], state["velocity"], Vector3.ZERO, false, state["on_ground"]
		)
		state = result
	t.assert_almost_eq(float(state["position"].y), 0.0, 0.0001)
	t.assert_true(bool(state["on_ground"]))
	return t


func test_jump_rises_then_lands_and_reports_both_events() -> SandboxTest:
	var t := SandboxTest.new("jump_rises_then_lands_and_reports_both_events")
	var position := Vector3.ZERO
	var velocity := Vector3.ZERO
	var on_ground: bool = true
	var jumped_events: int = 0
	var landed_events: int = 0
	var apex: float = 0.0
	var airborne_ticks: int = 0
	for step_index in range(180):
		var jump: bool = step_index == 0
		var result: Dictionary = _step(null, position, velocity, Vector3.ZERO, jump, on_ground)
		position = result["position"]
		velocity = result["velocity"]
		on_ground = bool(result["on_ground"])
		if bool(result["jumped"]):
			jumped_events += 1
		if bool(result["landed"]):
			landed_events += 1
		if not on_ground:
			airborne_ticks += 1
		apex = maxf(apex, position.y)
	t.assert_eq(jumped_events, 1, "exactly one jump must be registered")
	t.assert_eq(landed_events, 1, "exactly one landing must be registered")
	t.assert_gt(apex, 0.6, "a jump must clear low cover")
	t.assert_lt(apex, 1.6, "a jump must not clear high cover")
	t.assert_gt(float(airborne_ticks), 10.0)
	t.assert_true(on_ground, "the character must be back on the ground at the end")
	return t


func test_jump_is_ignored_while_airborne() -> SandboxTest:
	var t := SandboxTest.new("jump_is_ignored_while_airborne")
	var first: Dictionary = _step(null, Vector3.ZERO, Vector3.ZERO, Vector3.ZERO, true, true)
	t.assert_true(bool(first["jumped"]))
	var second: Dictionary = _step(
		null, first["position"], first["velocity"], Vector3.ZERO, true, bool(first["on_ground"])
	)
	t.assert_false(bool(second["jumped"]), "a second jump must not trigger mid-air")
	return t


func test_air_control_limits_mid_air_steering() -> SandboxTest:
	var t := SandboxTest.new("air_control_limits_mid_air_steering")
	# Airborne with zero horizontal velocity, asking for full forward input:
	# after one tick the speed must be well below the grounded speed.
	var airborne: Dictionary = _step(
		null, Vector3(0.0, 1.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, -1.0), false, false
	)
	var air_speed: float = Vector2(
		float(airborne["velocity"].x), float(airborne["velocity"].z)
	).length()
	var grounded: Dictionary = _step(
		null, Vector3.ZERO, Vector3.ZERO, Vector3(0.0, 0.0, -1.0), false, true
	)
	var ground_speed: float = Vector2(
		float(grounded["velocity"].x), float(grounded["velocity"].z)
	).length()
	t.assert_almost_eq(ground_speed, 4.5, 0.001)
	t.assert_lt(air_speed, ground_speed, "air control must be weaker than ground acceleration")
	t.assert_almost_eq(air_speed, 4.5 * SandboxConfig.AIR_CONTROL, 0.01)
	return t


func test_character_can_jump_onto_a_low_box_and_stand_on_it() -> SandboxTest:
	var t := SandboxTest.new("character_can_jump_onto_a_low_box_and_stand_on_it")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 0.4, 0.0), Vector3(1.5, 0.4, 1.5), Obstacle.Kind.PLATFORM)
	var position := Vector3(0.0, 0.0, 3.0)
	var velocity := Vector3.ZERO
	var on_ground: bool = true
	for step_index in range(240):
		var jump: bool = step_index == 30
		var result: Dictionary = CharacterMotor.step(
			world,
			position,
			velocity,
			Vector3(0.0, 0.0, -1.0),
			4.5,
			DT,
			0.4,
			1.8,
			jump,
			on_ground,
			10.0
		)
		position = result["position"]
		velocity = result["velocity"]
		on_ground = bool(result["on_ground"])
	t.assert_almost_eq(position.y, 0.8, 0.05, "the character must end up standing on the box top")
	t.assert_true(on_ground)
	return t


func test_collision_stops_horizontal_velocity_instead_of_accumulating() -> SandboxTest:
	var t := SandboxTest.new("collision_stops_horizontal_velocity_instead_of_accumulating")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(0.0, 1.5, 0.0), Vector3(0.5, 1.5, 5.0), Obstacle.Kind.WALL)
	var position := Vector3(-1.0, 0.0, 0.0)
	var velocity := Vector3.ZERO
	for _i in range(60):
		var result: Dictionary = CharacterMotor.step(
			world, position, velocity, Vector3(1.0, 0.0, 0.0), 4.5, DT, 0.4, 1.8, false, true, 10.0
		)
		position = result["position"]
		velocity = result["velocity"]
	t.assert_lt(position.x, -0.8, "the character must not tunnel through the wall")
	t.assert_almost_eq(velocity.x, 0.0, 0.001, "blocked velocity must be zeroed, not accumulated")
	return t


func test_motion_is_deterministic() -> SandboxTest:
	var t := SandboxTest.new("motion_is_deterministic")
	var world: ArenaWorld = ArenaWorld.create(10.0)
	world.add_box(Vector3(1.0, 0.5, 1.0), Vector3(1.0, 0.5, 1.0), Obstacle.Kind.CRATE)
	var trace_a: Array = []
	var trace_b: Array = []
	for trace in [trace_a, trace_b]:
		var position := Vector3(-3.0, 0.0, -3.0)
		var velocity := Vector3.ZERO
		var on_ground: bool = true
		for step_index in range(120):
			var result: Dictionary = CharacterMotor.step(
				world,
				position,
				velocity,
				Vector3(1.0, 0.0, 1.0).normalized(),
				4.5,
				DT,
				0.4,
				1.8,
				step_index == 40,
				on_ground,
				10.0
			)
			position = result["position"]
			velocity = result["velocity"]
			on_ground = bool(result["on_ground"])
			trace.append(position)
	for index in range(trace_a.size()):
		t.assert_vec_almost_eq(trace_a[index], trace_b[index], 0.000001)
	return t
