## EnemyBrain
##
## Tactical behavior layer for one enemy. It is intentionally a stateless
## static driver over `EnemyState` fields (the enemy owns its own state) so
## an environment can hold N enemies without N extra objects, and so the
## whole thing stays trivially deterministic.
##
## Behavior graph:
##
##   IDLE      no contact, no memory              -> holds position
##   ALERT     contact detected, not yet confirmed-> turns toward it
##   ENGAGE    confirmed target, LOS              -> holds preferred range,
##                                                   strafes and shoots
##   TAKE_COVER hurt or losing the trade          -> moves to a position
##                                                   the agent cannot see
##   PEEK      cover dwell elapsed                -> steps to a spot that
##                                                   does see the agent
##   SEARCH    contact lost                       -> walks to the last known
##                                                   position, then wanders
##                                                   nearby, then gives up
##   DEAD      corpse                             -> does nothing forever
##
## Everything it knows comes from `PerceptionSystem` (FOV + LOS), the
## `SoundBus` and its own `EnemyMemory`. It is never handed the agent's
## live position directly: `update()` receives the true agent state only to
## run the perception queries, and every decision downstream reads the
## memory track instead. That is what makes an enemy actually lose you
## around a corner.
class_name EnemyBrain
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const NavigationAgent = preload("res://scripts/world/navigation_agent.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

## Memory key used for "the player". Enemies do not track each other.
const AGENT_TRACK_ID: int = -1


## Advances one enemy by one tick.
##
## `context` carries the shared per-environment objects and flags:
##   world, navigation, lighting, sound_bus, rng, arena_half_extent, agent_position,
##   agent_eye, agent_height, agent_alive, dt, time_seconds,
##   allow_ranged, allow_movement, allow_jump
##
## Returns the events the environment needs:
##   {"damage": float, "shot": bool, "hit": bool, "footstep": bool,
##    "jumped": bool, "landed": bool, "sound_position": Vector3}
static func update(enemy: EnemyState, context: Dictionary) -> Dictionary:
	var events: Dictionary = {
		"damage": 0.0,
		"shot": false,
		"hit": false,
		"footstep": false,
		"jumped": false,
		"landed": false,
	}
	var dt: float = float(context.get("dt", SandboxConfig.SIMULATION_DT))
	enemy.weapon.tick(dt)
	if enemy.attack_cooldown_remaining > 0.0:
		enemy.attack_cooldown_remaining = maxf(0.0, enemy.attack_cooldown_remaining - dt)
	if not enemy.is_targetable():
		enemy.ai_state = EnemyState.AIState.DEAD
		return events

	enemy.time_alive += dt
	enemy.state_time += dt
	enemy.memory.tick(dt)

	_perceive(enemy, context, dt)
	_decide(enemy, context)
	var motion: Dictionary = _act(enemy, context, dt, events)
	events["footstep"] = bool(motion.get("footstep", false))
	events["jumped"] = bool(motion.get("jumped", false))
	events["landed"] = bool(motion.get("landed", false))
	return events


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


## Updates visual contact timers, the memory track and the reaction-gated
## `target_confirmed` flag.
static func _perceive(enemy: EnemyState, context: Dictionary, dt: float) -> void:
	var world = context.get("world")
	var agent_position: Vector3 = context.get("agent_position", Vector3.ZERO)
	var agent_alive: bool = bool(context.get("agent_alive", true))
	var agent_height: float = float(context.get("agent_height", SandboxConfig.AGENT_HEIGHT))

	# The opponents obey exactly the same visibility rules as the policy,
	# including lighting: an enemy in the dark also loses you.
	var lighting = context.get("lighting")
	var range_limit: float = SandboxConfig.VISION_RANGE
	var delay_scale: float = 1.0
	if lighting != null:
		var profile: LightingProfile = lighting
		range_limit = profile.detection_range(SandboxConfig.VISION_RANGE, agent_position)
		delay_scale = profile.detection_delay_scale(agent_position)

	var visible: bool = false
	if agent_alive:
		var evaluation: Dictionary = PerceptionSystem.evaluate_target(
			world,
			enemy.get_eye_position(),
			enemy.position,
			enemy.get_forward_horizontal(),
			agent_position,
			agent_height,
			SandboxConfig.ENEMY_FOV_DEG,
			range_limit
		)
		visible = bool(evaluation["visible"])

	# Line of sight is symmetric: while the enemy can see the agent, the
	# agent could see it back, so this doubles as an exposure clock.
	if visible:
		enemy.exposure_time += dt
	else:
		enemy.exposure_time = 0.0

	if visible:
		enemy.visual_contact_time += dt
		enemy.time_since_visual = 0.0
		if enemy.visual_contact_time >= enemy.reaction.visual_detection_delay * delay_scale:
			var direction: Vector3 = (agent_position - enemy.position)
			direction.y = 0.0
			enemy.memory.observe_visual(
				AGENT_TRACK_ID,
				agent_position,
				direction.normalized() if not direction.is_zero_approx() else Vector3.ZERO
			)
	else:
		enemy.visual_contact_time = 0.0
		enemy.time_since_visual += dt
		_hear(enemy, context)

	# Reaction gating: a confirmed target requires detection AND the extra
	# confirmation latency on top of it.
	var track: Dictionary = enemy.memory.get_track(AGENT_TRACK_ID)
	if track.is_empty():
		enemy.target_confirmed = false
		return
	var required: float = (
		enemy.reaction.visual_detection_delay * delay_scale + enemy.reaction.target_confirm_delay
	)
	if visible and enemy.visual_contact_time >= required:
		enemy.target_confirmed = true
	elif not visible and enemy.time_since_visual > SandboxConfig.VISUAL_LOSS_GRACE:
		enemy.target_confirmed = false


## Folds the loudest audible event into memory as a low-confidence track.
static func _hear(enemy: EnemyState, context: Dictionary) -> void:
	var bus = context.get("sound_bus")
	if bus == null:
		return
	var event: Dictionary = (bus as SoundBus).loudest(
		enemy.position,
		enemy.get_forward_horizontal(),
		context.get("world"),
		enemy.enemy_id,
		enemy.reaction.sound_detection_delay
	)
	if event.is_empty():
		return
	var direction: Vector3 = event["direction"]
	var distance: float = float(event["distance"])
	var estimate: Vector3 = enemy.position + direction * distance
	estimate.y = 0.0
	enemy.memory.observe_sound(
		AGENT_TRACK_ID, estimate, direction, float(event["loudness"])
	)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


static func _decide(enemy: EnemyState, context: Dictionary) -> void:
	var track: Dictionary = enemy.memory.get_track(AGENT_TRACK_ID)
	var previous: int = enemy.ai_state

	if track.is_empty():
		_set_state(enemy, EnemyState.AIState.IDLE, "no contact")
	elif _should_retreat(enemy):
		_set_state(enemy, EnemyState.AIState.RETREAT, "critically hurt, breaking away")
		_choose_retreat(enemy, context, track)
	elif enemy.target_confirmed and _should_take_cover(enemy):
		_set_state(enemy, EnemyState.AIState.TAKE_COVER, "low health, breaking contact")
		_choose_cover(enemy, context, track)
	elif enemy.target_confirmed and _over_exposed(enemy):
		_set_state(enemy, EnemyState.AIState.TAKE_COVER, "exposed too long, repositioning")
		_choose_cover(enemy, context, track)
	elif _only_heard(track) and not enemy.target_confirmed:
		if enemy.ai_state != EnemyState.AIState.INVESTIGATE:
			_set_state(enemy, EnemyState.AIState.INVESTIGATE, "heard something, investigating")
			enemy.search_time = 0.0
		enemy.search_time += float(context.get("dt", SandboxConfig.SIMULATION_DT))
		if enemy.search_time >= SandboxConfig.ENEMY_SEARCH_DURATION:
			enemy.memory.forget(AGENT_TRACK_ID)
			_set_state(enemy, EnemyState.AIState.IDLE, "noise led nowhere")
		elif _reached_destination(enemy):
			enemy.memory.mark_investigated(AGENT_TRACK_ID)
			_choose_search(enemy, context, track)
		else:
			enemy.tactical_destination = Vector3(track.get("position", enemy.position))
			enemy.has_tactical_destination = true
	elif enemy.ai_state == EnemyState.AIState.TAKE_COVER:
		if enemy.state_time >= SandboxConfig.ENEMY_COVER_DWELL:
			_set_state(enemy, EnemyState.AIState.PEEK, "cover dwell elapsed, peeking")
			_choose_peek(enemy, context, track)
	elif enemy.ai_state == EnemyState.AIState.PEEK and not enemy.target_confirmed:
		if enemy.state_time >= SandboxConfig.ENEMY_COVER_DWELL:
			_set_state(enemy, EnemyState.AIState.SEARCH, "peek found nothing")
			_choose_search(enemy, context, track)
	elif enemy.target_confirmed:
		_set_state(enemy, EnemyState.AIState.ENGAGE, "target confirmed")
		enemy.search_time = 0.0
		enemy.has_tactical_destination = false
	elif float(track.get("confidence", 0.0)) > 0.0:
		if enemy.ai_state != EnemyState.AIState.SEARCH:
			_set_state(enemy, EnemyState.AIState.ALERT, "contact lost, investigating")
			enemy.search_time = 0.0
			_choose_search(enemy, context, track)
		else:
			enemy.search_time += float(context.get("dt", SandboxConfig.SIMULATION_DT))
			if enemy.search_time >= SandboxConfig.ENEMY_SEARCH_DURATION:
				enemy.memory.forget(AGENT_TRACK_ID)
				_set_state(enemy, EnemyState.AIState.IDLE, "gave up searching")
			elif _reached_destination(enemy):
				enemy.memory.mark_investigated(AGENT_TRACK_ID)
				_choose_search(enemy, context, track)

	if enemy.ai_state == EnemyState.AIState.ALERT and enemy.state_time > 0.35:
		_set_state(enemy, EnemyState.AIState.SEARCH, "moving to last known position")
	if previous != enemy.ai_state:
		enemy.state_time = 0.0


## Critically hurt: stop trading entirely. Separate from `_should_take_cover`
## because cover may not exist, and running is still better than dying.
static func _should_retreat(enemy: EnemyState) -> bool:
	if enemy.max_health <= 0.0:
		return false
	return enemy.health / enemy.max_health <= SandboxConfig.ENEMY_CRITICAL_HEALTH_FRACTION


## Standing in the open for too long during an engagement. This is what
## produces repositioning instead of a static firefight, and it comes from
## the enemy's own (symmetric) line-of-sight information, not from anything
## privileged.
static func _over_exposed(enemy: EnemyState) -> bool:
	return enemy.exposure_time >= SandboxConfig.ENEMY_MAX_EXPOSURE_TIME


## True when the only information about the target came from hearing. A
## sound track is an approximate position with no confirmation, so it earns
## a cautious approach rather than an engagement.
static func _only_heard(track: Dictionary) -> bool:
	if track.is_empty():
		return false
	return int(track.get("source", EnemyMemory.Source.NONE)) == EnemyMemory.Source.SOUND


## Destination directly away from the believed threat, clamped inside the
## arena and rejected onto the enemy's own position when blocked.
static func _choose_retreat(enemy: EnemyState, context: Dictionary, track: Dictionary) -> void:
	var threat: Vector3 = Vector3(track.get("position", enemy.position))
	var away := Vector3(enemy.position.x - threat.x, 0.0, enemy.position.z - threat.z)
	if away.length() < 0.0001:
		away = Vector3(1.0, 0.0, 0.0)
	var limit: float = (
		float(context.get("arena_half_extent", SandboxConfig.ARENA_HALF_EXTENT)) - enemy.radius
	)
	var candidate: Vector3 = enemy.position + away.normalized() * SandboxConfig.ENEMY_RETREAT_DISTANCE
	candidate.x = clampf(candidate.x, -limit, limit)
	candidate.z = clampf(candidate.z, -limit, limit)
	candidate.y = 0.0
	enemy.tactical_destination = candidate
	enemy.has_tactical_destination = true


static func _should_take_cover(enemy: EnemyState) -> bool:
	if enemy.max_health <= 0.0:
		return false
	return enemy.health / enemy.max_health <= SandboxConfig.ENEMY_RETREAT_HEALTH_FRACTION


static func _reached_destination(enemy: EnemyState) -> bool:
	if not enemy.has_tactical_destination:
		return true
	return enemy.position.distance_to(enemy.tactical_destination) <= 0.75


static func _choose_cover(enemy: EnemyState, context: Dictionary, track: Dictionary) -> void:
	var world = context.get("world")
	if world == null:
		enemy.has_tactical_destination = false
		return
	var threat_eye: Vector3 = Vector3(track.get("position", enemy.position))
	threat_eye.y += SandboxConfig.AGENT_EYE_HEIGHT
	enemy.tactical_destination = (world as ArenaWorld).find_cover_position(
		enemy.position,
		threat_eye,
		SandboxConfig.ENEMY_COVER_SEARCH_RADIUS,
		enemy.radius,
		enemy.height,
		enemy.eye_height
	)
	enemy.has_tactical_destination = true


static func _choose_peek(enemy: EnemyState, context: Dictionary, track: Dictionary) -> void:
	var world = context.get("world")
	if world == null:
		enemy.has_tactical_destination = false
		return
	var threat_eye: Vector3 = Vector3(track.get("position", enemy.position))
	threat_eye.y += SandboxConfig.AGENT_EYE_HEIGHT
	enemy.tactical_destination = (world as ArenaWorld).find_peek_position(
		enemy.position,
		threat_eye,
		SandboxConfig.ENEMY_COVER_SEARCH_RADIUS * 0.6,
		enemy.radius,
		enemy.height,
		enemy.eye_height
	)
	enemy.has_tactical_destination = true


## Picks the next search waypoint: the last known position first, then
## deterministic points around it. Uses the environment RNG so the wander
## pattern is reproducible for a given seed.
static func _choose_search(enemy: EnemyState, context: Dictionary, track: Dictionary) -> void:
	var last_known: Vector3 = Vector3(track.get("position", enemy.position))
	if not bool(track.get("investigated", false)):
		enemy.tactical_destination = last_known
		enemy.has_tactical_destination = true
		return
	# Typed as Variant on purpose: callers may inject a deterministic RNG
	# stub (see tests/test_bug_regressions.gd). A `RandomNumberGenerator`
	# static type makes that assignment fail at runtime with
	# "Trying to assign value of type 'RefCounted'".
	var rng: Variant = context.get("rng")
	var angle: float = 0.0
	var distance: float = SandboxConfig.ENEMY_SEARCH_RADIUS
	if rng != null:
		angle = rng.randf_range(0.0, TAU)
		distance = rng.randf_range(1.0, SandboxConfig.ENEMY_SEARCH_RADIUS)
	var candidate := Vector3(
		last_known.x + cos(angle) * distance, 0.0, last_known.z + sin(angle) * distance
	)
	var world = context.get("world")
	if world != null:
		var arena: ArenaWorld = world
		if not arena.is_position_free(candidate, enemy.radius, enemy.height):
			candidate = last_known
	enemy.tactical_destination = candidate
	enemy.has_tactical_destination = true


static func _set_state(enemy: EnemyState, state: int, reason: String) -> void:
	if enemy.ai_state != state:
		enemy.ai_state = state
		enemy.state_time = 0.0
		# A route planned for the previous behavior's destination is stale
		# the moment the behavior changes; keeping it would walk an
		# engaging enemy back toward an abandoned cover spot.
		enemy.navigation.abandon()
	enemy.tactical_reason = reason


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


static func _act(
	enemy: EnemyState, context: Dictionary, dt: float, events: Dictionary
) -> Dictionary:
	var world = context.get("world")
	var arena_half_extent: float = float(
		context.get("arena_half_extent", SandboxConfig.ARENA_HALF_EXTENT)
	)
	var allow_movement: bool = bool(context.get("allow_movement", true))
	var track: Dictionary = enemy.memory.get_track(AGENT_TRACK_ID)
	var believed_position: Vector3 = (
		Vector3(track.get("position", enemy.position)) if not track.is_empty() else enemy.position
	)

	enemy.face_towards(believed_position, dt, enemy.reaction.aim_speed_deg)

	var destination: Vector3 = enemy.position
	var speed_scale: float = 1.0
	match enemy.ai_state:
		EnemyState.AIState.ENGAGE:
			destination = _engage_destination(enemy, believed_position, context)
		EnemyState.AIState.TAKE_COVER, EnemyState.AIState.PEEK, EnemyState.AIState.SEARCH:
			destination = (
				enemy.tactical_destination if enemy.has_tactical_destination else enemy.position
			)
			speed_scale = 0.85 if enemy.ai_state == EnemyState.AIState.SEARCH else 1.0
		EnemyState.AIState.INVESTIGATE:
			destination = (
				enemy.tactical_destination if enemy.has_tactical_destination else enemy.position
			)
			speed_scale = SandboxConfig.ENEMY_INVESTIGATE_SPEED_SCALE
		EnemyState.AIState.RETREAT:
			destination = (
				enemy.tactical_destination if enemy.has_tactical_destination else enemy.position
			)
			speed_scale = 1.0
		_:
			destination = enemy.position
			speed_scale = 0.0

	if not allow_movement:
		speed_scale = 0.0

	# Navigation is consulted only when the enemy is demonstrably blocked
	# (see NavigationAgent): in the open field this is two float compares.
	var steering: Vector3 = enemy.navigation.update(
		context.get("navigation"),
		world,
		enemy.position,
		destination,
		speed_scale > 0.0,
		dt
	)
	if enemy.navigation.status != NavigationAgent.STATUS_DIRECT:
		enemy.tactical_reason = "%s (nav: %s)" % [
			enemy.tactical_reason, enemy.navigation.status
		]
	destination = steering

	var jump_requested: bool = _wants_jump(enemy, destination, context)
	var motion: Dictionary = enemy.move_towards(
		destination, dt, arena_half_extent, world, speed_scale, jump_requested
	)

	if enemy.ai_state == EnemyState.AIState.ENGAGE or enemy.ai_state == EnemyState.AIState.PEEK:
		_try_attack(enemy, context, events)
	return motion


## Holds the preferred engagement distance and adds a lateral strafe
## component, so an engaging enemy circles rather than walking into melee.
static func _engage_destination(
	enemy: EnemyState, target_position: Vector3, context: Dictionary
) -> Vector3:
	var to_target := Vector3(
		target_position.x - enemy.position.x, 0.0, target_position.z - enemy.position.z
	)
	var distance: float = to_target.length()
	if distance < 0.0001:
		return enemy.position
	var direction: Vector3 = to_target / distance
	var preferred: float = SandboxConfig.ENEMY_PREFERRED_RANGE
	var radial: float = distance - preferred
	var lateral := Vector3(-direction.z, 0.0, direction.x)
	var strafe_signal: float = (
		sin(enemy.time_alive * SandboxConfig.ENEMY_STRAFE_ANGULAR_SPEED + enemy.strafe_phase)
		* enemy.strafe_direction
	)
	var strafe_enabled: bool = bool(context.get("allow_strafe", true))
	var lateral_offset: float = (
		strafe_signal * SandboxConfig.ENEMY_STRAFE_SPEED_SCALE * 2.5 if strafe_enabled else 0.0
	)
	return enemy.position + direction * clampf(radial, -2.0, 2.0) + lateral * lateral_offset


## Jump when a standable box blocks the direct path and the curriculum
## allows it. Deterministic given the environment RNG.
static func _wants_jump(enemy: EnemyState, destination: Vector3, context: Dictionary) -> bool:
	if not bool(context.get("allow_jump", false)) or not enemy.on_ground:
		return false
	var world = context.get("world")
	if world == null:
		return false
	var delta := Vector3(destination.x - enemy.position.x, 0.0, destination.z - enemy.position.z)
	if delta.length() < 0.2:
		return false
	var probe: Vector3 = enemy.position + delta.normalized() * (enemy.radius + 0.4)
	probe.y = enemy.position.y
	var arena: ArenaWorld = world
	if arena.is_position_free(probe, enemy.radius, enemy.height):
		return false
	var rng: Variant = context.get("rng")
	if rng == null:
		return true
	return rng.randf() < SandboxConfig.ENEMY_JUMP_PROBABILITY


## Ranged or melee attack, gated by reaction latency, line of sight and
## aim error. Returns nothing; damage is written into `events`.
static func _try_attack(enemy: EnemyState, context: Dictionary, events: Dictionary) -> void:
	if not bool(context.get("allow_attack", true)):
		return
	if not bool(context.get("agent_alive", true)):
		return
	if enemy.state_time < enemy.reaction.aim_reaction_delay + enemy.reaction.shoot_reaction_delay:
		return
	var agent_position: Vector3 = context.get("agent_position", Vector3.ZERO)
	var distance: float = enemy.position.distance_to(agent_position)

	if bool(context.get("allow_ranged", false)):
		if distance > SandboxConfig.ENEMY_FIRE_RANGE:
			return
		var world = context.get("world")
		var agent_height: float = float(context.get("agent_height", SandboxConfig.AGENT_HEIGHT))
		if not PerceptionSystem.has_line_of_sight(
			world, enemy.get_eye_position(), agent_position, agent_height
		):
			return
		if not enemy.weapon.try_fire():
			return
		events["shot"] = true
		# Aim error is converted into a distance-scaled hit probability so
		# the enemy is dangerous up close and unreliable far away.
		var spread_m: float = tan(deg_to_rad(enemy.reaction.aim_error_deg)) * distance
		var hit_chance: float = clampf(
			enemy.weapon.hit_radius / maxf(enemy.weapon.hit_radius + spread_m, 0.0001), 0.05, 1.0
		)
		var rng: Variant = context.get("rng")
		var roll: float = float(rng.randf()) if rng != null else 0.0
		if roll <= hit_chance:
			events["hit"] = true
			events["damage"] = float(events.get("damage", 0.0)) + SandboxConfig.ENEMY_FIRE_DAMAGE
		# The ranged attack was made this tick (the weapon cooldown was
		# consumed whether the roll hit or missed), so the enemy is done:
		# falling through to the melee check made a MISSED point-blank shot
		# deal melee damage on top (and more than a hit: 10 vs 9) while a
		# hit or a cooling-down weapon dealt none. Ranged enemies melee
		# only when `allow_ranged` is off, per the curriculum contract
		# ("enemies shoot instead of meleeing").
		return

	if distance <= enemy.attack_range and enemy.attack_cooldown_remaining <= 0.0:
		enemy.attack_cooldown_remaining = enemy.attack_cooldown_time
		events["hit"] = true
		events["damage"] = float(events.get("damage", 0.0)) + enemy.attack_damage
