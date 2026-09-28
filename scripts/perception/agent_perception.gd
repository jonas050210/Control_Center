## AgentPerception
##
## The learning agent's side of the perception pipeline: what the policy is
## ALLOWED to know about the world this tick.
##
## It owns
##   * per-enemy visual contact timers (reaction latency, Phase 7),
##   * an `EnemyMemory` of last-known positions with decay (Phase 4),
##   * the most recent audible `SoundBus` sample (Phase 3),
##
## and produces a `belief` list that `Observation.build()` consumes. When
## perception gating is DISABLED (`enabled = false`) it still runs the
## geometry so the Control Center can visualize it, but the belief entries
## carry ground-truth positions — this is exactly what the pre-perception
## curriculum levels used, preserved so previously trained policies keep
## their dynamics.
##
## Nothing here ever invents information. An enemy that is neither visible
## nor remembered simply is not in the belief list, and the observation
## reports "no contact" for that slot rather than a fabricated position.
class_name AgentPerception
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const EnemyMemory = preload("res://scripts/perception/enemy_memory.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const PerceptionSystem = preload("res://scripts/perception/perception_system.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")

const SELF_PATH: String = "res://scripts/perception/agent_perception.gd"

## When false the agent perceives every living enemy regardless of FOV,
## occlusion and latency (legacy analytic behavior).
var enabled: bool = false
## When false, sound events are not sampled at all.
var sound_enabled: bool = false
## When false, lost contacts are dropped immediately instead of decaying.
var memory_enabled: bool = false

var fov_deg: float = SandboxConfig.AGENT_FOV_DEG
var vision_range: float = SandboxConfig.VISION_RANGE
var detection_delay: float = SandboxConfig.AGENT_VISUAL_DETECTION_DELAY

## Environmental visibility conditions. Never observed as a mode; it acts
## by shortening the acquisition range, slowing detection and shortening
## the loss grace, so the policy experiences consequences rather than a
## label. Defaults to NORMAL, i.e. exactly the pre-lighting behavior.
var lighting: LightingProfile = LightingProfile.create()
## Perceived brightness where the agent is standing, in [0, 1]. This IS
## exposed to the policy: a human standing in a dark room knows it is dark.
var local_illumination: float = 1.0

var memory: EnemyMemory = EnemyMemory.create()
## enemy id -> continuous seconds of unbroken geometric visibility.
var contact_timers: Dictionary = {}
## enemy id -> seconds since the target was last confirmed visible.
var loss_timers: Dictionary = {}
## Last audible sound sample (loudest first) and its count.
var heard: Array = []
## Aggregate description of that sample: how many distinct directions noise
## came from, how much of it was masked, and how much the agent should
## trust the bearings (SoundBus.summarize()).
var sound_summary: Dictionary = SoundBus.summarize([])
## Distance to the first occluder straight ahead.
var forward_clearance: float = SandboxConfig.VISION_RANGE
## Whether geometry currently hides the agent from every living enemy.
var in_cover: bool = true
## How many living enemies currently have line of sight to the agent.
var threat_count: int = 0


static func create() -> AgentPerception:
	return (load(SELF_PATH) as GDScript).new() as AgentPerception


func configure(
	p_enabled: bool, p_sound: bool, p_memory: bool, p_detection_delay: float = -1.0
) -> void:
	enabled = p_enabled
	sound_enabled = p_sound
	memory_enabled = p_memory
	if p_detection_delay >= 0.0:
		detection_delay = p_detection_delay


func set_lighting(profile) -> void:
	lighting = profile if profile != null else LightingProfile.create()


func reset() -> void:
	memory.clear()
	local_illumination = 1.0
	contact_timers.clear()
	loss_timers.clear()
	heard.clear()
	sound_summary = SoundBus.summarize([])
	forward_clearance = vision_range
	in_cover = true
	threat_count = 0


## Drops every trace of an enemy. Called when it dies: the agent saw the
## kill it scored, so continuing to remember a corpse as a live threat
## would be wrong, and a corpse must never be a target.
func forget(enemy_id: int) -> void:
	memory.forget(enemy_id)
	contact_timers.erase(enemy_id)
	loss_timers.erase(enemy_id)


## Runs one perception tick and returns the belief list.
##
## Each belief entry:
##   {id, visible, in_fov, los_clear, distance, bearing_deg, elevation_deg,
##    position, health_norm, age, confidence, source, alive}
## `position` is the live position while visible and the last known
## position afterwards; `age`/`confidence`/`source` tell the policy which.
func update(agent, enemies: Array, world, sound_bus, dt: float) -> Array:
	if memory_enabled:
		memory.tick(dt)
	else:
		memory.clear()

	var eye: Vector3 = agent.get_eye_position()
	var forward: Vector3 = agent.get_forward_horizontal()
	forward_clearance = PerceptionSystem.forward_clearance(world, eye, forward, vision_range)
	local_illumination = lighting.illumination_at(agent.position)
	threat_count = 0

	var beliefs: Array = []
	for enemy_value in enemies:
		var enemy: EnemyState = enemy_value
		if not enemy.is_targetable():
			forget(enemy.enemy_id)
			continue
		var belief: Dictionary = _evaluate_enemy(agent, enemy, world, eye, forward, dt)
		if not belief.is_empty():
			beliefs.append(belief)

	in_cover = threat_count == 0
	_sample_sound(agent, world, sound_bus)
	return beliefs


## Beliefs ranked for target selection (Phase 5): visible contacts first,
## then by confidence, then by distance. A caller may pass `damage_source`
## (the id of whatever last hurt the agent) to bias selection toward it.
static func rank_beliefs(beliefs: Array, damage_source: int = -1) -> Array:
	var ranked: Array = beliefs.duplicate()
	ranked.sort_custom(
		func(a, b):
			var score_a: float = _selection_score(a, damage_source)
			var score_b: float = _selection_score(b, damage_source)
			if absf(score_a - score_b) > 0.000001:
				return score_a > score_b
			return float(a["distance"]) < float(b["distance"])
	)
	return ranked


## Higher is a better target. Visibility dominates, then recent damage,
## then confidence, then proximity — which is the ordering a human player
## uses and the one the Control Center displays as "target reason".
static func _selection_score(belief: Dictionary, damage_source: int) -> float:
	var score: float = 0.0
	if bool(belief.get("visible", false)):
		score += 100.0
	if int(belief.get("id", -1)) == damage_source:
		score += 40.0
	score += float(belief.get("confidence", 0.0)) * 20.0
	score += clampf(
		1.0 - float(belief.get("distance", 0.0)) / SandboxConfig.ARENA_MAX_DISTANCE, 0.0, 1.0
	) * 10.0
	return score


## Human-readable justification for why a belief was chosen. Debug/UI only.
static func selection_reason(belief: Dictionary, damage_source: int) -> String:
	if belief.is_empty():
		return "no target"
	if bool(belief.get("visible", false)):
		if int(belief.get("id", -1)) == damage_source:
			return "visible and recently damaged me"
		return "visible, nearest threat"
	if int(belief.get("source", EnemyMemory.Source.NONE)) == EnemyMemory.Source.SOUND:
		return "heard only, last known position"
	return "remembered, %.1fs since contact" % float(belief.get("age", 0.0))


func _evaluate_enemy(
	agent, enemy: EnemyState, world, eye: Vector3, forward: Vector3, dt: float
) -> Dictionary:
	# Lighting acts HERE, on the acquisition range, rather than being
	# reported to the policy: a target standing in shadow simply has to be
	# closer before it resolves at all.
	var effective_range: float = lighting.detection_range(vision_range, enemy.position)
	var evaluation: Dictionary = PerceptionSystem.evaluate_target(
		world,
		eye,
		agent.position,
		forward,
		enemy.position,
		enemy.height,
		fov_deg,
		effective_range
	)
	# Threat accounting uses pure geometry (does the enemy see me?), not the
	# agent's own FOV, and never leaks into the observation as a position.
	if PerceptionSystem.has_line_of_sight(world, enemy.get_eye_position(), agent.position, 1.8):
		threat_count += 1

	var health_norm: float = enemy.health / maxf(enemy.max_health, 0.0001)
	if not enabled:
		# Legacy path: full ground truth, no latency, no memory.
		return {
			"id": enemy.enemy_id,
			"visible": true,
			"in_fov": bool(evaluation["in_fov"]),
			"los_clear": bool(evaluation["los_clear"]),
			"distance": float(evaluation["distance"]),
			"bearing_deg": float(evaluation["bearing_deg"]),
			"elevation_deg": float(evaluation["elevation_deg"]),
			"position": enemy.position,
			"health_norm": health_norm,
			"age": 0.0,
			"confidence": 1.0,
			"source": EnemyMemory.Source.VISUAL,
			"alive": true,
		}

	var enemy_id: int = enemy.enemy_id
	var geometric: bool = bool(evaluation["visible"])
	var contact: float = float(contact_timers.get(enemy_id, 0.0))
	if geometric:
		contact += dt
		loss_timers[enemy_id] = 0.0
	else:
		contact = 0.0
		loss_timers[enemy_id] = float(loss_timers.get(enemy_id, 0.0)) + dt
	contact_timers[enemy_id] = contact

	# Poor light also costs reaction time and shortens how long a lost
	# silhouette keeps being reported.
	var required_contact: float = detection_delay * lighting.detection_delay_scale(enemy.position)
	var grace: float = SandboxConfig.VISUAL_LOSS_GRACE * lighting.loss_grace_scale(enemy.position)
	var confirmed: bool = geometric and contact >= required_contact
	var within_grace: bool = (
		not geometric
		and float(loss_timers.get(enemy_id, 0.0)) <= grace
		and memory.has(enemy_id)
	)
	var visible: bool = confirmed or within_grace

	if confirmed and memory_enabled:
		var direction: Vector3 = enemy.position - agent.position
		direction.y = 0.0
		memory.observe_visual(
			enemy_id,
			enemy.position,
			direction.normalized() if not direction.is_zero_approx() else Vector3.ZERO,
			health_norm
		)

	if visible:
		return {
			"id": enemy_id,
			"visible": true,
			"in_fov": bool(evaluation["in_fov"]),
			"los_clear": bool(evaluation["los_clear"]),
			"distance": float(evaluation["distance"]),
			"bearing_deg": float(evaluation["bearing_deg"]),
			"elevation_deg": float(evaluation["elevation_deg"]),
			"position": enemy.position,
			"health_norm": health_norm,
			"age": 0.0,
			"confidence": 1.0,
			"source": EnemyMemory.Source.VISUAL,
			"alive": true,
		}

	if not memory_enabled or not memory.has(enemy_id):
		return {}

	var track: Dictionary = memory.get_track(enemy_id)
	var remembered: Vector3 = track["position"]
	return {
		"id": enemy_id,
		"visible": false,
		"in_fov": bool(evaluation["in_fov"]),
		"los_clear": false,
		"distance": agent.position.distance_to(remembered),
		"bearing_deg": PerceptionSystem.bearing_deg(forward, agent.position, remembered),
		"elevation_deg": PerceptionSystem.elevation_deg(eye, remembered),
		"position": remembered,
		"health_norm": float(track.get("health_norm", 1.0)),
		"age": float(track["age"]),
		"confidence": float(track["confidence"]),
		"source": int(track["source"]),
		"alive": true,
	}


func _sample_sound(agent, world, sound_bus) -> void:
	heard.clear()
	sound_summary = SoundBus.summarize(heard)
	if not sound_enabled or sound_bus == null:
		return
	heard = (sound_bus as SoundBus).sample(
		agent.position,
		agent.get_forward_horizontal(),
		world,
		-1,
		SandboxConfig.SOUND_DETECTION_DELAY
	)
	sound_summary = SoundBus.summarize(heard)
	if not memory_enabled or heard.is_empty():
		return
	# The loudest event becomes a low-confidence memory track keyed to an
	# UNKNOWN source: hearing tells you something is over there, not who.
	var loudest: Dictionary = heard[0]
	var direction: Vector3 = loudest["direction"]
	var estimate: Vector3 = agent.position + direction * float(loudest["distance"])
	estimate.y = 0.0
	memory.observe_sound(
		SandboxConfig.SOUND_UNKNOWN_SOURCE_ID, estimate, direction, float(loudest["loudness"])
	)


## Current sound snapshot for telemetry / the Control Center SOUND panel.
func sound_snapshot() -> Array:
	return heard.duplicate(true)
