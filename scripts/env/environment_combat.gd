## EnvironmentCombat
##
## Resolution of the agent's trigger pull for `EnvironmentCore`: hitscan,
## hit zones, the near-miss/useless-shot classification, occlusion and the
## death bookkeeping that follows a kill.
##
## Split out of `EnvironmentCore` along the same seam as `EnvironmentReset`
## and `EnvironmentIntrospection`: this is one self-contained question -
## "what did that shot do?" - answered from the agent, the enemies and the
## geometry, and nothing outside this file needs the intermediate steps of
## the answer. `EnvironmentCore` keeps two-line wrappers so the simulation
## loop still reads as a sequence of named sub-steps.
##
## The classification the loop depends on is unchanged and deliberate. A
## fired shot is exactly one of:
##   * hit          - exact ray/sphere hit, geometry tested along the ray;
##   * missed_shot  - near miss at a clear, in-range, live target;
##   * useless_shot - cooldown, no target, out of range, blocked, spray.
## Near misses stay cheap while the policy learns fine aim; wall shots and
## trigger spam must stay strongly negative.
##
## `env` is intentionally untyped: typing it would require preloading
## `environment_core.gd`, which preloads this file.
class_name EnvironmentCombat
extends RefCounted

const Action = preload("res://scripts/core/action.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")
const WeaponState = preload("res://scripts/weapon/weapon_state.gd")


## Cosine of the angle between where the agent looks and where the target
## is. Zero for a dead or absent target, which makes the aiming-delta term
## in the reward collapse to nothing rather than to a stale bearing.
static func target_alignment(env, target: EnemyState) -> float:
	if target == null or not target.is_targetable():
		return 0.0
	var direction: Vector3 = (
		(target.get_chest_position() - env.agent.get_eye_position()).normalized()
	)
	return env.agent.get_forward_vector().dot(direction)


## Resolves the agent's trigger pull.
##
## The center-screen crosshair and this function both use
## AgentState.get_eye_position() + AgentState.get_forward_vector(): the red
## dot is the exact hitscan ray, not an approximate UI marker. A fired shot
## is classified into three distinct cases:
##   * hit: exact ray/sphere hit, with geometry tested along the same ray;
##   * missed_shot: near-miss at a clear, in-range live target;
##   * useless_shot: cooldown/no target/out of range/blocked/random spray.
##
## That distinction matters for PPO: near misses should be cheap while the
## policy learns fine aim, but wall shots and trigger spam must be strongly
## negative.
static func resolve_agent_shot(env, action: Action, sound_on: bool) -> Dictionary:
	var result: Dictionary = {
		"hit": false,
		"kill": false,
		"useless_shot": false,
		"missed_shot": false,
		"trigger_discipline": false,
		"headshot": false,
		"shot_fired": false,
		"shot_result": "none",
		"damage_dealt": 0.0,
		"projectiles_fired": 0,
		"projectiles_hit": 0,
		"spread_deg": 0.0,
	}
	if not env.agent.alive:
		# Still consume the trigger edge so a dead-then-respawned agent does
		# not inherit a stale "already held" state.
		env.agent.weapon.trigger_held = false
		return result

	var trigger: Dictionary = env.agent.weapon.pull_trigger(
		action.shoot, env.agent.speed_fraction, not env.agent.on_ground
	)
	if not bool(trigger["fired"]):
		_classify_blocked_trigger(env, str(trigger["blocked_reason"]), result)
		return result
	result["shot_fired"] = true
	result["spread_deg"] = float(trigger["spread_deg"])
	if sound_on:
		env.sound_bus.emit_sound(SoundBus.Category.SHOT, env.agent.position, env.AGENT_SOUND_SOURCE)

	var any_alive: bool = false
	var plausible_target: bool = false
	var eye: Vector3 = env.agent.get_eye_position()
	var forward: Vector3 = env.agent.get_forward_vector()

	# First classify whether the trigger pull was aimed near a real target.
	# This uses the INTENDED aim ray, never the bloom-perturbed one: a shot
	# is judged on where the agent pointed, not on where the cone happened
	# to throw the bullet. Pellet profiles likewise use the center ray, so
	# spread can still land a close hit but cannot turn random spray into a
	# cheap near-miss reward.
	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		if not enemy.is_targetable():
			continue
		any_alive = true
		if _shot_is_near_live_target(env, eye, forward, enemy):
			plausible_target = true

	# Bloom deviates the whole pattern; the pellet pattern is then built
	# around the deviated axis, so a shotgun keeps its shape while moving.
	var aim_dir: Vector3 = env.agent.weapon.apply_spread(
		forward, float(trigger["spread_deg"]), int(trigger["shot_index"])
	)
	var head_zones: bool = env.curriculum.hit_zones_enabled()
	var impact_sources: Dictionary = {}
	var projectile_dirs: Array = env.agent.weapon.projectile_directions(aim_dir)
	result["projectiles_fired"] = projectile_dirs.size()
	for projectile_dir_value in projectile_dirs:
		var projectile_dir: Vector3 = projectile_dir_value
		var best_hit_enemy: EnemyState = null
		var best_hit_distance: float = INF
		var best_zone: String = WeaponState.ZONE_NONE
		var best_multiplier: float = 1.0
		for enemy_value in env.enemies:
			var enemy: EnemyState = enemy_value
			if not enemy.is_targetable():
				continue
			var zone_hit: Dictionary = env.agent.weapon.resolve_hit_zone(
				eye,
				projectile_dir,
				enemy.get_chest_position(),
				enemy.get_head_position(),
				head_zones
			)
			var hit_dist: float = float(zone_hit["distance"])
			if hit_dist < 0.0 or hit_dist >= best_hit_distance:
				continue
			if _weapon_ray_blocked_before(env, eye, projectile_dir, hit_dist):
				continue
			best_hit_distance = hit_dist
			best_hit_enemy = enemy
			best_zone = str(zone_hit["zone"])
			best_multiplier = float(zone_hit["multiplier"])
		if best_hit_enemy == null:
			continue
		var applied: float = best_hit_enemy.take_damage(
			env.agent.weapon.projectile_damage_at_distance(best_hit_distance) * best_multiplier
		)
		if applied <= 0.0:
			continue
		result["hit"] = true
		if best_zone == WeaponState.ZONE_HEAD:
			result["headshot"] = true
		result["projectiles_hit"] = int(result["projectiles_hit"]) + 1
		result["shot_result"] = "hit"
		result["damage_dealt"] = float(result["damage_dealt"]) + applied
		env.episode.record_damage_dealt(applied)
		if sound_on and not impact_sources.has(best_hit_enemy.enemy_id):
			env.sound_bus.emit_sound(
				SoundBus.Category.IMPACT, best_hit_enemy.position, best_hit_enemy.enemy_id
			)
		impact_sources[best_hit_enemy.enemy_id] = true
		if not best_hit_enemy.alive:
			result["kill"] = true
			env.episode.record_kill()
			_on_enemy_died(env, best_hit_enemy, sound_on)

	# A genuine miss requires a plausible target close to the true aim ray.
	# Shooting into empty space, through a wall, out of range or nowhere near a
	# target is useless-shot spam and receives the larger penalty.
	result["missed_shot"] = any_alive and plausible_target and not bool(result["hit"])
	result["useless_shot"] = (not any_alive) or (not bool(result["hit"]) and not plausible_target)
	if bool(result["missed_shot"]):
		result["shot_result"] = "near_miss"
	elif bool(result["useless_shot"]):
		result["shot_result"] = "useless_spam" if any_alive else "useless_no_target"
	env.episode.record_shot(bool(result["hit"]), bool(result["headshot"]))
	# The kick lands AFTER the shot is resolved: recoil disturbs the NEXT
	# shot, never the one that produced it. That ordering is what makes
	# recoil something the policy learns to pre-compensate.
	env.agent.apply_recoil(float(trigger["recoil_pitch_deg"]), float(trigger["recoil_yaw_deg"]))
	return result


## Turns a `WeaponState.pull_trigger()` refusal into the reward-facing shot
## classification.
##
## Legacy (handling off) keeps the original contract exactly: any trigger
## pull on a cycling weapon is a `useless_shot` worth PENALTY_USELESS_SHOT.
##
## With handling on, holding the trigger on an automatic weapon is CORRECT
## play, so the cycling/reloading/empty cases move to the much cheaper
## `trigger_discipline` term. Waste is punished organically instead: bloom
## widens, the magazine drains and the reload leaves the agent exposed.
static func _classify_blocked_trigger(env, reason: String, result: Dictionary) -> void:
	match reason:
		"released":
			return
		"cycling":
			result["shot_result"] = "cooldown"
		"reloading":
			result["shot_result"] = "reloading"
		"empty":
			result["shot_result"] = "reload_started"
		"needs_release":
			result["shot_result"] = "needs_release"
		_:
			result["shot_result"] = "cooldown"
	if env.agent.weapon.handling_enabled:
		result["trigger_discipline"] = true
	else:
		result["useless_shot"] = true


## True when the primary target is a valid aim-shaping target: alive, within
## weapon range and not geometrically hidden. It intentionally does NOT
## require the crosshair to already be inside the near-miss cone — this is
## what lets the agent receive bounded reward while turning toward a real,
## shootable threat, but not while staring at a wall or a stale memory.
static func target_is_hittable(env, target: EnemyState) -> bool:
	if target == null or not target.is_targetable() or not env.agent.alive:
		return false
	var eye: Vector3 = env.agent.get_eye_position()
	var chest: Vector3 = target.get_chest_position()
	if eye.distance_to(chest) > env.agent.weapon.range_m + env.agent.weapon.hit_radius:
		return false
	if env.world != null and env.world.segment_blocked(eye, chest):
		return false
	return true


## Near-miss classifier only. It never changes whether a shot hits; it only
## decides whether an exact miss was a meaningful aiming attempt or spam.
static func _shot_is_near_live_target(
	env, eye: Vector3, forward: Vector3, enemy: EnemyState
) -> bool:
	if enemy == null or not enemy.is_targetable() or forward.is_zero_approx():
		return false
	var chest: Vector3 = enemy.get_chest_position()
	var to_target: Vector3 = chest - eye
	var distance: float = to_target.length()
	if distance <= 0.0001 or distance > env.agent.weapon.range_m + env.agent.weapon.hit_radius:
		return false
	var dir: Vector3 = forward.normalized()
	var target_dir: Vector3 = to_target / distance
	var angle_deg: float = rad_to_deg(acos(clampf(dir.dot(target_dir), -1.0, 1.0)))
	if angle_deg > SandboxConfig.WEAPON_NEAR_MISS_CONE_DEG:
		return false
	return not _weapon_ray_blocked_before(env, eye, dir, minf(distance, env.agent.weapon.range_m))


## Whether sight-blocking geometry intersects the exact weapon ray before a
## target distance. This is the same occlusion test used by hit resolution.
static func _weapon_ray_blocked_before(
	env, eye: Vector3, forward: Vector3, distance: float
) -> bool:
	if env.world == null or distance <= 0.0:
		return false
	var wall_distance: float = env.world.ray_hit_distance(
		eye, forward, minf(distance, env.agent.weapon.range_m)
	)
	return wall_distance >= 0.0 and wall_distance + 0.001 < distance


## Death bookkeeping. A corpse must stop being a target, stop being
## perceived and stop being remembered by anyone, immediately.
static func _on_enemy_died(env, enemy: EnemyState, sound_on: bool) -> void:
	enemy.mark_dead(env.episode.step_count * SandboxConfig.SIMULATION_DT)
	if sound_on:
		env.sound_bus.emit_sound(SoundBus.Category.DEATH, enemy.death_position, enemy.enemy_id)
	env.perception.forget(enemy.enemy_id)
	if env.last_damage_source == enemy.enemy_id:
		env.last_damage_source = -1
