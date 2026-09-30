## EnvironmentEnemies
##
## The per-tick opponent update for `EnvironmentCore`: advancing every
## enemy, applying the damage they deal to the agent, and turning their
## motion events into audible sounds.
##
## Split out along the same seam as `EnvironmentReset`,
## `EnvironmentIntrospection` and `EnvironmentCombat`. This half of the
## step answers "what did the opponents do?", the combat half answers
## "what did the agent's shot do", and neither needs the other's
## intermediate state.
##
## Two performance contracts live here and must survive any edit:
##
## * the `EnemyBrain` context is a REUSED dictionary on the environment,
##   not a fresh literal per tick - 64 parallel environments at 60 Hz
##   would otherwise allocate roughly 230k short-lived dictionaries per
##   simulated second;
## * the A* navigation graph is baked LAZILY, only once an enemy has
##   actually accumulated stuck time. Tactical episodes where everyone
##   walks straight never pay for a graph.
##
## `env` is intentionally untyped: typing it would require preloading
## `environment_core.gd`, which preloads this file.
class_name EnvironmentEnemies
extends RefCounted

const EnemyBrain = preload("res://scripts/enemy/enemy_brain.gd")
const EnemyState = preload("res://scripts/enemy/enemy_state.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SoundBus = preload("res://scripts/perception/sound_bus.gd")


## Advances every enemy and returns the total damage applied to the agent.
static func update_enemies(env, dt: float, sound_on: bool) -> float:
	var damage_taken: float = 0.0
	var tactical: bool = env.curriculum.tactical_enemies_enabled()
	# The brain context is a REUSED member dictionary rather than a fresh
	# literal per step: with 64 parallel environments at 60 Hz this would
	# otherwise allocate ~230k short-lived dictionaries per simulated
	# second, all of which the GC then has to sweep.
	var context: Dictionary = env._brain_context
	if tactical:
		context["world"] = env.world
		# Preserve the navigation layer's exception-path contract: baking the
		# A* grid for every tactical episode up front made the first step pay
		# the full graph cost even when every enemy moved directly forever.
		# A null graph still lets NavigationAgent accumulate stuck_time; the
		# per-enemy loop below bakes once only after that evidence exists.
		context["navigation"] = env.navigation
		context["lighting"] = env.lighting
		context["sound_bus"] = env.sound_bus if sound_on else null
		context["rng"] = env.rng
		context["dt"] = dt
		context["arena_half_extent"] = env.arena_half_extent
		context["agent_position"] = env.agent.position
		context["agent_eye"] = env.agent.get_eye_position()
		context["agent_height"] = env.agent.height
		context["agent_alive"] = env.agent.alive
		context["allow_movement"] = env.curriculum.enemy_movement_enabled()
		context["allow_attack"] = env.curriculum.enemy_attacks_enabled()
		context["allow_ranged"] = env.curriculum.ranged_enemies_enabled()
		context["allow_strafe"] = env.curriculum.strafing_enabled()
		context["allow_jump"] = env.curriculum.vertical_enabled()

	for enemy_value in env.enemies:
		var enemy: EnemyState = enemy_value
		if not enemy.is_targetable():
			continue
		var damage: float = 0.0
		if tactical:
			# NavigationAgent increments stuck_time even with a null graph. On
			# the following tick this supplies the shared graph and planning can
			# begin. Once baked, later enemies reuse the same graph.
			if (
				env.navigation == null
				and enemy.navigation.stuck_time >= SandboxConfig.NAV_STUCK_TIME
			):
				context["navigation"] = env._ensure_navigation()
			var brain_events: Dictionary = EnemyBrain.update(enemy, context)
			damage = float(brain_events["damage"])
			if sound_on:
				if bool(brain_events["shot"]):
					env.sound_bus.emit_sound(SoundBus.Category.SHOT, enemy.position, enemy.enemy_id)
				emit_motion_sounds(env, brain_events, enemy.position, enemy.enemy_id)
		else:
			damage = enemy.update_ai(
				dt,
				env.agent.position,
				env.arena_half_extent,
				env.curriculum.enemy_movement_enabled(),
				env.curriculum.enemy_attacks_enabled(),
				env.curriculum.strafing_enabled()
			)
		if damage > 0.0:
			var applied: float = env.agent.take_damage(damage)
			if applied > 0.0:
				damage_taken += applied
				env.last_damage_source = enemy.enemy_id
				if sound_on:
					env.sound_bus.emit_sound(
						SoundBus.Category.IMPACT, env.agent.position, env.AGENT_SOUND_SOURCE
					)
	return damage_taken


## Turns a CharacterMotor/EnemyBrain motion event dictionary into sounds.
static func emit_motion_sounds(env, motion: Dictionary, position: Vector3, source_id: int) -> void:
	if bool(motion.get("footstep", false)):
		env.sound_bus.emit_sound(SoundBus.Category.FOOTSTEP, position, source_id)
	if bool(motion.get("jumped", false)):
		env.sound_bus.emit_sound(SoundBus.Category.JUMP, position, source_id)
	if bool(motion.get("landed", false)):
		env.sound_bus.emit_sound(SoundBus.Category.LAND, position, source_id)
