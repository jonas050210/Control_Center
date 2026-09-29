## Self-play foundation tests: two independent policy/action slots,
## world geometry parity, perception gating, sound events, and match metrics.
class_name TestSelfPlay
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const Observation = preload("res://scripts/core/observation.gd")
const Obstacle = preload("res://scripts/world/obstacle.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")
const SelfPlayAdapter = preload("res://scripts/rl/self_play_adapter.gd")
const SelfPlayEnvironmentCore = preload("res://scripts/self_play/self_play_environment.gd")


func test_self_play_reset_has_two_observations() -> SandboxTest:
	var t := SandboxTest.new("self_play_reset_has_two_observations")
	var match_env := SelfPlayEnvironmentCore.new()
	var observations: Array = match_env.reset(10, 20)
	t.assert_eq(observations.size(), 2)
	t.assert_eq(observations[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(observations[1].to_array().size(), Observation.FIELD_COUNT)
	return t


func test_self_play_adapter_reset_serializes_both_contract_vectors() -> SandboxTest:
	var t := SandboxTest.new("self_play_adapter_reset_serializes_both_contract_vectors")
	var adapter := SelfPlayAdapter.new(1, 10)
	var batch: Array = adapter.reset(777)
	t.assert_eq(batch.size(), 1)
	t.assert_eq(batch[0].size(), 2)
	t.assert_eq(batch[0][0].size(), Observation.FIELD_COUNT)
	t.assert_eq(batch[0][1].size(), Observation.FIELD_COUNT)
	return t


func test_self_play_step_returns_per_agent_rewards_and_infos() -> SandboxTest:
	var t := SandboxTest.new("self_play_step_returns_per_agent_metrics")
	var match_env := SelfPlayEnvironmentCore.new()
	match_env.reset(1, 2)
	var result: Dictionary = match_env.step([Action.idle(), Action.idle()])
	t.assert_eq(result.rewards.size(), 2)
	t.assert_eq(result.infos.size(), 2)
	t.assert_true(result.infos[0].metrics.has("kills"))
	t.assert_true(result.infos[1].metrics.has("kills"))
	return t


## Geometry between two agents must stop a hitscan shot.
##
## The occluder is placed EXPLICITLY instead of trusting whatever the seeded
## "corner" generator happened to build: the L-wall's arms, flips and offsets
## are randomized, so for most seeds the diagonal between the two spawn
## points is wide open and the test proved nothing (or failed outright). The
## clear-world case is kept as the positive control, so "no damage" can only
## mean "occluded", never "the ray missed".
func test_self_play_with_world_blocks_weapon_ray_occlusion() -> SandboxTest:
	var t := SandboxTest.new("self_play_with_world_blocks_weapon_ray_occlusion")
	var clear_result: Dictionary = _fire_across_arena(false)
	t.assert_true(bool(clear_result["hit"]), "an unobstructed shot must connect")
	t.assert_gt(float(clear_result["damage"]), 0.0)

	var blocked_result: Dictionary = _fire_across_arena(true)
	t.assert_false(bool(blocked_result["hit"]), "a wall between the agents must stop the shot")
	t.assert_eq(blocked_result["damage"], 0.0)
	t.assert_eq(blocked_result["health_b"], blocked_result["max_health_b"])
	return t


## Fires one shot from agent A at agent B across the arena diagonal, with or
## without a sight-blocking wall halfway between them. Returns the shot
## outcome for slot 0 plus agent B's health.
static func _fire_across_arena(with_occluder: bool) -> Dictionary:
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	env.reset(42, 42)

	# Deterministic geometry: only the wall we place ourselves is between the
	# two agents.
	env.world.clear()
	if with_occluder:
		env.world.add_box(Vector3.ZERO, Vector3(2.0, 2.0, 2.0), Obstacle.Kind.WALL)

	env.agent_a.reset(Vector3(-4.0, 0.0, -4.0), 0.0)
	env.agent_b.reset(Vector3(4.0, 0.0, 4.0), 180.0)
	env._sync_proxies()

	# AgentState stores aim as yaw/pitch; it has no writable `forward`.
	env.agent_a.set_forward_horizontal(env.agent_b.position - env.agent_a.position)

	var shoot_action := Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)
	var res: Dictionary = env.step([shoot_action, Action.idle()])
	var events: Dictionary = res.infos[0].events
	return {
		"hit": bool(events["hit"]),
		"damage": float(events["damage_dealt"]),
		"shot_fired": bool(events["shot_fired"]),
		"health_b": env.agent_b.health,
		"max_health_b": env.agent_b.max_health,
	}


## Perception must be gated by geometry, not by ground truth: an agent behind
## a wall is neither visible nor LOS-clear, and the same agent in the open is.
func test_self_play_perception_gating_detects_visible_and_hidden() -> SandboxTest:
	var t := SandboxTest.new("self_play_perception_gating_detects_visible_and_hidden")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.FOV_LOS)
	env.reset(100, 100)

	# Deterministic geometry again: one wall, placed by us, halfway between
	# the two agents.
	env.world.clear()
	env.world.add_box(Vector3.ZERO, Vector3(2.0, 2.0, 2.0), Obstacle.Kind.WALL)

	env.agent_a.reset(Vector3(-4.0, 0.0, -4.0), 0.0)
	env.agent_b.reset(Vector3(4.0, 0.0, 4.0), 180.0)
	env.agent_a.set_forward_horizontal(env.agent_b.position - env.agent_a.position)
	env._sync_proxies()
	# Perception is refreshed by a step; reading observations straight after
	# repositioning would report the beliefs computed during reset().
	for _index in range(_visual_confirmation_ticks()):
		env.step([Action.idle(), Action.idle()])

	var obs: Array = env.get_observations()
	t.assert_false(obs[0].primary_enemy_visible, "a wall must hide agent B")
	t.assert_false(obs[0].primary_enemy_los_clear)

	# Same bearing, but now in the open: agent B moves to the near side of
	# the wall, straight ahead of agent A.
	env.agent_b.reset(Vector3(-4.0, 0.0, 4.0), 180.0)
	env.agent_a.set_forward_horizontal(Vector3(0.0, 0.0, 1.0))
	env._sync_proxies()
	# Visual contact is confirmed only after AGENT_VISUAL_DETECTION_DELAY of
	# uninterrupted sight (perception has reaction latency by design), so a
	# single tick is not enough to report the target.
	for _index in range(_visual_confirmation_ticks()):
		env.step([Action.idle(), Action.idle()])

	var obs_los: Array = env.get_observations()
	t.assert_true(obs_los[0].primary_enemy_los_clear, "an unobstructed enemy must be LOS-clear")
	t.assert_true(obs_los[0].primary_enemy_visible)
	return t


## Ticks of uninterrupted sight needed before perception confirms a target,
## plus one so the comparison is strictly satisfied.
static func _visual_confirmation_ticks() -> int:
	return (
		int(ceil(SandboxConfig.AGENT_VISUAL_DETECTION_DELAY / SandboxConfig.SIMULATION_DT)) + 1
	)


## A shot is audible to the other agent, after (and only after) the sound
## model's detection delay has elapsed.
func test_self_play_sound_emission_and_hearing() -> SandboxTest:
	var t := SandboxTest.new("self_play_sound_emission_and_hearing")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("open_arena")
	env.set_curriculum_level(CurriculumConfig.Level.SOUND)
	env.reset(1, 2)

	# Place agents near each other.
	env.agent_a.reset(Vector3(0.0, 0.0, -2.0), 0.0)
	env.agent_b.reset(Vector3(0.0, 0.0, 2.0), 180.0)
	env._sync_proxies()

	# Agent A fires a shot.
	var shoot_action := Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)
	env.step([shoot_action, Action.idle()])
	t.assert_true(
		env.perception_b.heard.is_empty(),
		"SOUND_DETECTION_DELAY means a shot is not registered on its own tick"
	)

	# Sounds are consciously registered once they are older than
	# SandboxConfig.SOUND_DETECTION_DELAY (a handful of ticks at 60 Hz).
	var delay_ticks: int = int(
		ceil(SandboxConfig.SOUND_DETECTION_DELAY / SandboxConfig.SIMULATION_DT)
	)
	for _index in range(delay_ticks):
		env.step([Action.idle(), Action.idle()])

	var heard_b: Array = env.perception_b.heard
	t.assert_false(heard_b.is_empty(), "agent B must hear agent A's shot")
	if not heard_b.is_empty():
		t.assert_eq(heard_b[0].category, 3)  # SHOT category
	t.assert_true(
		env.perception_a.heard.is_empty(), "an agent never hears its own weapon"
	)
	return t


func test_self_play_map_and_lighting_configuration() -> SandboxTest:
	var t := SandboxTest.new("self_play_map_and_lighting_configuration")
	var env := SelfPlayEnvironmentCore.new()
	t.assert_true(env.set_map("two_rooms"))
	t.assert_true(env.set_lighting_mode("low_light"))
	t.assert_false(env.set_map("non_existent_map_id_12345"))
	t.assert_false(env.set_lighting_mode("invalid_lighting_mode_xyz"))

	var cond: Dictionary = env.get_episode_condition()
	t.assert_eq(cond.map_id, "two_rooms")
	t.assert_eq(cond.lighting, "low_light")
	return t


## Regression test for the reset-path lighting resolution. The reset()
## lighting branches used to call `LightingProfile.mode(...)`, an instance
## VARIABLE, through the script class — a Godot COMPILE error that
## invalidated the whole self-play environment script, made
## `SelfPlayEnvironmentCore.new()` return null, and turned every self-play
## reset over the bridge into a silent `observations: []`. These branches
## (map default lighting, explicit lighting override with and without a map,
## and the world-less branch) were previously never executed by any test.
func test_self_play_reset_resolves_lighting_in_all_branches() -> SandboxTest:
	var t := SandboxTest.new("self_play_reset_resolves_lighting_in_all_branches")

	# Map set, no explicit lighting: the map's declared lighting must apply.
	var with_map := SelfPlayEnvironmentCore.new()
	with_map.set_map("two_rooms")
	var obs_map: Array = with_map.reset(31, 32)
	t.assert_eq(obs_map.size(), 2)
	t.assert_eq(obs_map[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(obs_map[1].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(
		LightingProfile.mode_id(with_map.lighting.mode), "normal",
		"the map's declared lighting must resolve"
	)

	# Map set plus explicit override: the override wins.
	var overridden := SelfPlayEnvironmentCore.new()
	overridden.set_map("two_rooms")
	overridden.set_lighting_mode("low_light")
	var obs_override: Array = overridden.reset(41, 42)
	t.assert_eq(obs_override[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(LightingProfile.mode_id(overridden.lighting.mode), "low_light")

	# No map (generated layout) plus explicit override.
	var layout_only := SelfPlayEnvironmentCore.new()
	layout_only.set_lighting_mode("night")
	var obs_layout: Array = layout_only.reset(51, 52)
	t.assert_eq(obs_layout[0].to_array().size(), Observation.FIELD_COUNT)
	t.assert_eq(LightingProfile.mode_id(layout_only.lighting.mode), "night")
	return t


func test_self_play_independent_rng_and_observations() -> SandboxTest:
	var t := SandboxTest.new("self_play_independent_rng_and_observations")
	var env := SelfPlayEnvironmentCore.new()
	env.reset(12345, 67890)
	t.assert_ne(env.rng_a.seed, env.rng_b.seed)
	var obs: Array = env.get_observations()
	t.assert_eq(obs.size(), 2)
	t.assert_ne(obs[0].to_array(), obs[1].to_array())
	return t


func test_self_play_zero_information_leakage_when_occluded() -> SandboxTest:
	var t := SandboxTest.new("self_play_zero_information_leakage_when_occluded")
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("corner")
	env.set_curriculum_level(CurriculumConfig.Level.FOV_LOS)
	env.reset(555, 777)

	# Place A and B on opposite sides of the corner
	env.agent_a.reset(Vector3(-5.0, 0.0, -5.0), 0.0)
	env.agent_b.reset(Vector3(5.0, 0.0, 5.0), 180.0)
	env._sync_proxies()

	# Perception update with no sounds
	env.perception_a.reset()
	env.perception_b.reset()
	var beliefs_a: Array = env.perception_a.update(
		env.agent_a, [env.proxy_b], env.world, env.sound_bus, 0.0, 0
	)
	var beliefs_b: Array = env.perception_b.update(
		env.agent_b, [env.proxy_a], env.world, env.sound_bus, 0.0, 1
	)
	env._beliefs_a = beliefs_a
	env._beliefs_b = beliefs_b

	var obs: Array = env.get_observations()
	var obs_a: Observation = obs[0]
	var obs_b: Observation = obs[1]

	# Both agents must have 0 visibility / 0 clear line of sight
	t.assert_false(obs_a.primary_enemy_visible)
	t.assert_false(obs_a.primary_enemy_los_clear)
	t.assert_false(obs_b.primary_enemy_visible)
	t.assert_false(obs_b.primary_enemy_los_clear)

	# Enemy health / speed / distance must not leak ground truth
	# The primary target's health field is `enemy_health_norm` on the
	# Observation object; `primary_enemy_health_norm` is only its name in
	# Observation.FIELD_SPEC (the flat-vector layout).
	t.assert_eq(obs_a.enemy_health_norm, 0.0)
	t.assert_eq(obs_b.enemy_health_norm, 0.0)
	return t


func test_self_play_deterministic_replay_trajectory() -> SandboxTest:
	var t := SandboxTest.new("self_play_deterministic_replay_trajectory")
	var env1 := SelfPlayEnvironmentCore.new()
	env1.set_layout("open_arena")
	env1.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	var obs1: Array = env1.reset(42, 42 + 1000003)

	var env2 := SelfPlayEnvironmentCore.new()
	env2.set_layout("open_arena")
	env2.set_curriculum_level(CurriculumConfig.Level.OBSTACLES_COVER)
	var obs2: Array = env2.reset(42, 42 + 1000003)

	t.assert_eq(obs1[0].to_array(), obs2[0].to_array())
	t.assert_eq(obs1[1].to_array(), obs2[1].to_array())

	var step_actions := [
		[Action.new(1, 0, 0, 0, false), Action.new(-1, 0, 0, 0, false)],
		[Action.new(1, 0, 0, 0, true), Action.new(0, 1, 0, 0, false)],
		[Action.new(0, 0, 1, 0, false), Action.new(0, -1, 0, 0, true)],
	]

	for act_pair in step_actions:
		var r1: Dictionary = env1.step(act_pair)
		var r2: Dictionary = env2.step(act_pair)
		t.assert_eq(r1.rewards, r2.rewards)
		t.assert_eq(r1.done, r2.done)
		t.assert_eq(r1.observations[0].to_array(), r2.observations[0].to_array())
		t.assert_eq(r1.observations[1].to_array(), r2.observations[1].to_array())

	return t


## Simultaneous lethal fire must kill BOTH agents.
##
## Regression test for a slot-order bias: fire used to be resolved slot A
## first, mutating agent B's health before agent B's own trigger was
## evaluated (and gating that trigger on `agent_b.alive`). A tie therefore
## always scored as an agent_a_win, which is structurally unfair in
## self-play, where one policy plays both slots and the win rate is read
## back as skill.
func test_self_play_simultaneous_lethal_exchange_kills_both() -> SandboxTest:
	var t := SandboxTest.new("self_play_simultaneous_lethal_exchange_kills_both")
	var res: Dictionary = _simultaneous_duel()

	t.assert_false(bool(res["alive_a"]), "slot A must not survive a lethal exchange")
	t.assert_false(bool(res["alive_b"]), "slot B must not survive a lethal exchange")
	t.assert_true(bool(res["events_a"]["kill"]), "slot A scored the lethal hit")
	t.assert_true(bool(res["events_b"]["kill"]), "slot B scored the lethal hit")
	t.assert_true(bool(res["events_a"]["died"]))
	t.assert_true(bool(res["events_b"]["died"]))
	t.assert_eq(res["done_reason"], "draw")
	t.assert_true(bool(res["done"]))
	return t


## The same exchange must be identical under a slot swap: with mirrored
## inputs the two slots see mirrored events and receive equal reward.
func test_self_play_lethal_exchange_is_slot_symmetric() -> SandboxTest:
	var t := SandboxTest.new("self_play_lethal_exchange_is_slot_symmetric")
	var res: Dictionary = _simultaneous_duel()
	var events_a: Dictionary = res["events_a"]
	var events_b: Dictionary = res["events_b"]

	t.assert_eq(res["reward_a"], res["reward_b"], "mirrored slots must earn equal reward")
	t.assert_eq(events_a["hit"], events_b["hit"])
	t.assert_eq(events_a["kill"], events_b["kill"])
	t.assert_eq(events_a["shot_result"], events_b["shot_result"])
	t.assert_eq(events_a["damage_dealt"], events_b["damage_dealt"])
	t.assert_eq(events_a["damage_taken"], events_b["damage_taken"])
	t.assert_eq(res["kills_a"], res["kills_b"])
	t.assert_eq(res["deaths_a"], res["deaths_b"])
	return t


## Damage bookkeeping must stay clamped: a volley can never drain more than
## the target's remaining health, and dealt damage must equal taken damage
## on the opposite slot.
func test_self_play_simultaneous_damage_is_clamped_and_mirrored() -> SandboxTest:
	var t := SandboxTest.new("self_play_simultaneous_damage_is_clamped_and_mirrored")
	var res: Dictionary = _simultaneous_duel(1.0)
	var events_a: Dictionary = res["events_a"]
	var events_b: Dictionary = res["events_b"]

	t.assert_eq(float(events_a["damage_dealt"]), 1.0, "damage is clamped to remaining health")
	t.assert_eq(float(events_b["damage_dealt"]), 1.0)
	t.assert_eq(events_a["damage_dealt"], events_b["damage_taken"])
	t.assert_eq(events_b["damage_dealt"], events_a["damage_taken"])
	t.assert_eq(res["health_a"], 0.0)
	t.assert_eq(res["health_b"], 0.0)
	return t


## A non-lethal simultaneous exchange must still land both shots: the fix is
## about resolution order, not about making every trade lethal.
func test_self_play_simultaneous_nonlethal_exchange_damages_both() -> SandboxTest:
	var t := SandboxTest.new("self_play_simultaneous_nonlethal_exchange_damages_both")
	var res: Dictionary = _simultaneous_duel(-1.0)
	var events_a: Dictionary = res["events_a"]
	var events_b: Dictionary = res["events_b"]

	t.assert_true(bool(events_a["hit"]))
	t.assert_true(bool(events_b["hit"]))
	t.assert_false(bool(events_a["kill"]))
	t.assert_false(bool(events_b["kill"]))
	t.assert_true(bool(res["alive_a"]))
	t.assert_true(bool(res["alive_b"]))
	t.assert_gt(float(events_a["damage_dealt"]), 0.0)
	t.assert_eq(events_a["damage_dealt"], events_b["damage_dealt"])
	t.assert_false(bool(res["done"]))
	return t


## Mirrored duel on a cleared arena: both agents face each other at equal
## distance and both pull the trigger on the same tick. `health` <= 0.0
## keeps full health (non-lethal trade); a positive value is assigned to
## both agents to make a single hit lethal. The default curriculum level
## keeps weapon handling (spread, bloom, recoil, magazines) OFF so the
## hitscan outcome is exact; symmetry under full handling is asserted
## separately.
static func _simultaneous_duel(
	health: float = 1.0, level: int = CurriculumConfig.Level.STATIONARY_TARGET
) -> Dictionary:
	var env := SelfPlayEnvironmentCore.new()
	env.set_layout("open_arena")
	env.set_curriculum_level(level)
	env.reset(7, 7)
	# Low ladder levels run without a world at all; higher ones must be
	# emptied so only the mirrored geometry below decides the outcome.
	if env.world != null:
		env.world.clear()

	env.agent_a.reset(Vector3(-2.0, 0.0, 0.0), 0.0)
	env.agent_b.reset(Vector3(2.0, 0.0, 0.0), 0.0)
	if health > 0.0:
		env.agent_a.health = health
		env.agent_b.health = health
	env.agent_a.set_forward_horizontal(env.agent_b.position - env.agent_a.position)
	env.agent_b.set_forward_horizontal(env.agent_a.position - env.agent_b.position)
	env._sync_proxies()

	var shoot := Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)
	var res: Dictionary = env.step([shoot, Action.new(0, 0, 0, 0, true, Vector2.ZERO, false)])
	var metrics_a: Dictionary = res.infos[0].metrics
	var metrics_b: Dictionary = res.infos[1].metrics
	return {
		"events_a": res.infos[0].events,
		"events_b": res.infos[1].events,
		"reward_a": res.rewards[0],
		"reward_b": res.rewards[1],
		"done": bool(res.done),
		"done_reason": str(res.infos[0].done_reason),
		"alive_a": env.agent_a.alive,
		"alive_b": env.agent_b.alive,
		"health_a": env.agent_a.health,
		"health_b": env.agent_b.health,
		"kills_a": metrics_a["kills"],
		"kills_b": metrics_b["kills"],
		"deaths_a": metrics_a["deaths"],
		"deaths_b": metrics_b["deaths"],
	}


## Symmetry must survive full weapon handling (spread, bloom, recoil,
## magazines): mirrored slots consume the same deterministic spread index,
## so whatever the outcome, both slots must observe the same one.
func test_self_play_simultaneous_fire_symmetric_with_weapon_handling() -> SandboxTest:
	var t := SandboxTest.new("self_play_simultaneous_fire_symmetric_with_weapon_handling")
	var res: Dictionary = _simultaneous_duel(1.0, CurriculumConfig.Level.AGENT_VS_AGENT)
	var events_a: Dictionary = res["events_a"]
	var events_b: Dictionary = res["events_b"]

	t.assert_eq(events_a["shot_fired"], events_b["shot_fired"])
	t.assert_eq(events_a["hit"], events_b["hit"])
	t.assert_eq(events_a["kill"], events_b["kill"])
	t.assert_eq(events_a["damage_dealt"], events_b["damage_dealt"])
	t.assert_eq(events_a["shot_result"], events_b["shot_result"])
	t.assert_eq(res["reward_a"], res["reward_b"])
	t.assert_eq(res["alive_a"], res["alive_b"])
	t.assert_ne(res["done_reason"], "agent_a_win", "a mirrored tick cannot favour slot A")
	t.assert_ne(res["done_reason"], "agent_b_win", "a mirrored tick cannot favour slot B")
	return t
