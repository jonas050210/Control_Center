extends Node
class_name SandboxScenarioDirector

@export var scenario_id: String = "mixed_encounter"
var rng: RandomNumberGenerator = RandomNumberGenerator.new()
var seed_value: int = 42
var active_scenario: String = ""
var scenario_names: Array[String] = ["doorway_ambush", "crossing_target", "cover_transition", "close_range", "long_range", "multi_target", "flank", "defensive_hold"]

func configure(seed: int, requested: String = "") -> void:
	seed_value = seed; rng.seed = seed
	active_scenario = requested if requested != "" else scenario_names[rng.randi_range(0, scenario_names.size() - 1)]
	apply_scenario()

func apply_scenario() -> void:
	var enemies: Array[Node] = get_tree().get_nodes_in_group("targets")
	for index: int in range(enemies.size()):
		var enemy: SandboxEnemy = enemies[index] as SandboxEnemy
		if enemy == null: continue
		enemy.configure_seed(seed_value + index * 977)
		match active_scenario:
			"doorway_ambush": enemy.behavior = "stationary"
			"crossing_target": enemy.behavior = "patrol"
			"cover_transition": enemy.behavior = "cover"
			"close_range": enemy.behavior = "aggressive"
			"long_range": enemy.behavior = "stationary"
			"multi_target": enemy.behavior = "patrol" if index % 2 == 0 else "stationary"
			"flank": enemy.behavior = "aggressive" if index == 0 else "patrol"
			"defensive_hold": enemy.behavior = "stationary"
		if active_scenario == "close_range": enemy.global_position = Vector3(0, 0.9, -5.0 + index * 2.0)
		elif active_scenario == "long_range": enemy.global_position = Vector3(-10.0 + index * 8.0, 0.9, -15.0)

func state() -> Dictionary:
	return {"scenario": active_scenario, "seed": seed_value, "enemy_count": get_tree().get_nodes_in_group("targets").size()}
