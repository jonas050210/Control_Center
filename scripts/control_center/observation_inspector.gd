## ObservationInspector
##
## Turns the raw numbers a policy actually receives into labelled rows for
## the Control Center's Observation Inspector.
##
## SOURCE OF TRUTH: every label comes from the existing contract
## definitions, never from a second hand-maintained list:
##   * observation rows  -> `Observation.field_names()` / `FIELD_SPEC` and
##                          `Observation.to_array()`
##   * action rows       -> `RLAdapter.action_space_info()` and
##                          `Action.MULTI_DISCRETE_NVECS`
##   * reward components -> `EpisodeState.get_reward_breakdown()`
##
## If the observation contract changes (extra fields, reordering, a new
## tracked enemy), this inspector follows automatically and
## python/tests/test_contract.py fails if the Godot and Python contract
## descriptions disagree.
##
## Everything here is a pure static function over plain data: no Node, no
## Control, no simulation mutation.
class_name ObservationInspector
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")


## One row per observation-vector index:
## {"index": int, "name": String, "group": String, "value": float}
## `observation` may be an Observation, a PackedFloat32Array or an Array.
static func build_rows(observation) -> Array:
	var values: PackedFloat32Array = _to_vector(observation)
	var names: PackedStringArray = Observation.field_names()
	var rows: Array = []
	for index in range(names.size()):
		(
			rows
			. append(
				{
					"index": index,
					"name": names[index],
					"group": Observation.field_group(index),
					"value": float(values[index]) if index < values.size() else 0.0,
				}
			)
		)
	return rows


## Rows grouped by FIELD_SPEC group, preserving contract order:
## [{"group": String, "rows": Array}, ...]
static func build_grouped_rows(observation) -> Array:
	var grouped: Array = []
	var index_by_group: Dictionary = {}
	for row_value in build_rows(observation):
		var row: Dictionary = row_value
		var group: String = str(row["group"])
		if not index_by_group.has(group):
			index_by_group[group] = grouped.size()
			grouped.append({"group": group, "rows": []})
		var bucket: Dictionary = grouped[int(index_by_group[group])]
		(bucket["rows"] as Array).append(row)
	return grouped


## One row per action field, using the adapter's canonical field list:
## {"index", "name", "cardinality", "multidiscrete", "canonical"}
## `canonical` is the -1/0/1 (or bool) value the simulation consumes.
static func build_action_rows(action) -> Array:
	var info: Dictionary = RLAdapter.action_space_info()
	var fields: Array = info.get("fields", [])
	var nvec: Array = info.get("nvec", [])
	var rows: Array = []
	if action == null:
		return rows
	var multidiscrete: Array = action.to_multidiscrete()
	var canonical: Array = action.to_array()
	for index in range(fields.size()):
		(
			rows
			. append(
				{
					"index": index,
					"name": str(fields[index]),
					"cardinality": int(nvec[index]) if index < nvec.size() else 0,
					"multidiscrete":
					int(multidiscrete[index]) if index < multidiscrete.size() else 0,
					"canonical": canonical[index] if index < canonical.size() else 0,
				}
			)
		)
	return rows


## Continuous look-delta values that are logged for human demonstrations but
## are deliberately NOT part of the PPO action space. Returned separately so
## the UI can show them without implying they are policy outputs.
static func build_continuous_action_rows(action) -> Array:
	if action == null:
		return []
	var info: Dictionary = RLAdapter.action_space_info()
	var reserved: Array = info.get("continuous_reserved", [])
	var values: Array = action.to_array()
	var rows: Array = []
	for offset in range(reserved.size()):
		var index: int = Action.MULTI_DISCRETE_SIZE + offset
		(
			rows
			. append(
				{
					"index": index,
					"name": str(reserved[offset]),
					"value": float(values[index]) if index < values.size() else 0.0,
				}
			)
		)
	return rows


## Reward component rows from EpisodeState.get_reward_breakdown(). The
## breakdown dictionary is the reward system's own accounting; this only
## orders and labels it.
static func build_reward_rows(breakdown: Dictionary) -> Array:
	var rows: Array = []
	if breakdown.is_empty():
		return rows
	var ordered: PackedStringArray = PackedStringArray(
		[
			"reward_hits",
			"reward_kills",
			"reward_damage",
			"reward_survive",
			"reward_positioning",
			"reward_aiming",
			"penalty_passivity",
			"penalty_combat_time",
			"penalty_damage",
			"penalty_death",
			"penalty_useless_shot",
			"penalty_missed_shot",
		]
	)
	for key in ordered:
		if breakdown.has(key):
			rows.append({"name": key, "value": float(breakdown[key])})
	# Anything the reward system adds later still shows up, unordered but
	# never silently dropped.
	for key in breakdown.keys():
		var name_key: String = str(key)
		if name_key == "total" or ordered.has(name_key):
			continue
		rows.append({"name": name_key, "value": float(breakdown[key])})
	if breakdown.has("total"):
		rows.append({"name": "total", "value": float(breakdown["total"])})
	return rows


## Monospace-friendly "idx  name  value" lines for a text panel.
static func format_observation_lines(rows: Array, decimals: int = 3) -> PackedStringArray:
	var lines := PackedStringArray()
	for row_value in rows:
		var row: Dictionary = row_value
		(
			lines
			. append(
				(
					"%2d  %-38s %s"
					% [
						int(row["index"]),
						str(row["name"]),
						String.num(float(row["value"]), decimals),
					]
				)
			)
		)
	return lines


static func format_action_lines(rows: Array) -> PackedStringArray:
	var lines := PackedStringArray()
	for row_value in rows:
		var row: Dictionary = row_value
		(
			lines
			. append(
				(
					"%-16s md=%d/%d  value=%s"
					% [
						str(row["name"]),
						int(row["multidiscrete"]),
						maxi(1, int(row["cardinality"])) - 1,
						str(row["canonical"]),
					]
				)
			)
		)
	return lines


## Sanity check used by tests: the inspector must cover exactly the
## contract's field count with no gaps.
static func covers_full_contract() -> bool:
	return Observation.field_names().size() == Observation.FIELD_COUNT


static func _to_vector(observation) -> PackedFloat32Array:
	if observation == null:
		return PackedFloat32Array()
	if observation is PackedFloat32Array:
		return observation
	if observation is Array:
		return PackedFloat32Array(observation)
	if observation.has_method("to_array"):
		return observation.to_array()
	return PackedFloat32Array()
