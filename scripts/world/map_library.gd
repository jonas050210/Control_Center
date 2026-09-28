## MapLibrary
##
## The authored-map catalog. A *map* is one level above a *layout*: a
## layout (`WorldGenerator`) is raw geometry, while a map binds that
## geometry to default lighting, an arena size, procedural variation
## parameters and human-facing metadata.
##
## Information boundary — this is the whole reason the class exists:
##
##   * The SIMULATION receives `world` (geometry) and `lighting`
##     (a `LightingProfile`). Both are things a character physically
##     interacts with.
##   * The POLICY receives NEITHER the map id nor the metadata. A map id in
##     the observation would be a free strategic shortcut ("map 7 means the
##     enemy comes from the left"), which is exactly the memorization this
##     project is trying to avoid.
##   * The CONTROL CENTER and any offline report receive `metadata()`,
##     which is labels, tags and prose for humans.
##
## Every entry is seedable: `resolve(map_id, seed)` is a pure function, and
## `variants` lets one authored map cover a family of layouts so training
## sees the same *concept* (a corner fight, a dark corridor) in different
## geometry.
class_name MapLibrary
extends RefCounted

const ArenaWorld = preload("res://scripts/world/arena_world.gd")
const LightingProfile = preload("res://scripts/perception/lighting_profile.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")
const WorldGenerator = preload("res://scripts/world/world_generator.gd")

## One entry per authored map.
##
## Keys:
##   id            stable string id (persisted in configs and telemetry)
##   label         human-readable name (Control Center only)
##   variants      layout ids this map may generate; one is chosen by seed
##   lighting      default lighting mode id (see LightingProfile.MODE_IDS)
##   lighting_pool optional alternative lighting ids sampled by seed
##   half_extent   arena half extent in meters
##   tags          human/curriculum tags; never observed
##   ambience      number of environmental noise emitters placed in the map
##   description   prose for the Control Center map panel
const MAPS: Array = [
	{
		"id": "open_field",
		"label": "Open Field",
		"variants": ["open_arena"],
		"lighting": "normal",
		"lighting_pool": [],
		"half_extent": 10.0,
		"tags": ["open", "aim", "no_cover"],
		"ambience": 0,
		"description": "Empty arena. Pure aiming and movement; nothing to hide behind.",
	},
	{
		"id": "training_yard",
		"label": "Training Yard",
		"variants": ["scattered_cover"],
		"lighting": "normal",
		"lighting_pool": [],
		"half_extent": 10.0,
		"tags": ["open", "cover", "introductory"],
		"ambience": 0,
		"description": "A few crates in an otherwise open yard. First contact with occlusion.",
	},
	{
		"id": "blind_corner",
		"label": "Blind Corner",
		"variants": ["corner"],
		"lighting": "normal",
		"lighting_pool": ["normal", "low_light"],
		"half_extent": 10.0,
		"tags": ["corner", "occlusion", "peek"],
		"ambience": 0,
		"description": "An L-shaped junction. Sight lines break and reform as you move.",
	},
	{
		"id": "cover_field",
		"label": "Cover Field",
		"variants": ["cover_field"],
		"lighting": "normal",
		"lighting_pool": [],
		"half_extent": 11.0,
		"tags": ["cover", "positioning"],
		"ambience": 0,
		"description": "Staggered low and high cover. Rewards choosing which line to hold.",
	},
	{
		"id": "long_corridor",
		"label": "Long Corridor",
		"variants": ["corridor"],
		"lighting": "low_light",
		"lighting_pool": ["low_light", "normal"],
		"half_extent": 10.0,
		"tags": ["corridor", "one_lane", "low_light"],
		"ambience": 2,
		"description": "A single dim lane with one side opening. Nowhere to go but forward.",
	},
	{
		"id": "two_rooms",
		"label": "Two Rooms",
		"variants": ["rooms"],
		"lighting": "normal",
		"lighting_pool": [],
		"half_extent": 10.0,
		"tags": ["rooms", "doorway", "memory"],
		"ambience": 2,
		"description": "Two rooms joined by a doorway plus an alcove to wait in.",
	},
	{
		"id": "pillar_hall",
		"label": "Pillar Hall",
		"variants": ["pillars"],
		"lighting": "high_contrast",
		"lighting_pool": ["high_contrast", "normal"],
		"half_extent": 11.0,
		"tags": ["pillars", "narrow_sightlines", "contrast"],
		"ambience": 1,
		"description": "A grid of pillars under harsh light. Many sight lines, all thin.",
	},
	{
		"id": "catwalks",
		"label": "Catwalks",
		"variants": ["vertical"],
		"lighting": "normal",
		"lighting_pool": [],
		"half_extent": 10.0,
		"tags": ["vertical", "platforms", "elevation"],
		"ambience": 2,
		"description": "Standable platforms at two heights. Elevation changes who sees whom.",
	},
	{
		"id": "compound",
		"label": "Compound",
		"variants": ["multi_room"],
		"lighting": "mixed",
		"lighting_pool": ["mixed", "normal"],
		"half_extent": 12.0,
		"tags": ["multi_room", "navigation", "mixed_light"],
		"ambience": 3,
		"description": "Four rooms behind doorways with uneven lighting. Navigation matters.",
	},
	{
		"id": "ambush_alley",
		"label": "Ambush Alley",
		"variants": ["ambush"],
		"lighting": "low_light",
		"lighting_pool": ["low_light", "night"],
		"half_extent": 11.0,
		"tags": ["ambush", "alcoves", "corner_check"],
		"ambience": 2,
		"description": "A lane flanked by blind alcoves. Whoever walks it first is exposed.",
	},
	{
		"id": "echo_maze",
		"label": "Echo Maze",
		"variants": ["sound_maze"],
		"lighting": "night",
		"lighting_pool": ["night"],
		"half_extent": 11.0,
		"tags": ["sound", "night", "occlusion"],
		"ambience": 4,
		"description": "Staggered stubs at night. Almost nothing is visible; sound carries.",
	},
	{
		"id": "night_yard",
		"label": "Night Yard",
		"variants": ["scattered_cover", "cover_field"],
		"lighting": "night",
		"lighting_pool": ["night", "low_light"],
		"half_extent": 10.0,
		"tags": ["night", "cover", "sound"],
		"ambience": 2,
		"description": "A familiar yard, after dark. The same geometry, far less information.",
	},
	{
		"id": "foggy_field",
		"label": "Foggy Field",
		"variants": ["open_arena", "scattered_cover"],
		"lighting": "fog",
		"lighting_pool": ["fog"],
		"half_extent": 12.0,
		"tags": ["fog", "distance_limited"],
		"ambience": 1,
		"description": "Open ground under heavy haze. Range, not geometry, is the limit.",
	},
	{
		"id": "random_ops",
		"label": "Randomized Operations",
		"variants": ["randomized"],
		"lighting": "normal",
		"lighting_pool": ["normal", "low_light", "night", "fog", "high_contrast", "mixed"],
		"half_extent": 11.0,
		"tags": ["randomized", "generalization"],
		"ambience": 2,
		"description": "A new layout and new lighting every episode. The generalization set.",
	},
]


static func ids() -> PackedStringArray:
	var out: PackedStringArray = PackedStringArray()
	for entry_value in MAPS:
		out.append(str((entry_value as Dictionary)["id"]))
	return out


static func has_map(map_id: String) -> bool:
	return not definition(map_id).is_empty()


## Raw catalog entry, or {} when the id is unknown.
static func definition(map_id: String) -> Dictionary:
	for entry_value in MAPS:
		var entry: Dictionary = entry_value
		if str(entry["id"]) == map_id:
			return entry
	return {}


## Human-facing description of a map. Control Center / reports only — the
## simulation never reads this and the policy never sees it.
static func metadata(map_id: String) -> Dictionary:
	var entry: Dictionary = definition(map_id)
	if entry.is_empty():
		return {"id": map_id, "label": map_id, "known": false, "tags": [], "description": ""}
	return {
		"id": entry["id"],
		"label": entry["label"],
		"known": true,
		"tags": (entry["tags"] as Array).duplicate(),
		"description": entry["description"],
		"variants": (entry["variants"] as Array).duplicate(),
		"default_lighting": entry["lighting"],
		"lighting_pool": (entry["lighting_pool"] as Array).duplicate(),
		"half_extent": entry["half_extent"],
		"ambience": int(entry.get("ambience", 0)),
	}


## Every map's metadata, for the Control Center map browser.
static func catalog() -> Array:
	var out: Array = []
	for entry_value in MAPS:
		out.append(metadata(str((entry_value as Dictionary)["id"])))
	return out


## Maps carrying every one of `tags`. Used by the training-side map sampler
## to build "the same concept in different geometry" distributions.
static func ids_with_tags(tags: Array) -> PackedStringArray:
	var out: PackedStringArray = PackedStringArray()
	for entry_value in MAPS:
		var entry: Dictionary = entry_value
		var entry_tags: Array = entry["tags"]
		var matches: bool = true
		for tag_value in tags:
			if not entry_tags.has(tag_value):
				matches = false
				break
		if matches:
			out.append(str(entry["id"]))
	return out


## Instantiates a map.
##
## Returns:
##   {map_id, label, layout_id, seed, world, lighting, half_extent,
##    metadata}
##
## `world` and `lighting` are the only two entries the simulation consumes.
## Unknown ids fall back to `open_field` rather than failing, matching
## `WorldGenerator.build()`'s degrade-don't-crash policy.
static func resolve(map_id: String, seed_value: int = 0) -> Dictionary:
	var entry: Dictionary = definition(map_id)
	if entry.is_empty():
		entry = definition("open_field")

	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value

	var variants: Array = entry["variants"]
	var layout_id: String = str(variants[rng.randi_range(0, variants.size() - 1)])

	var pool: Array = entry["lighting_pool"]
	var lighting_id: String = str(entry["lighting"])
	if pool.size() > 0:
		lighting_id = str(pool[rng.randi_range(0, pool.size() - 1)])

	var half_extent: float = float(entry["half_extent"])
	# The layout seed is derived from the map seed so that changing the map
	# id changes the geometry even when the episode seed is identical.
	var layout_seed: int = rng.randi()
	var world: ArenaWorld = WorldGenerator.build(
		layout_id, layout_seed, half_extent, SandboxConfig.ARENA_WALL_HEIGHT
	)
	var lighting: LightingProfile = LightingProfile.from_id(lighting_id, rng.randi())

	# Environmental noise emitters. Fixed for the episode and placed from
	# the same seeded stream as the geometry, so ambience replays exactly.
	var ambient_sources: Array = []
	for _i in range(int(entry.get("ambience", 0))):
		ambient_sources.append(
			world.sample_free_position(rng, SandboxConfig.ENEMY_RADIUS, SandboxConfig.AGENT_HEIGHT)
		)

	return {
		"map_id": str(entry["id"]),
		"label": str(entry["label"]),
		"layout_id": layout_id,
		"layout_seed": layout_seed,
		"seed": seed_value,
		"world": world,
		"lighting": lighting,
		"lighting_id": lighting_id,
		"half_extent": half_extent,
		"ambient_sources": ambient_sources,
		"metadata": metadata(str(entry["id"])),
	}
