## ControlCenterConfig
##
## Plain data model for the Control Center: which mode is active, which
## environment/agent is selected, how fast the simulation runs and which
## panels are visible. It holds NO simulation state and performs no work —
## `ControlCenterSession` is the only thing that acts on it, and the UI only
## ever reads/edits these values.
##
## Every setting here maps onto something the existing simulation can
## genuinely do. Settings that cannot be changed without restarting the
## environments are listed in `REBUILD_SETTINGS`, so the UI can mark them
## "requires reset" instead of pretending a live change happened.
class_name ControlCenterConfig
extends RefCounted

## Operating modes. All three drive the SAME EnvironmentCore/Action/
## Observation pipeline; they only differ in who produces the actions and
## how much presentation work is allowed to run.
enum Mode {
	## Headless-style throughput: no views, no telemetry, no logging.
	TRAINING = 0,
	## One rendered environment driven by an in-engine AI controller.
	WATCH = 1,
	## One rendered environment driven by the human input pipeline.
	HUMAN = 2,
}

## Presentation-only cameras. The simulation never reads these.
enum CameraMode {
	FIRST_PERSON = 0,
	THIRD_PERSON = 1,
	FREE = 2,
	TOP_DOWN = 3,
}

## Who produces the Action for the selected environment in WATCH mode.
## EXTERNAL_POLICY is deliberately present but unavailable: Godot has no
## neural-network runtime, trained policies are executed by the Python
## trainer over the JSON-lines bridge (see docs/CONTROL_CENTER.md).
enum PolicySource {
	HEURISTIC = 0,
	IDLE = 1,
	EXTERNAL_POLICY = 2,
}

## Python training workflows exposed by the Control Center. SELF_PLAY is
## visible because the repository has a real two-slot bridge and league, but
## is reported unavailable for optimization until a self-play trainer exists.
enum TrainingType {
	PPO = 0,
	BEHAVIOR_CLONING = 1,
	SELF_PLAY = 2,
}

enum TrainingMode {
	VISUAL = 0,
	HEADLESS = 1,
}

enum TrainingDevice {
	AUTO = 0,
	CPU = 1,
	GPU = 2,
}

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const CurriculumConfig = preload("res://scripts/core/curriculum_config.gd")
const SandboxConfig = preload("res://scripts/core/sandbox_config.gd")

const SPEED_PRESETS: Array = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
const MIN_SPEED: float = 0.05
const MAX_SPEED: float = 16.0
## Hard ceiling on simulation steps executed in one rendered frame while in
## WATCH/HUMAN mode. Prevents a high speed multiplier from freezing the UI.
const MAX_STEPS_PER_FRAME_INTERACTIVE: int = 32
## Milliseconds of each frame TRAINING mode may spend stepping environments.
## Keeps the window responsive while still running far faster than realtime.
const TRAINING_FRAME_BUDGET_MS: float = 8.0
const MAX_ENVIRONMENT_COUNT: int = 64
const MAX_ENEMY_COUNT: int = 12
const MAX_TRAINING_STEPS: int = 2_000_000_000
const MAX_BC_EPOCHS: int = 100_000
const PREFERENCES_PATH: String = "user://control_center.cfg"

## Stable tile identifiers. The presentation uses these rather than node
## names so visibility/order survives a UI refactor.
const TILE_IDS: Array = [
	"simulation", "agent", "inspector", "training", "controls", "logs"
]
const DEFAULT_TILE_ORDER: Array = [
	"simulation", "agent", "training", "inspector", "controls", "logs"
]

## Settings that cannot be applied to running environments and therefore
## need an explicit rebuild/reset. The UI marks them and only applies them
## when "Apply & reset" is pressed.
## NOTE: declared as a plain Array literal, not `PackedStringArray(...)`.
## A built-in constructor call taking an Array argument is not a constant
## expression in Godot 4.7 ("Not a constant expression"), which made this
## whole script fail to compile and cascaded into every Control Center test.
const REBUILD_SETTINGS: Array = ["environment_count", "enemy_count", "seed"]

## Scenario presets. Each one is only a bundle of settings the simulation
## already supports (curriculum level + enemy count) — no scenario-specific
## gameplay code exists or is implied.
const SCENARIOS: Array = [
	{
		"id": "custom",
		"label": "Custom",
		"description": "Use the curriculum level and enemy count set below.",
	},
	{
		"id": "target_practice",
		"label": "Target practice",
		"curriculum_level": CurriculumConfig.Level.STATIONARY_TARGET,
		"enemy_count": 1,
		"description": "Level 1: one stationary target directly ahead.",
	},
	{
		"id": "duel",
		"label": "Duel",
		"curriculum_level": CurriculumConfig.Level.ENEMY_ATTACKS,
		"enemy_count": 1,
		"description": "Level 3: one strafing, attacking enemy.",
	},
	{
		"id": "three_way",
		"label": "Three-way fight",
		"curriculum_level": CurriculumConfig.Level.MULTIPLE_ENEMIES,
		"enemy_count": 3,
		"description": "Level 4: three simultaneous enemies (all observable).",
	},
	{
		"id": "overwhelmed",
		"label": "Overwhelmed",
		"curriculum_level": CurriculumConfig.Level.MULTIPLE_ENEMIES,
		"enemy_count": 6,
		"description": "Level 4 with 6 enemies: more enemies than the "
		+ "observation tracks, so some are invisible to the policy.",
	},
	{
		"id": "cover_fight",
		"label": "Cover fight",
		"curriculum_level": CurriculumConfig.Level.OBSTACLES_COVER,
		"enemy_count": 2,
		"description": "Level 5: staggered cover, ranged enemies that peek and retreat.",
	},
	{
		"id": "corner_fight",
		"label": "Corner fight",
		"curriculum_level": CurriculumConfig.Level.FOV_LOS,
		"enemy_count": 1,
		"description": "Level 6: FOV + occlusion. The enemy starts out of sight around a corner.",
	},
	{
		"id": "sound_only",
		"label": "Sound-only contact",
		"curriculum_level": CurriculumConfig.Level.SOUND,
		"enemy_count": 1,
		"description": "Level 7: the enemy is fully occluded; footsteps and shots are the cue.",
	},
	{
		"id": "lost_target",
		"label": "Target breaks contact",
		"curriculum_level": CurriculumConfig.Level.MEMORY_LOST_TARGETS,
		"enemy_count": 1,
		"description": "Level 8: memory decay. The target leaves your sight line on purpose.",
	},
	{
		"id": "vertical",
		"label": "Vertical encounter",
		"curriculum_level": CurriculumConfig.Level.VERTICAL_COMBAT,
		"enemy_count": 2,
		"description": "Level 9: platforms; jumping changes the available sight lines.",
	},
	{
		"id": "randomized",
		"label": "Randomized arena",
		"curriculum_level": CurriculumConfig.Level.MIXED_RANDOMIZED,
		"enemy_count": 3,
		"description": "Level 10: a new seeded layout and scenario every episode.",
	},
]

var mode: int = Mode.WATCH
var environment_count: int = SandboxConfig.DEFAULT_ENVIRONMENT_COUNT
var enemy_count: int = SandboxConfig.ENEMY_COUNT_DEFAULT
var curriculum_level: int = CurriculumConfig.Level.ENEMY_ATTACKS
var seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
var scenario_id: String = "custom"

## Real Python-backend training configuration. Defaults mirror
## python/sandboxai/config.py and python/sandboxai/cli.py.
var training_type: int = TrainingType.PPO
var training_mode: int = TrainingMode.VISUAL
var training_device: int = TrainingDevice.AUTO
var total_training_steps: int = 1_000_000
var bc_epochs: int = 25
var bc_dataset_path: String = "training/datasets/human_demo.jsonl"
var checkpoint_path: String = ""
var resume_from_checkpoint: bool = false
var learning_rate: float = 0.0003
var rollout_length: int = 2048
var batch_size: int = 256
var gamma: float = 0.99
var gae_lambda: float = 0.95
var entropy_coefficient: float = 0.01
var clip_range: float = 0.2
var checkpoint_frequency: int = 100_000
var evaluation_frequency: int = 50_000
var python_executable: String = "python"
var godot_executable: String = "godot"

var simulation_speed: float = 1.0
var selected_environment: int = 0
## Agent/policy slot inside the selected environment. The canonical
## EnvironmentCore has exactly one controllable agent (slot 0); slot 1 only
## exists in the self-play foundation, which SimulationManager does not
## build yet, so the UI shows it as unavailable instead of faking it.
var selected_agent_slot: int = 0
var policy_source: int = PolicySource.HEURISTIC
var camera_mode: int = CameraMode.FIRST_PERSON

## Panel/overlay visibility (presentation only).
var show_left_panel: bool = true
var show_right_panel: bool = true
var show_bottom_panel: bool = true
var show_perception_overlay: bool = true
var show_reward_components: bool = true
var log_filter: int = -1  # ControlCenterEventLog.FILTER_ALL
var tile_order: Array = DEFAULT_TILE_ORDER.duplicate()
var tile_visibility: Dictionary = {
	"simulation": true,
	"agent": true,
	"inspector": true,
	"training": true,
	"controls": true,
	"logs": true,
}
var left_dock_width: int = 310
var right_dock_width: int = 380
var bottom_dock_height: int = 250


func _init(p_mode: int = Mode.WATCH) -> void:
	mode = clampi(p_mode, Mode.TRAINING, Mode.HUMAN)


static func mode_name(value: int) -> String:
	match value:
		Mode.TRAINING:
			return "TRAINING"
		Mode.WATCH:
			return "WATCH"
		Mode.HUMAN:
			return "HUMAN"
		_:
			return "UNKNOWN"


static func mode_from_name(value: String) -> int:
	match value.strip_edges().to_upper():
		"TRAINING":
			return Mode.TRAINING
		"WATCH":
			return Mode.WATCH
		"HUMAN", "PLAY":
			return Mode.HUMAN
		_:
			return Mode.WATCH


static func training_type_name(value: int) -> String:
	match value:
		TrainingType.PPO:
			return "PPO"
		TrainingType.BEHAVIOR_CLONING:
			return "Behavior Cloning"
		TrainingType.SELF_PLAY:
			return "Self-Play"
		_:
			return "Unknown"


static func training_type_available(value: int) -> bool:
	return value == TrainingType.PPO or value == TrainingType.BEHAVIOR_CLONING


static func training_type_unavailable_reason(value: int) -> String:
	if value == TrainingType.SELF_PLAY:
		return (
			"The two-policy bridge and league are implemented for evaluation, "
			+ "but no self-play optimizer exists yet. Start is disabled rather "
			+ "than launching single-agent PPO under a misleading label."
		)
	return ""


static func training_mode_name(value: int) -> String:
	return "Visual Mode" if value == TrainingMode.VISUAL else "Headless Mode"


static func training_device_argument(value: int) -> String:
	match value:
		TrainingDevice.CPU:
			return "cpu"
		TrainingDevice.GPU:
			return "cuda"
		_:
			return "auto"


static func camera_mode_name(value: int) -> String:
	match value:
		CameraMode.FIRST_PERSON:
			return "First person (agent)"
		CameraMode.THIRD_PERSON:
			return "Third person"
		CameraMode.FREE:
			return "Free camera"
		CameraMode.TOP_DOWN:
			return "Top-down"
		_:
			return "Unknown"


static func policy_source_name(value: int) -> String:
	match value:
		PolicySource.HEURISTIC:
			return "Heuristic AI (AIStubController)"
		PolicySource.IDLE:
			return "Idle (no actions)"
		PolicySource.EXTERNAL_POLICY:
			return "Trained policy (unavailable in-engine)"
		_:
			return "Unknown"


## Whether a policy source can actually drive the simulation right now.
## EXTERNAL_POLICY needs an inference runtime Godot does not have.
static func policy_source_available(value: int) -> bool:
	return value == PolicySource.HEURISTIC or value == PolicySource.IDLE


## True when changing `key` requires rebuilding the environments.
static func requires_rebuild(key: String) -> bool:
	return REBUILD_SETTINGS.has(key)


static func scenario(scenario_identifier: String) -> Dictionary:
	for entry_value in SCENARIOS:
		var entry: Dictionary = entry_value
		if str(entry.get("id", "")) == scenario_identifier:
			return entry
	return {}


## Applies a scenario preset onto this config. Returns the setting keys that
## actually changed, so the caller can decide whether a rebuild is needed.
func apply_scenario(scenario_identifier: String) -> PackedStringArray:
	var changed := PackedStringArray()
	var entry: Dictionary = scenario(scenario_identifier)
	if entry.is_empty():
		return changed
	scenario_id = str(entry.get("id", "custom"))
	if entry.has("curriculum_level"):
		var level: int = int(entry["curriculum_level"])
		if level != curriculum_level:
			curriculum_level = level
			changed.append("curriculum_level")
	if entry.has("enemy_count"):
		var count: int = int(entry["enemy_count"])
		if count != enemy_count:
			enemy_count = count
			changed.append("enemy_count")
	return changed


## Clamps every numeric field into a range the simulation accepts.
func sanitize() -> void:
	mode = clampi(mode, Mode.TRAINING, Mode.HUMAN)
	environment_count = clampi(environment_count, 1, MAX_ENVIRONMENT_COUNT)
	enemy_count = clampi(enemy_count, 1, MAX_ENEMY_COUNT)
	curriculum_level = clampi(
		curriculum_level,
		CurriculumConfig.Level.STATIONARY_TARGET,
		CurriculumConfig.Level.AGENT_VS_AGENT
	)
	seed = maxi(0, seed)
	simulation_speed = clampf(simulation_speed, MIN_SPEED, MAX_SPEED)
	selected_environment = clampi(selected_environment, 0, environment_count - 1)
	selected_agent_slot = maxi(0, selected_agent_slot)
	camera_mode = clampi(camera_mode, CameraMode.FIRST_PERSON, CameraMode.TOP_DOWN)
	policy_source = clampi(policy_source, PolicySource.HEURISTIC, PolicySource.EXTERNAL_POLICY)
	training_type = clampi(training_type, TrainingType.PPO, TrainingType.SELF_PLAY)
	training_mode = clampi(training_mode, TrainingMode.VISUAL, TrainingMode.HEADLESS)
	training_device = clampi(training_device, TrainingDevice.AUTO, TrainingDevice.GPU)
	total_training_steps = clampi(total_training_steps, 1, MAX_TRAINING_STEPS)
	bc_epochs = clampi(bc_epochs, 1, MAX_BC_EPOCHS)
	learning_rate = maxf(0.0000001, learning_rate)
	rollout_length = maxi(1, rollout_length)
	batch_size = clampi(batch_size, 1, rollout_length * environment_count)
	gamma = clampf(gamma, 0.000001, 1.0)
	gae_lambda = clampf(gae_lambda, 0.0, 1.0)
	entropy_coefficient = maxf(0.0, entropy_coefficient)
	clip_range = maxf(0.000001, clip_range)
	checkpoint_frequency = maxi(1, checkpoint_frequency)
	evaluation_frequency = maxi(1, evaluation_frequency)
	python_executable = python_executable.strip_edges()
	if python_executable.is_empty():
		python_executable = "python"
	godot_executable = godot_executable.strip_edges()
	if godot_executable.is_empty():
		godot_executable = "godot"
	left_dock_width = clampi(left_dock_width, 220, 700)
	right_dock_width = clampi(right_dock_width, 280, 800)
	bottom_dock_height = clampi(bottom_dock_height, 140, 600)
	_sanitize_tiles()


func to_dict() -> Dictionary:
	return {
		"mode": mode,
		"mode_name": mode_name(mode),
		"environment_count": environment_count,
		"enemy_count": enemy_count,
		"curriculum_level": curriculum_level,
		"seed": seed,
		"scenario_id": scenario_id,
		"training_type": training_type,
		"training_mode": training_mode,
		"training_device": training_device,
		"total_training_steps": total_training_steps,
		"bc_epochs": bc_epochs,
		"bc_dataset_path": bc_dataset_path,
		"checkpoint_path": checkpoint_path,
		"resume_from_checkpoint": resume_from_checkpoint,
		"learning_rate": learning_rate,
		"rollout_length": rollout_length,
		"batch_size": batch_size,
		"gamma": gamma,
		"gae_lambda": gae_lambda,
		"entropy_coefficient": entropy_coefficient,
		"clip_range": clip_range,
		"checkpoint_frequency": checkpoint_frequency,
		"evaluation_frequency": evaluation_frequency,
		"python_executable": python_executable,
		"godot_executable": godot_executable,
		"simulation_speed": simulation_speed,
		"selected_environment": selected_environment,
		"selected_agent_slot": selected_agent_slot,
		"policy_source": policy_source,
		"camera_mode": camera_mode,
		"show_left_panel": show_left_panel,
		"show_right_panel": show_right_panel,
		"show_bottom_panel": show_bottom_panel,
		"show_perception_overlay": show_perception_overlay,
		"show_reward_components": show_reward_components,
		"log_filter": log_filter,
		"tile_order": tile_order.duplicate(),
		"tile_visibility": tile_visibility.duplicate(),
		"left_dock_width": left_dock_width,
		"right_dock_width": right_dock_width,
		"bottom_dock_height": bottom_dock_height,
	}


func apply_dict(values: Dictionary) -> void:
	mode = int(values.get("mode", mode))
	environment_count = int(values.get("environment_count", environment_count))
	enemy_count = int(values.get("enemy_count", enemy_count))
	curriculum_level = int(values.get("curriculum_level", curriculum_level))
	seed = int(values.get("seed", seed))
	scenario_id = str(values.get("scenario_id", scenario_id))
	training_type = int(values.get("training_type", training_type))
	training_mode = int(values.get("training_mode", training_mode))
	training_device = int(values.get("training_device", training_device))
	total_training_steps = int(values.get("total_training_steps", total_training_steps))
	bc_epochs = int(values.get("bc_epochs", bc_epochs))
	bc_dataset_path = str(values.get("bc_dataset_path", bc_dataset_path))
	checkpoint_path = str(values.get("checkpoint_path", checkpoint_path))
	resume_from_checkpoint = bool(values.get("resume_from_checkpoint", resume_from_checkpoint))
	learning_rate = float(values.get("learning_rate", learning_rate))
	rollout_length = int(values.get("rollout_length", rollout_length))
	batch_size = int(values.get("batch_size", batch_size))
	gamma = float(values.get("gamma", gamma))
	gae_lambda = float(values.get("gae_lambda", gae_lambda))
	entropy_coefficient = float(values.get("entropy_coefficient", entropy_coefficient))
	clip_range = float(values.get("clip_range", clip_range))
	checkpoint_frequency = int(values.get("checkpoint_frequency", checkpoint_frequency))
	evaluation_frequency = int(values.get("evaluation_frequency", evaluation_frequency))
	python_executable = str(values.get("python_executable", python_executable))
	godot_executable = str(values.get("godot_executable", godot_executable))
	simulation_speed = float(values.get("simulation_speed", simulation_speed))
	selected_environment = int(values.get("selected_environment", selected_environment))
	selected_agent_slot = int(values.get("selected_agent_slot", selected_agent_slot))
	policy_source = int(values.get("policy_source", policy_source))
	camera_mode = int(values.get("camera_mode", camera_mode))
	show_left_panel = bool(values.get("show_left_panel", show_left_panel))
	show_right_panel = bool(values.get("show_right_panel", show_right_panel))
	show_bottom_panel = bool(values.get("show_bottom_panel", show_bottom_panel))
	show_perception_overlay = bool(values.get("show_perception_overlay", show_perception_overlay))
	show_reward_components = bool(values.get("show_reward_components", show_reward_components))
	log_filter = int(values.get("log_filter", log_filter))
	tile_order = (values.get("tile_order", tile_order) as Array).duplicate()
	tile_visibility = (values.get("tile_visibility", tile_visibility) as Dictionary).duplicate()
	left_dock_width = int(values.get("left_dock_width", left_dock_width))
	right_dock_width = int(values.get("right_dock_width", right_dock_width))
	bottom_dock_height = int(values.get("bottom_dock_height", bottom_dock_height))
	sanitize()


func set_tile_visible(tile_id: String, visible: bool) -> bool:
	if not TILE_IDS.has(tile_id):
		return false
	tile_visibility[tile_id] = visible
	match tile_id:
		"agent":
			show_left_panel = visible
		"inspector":
			show_right_panel = visible
		"logs":
			show_bottom_panel = visible
	return true


func is_tile_visible(tile_id: String) -> bool:
	return bool(tile_visibility.get(tile_id, true))


func move_tile(tile_id: String, direction: int) -> bool:
	var index: int = tile_order.find(tile_id)
	if index < 0:
		return false
	var destination: int = clampi(index + signi(direction), 0, tile_order.size() - 1)
	if destination == index:
		return false
	var swap_value = tile_order[destination]
	tile_order[destination] = tile_order[index]
	tile_order[index] = swap_value
	return true


func save_preferences(path: String = PREFERENCES_PATH) -> bool:
	var file := ConfigFile.new()
	file.set_value("control_center", "configuration", to_dict())
	return file.save(path) == OK


func load_preferences(path: String = PREFERENCES_PATH) -> bool:
	var file := ConfigFile.new()
	if file.load(path) != OK:
		return false
	var values = file.get_value("control_center", "configuration", {})
	if not (values is Dictionary):
		return false
	apply_dict(values)
	return true


func _sanitize_tiles() -> void:
	var cleaned: Array = []
	for tile_value in tile_order:
		var tile_id: String = str(tile_value)
		if TILE_IDS.has(tile_id) and not cleaned.has(tile_id):
			cleaned.append(tile_id)
	for tile_id in TILE_IDS:
		if not cleaned.has(tile_id):
			cleaned.append(tile_id)
	tile_order = cleaned
	for tile_id in TILE_IDS:
		if not tile_visibility.has(tile_id):
			tile_visibility[tile_id] = true
	set_tile_visible("agent", show_left_panel)
	set_tile_visible("inspector", show_right_panel)
	set_tile_visible("logs", show_bottom_panel)


func duplicate_config() -> ControlCenterConfig:
	var copy: ControlCenterConfig = (load("res://scripts/control_center/control_center_config.gd")
		as GDScript).new()
	copy.apply_dict(to_dict())
	return copy
