## SelfPlayConfig
## Configuration for the two-policy combat hook. Slot 1 may point at a frozen
## checkpoint while slot 0 continues learning; seeds are intentionally
## independent so mirrored matches do not share RNG state.
class_name SelfPlayConfig
extends RefCounted

var slot_a_checkpoint: String = ""
var slot_b_checkpoint: String = ""
var slot_a_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED
var slot_b_seed: int = SandboxConfig.DEFAULT_RANDOM_SEED + 1000003
var environment_count: int = 1


func to_dict() -> Dictionary:
	return {
		"slot_a_checkpoint": slot_a_checkpoint,
		"slot_b_checkpoint": slot_b_checkpoint,
		"slot_a_seed": slot_a_seed,
		"slot_b_seed": slot_b_seed,
		"environment_count": environment_count,
	}
