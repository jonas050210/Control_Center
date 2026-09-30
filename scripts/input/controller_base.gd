## ControllerBase
##
## Common interface implemented by both the human debug controller and any
## AI/agent controller. SimulationManager only ever talks to this interface,
## which is exactly what "human mode uses the same underlying action/state
## system as the AI" means in practice: both sides just produce an `Action`
## for a given `EnvironmentCore` every tick.
##
## Extends Node (rather than RefCounted) purely so concrete controllers can
## receive engine input callbacks (`_input`, `_unhandled_input`, `_process`)
## when they need to read keyboard/mouse state.
class_name ControllerBase
extends Node

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const Action = preload("res://scripts/core/action.gd")
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")


## Returns the Action to apply to `env` for the current tick. The default
## implementation is a safe no-op so a controller can be attached without
## overriding anything yet.
func get_action(_env: EnvironmentCore) -> Action:
	return Action.idle()
