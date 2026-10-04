## RemotePolicyController
##
## A ControllerBase whose actions come from a trained checkpoint running in
## Python. Every tick it sends the environment's current 126-value
## observation - built by exactly the same code as in training - and decodes
## the reply through RLAdapter._resolve_action, the decoder the headless
## bridge uses. The viewer therefore shows the policy acting on precisely the
## observation/action contract it was trained on; only rendering is added.
class_name RemotePolicyController
extends ControllerBase

## ControllerBase provides the shared Action and EnvironmentCore dependencies.
## Keep the controller-specific dependencies explicit for standalone/headless parsing.
const ControllerBase = preload("res://scripts/input/controller_base.gd")
const RemotePolicyClient = preload("res://scripts/viewer/remote_policy_client.gd")
const RLAdapter = preload("res://scripts/rl/rl_adapter.gd")

var client: RemotePolicyClient
## The last raw MultiDiscrete action the policy returned (HUD only).
var last_action_values: Array = []
## Total number of answered action requests (HUD only).
var requests_answered: int = 0


func _init(p_client: RemotePolicyClient = null) -> void:
	client = p_client


func is_connected_to_policy() -> bool:
	return client != null and client.connected


func get_action(env: EnvironmentCore) -> Action:
	if env.is_done() or not is_connected_to_policy():
		return Action.idle()
	var observation: Array = RLAdapter._observation_to_array(env.get_observations())
	var reply: Dictionary = client.request({"type": "act", "obs": observation})
	if not reply.has("action"):
		return Action.idle()
	var values = reply["action"]
	if values is Array:
		last_action_values = values
	requests_answered += 1
	return RLAdapter._resolve_action(values)
