extends Node
## Local newline-delimited JSON bridge for the controlled sandbox.

var enabled: bool = false
var port: int = 11008
var obs_width: int = 84
var obs_height: int = 84
var max_steps: int = 500
var server: TCPServer = TCPServer.new()
var peer: StreamPeerTCP
var receive_buffer: String = ""
var controller: SandboxAIController

func _ready() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	enabled = args.has("--rl-server")
	if not enabled:
		return
	for argument: String in args:
		if argument.begins_with("--rl-port="): port = int(argument.get_slice("=", 1))
		elif argument.begins_with("--obs-width="): obs_width = maxi(1, int(argument.get_slice("=", 1)))
		elif argument.begins_with("--obs-height="): obs_height = maxi(1, int(argument.get_slice("=", 1)))
		elif argument.begins_with("--max-steps="): max_steps = maxi(1, int(argument.get_slice("=", 1)))
	var error: Error = server.listen(port, "127.0.0.1")
	if error != OK:
		push_error("Could not start SandboxAI bridge on port %d: %s" % [port, error_string(error)])
		get_tree().quit(2)
		return
	print("SandboxAI RL bridge listening on 127.0.0.1:%d" % port)

func _process(_delta: float) -> void:
	if not enabled: return
	_resolve_controller()
	if peer == null and server.is_connection_available(): peer = server.take_connection()
	if peer == null: return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		peer = null
		receive_buffer = ""
		return
	var available: int = peer.get_available_bytes()
	if available > 0: receive_buffer += peer.get_utf8_string(available)
	while receive_buffer.contains("\n"):
		var line: String = receive_buffer.get_slice("\n", 0)
		receive_buffer = receive_buffer.substr(line.length() + 1)
		if not line.strip_edges().is_empty(): _handle_line(line)

func _resolve_controller() -> void:
	if controller != null: return
	var scene: Node = get_tree().current_scene
	if scene == null: return
	controller = scene.find_child("AIController3D", true, false) as SandboxAIController
	if controller:
		controller.max_steps = max_steps
		controller.set_agent_mode(true)

func _handle_line(line: String) -> void:
	var request: Variant = JSON.parse_string(line)
	if not request is Dictionary:
		_send({"ok": false, "error": "request must be a JSON object"})
		return
	var command: String = str(request.get("command", ""))
	if command == "hello":
		_send({"ok": true, "protocol": 1, "action_dims": [3,3,2,2,2,2,2,2,21,21]})
		return
	if controller == null:
		_send({"ok": false, "error": "arena controller is not ready"})
		return
	if command == "reset":
		controller.reset(int(request.get("seed", 42)))
		_send_state(0.0)
	elif command == "step":
		var action: Variant = request.get("action", [])
		if not action is Array or action.size() < 10:
			_send({"ok": false, "error": "action must contain 10 integers"})
			return
		controller.set_action(action)
		_send_state(controller.consume_reward())
	elif command == "close":
		_send({"ok": true})
		peer.disconnect_from_host()
	else: _send({"ok": false, "error": "unknown command: " + command})

func _send_state(step_reward: float) -> void:
	var image: Image = controller.get_observation_image()
	if image == null or image.is_empty():
		_send({"ok": false, "error": "observation image is unavailable"})
		return
	image.resize(obs_width, obs_height, Image.INTERPOLATE_BILINEAR)
	var encoded: String = Marshalls.raw_to_base64(image.save_png_to_buffer())
	_send({"ok": true, "observation_png": encoded, "reward": step_reward, "terminated": controller.terminated, "truncated": controller.truncated, "info": controller.get_state_info()})

func _send(payload: Dictionary) -> void:
	if peer == null: return
	peer.put_data((JSON.stringify(payload) + "\n").to_utf8_buffer())
