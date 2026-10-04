## Tests for the 3D checkpoint viewer (scripts/viewer/): its pure helpers,
## the HUD text, a headless boot of the viewer scene with the built-in
## heuristic, and the TCP policy link (framing + action decoding) against an
## in-process TCPServer standing in for python/sandboxai/viewer.py.
class_name TestViewer
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const EnvironmentCore = preload("res://scripts/env/environment_core.gd")
const Observation = preload("res://scripts/core/observation.gd")
const RemotePolicyClient = preload("res://scripts/viewer/remote_policy_client.gd")
const RemotePolicyController = preload("res://scripts/viewer/remote_policy_controller.gd")
const SimulationManager = preload("res://scripts/core/simulation_manager.gd")
const ViewerCameraRig = preload("res://scripts/viewer/viewer_camera_rig.gd")
const ViewerHud = preload("res://scripts/viewer/viewer_hud.gd")
const ViewerMain = preload("res://scripts/viewer/viewer_main.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")

const READ_TIMEOUT_MS: int = 3000


func test_user_args_are_parsed_as_key_value_pairs() -> SandboxTest:
	var t := SandboxTest.new("viewer_user_args_are_parsed_as_key_value_pairs")
	var parsed: Dictionary = ViewerMain.parse_user_args(
		PackedStringArray(["--map", "compound", "--level", "7", "--orphan", "--seed", "3"])
	)
	t.assert_eq(parsed.get("map"), "compound")
	t.assert_eq(parsed.get("level"), "7")
	t.assert_eq(parsed.get("seed"), "3")
	t.assert_false(parsed.has("orphan"), "a flag without a value is ignored")
	return t


func test_speed_and_map_cycling_helpers() -> SandboxTest:
	var t := SandboxTest.new("viewer_speed_and_map_cycling_helpers")
	t.assert_eq(ViewerMain.SPEEDS[ViewerMain.speed_index_for(1.0)], 1.0)
	t.assert_eq(ViewerMain.SPEEDS[ViewerMain.speed_index_for(3.1)], 4.0)
	t.assert_eq(ViewerMain.SPEEDS[ViewerMain.speed_index_for(100.0)], 8.0)
	# -1 is the generated arena; the cycle visits it between the last and first map.
	t.assert_eq(ViewerMain.next_map_index(-1, 1, 3), 0)
	t.assert_eq(ViewerMain.next_map_index(2, 1, 3), -1)
	t.assert_eq(ViewerMain.next_map_index(-1, -1, 3), 2)
	t.assert_eq(ViewerMain.next_map_index(0, -1, 3), -1)
	t.assert_eq(ViewerMain.next_map_index(5, 1, 0), -1)
	return t


func test_episode_outcome_classification() -> SandboxTest:
	var t := SandboxTest.new("viewer_episode_outcome_classification")
	t.assert_eq(ViewerMain.outcome_of({"win": true}), "win")
	t.assert_eq(ViewerMain.outcome_of({"win": false, "loss": true}), "loss")
	t.assert_eq(ViewerMain.outcome_of({"truncated": true}), "timeout")
	return t


func test_hud_describes_actions_and_results() -> SandboxTest:
	var t := SandboxTest.new("viewer_hud_describes_actions_and_results")
	t.assert_eq(ViewerHud.describe_action([1, 1, 1, 1, 0, 0]), "idle")
	t.assert_eq(ViewerHud.describe_action([2, 1, 1, 1, 1, 0]), "forward + SHOOT")
	t.assert_eq(ViewerHud.describe_action([0, 2, 0, 1, 0, 1]), "back + right + turn L + JUMP")
	t.assert_eq(ViewerHud.describe_action([]), "-")
	t.assert_eq(ViewerHud.format_results([]), "-")
	t.assert_eq(ViewerHud.format_results(["win", "loss", "timeout"]), "W L T")
	var many: Array = []
	for _index in range(12):
		many.append("win")
	t.assert_eq(ViewerHud.format_results(many).split(" ").size(), ViewerHud.RESULT_HISTORY)
	return t


func test_hud_status_lines_report_policy_and_disconnects() -> SandboxTest:
	var t := SandboxTest.new("viewer_hud_status_lines_report_policy_and_disconnects")
	var lines: PackedStringArray = ViewerHud.format_status_lines(
		{"policy": "best_eval.zip", "connected": false, "error": "timeout", "map": "compound"}
	)
	var text: String = "\n".join(lines)
	t.assert_true(text.contains("best_eval.zip"), "the policy label is shown")
	t.assert_true(text.contains("disconnected"), "a broken policy link is shown")
	t.assert_true(text.contains("compound"), "the map is shown")
	return t


func test_viewer_scene_boots_steps_and_switches_maps_offline() -> SandboxTest:
	var t := SandboxTest.new("viewer_scene_boots_steps_and_switches_maps_offline")
	var packed: PackedScene = load("res://scenes/viewer.tscn") as PackedScene
	t.assert_not_null(packed, "scenes/viewer.tscn should load")
	if packed == null:
		return t
	var loop: SceneTree = Engine.get_main_loop() as SceneTree
	var viewer: ViewerMain = packed.instantiate() as ViewerMain
	t.assert_not_null(viewer, "the viewer scene root is a ViewerMain")
	if viewer == null or loop == null:
		return t
	loop.root.add_child(viewer)  # _ready(): no --policy-port, so the heuristic plays

	t.assert_not_null(viewer.simulation_manager, "a SimulationManager is built")
	t.assert_not_null(viewer.get_node_or_null("HeuristicController"), "offline controller")
	t.assert_not_null(viewer.camera_rig, "a camera rig exists")
	t.assert_eq(viewer.simulation_manager.environments.size(), 1)
	t.assert_false(viewer.simulation_manager.views.is_empty(), "the environment is rendered")

	for _index in range(30):
		viewer._advance_one_step()
	t.assert_eq(viewer.total_steps, 30, "every advance is one simulation tick")

	var env: EnvironmentCore = viewer.simulation_manager.environments[0]
	viewer._change_map(1)
	t.assert_eq(
		str(env.get_episode_condition().get("map_id")),
		viewer.map_ids[0],
		"M selects the first authored map"
	)
	t.assert_eq(viewer.episode_step, 0, "a map change starts a fresh episode")
	viewer._change_map(-1)
	t.assert_eq(viewer.map_index, -1, "Shift+M goes back to the generated arena")

	for mode in range(ViewerCameraRig.MODE_NAMES.size()):
		viewer.camera_rig.set_mode(mode)
		viewer._process(1.0 / 60.0)
	t.assert_eq(viewer.camera_rig.mode, ViewerCameraRig.Mode.FIRST_PERSON)

	loop.root.remove_child(viewer)
	viewer.free()
	return t


func test_remote_policy_link_round_trip() -> SandboxTest:
	var t := SandboxTest.new("viewer_remote_policy_link_round_trip")
	var server := TCPServer.new()
	t.assert_eq(server.listen(0, "127.0.0.1"), OK, "a local test server can listen")
	var client := RemotePolicyClient.new()
	t.assert_true(client.connect_to("127.0.0.1", server.get_local_port()), "client connects")
	var peer: StreamPeerTCP = _accept(server)
	t.assert_not_null(peer, "the server accepts the viewer")
	if peer == null:
		server.stop()
		return t

	# Framing: one JSON object per line in each direction.
	t.assert_true(client.send({"type": "hello"}))
	var hello: Dictionary = _read_line(peer)
	t.assert_eq(hello.get("type"), "hello")

	# Pre-queue the reply so the synchronous request can complete in-process.
	_write_line(peer, {"action": [2, 1, 2, 1, 1, 0]})
	var sim := SimulationManager.new()
	sim.create_visuals = false
	sim.build(1, 1)
	var controller := RemotePolicyController.new(client)
	var action = controller.get_action(sim.environments[0])
	t.assert_eq(action.move_axis, 1, "MultiDiscrete 2 decodes to forward")
	t.assert_eq(action.look_yaw_axis, 1)
	t.assert_true(action.shoot, "shoot is decoded")
	t.assert_eq(controller.requests_answered, 1)
	var act: Dictionary = _read_line(peer)
	t.assert_eq(act.get("type"), "act")
	t.assert_eq((act.get("obs", []) as Array).size(), Observation.FIELD_COUNT)

	# A closed server must not hang the viewer: the controller goes idle.
	peer.disconnect_from_host()
	server.stop()
	var idle = controller.get_action(sim.environments[0])
	t.assert_false(idle.shoot, "a dead link yields an idle action")
	t.assert_false(controller.is_connected_to_policy(), "the failure is sticky and visible")
	controller.free()
	sim.free()
	return t


static func _accept(server: TCPServer) -> StreamPeerTCP:
	var deadline: int = Time.get_ticks_msec() + READ_TIMEOUT_MS
	while Time.get_ticks_msec() < deadline:
		if server.is_connection_available():
			return server.take_connection()
		OS.delay_msec(2)
	return null


static func _write_line(peer: StreamPeerTCP, message: Dictionary) -> void:
	peer.put_data((JSON.stringify(message) + "\n").to_utf8_buffer())


static func _read_line(peer: StreamPeerTCP) -> Dictionary:
	var buffer := PackedByteArray()
	var deadline: int = Time.get_ticks_msec() + READ_TIMEOUT_MS
	while Time.get_ticks_msec() < deadline:
		peer.poll()
		var available: int = peer.get_available_bytes()
		if available > 0:
			var chunk: Array = peer.get_partial_data(available)
			buffer.append_array(chunk[1])
			var newline_at: int = buffer.find(10)
			if newline_at >= 0:
				var parsed = JSON.parse_string(buffer.slice(0, newline_at).get_string_from_utf8())
				return parsed if parsed is Dictionary else {}
		OS.delay_msec(2)
	return {}
