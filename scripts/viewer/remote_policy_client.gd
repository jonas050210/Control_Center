## RemotePolicyClient
##
## The viewer's side of the checkpoint-viewer wire: one local TCP connection
## to the Python process that holds the trained policy
## (python/sandboxai/viewer.py). Strict request/response, one JSON object per
## line in each direction - the same framing as the headless bridge, only
## over a socket instead of stdin/stdout, because a rendered Godot window
## cannot block its main loop on stdin.
##
## Messages sent (see docs/CHECKPOINT_VIEWER.md for the full protocol):
##   {"type": "hello", "observation_size": 126, "action_nvec": [...]}
##   {"type": "act", "obs": [126 floats]}          -> {"action": [6 ints]}
##   {"type": "episode", "summary": {...}}         -> {"ok": true}
##   {"type": "bye"}                               (no reply expected)
##
## Every call has a timeout and failure is sticky: once the link breaks the
## client reports `connected == false` and the viewer falls back to an idle
## agent instead of freezing the window.
class_name RemotePolicyClient
extends RefCounted

const CONNECT_TIMEOUT_MS: int = 10000
const REPLY_TIMEOUT_MS: int = 10000
const NEWLINE: int = 10

var connected: bool = false
var last_error: String = ""
var _peer: StreamPeerTCP = StreamPeerTCP.new()
var _buffer: PackedByteArray = PackedByteArray()


## Opens the connection and waits (bounded) until it is established.
func connect_to(host: String, port: int, timeout_ms: int = CONNECT_TIMEOUT_MS) -> bool:
	connected = false
	var error: int = _peer.connect_to_host(host, port)
	if error != OK:
		last_error = "connect_to_host(%s:%d) failed with error %d" % [host, port, error]
		return false
	var deadline: int = Time.get_ticks_msec() + timeout_ms
	while Time.get_ticks_msec() < deadline:
		_peer.poll()
		var status: int = _peer.get_status()
		if status == StreamPeerTCP.STATUS_CONNECTED:
			_peer.set_no_delay(true)
			connected = true
			return true
		if status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
			break
		OS.delay_msec(5)
	last_error = "could not connect to the policy server at %s:%d" % [host, port]
	return false


## Sends one message and waits for its one-line reply. Returns {} (and
## marks the client disconnected) on any failure.
func request(message: Dictionary, timeout_ms: int = REPLY_TIMEOUT_MS) -> Dictionary:
	if not send(message):
		return {}
	return _read_reply(timeout_ms)


## Sends one message without waiting for a reply.
func send(message: Dictionary) -> bool:
	if not connected:
		return false
	var payload: PackedByteArray = (JSON.stringify(message) + "\n").to_utf8_buffer()
	var error: int = _peer.put_data(payload)
	if error != OK:
		_fail("sending to the policy server failed with error %d" % error)
		return false
	return true


## Polite shutdown: tells the server we are leaving, then closes.
func close() -> void:
	if connected:
		send({"type": "bye"})
	_peer.disconnect_from_host()
	connected = false


func _read_reply(timeout_ms: int) -> Dictionary:
	var deadline: int = Time.get_ticks_msec() + timeout_ms
	while connected:
		var newline_at: int = _buffer.find(NEWLINE)
		if newline_at >= 0:
			var line: String = _buffer.slice(0, newline_at).get_string_from_utf8()
			_buffer = _buffer.slice(newline_at + 1)
			var parsed = JSON.parse_string(line)
			if parsed is Dictionary:
				return parsed
			_fail("the policy server sent an invalid reply: %s" % line.left(120))
			return {}
		_peer.poll()
		if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			_fail("the policy server closed the connection")
			return {}
		var available: int = _peer.get_available_bytes()
		if available > 0:
			var chunk: Array = _peer.get_partial_data(available)
			if int(chunk[0]) != OK:
				_fail("reading from the policy server failed with error %d" % int(chunk[0]))
				return {}
			_buffer.append_array(chunk[1])
			continue
		if Time.get_ticks_msec() > deadline:
			_fail("the policy server did not answer within %d ms" % timeout_ms)
			return {}
		OS.delay_usec(200)
	return {}


func _fail(message: String) -> void:
	last_error = message
	connected = false
	_peer.disconnect_from_host()
	push_warning("RemotePolicyClient: " + message)
