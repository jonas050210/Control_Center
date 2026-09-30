## ControlCenterSystemMonitor
##
## Local PC telemetry (GPU / CPU / RAM) for the Control Center dashboard.
##
## Design rules:
##   * Every metric is measured, never derived from guesses. A metric that
##     cannot be obtained on this machine is `null` in the snapshot and the
##     UI renders it as "N/A" via `format_metric()`.
##   * External probes (nvidia-smi, wmic/powershell) run on a worker thread
##     at a slow interval so the render loop and the RL trainer never wait
##     on a subprocess.
##   * The monitor is presentation-support only: nothing here reads or
##     writes simulation or training state, and headless RL runs never
##     construct it.
class_name ControlCenterSystemMonitor
extends Node

signal sample_updated(snapshot: Dictionary)

## External probes are expensive (subprocess spawn); poll slowly.
const POLL_INTERVAL_SECONDS: float = 2.0
## After this many consecutive probe failures the probe is parked and only
## retried occasionally, so a machine without an NVIDIA GPU does not spawn
## a failing process every two seconds forever.
const PROBE_FAILURE_LIMIT: int = 3
const PROBE_RETRY_INTERVAL_SECONDS: float = 30.0

const NVIDIA_SMI_ARGUMENTS: PackedStringArray = [
	"--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
	"--format=csv,noheader,nounits",
]

## Thermal zone types accepted as "the CPU" on Linux. Anything else (WiFi
## module, battery, ambient) must not be presented as a CPU temperature.
const LINUX_CPU_THERMAL_TYPES: Array = ["x86_pkg_temp", "cpu_thermal", "coretemp", "k10temp"]

var _latest: Dictionary = {}
var _mutex: Mutex = Mutex.new()
var _thread: Thread
var _poll_accumulator: float = POLL_INTERVAL_SECONDS
var _prev_proc_stat: Dictionary = {}
var _gpu_probe_failures: int = 0
var _gpu_probe_parked_for: float = 0.0
var _cpu_probe_failures: int = 0
var _cpu_probe_parked_for: float = 0.0
var _active: bool = false


func _init() -> void:
	_latest = empty_snapshot()


func _exit_tree() -> void:
	_join_thread()


## Starts/stops polling. The monitor never polls unless explicitly enabled,
## so a session created for the RL/headless path pays nothing.
func set_active(value: bool) -> void:
	_active = value
	set_process(value)
	if value:
		# Deliver the first sample promptly instead of after a full interval.
		_poll_accumulator = POLL_INTERVAL_SECONDS


func is_active() -> bool:
	return _active


## Latest measured values. Always structurally complete; unavailable
## metrics are null and each section carries an `available` flag.
func snapshot() -> Dictionary:
	_mutex.lock()
	var copy: Dictionary = _latest.duplicate(true)
	_mutex.unlock()
	return copy


func _process(delta: float) -> void:
	_poll_accumulator += delta
	if _gpu_probe_parked_for > 0.0:
		_gpu_probe_parked_for = maxf(0.0, _gpu_probe_parked_for - delta)
	if _cpu_probe_parked_for > 0.0:
		_cpu_probe_parked_for = maxf(0.0, _cpu_probe_parked_for - delta)
	if _poll_accumulator < POLL_INTERVAL_SECONDS:
		return
	if _thread != null and _thread.is_alive():
		# The previous probe is still running (slow machine); skip this tick.
		return
	_poll_accumulator = 0.0
	_join_thread()
	_sample_main_thread_metrics()
	var jobs: Dictionary = {
		"gpu": _gpu_probe_parked_for <= 0.0,
		"cpu_command": _needs_cpu_command() and _cpu_probe_parked_for <= 0.0,
		"os": OS.get_name(),
	}
	_thread = Thread.new()
	_thread.start(_probe_worker.bind(jobs))


## RAM and /proc-based CPU sampling are file/API reads: cheap enough for
## the main thread and free of subprocess latency.
func _sample_main_thread_metrics() -> void:
	var ram: Dictionary = ram_snapshot_from(OS.get_memory_info())
	var cpu_update: Dictionary = {}
	if OS.get_name() == "Linux":
		var totals: Dictionary = parse_proc_stat_totals(_read_text_file("/proc/stat"))
		var percent = cpu_percent_between(_prev_proc_stat, totals)
		if bool(totals.get("valid", false)):
			_prev_proc_stat = totals
		cpu_update["utilization_percent"] = percent
		cpu_update["source"] = "/proc/stat" if percent != null else ""
		cpu_update["temperature_c"] = _read_linux_cpu_temperature()
	_mutex.lock()
	_latest["ram"] = ram
	if not cpu_update.is_empty():
		var cpu: Dictionary = _latest["cpu"]
		cpu["utilization_percent"] = cpu_update["utilization_percent"]
		cpu["temperature_c"] = cpu_update["temperature_c"]
		cpu["source"] = cpu_update["source"]
		cpu["available"] = cpu_update["utilization_percent"] != null
	_latest["sampled_at"] = Time.get_unix_time_from_system()
	_mutex.unlock()
	sample_updated.emit(snapshot())


## Runs on the worker thread: only subprocess probes live here.
func _probe_worker(jobs: Dictionary) -> void:
	var gpu: Dictionary = {}
	var gpu_ran: bool = false
	if bool(jobs.get("gpu", false)):
		gpu_ran = true
		gpu = _probe_nvidia_smi()
	var cpu_percent = null
	var cpu_source: String = ""
	var cpu_ran: bool = false
	if bool(jobs.get("cpu_command", false)):
		cpu_ran = true
		var probed: Dictionary = _probe_windows_cpu()
		cpu_percent = probed.get("percent")
		cpu_source = str(probed.get("source", ""))
	call_deferred("_apply_worker_results", gpu_ran, gpu, cpu_ran, cpu_percent, cpu_source)


func _apply_worker_results(
	gpu_ran: bool, gpu: Dictionary, cpu_ran: bool, cpu_percent, cpu_source: String
) -> void:
	_mutex.lock()
	if gpu_ran:
		if bool(gpu.get("available", false)):
			_gpu_probe_failures = 0
			_latest["gpu"] = gpu
		else:
			_gpu_probe_failures += 1
			_latest["gpu"] = _merge_unavailable_gpu(gpu)
			if _gpu_probe_failures >= PROBE_FAILURE_LIMIT:
				_gpu_probe_parked_for = PROBE_RETRY_INTERVAL_SECONDS
	if cpu_ran:
		var cpu: Dictionary = _latest["cpu"]
		cpu["utilization_percent"] = cpu_percent
		cpu["source"] = cpu_source
		cpu["available"] = cpu_percent != null
		if cpu_percent == null:
			_cpu_probe_failures += 1
			if _cpu_probe_failures >= PROBE_FAILURE_LIMIT:
				_cpu_probe_parked_for = PROBE_RETRY_INTERVAL_SECONDS
		else:
			_cpu_probe_failures = 0
	_latest["sampled_at"] = Time.get_unix_time_from_system()
	_mutex.unlock()
	sample_updated.emit(snapshot())


## nvidia-smi ships with every NVIDIA driver (Windows: System32, on PATH).
## When it is missing or fails, GPU metrics stay null — the renderer's
## adapter name alone is NOT utilization/VRAM data and is kept separate.
func _probe_nvidia_smi() -> Dictionary:
	var output: Array = []
	var code: int = OS.execute("nvidia-smi", NVIDIA_SMI_ARGUMENTS, output, true)
	if code != 0 or output.is_empty():
		var fallback: Dictionary = unavailable_gpu()
		fallback["name"] = _adapter_name_fallback()
		return fallback
	return parse_nvidia_smi(str(output[0]))


## Windows has no /proc/stat: CPU utilization there can only be measured
## by asking the OS through a command (wmic / PowerShell CIM). Linux reads
## /proc/stat directly on the main thread, so no command is scheduled;
## platforms with neither source keep utilization null ("N/A") instead of
## guessing.
func _needs_cpu_command() -> bool:
	return OS.get_name() == "Windows"


func _probe_windows_cpu() -> Dictionary:
	# wmic is present on most Windows installs; newer builds removed it, so
	# a CIM query through PowerShell is the fallback. Both return the same
	# Win32_Processor LoadPercentage and are locale independent.
	var output: Array = []
	var code: int = OS.execute("wmic", ["cpu", "get", "loadpercentage", "/value"], output, true)
	if code == 0 and not output.is_empty():
		var value = parse_windows_cpu_load(str(output[0]))
		if value != null:
			return {"percent": value, "source": "wmic"}
	output.clear()
	code = (
		OS
		. execute(
			"powershell",
			[
				"-NoProfile",
				"-Command",
				(
					"(Get-CimInstance Win32_Processor | "
					+ "Measure-Object -Property LoadPercentage -Average).Average"
				),
			],
			output,
			true
		)
	)
	if code == 0 and not output.is_empty():
		var value = parse_windows_cpu_load(str(output[0]))
		if value != null:
			return {"percent": value, "source": "powershell-cim"}
	return {"percent": null, "source": ""}


func _adapter_name_fallback() -> String:
	# Presentation-only fallback so the GPU card can at least be named when
	# NVIDIA telemetry is absent. Never used for utilization/VRAM numbers.
	if DisplayServer.get_name() == "headless":
		return ""
	return RenderingServer.get_video_adapter_name()


func _read_linux_cpu_temperature():
	for zone in range(10):
		var base: String = "/sys/class/thermal/thermal_zone%d" % zone
		var zone_type: String = _read_text_file(base + "/type").strip_edges()
		if zone_type.is_empty():
			break
		if not LINUX_CPU_THERMAL_TYPES.has(zone_type):
			continue
		var raw: String = _read_text_file(base + "/temp").strip_edges()
		if raw.is_valid_int():
			return float(raw.to_int()) / 1000.0
	return null


func _read_text_file(path: String) -> String:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return ""
	return file.get_as_text()


func _join_thread() -> void:
	if _thread != null:
		if _thread.is_started():
			_thread.wait_to_finish()
		_thread = null


func _merge_unavailable_gpu(probe: Dictionary) -> Dictionary:
	var gpu: Dictionary = unavailable_gpu()
	var previous_name: String = str((_latest.get("gpu", {}) as Dictionary).get("name", ""))
	var probe_name: String = str(probe.get("name", ""))
	gpu["name"] = probe_name if not probe_name.is_empty() else previous_name
	return gpu


# ---------------------------------------------------------------------------
# Pure parsing/formatting helpers (unit-tested without running any command)
# ---------------------------------------------------------------------------


## Fully-unavailable snapshot: the shape every consumer can rely on before
## the first sample and on machines where nothing can be measured.
static func empty_snapshot() -> Dictionary:
	return {
		"sampled_at": null,
		"gpu": unavailable_gpu(),
		"cpu":
		{
			"available": false,
			"utilization_percent": null,
			"temperature_c": null,
			"source": "",
		},
		"ram":
		{
			"available": false,
			"used_mb": null,
			"total_mb": null,
			"utilization_percent": null,
		},
	}


static func unavailable_gpu() -> Dictionary:
	return {
		"available": false,
		"source": "",
		"name": "",
		"utilization_percent": null,
		"vram_used_mb": null,
		"vram_total_mb": null,
		"temperature_c": null,
	}


## Parses one CSV line of
## `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,
## temperature.gpu --format=csv,noheader,nounits`.
## Individual "[N/A]" fields become null; a malformed line yields the
## unavailable shape instead of invented numbers.
static func parse_nvidia_smi(text: String) -> Dictionary:
	var line: String = ""
	for candidate in text.split("\n"):
		if not str(candidate).strip_edges().is_empty():
			line = str(candidate).strip_edges()
			break
	if line.is_empty():
		return unavailable_gpu()
	var parts: PackedStringArray = line.split(",")
	if parts.size() < 5:
		return unavailable_gpu()
	# The four trailing fields are numeric; everything before them is the
	# device name (which may itself contain commas). nvidia-smi separates
	# CSV fields with ", ", so every fragment after a comma keeps its
	# leading space — strip each fragment before re-joining, or a name
	# containing ", " would come back with doubled spaces.
	var name_parts: PackedStringArray = parts.slice(0, parts.size() - 4)
	var name_fragments := PackedStringArray()
	for fragment in name_parts:
		name_fragments.append(str(fragment).strip_edges())
	var gpu: Dictionary = unavailable_gpu()
	gpu["available"] = true
	gpu["source"] = "nvidia-smi"
	gpu["name"] = ", ".join(name_fragments)
	gpu["utilization_percent"] = _parse_optional_number(parts[parts.size() - 4])
	gpu["vram_used_mb"] = _parse_optional_number(parts[parts.size() - 3])
	gpu["vram_total_mb"] = _parse_optional_number(parts[parts.size() - 2])
	gpu["temperature_c"] = _parse_optional_number(parts[parts.size() - 1])
	return gpu


## One numeric CSV field from nvidia-smi. "[N/A]", blank or non-numeric
## text becomes null — a field the driver did not publish is never coerced
## into a fabricated number.
static func _parse_optional_number(text: String):
	var value: String = text.strip_edges()
	if not value.is_valid_float():
		return null
	return value.to_float()


## Parses `wmic cpu get loadpercentage /value` ("LoadPercentage=12", one
## line per socket) or a bare number printed by the PowerShell CIM
## fallback. Returns null when no load value is present.
static func parse_windows_cpu_load(text: String):
	var total: float = 0.0
	var count: int = 0
	for raw_line in text.split("\n"):
		var line: String = str(raw_line).strip_edges()
		if line.is_empty():
			continue
		var value: String = line
		if line.to_lower().begins_with("loadpercentage"):
			var separator: int = line.find("=")
			if separator < 0:
				continue
			value = line.substr(separator + 1).strip_edges()
		value = value.replace(",", ".")
		if value.is_valid_float():
			total += value.to_float()
			count += 1
	if count == 0:
		return null
	return clampf(total / float(count), 0.0, 100.0)


## Aggregate jiffy counters from the `cpu ` summary line of /proc/stat.
static func parse_proc_stat_totals(text: String) -> Dictionary:
	for raw_line in text.split("\n"):
		var line: String = str(raw_line).strip_edges()
		if not line.begins_with("cpu "):
			continue
		var fields: PackedStringArray = line.split(" ", false)
		if fields.size() < 5:
			break
		var total: float = 0.0
		var idle: float = 0.0
		for index in range(1, fields.size()):
			if not fields[index].is_valid_int():
				continue
			var jiffies: float = float(fields[index].to_int())
			total += jiffies
			# idle (field 4) + iowait (field 5) both count as not-working.
			if index == 4 or index == 5:
				idle += jiffies
		return {"valid": true, "total": total, "idle": idle}
	return {"valid": false, "total": 0.0, "idle": 0.0}


## CPU utilization percentage between two /proc/stat samples, or null when
## the samples cannot produce a real measurement (first sample, clock
## reset, identical totals).
static func cpu_percent_between(previous: Dictionary, current: Dictionary):
	if not bool(previous.get("valid", false)) or not bool(current.get("valid", false)):
		return null
	var delta_total: float = float(current.get("total", 0.0)) - float(previous.get("total", 0.0))
	var delta_idle: float = float(current.get("idle", 0.0)) - float(previous.get("idle", 0.0))
	if delta_total <= 0.0 or delta_idle < 0.0:
		return null
	return clampf((delta_total - delta_idle) / delta_total * 100.0, 0.0, 100.0)


## RAM section from OS.get_memory_info(). Godot reports bytes; zero/absent
## totals mean the platform did not provide the counter.
static func ram_snapshot_from(info: Dictionary) -> Dictionary:
	var ram: Dictionary = {
		"available": false,
		"used_mb": null,
		"total_mb": null,
		"utilization_percent": null,
	}
	var total_bytes: float = float(info.get("physical", 0))
	var free_bytes: float = float(info.get("free", 0))
	if total_bytes <= 0.0 or free_bytes < 0.0 or free_bytes > total_bytes:
		return ram
	ram["available"] = true
	ram["total_mb"] = total_bytes / (1024.0 * 1024.0)
	ram["used_mb"] = (total_bytes - free_bytes) / (1024.0 * 1024.0)
	ram["utilization_percent"] = (total_bytes - free_bytes) / total_bytes * 100.0
	return ram


## "N/A" for null, formatted number otherwise. The single place deciding
## how an unavailable metric is displayed.
static func format_metric(value, decimals: int = 0, suffix: String = "") -> String:
	if value == null:
		return "N/A"
	return String.num(float(value), maxi(0, decimals)) + suffix
