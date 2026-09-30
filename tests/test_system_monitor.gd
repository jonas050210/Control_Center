## Focused tests for the PC telemetry layer (ControlCenterSystemMonitor).
## Parsers are exercised with fixed command output — no external process
## runs here — and the payload shape/N-A rules are pinned down.
class_name TestSystemMonitor
extends RefCounted

const ControlCenterSystemMonitor = preload("res://scripts/control_center/system_monitor.gd")
const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_snapshot_payload_shape_is_complete_before_any_sample() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_payload_shape")
	var snapshot: Dictionary = ControlCenterSystemMonitor.empty_snapshot()
	t.assert_true(snapshot.has("sampled_at"))
	t.assert_null(snapshot["sampled_at"], "no timestamp may exist before a real sample")
	for section_name in ["gpu", "cpu", "ram"]:
		t.assert_true(snapshot.has(section_name), "missing section %s" % section_name)
		var section: Dictionary = snapshot[section_name]
		t.assert_false(bool(section["available"]), "%s must start unavailable" % section_name)
	var gpu: Dictionary = snapshot["gpu"]
	for key in ["name", "utilization_percent", "vram_used_mb", "vram_total_mb", "temperature_c"]:
		t.assert_true(gpu.has(key), "gpu section is missing %s" % key)
	t.assert_null(gpu["utilization_percent"])
	t.assert_null(gpu["vram_used_mb"])
	t.assert_null(gpu["vram_total_mb"])
	t.assert_null(gpu["temperature_c"])
	var cpu: Dictionary = snapshot["cpu"]
	t.assert_null(cpu["utilization_percent"])
	t.assert_null(cpu["temperature_c"])
	var ram: Dictionary = snapshot["ram"]
	for key in ["used_mb", "total_mb", "utilization_percent"]:
		t.assert_null(ram[key], "ram %s must be null before a sample" % key)
	return t


func test_live_monitor_reports_unavailable_not_fake_before_polling() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_no_fabrication_before_polling")
	var monitor := ControlCenterSystemMonitor.new()
	var snapshot: Dictionary = monitor.snapshot()
	t.assert_false(bool((snapshot["gpu"] as Dictionary)["available"]))
	t.assert_false(bool((snapshot["cpu"] as Dictionary)["available"]))
	t.assert_false(bool((snapshot["ram"] as Dictionary)["available"]))
	t.assert_false(monitor.is_active(), "the monitor never polls unless enabled")
	monitor.free()
	return t


func test_nvidia_smi_parsing_extracts_all_fields() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_nvidia_smi_parse")
	var gpu: Dictionary = ControlCenterSystemMonitor.parse_nvidia_smi(
		"NVIDIA GeForce RTX 4060 Ti, 34, 2048, 8188, 52\n"
	)
	t.assert_true(bool(gpu["available"]))
	t.assert_eq(gpu["name"], "NVIDIA GeForce RTX 4060 Ti")
	t.assert_eq(gpu["source"], "nvidia-smi")
	t.assert_almost_eq(float(gpu["utilization_percent"]), 34.0)
	t.assert_almost_eq(float(gpu["vram_used_mb"]), 2048.0)
	t.assert_almost_eq(float(gpu["vram_total_mb"]), 8188.0)
	t.assert_almost_eq(float(gpu["temperature_c"]), 52.0)
	return t


func test_nvidia_smi_na_fields_become_null_not_numbers() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_nvidia_smi_na_fields")
	var gpu: Dictionary = ControlCenterSystemMonitor.parse_nvidia_smi(
		"Quadro P400, [N/A], 512, 2048, [N/A]"
	)
	t.assert_true(bool(gpu["available"]), "memory numbers are real, so the probe worked")
	t.assert_null(gpu["utilization_percent"], "[N/A] must not be coerced to a number")
	t.assert_null(gpu["temperature_c"])
	t.assert_almost_eq(float(gpu["vram_used_mb"]), 512.0)
	t.assert_almost_eq(float(gpu["vram_total_mb"]), 2048.0)
	return t


func test_nvidia_smi_garbage_yields_unavailable() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_nvidia_smi_garbage")
	for text in ["", "\n", "command not found", "a, b"]:
		var gpu: Dictionary = ControlCenterSystemMonitor.parse_nvidia_smi(text)
		t.assert_false(bool(gpu["available"]), "garbage input '%s' must be unavailable" % text)
		t.assert_null(gpu["utilization_percent"])
	return t


func test_gpu_name_with_comma_survives_parsing() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_nvidia_smi_comma_name")
	var gpu: Dictionary = ControlCenterSystemMonitor.parse_nvidia_smi(
		"NVIDIA RTX, Special Edition, 10, 100, 1000, 40"
	)
	t.assert_true(bool(gpu["available"]))
	t.assert_eq(gpu["name"], "NVIDIA RTX, Special Edition")
	t.assert_almost_eq(float(gpu["utilization_percent"]), 10.0)
	return t


func test_windows_cpu_load_parsing() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_windows_cpu_load")
	var single = ControlCenterSystemMonitor.parse_windows_cpu_load("\r\nLoadPercentage=37\r\n\r\n")
	t.assert_not_null(single)
	t.assert_almost_eq(float(single), 37.0)
	var multi = ControlCenterSystemMonitor.parse_windows_cpu_load(
		"LoadPercentage=20\nLoadPercentage=40\n"
	)
	t.assert_almost_eq(float(multi), 30.0, 0.001, "multi-socket loads are averaged")
	var bare = ControlCenterSystemMonitor.parse_windows_cpu_load("12,5\n")
	t.assert_almost_eq(float(bare), 12.5, 0.001, "PowerShell decimal-comma output parses")
	t.assert_null(
		ControlCenterSystemMonitor.parse_windows_cpu_load("no counters here"),
		"unparsable output must be null, never a made-up load"
	)
	return t


func test_proc_stat_cpu_percent_needs_two_valid_samples() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_proc_stat_percent")
	var first: Dictionary = ControlCenterSystemMonitor.parse_proc_stat_totals(
		"cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 50 0 50 350 50 0 0 0 0 0\n"
	)
	t.assert_true(bool(first["valid"]))
	var second: Dictionary = ControlCenterSystemMonitor.parse_proc_stat_totals(
		"cpu  200 0 200 1000 200 0 0 0 0 0\n"
	)
	# Delta: total 1600 -> 200+200+1000+200=... totals: first=1000, second=1600,
	# idle first=800, second=1200 -> busy 600-400=200 of 600 => 33.3%
	var percent = ControlCenterSystemMonitor.cpu_percent_between(first, second)
	t.assert_not_null(percent)
	t.assert_almost_eq(float(percent), 33.333, 0.1)
	t.assert_null(
		ControlCenterSystemMonitor.cpu_percent_between({}, second),
		"the first sample alone cannot produce a percentage"
	)
	t.assert_null(
		ControlCenterSystemMonitor.cpu_percent_between(second, second),
		"identical samples (no elapsed time) must not fake 0% or 100%"
	)
	t.assert_false(bool(ControlCenterSystemMonitor.parse_proc_stat_totals("intr 12345")["valid"]))
	return t


func test_ram_snapshot_from_memory_info() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_ram_snapshot")
	var gib: float = 1024.0 * 1024.0 * 1024.0
	var ram: Dictionary = ControlCenterSystemMonitor.ram_snapshot_from(
		{"physical": 16.0 * gib, "free": 4.0 * gib, "available": 6.0 * gib, "stack": 0}
	)
	t.assert_true(bool(ram["available"]))
	t.assert_almost_eq(float(ram["total_mb"]), 16384.0, 0.5)
	t.assert_almost_eq(float(ram["used_mb"]), 12288.0, 0.5)
	t.assert_almost_eq(float(ram["utilization_percent"]), 75.0, 0.1)
	var missing: Dictionary = ControlCenterSystemMonitor.ram_snapshot_from({})
	t.assert_false(bool(missing["available"]))
	t.assert_null(missing["used_mb"])
	var nonsense: Dictionary = ControlCenterSystemMonitor.ram_snapshot_from(
		{"physical": 100, "free": 200}
	)
	t.assert_false(bool(nonsense["available"]), "free > total is not a real measurement")
	return t


func test_real_memory_info_produces_a_real_ram_sample() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_real_ram")
	var ram: Dictionary = ControlCenterSystemMonitor.ram_snapshot_from(OS.get_memory_info())
	# Every supported desktop platform reports physical memory.
	t.assert_true(bool(ram["available"]), "OS.get_memory_info() should yield real RAM data")
	t.assert_gt(float(ram["total_mb"]), 0.0)
	t.assert_gte(float(ram["used_mb"]), 0.0)
	t.assert_lte(float(ram["utilization_percent"]), 100.0)
	return t


func test_unavailable_metrics_format_as_na() -> SandboxTest:
	var t := SandboxTest.new("system_monitor_na_formatting")
	t.assert_eq(ControlCenterSystemMonitor.format_metric(null), "N/A")
	t.assert_eq(ControlCenterSystemMonitor.format_metric(null, 1, "%"), "N/A")
	t.assert_eq(ControlCenterSystemMonitor.format_metric(42.6, 0, "%"), "43%")
	t.assert_eq(ControlCenterSystemMonitor.format_metric(8188.0, 0, " MB"), "8188 MB")
	return t
