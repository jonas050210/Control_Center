## Tests for ControlCenterEventLog: filtering, throttling, the per-second
## budget and the hard "disabled in TRAINING" early-out that keeps logging
## off the RL hot path.
class_name TestControlCenterEventLog
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ControlCenterEventLog = preload("res://scripts/control_center/control_center_event_log.gd")


const SandboxTest = preload("res://tests/sandbox_test.gd")


func test_entries_are_filterable_by_category() -> SandboxTest:
	var t := SandboxTest.new("event_log_filters_by_category")
	var log_ref := ControlCenterEventLog.new()
	log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "hit enemy 0", {}, "", 0.0)
	log_ref.log_event(ControlCenterEventLog.Category.SYSTEM, "mode -> WATCH", {}, "", 0.1)
	log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "killed enemy 0", {}, "", 0.2)

	t.assert_eq(log_ref.size(), 3)
	t.assert_eq(log_ref.entries(ControlCenterEventLog.Category.COMBAT).size(), 2)
	t.assert_eq(log_ref.entries(ControlCenterEventLog.Category.SYSTEM).size(), 1)
	t.assert_eq(log_ref.entries(ControlCenterEventLog.Category.REWARD).size(), 0)
	t.assert_eq(
		log_ref.entries(ControlCenterEventLog.FILTER_ALL).size(),
		3,
		"FILTER_ALL must not filter anything out"
	)
	var newest: Array = log_ref.entries(ControlCenterEventLog.FILTER_ALL, 1)
	t.assert_eq(str((newest[0] as Dictionary)["message"]), "killed enemy 0")
	return t


func test_throttle_key_collapses_repeated_events() -> SandboxTest:
	var t := SandboxTest.new("event_log_throttles_repeats")
	var log_ref := ControlCenterEventLog.new()
	log_ref.throttle_seconds = 0.5

	t.assert_true(
		log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "took damage", {}, "dmg", 0.0)
	)
	t.assert_false(
		log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "took damage", {}, "dmg", 0.2),
		"a repeat inside the throttle window is dropped"
	)
	t.assert_true(
		log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "took damage", {}, "dmg", 0.6),
		"the same key is accepted again after the window"
	)
	t.assert_eq(log_ref.size(), 2)
	t.assert_eq(log_ref.dropped_count, 1, "drops are counted, never silent")
	t.assert_true(
		log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "other event", {}, "other", 0.61),
		"a different throttle key is unaffected"
	)
	return t


func test_events_per_second_budget_bounds_a_storm() -> SandboxTest:
	var t := SandboxTest.new("event_log_per_second_budget")
	var log_ref := ControlCenterEventLog.new()
	log_ref.max_events_per_second = 5.0
	var accepted: int = 0
	for index in range(50):
		if log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "shot %d" % index, {}, "", 0.5):
			accepted += 1
	t.assert_eq(accepted, 5, "the per-second budget caps the frame cost of logging")
	t.assert_eq(log_ref.dropped_count, 45)

	# A new one-second window resets the budget.
	t.assert_true(log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "next", {}, "", 1.6))
	return t


func test_capacity_bounds_memory() -> SandboxTest:
	var t := SandboxTest.new("event_log_capacity_bounds_memory")
	var log_ref := ControlCenterEventLog.new()
	log_ref.capacity = 10
	log_ref.max_events_per_second = 0.0  # disable the budget for this test
	for index in range(100):
		log_ref.log_event(
			ControlCenterEventLog.Category.SYSTEM, "entry %d" % index, {}, "", float(index)
		)
	t.assert_eq(log_ref.size(), 10, "the ring buffer never grows past capacity")
	t.assert_eq(log_ref.total_accepted, 100, "the monotonic counter still sees every event")
	var entries: Array = log_ref.entries()
	t.assert_eq(str((entries[entries.size() - 1] as Dictionary)["message"]), "entry 99")
	return t


func test_disabled_log_is_a_hard_early_out() -> SandboxTest:
	var t := SandboxTest.new("event_log_disabled_early_out")
	var log_ref := ControlCenterEventLog.new()
	log_ref.enabled = false
	for index in range(20):
		t.assert_false(
			log_ref.log_event(ControlCenterEventLog.Category.COMBAT, "x %d" % index, {}, "", 0.0)
		)
	t.assert_eq(log_ref.size(), 0, "TRAINING mode must not buffer a single event")
	t.assert_eq(log_ref.total_accepted, 0)
	t.assert_eq(log_ref.dropped_count, 0, "an early-out is not a drop; it costs nothing")
	return t


func test_format_lines_and_clear() -> SandboxTest:
	var t := SandboxTest.new("event_log_format_and_clear")
	var log_ref := ControlCenterEventLog.new()
	log_ref.log_event(ControlCenterEventLog.Category.REWARD, "reward spike +2.5", {}, "", 1.25)
	var lines: PackedStringArray = log_ref.format_lines()
	t.assert_eq(lines.size(), 1)
	t.assert_true(lines[0].contains("REWARD"), "the category is visible in the rendered line")
	t.assert_true(lines[0].contains("reward spike"))
	t.assert_true(lines[0].contains("1.25"), "the event time is rendered")

	log_ref.clear()
	t.assert_eq(log_ref.size(), 0)
	t.assert_eq(log_ref.total_accepted, 0)
	t.assert_eq(log_ref.format_lines().size(), 0)
	return t


func test_filter_options_cover_every_category() -> SandboxTest:
	var t := SandboxTest.new("event_log_filter_options")
	var options: Array = ControlCenterEventLog.filter_options()
	t.assert_eq(options[0], ControlCenterEventLog.FILTER_ALL, "ALL is offered first")
	for category in [
		ControlCenterEventLog.Category.COMBAT,
		ControlCenterEventLog.Category.PERCEPTION,
		ControlCenterEventLog.Category.SYSTEM,
		ControlCenterEventLog.Category.REWARD,
		ControlCenterEventLog.Category.ERROR
	]:
		t.assert_true(options.has(category), "category %d must be filterable" % category)
		t.assert_eq(
			ControlCenterEventLog.category_from_name(
				ControlCenterEventLog.category_name(category)
			),
			category
		)
	return t
