## Tests Map Analyzer 2.0's analysis layer: heatmaps, region classification,
## routes, sightlines, cover, metrics and the exploration curve.
##
## The load-bearing assertion in this file is the negative one: an
## ExplorationReport built from a SpatialMemory that has observed nothing
## must report everything as unknown. If hidden geometry ever leaks into
## the analyzer, that test is what fails.
class_name TestExplorationReport
extends RefCounted

## Explicit dependencies keep standalone/headless execution independent of the editor class cache.
const ExplorationReport = preload("res://scripts/exploration/exploration_report.gd")
const MapAnalyzer = preload("res://scripts/exploration/map_analyzer.gd")
const SpatialMemory = preload("res://scripts/exploration/spatial_memory.gd")

const SandboxTest = preload("res://tests/sandbox_test.gd")


func _memory() -> SpatialMemory:
	return SpatialMemory.create(8.0, 2.0)


func test_untouched_memory_is_entirely_unknown() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_starts_unknown")
	var memory := _memory()
	var report := ExplorationReport.create(memory)
	var counts: Dictionary = report.regions()
	t.assert_eq(int(counts["observed"]), 0)
	t.assert_eq(int(counts["unknown"]), memory.cell_count)
	t.assert_eq(report.unknown_cells().size(), memory.cell_count)
	t.assert_eq(report.sightlines().size(), 0)
	t.assert_eq(report.cover_positions().size(), 0)
	t.assert_almost_eq(float(report.metrics()["coverage"]), 0.0)
	# Every heatmap cell must read as unknown, not as a plausible zero.
	var coverage_map: PackedFloat32Array = report.heatmap("coverage")
	t.assert_eq(coverage_map.size(), memory.cell_count)
	for value in coverage_map:
		t.assert_almost_eq(value, ExplorationReport.UNKNOWN)
	return t


func test_walking_marks_only_what_was_perceived() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_perception_only")
	var memory := _memory()
	memory.observe_self(Vector3(0.0, 0.0, 0.0), 0.9)
	memory.tick(0.1)
	memory.observe_self(Vector3(2.0, 0.0, 0.0), 0.9)
	var report := ExplorationReport.create(memory)
	var counts: Dictionary = report.regions()
	t.assert_eq(int(counts["visited"]), 2)
	t.assert_eq(int(counts["observed"]), 2)
	t.assert_eq(int(counts["unknown"]), memory.cell_count - 2)
	t.assert_true(int(counts["unknown"]) > 0, "a two-step walk cannot know the whole map")
	return t


func test_heatmap_layers_are_bounded_and_named() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_heatmaps")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 0.8)
	memory.mark_danger(Vector3.ZERO, 0.5)
	var report := ExplorationReport.create(memory)
	for layer_value in ExplorationReport.LAYERS:
		var layer: String = str(layer_value)
		var values: PackedFloat32Array = report.heatmap(layer)
		t.assert_eq(values.size(), memory.cell_count, layer)
		for value in values:
			t.assert_true(
				value == ExplorationReport.UNKNOWN or (value >= 0.0 and value <= 1.0),
				"%s produced %f" % [layer, value]
			)
	t.assert_eq(report.heatmap("wallhack").size(), 0, "unknown layers must not be invented")
	return t


func test_recency_layer_decays_with_age() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_recency")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	var report := ExplorationReport.create(memory)
	var index: int = memory.cell_index(Vector3.ZERO)
	var fresh: float = report.heatmap("recency")[index]
	memory.tick(30.0)
	var stale: float = report.heatmap("recency")[index]
	t.assert_true(stale < fresh, "an old observation must look older")
	t.assert_true(stale > 0.0)
	return t


func test_observed_but_unvisited_is_distinguished_from_visited() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_observed_vs_visited")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	memory.mark_danger(Vector3(4.0, 0.0, 4.0), 0.4)
	var report := ExplorationReport.create(memory)
	# The danger cell is known (something happened there) but was never
	# walked through.
	t.assert_eq(report.observed_but_unvisited_cells().size(), 1)
	t.assert_eq(int(report.regions()["visited"]), 1)
	t.assert_eq(int(report.regions()["dangerous"]), 1)
	return t


func test_route_and_efficiency() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_route")
	var memory := _memory()
	for step in range(4):
		memory.observe_self(Vector3(float(step) * 2.0 - 4.0, 0.0, 0.0), 1.0)
		memory.tick(0.1)
	var report := ExplorationReport.create(memory)
	var route: Array = report.route_positions()
	t.assert_eq(route.size(), 4)
	t.assert_true(report.route_length() > 0.0)
	t.assert_true(report.exploration_efficiency() > 0.0)

	var empty_report := ExplorationReport.create(_memory())
	t.assert_almost_eq(empty_report.route_length(), 0.0)
	t.assert_almost_eq(empty_report.exploration_efficiency(), 0.0)
	return t


func test_coverage_curve_samples() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_curve")
	var memory := _memory()
	var report := ExplorationReport.create(memory)
	report.sample(0.0)
	memory.observe_self(Vector3.ZERO, 1.0)
	report.sample(1.0)
	t.assert_eq(report.coverage_samples.size(), 2)
	var first: Dictionary = report.coverage_samples[0]
	var second: Dictionary = report.coverage_samples[1]
	t.assert_almost_eq(float(first["coverage"]), 0.0)
	t.assert_true(float(second["coverage"]) > float(first["coverage"]))
	report.clear_samples()
	t.assert_eq(report.coverage_samples.size(), 0)
	return t


func test_metrics_are_diagnostic_and_complete() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_metrics")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	var report := ExplorationReport.create(memory)
	var metrics: Dictionary = report.metrics()
	for key in [
		"coverage",
		"visited_fraction",
		"unknown_fraction",
		"dark_fraction",
		"cover_fraction",
		"danger_fraction",
		"mean_uncertainty",
		"route_length",
		"route_cells",
		"exploration_efficiency",
		"samples",
	]:
		t.assert_true(metrics.has(key), "missing metric %s" % str(key))
	t.assert_true(float(metrics["unknown_fraction"]) > 0.9)
	return t


func test_full_payload_shape() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_payload")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	var report := ExplorationReport.create(memory)
	report.sample(0.0)
	var payload: Dictionary = report.to_dict(["coverage", "danger"])
	t.assert_true(payload.has("metrics"))
	t.assert_true(payload.has("regions"))
	t.assert_true(payload.has("grid"))
	t.assert_eq(int((payload["grid"] as Dictionary)["cell_count"]), memory.cell_count)
	var heatmaps: Dictionary = payload["heatmaps"]
	t.assert_eq(heatmaps.size(), 2, "only the requested layers should be produced")
	t.assert_true(heatmaps.has("coverage"))
	t.assert_true(heatmaps.has("danger"))
	t.assert_eq((payload["coverage_curve"] as Array).size(), 1)
	var all_layers: Dictionary = report.to_dict()["heatmaps"]
	t.assert_eq(all_layers.size(), ExplorationReport.LAYERS.size())
	return t


func test_summary_is_formatted() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_summary")
	var memory := _memory()
	memory.observe_self(Vector3.ZERO, 1.0)
	var text: String = ExplorationReport.create(memory).format_summary()
	t.assert_true(text.contains("exploration"))
	t.assert_true(text.contains("route"))
	var no_memory := ExplorationReport.new()
	t.assert_true(no_memory.format_summary().contains("no spatial memory"))
	t.assert_eq(no_memory.metrics().size(), 0)
	t.assert_eq(no_memory.heatmap("coverage").size(), 0)
	t.assert_eq(no_memory.unknown_cells().size(), 0)
	return t


func test_report_integrates_with_the_map_analyzer() -> SandboxTest:
	var t := SandboxTest.new("exploration_report_with_analyzer")
	var analyzer := MapAnalyzer.create(8.0, 2.0)
	analyzer.reset()
	var report := ExplorationReport.create(analyzer.memory)
	for step in range(5):
		analyzer.update(0.1, {"position": Vector3(float(step) - 2.0, 0.0, 0.0)})
		report.sample(analyzer.elapsed)
	t.assert_true(float(report.metrics()["coverage"]) > 0.0)
	t.assert_eq(report.coverage_samples.size(), 5)
	t.assert_almost_eq(
		float(report.metrics()["coverage"]),
		analyzer.coverage(),
		0.0001,
		"report must agree with the analyzer"
	)
	return t
