## SandboxTest
##
## Minimal, dependency-free assertion framework used by the automated test
## suite in tests/. Deliberately not an external addon (e.g. GUT) so the
## project has zero third-party dependencies; this is small enough to
## audit at a glance.
class_name SandboxTest
extends RefCounted

var test_name: String = ""
var failures: Array = []  # Array[String]


func _init(p_test_name: String) -> void:
	test_name = p_test_name


func assert_true(condition: bool, message: String = "") -> void:
	if not condition:
		failures.append("expected true but was false. %s" % message)


func assert_false(condition: bool, message: String = "") -> void:
	if condition:
		failures.append("expected false but was true. %s" % message)


func assert_eq(actual, expected, message: String = "") -> void:
	if actual != expected:
		failures.append("expected %s but got %s. %s" % [str(expected), str(actual), message])


func assert_almost_eq(
	actual: float, expected: float, tolerance: float = 0.001, message: String = ""
) -> void:
	if absf(actual - expected) > tolerance:
		failures.append(
			(
				"expected ~%s (+/-%s) but got %s. %s"
				% [str(expected), str(tolerance), str(actual), message]
			)
		)


func assert_vec_almost_eq(
	actual: Vector3, expected: Vector3, tolerance: float = 0.01, message: String = ""
) -> void:
	if actual.distance_to(expected) > tolerance:
		failures.append(
			(
				"expected ~%s (+/-%s) but got %s. %s"
				% [str(expected), str(tolerance), str(actual), message]
			)
		)


func assert_gt(actual: float, threshold: float, message: String = "") -> void:
	if not (actual > threshold):
		failures.append("expected %s > %s. %s" % [str(actual), str(threshold), message])


func assert_lt(actual: float, threshold: float, message: String = "") -> void:
	if not (actual < threshold):
		failures.append("expected %s < %s. %s" % [str(actual), str(threshold), message])


func assert_null(value, message: String = "") -> void:
	if value != null:
		failures.append("expected null. %s" % message)


func assert_not_null(value, message: String = "") -> void:
	if value == null:
		failures.append("expected non-null value. %s" % message)


func passed() -> bool:
	return failures.is_empty()
