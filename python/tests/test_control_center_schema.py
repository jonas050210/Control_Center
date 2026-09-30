from sandboxai.control_center_schema import (
    EVENT_SCHEMA_VERSION,
    METRICS,
    STATUS_SCHEMA_VERSION,
    metric_definition,
    validate_event,
    validate_status,
)


def test_registry_has_unique_self_describing_metrics():
    assert METRICS
    assert all(key == metric.key for key, metric in METRICS.items())
    assert all(metric.label and metric.description for metric in METRICS.values())
    assert metric_definition("steps_per_second").unit == "steps/s"
    assert metric_definition("does_not_exist") is None


def test_status_schema_accepts_a_valid_status_and_legacy_missing_version():
    assert validate_status({
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": "Running",
        "pid": 42,
        "updated_at": 1.0,
        "timesteps": 10,
    }) == []
    assert validate_status({"state": "Finished"}) == []


def test_status_schema_reports_all_obvious_contract_errors():
    problems = validate_status({
        "schema_version": 99,
        "state": "Maybe",
        "pid": "42",
        "progress": "half",
    })
    assert len(problems) == 4


def test_event_schema_is_versioned_and_validated():
    assert validate_event({
        "schema_version": EVENT_SCHEMA_VERSION,
        "category": "system",
        "message": "started",
    }) == []
    assert validate_event({"category": "", "message": 2})
