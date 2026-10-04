from rocketai import doctor


def test_python_check_reports_only_supported_versions():
    for version in ((3, 11), (3, 12), (3, 13)):
        ok, detail = doctor._python_check(version)
        assert ok
        assert "unterstützt" not in detail

    for version in ((3, 10), (3, 14), (3, 15)):
        ok, detail = doctor._python_check(version)
        assert not ok
        assert "3.11–3.13" in detail


def test_doctor_reports_teacher_import_failure_without_crashing(monkeypatch):
    monkeypatch.setattr(doctor, "_module", lambda _name: (True, "ok"))

    def fail_teacher_check():
        raise NameError("name '__annotations__' is not defined")

    monkeypatch.setattr(doctor, "teacher_check", fail_teacher_check)
    checks = doctor.run_checks()

    teacher = next(check for check in checks if check["key"] == "teacher")
    assert not teacher["ok"]
    assert "__annotations__" in teacher["detail"]
