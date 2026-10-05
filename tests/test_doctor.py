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


def test_rlbot_check_spots_missing_and_wrong_versions(monkeypatch):
    """Nur "rlbot importierbar" reicht nicht: der Bot braucht flat + managers."""
    # Nichts installiert.
    check = doctor.rlbot_check(find_spec=lambda _name: None)
    assert check["ok"] is False
    assert "nicht installiert" in check["detail"]

    # Falsche Reihe installiert (1.x hat rlbot.flat nicht).
    def only_rlbot(name):
        return object() if name == "rlbot" else None

    monkeypatch.setattr(doctor, "_module", lambda _name: (True, "1.68.0"))

    check = doctor.rlbot_check(find_spec=only_rlbot)
    assert check["ok"] is False
    assert "1.68.0" in check["detail"]
    assert doctor.RLBOT_REQUIREMENT in check["detail"]
    assert "flat" in check["detail"]

    # Passende Version.
    check = doctor.rlbot_check(find_spec=lambda _name: object())
    assert check["ok"] is True
    assert "flat + managers" in check["detail"]


def test_rlbot_requirement_matches_pyproject():
    """Der Hinweis im Doctor darf nicht von der echten Abhängigkeit abweichen."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = pyproject["project"]["optional-dependencies"]["rlbot"]
    assert pinned == [doctor.RLBOT_REQUIREMENT]


def test_require_rlbot_explains_missing_package(monkeypatch):
    """Statt eines nackten ImportError gibt es einen Satz, der weiterhilft."""
    import pytest

    from rocketai import play

    monkeypatch.setattr(
        doctor,
        "rlbot_check",
        lambda: {
            "label": "RLBot-Python (für das echte Spiel)",
            "ok": False,
            "detail": "Version 1.68.0 passt nicht: flat fehlt.",
            "required": False,
        },
    )
    with pytest.raises(RuntimeError) as error:
        play.require_rlbot()
    assert "1.68.0" in str(error.value)
    assert "install.py" in str(error.value)
    assert "rlbot.flat" in str(error.value)


def test_require_rlbot_is_silent_with_the_right_version():
    from rocketai import play

    play.require_rlbot()  # installierte Reihe muss durchgehen
