import pytest

from install import check_python


@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 13)])
def test_installer_accepts_supported_python_versions(version):
    check_python(version)


@pytest.mark.parametrize("version", [(3, 10), (3, 14), (3, 15)])
def test_installer_rejects_unsupported_python_versions(version):
    with pytest.raises(SystemExit):
        check_python(version)


def test_windows_shortcuts_point_at_the_real_scripts():
    """Doppelklick-Dateien müssen auf die vorhandenen Skripte zeigen (CRLF bleibt)."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    start = (root / "start.cmd").read_bytes()
    install = (root / "install.cmd").read_bytes()
    # cmd.exe mag CRLF; gemischte Zeilenenden sind der klassische Stolperstein.
    for name, raw in (("start.cmd", start), ("install.cmd", install)):
        assert b"\r\n" in raw, name
        assert b"\n" not in raw.replace(b"\r\n", b""), name

    start_text = start.decode("utf-8")
    assert ".venv\\Scripts\\python.exe" in start_text
    assert "start.py" in start_text
    # Derselbe Pfad, den start.py selbst erwartet.
    assert '"Scripts/python.exe"' in (root / "start.py").read_text(encoding="utf-8")
    assert (root / "start.py").is_file()

    install_text = install.decode("utf-8")
    assert "install.py" in install_text
    assert (root / "install.py").is_file()
