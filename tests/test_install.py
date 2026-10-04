import pytest

from install import check_python


@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 13)])
def test_installer_accepts_supported_python_versions(version):
    check_python(version)


@pytest.mark.parametrize("version", [(3, 10), (3, 14), (3, 15)])
def test_installer_rejects_unsupported_python_versions(version):
    with pytest.raises(SystemExit):
        check_python(version)
