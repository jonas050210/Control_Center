import pytest


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Every test gets its own empty runs folder."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("ROCKETAI_RUNS", str(runs))
    return runs
