from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_bridge_has_correlated_request_responses():
    godot = (ROOT / "sandbox/godot_project/scripts/rl_bridge.gd").read_text()
    python = (ROOT / "sandbox/godot_env.py").read_text()
    assert "current_request_id" in godot
    assert 'response["request_id"]' in godot
    assert "self._request_id" in python
    assert "response mismatch" in python
    assert "ACTION_CONTRACT_VERSION" in python
    assert '"action_contract"' in godot
