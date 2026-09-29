"""Two-policy/frozen-opponent foundation for multi-agent training and evaluation.

Also hosts the Python half of the headless self-play bridge (rl_server
``--self-play`` mode): a deterministic two-slot match channel used by the
league to play checkpoint tournaments. The bridge is seeded per match and
never auto-resets, so a finished match is exactly the episode it was
seeded to be.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable


@dataclass
class PolicySlot:
    name: str
    checkpoint: str = ""
    frozen: bool = False
    seed: int = 1234
    model: Any = None

    def load(self, device: str = "cpu") -> Any:
        if not self.checkpoint:
            raise ValueError(f"policy slot {self.name} has no checkpoint")
        try:
            from stable_baselines3 import PPO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("self-play checkpoint loading requires stable-baselines3") from exc
        self.model = PPO.load(Path(self.checkpoint), device=device)
        return self.model

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError(f"policy slot {self.name} is not loaded")
        return self.model.predict(observation, deterministic=deterministic)


class SelfPlayCoordinator:
    """Holds explicit learning and frozen slots and per-agent metrics.

    The actual match dynamics are implemented by Godot's
    SelfPlayEnvironmentCore; this class owns policy/checkpoint lifecycle and
    makes it difficult to accidentally update a frozen opponent.
    """

    def __init__(
        self,
        learning_slot: PolicySlot,
        opponent_slot: PolicySlot,
        opponent_pool: list[str] | None = None,
    ) -> None:
        if learning_slot.name == opponent_slot.name:
            raise ValueError("self-play slots need distinct names")
        self.learning_slot = learning_slot
        self.opponent_slot = opponent_slot
        self.opponent_slot.frozen = True
        self.opponent_pool: list[str] = list(opponent_pool or [])
        if opponent_slot.checkpoint and opponent_slot.checkpoint not in self.opponent_pool:
            self.opponent_pool.append(opponent_slot.checkpoint)
        self.metrics: dict[str, list[dict[str, Any]]] = {learning_slot.name: [], opponent_slot.name: []}
        # Lazily created, seeded fallback RNG used only when a caller does not
        # supply its own `rng` to sample_opponent(). Seeded from the opponent
        # slot's own `seed` field rather than the unseeded global `random`
        # module, so a coordinator built with the same slots draws the same
        # opponent sequence on every run -- a training-determinism hazard
        # otherwise, since the global module seeds from OS entropy.
        self._fallback_rng: random.Random | None = None

    def add_to_pool(self, checkpoint_path: str | Path) -> None:
        path_str = str(checkpoint_path)
        if path_str not in self.opponent_pool:
            self.opponent_pool.append(path_str)

    def sample_opponent(self, device: str = "cpu", rng: random.Random | None = None) -> PolicySlot:
        if not self.opponent_pool:
            return self.opponent_slot
        picker: random.Random
        if rng is not None:
            picker = rng
        else:
            if self._fallback_rng is None:
                self._fallback_rng = random.Random(self.opponent_slot.seed)
            picker = self._fallback_rng
        chosen = picker.choice(self.opponent_pool)
        self.opponent_slot.checkpoint = chosen
        self.opponent_slot.load(device)
        return self.opponent_slot

    def load_opponent(self, device: str = "cpu") -> Any:
        return self.opponent_slot.load(device)

    def actions_for_observations(self, observations, deterministic: bool = True) -> list[Any]:
        if len(observations) != 2:
            raise ValueError("self-play needs one observation for each policy slot")
        learning_action, _ = self.learning_slot.predict(observations[0], deterministic)
        opponent_action, _ = self.opponent_slot.predict(observations[1], deterministic)
        return [learning_action, opponent_action]

    def record_match(self, learning_info: dict[str, Any], opponent_info: dict[str, Any]) -> None:
        self.metrics[self.learning_slot.name].append(dict(learning_info))
        self.metrics[self.opponent_slot.name].append(dict(opponent_info))

    def summary(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, rows in self.metrics.items():
            wins = sum(float(row.get("win", 0.0)) for row in rows)
            output[name] = {
                "matches": len(rows),
                "win_rate": wins / len(rows) if rows else 0.0,
                "mean_reward": sum(float(row.get("episode_reward", 0.0)) for row in rows) / len(rows) if rows else 0.0,
                "mean_kills": sum(float(row.get("kills", 0.0)) for row in rows) / len(rows) if rows else 0.0,
                "mean_accuracy": sum(float(row.get("accuracy", 0.0)) for row in rows) / len(rows) if rows else 0.0,
            }
        return output


# ---------------------------------------------------------------------------
# Headless self-play bridge client (rl_server --self-play mode)
# ---------------------------------------------------------------------------


class SelfPlayBatchClient:
    """Batch of deterministic two-agent matches over the JSON-lines bridge.

    Wire shapes mirror the normal environment batch with a slot axis of 2:
    ``reset`` returns ``[[obs_a, obs_b], ...]``, ``step`` takes
    ``[[action_a, action_b], ...]`` and returns both slots'
    observations/rewards/infos. Every reset is explicitly seeded (slot B
    derives +1000003 inside the environment), so a match is reproducible
    from its seed alone; there is no auto-reset mid-match.
    """

    def __init__(self, **kwargs: Any) -> None:
        from .contract import OBSERVATION_FIELD_COUNT
        from .godot_env import GodotProcessTransport

        kwargs["self_play"] = True
        self.transport = GodotProcessTransport(**kwargs)
        self.environment_count = int(kwargs.get("environment_count", 1))
        spaces = self.transport.spaces
        dim = int(spaces["observation_space"]["size"])
        if dim != OBSERVATION_FIELD_COUNT:
            self.transport.close()
            raise RuntimeError(
                "self-play bridge reports a %d-float observation space, but the "
                "contract defines %d floats" % (dim, OBSERVATION_FIELD_COUNT)
            )

    def reset(self, seed: int):
        """Resets every environment with seed + index. Returns observation
        pairs ``[[obs_a, obs_b], ...]`` as float lists."""
        response = self.transport.request({"cmd": "reset", "seed": int(seed)})
        return response.get("observations", [])

    def step(self, action_pairs: list[Any]):
        response = self.transport.request({"cmd": "step", "actions": action_pairs})
        return (
            response.get("observations", []),
            response.get("rewards", []),
            response.get("dones", []),
            response.get("infos", []),
        )

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "SelfPlayBatchClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _as_action_list(action: Any) -> list[int]:
    if hasattr(action, "tolist"):
        action = action.tolist()
    if action and isinstance(action[0], (list, tuple)):
        action = list(action[0])
    return [int(value) for value in action]


def play_self_play_match(
    client: SelfPlayBatchClient,
    predict_a: Callable[[Any], Any],
    predict_b: Callable[[Any], Any],
    seed: int,
    max_steps: int = 3600,
) -> dict[str, Any]:
    """Plays one deterministic match and returns both slots' results.

    ``predict_*`` are callables observation -> action (6-component
    MultiDiscrete). The outcome is taken from the environment's own
    episode metrics, never re-derived here: ``win`` flags decide the
    score, and a timeout without a winner is a draw with ``truncated``
    set, matching the main environment's loss/timeout accounting.
    """
    pairs = client.reset(seed)
    if not pairs or len(pairs[0]) != 2:
        raise RuntimeError(
            "self-play bridge returned malformed observations on reset: "
            "expected [[obs_a, obs_b], ...]; is Godot running with rl_server.gd --self-play 1?"
        )
    observations = pairs[0]
    metrics_a: dict[str, Any] = {}
    metrics_b: dict[str, Any] = {}
    done_reason = ""
    truncated = False
    steps = 0
    for _ in range(max_steps):
        action_a = _as_action_list(predict_a(observations[0]))
        action_b = _as_action_list(predict_b(observations[1]))
        observations, _rewards, dones, infos = client.step([[action_a, action_b]])
        if not observations or len(observations[0]) != 2:
            raise RuntimeError(
                "self-play bridge returned malformed observations on step: expected [[obs_a, obs_b], ...]"
            )
        observations = observations[0]
        steps += 1
        info_a = infos[0][0] if infos and infos[0] else {}
        info_b = infos[0][1] if infos and len(infos[0]) > 1 else {}
        metrics_a = dict(info_a.get("metrics", metrics_a))
        metrics_b = dict(info_b.get("metrics", metrics_b))
        if dones[0]:
            done_reason = str(info_a.get("done_reason", ""))
            truncated = done_reason == "timeout"
            break
    else:
        truncated = True
        done_reason = "timeout"
    won_a = bool(metrics_a.get("win", False))
    won_b = bool(metrics_b.get("win", False))
    if won_a and not won_b:
        score_a = 1.0
    elif won_b and not won_a:
        score_a = 0.0
    else:
        score_a = 0.5
    metrics_a.setdefault("episode_length", steps)
    metrics_b.setdefault("episode_length", steps)
    return {
        "score_a": score_a,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "truncated": truncated,
        "done_reason": done_reason,
        "steps": steps,
        "seed": seed,
    }
