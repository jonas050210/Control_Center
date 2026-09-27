"""Two-policy/frozen-opponent foundation for multi-agent training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any


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

    def add_to_pool(self, checkpoint_path: str | Path) -> None:
        path_str = str(checkpoint_path)
        if path_str not in self.opponent_pool:
            self.opponent_pool.append(path_str)

    def sample_opponent(self, device: str = "cpu", rng: random.Random | None = None) -> PolicySlot:
        if not self.opponent_pool:
            return self.opponent_slot
        picker = rng or random
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
