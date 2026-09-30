"""Two-policy/frozen-opponent foundation for multi-agent training and evaluation.

Also hosts the Python half of the headless self-play bridge (rl_server
``--self-play`` mode): a deterministic two-slot match channel used by the
league to play checkpoint tournaments. The bridge is seeded per match and
never auto-resets, so a finished match is exactly the episode it was
seeded to be.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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


#: Opponent-sampling strategies for :class:`SelfPlayCoordinator`.
#:
#: ``uniform``
#:     Every pooled checkpoint equally likely. Safe default: cannot
#:     collapse onto a single opponent and overfit to it.
#: ``latest``
#:     Always the most recently added checkpoint. Fast progress, classic
#:     catastrophic forgetting against older versions.
#: ``recency_weighted``
#:     Linearly weighted toward recent checkpoints while keeping the whole
#:     pool reachable. Compromise between the two above.
#: ``round_robin``
#:     Deterministic cycle through the pool, no RNG at all. Gives every
#:     opponent exactly equal exposure, which is what evaluation wants.
OPPONENT_STRATEGIES = ("uniform", "latest", "recency_weighted", "round_robin")


class SelfPlayCoordinator:
    """Holds explicit learning and frozen slots and per-agent metrics.

    The actual match dynamics are implemented by Godot's
    SelfPlayEnvironmentCore; this class owns policy/checkpoint lifecycle and
    makes it difficult to accidentally update a frozen opponent.

    Opponent sampling is deterministic by construction: the coordinator
    owns a seeded ``random.Random`` and never touches the global ``random``
    module, so a run's opponent stream is reproducible from
    ``(seed, strategy, pool order)`` alone. Pool order is insertion order,
    which is why ``add_to_pool`` is append-only and de-duplicating.
    """

    STRATEGIES = OPPONENT_STRATEGIES

    def __init__(
        self,
        learning_slot: PolicySlot,
        opponent_slot: PolicySlot,
        opponent_pool: list[str] | None = None,
        strategy: str = "uniform",
        seed: int = 0,
    ) -> None:
        if learning_slot.name == opponent_slot.name:
            raise ValueError("self-play slots need distinct names")
        if strategy not in OPPONENT_STRATEGIES:
            raise ValueError(
                "unknown opponent strategy %r; expected one of %s"
                % (strategy, ", ".join(OPPONENT_STRATEGIES))
            )
        self.learning_slot = learning_slot
        self.opponent_slot = opponent_slot
        self.opponent_slot.frozen = True
        self.opponent_pool: list[str] = list(opponent_pool or [])
        if opponent_slot.checkpoint and opponent_slot.checkpoint not in self.opponent_pool:
            self.opponent_pool.append(opponent_slot.checkpoint)
        self.opponent_strategy = strategy
        self.opponent_seed = int(seed)
        self._rng = random.Random(self.opponent_seed)
        self._draws = 0
        self.opponent_history: list[str] = []
        self.metrics: dict[str, list[dict[str, Any]]] = {
            learning_slot.name: [],
            opponent_slot.name: [],
        }

    @classmethod
    def from_config(
        cls,
        config: Any,
        learning_name: str = "learner",
        opponent_name: str = "frozen_opponent",
    ) -> SelfPlayCoordinator:
        """Builds a coordinator from a :class:`~sandboxai.config.SelfPlayConfig`.

        Keeps the opponent-selection rule and its seed in the run's config
        snapshot instead of at the call site, so the opponent stream is
        reproducible from the saved configuration alone.
        """
        return cls(
            PolicySlot(name=learning_name, seed=int(getattr(config, "seed", 1234))),
            PolicySlot(
                name=opponent_name,
                checkpoint=str(getattr(config, "opponent_checkpoint", "") or ""),
                frozen=True,
                seed=int(getattr(config, "seed", 1234)),
            ),
            opponent_pool=list(getattr(config, "opponent_pool", []) or []),
            strategy=str(getattr(config, "opponent_strategy", "uniform")),
            seed=int(getattr(config, "opponent_seed", 0)),
        )

    def add_to_pool(self, checkpoint_path: str | Path) -> None:
        path_str = str(checkpoint_path)
        if path_str not in self.opponent_pool:
            self.opponent_pool.append(path_str)

    def reset_sampling(self) -> None:
        """Rewinds the opponent stream to its seeded start.

        Mirrors :meth:`League.reset_sampling` so a resumed or repeated run
        draws exactly the same opponents again.
        """
        self._rng = random.Random(self.opponent_seed)
        self._draws = 0
        self.opponent_history.clear()

    def choose_opponent_checkpoint(
        self, rng: random.Random | None = None, strategy: str | None = None
    ) -> str | None:
        """Picks the next opponent checkpoint without loading a model.

        Split out from :meth:`sample_opponent` so the selection rule can be
        tested (and logged into a run manifest) without stable-baselines3
        or an actual checkpoint file.
        """
        if not self.opponent_pool:
            return None
        chosen_strategy = strategy or self.opponent_strategy
        if chosen_strategy not in OPPONENT_STRATEGIES:
            raise ValueError(f"unknown opponent strategy: {chosen_strategy}")
        picker = rng if rng is not None else self._rng
        pool = self.opponent_pool
        if chosen_strategy == "latest":
            chosen = pool[-1]
        elif chosen_strategy == "round_robin":
            chosen = pool[self._draws % len(pool)]
        elif chosen_strategy == "recency_weighted":
            weights = [float(index + 1) for index in range(len(pool))]
            chosen = picker.choices(pool, weights=weights, k=1)[0]
        else:
            chosen = picker.choice(pool)
        self._draws += 1
        self.opponent_history.append(chosen)
        return chosen

    def sample_opponent(
        self,
        device: str = "cpu",
        rng: random.Random | None = None,
        strategy: str | None = None,
        load: bool = True,
    ) -> PolicySlot:
        """Selects (and by default loads) the next frozen opponent.

        ``rng`` overrides the coordinator's own generator for callers that
        manage seeding themselves (the league does). It is never the global
        ``random`` module: an unseeded default would make a training run
        irreproducible in a way that is invisible in the logs.
        """
        chosen = self.choose_opponent_checkpoint(rng=rng, strategy=strategy)
        if chosen is None:
            return self.opponent_slot
        self.opponent_slot.checkpoint = chosen
        if load:
            self.opponent_slot.load(device)
        return self.opponent_slot

    def sampling_snapshot(self, history_limit: int = 32) -> dict[str, Any]:
        """Reproducibility record for run manifests and eval reports."""
        return {
            "strategy": self.opponent_strategy,
            "seed": self.opponent_seed,
            "draws": self._draws,
            "pool_size": len(self.opponent_pool),
            "pool": list(self.opponent_pool),
            "recent_opponents": list(self.opponent_history[-history_limit:]),
        }

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
                "mean_reward": sum(float(row.get("episode_reward", 0.0)) for row in rows)
                / len(rows)
                if rows
                else 0.0,
                "mean_kills": sum(float(row.get("kills", 0.0)) for row in rows) / len(rows)
                if rows
                else 0.0,
                "mean_accuracy": sum(float(row.get("accuracy", 0.0)) for row in rows) / len(rows)
                if rows
                else 0.0,
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

    def __enter__(self) -> SelfPlayBatchClient:
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
