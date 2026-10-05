"""The policy network and its checkpoint format."""

from __future__ import annotations

import io
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import N_ACTIONS, OBS_SIZE

CHECKPOINT_FORMAT = "rocketai.policy.v1"


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` = Grafikkarte, wenn eine vorhanden ist, sonst CPU."""
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "device='cuda' verlangt eine Grafikkarte mit CUDA. Bitte 'auto' oder 'cpu' nutzen."
        )
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Alle Zufallsquellen auf denselben Startwert setzen (wiederholbare Läufe)."""
    import random as _random

    _random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))


def rng_payload(rng: np.random.Generator) -> dict[str, Any]:
    """Zustand der Zufallsgeneratoren für den Checkpoint (Fortsetzen ohne Sprung).

    Ohne diesen Zustand beginnt ein fortgesetztes Training mit neuen Zufallszahlen;
    dann sind zwei Läufe mit gleichem Startwert nicht mehr vergleichbar.
    """
    state = dict(rng.bit_generator.state)
    inner = dict(state.get("state", {}))
    return {
        "numpy_state": int(inner.get("state", 0)),
        "numpy_inc": int(inner.get("inc", 0)),
        "numpy_has_uint32": int(state.get("has_uint32", 0)),
        "numpy_uinteger": int(state.get("uinteger", 0)),
        "torch": torch.get_rng_state(),
    }


def set_rng_payload(rng: np.random.Generator, payload: dict[str, Any] | None) -> bool:
    """Zustand aus dem Checkpoint wiederherstellen. True, wenn es geklappt hat."""
    if not payload:
        return False
    try:
        state = dict(rng.bit_generator.state)
        state["state"] = {
            "state": int(payload["numpy_state"]),
            "inc": int(payload["numpy_inc"]),
        }
        state["has_uint32"] = int(payload.get("numpy_has_uint32", 0))
        state["uinteger"] = int(payload.get("numpy_uinteger", 0))
        rng.bit_generator.state = state
        torch_state = payload.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _mlp(sizes: Sequence[int], out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=False):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor (90 action logits) and critic (state value) MLPs."""

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        n_actions: int = N_ACTIONS,
        hidden_sizes: Sequence[int] = (512, 512, 256),
    ):
        super().__init__()
        self.obs_size = int(obs_size)
        self.n_actions = int(n_actions)
        self.hidden_sizes = [int(size) for size in hidden_sizes]
        self.actor = _mlp([self.obs_size, *self.hidden_sizes], self.n_actions)
        self.critic = _mlp([self.obs_size, *self.hidden_sizes], 1)
        # Small final actor layer: start close to a uniform policy.
        final = self.actor[-1]
        assert isinstance(final, nn.Linear)
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @property
    def device(self) -> torch.device:
        """Gerät des Netzes (``cuda``, wenn das Lernen auf der Grafikkarte läuft)."""
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch inference: returns (actions, log-probabilities, values) as numpy arrays."""
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.logits(tensor)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return (
            actions.cpu().numpy(),
            dist.log_prob(actions).cpu().numpy(),
            self.value(tensor).cpu().numpy(),
        )

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.logits(obs))
        return dist.log_prob(actions), dist.entropy(), self.value(obs)


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    *,
    steps: int,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomic write: a crash mid-save never leaves a truncated checkpoint behind."""
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "obs_size": model.obs_size,
        "n_actions": model.n_actions,
        "hidden_sizes": model.hidden_sizes,
        "steps": int(steps),
        "created": time.time(),
        "config": config,
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buffer.getvalue())
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a RocketAI checkpoint")
    return payload


def model_from_checkpoint(payload: dict[str, Any]) -> ActorCritic:
    model = ActorCritic(payload["obs_size"], payload["n_actions"], payload["hidden_sizes"])
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def load_policy(path: Path) -> ActorCritic:
    return model_from_checkpoint(load_checkpoint(path))
