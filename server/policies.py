"""Saved-policy loading and inference for Arena matches and Playground bots."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from server.actions import HEURISTIC


def model_names(models_dir: Path) -> list[str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    return [path.name for path in sorted(models_dir.glob("*.zip"))]


def policy_choices(models_dir: Path) -> list[str]:
    return [HEURISTIC, *model_names(models_dir)]


# A loaded PPO policy costs tens of MB; keep only a few checkpoints resident.
MAX_CACHED_POLICIES = 3


class PolicyCache:
    """Cache ``PPO`` models keyed by path + mtime so reloads are cheap."""

    def __init__(self, max_entries: int = MAX_CACHED_POLICIES) -> None:
        self.max_entries = max(1, int(max_entries))
        self._cache: dict[str, tuple[Any | None, Any | None, str | None]] = {}

    def _remember(self, key: str, value: tuple[Any | None, Any | None, str | None]) -> None:
        """Insert into the cache and evict the least recently used entry."""
        self._cache.pop(key, None)
        self._cache[key] = value
        while len(self._cache) > self.max_entries:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)

    def load(self, model_name: str, models_dir: Path) -> tuple[Any | None, Any | None, str | None]:
        if model_name == HEURISTIC:
            return None, None, None
        path = models_dir / model_name
        if not path.exists():
            return None, None, f"Model file not found: {path.name}"
        try:
            cache_key = f"{path}:{path.stat().st_mtime_ns}"
        except OSError as exc:
            return None, None, f"Could not stat {path.name}: {exc}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self._remember(cache_key, cached)  # refresh the LRU position
            return cached
        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(path), device="cpu")
            normalizer = None
            sidecar = path.with_name(f"{path.stem}_vecnormalize.pkl")
            if sidecar.exists():
                with sidecar.open("rb") as handle:
                    normalizer = pickle.load(handle)
            result: tuple[Any | None, Any | None, str | None] = (model, normalizer, None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
            result = (None, None, f"Could not load {path.name}: {type(exc).__name__}: {exc}")
        self._remember(cache_key, result)
        return result

    def predict(
        self,
        model_name: str,
        observation: np.ndarray,
        agent_index: int,
        env: Any,
        models_dir: Path,
    ) -> tuple[np.ndarray, str | None]:
        """Return an action plus an optional human-readable error message."""
        if model_name == HEURISTIC:
            return env.heuristic_action(agent_index), None
        model, normalizer, error = self.load(model_name, models_dir)
        if model is None:
            return env.heuristic_action(agent_index), error
        policy_observation = observation
        if normalizer is not None and getattr(normalizer, "obs_rms", None) is not None:
            mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float32)
            variance = np.asarray(normalizer.obs_rms.var, dtype=np.float32)
            policy_observation = np.clip(
                (observation - mean) / np.sqrt(np.maximum(variance, 1e-8) + 1e-8), -10.0, 10.0
            ).astype(np.float32)
        action, _ = model.predict(policy_observation, deterministic=True)
        return np.asarray(action, dtype=np.int64).reshape(-1), None
