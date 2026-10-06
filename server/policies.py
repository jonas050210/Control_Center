"""Saved-policy loading and inference for Arena matches and Playground bots."""

from __future__ import annotations

import json
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

    @staticmethod
    def _observation_guard(path: Path) -> str | None:
        """Reject checkpoints trained on an older observation layout.

        Feeding a policy the wrong layout silently produces nonsense actions, so
        a version stamp next to the model is checked before loading. Missing
        metadata is accepted for the models that ship with the repository.
        """
        from env.shooter_env import OBSERVATION_VERSION

        meta_path = path.with_name(f"{path.stem}_meta.json")
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as exc:
            return f"Could not read {meta_path.name}: {exc}"
        version = int(meta.get("observation_version", OBSERVATION_VERSION))
        if version != OBSERVATION_VERSION:
            return (
                f"{path.name} was trained for observation version {version}; "
                f"this build uses version {OBSERVATION_VERSION}. Please retrain."
            )
        return None

    def load(self, model_name: str, models_dir: Path) -> tuple[Any | None, Any | None, str | None]:
        if model_name == HEURISTIC:
            return None, None, None
        path = models_dir / model_name
        if not path.exists():
            return None, None, f"Model file not found: {path.name}"
        guard = self._observation_guard(path)
        if guard is not None:
            return None, None, guard
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
                # Share the loader with the trainer: it survives sidecars whose
                # wrapped vector env was stripped while pickling.
                from training.evaluation import load_normalizer

                normalizer = load_normalizer(sidecar)
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
        from training.evaluation import normalize

        # ``norm_obs=False`` checkpoints (the current default) expect the raw
        # perception vector; only legacy runs were trained on rescaled values.
        policy_observation = normalize(np.asarray(observation, dtype=np.float32), normalizer)
        action, _ = model.predict(policy_observation, deterministic=True)
        return np.asarray(action, dtype=np.int64).reshape(-1), None
