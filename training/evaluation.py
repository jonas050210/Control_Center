"""Play episodes with a policy and report honest combat metrics.

Used by ``tools/evaluate_policy.py`` for manual checks and by the trainer for the
automatic evaluation right after a run. Deliberately independent of Stable
Baselines: the pure functions work with any ``predict``-like callable, so the
whole module is testable without PyTorch.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from env.shooter_env import OBSERVATION_VERSION, ShooterEnv, VISION_MODES

DEFAULT_EPISODES = 30
DEFAULT_BOTS = ("stationary", "walker", "shooter", "full")


class RunningStatistics:
    """Minimal stand-in for a running mean/variance accumulator."""

    __slots__ = ("mean", "var")

    def __init__(self, mean: Any, var: Any) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)


class ObservationStatistics:
    """The observation statistics of a checkpoint, without Stable Baselines.

    ``VecNormalize.save`` drops the wrapped environment, so an unpickled
    ``VecNormalize`` raises ``RecursionError`` the moment a *missing* attribute is
    looked up. Only the plain numbers are kept here, which is all inference needs.
    """

    __slots__ = ("obs_rms", "norm_obs", "clip_obs")

    def __init__(self, obs_rms: RunningStatistics, norm_obs: bool, clip_obs: float) -> None:
        self.obs_rms = obs_rms
        self.norm_obs = bool(norm_obs)
        self.clip_obs = float(clip_obs)


def _instance_state(obj: Any) -> dict[str, Any]:
    """Return ``obj.__dict__`` without triggering a lazy ``__getattr__``."""
    state = getattr(obj, "__dict__", None)
    return state if isinstance(state, dict) else {}


def _statistic(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute, preferring the instance dict over ``__getattr__``.

    Some wrappers (SB3's ``VecEnv``) recurse infinitely during attribute lookup
    once their internal environment was removed while pickling.
    """
    state = _instance_state(obj)
    if name in state:
        return state[name]
    try:
        return getattr(obj, name, default)
    except RecursionError:
        return default


def load_normalizer(path: str | Path) -> ObservationStatistics | None:
    """Read the ``*_vecnormalize.pkl`` sidecar and keep only its statistics."""
    import pickle

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    state = _instance_state(payload)
    statistics = _statistic(payload, "obs_rms", None)
    if statistics is None:
        return None
    mean = _statistic(statistics, "mean", None)
    variance = _statistic(statistics, "var", None)
    if mean is None or variance is None:
        return None
    return ObservationStatistics(
        obs_rms=RunningStatistics(mean, variance),
        norm_obs=bool(_statistic(payload, "norm_obs", False)),
        clip_obs=float(_statistic(payload, "clip_obs", 10.0) or 10.0),
    )


def resolve_model_path(name: str | Path) -> Path:
    """Accept a bare name (``best_model.zip``), a relative or an absolute path."""
    from server.config import MODELS_DIR

    candidate = Path(name)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    return Path(MODELS_DIR) / candidate


def load_policy(model_path: str | Path | None, *, project_root: Path | None = None) -> tuple[Any, Any]:
    """Load a PPO checkpoint plus its VecNormalize sidecar; ``None`` = random policy.

    The ``observation_version`` stamp is enforced: an old checkpoint would still
    load, but every number in its observation would mean something else.
    """
    if model_path is None:
        return None, None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    meta_path = path.with_name(f"{path.stem}_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("observation_version", OBSERVATION_VERSION))
        if version != OBSERVATION_VERSION:
            raise ValueError(
                f"{path.name} was trained for observation version {version}; this build "
                f"uses {OBSERVATION_VERSION}. Retrain before evaluating."
            )
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    normalizer = None
    sidecar = path.with_name(f"{path.stem}_vecnormalize.pkl")
    if sidecar.exists():
        normalizer = load_normalizer(sidecar)
    return model, normalizer


def normalize(observation: np.ndarray, normalizer: Any) -> np.ndarray:
    """Apply the saved ``VecNormalize`` observation statistics, if any.

    Only checkpoints trained with ``norm_obs=True`` expect rescaled values; new
    runs train on the raw, already bounded perception vector.
    """
    if normalizer is None:
        return observation
    if not bool(_statistic(normalizer, "norm_obs", False)):
        return observation
    statistics = _statistic(normalizer, "obs_rms", None)
    if statistics is None:
        return observation
    mean = np.asarray(_statistic(statistics, "mean", 0.0), dtype=np.float32)
    variance = np.asarray(_statistic(statistics, "var", 1.0), dtype=np.float32)
    limit = float(_statistic(normalizer, "clip_obs", 10.0) or 10.0)
    return np.clip((observation - mean) / np.sqrt(np.maximum(variance, 1e-8) + 1e-8),
                   -limit, limit).astype(np.float32)


def build_env(*, map_name: str = "Dust", weapon: str = "AK-47", opponent_weapon: str = "AK-47",
              bot: str = "full", vision_mode: str = "coarse_los", phase: int = 4,
              episode_seconds: float = 60.0, frame_skip: int = 4, seed: int = 1) -> ShooterEnv:
    """Create an evaluation environment.

    ``phase < 4`` reproduces the curriculum conditions (spawn distance) a
    checkpoint was trained in - phase 4 means the real map spawns.
    """
    if vision_mode not in VISION_MODES:
        raise ValueError(f"Unknown vision mode {vision_mode!r}.")
    phase = min(4, max(1, int(phase)))
    return ShooterEnv(
        map_name=map_name,
        weapon_name=weapon,
        opponent_weapon=opponent_weapon,
        curriculum=phase < 4,
        curriculum_phase=phase,
        opponent_mode=bot,
        frame_skip=frame_skip,
        max_episode_seconds=episode_seconds,
        vision_mode=vision_mode,
        seed=seed,
    )


def episode_action(model: Any, normalizer: Any, observation: np.ndarray,
                   deterministic: bool = True) -> np.ndarray:
    if model is None:
        return np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    prepared = normalize(np.asarray(observation, dtype=np.float32), normalizer)
    action, _state = model.predict(prepared, deterministic=deterministic)
    return np.asarray(action, dtype=np.int64)


def evaluate(*, model_path: str | Path | None = None, model: Any = None, normalizer: Any = None,
             episodes: int = DEFAULT_EPISODES, map_name: str = "Dust", weapon: str = "AK-47",
             opponent_weapon: str = "AK-47", bot: str = "full", vision_mode: str = "coarse_los",
             phase: int = 4, episode_seconds: float = 60.0, frame_skip: int = 4,
             seed: int = 1, deterministic: bool = True) -> dict[str, Any]:
    """Play ``episodes`` matches and return wins, draws, losses and fight statistics."""
    if model is None and model_path is not None:
        model, normalizer = load_policy(model_path)
    episodes = max(1, int(episodes))
    wins = draws = losses = 0
    ttks: list[float] = []
    frames: list[int] = []
    blind_ratios: list[float] = []
    accuracy: list[float] = []
    contacts = 0
    started = time.monotonic()
    env = build_env(map_name=map_name, weapon=weapon, opponent_weapon=opponent_weapon, bot=bot,
                    vision_mode=vision_mode, phase=phase, episode_seconds=episode_seconds,
                    frame_skip=frame_skip, seed=seed)
    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            info: dict[str, Any] = {}
            while True:
                action = episode_action(model, normalizer, observation, deterministic)
                observation, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            metrics = info.get("episode_metrics", {})
            killed = bool(metrics.get("killed"))
            if killed:
                wins += 1
                ttks.append(float(metrics.get("ttk", 0.0)))
            elif metrics.get("win"):
                wins += 1          # decision on health at the time limit
            elif metrics.get("draw"):
                draws += 1
            else:
                losses += 1
            frames.append(int(env.physics_frames))
            blind_ratios.append(float(metrics.get("player_blind_ratio", 0.0)))
            accuracy.append(float(metrics.get("accuracy", 0.0)))
            if float(metrics.get("player_visible_steps", 0)) > 0:
                contacts += 1
    finally:
        env.close()
    runtime = time.monotonic() - started
    return {
        "policy": "random" if model is None else str(model_path or "checkpoint"),
        "map": map_name,
        "bot": bot,
        "weapon": weapon,
        "opponent_weapon": opponent_weapon,
        "vision_mode": vision_mode,
        "phase": phase,
        "episodes": episodes,
        "wins": wins,
        "kill_wins": len(ttks),
        "draws": draws,
        "losses": losses,
        "win_rate": wins / episodes,
        "kill_rate": len(ttks) / episodes,
        "avg_ttk": float(np.mean(ttks)) if ttks else math.nan,
        "avg_frames": float(np.mean(frames)) if frames else 0.0,
        "avg_seconds": float(np.mean(frames)) * frame_skip * (1.0 / 60.0) if frames else 0.0,
        "contact_episodes": contacts,
        "avg_blind_ratio": float(np.mean(blind_ratios)) if blind_ratios else 0.0,
        "avg_accuracy": float(np.mean(accuracy)) if accuracy else 0.0,
        "runtime_seconds": runtime,
    }


def evaluate_matrix(*, model_path: str | Path | None = None, bots: tuple[str, ...] = DEFAULT_BOTS,
                    phases: tuple[int, ...] = (4,), episodes: int = 8,
                    vision_mode: str = "coarse_los", map_name: str = "Dust",
                    episode_seconds: float = 60.0, frame_skip: int = 4,
                    seed: int = 1) -> dict[str, Any]:
    """Evaluate one checkpoint against several opponents (and curriculum phases)."""
    model, normalizer = (None, None)
    if model_path is not None:
        model, normalizer = load_policy(model_path)
    runs: list[dict[str, Any]] = []
    for phase in phases:
        for bot in bots:
            result = evaluate(model=model, normalizer=normalizer, episodes=episodes, bot=bot,
                              phase=phase, vision_mode=vision_mode, map_name=map_name,
                              episode_seconds=episode_seconds, frame_skip=frame_skip, seed=seed)
            result["policy"] = "random" if model is None else str(model_path)
            runs.append(result)
    best = max(runs, key=lambda run: (run["kill_rate"], run["win_rate"])) if runs else {}
    return {
        "model": None if model_path is None else str(model_path),
        "episodes_per_matchup": episodes,
        "runs": runs,
        "best": {"bot": best.get("bot"), "phase": best.get("phase"),
                 "win_rate": best.get("win_rate"), "kill_rate": best.get("kill_rate")} if best else {},
        "kill_rate_overall": float(np.mean([run["kill_rate"] for run in runs])) if runs else 0.0,
        "win_rate_overall": float(np.mean([run["win_rate"] for run in runs])) if runs else 0.0,
    }


def summarize(result: dict[str, Any]) -> str:
    """One-line summary for logs and job snapshots."""
    return (f"{result['bot']}/Phase {result['phase']}: "
            f"{result['win_rate']:.0%} Siege, davon {result['kill_rate']:.0%} Kills, "
            f"{result['avg_blind_ratio']:.0%} blind")


def policy_button(model_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Small adapter that turns a checkpoint into an ``observation -> action`` callable."""
    model, normalizer = load_policy(model_path)

    def act(observation: np.ndarray) -> np.ndarray:
        return episode_action(model, normalizer, observation)

    return act
