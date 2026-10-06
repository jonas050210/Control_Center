"""Evaluate a PPO checkpoint (or a random policy) against the built-in bot.

Usage (from the repository root, inside .venv):

    python3 tools/evaluate_policy.py --random --episodes 30
    python3 tools/evaluate_policy.py --model best_model.zip --episodes 50
    python3 tools/evaluate_policy.py --model best_model.zip --vision exact --bots full

The script reports win rate, average time-to-kill (only confirmed kills) and the
perception statistics, so a training run can be judged against the random
baseline. It never trains - it only plays episodes.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.shooter_env import OBSERVATION_VERSION, ShooterEnv, VISION_MODES  # noqa: E402


def build_env(args: argparse.Namespace, seed: int) -> ShooterEnv:
    # ``--phase`` reproduces the curriculum conditions a checkpoint was trained
    # in (the early phases start inside weapon range). Default is phase 4, i.e.
    # the real map spawns.
    curriculum = args.phase < 4
    return ShooterEnv(
        map_name=args.map,
        weapon_name=args.weapon,
        opponent_weapon=args.opponent_weapon,
        curriculum=curriculum,
        curriculum_phase=args.phase,
        opponent_mode=args.bots,
        frame_skip=4,
        max_episode_seconds=args.episode_seconds,
        vision_mode=args.vision,
        seed=seed,
    )


def load_policy(args: argparse.Namespace):
    if args.random or not args.model:
        return None, None
    # Accept a bare name ("best_model.zip"), a relative path or an absolute path.
    candidate = Path(args.model)
    model_path = candidate if candidate.is_absolute() or candidate.parent != Path(".") \
        else PROJECT_ROOT / "models" / candidate
    if not model_path.exists():
        raise SystemExit(f"Checkpoint not found: {model_path}")
    meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("observation_version", OBSERVATION_VERSION))
        if version != OBSERVATION_VERSION:
            raise SystemExit(
                f"{model_path.name} was trained for observation version {version}; "
                f"this build uses {OBSERVATION_VERSION}. Retrain before evaluating."
            )
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    normalizer = None
    sidecar = model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")
    if sidecar.exists():
        import pickle

        with sidecar.open("rb") as handle:
            normalizer = pickle.load(handle)
    return model, normalizer


def normalize(observation, normalizer):
    if normalizer is None or getattr(normalizer, "obs_rms", None) is None:
        return observation
    import numpy as np

    mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float32)
    variance = np.asarray(normalizer.obs_rms.var, dtype=np.float32)
    return np.clip((observation - mean) / np.sqrt(np.maximum(variance, 1e-8) + 1e-8),
                   -10.0, 10.0).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a NEURAL ARENA policy.")
    parser.add_argument("--model", default=None, help="Checkpoint in models/ (z. B. best_model.zip)")
    parser.add_argument("--random", action="store_true", help="Zufallspolitik statt Modell")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--map", default="Dust")
    parser.add_argument("--weapon", default="AK-47")
    parser.add_argument("--opponent-weapon", dest="opponent_weapon", default="AK-47")
    parser.add_argument("--phase", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Curriculum-Bedingungen nachstellen (1 = kurzer Abstand)")
    parser.add_argument("--bots", default="full",
                        choices=["stationary", "walker", "shooter", "full"])
    parser.add_argument("--vision", default="coarse_los", choices=list(VISION_MODES))
    parser.add_argument("--episode-seconds", dest="episode_seconds", type=float, default=60.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args(argv)

    model, normalizer = load_policy(args)
    wins = draws = losses = 0
    ttks: list[float] = []
    frames: list[int] = []
    blind_ratios: list[float] = []
    contact_episodes = 0
    started = time.monotonic()

    for episode in range(max(1, args.episodes)):
        env = build_env(args, args.seed + episode)
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(normalize(observation, normalizer), deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        metrics = info.get("episode_metrics", {})
        win = bool(metrics.get("win"))
        draw = bool(metrics.get("draw"))
        wins += int(win)
        draws += int(draw)
        losses += int(not win and not draw)
        if metrics.get("killed"):
            ttks.append(float(metrics.get("ttk", 0.0)))
        frames.append(int(info.get("episode", {}).get("l", 0)) or env.physics_frames)
        ratio = float(metrics.get("player_blind_ratio", 0.0))
        blind_ratios.append(ratio)
        contact_episodes += int(1.0 - ratio > 0.0)
        env.close()

    total = max(1, args.episodes)
    print(f"Modell         : {'Zufallspolitik' if model is None else args.model}")
    print(f"Wahrnehmung    : {args.vision} (observation version {OBSERVATION_VERSION})")
    print(f"Karte/Gegner   : {args.map} · Bot '{args.bots}' · {args.weapon} vs {args.opponent_weapon}"
          + (f" · Curriculum-Phase {args.phase}" if args.phase < 4 else ""))
    print(f"Episoden       : {total}")
    print(f"Siege          : {wins} ({wins / total:.0%})")
    print(f"Unentschieden  : {draws} ({draws / total:.0%})")
    print(f"Niederlagen    : {losses} ({losses / total:.0%})")
    print(f"Ø Episodendauer: {statistics.fmean(frames):.0f} Physics-Frames "
          f"({statistics.fmean(frames) * env.dt:.1f}s)")
    print(f"Episoden mit Sichtkontakt: {contact_episodes}/{total}")
    print(f"Ø Blindanteil  : {statistics.fmean(blind_ratios):.0%} der Physik-Frames ohne Sicht")
    if ttks:
        print(f"Ø Time-to-Kill : {statistics.fmean(ttks):.2f}s (nur bestätigte Kills, n={len(ttks)})")
    else:
        print("Ø Time-to-Kill : kein Kill")
    print(f"Laufzeit       : {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
