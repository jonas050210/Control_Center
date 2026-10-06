"""Evaluate a PPO checkpoint (or a random policy) against the built-in bot.

Usage (from the repository root, inside .venv):

    python3 tools/evaluate_policy.py --random --episodes 30
    python3 tools/evaluate_policy.py --model best_model.zip --episodes 50
    python3 tools/evaluate_policy.py --model best_model.zip --phase 1 --bots stationary
    python3 tools/evaluate_policy.py --model best_model.zip --bots stationary,walker,full --json

The script reports wins, draws, losses, the average time-to-kill (confirmed kills
only) and the perception statistics, so a training run can be judged against the
random baseline. It never trains - it only plays episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.shooter_env import OBSERVATION_VERSION, OPPONENT_MODES, VISION_MODES  # noqa: E402
from training.evaluation import evaluate, resolve_model_path  # noqa: E402


def format_result(result: dict) -> str:
    lines = [
        f"Modell         : {result['policy']}",
        f"Wahrnehmung    : {result['vision_mode']} (observation version {OBSERVATION_VERSION})",
        f"Karte/Gegner   : {result['map']} · Bot '{result['bot']}' · "
        f"{result['weapon']} vs {result['opponent_weapon']}"
        + (f" · Curriculum-Phase {result['phase']}" if result["phase"] < 4 else ""),
        f"Episoden       : {result['episodes']}",
        f"Siege          : {result['wins']} ({result['win_rate']:.0%})",
        f"Unentschieden  : {result['draws']} ({result['draws'] / result['episodes']:.0%})",
        f"Niederlagen    : {result['losses']} ({result['losses'] / result['episodes']:.0%})",
        f"Ø Episodendauer: {result['avg_frames']:.0f} Physics-Frames ({result['avg_seconds']:.1f}s)",
        f"Episoden mit Sichtkontakt: {result['contact_episodes']}/{result['episodes']}",
        f"Ø Blindanteil  : {result['avg_blind_ratio']:.0%} der Physik-Frames ohne Sicht",
    ]
    if result["kill_wins"]:
        lines.append(f"Ø Time-to-Kill : {result['avg_ttk']:.2f}s "
                     f"(nur bestätigte Kills, n={result['kill_wins']})")
    else:
        lines.append("Ø Time-to-Kill : kein Kill")
    lines.append(f"Ø Trefferquote : {result['avg_accuracy']:.1%}")
    lines.append(f"Laufzeit       : {result['runtime_seconds']:.1f}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a NEURAL ARENA policy.")
    parser.add_argument("--model", default=None, help="Checkpoint in models/ (z. B. best_model.zip)")
    parser.add_argument("--random", action="store_true", help="Zufallspolitik statt Modell")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--map", default="Dust")
    parser.add_argument("--weapon", default="AK-47")
    parser.add_argument("--opponent-weapon", dest="opponent_weapon", default="AK-47")
    parser.add_argument("--bots", default="full",
                        help="Komma-Liste: " + ",".join(OPPONENT_MODES))
    parser.add_argument("--phase", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Curriculum-Bedingungen nachstellen (1 = kurzer Abstand)")
    parser.add_argument("--vision", default="coarse_los", choices=list(VISION_MODES))
    parser.add_argument("--episode-seconds", dest="episode_seconds", type=float, default=60.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--json", action="store_true", help="Rohdaten als JSON ausgeben")
    args = parser.parse_args(argv)

    bots = tuple(name.strip() for name in args.bots.split(",") if name.strip())
    unknown = [name for name in bots if name not in OPPONENT_MODES]
    if unknown:
        raise SystemExit(f"Unknown bot(s) {', '.join(unknown)}. "
                         f"Choose from: {', '.join(OPPONENT_MODES)}")
    model_path = None
    if not args.random and args.model:
        model_path = resolve_model_path(args.model)

    results = []
    for bot in bots:
        result = evaluate(
            model_path=model_path,
            episodes=args.episodes,
            map_name=args.map,
            weapon=args.weapon,
            opponent_weapon=args.opponent_weapon,
            bot=bot,
            vision_mode=args.vision,
            phase=args.phase,
            episode_seconds=args.episode_seconds,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        result["policy"] = "Zufallspolitik" if model_path is None else str(args.model)
        results.append(result)
        if not args.json:
            if len(bots) > 1:
                print(f"=== Bot: {bot} ===")
            print(format_result(result))
            print()
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
