"""Command line: ``python -m rocketai <command>`` (or ``rocketai <command>`` after install)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from .config import TrainConfig, preset_config, run_paths
    from .trainer import train

    overrides = {
        "total_steps": args.steps,
        "n_workers": args.workers,
        "team_size": args.team_size,
        "reward_stage": args.stage,
        "envs_per_worker": args.envs,
        "seed": args.seed,
    }
    if args.resume:
        paths = run_paths(args.resume)
        if not paths.config.exists():
            print(f"Run '{args.resume}' gibt es nicht ({paths.root})", file=sys.stderr)
            return 2
        data = TrainConfig.load(paths.config).to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        config = TrainConfig.from_dict(data)
    elif args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        if args.name:
            data["name"] = args.name
        config = TrainConfig.from_dict(data)
    else:
        config = preset_config(args.preset, name=args.name, **overrides)
    try:
        config.validate()
    except ValueError as error:
        print(f"Ungültige Einstellungen: {error}", file=sys.stderr)
        return 2
    train(config)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .match import evaluate
    from .opponents import PolicyPlayer, make_player

    policy = PolicyPlayer.from_checkpoint(Path(args.checkpoint))
    for spec in args.opponent:
        result = evaluate(policy, make_player(spec), games=args.games, team_size=args.team_size)
        print(json.dumps(result))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from .match import play_match, save_replay
    from .opponents import make_player

    result = play_match(
        make_player(args.blue),
        make_player(args.orange),
        team_size=args.team_size,
        seconds=args.seconds,
        record=True,
    )
    save_replay(Path(args.out), result, {"kind": "match"})
    print(json.dumps(result.summary()))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    from .play import PlaySettings, start_match

    settings = PlaySettings(
        checkpoint=str(Path(args.checkpoint).resolve()),
        mode=args.mode,
        team_size=args.team_size,
        skill=args.skill,
        launcher=args.launcher,
        opponent_bot=args.opponent_bot,
    )
    start_match(settings, wait=True)
    return 0


def _cmd_bot(args: argparse.Namespace) -> int:
    from .rlbot_bot.bot import main as bot_main

    bot_main(["--checkpoint", args.checkpoint] + (["--sample"] if args.sample else []))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_checks

    ok = True
    for check in run_checks():
        mark = "OK " if check["ok"] else ("!! " if check["required"] else "-- ")
        ok &= check["ok"] or not check["required"]
        print(f"{mark}{check['label']}: {check['detail']}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    from .config import PRESETS

    parser = argparse.ArgumentParser(
        prog="rocketai", description="Rocket-League-KI: trainieren, testen, spielen."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="KI trainieren (Simulation)")
    train.add_argument("--preset", choices=sorted(PRESETS), default="beginner")
    train.add_argument("--name", help="Name des Runs (Ordner unter runs/)")
    train.add_argument("--resume", metavar="RUN", help="bestehenden Run fortsetzen")
    train.add_argument("--config", help="TrainConfig als JSON-Datei")
    train.add_argument("--steps", type=int, help="Gesamtschritte")
    train.add_argument("--workers", type=int, help="Simulations-Prozesse (0 = automatisch)")
    train.add_argument("--envs", type=int, help="Spiele pro Prozess")
    train.add_argument("--team-size", type=int, choices=(1, 2, 3))
    train.add_argument("--stage", type=int, choices=(1, 2, 3), help="Belohnungsstufe")
    train.add_argument("--seed", type=int)
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="Checkpoint gegen eingebaute Gegner testen")
    ev.add_argument("checkpoint")
    ev.add_argument("--opponent", nargs="+", default=["chaser", "defender"])
    ev.add_argument("--games", type=int, default=6)
    ev.add_argument("--team-size", type=int, default=1, choices=(1, 2, 3))
    ev.set_defaults(func=_cmd_eval)

    rp = sub.add_parser("replay", help="Simuliertes Match aufzeichnen")
    rp.add_argument("blue", help="Checkpoint-Pfad oder idle/random/chaser/defender")
    rp.add_argument("orange")
    rp.add_argument("--out", default="replay.json")
    rp.add_argument("--seconds", type=float, default=120.0)
    rp.add_argument("--team-size", type=int, default=1, choices=(1, 2, 3))
    rp.set_defaults(func=_cmd_replay)

    sv = sub.add_parser("serve", help="Web-App starten")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--no-browser", action="store_true")
    sv.set_defaults(func=_cmd_serve)

    pl = sub.add_parser("play", help="Im echten Rocket League spielen (RLBot, offline)")
    pl.add_argument("checkpoint")
    pl.add_argument("--mode", choices=("psyonix", "human", "bot", "self"), default="psyonix")
    pl.add_argument("--team-size", type=int, default=1, choices=(1, 2, 3))
    pl.add_argument("--skill", choices=("beginner", "rookie", "pro", "allstar"), default="rookie")
    pl.add_argument("--launcher", choices=("steam", "epic"), default="epic")
    pl.add_argument(
        "--opponent-bot", default="", help="bot.toml eines Community-Bots (bei --mode bot)"
    )
    pl.set_defaults(func=_cmd_play)

    bot = sub.add_parser("bot", help="(intern) RLBot-Bot-Prozess")
    bot.add_argument("--checkpoint", default=os.environ.get("ROCKETAI_CHECKPOINT", ""))
    bot.add_argument("--sample", action="store_true")
    bot.set_defaults(func=_cmd_bot)

    doc = sub.add_parser("doctor", help="Installation prüfen")
    doc.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "train" and not (args.name or args.resume or args.config):
        args.name = args.preset
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
