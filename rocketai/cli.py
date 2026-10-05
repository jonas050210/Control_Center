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
        "teacher_opponent_prob": args.teacher_opponent,
        "teacher_weight": args.teacher_weight,
        "teacher_final_weight": args.teacher_final_weight,
        "teacher_decay_steps": args.teacher_decay_steps,
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
    from .runtime import AlreadyRunning

    try:
        train(config)
    except AlreadyRunning as error:
        print(f"{error}\nEin Run darf nur einmal gleichzeitig trainieren.", file=sys.stderr)
        return 3
    return 0


def _cmd_teacher(args: argparse.Namespace) -> int:
    """Lehrer laden, prüfen und auf Wunsch ein Testspiel zeigen."""
    from .teacher import NextoTeacher, describe_teacher, ensure_teacher

    try:
        ensure_teacher(progress=print)
    except Exception as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 2
    info = describe_teacher()
    print(f"Lehrer {info['name']}: {'bereit' if info['ready'] else 'fehlt'}")
    print(f"  Ordner: {info['directory']}")
    print(f"  Quelle: {info['source']} ({info['license']})")
    if not args.test:
        print("Hinweis: nur offline verwenden. Testspiel mit '--test'.")
        return 0
    from .match import play_match
    from .opponents import make_player

    print("Testspiel: Lehrer (blau) gegen Balljäger (orange), 60 Sekunden ...")
    teacher = make_player("teacher")
    _ = NextoTeacher()  # früher Fehler statt mitten im Match
    result = play_match(teacher, make_player(args.opponent), team_size=1, seconds=60.0)
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
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
        checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        brain=args.brain,
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

    bot_main(
        ["--checkpoint", args.checkpoint]
        + (["--sample"] if args.sample else [])
        + (["--teacher"] if args.teacher else [])
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import format_report, run_benchmark

    report = run_benchmark(
        seconds=args.seconds,
        workers=args.workers,
        envs_per_worker=args.envs,
        team_size=args.team_size,
        with_update=not args.no_update,
    )
    print(format_report(report))
    if args.json:
        import json as _json

        print(_json.dumps(report, indent=2))
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
    train.add_argument(
        "--teacher-opponent",
        type=float,
        help="Anteil der Trainings-Matches gegen den Lehrer (z. B. 0.25)",
    )
    train.add_argument(
        "--teacher-weight", type=float, help="Gewicht der Nachahmung am Anfang (0 = aus)"
    )
    train.add_argument(
        "--teacher-final-weight",
        type=float,
        help="Gewicht der Nachahmung am Ende (Standard 0 = ganz auslaufen, nur mit "
        "--teacher-weight > 0 sinnvoll)",
    )
    train.add_argument(
        "--teacher-decay-steps",
        type=int,
        help="Schritte, nach denen die Nachahmung ihren Endwert erreicht (0 = nie abfallen)",
    )
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
    pl.add_argument("checkpoint", nargs="?", default="")
    pl.add_argument(
        "--brain",
        choices=("policy", "teacher"),
        default="policy",
        help="'policy' = eigene KI, 'teacher' = Nexto spielt (Lehrer)",
    )
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
    bot.add_argument("--teacher", action="store_true", help="Lehrer statt Checkpoint fahren")
    bot.set_defaults(func=_cmd_bot)

    th = sub.add_parser("teacher", help="Lehrer (Nexto) laden und prüfen")
    th.add_argument("--test", action="store_true", help="Testspiel gegen einen Bot zeigen")
    th.add_argument("--opponent", default="chaser", help="Gegner im Testspiel")
    th.set_defaults(func=_cmd_teacher)

    doc = sub.add_parser("doctor", help="Installation prüfen")
    doc.set_defaults(func=_cmd_doctor)

    bench = sub.add_parser("benchmark", help="Messen, wie schnell dieser Rechner trainiert")
    bench.add_argument("--seconds", type=float, default=6.0, help="Messdauer pro Einstellung")
    bench.add_argument("--workers", type=int, default=0, help="0 = automatisch")
    bench.add_argument(
        "--envs", type=int, default=0, help="Spiele pro Prozess (0 = automatisch testen)"
    )
    bench.add_argument("--team-size", type=int, default=1, choices=(1, 2, 3))
    bench.add_argument("--no-update", action="store_true", help="Lernschritt nicht messen")
    bench.add_argument("--json", action="store_true", help="Ergebnis zusätzlich als JSON")
    bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "train" and not (args.name or args.resume or args.config):
        args.name = args.preset
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
