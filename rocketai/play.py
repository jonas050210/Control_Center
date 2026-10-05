"""Start matches in the real Rocket League through RLBot v5 (offline only).

Rocket League must be installed (Steam or Epic). Since April 2026 RL ships
with Easy Anti-Cheat; bots only work in offline matches, which is exactly
what RLBot starts. Never use this for online play.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import PYTHON_CMD, ROOT, tools_root

MODES = {
    "psyonix": "Gegen Psyonix-Bots",
    "human": "Gegen dich (Controller/Tastatur)",
    "bot": "Gegen einen Community-Bot",
    "self": "KI gegen sich selbst",
}
SKILLS = {"beginner": "Beginner", "rookie": "Rookie", "pro": "Pro", "allstar": "AllStar"}
LAUNCHERS = {"steam": "Steam", "epic": "Epic"}
BRAINS = {"policy": "Eigene KI", "teacher": "Lehrer (Nexto)"}


@dataclass
class PlaySettings:
    checkpoint: str = ""
    brain: str = "policy"  # "policy" = own AI, "teacher" = Nexto plays
    mode: str = "psyonix"
    team_size: int = 1
    skill: str = "rookie"
    launcher: str = "epic"
    opponent_bot: str = ""  # path to another bot's bot.toml (mode "bot")
    match_length: str = "FiveMinutes"

    def validate(self) -> None:
        if self.brain not in BRAINS:
            raise ValueError(f"Unbekanntes Gehirn {self.brain!r}")
        if self.brain == "teacher":
            from .teacher import teacher_ready

            if not teacher_ready():
                raise ValueError(
                    f"Der Lehrer ist nicht geladen. Einmal '{PYTHON_CMD} -m rocketai teacher' ausführen."
                )
        elif not Path(self.checkpoint).is_file():
            raise ValueError(f"Checkpoint nicht gefunden: {self.checkpoint}")
        if self.mode not in MODES:
            raise ValueError(f"Unbekannter Modus {self.mode!r}")
        if self.team_size not in (1, 2, 3):
            raise ValueError("Teamgröße muss 1, 2 oder 3 sein")
        if self.skill not in SKILLS:
            raise ValueError(f"Unbekannte Stufe {self.skill!r}")
        if self.launcher not in LAUNCHERS:
            raise ValueError(f"Unbekannter Launcher {self.launcher!r}")
        if self.mode == "bot" and not Path(self.opponent_bot).is_file():
            raise ValueError("Für 'Community-Bot' bitte den Pfad zur bot.toml angeben")
        if self.mode == "bot" and self.team_size != 1:
            # Ein fremder Bot bringt genau eine Kennung mit; zweimal dieselbe
            # Kennung ergibt kein Match.
            raise ValueError("Community-Bots lassen sich nur 1v1 spielen")


def server_path() -> Path:
    name = "RLBotServer.exe" if sys.platform == "win32" else "RLBotServer"
    return tools_root() / "rlbot" / name


def match_dir() -> Path:
    return ROOT / ".rocketai" / "match"


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def bot_command(settings: PlaySettings, agent_id: str) -> str:
    """Der Befehl, mit dem RLBotServer unseren Bot startet."""
    python = sys.executable
    if settings.brain == "teacher":
        args = "--teacher"
    else:
        args = f'--checkpoint "{Path(settings.checkpoint).resolve()}"'
    # Die agent_id muss in beiden Richtungen gleich sein — sonst ordnet RLBot
    # den laufenden Prozess keinem Auto zu und der Bot bewegt sich nie.
    return f'"{python}" -m rocketai bot {args} --agent-id {agent_id}'


def bot_toml(
    settings: PlaySettings,
    agent_id: str = "rocketai/policy",
    name: str | None = None,
) -> str:
    command = bot_command(settings, agent_id)
    return "\n".join(
        [
            "[settings]",
            f'name = "{name or BRAINS[settings.brain]}"',
            f'agent_id = "{agent_id}"',
            f"root_dir = {_toml_str(str(ROOT))}",
            f"run_command = {_toml_str(command)}",
            f"run_command_linux = {_toml_str(command)}",
            "",
        ]
    )


def _car(kind: str, team: int, **extra: str) -> list[str]:
    lines = ["[[cars]]", f'type = "{kind}"', f"team = {team}"]
    lines += [f"{key} = {_toml_str(value)}" for key, value in extra.items()]
    return lines + [""]


def config_names(settings: PlaySettings, team: str) -> list[str]:
    """Dateinamen der Bot-Konfigurationen für ein Team (``"blue"``/``"orange"``)."""
    size = settings.team_size
    if team == "orange" and settings.mode != "self":
        return []
    if size == 1:
        return ["bot.toml"] if team == "blue" else ["bot-orange.toml"]
    prefix = "bot" if team == "blue" else "bot-orange"
    return [f"{prefix}-{index}.toml" for index in range(size)]


def agent_ids(settings: PlaySettings, team: str) -> list[str]:
    """RLBot-Kennungen je Auto.

    Jedes Auto braucht eine **eigene** Kennung: Bei „KI gegen sich selbst“ und
    im 2v2 laufen mehrere Bot-Prozesse gleichzeitig, und zwei Prozesse mit
    derselben Kennung bekommen kein Auto zugeteilt. Genau das war der Fehler,
    durch den der zweite Bot einfach stehen blieb.
    """
    size = settings.team_size
    if team == "orange" and settings.mode != "self":
        return []
    if size == 1:
        return [primary_agent_id(settings)] if team == "blue" else ["rocketai/orange"]
    return [f"rocketai/{team}-{index}" for index in range(size)]


def match_toml(settings: PlaySettings) -> str:
    """The match config: our bots are always blue (team 0)."""
    lines = [
        "[rlbot]",
        f'launcher = "{LAUNCHERS[settings.launcher]}"',
        "auto_start_agents = true",
        "wait_for_agents = true",
        "",
        "[match]",
        'game_mode = "Soccar"',
        'game_map_upk = "Stadium_P"',
        "skip_replays = false",
        "instant_start = false",
        'existing_match_behavior = "Restart"',
        "enable_rendering = false",
        "enable_state_setting = false",
        "",
        "[mutators]",
        f'match_length = "{settings.match_length}"',
        "",
    ]
    size = settings.team_size
    skill = SKILLS[settings.skill]
    # Ein Auto pro Datei: jedes bekommt eine eigene Kennung (siehe agent_ids).
    for name in config_names(settings, "blue"):
        lines += _car("rlbot", 0, config_file=name)
    if settings.mode == "psyonix":
        for _ in range(size):
            lines += _car("psyonix", 1, skill=skill)
    elif settings.mode == "human":
        lines += _car("human", 1)
        for _ in range(size - 1):
            lines += _car("psyonix", 1, skill=skill)
    elif settings.mode == "bot":
        for _ in range(size):
            lines += _car("rlbot", 1, config_file=str(Path(settings.opponent_bot).resolve()))
    else:  # self: dieselbe KI, aber je Auto ein eigener Bot-Prozess
        for name in config_names(settings, "orange"):
            lines += _car("rlbot", 1, config_file=name)
    return "\n".join(lines)


def primary_agent_id(settings: PlaySettings) -> str:
    """Kennung für das blaue Auto (die eigene KI bzw. der Lehrer)."""
    return "rocketai/teacher" if settings.brain == "teacher" else "rocketai/policy"


def write_match_files(settings: PlaySettings) -> Path:
    folder = match_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for team, label in (("blue", "Blau"), ("orange", "Orange")):
        for name, agent in zip(
            config_names(settings, team), agent_ids(settings, team), strict=True
        ):
            (folder / name).write_text(
                bot_toml(settings, agent, name=f"{BRAINS[settings.brain]} ({label})"),
                encoding="utf-8",
            )
    path = folder / "match.toml"
    path.write_text(match_toml(settings), encoding="utf-8")
    return path


class PlaySession:
    """One RLBot match started from the web app; at most one at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.manager: Any = None
        self.settings: PlaySettings | None = None
        self.state = "idle"
        self.message = ""

    def info(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "settings": asdict(self.settings) if self.settings else None,
        }

    def start(self, settings: PlaySettings) -> None:
        settings.validate()
        with self.lock:
            if self.state in ("starting", "running"):
                raise RuntimeError("Es läuft schon ein Match")
            self.settings = settings
            self.state = "starting"
            self.message = "Starte RLBot und Rocket League ..."
        threading.Thread(target=self._run, args=(settings,), daemon=True).start()

    def _run(self, settings: PlaySettings) -> None:
        try:
            self.manager = start_match(settings, wait=True)
            self.state = "running"
            self.message = "Match läuft in Rocket League"
        except Exception as error:  # surfaced in the UI
            self.state = "failed"
            self.message = f"{type(error).__name__}: {error}"

    def stop(self) -> None:
        manager, self.manager = self.manager, None
        if manager is not None:
            try:
                manager.stop_match()
            finally:
                manager.disconnect()
        self.state = "idle"
        self.message = "Match beendet"


def require_rlbot() -> None:
    """Klartext, wenn RLBot fehlt oder die falsche Reihe installiert ist."""
    from .doctor import RLBOT_PARTS, rlbot_check

    check = rlbot_check()
    if not check["ok"]:
        raise RuntimeError(
            f"{check['detail']} Der Bot importiert {', '.join(RLBOT_PARTS)}; "
            f"installiert wird das mit '{PYTHON_CMD} install.py'."
        )


def start_match(settings: PlaySettings, wait: bool = True) -> Any:
    """Write the configs, start RLBotServer (and RL) and the match. Returns the MatchManager."""
    settings.validate()
    require_rlbot()
    from rlbot.managers import MatchManager

    server = server_path()
    if not server.exists():
        raise FileNotFoundError(
            f"RLBotServer fehlt ({server}). Bitte '{PYTHON_CMD} install.py' erneut ausführen."
        )
    config = write_match_files(settings)
    if settings.brain == "policy":
        os.environ["ROCKETAI_CHECKPOINT"] = str(Path(settings.checkpoint).resolve())
    manager = MatchManager(server)
    manager.start_match(config, wait_for_start=wait)
    return manager
