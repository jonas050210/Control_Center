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

from .config import ROOT, tools_root

MODES = {
    "psyonix": "Gegen Psyonix-Bots",
    "human": "Gegen dich (Controller/Tastatur)",
    "bot": "Gegen einen Community-Bot",
    "self": "KI gegen sich selbst",
}
SKILLS = {"beginner": "Beginner", "rookie": "Rookie", "pro": "Pro", "allstar": "AllStar"}
LAUNCHERS = {"steam": "Steam", "epic": "Epic"}


@dataclass
class PlaySettings:
    checkpoint: str
    mode: str = "psyonix"
    team_size: int = 1
    skill: str = "rookie"
    launcher: str = "epic"
    opponent_bot: str = ""  # path to another bot's bot.toml (mode "bot")
    match_length: str = "FiveMinutes"

    def validate(self) -> None:
        if not Path(self.checkpoint).is_file():
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


def server_path() -> Path:
    name = "RLBotServer.exe" if sys.platform == "win32" else "RLBotServer"
    return tools_root() / "rlbot" / name


def match_dir() -> Path:
    return ROOT / ".rocketai" / "match"


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def bot_toml(settings: PlaySettings) -> str:
    python = sys.executable
    command = f'"{python}" -m rocketai bot --checkpoint "{Path(settings.checkpoint).resolve()}"'
    return "\n".join(
        [
            "[settings]",
            'name = "RocketAI"',
            'agent_id = "rocketai/policy"',
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
    for _ in range(size):
        lines += _car("rlbot", 0, config_file="bot.toml")
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
    else:  # self
        for _ in range(size):
            lines += _car("rlbot", 1, config_file="bot.toml")
    return "\n".join(lines)


def write_match_files(settings: PlaySettings) -> Path:
    folder = match_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bot.toml").write_text(bot_toml(settings), encoding="utf-8")
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


def start_match(settings: PlaySettings, wait: bool = True) -> Any:
    """Write the configs, start RLBotServer (and RL) and the match. Returns the MatchManager."""
    settings.validate()
    from rlbot.managers import MatchManager

    server = server_path()
    if not server.exists():
        raise FileNotFoundError(
            f"RLBotServer fehlt ({server}). Bitte 'python install.py' erneut ausführen."
        )
    config = write_match_files(settings)
    os.environ["ROCKETAI_CHECKPOINT"] = str(Path(settings.checkpoint).resolve())
    manager = MatchManager(server)
    manager.start_match(config, wait_for_start=wait)
    return manager
