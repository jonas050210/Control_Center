"""Der Lehrer: die eigene KI lernt durch Nachahmung von Nexto.

Nexto ist der stärkste frei verfügbare Rocket-League-Bot der Community
(Grand-Champion-Niveau). Sein neuronales Netz sagt zu jeder Spielsituation,
welche Taste er drücken würde. Genau das nutzt dieses Modul: Die eigene KI
bekommt dieselbe Situation, sieht Nextos Antwort und wird ein Stück in diese
Richtung gezogen ("Wissens-Destillation"). Ihre Mechanics lernt sie damit in
Stunden statt in Monaten.

Wichtig:

* Nextos Dateien sind **GPL-lizenziert** und werden deshalb nicht mit diesem
  Projekt ausgeliefert. Sie werden bei Bedarf von GitHub geladen und landen
  unter ``tools/nexto`` (nicht im Repository). Nur für den privaten,
  Offline-Gebrauch gedacht.
* Für einen Menschen unsichtbar, aber technisch dasselbe Spiel: Nexto sieht
  den Zustand über ``nexto_obs.py``, wir bauen ihm exakt dieses Format aus
  der RocketSim-Simulation bzw. aus einem RLBot-Paket.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import types
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import tools_root
from .env import lookup_table
from .opponents import IDLE_ACTION, Player, explain_decision

#: Woher der Lehrer kommt (RLBot-Botpack) und wie er geprüft wird.
SOURCE_REPO = "RLBot/RLBotPack"
SOURCE_BRANCH = "master"
SOURCE_DIR = "RLBotPack/Necto/Nexto"
TEACHER_LABEL = "Nexto"

#: sha256 of exactly the bytes we expect. Anything else is refused: a changed
#: ``.pt`` file is code, and we do not load code we cannot vouch for.
FILES: dict[str, str] = {
    "nexto-model.pt": "bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa",
    "nexto_obs.py": "7247046d836059ab4428fb8f08968af1a977cf3228f1462a0235cd7a825362b0",
    "agent.py": "af5e20b5741d58da2c12d769cb72c74590653f25fda3243cef73fe7127fe61bd",
}

NOTICE = f"""Nexto - Community-Rocket-League-Bot der RLBot-Community.
Quelle: https://github.com/{SOURCE_REPO} (Ordner {SOURCE_DIR})
Lizenz: GNU General Public License v3 (siehe Repository).
Diese Dateien werden nur lokal geladen und sind nicht Teil von RocketAI.
Nutzung nur offline / privat. Nicht weitergeben, nicht mitverbreiten.
"""

#: Ein kompakter Spielstand hat so viele Zahlen:
BALL_VALUES = 9  # Ball: Position, Geschwindigkeit, Drehung
PAD_VALUES = 34  # Boost-Pads (1 = da, 0 = weg)
PLAYER_VALUES = 19  # pro Auto: Team, Position, Quaternion, Tempo, Drehung, Boost, Flags
HEADER_VALUES = 3 + PAD_VALUES + 2 * BALL_VALUES  # Nextos Kopfbereich
PLAYER_BLOCK = 2 + 2 * 13 + 10  # ein Auto in Nextos Format


class TeacherError(RuntimeError):
    """Der Lehrer fehlt, ist nicht erreichbar oder passt nicht."""


# --------------------------------------------------------------------------
# Herunterladen und Prüfen
# --------------------------------------------------------------------------


def teacher_dir() -> Path:
    import os

    override = os.environ.get("ROCKETAI_TEACHER")
    return Path(override) if override else tools_root() / "nexto"


def teacher_ready(directory: Path | None = None) -> bool:
    directory = directory or teacher_dir()
    for name, digest in FILES.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "RocketAI"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _fetch_raw(name: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/{SOURCE_DIR}/{name}"
    return _get(url)


def _fetch_api(name: str) -> bytes:
    """GitHub-API. Große Dateien (unser Modell) liefert nur der Blob-Endpunkt."""
    url = (
        f"https://api.github.com/repos/{SOURCE_REPO}/contents/{SOURCE_DIR}/{name}"
        f"?ref={SOURCE_BRANCH}"
    )
    meta = json.loads(_get(url).decode("utf-8"))
    if meta.get("content"):
        return base64.b64decode(meta["content"])
    blob = meta.get("git_url") or (
        f"https://api.github.com/repos/{SOURCE_REPO}/git/blobs/{meta['sha']}"
    )
    payload = json.loads(_get(blob).decode("utf-8"))
    return base64.b64decode(payload["content"])


#: Erst die schnelle Rohdatei, dann die GitHub-API (manche Netze blocken eine davon).
FETCHERS: tuple[tuple[str, Callable[[str], bytes]], ...] = (
    ("raw.githubusercontent.com", _fetch_raw),
    ("api.github.com", _fetch_api),
)


def _fetch_verified(name: str, digest: str) -> bytes:
    """First source that delivers the exact bytes we expect wins."""
    problems: list[str] = []
    for label, fetch in FETCHERS:
        try:
            data = fetch(name)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as error:
            problems.append(f"{label}: {error}")
            continue
        except (ValueError, KeyError) as error:  # broken JSON from a captive portal
            problems.append(f"{label}: unlesbare Antwort ({error})")
            continue
        got = hashlib.sha256(data).hexdigest()
        if got == digest:
            return data
        problems.append(f"{label}: falscher Inhalt ({got[:12]}…)")
    raise TeacherError(
        f"{name} konnte nicht geladen werden. Versuche: "
        + "; ".join(problems)
        + ". Aus Sicherheitsgründen wird eine unbekannte Modelldatei nicht geladen."
    )


def ensure_teacher(
    directory: Path | None = None,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> Path:
    """Load Nexto's files if they are missing; returns the folder holding them."""
    directory = (directory or teacher_dir()).expanduser()
    say = progress or (lambda _message: None)
    directory.mkdir(parents=True, exist_ok=True)
    if teacher_ready(directory) and not force:
        return directory
    say(f"Lade den Lehrer ({TEACHER_LABEL}) von GitHub ...")
    for name, digest in FILES.items():
        path = directory / name
        if path.is_file() and _sha256(path) == digest and not force:
            continue
        try:
            data = _fetch_verified(name, digest)
        except TeacherError as error:
            raise TeacherError(
                f"Der Lehrer konnte nicht geladen werden. Einmal Internet nötig, "
                f"danach läuft alles offline.\n{error}"
            ) from error
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        say(f"  {name} geladen ({len(data) / 1e6:.1f} MB)")
    (directory / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    return directory


# --------------------------------------------------------------------------
# Nextos Code einbinden
# --------------------------------------------------------------------------
#
# ``nexto_obs.py`` importiert ``rlgym_compat`` (das alte RLGym-1-Zustandsformat).
# Wir stellen diese Module als winzige Attrappen bereit, damit die
# heruntergeladene Datei unverändert importiert werden kann.


class _CompatPhysics:
    """Steht für ``rlgym_compat.PhysicsObject``: nur was Nexto wirklich liest."""

    __slots__ = ("position", "quaternion", "linear_velocity", "angular_velocity", "_matrix")

    def __init__(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        matrix: np.ndarray | None = None,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.quaternion = np.asarray(quaternion, dtype=np.float64)
        self.linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        self._matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)

    def rotation_mtx(self) -> np.ndarray:  # Nexto ruft das als Methode auf
        if self._matrix is None:
            from rlgym.rocket_league.math import quat_to_rot_mtx

            self._matrix = quat_to_rot_mtx(np.asarray(self.quaternion))
        return self._matrix

    def euler_angles(self) -> np.ndarray:  # pragma: no cover - Nexto liest das nicht
        raise NotImplementedError

    def invert(self, other: _CompatPhysics) -> _CompatPhysics:
        mirror = np.array([-1.0, -1.0, 1.0])
        self.position = other.position * mirror
        self.quaternion = other.quaternion
        self.linear_velocity = other.linear_velocity * mirror
        self.angular_velocity = other.angular_velocity * mirror
        self._matrix = other.rotation_mtx() * np.array([[-1], [-1], [1]], dtype=np.float64)
        return self


class _CompatPlayer:
    """Steht für ``rlgym_compat.PlayerData``."""

    def __init__(self, index: int, team: int, car: Any):
        self.car_id = index
        self.team_num = int(team)
        self.is_demoed = bool(car.is_demoed)
        self.on_ground = bool(car.on_ground)
        self.ball_touched = bool(getattr(car, "ball_touched", False))
        self.has_flip = bool(car.has_flip)
        self.boost_amount = float(car.boost_amount) / 100.0
        physics = car.physics
        self.car_data = _CompatPhysics(
            physics.position,
            physics.quaternion,
            physics.linear_velocity,
            physics.angular_velocity,
            getattr(physics, "_rotation_mtx", None),
        )
        self.inverted_car_data = _CompatPhysics(
            np.zeros(3), np.ones(4), np.zeros(3), np.zeros(3)
        ).invert(self.car_data)


class _CompatState:
    """Steht für ``rlgym_compat.GameState``."""

    def __init__(
        self,
        ball: np.ndarray,
        pads: np.ndarray,
        players: list[_CompatPlayer],
        ball_velocity: np.ndarray | None = None,
        ball_angular: np.ndarray | None = None,
    ):
        physics = _CompatPhysics(
            ball,
            np.ones(4),
            np.zeros(3) if ball_velocity is None else ball_velocity,
            np.zeros(3) if ball_angular is None else ball_angular,
        )
        self.ball = physics
        self.inverted_ball = _CompatPhysics(
            np.zeros(3), np.ones(4), np.zeros(3), np.zeros(3)
        ).invert(physics)
        self.boost_pads = np.asarray(pads, dtype=np.float64)
        self.inverted_boost_pads = self.boost_pads[::-1].copy()
        self.players = players
        self.blue_score = 0
        self.orange_score = 0
        self.last_touch = None


def _install_compat() -> None:
    if "rlgym_compat" in sys.modules:
        return
    package = types.ModuleType("rlgym_compat")
    constants = types.ModuleType("rlgym_compat.common_values")
    constants.ORANGE_GOAL_CENTER = [0, 5120, 642.775 / 2]
    constants.BLUE_GOAL_CENTER = [0, -5120, 642.775 / 2]
    constants.BLUE_TEAM = 0
    constants.ORANGE_TEAM = 1
    constants.NUM_ACTIONS = 8
    state_module = types.ModuleType("rlgym_compat.game_state")
    state_module.GameState = _CompatState
    state_module.PlayerData = _CompatPlayer
    physics_module = types.ModuleType("rlgym_compat.physics_object")
    physics_module.PhysicsObject = _CompatPhysics
    package.common_values = constants
    package.game_state = state_module
    package.physics_object = physics_module
    sys.modules["rlgym_compat"] = package
    sys.modules["rlgym_compat.common_values"] = constants
    sys.modules["rlgym_compat.game_state"] = state_module
    sys.modules["rlgym_compat.physics_object"] = physics_module


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise TeacherError(f"{path} lässt sich nicht laden")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Spielstand -> Nextos Sicht
# --------------------------------------------------------------------------


def canonical_players(state: Any) -> list[str]:
    """Alle Autos in der Reihenfolge, die Nexto erwartet: erst Blau, dann Orange.

    Innerhalb eines Teams zählt die Reihenfolge der Agenten, genau wie bei
    RLBot (Spieler 0, 1, 2 je Team).
    """
    keys = list(state.cars)
    return sorted(keys, key=lambda agent: (int(state.cars[agent].team_num), keys.index(agent)))


def encode_compact(state: Any, players: list[str] | None = None) -> np.ndarray:
    """Kompakter Spielstand (siehe ``BALL_VALUES``/``PAD_VALUES``/``PLAYER_VALUES``).

    Enthält bewusst nur Zahlen, die sich billig kopieren lassen; beim
    Berechnen der Lehrer-Antworten wird daraus Nextos volles Format.
    """
    players = players or canonical_players(state)
    values = np.empty(BALL_VALUES + PAD_VALUES + PLAYER_VALUES * len(players), dtype=np.float32)
    ball = state.ball
    values[0:3] = ball.position
    values[3:6] = ball.linear_velocity
    values[6:9] = ball.angular_velocity
    values[9 : 9 + PAD_VALUES] = state.boost_pad_timers <= 0
    offset = 9 + PAD_VALUES
    for agent in players:
        car = state.cars[agent]
        physics = car.physics
        values[offset] = car.team_num
        values[offset + 1 : offset + 4] = physics.position
        values[offset + 4 : offset + 8] = physics.quaternion
        values[offset + 8 : offset + 11] = physics.linear_velocity
        values[offset + 11 : offset + 14] = physics.angular_velocity
        values[offset + 14] = float(car.boost_amount) / 100.0
        values[offset + 15] = 1.0 if car.is_demoed else 0.0
        values[offset + 16] = 1.0 if car.on_ground else 0.0
        values[offset + 17] = 1.0 if car.has_flip else 0.0
        values[offset + 18] = 1.0 if getattr(car, "ball_touched", False) else 0.0
        offset += PLAYER_VALUES
    return values


def compact_players(compact: np.ndarray) -> int:
    return (len(compact) - BALL_VALUES - PAD_VALUES) // PLAYER_VALUES


def expand_compacts(compacts: np.ndarray) -> np.ndarray:
    """``(B, D)`` kompakt -> ``(B, N)`` in Nextos ``encode_gamestate``-Format.

    Reine Vektorrechnung; ``tests/test_teacher.py`` prüft das Ergebnis gegen
    Nextos eigene Funktion.
    """
    compacts = np.atleast_2d(np.asarray(compacts, dtype=np.float64))
    batch = compacts.shape[0]
    n_players = compact_players(compacts[0])
    size = HEADER_VALUES + PLAYER_BLOCK * n_players
    encoded = np.zeros((batch, size), dtype=np.float64)
    # Kopf: Modus/Punkte (ungenutzt), Boost-Pads, Ball hin und zurück
    encoded[:, 3 : 3 + PAD_VALUES] = compacts[:, BALL_VALUES : BALL_VALUES + PAD_VALUES]
    ball = compacts[:, 0:BALL_VALUES]
    encoded[:, 37:46] = ball
    inverted = ball.copy()
    inverted[:, 0:2] *= -1  # x/y gespiegelt, Drehung unverändert
    inverted[:, 3:6] *= -1
    inverted[:, 6:9] *= -1
    encoded[:, 46:55] = inverted
    mirror = np.array([-1.0, -1.0, 1.0])
    for index in range(n_players):
        start = BALL_VALUES + PAD_VALUES + index * PLAYER_VALUES
        block = compacts[:, start : start + PLAYER_VALUES]
        base = HEADER_VALUES + index * PLAYER_BLOCK
        car = np.zeros((batch, 13))
        car[:, 0:3] = block[:, 1:4]
        car[:, 3:7] = block[:, 4:8]
        car[:, 7:10] = block[:, 8:11]
        car[:, 10:13] = block[:, 11:14]
        encoded[:, base] = index
        encoded[:, base + 1] = block[:, 0]
        encoded[:, base + 2 : base + 15] = car
        inv = car.copy()
        inv[:, 0:3] *= mirror
        inv[:, 7:10] *= mirror
        inv[:, 10:13] *= mirror
        encoded[:, base + 15 : base + 28] = inv
        encoded[:, base + 33] = block[:, 15]  # demoed
        encoded[:, base + 34] = block[:, 16]  # on_ground
        encoded[:, base + 35] = block[:, 18]  # ball_touched
        encoded[:, base + 36] = block[:, 17]  # has_flip
        encoded[:, base + 37] = block[:, 14]  # boost 0..1
    return encoded


def encode_reference(state: Any, players: list[str] | None = None) -> np.ndarray:
    """Derselbe Spielstand, aber über Nextos eigene Funktion (langsam, Prüfweg)."""
    builder = _reference_builder()
    players = players or canonical_players(state)
    compat = _CompatState(
        state.ball.position,
        np.asarray(state.boost_pad_timers <= 0, dtype=np.float64),
        [
            _CompatPlayer(index, state.cars[agent].team_num, state.cars[agent])
            for index, agent in enumerate(players)
        ],
        ball_velocity=state.ball.linear_velocity,
        ball_angular=state.ball.angular_velocity,
    )
    return np.asarray(builder.encode_gamestate(compat), dtype=np.float64)


def _reference_builder() -> Any:
    _install_compat()
    directory = ensure_teacher()
    obs = _load_module("nexto_obs_ref", directory / "nexto_obs.py")
    return obs


# --------------------------------------------------------------------------
# Der Lehrer selbst
# --------------------------------------------------------------------------


class NextoTeacher:
    """Nextos Netz, benutzbar für Einzel- und Stapel-Abfragen."""

    def __init__(
        self,
        directory: Path | None = None,
        progress: Callable[[str], None] | None = None,
        chunk: int = 2048,
    ):
        import torch

        directory = ensure_teacher(directory, progress)
        _install_compat()
        self.directory = directory
        self.obs = _load_module("nexto_obs", directory / "nexto_obs.py")
        self.builder = self.obs.NextoObsBuilder(field_info=None)
        with warnings.catch_warnings():
            # torch.jit.load ist in neuen Torch-Versionen als "deprecated"
            # markiert, funktioniert aber genau wie beim Original.
            warnings.simplefilter("ignore")
            self.model = torch.jit.load(str(directory / "nexto-model.pt"), map_location="cpu")
        self.model.eval()
        self.chunk = max(1, int(chunk))
        self.calls = 0
        self.samples = 0
        self._last: dict[str, np.ndarray] = {}

    # -- Antworten ---------------------------------------------------------
    def logits(self, compacts: np.ndarray, rows: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Lehrer-Antworten für ``B`` Spielstände.

        ``compacts``: ``(B, D)`` aus :func:`encode_compact`, ``rows``: welches
        Auto in jedem Spielstand gefragt ist, ``previous``: ``(B, 8)`` dessen
        letzte Eingabe.
        """
        import torch

        compacts = np.atleast_2d(np.asarray(compacts, dtype=np.float32))
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        previous = np.asarray(previous, dtype=np.float32).reshape(len(compacts), 8)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(compacts), self.chunk):
                stop = min(start + self.chunk, len(compacts))
                encoded = expand_compacts(compacts[start:stop])
                # Nexto liefert eine (q, kv, mask)-Gruppe je Auto.
                built = self.builder.batched_build_obs(encoded)
                count = stop - start
                index = np.arange(count)
                pick = rows[start:stop]
                q = np.stack([part[0] for part in built])[pick, index]  # (B, 1, 32)
                q[:, 0, 24:32] = previous[start:stop]
                kv = np.stack([part[1] for part in built])[pick, index]
                mask = np.stack([part[2] for part in built])[pick, index]
                answer, _ = self.model(
                    (
                        torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)),
                        torch.from_numpy(np.ascontiguousarray(kv, dtype=np.float32)),
                        torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
                    )
                )
                out.append(answer.numpy())
                self.samples += count
        self.calls += 1
        return np.concatenate(out, axis=0)

    def probs(
        self, compacts: np.ndarray, rows: np.ndarray, previous: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        logits = self.logits(compacts, rows, previous) / max(1e-3, float(temperature))
        logits = logits - logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True)

    def actions(self, compacts: np.ndarray, rows: np.ndarray, previous: np.ndarray) -> np.ndarray:
        return self.logits(compacts, rows, previous).argmax(axis=-1)

    # -- Einzelabfrage (Live-Ansicht, echtes Spiel) ------------------------
    def act_state(self, state: Any, agents: list[str]) -> dict[str, int]:
        players = canonical_players(state)
        table = lookup_table()
        compacts = np.stack([encode_compact(state, players) for _ in agents])
        rows = np.asarray([players.index(agent) for agent in agents], dtype=np.int64)
        previous = np.stack([self._last.get(agent, table[IDLE_ACTION]) for agent in agents])
        actions = self.actions(compacts, rows, previous)
        self._last = {
            agent: table[int(action)] for agent, action in zip(agents, actions, strict=True)
        }
        return {agent: int(action) for agent, action in zip(agents, actions, strict=True)}


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(1e-3, float(temperature))
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(scaled)
    return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float64)


class TeacherPlayer(Player):
    """Der Lehrer als Spieler: für Live-Ansicht, Replays und echte Spiele."""

    def __init__(
        self,
        teacher: NextoTeacher | None = None,
        deterministic: bool = True,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.teacher = teacher or NextoTeacher()
        self.deterministic = deterministic
        self.temperature = temperature
        self.name = f"Lehrer ({TEACHER_LABEL})"
        self.explain = False
        self.last_info: dict[str, Any] = {}
        self._last: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    def act(self, agents: list[str], obs: dict[str, np.ndarray], state: Any) -> dict[str, int]:
        if not agents:
            return {}
        table = lookup_table()
        players = canonical_players(state)
        compacts = np.stack([encode_compact(state, players) for _ in agents])
        rows = np.asarray([players.index(agent) for agent in agents], dtype=np.int64)
        previous = np.stack([self._last.get(agent, table[IDLE_ACTION]) for agent in agents])
        probs = self.teacher.probs(compacts, rows, previous, self.temperature)
        if self.deterministic:
            actions = probs.argmax(axis=-1)
        else:
            actions = np.array([self._rng.choice(len(p), p=p / p.sum()) for p in probs])
        self._last = {agent: table[int(a)] for agent, a in zip(agents, actions, strict=True)}
        if self.explain:
            self.last_info = {
                agent: explain_decision(probs[i], 0.0, int(actions[i]))
                for i, agent in enumerate(agents)
            }
        return {agent: int(action) for agent, action in zip(agents, actions, strict=True)}


# --------------------------------------------------------------------------
# Fürs Training: Antworten im Stapel nachrechnen
# --------------------------------------------------------------------------


class TeacherLabeler:
    """Rechnet die Lehrer-Antworten für gesammelte Trainingsschritte aus."""

    def __init__(self, progress: Callable[[str], None] | None = None, threads: int = 0):
        import torch

        if threads:
            torch.set_num_threads(int(threads))
        self.teacher = NextoTeacher(progress=progress)

    def targets(
        self,
        compacts: np.ndarray,
        rows: np.ndarray,
        previous: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Wahrscheinlichkeiten des Lehrers, ``(S, 90)`` float32."""
        return self.teacher.probs(compacts, rows, previous, temperature).astype(np.float32)


def teacher_weight_at(steps: int, start: float, final: float, decay_steps: int) -> float:
    """Der Anteil, mit dem der Lehrer anfangs mitspricht (fällt mit der Zeit)."""
    if start <= 0 and final <= 0:
        return 0.0
    if decay_steps <= 0:
        return float(start)
    done = min(1.0, max(0.0, steps / float(decay_steps)))
    return float(start + (final - start) * done)


def describe_teacher() -> dict[str, Any]:
    """Kurzbericht für die Oberfläche / ``doctor``."""
    directory = teacher_dir()
    ready = teacher_ready(directory)
    return {
        "name": TEACHER_LABEL,
        "ready": ready,
        "directory": str(directory),
        "files": sorted(FILES),
        "source": f"https://github.com/{SOURCE_REPO}/tree/{SOURCE_BRANCH}/{SOURCE_DIR}",
        "license": "GPL-3.0 (nur lokal, nur offline)",
    }
