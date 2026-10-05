"""Was die KI sieht: die Beobachtung für Training und echtes Spiel.

Basis ist RLGyms ``DefaultObs`` (Ball, Boost-Pads, eigenes Auto, Teamkollegen
und Gegner — jeweils relativ zum eigenen Tor gespiegelt). Darauf setzt
:class:`ExtraObs` einen kleinen, dokumentierten Zusatzblock, der genau die
Informationen liefert, die in der Basis fehlen oder dort mühsam zu lernen sind:

======================  ====  ==================================================
Wert                    Größe  Bedeutung
======================  ====  ==================================================
``ball_relative``          3   Ballposition relativ zum eigenen Auto
``ball_distance``          1   Abstand zum Ball (0 … 1, 0 = direkt dran)
``ball_speed``             1   Ballgeschwindigkeit (0 … 1)
``ball_spin``              1   Balldrehung (0 … 1)
``ball_height``            1   Ballhöhe über dem Boden (0 … 1)
``own_goal_danger``        1   +1, wenn der Ball direkt auf unser Tor zufliegt
``ball_in_our_half``       1   1, wenn der Ball in unserer Hälfte ist
``boost``                  1   eigener Boost (0 … 1)
``on_ground``              1   1, wenn die Räder den Boden berühren
``supersonic``             1   1, wenn das Auto Überschall fährt
======================  ====  ==================================================

Training und echtes Spiel benutzen **denselben** Bauplan; ``tests/test_rlbot.py``
vergleicht beide Wege Schritt für Schritt. Die Größe wird im Checkpoint
gespeichert (``obs_size``), damit ein Bot immer die Beobachtung baut, mit der
seine Gewichte trainiert wurden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BACK_NET_Y,
    BALL_MAX_SPEED,
    CEILING_Z,
    ORANGE_TEAM,
)
from rlgym.rocket_league.obs_builders import DefaultObs

from .config import OBS_BASE_SIZE, OBS_EXTRA_SIZE, OBS_PADDING, OBS_SIZE

#: Namen der Zusatzwerte in Reihenfolge — die Oberfläche zeigt sie so an.
EXTRA_FEATURES: tuple[str, ...] = (
    "ball_x_rel",
    "ball_y_rel",
    "ball_z_rel",
    "ball_distance",
    "ball_speed",
    "ball_spin",
    "ball_height",
    "own_goal_danger",
    "ball_in_our_half",
    "boost",
    "on_ground",
    "supersonic",
)
assert len(EXTRA_FEATURES) == OBS_EXTRA_SIZE

#: Deutsche Beschriftungen für die Oberfläche ("Was die KI sieht").
EXTRA_LABELS: dict[str, str] = {
    "ball_x_rel": "Ball seitlich (relativ)",
    "ball_y_rel": "Ball vor/hinter (relativ)",
    "ball_z_rel": "Ball über/unter (relativ)",
    "ball_distance": "Abstand zum Ball",
    "ball_speed": "Ballgeschwindigkeit",
    "ball_spin": "Balldrehung",
    "ball_height": "Ballhöhe",
    "own_goal_danger": "Ball fliegt auf unser Tor",
    "ball_in_our_half": "Ball in unserer Hälfte",
    "boost": "Eigener Boost",
    "on_ground": "Räder am Boden",
    "supersonic": "Überschall",
}

#: Einheit/Zahlenformat je Wert: "percent" (0…1), "signed" (−1…1), "raw".
EXTRA_FORMAT: dict[str, str] = {
    "ball_distance": "percent",
    "ball_speed": "percent",
    "ball_spin": "percent",
    "ball_height": "percent",
    "own_goal_danger": "signed",
    "ball_in_our_half": "flag",
    "boost": "percent",
    "on_ground": "flag",
    "supersonic": "flag",
}
EXTRA_GROUPS: dict[str, str] = {
    "ball_x_rel": "ball",
    "ball_y_rel": "ball",
    "ball_z_rel": "ball",
    "ball_distance": "ball",
    "ball_speed": "ball",
    "ball_spin": "ball",
    "ball_height": "ball",
    "own_goal_danger": "tactics",
    "ball_in_our_half": "tactics",
    "boost": "car",
    "on_ground": "car",
    "supersonic": "car",
}

GROUP_LABELS: dict[str, str] = {
    "ball": "Ball & Distanz",
    "tactics": "Taktische Lage",
    "car": "Eigenes Auto",
}



def describe_features() -> list[dict[str, str]]:
    """Beschreibung aller Zusatzwerte für die Weboberfläche."""
    return [
        {
            "key": key,
            "label": EXTRA_LABELS.get(key, key),
            "format": EXTRA_FORMAT.get(key, "signed"),
            "group": EXTRA_GROUPS.get(key, "other"),
            "group_label": GROUP_LABELS.get(EXTRA_GROUPS.get(key, "other"), "Allgemein"),
        }
        for key in EXTRA_FEATURES
    ]


POS_COEF = 1 / 2300
SPIN_COEF = 1 / 6.0  # sehr schnelle Bälle drehen mit ~6 rad/s


def obs_size(extras: bool = True) -> int:
    """Eingabegröße des Netzes für diesen Beobachtungs-Bauplan."""
    return OBS_SIZE if extras else OBS_BASE_SIZE


def build_obs_builder(extras: bool = True, zero_padding: int = OBS_PADDING) -> ObsBuilder:
    """``DefaultObs`` oder die erweiterte Variante — sonst identisch konfiguriert."""
    if extras:
        return ExtraObs(zero_padding=zero_padding)
    return DefaultObs(zero_padding=zero_padding)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros_like(vector)


def extra_values(agent: AgentID, state: GameState) -> np.ndarray:
    """Die :data:`EXTRA_FEATURES` für ein Auto (eigener Test- und Prüfpfad)."""
    car = state.cars[agent]
    inverted = int(car.team_num) == ORANGE_TEAM
    ball = state.inverted_ball if inverted else state.ball
    physics = car.inverted_physics if inverted else car.physics
    # Im (ggf. gespiegelten) Blickwinkel liegt das eigene Tor immer bei -y.
    own_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)

    relative = (ball.position - physics.position).astype(np.float32)
    distance = float(np.linalg.norm(relative))
    ball_velocity = np.asarray(ball.linear_velocity, dtype=np.float32)
    speed = float(np.linalg.norm(ball_velocity))
    spin = float(np.linalg.norm(np.asarray(ball.angular_velocity, dtype=np.float32)))
    to_own_goal = _unit(own_goal - np.asarray(ball.position, dtype=np.float32))
    danger = -float(np.dot(_unit(ball_velocity), to_own_goal)) if speed > 1e-3 else 0.0

    values = np.array(
        [
            relative[0] * POS_COEF,
            relative[1] * POS_COEF,
            relative[2] * POS_COEF,
            min(1.0, distance / 6000.0),
            min(1.0, speed / BALL_MAX_SPEED),
            min(1.0, spin * SPIN_COEF),
            min(1.0, max(0.0, float(ball.position[2]) / CEILING_Z)),
            danger,
            1.0 if float(ball.position[1]) < 0.0 else 0.0,
            float(car.boost_amount) / 100.0,
            1.0 if car.on_ground else 0.0,
            1.0 if car.is_supersonic else 0.0,
        ],
        dtype=np.float32,
    )
    return np.clip(values, -1.0, 1.0)


class ExtraObs(DefaultObs):
    """``DefaultObs`` plus :data:`EXTRA_FEATURES` (siehe Modulbeschreibung)."""

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        space, size = super().get_obs_space(agent)
        return space, (size + OBS_EXTRA_SIZE if size > 0 else size)

    def _build_obs(
        self, agent: AgentID, state: GameState, shared_info: dict[str, Any]
    ) -> np.ndarray:
        base = super()._build_obs(agent, state, shared_info)
        return np.concatenate([base, extra_values(agent, state)])
