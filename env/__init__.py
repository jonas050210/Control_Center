"""Headless Gymnasium environments for NEURAL ARENA."""

from env.maps import ArenaMap, ArenaObject, MAP_NAMES, create_map
from env.shooter_env import ACTION_NVECS, OBSERVATION_SIZE, ShooterEnv
from env.weapons import WEAPON_NAMES, get_weapon

__all__ = [
    "ACTION_NVECS",
    "ArenaMap",
    "ArenaObject",
    "MAP_NAMES",
    "OBSERVATION_SIZE",
    "ShooterEnv",
    "WEAPON_NAMES",
    "create_map",
    "get_weapon",
]
