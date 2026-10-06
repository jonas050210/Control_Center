"""Pure game-state helpers for the browser-playable RL training drills.

Moved verbatim from ``gui/games.py`` so the drills no longer depend on a UI
framework: the state is a plain dictionary that the API serializes.
"""

from __future__ import annotations

import random
import time
from typing import Any


DODGE_GRID_SIZE = 5


def new_dodge_game(seed: int | None = None) -> dict[str, Any]:
    """Create a 5×5 survival run controlled with a compact desktop D-pad."""
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    return {
        "active": True,
        "seed": int(seed),
        "tick": 0,
        "player": [DODGE_GRID_SIZE // 2, DODGE_GRID_SIZE - 1],
        "projectiles": [],
        "score": 0,
        "dodged": 0,
        "hit": False,
    }


def step_dodge_game(game: dict[str, Any], dx: int = 0, dy: int = 0) -> dict[str, Any]:
    """Advance one dodge turn and mutate/return the game object."""
    if not game.get("active", False):
        return game
    size = DODGE_GRID_SIZE
    player_x, player_y = game["player"]
    player_x = min(size - 1, max(0, int(player_x) + int(dx)))
    player_y = min(size - 1, max(0, int(player_y) + int(dy)))
    tick = int(game["tick"]) + 1
    rng = random.Random(int(game["seed"]) + tick * 1_000_003)

    shifted: list[list[int]] = []
    cleared = 0
    for projectile_x, projectile_y in game["projectiles"]:
        next_y = int(projectile_y) + 1
        if next_y >= size:
            cleared += 1
        else:
            shifted.append([int(projectile_x), next_y])

    spawn_probability = min(0.62, 0.22 + tick * 0.008)
    if rng.random() < spawn_probability:
        shifted.append([rng.randrange(size), 0])
    if tick >= 18 and rng.random() < min(0.24, (tick - 17) * 0.006):
        second_lane = rng.randrange(size)
        if not any(projectile[0] == second_lane and projectile[1] == 0 for projectile in shifted):
            shifted.append([second_lane, 0])

    hit = any(projectile_x == player_x and projectile_y == player_y
              for projectile_x, projectile_y in shifted)
    game["player"] = [player_x, player_y]
    game["projectiles"] = shifted
    game["tick"] = tick
    game["dodged"] = int(game["dodged"]) + cleared
    game["score"] = tick + int(game["dodged"]) * 2
    game["hit"] = bool(hit)
    game["active"] = not hit
    return game


def new_aim_game(duration_seconds: float = 30.0, seed: int | None = None) -> dict[str, Any]:
    """Create a timed aim drill with a seeded but unpredictable target sequence."""
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    rng = random.Random(int(seed))
    target = rng.randrange(9)
    return {
        "active": True,
        "seed": int(seed),
        "rng": rng,
        "started_at": time.monotonic(),
        "duration": max(5.0, float(duration_seconds)),
        "target": target,
        "last_target_at": time.monotonic(),
        "hits": 0,
        "misses": 0,
        "streak": 0,
        "best_streak": 0,
        "reaction_ms": [],
    }


def aim_tap(game: dict[str, Any], cell: int, now: float | None = None) -> bool:
    """Register one grid tap; return whether it hit the current target."""
    if not game.get("active", False):
        return False
    if now is None:
        now = time.monotonic()
    if now - float(game["started_at"]) >= float(game["duration"]):
        game["active"] = False
        return False
    if int(cell) == int(game["target"]):
        game["hits"] += 1
        game["streak"] += 1
        game["best_streak"] = max(game["best_streak"], game["streak"])
        game["reaction_ms"].append(max(0.0, (now - float(game["last_target_at"])) * 1000.0))
        rng = game["rng"]
        next_target = rng.randrange(8)
        current = int(game["target"])
        game["target"] = next_target if next_target < current else next_target + 1
        game["last_target_at"] = now
        return True
    game["misses"] += 1
    game["streak"] = 0
    return False


def public_aim_game(game: dict[str, Any] | None, now: float | None = None) -> dict[str, Any] | None:
    """Return a JSON-serializable aim-drill view (the RNG object is stripped)."""
    if game is None:
        return None
    if now is None:
        now = time.monotonic()
    duration = float(game["duration"])
    elapsed = max(0.0, now - float(game["started_at"]))
    hits = int(game["hits"])
    misses = int(game["misses"])
    reactions = [float(value) for value in game["reaction_ms"]]
    return {
        "active": bool(game["active"]) and elapsed < duration,
        "target": int(game["target"]),
        "hits": hits,
        "misses": misses,
        "streak": int(game["streak"]),
        "best_streak": int(game["best_streak"]),
        "duration": duration,
        "remaining": max(0.0, duration - elapsed),
        "accuracy": hits / max(1, hits + misses),
        "avg_reaction_ms": (sum(reactions) / len(reactions)) if reactions else 0.0,
        "reaction_ms": reactions[-40:],
        "seed": int(game["seed"]),
    }
