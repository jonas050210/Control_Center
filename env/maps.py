"""Data-driven maps used by the simulation and browser visualizations.

Coordinates use metres. Object x/y are the centre of the footprint and z is
its bottom elevation, which keeps map authoring and collision math intuitive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random


@dataclass(frozen=True)
class ArenaObject:
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float
    kind: str
    name: str = ""
    yaw: float = 0.0

    @property
    def x_min(self) -> float:
        return self.x - self.width / 2.0

    @property
    def x_max(self) -> float:
        return self.x + self.width / 2.0

    @property
    def y_min(self) -> float:
        return self.y - self.depth / 2.0

    @property
    def y_max(self) -> float:
        return self.y + self.depth / 2.0

    @property
    def z_max(self) -> float:
        return self.z + self.height


@dataclass
class ArenaMap:
    name: str
    width: float
    depth: float
    objects: list[ArenaObject]
    spawn_points: tuple[tuple[float, float], tuple[float, float]]
    description: str = ""

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def cover_density(self) -> float:
        """Approximate percentage of floor footprint occupied by cover."""
        covered = sum(obj.width * obj.depth for obj in self.objects if obj.kind != "ramp")
        return min(100.0, 100.0 * covered / max(1.0, self.area))

    @property
    def average_sightline(self) -> float:
        """A useful map-scale sightline estimate, reduced by dense cover."""
        diagonal_scale = math.hypot(self.width, self.depth) * 0.55
        return max(4.0, diagonal_scale / (1.0 + self.cover_density / 35.0))

    def copy(self) -> "ArenaMap":
        return ArenaMap(
            name=self.name,
            width=self.width,
            depth=self.depth,
            objects=list(self.objects),
            spawn_points=tuple(self.spawn_points),
            description=self.description,
        )


MAP_NAMES = ("Dust", "Warehouse", "Highrise", "Arena", "Sniper Alley", "Custom")


def _obj(x: float, y: float, width: float, height: float, depth: float, kind: str,
         name: str = "", z: float = 0.0, yaw: float = 0.0) -> ArenaObject:
    return ArenaObject(x=x, y=y, z=z, width=width, height=height, depth=depth,
                       kind=kind, name=name, yaw=yaw)


def _dust() -> ArenaMap:
    objects = [
        _obj(-8, -8, 3, 1.5, 3, "crate", "Crate A"),
        _obj(8, -8, 3, 1.5, 3, "crate", "Crate B"),
        _obj(-8, 8, 3, 1.5, 3, "crate", "Crate C"),
        _obj(8, 8, 3, 1.5, 3, "crate", "Crate D"),
        _obj(-2, 1, 1.2, 1.1, 1.2, "barrel", "Barrel A"),
        _obj(3, -1, 1.2, 1.1, 1.2, "barrel", "Barrel B"),
    ]
    return ArenaMap("Dust", 50, 50, objects, ((-18, 0), (18, 0)),
                    "Open desert test range with scattered crates and barrels.")


def _warehouse() -> ArenaMap:
    objects = [
        _obj(-6.5, 0, 0.7, 4.5, 13, "wall", "West partition"),
        _obj(6.5, 0, 0.7, 4.5, 13, "wall", "East partition"),
        _obj(-13, -5, 1.0, 2.4, 5, "shelf", "West shelf 1"),
        _obj(-13, 5, 1.0, 2.4, 5, "shelf", "West shelf 2"),
        _obj(13, -5, 1.0, 2.4, 5, "shelf", "East shelf 1"),
        _obj(13, 5, 1.0, 2.4, 5, "shelf", "East shelf 2"),
        _obj(-1.8, -8.5, 2.8, 1.5, 2.3, "crate", "Pallet stack A"),
        _obj(2.2, 8.0, 2.8, 1.5, 2.3, "crate", "Pallet stack B"),
        _obj(0, -13.5, 40, 4.5, 0.5, "wall", "South warehouse wall"),
        _obj(0, 13.5, 40, 4.5, 0.5, "wall", "North warehouse wall"),
    ]
    return ArenaMap("Warehouse", 40, 30, objects, ((-16, 0), (16, 0)),
                    "Tight industrial corridors with shelves, partition walls, and pallet stacks.")


def _highrise() -> ArenaMap:
    objects = [
        # A walkable inclined surface reaches the upper platform at z=3m.
        _obj(-7.0, -2.5, 4.0, 3.0, 12.0, "ramp", "Service ramp"),
        _obj(-7.0, 7.5, 11.0, 0.25, 8.0, "platform", "Upper deck", z=2.75),
        _obj(-12.6, 7.2, 0.3, 1.2, 7.8, "rail", "Upper rail west", z=3.0),
        _obj(-1.4, 7.2, 0.3, 1.2, 7.8, "rail", "Upper rail east", z=3.0),
        _obj(5, -6, 3.0, 1.5, 2.5, "crate", "Ground crate A"),
        _obj(10, 2, 3.0, 1.5, 2.5, "crate", "Ground crate B"),
        _obj(-11, -9, 3.0, 1.5, 2.5, "crate", "Ground crate C"),
        _obj(4.0, 8.0, 1.0, 1.2, 1.0, "barrel", "Ground barrel"),
        _obj(-7.0, 8.0, 3.0, 1.4, 2.4, "crate", "Upper cover", z=3.0),
    ]
    return ArenaMap("Highrise", 30, 30, objects, ((11, -10), (11, 10)),
                    "Two-level rooftop with a walkable ramp, elevated deck, railings, and cover.")


def _arena() -> ArenaMap:
    objects: list[ArenaObject] = []
    radius = 12.0
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        objects.append(_obj(radius * math.cos(angle), radius * math.sin(angle),
                            1.5, 4.2, 1.5, "pillar", f"Ring pillar {index + 1}"))
    objects.extend([
        _obj(0, 0, 4.0, 1.4, 2.0, "crate", "Centre block"),
        _obj(-3.4, 0, 1.1, 1.2, 1.1, "barrel", "Centre barrel west"),
        _obj(3.4, 0, 1.1, 1.2, 1.1, "barrel", "Centre barrel east"),
    ])
    return ArenaMap("Arena", 40, 40, objects, ((-15, 0), (15, 0)),
                    "Circular combat arena with a pillar ring and a compact central cover cluster.")


def _sniper_alley() -> ArenaMap:
    objects = [
        _obj(0, -9.2, 80, 4.2, 0.6, "wall", "South side wall"),
        _obj(0, 9.2, 80, 4.2, 0.6, "wall", "North side wall"),
        _obj(-18, -3.7, 5.0, 2.2, 1.2, "crate", "West lane crate"),
        _obj(14, 3.7, 5.0, 2.2, 1.2, "crate", "East lane crate"),
        _obj(-4, 0, 1.2, 1.0, 1.2, "barrel", "Lane barrel west"),
        _obj(4, 0, 1.2, 1.0, 1.2, "barrel", "Lane barrel east"),
        # Gaps in each segmented side wall act as windows and firing ports.
        _obj(-30, -8.0, 11.0, 3.2, 0.5, "wall", "West window wall A"),
        _obj(-10, -8.0, 11.0, 3.2, 0.5, "wall", "West window wall B"),
        _obj(10, -8.0, 11.0, 3.2, 0.5, "wall", "West window wall C"),
        _obj(30, -8.0, 11.0, 3.2, 0.5, "wall", "West window wall D"),
        _obj(-30, 8.0, 11.0, 3.2, 0.5, "wall", "East window wall A"),
        _obj(-10, 8.0, 11.0, 3.2, 0.5, "wall", "East window wall B"),
        _obj(10, 8.0, 11.0, 3.2, 0.5, "wall", "East window wall C"),
        _obj(30, 8.0, 11.0, 3.2, 0.5, "wall", "East window wall D"),
    ]
    return ArenaMap("Sniper Alley", 80, 20, objects, ((-34, 0), (34, 0)),
                    "Long, straight firing lane bordered by segmented side walls and window ports.")


def _custom() -> ArenaMap:
    return ArenaMap("Custom", 50, 50, [], ((-18, 0), (18, 0)),
                    "Empty 50 by 50 metre sandbox for custom or randomized cover layouts.")


_MAP_BUILDERS = {
    "Dust": _dust,
    "Warehouse": _warehouse,
    "Highrise": _highrise,
    "Arena": _arena,
    "Sniper Alley": _sniper_alley,
    "Custom": _custom,
}


def create_map(name: str = "Dust", seed: int | None = None, randomize: bool = False) -> ArenaMap:
    """Build a fresh map. ``randomize`` perturbs cover without mutating templates."""
    canonical_name = next((item for item in MAP_NAMES if item.lower() == name.lower()), None)
    if canonical_name is None:
        raise ValueError(f"Unknown map {name!r}. Choose one of: {', '.join(MAP_NAMES)}")
    arena_map = _MAP_BUILDERS[canonical_name]()
    if randomize:
        return randomize_cover(arena_map, seed=seed)
    return arena_map


def randomize_cover(arena_map: ArenaMap, seed: int | None = None) -> ArenaMap:
    """Return a new layout with safe bounds and deterministic seeded placement."""
    rng = random.Random(seed)
    result = arena_map.copy()
    if result.name == "Custom" and not result.objects:
        candidates = ["crate", "barrel", "pillar"]
        count = 10
        for index in range(count):
            kind = rng.choice(candidates)
            width = rng.uniform(1.0, 3.2)
            depth = rng.uniform(1.0, 3.2)
            height = rng.uniform(1.0, 4.0) if kind != "barrel" else rng.uniform(0.8, 1.5)
            x = rng.uniform(-result.width * 0.34, result.width * 0.34)
            y = rng.uniform(-result.depth * 0.34, result.depth * 0.34)
            if min(math.hypot(x - sx, y - sy) for sx, sy in result.spawn_points) < 6.0:
                x = -x * 0.55
                y = -y * 0.55
            result.objects.append(_obj(x, y, width, height, depth, kind,
                                       f"Random cover {index + 1}"))
        return result

    perturbed: list[ArenaObject] = []
    for item in result.objects:
        # Keep structural walls and the Highrise traversal geometry fixed.
        if item.kind in {"wall", "rail", "ramp", "platform"}:
            perturbed.append(item)
            continue
        dx = rng.uniform(-2.2, 2.2)
        dy = rng.uniform(-2.2, 2.2)
        x = max(-result.width / 2 + item.width / 2 + 1.0,
                min(result.width / 2 - item.width / 2 - 1.0, item.x + dx))
        y = max(-result.depth / 2 + item.depth / 2 + 1.0,
                min(result.depth / 2 - item.depth / 2 - 1.0, item.y + dy))
        perturbed.append(replace(item, x=x, y=y))
    result.objects = perturbed
    return result


def map_names() -> tuple[str, ...]:
    return MAP_NAMES


def serialize_map(arena_map: ArenaMap) -> dict[str, object]:
    """Convert a map and its objects to a JSON-friendly snapshot."""
    return {
        "name": arena_map.name,
        "width": arena_map.width,
        "depth": arena_map.depth,
        "description": arena_map.description,
        "spawn_points": [list(point) for point in arena_map.spawn_points],
        "objects": [
            {
                "x": item.x,
                "y": item.y,
                "z": item.z,
                "width": item.width,
                "height": item.height,
                "depth": item.depth,
                "kind": item.kind,
                "name": item.name,
                "yaw": item.yaw,
            }
            for item in arena_map.objects
        ],
    }
