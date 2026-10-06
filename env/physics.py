"""Small deterministic 3D movement and ray-collision helpers.

There is no renderer or display dependency here. Actors are vertical capsules
approximated by cylinders for shots and circles against map AABBs for movement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from env.maps import ArenaMap, ArenaObject


STANCE_HEIGHTS = {0: 1.8, 1: 1.2, 2: 0.5}


@dataclass
class AgentBody:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    hp: float = 100.0
    stance: int = 0
    lean: int = 0
    sprinting: bool = False
    in_air: bool = False
    vertical_velocity: float = 0.0
    movement_speed: float = 4.5

    @property
    def height(self) -> float:
        return STANCE_HEIGHTS.get(int(self.stance), 1.8)

    @property
    def eye_height(self) -> float:
        return self.height * 0.88

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def direction_from_angles(yaw: float, pitch: float) -> tuple[float, float, float]:
    cos_pitch = math.cos(pitch)
    return (
        math.sin(yaw) * cos_pitch,
        math.cos(yaw) * cos_pitch,
        math.sin(pitch),
    )


def ground_height(arena_map: ArenaMap, x: float, y: float) -> float:
    """Return the highest traversable floor at a world x/y coordinate."""
    floor = 0.0
    for item in arena_map.objects:
        if item.kind == "platform":
            if item.x_min <= x <= item.x_max and item.y_min <= y <= item.y_max:
                floor = max(floor, item.z + item.height)
        elif item.kind == "ramp":
            # Ramps rise along their local +y axis and join the upper deck.
            dx, dy = x - item.x, y - item.y
            local_y = -math.sin(item.yaw) * dx + math.cos(item.yaw) * dy
            local_x = math.cos(item.yaw) * dx + math.sin(item.yaw) * dy
            if abs(local_x) <= item.width / 2 and abs(local_y) <= item.depth / 2:
                fraction = min(1.0, max(0.0, local_y / max(item.depth, 1e-6) + 0.5))
                floor = max(floor, item.z + item.height * fraction)
    return floor


def _circle_overlaps_aabb(x: float, y: float, radius: float, item: ArenaObject) -> bool:
    nearest_x = min(item.x_max, max(item.x_min, x))
    nearest_y = min(item.y_max, max(item.y_min, y))
    dx, dy = x - nearest_x, y - nearest_y
    return dx * dx + dy * dy < radius * radius


def _body_collides(body: AgentBody, arena_map: ArenaMap, x: float, y: float, z: float,
                   radius: float = 0.34) -> bool:
    if x < -arena_map.width / 2 + radius or x > arena_map.width / 2 - radius:
        return True
    if y < -arena_map.depth / 2 + radius or y > arena_map.depth / 2 - radius:
        return True
    body_top = z + body.height
    for item in arena_map.objects:
        if item.kind in {"ramp", "platform"}:
            continue
        if body_top <= item.z + 0.02 or z >= item.z_max - 0.02:
            continue
        if _circle_overlaps_aabb(x, y, radius, item):
            return True
    return False


def move_body(
    body: AgentBody,
    arena_map: ArenaMap,
    move_forward: float,
    move_strafe: float,
    look_yaw: float,
    look_pitch: float,
    sprint: bool,
    crouch_or_prone: int,
    lean: int,
    jump: bool,
    dt: float,
) -> None:
    """Advance one physics tick with slide collision, stance and gravity."""
    dt = max(0.0, float(dt))
    body.yaw = wrap_angle(body.yaw + float(look_yaw) * 2.7 * dt)
    body.pitch = min(math.radians(80), max(math.radians(-80),
                                            body.pitch + float(look_pitch) * 2.0 * dt))
    body.stance = int(min(2, max(0, crouch_or_prone)))
    body.lean = int(min(1, max(-1, lean)))
    body.sprinting = bool(sprint and body.stance == 0)

    forward = max(-1.0, min(1.0, float(move_forward)))
    strafe = max(-1.0, min(1.0, float(move_strafe)))
    norm = math.hypot(forward, strafe)
    if norm > 1.0:
        forward /= norm
        strafe /= norm
    speed_scale = 1.0
    if body.stance == 1:
        speed_scale = 0.62
    elif body.stance == 2:
        speed_scale = 0.32
    if body.sprinting:
        speed_scale = 1.48
    body.movement_speed = 4.5 * speed_scale

    forward_x, forward_y = math.sin(body.yaw), math.cos(body.yaw)
    right_x, right_y = math.cos(body.yaw), -math.sin(body.yaw)
    dx = (forward * forward_x + strafe * right_x) * body.movement_speed * dt
    dy = (forward * forward_y + strafe * right_y) * body.movement_speed * dt

    old_x, old_y, old_z = body.x, body.y, body.z
    candidate_x = old_x + dx
    candidate_y = old_y + dy
    candidate_ground = ground_height(arena_map, candidate_x, candidate_y)
    # A short ledge remains traversable, while falling from upper floors is physical.
    if candidate_ground - old_z > 0.42:
        candidate_ground = old_z
        candidate_x, candidate_y = old_x, old_y

    if not _body_collides(body, arena_map, candidate_x, old_y, max(old_z, candidate_ground)):
        body.x = candidate_x
    if not _body_collides(body, arena_map, body.x, candidate_y, max(old_z, candidate_ground)):
        body.y = candidate_y

    floor = ground_height(arena_map, body.x, body.y)
    if jump and not body.in_air:
        body.vertical_velocity = 5.25
        body.in_air = True
    if body.in_air:
        body.z += body.vertical_velocity * dt
        body.vertical_velocity -= 9.81 * dt
        floor = ground_height(arena_map, body.x, body.y)
        if body.z <= floor and body.vertical_velocity <= 0.0:
            body.z = floor
            body.vertical_velocity = 0.0
            body.in_air = False
    else:
        if floor < body.z - 0.05:
            body.in_air = True
            body.vertical_velocity = min(body.vertical_velocity, 0.0)
        else:
            body.z = floor

    # Keep an actor inside the playable volume even after a high-speed action.
    body.x = min(arena_map.width / 2 - 0.34, max(-arena_map.width / 2 + 0.34, body.x))
    body.y = min(arena_map.depth / 2 - 0.34, max(-arena_map.depth / 2 + 0.34, body.y))


def ray_aabb_distance(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    item: ArenaObject,
    max_distance: float = math.inf,
) -> float | None:
    """Ray/AABB slab intersection; return the nearest non-negative distance."""
    bounds = (
        (item.x_min, item.x_max),
        (item.y_min, item.y_max),
        (item.z, item.z_max),
    )
    near, far = 0.0, max_distance
    for axis in range(3):
        origin_axis = origin[axis]
        direction_axis = direction[axis]
        low, high = bounds[axis]
        if abs(direction_axis) < 1e-12:
            if origin_axis < low or origin_axis > high:
                return None
            continue
        first = (low - origin_axis) / direction_axis
        second = (high - origin_axis) / direction_axis
        if first > second:
            first, second = second, first
        near = max(near, first)
        far = min(far, second)
        if near > far:
            return None
    if far < 0.0:
        return None
    return near if near >= 0.0 else far


def raycast_scene(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    arena_map: ArenaMap,
    max_distance: float = math.inf,
) -> tuple[float | None, ArenaObject | None]:
    """Find the nearest map object intersected by a normalized ray."""
    nearest: float | None = None
    nearest_object: ArenaObject | None = None
    for item in arena_map.objects:
        distance = ray_aabb_distance(origin, direction, item,
                                     max_distance if nearest is None else min(max_distance, nearest))
        if distance is not None and (nearest is None or distance < nearest):
            nearest = distance
            nearest_object = item
    return nearest, nearest_object


def distance_to_map_boundary(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    arena_map: ArenaMap,
) -> float:
    """Distance to the rectangular playable boundary along a horizontal ray."""
    candidates: list[float] = []
    dx, dy = direction[0], direction[1]
    half_w, half_d = arena_map.width / 2, arena_map.depth / 2
    if dx > 1e-12:
        candidates.append((half_w - origin[0]) / dx)
    elif dx < -1e-12:
        candidates.append((-half_w - origin[0]) / dx)
    if dy > 1e-12:
        candidates.append((half_d - origin[1]) / dy)
    elif dy < -1e-12:
        candidates.append((-half_d - origin[1]) / dy)
    positive = [value for value in candidates if value >= 0.0]
    return min(positive) if positive else math.inf


def raycast_distance(
    origin: tuple[float, float, float],
    yaw: float,
    pitch: float,
    arena_map: ArenaMap,
    max_distance: float = 100.0,
) -> float:
    """Distance to a wall/cover object or map edge along an aim ray."""
    direction = direction_from_angles(yaw, pitch)
    object_distance, _ = raycast_scene(origin, direction, arena_map, max_distance)
    edge_distance = distance_to_map_boundary(origin, direction, arena_map)
    distances = [max_distance, edge_distance]
    if object_distance is not None:
        distances.append(object_distance)
    return float(min(distances))


def ray_cylinder_hit(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    body: AgentBody,
    radius: float = 0.38,
) -> tuple[float, bool] | None:
    """Intersect a ray with a standing/crouched vertical cylinder.

    The upper 20 percent of the visible hit volume is the headshot zone.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    rx, ry = ox - body.x, oy - body.y
    a = dx * dx + dy * dy
    if a < 1e-12:
        if rx * rx + ry * ry > radius * radius:
            return None
        if abs(dz) < 1e-12:
            return None
        candidates = [(body.z - oz) / dz, (body.z + body.height - oz) / dz]
        for distance in sorted(value for value in candidates if value >= 0.0):
            hit_z = oz + distance * dz
            if body.z <= hit_z <= body.z + body.height:
                return distance, hit_z >= body.z + 0.8 * body.height
        return None

    b = 2.0 * (rx * dx + ry * dy)
    c = rx * rx + ry * ry - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    for distance in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
        if distance < 0.0:
            continue
        hit_z = oz + distance * dz
        if body.z <= hit_z <= body.z + body.height:
            return distance, hit_z >= body.z + 0.8 * body.height
    return None


def has_line_of_sight(source: AgentBody, target: AgentBody, arena_map: ArenaMap) -> bool:
    origin = (source.x, source.y, source.z + source.eye_height)
    target_point = (target.x, target.y, target.z + target.height * 0.75)
    vector = tuple(target_point[index] - origin[index] for index in range(3))
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-6:
        return True
    direction = tuple(value / length for value in vector)
    blocker_distance, _ = raycast_scene(origin, direction, arena_map, length)
    return blocker_distance is None or blocker_distance >= length - 0.4
