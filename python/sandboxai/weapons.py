"""Weapon roles, TTK and handling — the Python mirror of the Godot tables.

The authoritative weapon simulation is ``scripts/weapon/weapon_state.gd``.
This module does **not** re-implement it from memory: it *parses* the
GDScript constant tables (``WeaponState.PROFILE_DEFINITIONS`` and the
handling constants in ``SandboxConfig``) and reproduces the small amount of
closed-form maths that follows from them — damage falloff, shots-to-kill,
time-to-kill, the deterministic recoil pattern and the bloom cone.

Why it exists
-------------
Weapon balance is the part of an FPS RL environment that is easiest to get
quietly wrong and hardest to notice: nothing crashes when the shotgun is
secretly the best rifle, the policy simply learns something uninteresting.
Balance therefore needs *regression tests*, and those tests have to run on
a machine without the Godot engine (see ``docs/ARCHITECTURE.md`` on why the
engine is not vendored). Parsing the engine's own tables means the tests
check the shipped numbers rather than a hand-copied duplicate: if somebody
edits the GDScript, these numbers move with it and the invariants below
either still hold or fail loudly.

What it is *not*
----------------
Not a second simulator. There is no movement, no perception, no geometry
and no RNG here. Anything that depends on the actual engine loop (hit
registration against a moving target, cover, enemy behaviour) is measured
by running the engine, not by this module.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gdscript_analysis import project_root

__all__ = [
    "WeaponProfile",
    "load_profiles",
    "load_handling_constants",
    "ttk_table",
    "role_ranking",
    "format_ttk_table",
    "RANGE_BANDS",
]

## Engagement bands used for role reporting, in metres. They match the way
## the research literature and the community weapon guides talk about
## roles ("close / mid / long"), compressed onto this arena's scale: the
## square arena is 20 m across, so its diagonal (~28 m) is the longest
## sight line that can physically exist here.
## The bands are clipped to what the weapons can physically reach: the
## longest-ranged profile (rifle) stops at 15 m, so a band centred beyond
## that would only ever report "nobody can shoot here" and would hide real
## regressions behind a row of infinities.
RANGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("point_blank", 0.0, 3.0),
    ("close", 3.0, 7.0),
    ("mid", 7.0, 11.0),
    ("long", 11.0, 15.0),
)

_ENEMY_MAX_HEALTH_DEFAULT: float = 100.0

_SCALAR_RE = re.compile(
    r"^\s*const\s+([A-Z][A-Z0-9_]*)\s*:\s*(?:float|int)\s*=\s*([^#\n]+?)\s*(?:#.*)?$"
)
_PROFILE_BLOCK_RE = re.compile(r"const PROFILE_DEFINITIONS: Dictionary = \{(.*?)\n\}", re.S)
_ENTRY_RE = re.compile(r'"(?P<id>[a-z_]+)"\s*:\s*\{(?P<body>.*?)\n\t\},', re.S)
_FIELD_RE = re.compile(r'"(?P<key>[a-z_]+)"\s*:\s*(?P<value>[^,\n]+),')


def _parse_scalar_constants(source: str) -> dict[str, float]:
    """Numeric ``const NAME: float|int = <expr>`` values from a GDScript file.

    Handles the small expression forms the config actually uses (plain
    literals and products/quotients of earlier constants). Anything it
    cannot evaluate is skipped rather than guessed.
    """
    values: dict[str, float] = {}
    for line in source.splitlines():
        match = _SCALAR_RE.match(line)
        if not match:
            continue
        name, raw = match.group(1), match.group(2).strip()
        try:
            values[name] = float(eval(raw, {"__builtins__": {}}, dict(values)))  # noqa: S307
        except Exception:  # pragma: no cover - defensive, non-numeric consts
            continue
    return values


def _coerce(raw: str, constants: dict[str, float]) -> Any:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith("SandboxConfig."):
        return constants.get(raw.split(".", 1)[1])
    if raw.startswith("FIRE_MODE_"):
        return {"FIRE_MODE_AUTO": "auto", "FIRE_MODE_SEMI": "semi", "FIRE_MODE_PUMP": "pump"}[raw]
    try:
        return float(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class WeaponProfile:
    """One weapon role, mirrored from ``WeaponState.PROFILE_DEFINITIONS``."""

    profile_id: str
    label: str
    category: str
    damage: float
    range_m: float
    cooldown_time: float
    hit_radius: float
    projectile_count: int
    spread_deg: float
    falloff_start_m: float
    minimum_damage_scale: float
    fire_mode: str = "auto"
    magazine_size: int = 30
    reload_time: float = 2.0
    recoil_vertical_deg: float = 0.0
    recoil_horizontal_deg: float = 0.0
    recoil_recovery_deg_per_s: float = 18.0
    spread_per_shot_deg: float = 0.0
    spread_move_deg: float = 0.0
    spread_air_deg: float = 0.0
    spread_max_deg: float = 0.0
    spread_recovery_deg_per_s: float = 4.0
    headshot_multiplier: float = 1.0
    move_speed_scale_firing: float = 1.0
    description: str = ""

    # -- ballistics --------------------------------------------------------

    @property
    def projectile_damage(self) -> float:
        return self.damage / max(1, self.projectile_count)

    @property
    def rounds_per_minute(self) -> float:
        return 60.0 / self.cooldown_time if self.cooldown_time > 0.0 else math.inf

    def damage_scale_at(self, distance_m: float) -> float:
        """Mirror of ``WeaponState.damage_scale_at_distance``."""
        if distance_m <= self.falloff_start_m or self.range_m <= self.falloff_start_m:
            return 1.0
        if distance_m >= self.range_m:
            return self.minimum_damage_scale
        alpha = (distance_m - self.falloff_start_m) / (self.range_m - self.falloff_start_m)
        return 1.0 + (self.minimum_damage_scale - 1.0) * min(max(alpha, 0.0), 1.0)

    def projectile_damage_at(self, distance_m: float) -> float:
        return self.projectile_damage * self.damage_scale_at(distance_m)

    def volley_damage_at(
        self, distance_m: float, pellets_landed: int | None = None, headshot: bool = False
    ) -> float:
        landed = self.projectile_count if pellets_landed is None else pellets_landed
        landed = min(max(landed, 0), self.projectile_count)
        multiplier = self.headshot_multiplier if headshot else 1.0
        return self.projectile_damage_at(distance_m) * landed * multiplier

    def shots_to_kill(
        self,
        distance_m: float,
        target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
        pellets_landed: int | None = None,
        headshot: bool = False,
    ) -> float:
        volley = self.volley_damage_at(distance_m, pellets_landed, headshot)
        if volley <= 0.0:
            return math.inf
        return math.ceil(max(target_health, 0.0) / volley)

    def ttk(
        self,
        distance_m: float,
        target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
        pellets_landed: int | None = None,
        headshot: bool = False,
    ) -> float:
        """Ideal TTK: the first shot lands at t=0, so N shots cost N-1 cycles."""
        shots = self.shots_to_kill(distance_m, target_health, pellets_landed, headshot)
        if not math.isfinite(shots):
            return math.inf
        return max(0.0, shots - 1) * self.cooldown_time

    def sustained_ttk(
        self,
        distance_m: float,
        target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
        pellets_landed: int | None = None,
        headshot: bool = False,
    ) -> float:
        """TTK including any reload the magazine forces mid-kill."""
        shots = self.shots_to_kill(distance_m, target_health, pellets_landed, headshot)
        if not math.isfinite(shots):
            return math.inf
        cycles = max(0, int(shots) - 1)
        reloads = cycles // self.magazine_size if self.magazine_size > 0 else 0
        return cycles * self.cooldown_time + reloads * self.reload_time

    def dps(self, distance_m: float) -> float:
        if self.cooldown_time <= 0.0:
            return math.inf
        return self.volley_damage_at(distance_m) / self.cooldown_time

    # -- handling ----------------------------------------------------------

    def recoil_kick(
        self, index: int, constants: dict[str, float] | None = None
    ) -> tuple[float, float]:
        """Mirror of ``WeaponState.recoil_kick`` -> ``(pitch_deg, yaw_deg)``."""
        consts = constants or load_handling_constants()
        pattern_length = max(1.0, consts["RECOIL_PATTERN_LENGTH"])
        sustain = consts["RECOIL_SUSTAIN_SCALE"]
        golden = consts["HANDLING_GOLDEN_ANGLE"]
        n = max(0, index)
        progress = min(max(n / pattern_length, 0.0), 1.0)
        vertical = self.recoil_vertical_deg * (1.0 + (sustain - 1.0) * progress)
        horizontal = self.recoil_horizontal_deg * math.sin(n * golden)
        return vertical, horizontal

    def spread_after(
        self, shots: int, speed_fraction: float = 0.0, airborne: bool = False
    ) -> float:
        """Cone half-angle (degrees) after ``shots`` uninterrupted shots."""
        bloom = min(self.spread_max_deg, shots * self.spread_per_shot_deg)
        total = bloom + self.spread_move_deg * min(max(speed_fraction, 0.0), 1.0)
        if airborne:
            total += self.spread_air_deg
        ceiling = max(self.spread_max_deg, self.spread_air_deg + self.spread_move_deg)
        return min(max(total, 0.0), ceiling)

    def spread_radius_m(
        self, distance_m: float, shots: int = 0, speed_fraction: float = 0.0
    ) -> float:
        """How far off-axis the cone can throw a round at ``distance_m``."""
        return math.tan(math.radians(self.spread_after(shots, speed_fraction))) * distance_m

    def hit_probability_for_cone(self, distance_m: float, cone_deg: float) -> float:
        """Chance a perfectly-aimed round still lands, given a cone angle.

        The engine places rounds deterministically on a sunflower pattern
        whose radius is ``spread * sqrt(fract(n * alpha))``, which fills the
        cone disc uniformly. The fraction of a disc of radius ``R`` covered
        by a target circle of radius ``hit_radius`` is therefore
        ``(hit_radius / R) ** 2``. This assumes a stationary target and an
        aim ray through its centre, so it is an upper bound on accuracy
        rather than a prediction of what a policy will achieve.
        """
        cone = math.tan(math.radians(max(0.0, cone_deg))) * max(0.0, distance_m)
        if cone <= self.hit_radius:
            return 1.0
        return min(1.0, (self.hit_radius / cone) ** 2)

    def hit_probability(
        self, distance_m: float, shots: int = 0, speed_fraction: float = 0.0
    ) -> float:
        """``hit_probability_for_cone`` after ``shots`` uninterrupted shots."""
        return self.hit_probability_for_cone(distance_m, self.spread_after(shots, speed_fraction))

    def _bloom_decay_per_cycle(self, constants: dict[str, float] | None = None) -> float:
        """Bloom recovered in one firing cycle.

        ``tick_handling`` refuses to recover for the first
        ``RECOIL_RECOVERY_DELAY`` seconds after a shot, so only the tail of
        the cooldown window counts.
        """
        consts = constants or load_handling_constants()
        delay = consts.get("RECOIL_RECOVERY_DELAY", 0.0)
        window = max(0.0, self.cooldown_time - delay)
        return self.spread_recovery_deg_per_s * window

    def effective_ttk(
        self,
        distance_m: float,
        target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
        speed_fraction: float = 0.0,
        max_shots: int = 400,
        constants: dict[str, float] | None = None,
    ) -> float:
        """TTK once bloom, magazine and reload are taken into account.

        Walks the trigger shot by shot, growing and decaying the cone the
        way ``WeaponState`` does, crediting each shot its expected damage
        (hit probability times falloff-scaled damage) and paying the reload
        whenever the magazine runs dry. This is the number that actually
        separates the roles: ideal TTK claims the SMG wins every fight,
        effective TTK shows what holding the trigger costs at range.

        Expected damage is used rather than a sampled trajectory because
        the result has to be deterministic and cheap enough for a test.
        """
        if distance_m > self.range_m or target_health <= 0.0:
            return math.inf
        volley = self.projectile_damage_at(distance_m) * self.projectile_count
        if volley <= 0.0:
            return math.inf
        consts = constants or load_handling_constants()
        decay = self._bloom_decay_per_cycle(consts)
        ceiling = max(self.spread_max_deg, self.spread_air_deg + self.spread_move_deg)
        move_cone = self.spread_move_deg * min(max(speed_fraction, 0.0), 1.0)
        remaining = target_health
        elapsed = 0.0
        bloom = 0.0
        ammo = self.magazine_size
        for _ in range(max_shots):
            if self.magazine_size > 0 and ammo <= 0:
                elapsed += self.reload_time
                ammo = self.magazine_size
                bloom = max(0.0, bloom - self.spread_recovery_deg_per_s * self.reload_time)
            cone = min(ceiling, bloom + move_cone)
            remaining -= volley * self.hit_probability_for_cone(distance_m, cone)
            ammo -= 1
            if remaining <= 0.0:
                return elapsed
            bloom = min(self.spread_max_deg, bloom + self.spread_per_shot_deg)
            bloom = max(0.0, bloom - decay)
            elapsed += self.cooldown_time
        return math.inf

    def burst_ttk(
        self,
        distance_m: float,
        burst: int = 3,
        pause: float = 0.4,
        target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
        constants: dict[str, float] | None = None,
        max_shots: int = 400,
    ) -> float:
        """Effective TTK when firing ``burst`` rounds then pausing ``pause`` s.

        Burst discipline is the behaviour the reference game's own guides
        recommend (a short reset pause lets the cone collapse), so the
        environment should reward it: this is the measurement that proves
        bursting beats spraying at range instead of merely asserting it.
        """
        if distance_m > self.range_m or target_health <= 0.0:
            return math.inf
        volley = self.projectile_damage_at(distance_m) * self.projectile_count
        if volley <= 0.0:
            return math.inf
        consts = constants or load_handling_constants()
        decay = self._bloom_decay_per_cycle(consts)
        pause_decay = self.spread_recovery_deg_per_s * max(
            0.0, pause - consts.get("RECOIL_RECOVERY_DELAY", 0.0)
        )
        remaining = target_health
        elapsed = 0.0
        bloom = 0.0
        ammo = self.magazine_size
        fired_in_burst = 0
        for _ in range(max_shots):
            if self.magazine_size > 0 and ammo <= 0:
                elapsed += self.reload_time
                ammo = self.magazine_size
                bloom = max(0.0, bloom - self.spread_recovery_deg_per_s * self.reload_time)
                fired_in_burst = 0
            remaining -= volley * self.hit_probability_for_cone(distance_m, bloom)
            ammo -= 1
            fired_in_burst += 1
            if remaining <= 0.0:
                return elapsed
            bloom = min(self.spread_max_deg, bloom + self.spread_per_shot_deg)
            elapsed += self.cooldown_time
            if burst > 0 and fired_in_burst >= burst:
                elapsed += pause
                bloom = max(0.0, bloom - pause_decay)
                fired_in_burst = 0
            else:
                bloom = max(0.0, bloom - decay)
        return math.inf

    def best_band(self, peers: Iterable[WeaponProfile] = ()) -> str:
        """The engagement band this profile *owns*.

        Absolute TTK always improves as the target gets closer, so "lowest
        TTK" would label every weapon a point-blank weapon and say nothing.
        The useful question is comparative: at which band is this profile
        the strongest option relative to the rest of the roster? Ties are
        broken toward the longer band, because reach is what distinguishes
        two weapons that kill equally fast.
        """
        others = [p for p in peers if p.profile_id != self.profile_id]
        best_band, best_score = RANGE_BANDS[0][0], -math.inf
        for name, low, high in RANGE_BANDS:
            probe = (low + high) / 2.0
            mine = math.inf if probe > self.range_m else self.effective_ttk(probe)
            if not math.isfinite(mine):
                continue
            if not others:
                score = -mine
            else:
                rivals = [math.inf if probe > p.range_m else p.effective_ttk(probe) for p in others]
                finite = [r for r in rivals if math.isfinite(r)]
                # Advantage over the best rival that can reach this band.
                # A zero TTK is a one-shot kill, i.e. unbeatable here.
                if not finite:
                    score = math.inf
                elif mine <= 0.0:
                    score = math.inf if min(finite) > 0.0 else 1.0
                else:
                    score = min(finite) / mine
            if score >= best_score:
                best_band, best_score = name, score
        return best_band

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weapon_source(root: Path | None = None) -> tuple[str, str]:
    base = Path(root) if root is not None else project_root()
    weapon = (base / "scripts/weapon/weapon_state.gd").read_text(encoding="utf-8")
    config = (base / "scripts/core/sandbox_config.gd").read_text(encoding="utf-8")
    return weapon, config


## Parsing the GDScript costs a couple of milliseconds, which is nothing
## once but ruinous when `effective_ttk` is called in a loop. The tables
## are keyed by resolved root so a test pointing at a fixture project does
## not poison the cache for the real one. Callers get copies, so the cache
## cannot be mutated from outside.
_CONSTANTS_CACHE: dict[Path, dict[str, float]] = {}
_PROFILE_CACHE: dict[Path, dict[str, WeaponProfile]] = {}


def clear_cache() -> None:
    """Drops the parsed-table caches. For tests that edit the GDScript."""
    _CONSTANTS_CACHE.clear()
    _PROFILE_CACHE.clear()


def load_handling_constants(root: Path | None = None) -> dict[str, float]:
    """Handling-relevant numeric constants from ``SandboxConfig``."""
    key = Path(root) if root is not None else project_root()
    cached = _CONSTANTS_CACHE.get(key)
    if cached is None:
        _, config = _weapon_source(key)
        cached = _parse_scalar_constants(config)
        _CONSTANTS_CACHE[key] = cached
    return dict(cached)


def load_profiles(root: Path | None = None) -> dict[str, WeaponProfile]:
    """Every weapon profile declared by the Godot ``WeaponState``.

    Raises ``ValueError`` when the GDScript table cannot be parsed at all,
    which is the honest outcome: silently returning an empty/partial table
    would turn a balance regression test into a no-op.
    """
    key = Path(root) if root is not None else project_root()
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    weapon_src, config_src = _weapon_source(key)
    constants = _parse_scalar_constants(config_src)
    block = _PROFILE_BLOCK_RE.search(weapon_src)
    if block is None:
        raise ValueError("WeaponState.PROFILE_DEFINITIONS could not be located")
    profiles: dict[str, WeaponProfile] = {}
    for entry in _ENTRY_RE.finditer(block.group(1)):
        fields: dict[str, Any] = {}
        for field_match in _FIELD_RE.finditer(entry.group("body")):
            value = _coerce(field_match.group("value"), constants)
            if value is not None:
                fields[field_match.group("key")] = value
        profile_id = entry.group("id")
        profiles[profile_id] = WeaponProfile(
            profile_id=profile_id,
            label=str(fields.get("label", profile_id)),
            category=str(fields.get("category", "primary")),
            damage=float(fields["damage"]),
            range_m=float(fields["range_m"]),
            cooldown_time=float(fields["cooldown_time"]),
            hit_radius=float(fields["hit_radius"]),
            projectile_count=int(fields.get("projectile_count", 1)),
            spread_deg=float(fields.get("spread_deg", 0.0)),
            falloff_start_m=float(fields.get("falloff_start_m", fields["range_m"])),
            minimum_damage_scale=float(fields.get("minimum_damage_scale", 1.0)),
            fire_mode=str(fields.get("fire_mode", "auto")),
            magazine_size=int(fields.get("magazine_size", 30)),
            reload_time=float(fields.get("reload_time", 2.0)),
            recoil_vertical_deg=float(fields.get("recoil_vertical_deg", 0.0)),
            recoil_horizontal_deg=float(fields.get("recoil_horizontal_deg", 0.0)),
            recoil_recovery_deg_per_s=float(fields.get("recoil_recovery_deg_per_s", 18.0)),
            spread_per_shot_deg=float(fields.get("spread_per_shot_deg", 0.0)),
            spread_move_deg=float(fields.get("spread_move_deg", 0.0)),
            spread_air_deg=float(fields.get("spread_air_deg", 0.0)),
            spread_max_deg=float(fields.get("spread_max_deg", 0.0)),
            spread_recovery_deg_per_s=float(fields.get("spread_recovery_deg_per_s", 4.0)),
            headshot_multiplier=float(fields.get("headshot_multiplier", 1.0)),
            move_speed_scale_firing=float(fields.get("move_speed_scale_firing", 1.0)),
            description=str(fields.get("description", "")),
        )
    if not profiles:
        raise ValueError("no weapon profiles parsed from WeaponState")
    _PROFILE_CACHE[key] = profiles
    return dict(profiles)


def ttk_table(
    distances: Iterable[float] = (2.0, 5.0, 8.0, 11.0, 14.0),
    profiles: dict[str, WeaponProfile] | None = None,
    target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
) -> dict[str, Any]:
    """TTK / shots-to-kill / DPS per profile at a set of distances."""
    table = profiles if profiles is not None else load_profiles()
    distances = [float(d) for d in distances]
    rows: dict[str, Any] = {}
    for profile_id, profile in table.items():
        rows[profile_id] = {
            "label": profile.label,
            "category": profile.category,
            "fire_mode": profile.fire_mode,
            "rpm": profile.rounds_per_minute,
            "magazine_size": profile.magazine_size,
            "reload_time": profile.reload_time,
            "headshot_multiplier": profile.headshot_multiplier,
            "range_m": profile.range_m,
            "best_band": profile.best_band(table.values()),
            "by_distance": {
                f"{d:g}m": {
                    "in_range": d <= profile.range_m,
                    "damage": profile.volley_damage_at(d),
                    "shots_to_kill": profile.shots_to_kill(d, target_health),
                    "ttk": profile.ttk(d, target_health),
                    "effective_ttk": profile.effective_ttk(d, target_health),
                    "moving_ttk": profile.effective_ttk(d, target_health, speed_fraction=1.0),
                    "sustained_ttk": profile.sustained_ttk(d, target_health),
                    "headshot_ttk": profile.ttk(d, target_health, headshot=True),
                    "dps": profile.dps(d),
                }
                for d in distances
            },
        }
    return {"target_health": target_health, "distances": distances, "profiles": rows}


def role_ranking(
    profiles: dict[str, WeaponProfile] | None = None,
    target_health: float = _ENEMY_MAX_HEALTH_DEFAULT,
) -> dict[str, list[tuple[str, float]]]:
    """Per range band, the profiles sorted by TTK (best first).

    Out-of-range profiles are reported with ``inf`` rather than dropped, so
    "the shotgun cannot reach that band at all" is visible instead of being
    silently missing from the table.
    """
    table = profiles if profiles is not None else load_profiles()
    ranking: dict[str, list[tuple[str, float]]] = {}
    for name, low, high in RANGE_BANDS:
        probe = (low + high) / 2.0
        scores: list[tuple[str, float]] = []
        for profile_id, profile in table.items():
            value = math.inf if probe > profile.range_m else profile.ttk(probe, target_health)
            scores.append((profile_id, value))
        scores.sort(key=lambda item: (item[1], item[0]))
        ranking[name] = scores
    return ranking


def format_ttk_table(table: dict[str, Any] | None = None) -> str:
    """Human-readable TTK matrix for the CLI and for run reports."""
    data = table if table is not None else ttk_table()
    distances = data["distances"]
    lines = [
        f"Weapon TTK matrix (target health {data['target_health']:.0f} HP)",
        "",
        "  ideal  = every shot lands on centre mass, no bloom",
        "  actual = bloom, magazine and reload included, target stationary",
        "",
    ]
    header = f"{'profile':<9}{'mode':<6}{'rpm':>6}{'mag':>5}{'range':>7}{'role':>13}  " + "".join(
        f"{d:g}m".rjust(15) for d in distances
    )
    lines.append(header)
    lines.append("-" * len(header))
    for profile_id, row in data["profiles"].items():
        cells = []
        for d in distances:
            cell = row["by_distance"][f"{d:g}m"]
            if not cell["in_range"]:
                cells.append("--".rjust(15))
            else:
                cells.append(f"{cell['ttk']:.2f}/{cell['effective_ttk']:.2f}s".rjust(15))
        lines.append(
            f"{profile_id:<9}{row['fire_mode']:<6}{row['rpm']:>6.0f}"
            f"{row['magazine_size']:>5}{row['range_m']:>6.1f}m{row['best_band']:>13}  "
            + "".join(cells)
        )
    lines.append("")
    lines.append("Cells are ideal/actual TTK. '--' is beyond the profile's maximum range.")
    return "\n".join(lines)
