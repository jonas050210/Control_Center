"""Weapon definitions and deterministic ammunition/reload mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    damage: float
    fire_rate: float
    mag_size: int
    reload_time: float
    spread: float
    range: float
    recoil: float
    projectile_speed: float
    headshot_multiplier: float = 2.5
    pellets: int = 1

    @property
    def pellet_damage(self) -> float:
        return self.damage / max(1, self.pellets)


class Weapon:
    """Immutable weapon tuning plus factories for per-agent runtime state."""

    spec: ClassVar[WeaponSpec]

    @property
    def name(self) -> str:
        return self.spec.name

    def create_runtime(self) -> "WeaponRuntime":
        return WeaponRuntime(self)

    def damage_at(self, distance: float, headshot: bool = False) -> float:
        """Apply a smooth distance falloff and optional upper-hitbox bonus."""
        if distance < 0 or distance > self.spec.range:
            return 0.0
        ratio = distance / max(self.spec.range, 1e-6)
        falloff = max(0.18, 1.0 - 0.72 * ratio ** 1.35)
        damage = self.spec.pellet_damage * falloff
        if headshot:
            damage *= self.spec.headshot_multiplier
        return damage

    def spread_radians(self, recoil_level: float = 0.0, moving: bool = False) -> float:
        """Return the one-sigma angular spread used by the ray-cast shot model."""
        movement_penalty = 1.35 if moving else 1.0
        degrees = self.spec.spread * movement_penalty + max(0.0, recoil_level) * 0.35
        return math.radians(degrees)


class Pistol(Weapon):
    spec = WeaponSpec("Pistol", damage=25, fire_rate=0.30, mag_size=12,
                      reload_time=1.5, spread=1.6, range=35, recoil=0.35,
                      projectile_speed=400)


class SMG(Weapon):
    spec = WeaponSpec("SMG", damage=15, fire_rate=0.08, mag_size=30,
                      reload_time=2.0, spread=5.0, range=22, recoil=0.50,
                      projectile_speed=380)


class AK47(Weapon):
    spec = WeaponSpec("AK-47", damage=28, fire_rate=0.10, mag_size=30,
                      reload_time=2.5, spread=3.0, range=55, recoil=0.62,
                      projectile_speed=500)


class Shotgun(Weapon):
    spec = WeaponSpec("Shotgun", damage=80, fire_rate=0.90, mag_size=6,
                      reload_time=3.0, spread=12.0, range=16, recoil=1.15,
                      projectile_speed=320, pellets=10)


class Sniper(Weapon):
    spec = WeaponSpec("Sniper", damage=95, fire_rate=1.50, mag_size=5,
                      reload_time=3.5, spread=0.08, range=100, recoil=1.35,
                      projectile_speed=850)


class LMG(Weapon):
    spec = WeaponSpec("LMG", damage=20, fire_rate=0.07, mag_size=100,
                      reload_time=5.0, spread=3.5, range=60, recoil=0.56,
                      projectile_speed=480)


WEAPON_CLASSES: dict[str, type[Weapon]] = {
    "Pistol": Pistol,
    "SMG": SMG,
    "AK-47": AK47,
    "Shotgun": Shotgun,
    "Sniper": Sniper,
    "LMG": LMG,
}
WEAPON_NAMES = tuple(WEAPON_CLASSES)


def get_weapon(name: str) -> Weapon:
    """Create a fresh weapon definition by its display name."""
    canonical = next((item for item in WEAPON_NAMES if item.lower() == name.lower()), None)
    if canonical is None:
        raise ValueError(f"Unknown weapon {name!r}. Choose one of: {', '.join(WEAPON_NAMES)}")
    return WEAPON_CLASSES[canonical]()


def all_weapon_specs() -> dict[str, WeaponSpec]:
    return {name: cls.spec for name, cls in WEAPON_CLASSES.items()}


@dataclass
class WeaponRuntime:
    """Mutable clip, cooldown, reload, recoil and timing state for one weapon."""

    weapon: Weapon
    ammo: int = 0
    cooldown: float = 0.0
    reload_remaining: float = 0.0
    recoil_level: float = 0.0
    time_since_last_shot: float = 10.0
    time_since_reload: float = 10.0

    def __post_init__(self) -> None:
        if self.ammo <= 0:
            self.ammo = self.weapon.spec.mag_size

    @property
    def spec(self) -> WeaponSpec:
        return self.weapon.spec

    @property
    def is_reloading(self) -> bool:
        return self.reload_remaining > 0.0

    def tick(self, dt: float, firing: bool = False) -> None:
        """Advance timers and gently recover recoil between shots."""
        dt = max(0.0, float(dt))
        self.cooldown = max(0.0, self.cooldown - dt)
        self.time_since_last_shot += dt
        self.time_since_reload += dt
        if self.reload_remaining > 0.0:
            self.reload_remaining = max(0.0, self.reload_remaining - dt)
            if self.reload_remaining == 0.0:
                self.ammo = self.spec.mag_size
        if not firing:
            self.recoil_level = 0.0

    def begin_reload(self) -> bool:
        """Start a reload if the magazine is not full and no reload is active."""
        if self.is_reloading or self.ammo >= self.spec.mag_size:
            return False
        self.reload_remaining = self.spec.reload_time
        self.time_since_reload = 0.0
        return True

    def try_fire(self) -> tuple[bool, str]:
        """Consume one round when ready; return (fired, reason)."""
        if self.is_reloading:
            return False, "reloading"
        if self.cooldown > 0.0:
            return False, "cooldown"
        if self.ammo <= 0:
            return False, "empty"
        self.ammo -= 1
        self.cooldown = self.spec.fire_rate
        self.recoil_level = min(5.0, self.recoil_level + self.spec.recoil)
        self.time_since_last_shot = 0.0
        return True, "fired"

    def reset(self) -> None:
        self.ammo = self.spec.mag_size
        self.cooldown = 0.0
        self.reload_remaining = 0.0
        self.recoil_level = 0.0
        self.time_since_last_shot = 10.0
        self.time_since_reload = 10.0
