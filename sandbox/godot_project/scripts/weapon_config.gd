extends Resource
class_name SandboxWeapon

@export var weapon_id: String = "pistol"
@export var damage: float = 30.0
@export var fire_interval: float = 0.25
@export var magazine_size: int = 12
@export var reserve_ammo: int = 48
@export var reload_seconds: float = 1.1
@export var recoil_degrees: float = 0.7
@export var spread_degrees: float = 1.2
@export var ads_fov: float = 58.0
@export var movement_multiplier: float = 1.0
@export var effective_range: float = 35.0
@export var pellets: int = 1

static func catalog() -> Dictionary:
	return {
		"pistol": _make("pistol", 30.0, 0.24, 12, 48, 1.1, 0.65, 1.0, 58.0, 1.0, 35.0, 1),
		"smg": _make("smg", 22.0, 0.075, 30, 120, 1.6, 0.45, 2.2, 62.0, 0.95, 24.0, 1),
		"rifle": _make("rifle", 42.0, 0.105, 30, 90, 1.8, 0.8, 0.75, 55.0, 0.88, 65.0, 1),
		"shotgun": _make("shotgun", 18.0, 0.85, 8, 40, 2.4, 1.1, 5.0, 62.0, 0.82, 18.0, 8),
		"marksman": _make("marksman", 78.0, 0.42, 10, 40, 2.0, 1.0, 0.25, 42.0, 0.78, 100.0, 1),
	}

static func _make(id: String, dmg: float, interval: float, mag: int, reserve: int, reload: float, recoil: float, spread: float, fov: float, movement: float, range: float, shot_count: int) -> SandboxWeapon:
	var weapon: SandboxWeapon = SandboxWeapon.new()
	weapon.weapon_id = id; weapon.damage = dmg; weapon.fire_interval = interval; weapon.magazine_size = mag; weapon.reserve_ammo = reserve; weapon.reload_seconds = reload; weapon.recoil_degrees = recoil; weapon.spread_degrees = spread; weapon.ads_fov = fov; weapon.movement_multiplier = movement; weapon.effective_range = range; weapon.pellets = shot_count
	return weapon
