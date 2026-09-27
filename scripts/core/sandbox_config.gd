## SandboxConfig
##
## Single centralized source of truth for every tunable numeric constant used
## across the SandboxAI simulator (arena size, agent/enemy stats, weapon
## stats, reward shaping, episode limits, simulation defaults).
##
## Design decision: this is a plain class with `const` values rather than an
## autoload singleton. GDScript constants declared on a globally-named class
## are reachable anywhere as `SandboxConfig.SOME_CONST` without requiring the
## class to be registered as an autoload or instanced first. That keeps
## project.godot free of autoload wiring and makes the constants trivially
## reachable from standalone test scripts that only `load()` individual
## files instead of running the full scene tree.
class_name SandboxConfig
extends RefCounted

## Future observation modes (vision-support hook; RGB is intentionally not
## implemented in milestone 1, see docs/ARCHITECTURE.md).
enum ObservationMode { STRUCTURED = 0, HUMAN_INPUT = 1, RGB = 2 }

# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
## Half-width/half-depth of the square arena floor, in meters.
const ARENA_HALF_EXTENT: float = 10.0
## Interior wall height, in meters.
const ARENA_WALL_HEIGHT: float = 3.0
## Maximum straight-line distance possible inside the arena (used to
## normalize distance-based observations).
const ARENA_MAX_DISTANCE: float = ARENA_HALF_EXTENT * 2.0 * 1.4142136

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
const AGENT_MAX_HEALTH: float = 100.0
const AGENT_MOVE_SPEED: float = 4.5  # meters / second
const AGENT_TURN_SPEED_DEG: float = 110.0  # degrees / second (yaw & pitch)
const AGENT_PITCH_LIMIT_DEG: float = 80.0  # +/- clamp from horizontal
const AGENT_EYE_HEIGHT: float = 1.6
const AGENT_RADIUS: float = 0.4
const AGENT_SPAWN_POSITION: Vector3 = Vector3(0.0, 0.0, 6.0)
## yaw=0 corresponds to forward=(0,0,-1); the enemy spawns at negative Z, so
## yaw=0 already faces the agent directly at the enemy on episode start.
const AGENT_SPAWN_YAW_DEG: float = 0.0

# ---------------------------------------------------------------------------
# Enemy
# ---------------------------------------------------------------------------
const ENEMY_MAX_HEALTH: float = 100.0
const ENEMY_MOVE_SPEED: float = 2.25  # meters / second, deliberately slower than the agent
const ENEMY_RADIUS: float = 0.45
const ENEMY_CHEST_HEIGHT: float = 1.2  # hit-sphere center offset above feet
const ENEMY_DETECTION_RANGE: float = ARENA_MAX_DISTANCE  # always aware for milestone 1
const ENEMY_ATTACK_RANGE: float = 2.0
const ENEMY_ATTACK_DAMAGE: float = 10.0
const ENEMY_ATTACK_COOLDOWN: float = 1.0  # seconds between enemy attacks
const ENEMY_SPAWN_POSITION: Vector3 = Vector3(0.0, 0.0, -6.0)
const ENEMY_COUNT_DEFAULT: int = 1

# ---------------------------------------------------------------------------
# Weapon
# ---------------------------------------------------------------------------
const WEAPON_DAMAGE: float = 25.0
const WEAPON_RANGE: float = 15.0
const WEAPON_FIRE_COOLDOWN: float = 0.5  # seconds between shots (2 shots/sec)
## Radius of the simplified spherical hitbox used by the ray test. Larger
## than the visual capsule radius on purpose: it stands in for a full-body
## hitbox (head to hip) rather than just the model's collision radius, so a
## roughly level shot at a standing target actually connects.
const WEAPON_HIT_RADIUS: float = 0.8

# ---------------------------------------------------------------------------
# Episode / simulation
# ---------------------------------------------------------------------------
const MAX_EPISODE_STEPS: int = 1200  # timeout safeguard (e.g. 20s @ 60 steps/s)
const SIMULATION_TICK_HZ: float = 60.0
const SIMULATION_DT: float = 1.0 / SIMULATION_TICK_HZ
const END_EPISODE_ON_AGENT_DEATH: bool = true
const END_EPISODE_ON_ALL_ENEMIES_DEAD: bool = true
const DEFAULT_ENVIRONMENT_COUNT: int = 4
const DEFAULT_RANDOM_SEED: int = 1234

# ---------------------------------------------------------------------------
# Reward shaping (centralized & intentionally modest so the task is not
# trivialized by shaping rewards).
# ---------------------------------------------------------------------------
const REWARD_HIT: float = 1.0
const REWARD_KILL: float = 10.0
const REWARD_SURVIVE_TICK: float = 0.01
const REWARD_POSITIONING_SCALE: float = 0.05  # per meter closed toward the enemy, clamped
const REWARD_POSITIONING_MAX: float = 0.05
const PENALTY_DAMAGE_TAKEN_PER_HP: float = -0.05
const PENALTY_DEATH: float = -10.0
const PENALTY_USELESS_SHOT: float = -0.1  # shot fired that could not possibly hit / on cooldown

# ---------------------------------------------------------------------------
# Observation mode currently active (see ObservationMode enum above).
# ---------------------------------------------------------------------------
const ACTIVE_OBSERVATION_MODE: int = ObservationMode.STRUCTURED
