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

## Safety margin subtracted from every arena-bounds clamp (meters).
## `half_extent - radius` is computed in 32-bit floats inside a Vector3, so
## a clamped coordinate can round a few 1e-7 OUTSIDE the nominal bound. The
## margin is 1/400 of a character radius — invisible to gameplay, but it
## makes "clamped means inside" exactly true.
const ARENA_BOUNDS_EPSILON: float = 0.001

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
const AGENT_MAX_HEALTH: float = 100.0
const AGENT_MOVE_SPEED: float = 4.5  # meters / second
const AGENT_TURN_SPEED_DEG: float = 110.0  # degrees / second (yaw & pitch)
const AGENT_PITCH_LIMIT_DEG: float = 80.0  # +/- clamp from horizontal
const AGENT_EYE_HEIGHT: float = 1.6
const AGENT_RADIUS: float = 0.4
## Standing collision height (feet -> top of head), used by the world
## collision queries. Slightly above the eye height so a character cannot
## slip under a box its eyes clear.
const AGENT_HEIGHT: float = 1.8
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
# Multi-enemy spawn variety / movement patterns (curriculum-driven).
#
# These constants exist so a curriculum level can place enemies at varied
# distances and horizontal angles around the agent (instead of always
# directly ahead) and can make enemies strafe while approaching/engaging
# instead of walking straight at the agent. All of it stays inside the
# existing flat, analytic arena: no physics, no verticality, no navmesh.
# ---------------------------------------------------------------------------
## Minimum/maximum spawn distance from the agent's spawn point once a
## curriculum level enables spawn-position variety (see
## CurriculumConfig.spawn_variety_enabled()).
const ENEMY_SPAWN_MIN_DISTANCE: float = 4.0
## Maximum sampled spawn distance from the agent's spawn point. NOTE: the
## agent's spawn point is offset from the arena center, so wide spawn angles
## can place the sampled point outside the walls; EnvironmentCore clamps
## those spawns back inside, which shortens the effective distance below
## this value (see _random_spawn_position). The clamp keeps every spawn
## valid; it just means the sampled 4-13 m range is an upper envelope, not
## a guarantee, for angles far off the forward axis.
const ENEMY_SPAWN_MAX_DISTANCE: float = 13.0
## Enemy lateral (strafe) speed as a fraction of its forward move speed.
const ENEMY_STRAFE_SPEED_SCALE: float = 0.6
## Angular frequency (radians/second) driving the deterministic sinusoidal
## strafe oscillation. Higher = faster left/right direction changes.
const ENEMY_STRAFE_ANGULAR_SPEED: float = 1.6
## Number of enemy "slots" the structured observation reports individually
## (nearest-alive-first). Extra enemies beyond this count still exist and
## affect the simulation/reward, they are simply not each individually
## observed — a policy trained on this contract must generalize from the
## nearest few threats, matching what a human/Roblox player could plausibly
## track. Kept small deliberately to keep the observation cheap.
const OBSERVATION_MAX_TRACKED_ENEMIES: int = 3

# ---------------------------------------------------------------------------
# Vertical movement (gravity / jumping)
#
# Analytic, fixed-timestep integration — no PhysicsServer. Numbers are
# chosen so a jump clears a 1.0 m low-cover crate but not a 2.4 m high
# cover wall, which is what makes "jump onto the platform to change the
# sight line" a real tactical decision instead of a free win.
# ---------------------------------------------------------------------------
const GRAVITY: float = -19.0  # meters / second^2
const JUMP_VELOCITY: float = 6.0  # meters / second, apex ~0.95 m
## Fraction of full ground acceleration available while airborne. Keeps
## bunny-hopping from being strictly better than running.
const AIR_CONTROL: float = 0.45
## Vertical speed below which a falling character is considered landed.
const LANDING_EPSILON: float = 0.01
## Maximum vertical extent used to normalize height-related observations.
const MAX_VERTICAL_EXTENT: float = 6.0

# ---------------------------------------------------------------------------
# Navigation
#
# A uniform walkable grid baked from the arena geometry (NavigationGraph).
# It is only consulted when direct steering is blocked, so the open-field
# case still costs nothing. The cell size is the single most important
# knob: smaller means finer paths through narrow doorways but a quadratic
# increase in bake cost, so it is tuned to just under the widest character
# diameter (enemy radius 0.45 -> 0.9 m) plus clearance.
# ---------------------------------------------------------------------------
const NAV_CELL_SIZE: float = 1.1
## Horizontal speed below which a character that IS asking to move counts as
## blocked (meters/second).
const NAV_STUCK_SPEED: float = 0.35
## How long a character must be blocked before navigation takes over.
const NAV_STUCK_TIME: float = 0.3
## Minimum interval between path re-plans for one character (seconds).
const NAV_REPATH_INTERVAL: float = 0.45
## Distance at which a waypoint counts as reached (meters).
const NAV_WAYPOINT_TOLERANCE: float = 0.55
## How long a recovery nudge is committed to once triggered (seconds).
## Committing prevents the character oscillating between "stuck" and
## "free" on alternating ticks.
const NAV_RECOVERY_TIME: float = 0.5
## Hard cap on cached waypoints per character; a longer route is re-planned
## when the tail is consumed.
const NAV_MAX_WAYPOINTS: int = 32

# ---------------------------------------------------------------------------
# Perception: field of view, line of sight, detection latency
# ---------------------------------------------------------------------------
## Horizontal field of view (total cone angle, degrees) used to decide
## whether a target is visually perceivable at all. 100 deg approximates a
## typical FPS horizontal FOV.
const AGENT_FOV_DEG: float = 100.0
const ENEMY_FOV_DEG: float = 110.0
## Maximum distance at which a target can be visually acquired.
const VISION_RANGE: float = 28.0
## How long a target must remain continuously inside FOV with clear line of
## sight before the observer registers it (seconds). Prevents impossible
## zero-latency reactions; set to 0.0 for instantaneous perception.
const AGENT_VISUAL_DETECTION_DELAY: float = 0.12
## How long a visible target keeps being reported after it leaves FOV/LOS.
## Models the fact that losing a target is not instantaneous either.
const VISUAL_LOSS_GRACE: float = 0.10

# ---------------------------------------------------------------------------
# Lighting / visibility conditions (LightingProfile)
#
# Visibility conditions are expressed as multipliers on the EXISTING
# perception pipeline rather than as a new observation flag. A policy is
# meant to notice "I am acquiring targets late and losing them early" and
# fall back on sound and memory, not to read a night bit.
# ---------------------------------------------------------------------------
## Fraction of the nominal vision range that survives total darkness. Never
## zero: a target close enough is still visible with no light at all.
const LIGHTING_MIN_RANGE_SCALE: float = 0.3
## How much the visual detection delay grows in total darkness, as a
## multiple of the nominal delay (1.0 = up to twice as slow).
const LIGHTING_DELAY_GAIN: float = 1.4
## Transmittance below which a sight line counts as lost in fog. Sets the
## fog horizon together with the profile's density.
const LIGHTING_MIN_TRANSMITTANCE: float = 0.25
## Default lighting mode id for maps that do not declare one.
const LIGHTING_DEFAULT_MODE_ID: String = "normal"

# ---------------------------------------------------------------------------
# Exploration / spatial memory (SpatialMemory, MapAnalyzer)
#
# The agent's map knowledge is a coarse grid it fills in by looking at
# things. Cell size is deliberately large: this is a memory of places, not
# a occupancy map, and it has to stay cheap enough to update inside the
# simulation loop.
# ---------------------------------------------------------------------------
const EXPLORATION_CELL_SIZE: float = 2.0
## Half-life (seconds) of per-cell confidence. Far longer than
## MEMORY_HALF_LIFE: terrain does not walk away, but the agent still cannot
## know whether a room it saw two minutes ago is still empty.
const EXPLORATION_HALF_LIFE: float = 45.0
## Confidence floor. A cell the agent has seen never returns to "unknown";
## it becomes "known but unreliable", which is a different thing.
const EXPLORATION_MIN_CONFIDENCE: float = 0.05
## Seconds between visibility sweeps. The sweep is the expensive part, so it
## runs on a fixed cadence driven by the simulation clock (deterministic),
## not every tick.
const EXPLORATION_UPDATE_INTERVAL: float = 0.2
## Age (seconds) used to normalize "how long since I was last here". Beyond
## this the field saturates: the agent only knows it has been a long time.
const EXPLORATION_MAX_RECALL_AGE: float = 60.0
## Cap on remembered route waypoints per episode.
const EXPLORATION_MAX_ROUTE: int = 512
## Map Analyzer mode: coverage fraction at which the map counts as explored
## and the episode ends successfully.
const EXPLORATION_TARGET_COVERAGE: float = 0.85
## Reward per newly observed cell in Map Analyzer mode. Scaled by the grid
## size at runtime so a big map is not worth more total reward than a small
## one.
const REWARD_EXPLORATION_COVERAGE: float = 5.0
## Reward for reaching EXPLORATION_TARGET_COVERAGE.
const REWARD_EXPLORATION_COMPLETE: float = 10.0

# ---------------------------------------------------------------------------
# Sound
#
# Sound is modelled as discrete, decaying events with a base audible radius
# that is attenuated once per sight-blocking box between source and
# listener. A listener receives direction + distance + category + age, never
# the emitter's identity or exact coordinates.
# ---------------------------------------------------------------------------
const SOUND_FOOTSTEP_RADIUS: float = 9.0
const SOUND_JUMP_RADIUS: float = 7.0
const SOUND_LAND_RADIUS: float = 11.0
const SOUND_SHOT_RADIUS: float = 26.0
const SOUND_IMPACT_RADIUS: float = 13.0
const SOUND_DEATH_RADIUS: float = 15.0
## Ambient / environmental noise (doors, debris, machinery). Not produced by
## any character, so it is a genuine distractor: it must NOT be treated as
## evidence of an enemy.
const SOUND_ENVIRONMENT_RADIUS: float = 12.0
## Multiplier applied to the audible radius for every wall between the
## source and the listener.
const SOUND_OCCLUSION_ATTENUATION: float = 0.55
## Seconds a sound event stays in the bus before being discarded.
const SOUND_EVENT_LIFETIME: float = 2.0
## Hard cap on simultaneously tracked sound events per environment. Keeps
## the perception update allocation-free and bounded.
const SOUND_MAX_ACTIVE: int = 24
## Seconds between footstep emissions while moving on the ground.
const FOOTSTEP_INTERVAL: float = 0.38
## Perceived-direction error, in degrees, applied deterministically to a
## heard event. Hearing is directional but not pinpoint.
const SOUND_DIRECTION_ERROR_DEG: float = 14.0
## Delay before a sound event is consciously registered (seconds).
const SOUND_DETECTION_DELAY: float = 0.08
## Memory-track id used for "something I only heard". Hearing never
## identifies WHICH enemy made the noise, so sound-only contacts are keyed
## separately from the per-enemy visual tracks.
const SOUND_UNKNOWN_SOURCE_ID: int = -999
## Masking between simultaneous sounds. A quiet event heard at the same
## time as a loud one loses this fraction of the loud event's perceived
## loudness; below SOUND_MASK_FLOOR it is not registered at all. This is
## what makes a footstep during a gunshot unhearable.
const SOUND_MASKING_STRENGTH: float = 0.7
const SOUND_MASK_FLOOR: float = 0.04
## Extra directional error per occluding wall, as a multiple of
## SOUND_DIRECTION_ERROR_DEG. A muffled sound is harder to place.
const SOUND_OCCLUSION_ERROR_GAIN: float = 0.8
## Extra directional error at the very edge of the audible radius, as a
## multiple of SOUND_DIRECTION_ERROR_DEG.
const SOUND_RANGE_ERROR_GAIN: float = 0.5
## Two heard events whose perceived bearings differ by more than this are
## counted as separate sources by SoundBus.summarize().
const SOUND_DISTINCT_SOURCE_ANGLE_DEG: float = 35.0
## Seconds between ambient emissions on a map that declares ambience.
const SOUND_AMBIENCE_INTERVAL: float = 3.5

# ---------------------------------------------------------------------------
# Target selection (TargetSelector)
#
# Weights on the factors that decide which contact fills the primary
# observation slot. They describe what information is WORTH, not a play
# style: the policy is free to shoot at something else entirely.
# ---------------------------------------------------------------------------
const TARGET_WEIGHT_VISIBLE: float = 100.0
const TARGET_WEIGHT_DAMAGE_SOURCE: float = 40.0
const TARGET_WEIGHT_CONFIDENCE: float = 20.0
const TARGET_WEIGHT_PROXIMITY: float = 10.0
## A contact that currently has line of sight to the agent.
const TARGET_WEIGHT_THREAT: float = 15.0
## Scales with missing health: a nearly dead contact is cheap to finish.
const TARGET_WEIGHT_WOUNDED: float = 8.0
## Hysteresis: a small bonus for keeping the current target, so two equal
## threats do not make the primary slot oscillate every tick.
const TARGET_WEIGHT_CONTINUITY: float = 6.0
## Subtracted when the navigation graph says the contact cannot be reached.
const TARGET_PENALTY_UNREACHABLE: float = 25.0
## Seconds after a target switch during which the switch is still reported
## as "recent" in the observation.
const TARGET_SWITCH_RECENT_WINDOW: float = 1.5

# ---------------------------------------------------------------------------
# Enemy memory
# ---------------------------------------------------------------------------
## Confidence decays exponentially with this half-life (seconds) once
## contact is lost.
const MEMORY_HALF_LIFE: float = 3.0
## A track below this confidence is forgotten entirely.
const MEMORY_FORGET_CONFIDENCE: float = 0.05
## Hard cap on remembered tracks per observer.
const MEMORY_MAX_TRACKS: int = 8
## Age (seconds) used to normalize the memory-age observation fields.
const MEMORY_MAX_AGE: float = 12.0

# ---------------------------------------------------------------------------
# Enemy tactical behavior
# ---------------------------------------------------------------------------
## Ranged enemy fire (enabled from the obstacles/cover curriculum level on).
const ENEMY_FIRE_RANGE: float = 16.0
const ENEMY_FIRE_DAMAGE: float = 9.0
const ENEMY_FIRE_COOLDOWN: float = 0.85
## Radius searched for a cover / peek position around the enemy.
const ENEMY_COVER_SEARCH_RADIUS: float = 4.0
## Distance the enemy tries to hold from the agent while engaging.
const ENEMY_PREFERRED_RANGE: float = 7.0
## Health fraction below which an engaging enemy breaks for cover.
const ENEMY_RETREAT_HEALTH_FRACTION: float = 0.35
## Below this fraction the enemy stops trying to trade at all and breaks
## away from the believed threat, even without cover to run to.
const ENEMY_CRITICAL_HEALTH_FRACTION: float = 0.15
## How far a retreating enemy tries to get from the believed threat.
const ENEMY_RETREAT_DISTANCE: float = 7.0
## Continuous seconds an engaging enemy tolerates standing in the open
## before repositioning to cover.
const ENEMY_MAX_EXPOSURE_TIME: float = 3.5
## Speed multiplier while walking toward a noise it has never seen.
const ENEMY_INVESTIGATE_SPEED_SCALE: float = 0.7
## Seconds an enemy stays behind cover before peeking again.
const ENEMY_COVER_DWELL: float = 1.2
## Seconds an enemy spends searching a lost target's last known position
## before giving up and returning to idle.
const ENEMY_SEARCH_DURATION: float = 7.0
## Radius the enemy wanders around the last known position while searching.
const ENEMY_SEARCH_RADIUS: float = 3.0
## Probability an enemy jumps when it needs to climb onto a standable box
## that is directly in its path.
const ENEMY_JUMP_PROBABILITY: float = 0.35

# ---------------------------------------------------------------------------
# Corpses
# ---------------------------------------------------------------------------
## Dead characters remain in the world as inert corpses for the rest of the
## episode. A corpse is never a target, never perceived as an enemy and
## never emits sound; it exists only as environmental information.
const CORPSES_PERSIST: bool = true

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
# Weapon handling (recoil, bloom, fire modes, magazines, hit zones)
#
# These model the "weighty handling" half of a tactical FPS: a weapon that
# climbs while you hold the trigger, loses precision while you move, has to
# be reloaded, and rewards hitting the head. Every one of them is
# DETERMINISTIC (no RNG anywhere: recoil follows a fixed pattern and bloom
# offsets come from a low-discrepancy sequence keyed on the shot index), so
# seeded replays stay byte-stable.
#
# The whole block is gated by CurriculumConfig.weapon_handling_enabled() /
# hit_zones_enabled(). With handling off, WeaponState behaves exactly like
# the pre-handling cooldown-only weapon, which is what keeps curriculum
# levels 1-4 bit-for-bit reproducible.
# ---------------------------------------------------------------------------
## Seconds after the last shot before accumulated recoil starts recovering.
## This is the "reset pause" a player takes between bursts.
const RECOIL_RECOVERY_DELAY: float = 0.12
## Hard cap on accumulated recoil offset in either axis (degrees). Without
## it a long spray would eventually point the camera at the ceiling.
const RECOIL_MAX_DEG: float = 12.0
## Golden angle (radians). Used to walk the horizontal recoil pattern and
## the bloom azimuth so consecutive shots spread evenly without RNG.
const HANDLING_GOLDEN_ANGLE: float = 2.39996323
## R1 low-discrepancy increment used for the bloom radius.
const HANDLING_R1_ALPHA: float = 0.7548776662
## How many shots a recoil pattern takes to reach its sustained (reduced)
## climb. Early shots kick hardest, exactly like a real muzzle-rise curve.
const RECOIL_PATTERN_LENGTH: int = 8
## Fraction of the initial vertical kick that remains once the pattern is
## fully developed.
const RECOIL_SUSTAIN_SCALE: float = 0.55
## Movement-induced inaccuracy is proportional to horizontal speed. This is
## the fraction of `spread_move_deg` applied at full move speed.
const SPREAD_MOVE_REFERENCE_SPEED: float = AGENT_MOVE_SPEED
## Bloom below this many degrees counts as "settled": the shot is treated
## as pinpoint. First-shot accuracy has to be exact or the early aiming
## curriculum stops being learnable.
const SPREAD_SETTLED_EPSILON: float = 0.01
## Seconds a shot keeps the shooter slowed down ("shooting slows you down").
const FIRE_MOVEMENT_SLOW_DURATION: float = 0.18
## Height above the feet of the head hit-sphere centre.
const ENEMY_HEAD_HEIGHT: float = 1.62
## Head hit-sphere radius as a fraction of the weapon's body hit radius.
const HEAD_HIT_RADIUS_SCALE: float = 0.42

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
## Weapon-normalization reference for the flat per-trigger-pull hit bonus.
##
## Without it the flat bonus is paid once per CONNECTING TRIGGER PULL, so a
## weapon that needs 8 shots to kill earns 8x the flat bonus of a weapon
## that needs 1 — the return of an episode would then depend mostly on
## which tool the scenario handed the agent, which corrupts cross-weapon
## evaluation, curriculum promotion thresholds and league comparisons.
## Scaling the bonus by `weapon_damage / REWARD_HIT_REFERENCE_DAMAGE`
## makes the total flat bonus per kill roughly weapon-independent, and the
## baseline rifle (damage 25 == the reference) is bit-for-bit unchanged.
const REWARD_HIT_REFERENCE_DAMAGE: float = WEAPON_DAMAGE
## Upper bound on the normalized hit bonus so a hypothetical very
## high-damage profile cannot dominate the kill reward.
const REWARD_HIT_SCALE_MAX: float = 6.0
const REWARD_KILL: float = 10.0
## Direct damage shaping keeps the optimizer focused on combat progress
## instead of passive survival. A 25 HP body shot is worth +0.5 before the
## discrete hit bonus, while a kill is still dominated by REWARD_KILL.
const REWARD_DAMAGE_DEALT_PER_HP: float = 0.02
## Kept for backwards-compatible reward breakdowns, but intentionally zero
## in combat. Survival is measured as a metric, not paid as a reward, so an
## agent cannot score by hiding until timeout.
const REWARD_SURVIVE_TICK: float = 0.0
const REWARD_POSITIONING_SCALE: float = 0.05  # per meter closed toward the enemy, clamped
const REWARD_POSITIONING_MAX: float = 0.05
## Small shaping for turning the true weapon ray toward a hittable target.
## It is gated by target_hittable, so aiming at an enemy through a wall or
## oscillating on a remembered/non-shootable contact cannot farm reward.
const REWARD_AIMING_SCALE: float = 0.10
const REWARD_AIMING_MAX: float = 0.02
## Applied while a live target exists and the agent neither moves toward,
## aims toward a hittable target, nor fires. This makes timeout farming
## unprofitable without forcing constant shooting.
const PENALTY_PASSIVITY: float = -0.012
## Small per-tick pressure whenever a live combat target exists. This breaks
## movement/reward cycles whose positive and negative shaping would otherwise
## net near zero over a long timeout.
const PENALTY_COMBAT_TIME: float = -0.002
const PENALTY_DAMAGE_TAKEN_PER_HP: float = -0.05
const PENALTY_DEATH: float = -10.0
## Trigger pulls that could not plausibly connect: weapon cooldown, no alive
## target, target out of range/behind cover, or the aim ray nowhere near any
## live target. Kept much harsher than a genuine near-miss so trigger
## discipline is learned before spray-and-pray.
const PENALTY_USELESS_SHOT: float = -0.1
## A shot actually fired near a live, clear, in-range target that failed to
## connect. Deliberately cheap: while fine aim is still being learned,
## near-misses must not erase the value of +hit/+damage/+kill rewards.
const PENALTY_MISSED_SHOT: float = -0.01
## Angular tolerance used to decide whether a fired shot was a genuine
## near-miss at a live target or just spam. This does NOT affect hit testing;
## hits still use the exact hitscan ray/sphere calculation.
const WEAPON_NEAR_MISS_CONE_DEG: float = 7.0
## Holding the trigger on a weapon that physically cannot fire yet
## (cycling, reloading, or a semi-auto that needs the trigger released).
##
## With weapon handling ON this REPLACES PENALTY_USELESS_SHOT for the
## cooldown case. Holding the trigger on an automatic weapon is correct
## play, not an error, and charging -0.1 per tick for 29 of every 30 ticks
## of a 0.5 s cycle made "never hold the trigger" worth more than the whole
## combat reward — it distorted every weapon role in proportion to its
## fire rate. Spray is instead punished organically: bloom widens, ammo
## drains, and the reload leaves the agent exposed. This residual cost only
## covers the genuinely wasted intent.
const PENALTY_TRIGGER_DISCIPLINE: float = -0.002

# ---------------------------------------------------------------------------
# Observation mode currently active (see ObservationMode enum above).
# ---------------------------------------------------------------------------
const ACTIVE_OBSERVATION_MODE: int = ObservationMode.STRUCTURED
## Vision hooks only: no RGB capture or image-processing dependency is enabled.
const RGB_OBSERVATION_ENABLED: bool = false
const RGB_FRAME_STACK: int = 1
const OBSERVATION_MODALITIES: Array = ["structured"]
