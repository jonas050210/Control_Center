## ReactionProfile
##
## Configurable reaction latency for one perceiving character. Real players
## cannot acquire, confirm, aim at and shoot a target in the same frame it
## becomes visible; an enemy that does is not "hard", it is impossible, and
## a policy trained against it learns nonsense.
##
## The profile splits latency into the components that behave differently:
##
##   visual_detection_delay -- continuous FOV+LOS time before "I see it"
##   sound_detection_delay  -- delay before a heard event registers
##   target_confirm_delay   -- extra time before a detected contact is
##                             accepted as the active target
##   aim_reaction_delay     -- time before the character starts turning
##   shoot_reaction_delay   -- time between being on target and pulling
##   aim_speed_deg          -- how fast it can actually turn (deg/s)
##   aim_error_deg          -- residual aim error, i.e. hit probability
##
## Profiles are archetypes, not difficulty multipliers: ROOKIE is slow and
## sloppy, ELITE is fast but still non-zero. INSTANT exists only for tests
## and for reproducing the pre-Phase-7 behavior exactly.
class_name ReactionProfile
extends RefCounted

enum Archetype { INSTANT = 0, ROOKIE = 1, REGULAR = 2, VETERAN = 3, ELITE = 4 }

const SELF_PATH: String = "res://scripts/perception/reaction_profile.gd"

var archetype: int = Archetype.REGULAR
var visual_detection_delay: float = 0.22
var sound_detection_delay: float = 0.12
var target_confirm_delay: float = 0.15
var aim_reaction_delay: float = 0.18
var shoot_reaction_delay: float = 0.12
var aim_speed_deg: float = 180.0
var aim_error_deg: float = 4.0


static func create(p_archetype: int = Archetype.REGULAR) -> ReactionProfile:
	var profile: ReactionProfile = (load(SELF_PATH) as GDScript).new() as ReactionProfile
	profile.apply_archetype(p_archetype)
	return profile


func apply_archetype(value: int) -> void:
	archetype = clampi(value, Archetype.INSTANT, Archetype.ELITE)
	match archetype:
		Archetype.INSTANT:
			visual_detection_delay = 0.0
			sound_detection_delay = 0.0
			target_confirm_delay = 0.0
			aim_reaction_delay = 0.0
			shoot_reaction_delay = 0.0
			aim_speed_deg = 720.0
			aim_error_deg = 0.0
		Archetype.ROOKIE:
			visual_detection_delay = 0.40
			sound_detection_delay = 0.25
			target_confirm_delay = 0.30
			aim_reaction_delay = 0.35
			shoot_reaction_delay = 0.25
			aim_speed_deg = 110.0
			aim_error_deg = 9.0
		Archetype.REGULAR:
			visual_detection_delay = 0.22
			sound_detection_delay = 0.12
			target_confirm_delay = 0.15
			aim_reaction_delay = 0.18
			shoot_reaction_delay = 0.12
			aim_speed_deg = 180.0
			aim_error_deg = 4.0
		Archetype.VETERAN:
			visual_detection_delay = 0.15
			sound_detection_delay = 0.08
			target_confirm_delay = 0.09
			aim_reaction_delay = 0.11
			shoot_reaction_delay = 0.08
			aim_speed_deg = 260.0
			aim_error_deg = 2.2
		Archetype.ELITE:
			visual_detection_delay = 0.10
			sound_detection_delay = 0.05
			target_confirm_delay = 0.05
			aim_reaction_delay = 0.07
			shoot_reaction_delay = 0.05
			aim_speed_deg = 340.0
			aim_error_deg = 1.2


## Total latency between a target becoming geometrically visible and the
## first shot leaving the barrel, assuming the character is already aimed.
func total_engagement_latency() -> float:
	return visual_detection_delay + target_confirm_delay + aim_reaction_delay + shoot_reaction_delay


static func archetype_name(value: int) -> String:
	match value:
		Archetype.INSTANT:
			return "instant"
		Archetype.ROOKIE:
			return "rookie"
		Archetype.REGULAR:
			return "regular"
		Archetype.VETERAN:
			return "veteran"
		Archetype.ELITE:
			return "elite"
		_:
			return "unknown"


func to_dict() -> Dictionary:
	return {
		"archetype": archetype,
		"archetype_name": archetype_name(archetype),
		"visual_detection_delay": visual_detection_delay,
		"sound_detection_delay": sound_detection_delay,
		"target_confirm_delay": target_confirm_delay,
		"aim_reaction_delay": aim_reaction_delay,
		"shoot_reaction_delay": shoot_reaction_delay,
		"aim_speed_deg": aim_speed_deg,
		"aim_error_deg": aim_error_deg,
		"total_engagement_latency": total_engagement_latency(),
	}
