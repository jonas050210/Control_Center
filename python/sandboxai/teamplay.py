"""Teamplay foundation: teams, teammate identity, comms and reward hooks.

Phase 8. This is deliberately a **foundation**, not a 3v3 implementation.
What it provides is the vocabulary and the boundaries that a future team
mode needs, expressed precisely enough to be tested now, so that adding
2v2 later is a matter of wiring an environment to an existing contract
rather than inventing one under deadline.

Three rules the design enforces:

1. **Disabled by default.** ``TeamConfig()`` is ``enabled=False`` with one
   team, which is exactly today's single-agent setup. Nothing in the
   training path changes unless a caller opts in.

2. **Teammate information is still perception.** A teammate does not grant
   omniscience. ``TeammateReport`` carries only what that teammate could
   itself perceive, plus an explicit ``age`` and ``confidence`` — the same
   decay semantics the single-agent memory already uses. Sharing a
   *ground-truth* enemy position between teammates would be the same leak
   as giving one agent the enemy's coordinates, so the report has no field
   for it.

3. **Communication is limited and costly to abuse.** ``CommsChannel`` is a
   small discrete vocabulary with a per-tick budget. An unbounded
   continuous channel is just a shared hidden state, which trains a single
   distributed brain rather than a team of agents — the opposite of what a
   team experiment is for.

Team rewards are *hooks*: ``TeamRewardHooks`` turns per-agent outcomes into
per-agent reward adjustments under an explicit shaping weight. It does not
add itself to any reward; the environment decides whether to call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

## Team id used when teamplay is off. Everything lives on team 0, which is
## bit-for-bit the current single-agent behaviour.
SOLO_TEAM: int = 0

## Discrete communication vocabulary. Small on purpose (see module note).
COMMS_SYMBOLS: tuple[str, ...] = (
    "contact",  # I can see something
    "lost",  # I lost the contact
    "hit",  # I damaged something
    "hurt",  # I am taking damage
    "moving",  # I am repositioning
    "hold",  # I am holding this position
    "help",  # I want support here
    "clear",  # nothing here
)

## Objectives a team run can be scored against. "eliminate" is today's
## behaviour; the others exist so an environment can declare its intent
## without a new enum every time.
TEAM_OBJECTIVES: tuple[str, ...] = ("eliminate", "survive", "control", "explore")


class TeamError(RuntimeError):
    """A team setup is inconsistent or would leak across teams."""


@dataclass
class TeamConfig:
    """How agent slots map to teams.

    ``slot_teams[i]`` is the team of agent slot ``i``. With teamplay
    disabled the list is ignored and every slot is on ``SOLO_TEAM``.
    """

    enabled: bool = False
    slot_teams: list[int] = field(default_factory=lambda: [SOLO_TEAM])
    friendly_fire: bool = False
    objective: str = "eliminate"
    ## Whether teammates may exchange CommsChannel messages at all.
    comms_enabled: bool = False
    ## Maximum messages one agent may send per tick.
    comms_budget: int = 1
    ## Weight applied to team-level reward shaping. 0.0 = pure individual
    ## reward, which is the default so enabling teams does not silently
    ## change the optimization target.
    team_reward_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.slot_teams:
            raise ValueError("slot_teams must not be empty")
        if any(team < 0 for team in self.slot_teams):
            raise ValueError("team ids must be non-negative")
        if self.objective not in TEAM_OBJECTIVES:
            raise ValueError(f"unknown team objective: {self.objective!r}")
        if self.comms_budget < 0:
            raise ValueError("comms_budget must be non-negative")
        if not 0.0 <= self.team_reward_weight <= 1.0:
            raise ValueError("team_reward_weight must be in [0, 1]")
        if not self.enabled:
            # Canonicalize the disabled case so no downstream code has to
            # remember to check `enabled` before reading a team id.
            self.slot_teams = [SOLO_TEAM for _ in self.slot_teams]
            self.comms_enabled = False
            self.team_reward_weight = 0.0

    @property
    def slot_count(self) -> int:
        return len(self.slot_teams)

    @property
    def teams(self) -> list[int]:
        seen: list[int] = []
        for team in self.slot_teams:
            if team not in seen:
                seen.append(team)
        return sorted(seen)

    def team_of(self, slot: int) -> int:
        if not 0 <= slot < len(self.slot_teams):
            raise IndexError(f"agent slot {slot} out of range (0..{len(self.slot_teams) - 1})")
        return self.slot_teams[slot]

    def slots_of(self, team: int) -> list[int]:
        return [slot for slot, value in enumerate(self.slot_teams) if value == team]

    def teammates_of(self, slot: int) -> list[int]:
        team = self.team_of(slot)
        return [other for other in self.slots_of(team) if other != slot]

    def are_teammates(self, slot_a: int, slot_b: int) -> bool:
        return self.team_of(slot_a) == self.team_of(slot_b)

    def can_damage(self, attacker_slot: int, victim_slot: int) -> bool:
        """Friendly fire gate. Self-damage is never allowed."""
        if attacker_slot == victim_slot:
            return False
        if self.are_teammates(attacker_slot, victim_slot):
            return bool(self.friendly_fire)
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamConfig":
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)

    @classmethod
    def solo(cls) -> "TeamConfig":
        return cls()

    @classmethod
    def versus(cls, per_team: int = 2, **kwargs: Any) -> "TeamConfig":
        """``TeamConfig.versus(2)`` -> a 2v2 layout (slots 0,1 vs 2,3)."""
        if per_team < 1:
            raise ValueError("per_team must be >= 1")
        slots = [0] * per_team + [1] * per_team
        return cls(enabled=True, slot_teams=slots, **kwargs)


@dataclass
class TeammateReport:
    """What one teammate tells another.

    Strictly perception-derived. There is deliberately no field for a
    ground-truth enemy position, enemy health or enemy intent: a teammate
    can only pass on what it perceived, with how old that is and how sure
    it is. ``validate`` rejects extra keys so the boundary cannot be
    widened by accident.
    """

    slot: int
    team: int
    ## Perception the teammate had. Bearings/distances are normalized the
    ## same way the observation vector is.
    contact_visible: bool = False
    contact_bearing_norm: float = 0.0
    contact_distance_norm: float = 0.0
    contact_confidence: float = 0.0
    ## Seconds since the teammate observed it.
    age: float = 0.0
    ## Teammate's own state, which it obviously knows about itself.
    health_norm: float = 1.0
    position_bearing_norm: float = 0.0
    position_distance_norm: float = 0.0
    alive: bool = True

    ALLOWED_FIELDS: tuple[str, ...] = ()

    def decayed(self, half_life: float) -> "TeammateReport":
        """Confidence decays with age, exactly like single-agent memory."""
        if half_life <= 0.0 or self.age <= 0.0:
            return self
        factor = 0.5 ** (self.age / half_life)
        return TeammateReport(
            slot=self.slot,
            team=self.team,
            contact_visible=self.contact_visible,
            contact_bearing_norm=self.contact_bearing_norm,
            contact_distance_norm=self.contact_distance_norm,
            contact_confidence=self.contact_confidence * factor,
            age=self.age,
            health_norm=self.health_norm,
            position_bearing_norm=self.position_bearing_norm,
            position_distance_norm=self.position_distance_norm,
            alive=self.alive,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ALLOWED_FIELDS", None)
        return payload


## Keys a teammate report may ever contain. Anything else is a leak.
TEAMMATE_REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "slot",
        "team",
        "contact_visible",
        "contact_bearing_norm",
        "contact_distance_norm",
        "contact_confidence",
        "age",
        "health_norm",
        "position_bearing_norm",
        "position_distance_norm",
        "alive",
    }
)

## Keys that would constitute privileged information if shared.
FORBIDDEN_REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "enemy_position",
        "enemy_health",
        "enemy_velocity",
        "enemy_id",
        "world_geometry",
        "map_id",
        "opponent_observation",
        "opponent_action",
    }
)


def validate_teammate_report(payload: dict[str, Any]) -> None:
    """Rejects a report that carries information a teammate cannot have."""
    keys = set(payload)
    leaked = keys & FORBIDDEN_REPORT_FIELDS
    if leaked:
        raise TeamError(f"teammate report leaks privileged information: {sorted(leaked)}")
    unknown = keys - TEAMMATE_REPORT_FIELDS
    if unknown:
        raise TeamError(f"teammate report has unknown fields: {sorted(unknown)}")


@dataclass
class CommsMessage:
    sender_slot: int
    team: int
    symbol: str
    ## Optional normalized bearing the symbol refers to ("contact, there").
    bearing_norm: float = 0.0
    tick: int = 0

    def __post_init__(self) -> None:
        if self.symbol not in COMMS_SYMBOLS:
            raise ValueError(f"unknown comms symbol: {self.symbol!r} (known: {COMMS_SYMBOLS})")
        if not -1.0 <= self.bearing_norm <= 1.0:
            raise ValueError("bearing_norm must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommsChannel:
    """Per-team, per-tick discrete message bus.

    Messages are only ever delivered to the sender's own team. There is no
    API that returns another team's traffic, which is what makes "team
    isolation" a property of the type rather than of caller discipline.
    """

    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self._messages: list[CommsMessage] = []
        self._sent_this_tick: dict[int, int] = {}
        self.tick: int = 0

    def send(self, slot: int, symbol: str, bearing_norm: float = 0.0) -> CommsMessage | None:
        """Sends one message; returns None when comms are off or over budget."""
        if not self.config.enabled or not self.config.comms_enabled:
            return None
        used = self._sent_this_tick.get(slot, 0)
        if used >= self.config.comms_budget:
            return None
        message = CommsMessage(
            sender_slot=slot,
            team=self.config.team_of(slot),
            symbol=symbol,
            bearing_norm=bearing_norm,
            tick=self.tick,
        )
        self._messages.append(message)
        self._sent_this_tick[slot] = used + 1
        return message

    def inbox(self, slot: int) -> list[CommsMessage]:
        """Messages visible to ``slot``: its own team's, minus its own."""
        team = self.config.team_of(slot)
        return [
            message
            for message in self._messages
            if message.team == team and message.sender_slot != slot
        ]

    def advance(self) -> None:
        """Ends the tick. Messages are transient, like sound events."""
        self._messages.clear()
        self._sent_this_tick.clear()
        self.tick += 1

    def reset(self) -> None:
        self._messages.clear()
        self._sent_this_tick.clear()
        self.tick = 0

    def pending(self) -> int:
        return len(self._messages)


@dataclass
class TeamOutcome:
    """Per-agent episode outcome, used by the reward hooks."""

    slot: int
    reward: float = 0.0
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_received: float = 0.0
    alive: bool = True
    objective_progress: float = 0.0


class TeamRewardHooks:
    """Turns per-agent outcomes into optional team-shaped rewards.

    Nothing here is applied automatically. The environment asks for an
    adjustment and decides what to do with it, which keeps the default
    (weight 0.0, teamplay off) mathematically identical to the current
    single-agent reward.
    """

    def __init__(self, config: TeamConfig) -> None:
        self.config = config

    def team_score(self, outcomes: Sequence[TeamOutcome], team: int) -> float:
        """Scores a team against its configured objective."""
        slots = set(self.config.slots_of(team))
        members = [outcome for outcome in outcomes if outcome.slot in slots]
        if not members:
            return 0.0
        objective = self.config.objective
        if objective == "eliminate":
            return float(sum(outcome.kills for outcome in members))
        if objective == "survive":
            return float(sum(1 for outcome in members if outcome.alive)) / len(members)
        if objective == "control":
            return sum(outcome.objective_progress for outcome in members) / len(members)
        # explore
        return sum(outcome.objective_progress for outcome in members)

    def shaped_rewards(self, outcomes: Sequence[TeamOutcome]) -> dict[int, float]:
        """Blends individual reward with the agent's team score."""
        weight = self.config.team_reward_weight
        if not self.config.enabled or weight <= 0.0:
            return {outcome.slot: outcome.reward for outcome in outcomes}
        scores = {team: self.team_score(outcomes, team) for team in self.config.teams}
        shaped: dict[int, float] = {}
        for outcome in outcomes:
            team = self.config.team_of(outcome.slot)
            shaped[outcome.slot] = (1.0 - weight) * outcome.reward + weight * scores[team]
        return shaped

    def friendly_fire_penalty(self, attacker_slot: int, victim_slot: int, damage: float) -> float:
        """Penalty magnitude for hitting a teammate. 0 when it is impossible."""
        if not self.config.enabled or not self.config.friendly_fire:
            return 0.0
        if not self.config.are_teammates(attacker_slot, victim_slot):
            return 0.0
        return abs(float(damage))


def build_teammate_reports(
    config: TeamConfig, slot: int, perceptions: dict[int, dict[str, Any]], half_life: float = 4.0
) -> list[TeammateReport]:
    """Builds the reports ``slot`` receives from its teammates.

    ``perceptions`` maps a slot to that agent's *own* perception summary.
    Only teammates' entries are read, and each is validated against the
    allowed field set before it is turned into a report — so a caller
    cannot smuggle a ground-truth enemy position through this function.
    """
    if not config.enabled:
        return []
    reports: list[TeammateReport] = []
    for teammate in config.teammates_of(slot):
        payload = perceptions.get(teammate)
        if payload is None:
            continue
        validate_teammate_report(payload)
        report = TeammateReport(
            slot=teammate,
            team=config.team_of(teammate),
            **{key: value for key, value in payload.items() if key not in ("slot", "team")},
        )
        reports.append(report.decayed(half_life))
    return reports


def team_summary(config: TeamConfig, outcomes: Iterable[TeamOutcome]) -> dict[str, Any]:
    """Per-team aggregate for a report or the Control Center."""
    rows = list(outcomes)
    hooks = TeamRewardHooks(config)
    return {
        "enabled": config.enabled,
        "objective": config.objective,
        "friendly_fire": config.friendly_fire,
        "teams": {
            str(team): {
                "slots": config.slots_of(team),
                "score": hooks.team_score(rows, team),
                "kills": sum(o.kills for o in rows if config.team_of(o.slot) == team),
                "deaths": sum(o.deaths for o in rows if config.team_of(o.slot) == team),
                "alive": sum(1 for o in rows if config.team_of(o.slot) == team and o.alive),
            }
            for team in config.teams
        },
    }
