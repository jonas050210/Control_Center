"""External-game adapter boundary: contract, validation and a mock.

Phase 9. The pipeline this module defines:

    External game -> ObservationAdapter -> SandboxAI policy
                  <- ActionAdapter      <-

``contract.GameAdapter`` already declared the *shape* of that seam. What
was missing is everything that makes the seam enforceable: what counts as
legal observation data, what timing guarantees an adapter must offer, what
an episode boundary means outside Godot, and a runnable mock that proves
the contract is implementable.

**Scope statement, unchanged and deliberate.** Nothing here connects to
Roblox or any other external game. There is no client injection, no memory
reading, no packet manipulation, no anti-cheat interaction and no
automation of another player's account. The only thing this module can
talk to is a process that *chooses* to implement
:class:`ExternalEnvironment` — i.e. a game or server that has opted in and
exposes its own supported interface. Anything else is out of scope by
design, not by omission.

**The hard rule the validator enforces:** an adapter may only report
information the controlled character could itself perceive. Every
observation channel has a defined "no information" encoding, and an
adapter that cannot supply a channel honestly must emit that encoding
rather than substituting privileged state. ``validate_observation`` and
``AdapterContractChecker`` turn that from a docstring into a test.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import math
import random
from typing import Any, Iterable, Sequence

from .contract import (
    ACTION_NVEC,
    ACTION_SPEC,
    OBSERVATION_FIELD_COUNT,
    OBSERVATION_GROUPS,
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    OBSERVATION_INDEX,
    OBSERVATION_SPEC,
    GameAdapter,
)

## Version of the *external* contract. Separate from the observation
## contract version because an adapter can be revised (timing rules,
## episode semantics) without the observation layout changing.
EXTERNAL_CONTRACT_VERSION: int = 1

## Episode lifecycle states an external environment reports. "unavailable"
## exists because a real external game can simply stop being there
## (session ended, player left), and pretending that is a normal
## termination corrupts the statistics.
EPISODE_STATES: tuple[str, ...] = ("idle", "running", "terminated", "truncated", "unavailable")

## Information an adapter is allowed to source. Anything else is
## privileged and must not be used, even if the target platform exposes it.
ALLOWED_DATA_SOURCES: tuple[str, ...] = (
    "own_character_state",  # position, orientation, velocity, health, grounded
    "own_weapon_state",  # ready/cooldown/ammo of the controlled character
    "rendered_visibility",  # what the character's camera/FOV can actually see
    "audible_events",  # sounds the character could hear, with attenuation
    "own_memory",  # anything the adapter itself remembered from the above
    "public_match_state",  # score/round timer any player can read on the HUD
)

## Explicitly forbidden sources. The validator does not scan an adapter's
## implementation (it cannot), but a declared source outside the allow-list
## fails the contract check, which makes the boundary reviewable.
FORBIDDEN_DATA_SOURCES: tuple[str, ...] = (
    "server_authoritative_state",
    "other_player_private_state",
    "hidden_entity_positions",
    "process_memory",
    "network_packet_inspection",
    "client_modification",
)


class AdapterContractError(RuntimeError):
    """An adapter violated the external environment contract."""


@dataclass
class TimingContract:
    """What the adapter promises about time.

    An RL policy trained at a fixed 60 Hz tick cannot be dropped onto a
    variable-rate external game without saying what happens to the
    mismatch. Three honest options, declared explicitly:

    ``fixed``     the external game steps at exactly ``tick_hz``
    ``polled``    the adapter samples at ``tick_hz`` and repeats/drops
                  observations; ``max_latency`` bounds the staleness
    ``event``     the external game drives the loop and the adapter blocks
    """

    mode: str = "polled"
    tick_hz: float = 60.0
    ## Worst-case age (seconds) of the observation handed to the policy.
    max_latency: float = 0.05
    ## Whether the adapter guarantees actions are applied in order.
    ordered_actions: bool = True
    ## Whether repeated reset(seed) reproduces the episode. Almost always
    ## False for a live external game, and saying so is the point.
    deterministic: bool = False

    MODES: tuple[str, ...] = ("fixed", "polled", "event")

    def validate(self) -> None:
        if self.mode not in self.MODES:
            raise AdapterContractError(f"unknown timing mode: {self.mode!r}")
        if self.tick_hz <= 0.0:
            raise AdapterContractError("tick_hz must be positive")
        if self.max_latency < 0.0:
            raise AdapterContractError("max_latency must be non-negative")
        if self.mode == "fixed" and self.max_latency > 1.0 / self.tick_hz:
            raise AdapterContractError(
                "a 'fixed' timing adapter cannot have a latency larger than one tick"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("MODES", None)
        return payload


@dataclass
class AdapterCapabilities:
    """Which observation channels an adapter can honestly supply.

    A channel that is False must be emitted as its neutral encoding
    (zeros / flags at 0). ``missing_channels`` is what the Control Center
    and the evaluation report show, so a result from a partially capable
    adapter is never presented as comparable to a full-perception run.
    """

    name: str = "external"
    version: int = EXTERNAL_CONTRACT_VERSION
    channels: dict[str, bool] = field(
        default_factory=lambda: {channel: True for channel in OBSERVATION_GROUPS}
    )
    data_sources: list[str] = field(default_factory=lambda: list(ALLOWED_DATA_SOURCES))
    timing: TimingContract = field(default_factory=TimingContract)
    ## Whether the external game can supply a scalar reward at all. Many
    ## cannot; evaluation-only adapters report False and the platform then
    ## treats the run as evaluation, not training.
    provides_reward: bool = False

    def validate(self) -> None:
        unknown = set(self.channels) - set(OBSERVATION_GROUPS)
        if unknown:
            raise AdapterContractError(f"unknown observation channels: {sorted(unknown)}")
        missing_declaration = set(OBSERVATION_GROUPS) - set(self.channels)
        if missing_declaration:
            raise AdapterContractError(
                f"adapter does not declare these channels: {sorted(missing_declaration)}"
            )
        forbidden = set(self.data_sources) & set(FORBIDDEN_DATA_SOURCES)
        if forbidden:
            raise AdapterContractError(
                f"adapter declares forbidden data sources: {sorted(forbidden)}"
            )
        unlisted = set(self.data_sources) - set(ALLOWED_DATA_SOURCES)
        if unlisted:
            raise AdapterContractError(f"adapter declares unknown data sources: {sorted(unlisted)}")
        self.timing.validate()

    @property
    def missing_channels(self) -> list[str]:
        return sorted(name for name, available in self.channels.items() if not available)

    def neutral_indices(self) -> list[int]:
        """Indices that MUST be zero given the declared capabilities."""
        indices: list[int] = []
        for channel in self.missing_channels:
            for name in OBSERVATION_GROUPS[channel]:
                start, width = OBSERVATION_INDEX[name]
                indices.extend(range(start, start + width))
        return sorted(indices)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timing"] = self.timing.to_dict()
        payload["missing_channels"] = self.missing_channels
        return payload


@dataclass
class EpisodeStatus:
    """Episode lifecycle as reported by an external environment."""

    state: str = "idle"
    tick: int = 0
    elapsed: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in EPISODE_STATES:
            raise AdapterContractError(f"unknown episode state: {self.state!r}")

    @property
    def finished(self) -> bool:
        return self.state in ("terminated", "truncated", "unavailable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_observation(
    observation: Sequence[float], capabilities: AdapterCapabilities | None = None
) -> None:
    """Rejects an observation that breaks the contract.

    Checks: width, finiteness, range, and — when capabilities are given —
    that every channel the adapter admits it cannot supply is actually
    zero. The last check is what stops "we don't have sound, so we put the
    enemy's real bearing there" from passing silently.
    """
    values = list(observation)
    if len(values) != OBSERVATION_FIELD_COUNT:
        raise AdapterContractError(
            f"observation has {len(values)} values, contract requires {OBSERVATION_FIELD_COUNT}"
        )
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise AdapterContractError(f"observation[{index}] is not finite")
        if number < OBSERVATION_LOW - 1e-6 or number > OBSERVATION_HIGH + 1e-6:
            raise AdapterContractError(
                f"observation[{index}] = {number} outside [{OBSERVATION_LOW}, {OBSERVATION_HIGH}]"
            )
    if capabilities is None:
        return
    for index in capabilities.neutral_indices():
        if abs(float(values[index])) > 1e-9:
            field_name = OBSERVATION_SPEC[0].name
            for spec in OBSERVATION_SPEC:
                if spec.index <= index < spec.index + spec.width:
                    field_name = spec.name
                    break
            raise AdapterContractError(
                f"adapter reports no data for this channel but observation[{index}] "
                f"({field_name}) is non-zero; emit the neutral encoding instead of "
                "substituting privileged information"
            )


def validate_action(action: Sequence[int]) -> None:
    """Rejects an action outside the MultiDiscrete space."""
    values = list(action)
    if len(values) != len(ACTION_NVEC):
        raise AdapterContractError(
            f"action has {len(values)} components, contract requires {len(ACTION_NVEC)}"
        )
    for index, value in enumerate(values):
        number = int(value)
        if not 0 <= number < ACTION_NVEC[index]:
            raise AdapterContractError(
                f"action[{index}] ({ACTION_SPEC[index].name}) = {number} outside "
                f"[0, {ACTION_NVEC[index]})"
            )


class ExternalEnvironment(GameAdapter, ABC):
    """The contract an external game integration implements.

    Extends the existing ``GameAdapter`` seam with the four things a real
    external integration needs and Godot got for free: declared
    capabilities, explicit episode status, an action-application
    acknowledgement, and the ability to say "I am not available right now"
    without that looking like a lost episode.
    """

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Static declaration of what this adapter can supply."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> EpisodeStatus:
        """Current episode lifecycle state."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Machine-readable description, for logs and the Control Center."""
        capabilities = self.capabilities
        return {
            "adapter": capabilities.name,
            "contract_version": EXTERNAL_CONTRACT_VERSION,
            "observation_dim": OBSERVATION_FIELD_COUNT,
            "action_nvec": list(ACTION_NVEC),
            "capabilities": capabilities.to_dict(),
            "status": self.status().to_dict(),
        }


class MockExternalEnvironment(ExternalEnvironment):
    """A runnable, dependency-free implementation of the contract.

    Its purpose is to prove the interface is implementable and to give the
    contract tests something concrete to run against. It simulates nothing
    interesting: a seeded pseudo-target drifts in bearing, the agent can
    "hit" it by firing while centred, and the episode ends after a fixed
    number of ticks. It is not a game, and it is explicitly not a stand-in
    for any real external product.
    """

    def __init__(
        self,
        capabilities: AdapterCapabilities | None = None,
        episode_length: int = 64,
        seed: int = 0,
    ) -> None:
        self._capabilities = capabilities or AdapterCapabilities(name="mock")
        self._capabilities.validate()
        self.episode_length = int(episode_length)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._status = EpisodeStatus(state="idle")
        self._bearing = 0.0
        self._health = 1.0
        self._target_health = 1.0
        self.closed = False

    # -- contract ----------------------------------------------------------

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def status(self) -> EpisodeStatus:
        return self._status

    def reset(self, seed: int | None = None):
        if self.closed:
            raise AdapterContractError("cannot reset a closed adapter")
        self._seed = self._seed if seed is None else int(seed)
        self._rng = random.Random(self._seed)
        self._status = EpisodeStatus(state="running", tick=0, elapsed=0.0)
        self._bearing = self._rng.uniform(-0.5, 0.5)
        self._health = 1.0
        self._target_health = 1.0
        return self._observation()

    def step(self, action: Sequence[int]):
        if self.closed:
            raise AdapterContractError("cannot step a closed adapter")
        if self._status.state != "running":
            raise AdapterContractError(
                f"cannot step while the episode is {self._status.state!r}; call reset()"
            )
        validate_action(action)

        yaw = int(action[2]) - 1
        shooting = int(action[4]) == 1
        self._bearing = max(-1.0, min(1.0, self._bearing - yaw * 0.05 + self._rng.uniform(-0.01, 0.01)))

        reward = 0.0
        hit = False
        if shooting and abs(self._bearing) < 0.05:
            hit = True
            self._target_health = max(0.0, self._target_health - 0.34)
            reward += 1.0
        elif shooting:
            reward -= 0.05

        self._status.tick += 1
        self._status.elapsed += 1.0 / self._capabilities.timing.tick_hz
        terminated = self._target_health <= 0.0
        truncated = self._status.tick >= self.episode_length
        if terminated:
            self._status = EpisodeStatus(
                state="terminated", tick=self._status.tick, elapsed=self._status.elapsed,
                reason="target_eliminated",
            )
        elif truncated:
            self._status = EpisodeStatus(
                state="truncated", tick=self._status.tick, elapsed=self._status.elapsed,
                reason="time_limit",
            )
        info = {
            "events": {"shot_fired": shooting, "hit": hit},
            "episode_state": self._status.state,
            "capabilities_missing": self._capabilities.missing_channels,
        }
        return self._observation(), reward, self._status.finished, info

    def close(self) -> None:
        self.closed = True
        self._status = EpisodeStatus(state="idle")

    # -- observation -------------------------------------------------------

    def _observation(self) -> list[float]:
        """Builds the vector by FIELD NAME, zeroing unavailable channels."""
        vector = [0.0] * OBSERVATION_FIELD_COUNT

        def put(name: str, value: float) -> None:
            index, width = OBSERVATION_INDEX[name]
            if width != 1:
                raise ValueError(f"{name} is not scalar")
            vector[index] = max(OBSERVATION_LOW, min(OBSERVATION_HIGH, float(value)))

        channels = self._capabilities.channels
        if channels.get("self_state", True):
            put("agent_health_norm", self._health)
        if channels.get("combat", True):
            put("weapon_ready", 1.0)
            put("in_combat", 1.0 if self._target_health > 0.0 else 0.0)
        if channels.get("targets", True):
            put("primary_enemy_bearing_norm", self._bearing)
            put("primary_enemy_distance_norm", 0.4)
            put("primary_enemy_health_norm", self._target_health)
            put("alive_enemy_count_norm", 1.0 if self._target_health > 0.0 else 0.0)
        if channels.get("perception", True):
            put("primary_enemy_visible", 1.0 if self._target_health > 0.0 else 0.0)
            put("primary_enemy_in_fov", 1.0 if abs(self._bearing) < 0.3 else 0.0)
            put("primary_enemy_los_clear", 1.0)
            put("visible_enemy_count_norm", 0.125 if self._target_health > 0.0 else 0.0)
        if channels.get("memory", True):
            put("primary_enemy_confidence", 1.0 if self._target_health > 0.0 else 0.0)
            put("primary_enemy_source_visual", 1.0)
        return vector


class AdapterContractChecker:
    """Runs an adapter through the contract and reports every violation.

    Used by the contract tests and usable as a pre-flight check before
    pointing a policy at a new integration. It returns problems rather
    than raising on the first one, because an adapter author wants the
    whole list.
    """

    def __init__(self, adapter: ExternalEnvironment) -> None:
        self.adapter = adapter

    def run(self, steps: int = 16, seed: int = 123) -> list[str]:
        problems: list[str] = []
        capabilities = self.adapter.capabilities
        try:
            capabilities.validate()
        except AdapterContractError as exc:
            problems.append(f"capabilities: {exc}")

        try:
            observation = self.adapter.reset(seed=seed)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the check
            return problems + [f"reset() raised {type(exc).__name__}: {exc}"]

        try:
            validate_observation(observation, capabilities)
        except AdapterContractError as exc:
            problems.append(f"reset observation: {exc}")

        status = self.adapter.status()
        if status.state != "running":
            problems.append(f"after reset the episode state is {status.state!r}, expected 'running'")

        idle_action = [1, 1, 1, 1, 0, 0]
        for index in range(steps):
            if self.adapter.status().finished:
                break
            try:
                observation, reward, done, info = self.adapter.step(idle_action)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"step {index} raised {type(exc).__name__}: {exc}")
                break
            try:
                validate_observation(observation, capabilities)
            except AdapterContractError as exc:
                problems.append(f"step {index} observation: {exc}")
            if not math.isfinite(float(reward)):
                problems.append(f"step {index}: non-finite reward")
            if not isinstance(info, dict):
                problems.append(f"step {index}: info is {type(info).__name__}, expected dict")
            if bool(done) != self.adapter.status().finished:
                problems.append(
                    f"step {index}: done={done} disagrees with status {self.adapter.status().state!r}"
                )

        # Stepping a finished episode must fail loudly rather than
        # returning made-up transitions.
        if self.adapter.status().finished:
            try:
                self.adapter.step(idle_action)
                problems.append("stepping a finished episode was accepted")
            except AdapterContractError:
                pass
            except Exception as exc:  # noqa: BLE001
                problems.append(f"stepping a finished episode raised {type(exc).__name__}: {exc}")

        description = self.adapter.describe()
        if description.get("observation_dim") != OBSERVATION_FIELD_COUNT:
            problems.append("describe() reports the wrong observation dimension")
        return problems


def contract_summary() -> dict[str, Any]:
    """The full external contract, as data. Used by docs and tests."""
    return {
        "version": EXTERNAL_CONTRACT_VERSION,
        "observation": {
            "dimension": OBSERVATION_FIELD_COUNT,
            "low": OBSERVATION_LOW,
            "high": OBSERVATION_HIGH,
            "channels": {name: list(fields) for name, fields in OBSERVATION_GROUPS.items()},
        },
        "action": {
            "type": "multi_discrete",
            "nvec": list(ACTION_NVEC),
            "fields": [field.name for field in ACTION_SPEC],
        },
        "timing": TimingContract().to_dict(),
        "episode_states": list(EPISODE_STATES),
        "allowed_data_sources": list(ALLOWED_DATA_SOURCES),
        "forbidden_data_sources": list(FORBIDDEN_DATA_SOURCES),
        "scope": (
            "Adapters connect only to games/servers that expose a supported, opted-in "
            "interface. No client modification, memory reading, packet manipulation or "
            "anti-cheat interaction is part of this contract or this repository."
        ),
    }


def iter_channel_fields(channel: str) -> Iterable[str]:
    return OBSERVATION_GROUPS[channel]
