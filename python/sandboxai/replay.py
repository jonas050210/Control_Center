"""Deterministic episode replay: recording, storage, validation and playback.

Phase 2 of the research-platform milestone. A replay is the only artifact
that makes a claim about a policy checkable after the fact: without one,
"the agent peeked the corner because it heard the shot" is an anecdote.

Design decisions, and why:

**One file, JSON Lines.** The first line is the header (format id, version,
seed, map, scenario, conditions, policy/checkpoint, contract dimensions);
every following line is one record. JSONL streams — a crashed run still
leaves a readable prefix — and stays greppable, which a pickle or a custom
binary format does not. There is no third-party dependency.

**Lightweight by default.** ``DetailLevel.LIGHT`` stores the tick, the
action and the reward, plus every discrete event. That is enough to *re-run*
the episode (the environment is seeded and the action stream is complete),
so observations are reconstructable rather than stored. ``DETAILED``
additionally stores the observation vector per tick for offline analysis
and for determinism verification without an engine. Storing 126 floats per
tick at 60 Hz is ~1.6 MB/minute/agent, which is why it is not the default.

**Versioned and validated.** ``REPLAY_FORMAT_VERSION`` is bumped whenever
the record layout changes; ``READABLE_VERSIONS`` lists what this build can
still load. A replay recorded against a different observation/action
contract is *rejected*, not silently played back against the current one —
a replay that does not line up with the contract is not a replay, it is a
different experiment.

**Playback is a cursor, not a simulator.** ``ReplayPlayer`` owns
pause/step/seek/speed and the event timeline. It never talks to Godot: the
Control Center drives the simulation itself, and a Python analysis script
wants the data without an engine. ``verify_determinism`` is the piece that
does involve an environment, and it takes an env factory so it can be
tested against a scripted environment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

## Magic string written into every header. Guards against handing a random
## JSONL file (a demonstration dataset, say) to the replay loader.
REPLAY_MAGIC: str = "sandboxai.replay"

## Current on-disk layout. History:
##   1 - initial format: header + tick/event records, 6-component
##       MultiDiscrete action. Each file stamps the observation contract it
##       was recorded under (format 1 was introduced with contract v3 / 84
##       floats and is still the format used by contract v5 / 126 floats);
##       loading a replay never requires the current contract to match.
REPLAY_FORMAT_VERSION: int = 1

## Versions this build can load. Older versions stay listed here (with an
## upgrade path in `_upgrade_header`) instead of being dropped, because a
## replay is evidence and evidence should not expire on a refactor.
READABLE_VERSIONS: tuple[int, ...] = (1,)


class ReplayError(RuntimeError):
    """A replay file is corrupt, truncated or not a replay at all."""


class ReplayIncompatibleError(ReplayError):
    """A replay is well-formed but was recorded against another contract."""


class DetailLevel:
    """How much per-tick data a recording keeps."""

    LIGHT: str = "light"
    DETAILED: str = "detailed"
    ALL: tuple[str, ...] = (LIGHT, DETAILED)


## Event kinds a recorder may emit. Kept as an explicit vocabulary so the
## timeline UI and the "jump to the next interesting thing" control have a
## fixed set to reason about, rather than whatever strings a caller typed.
EVENT_KINDS: tuple[str, ...] = (
    "episode_start",
    "episode_end",
    "target_change",
    "perception",  # contact acquired/lost, FOV/LOS transitions
    "memory",  # a remembered contact was created, refreshed or expired
    "sound",  # an audible event the agent actually perceived
    "combat",  # shot fired / hit / damage
    "death",
    "curriculum",
    "note",  # free-form annotation from a tool; never simulation state
)

## Events that a human or a policy-comparison run almost always wants to
## jump between. Used as the default filter of `ReplayPlayer.next_event`.
IMPORTANT_EVENT_KINDS: tuple[str, ...] = (
    "episode_start",
    "episode_end",
    "target_change",
    "combat",
    "death",
)


@dataclass
class ReplayHeader:
    """Everything needed to reproduce and to validate an episode.

    ``policy_id``/``checkpoint`` identify *which brain* produced the
    actions. They are metadata about the recording, never simulation
    inputs: replaying does not need a policy, only the action stream.
    """

    magic: str = REPLAY_MAGIC
    version: int = REPLAY_FORMAT_VERSION
    detail: str = DetailLevel.LIGHT
    ## Reproduction inputs.
    seed: int = 0
    map_id: str = ""
    scenario: str = ""
    lighting: str = ""
    enemy_count: int = 1
    curriculum_level: int = 1
    ## Provenance.
    policy_id: str = ""
    checkpoint: str = ""
    environment_index: int = 0
    agent_slot: int = 0
    team_id: int = 0
    ## Contract fingerprint; a mismatch makes the replay unplayable.
    observation_dim: int = OBSERVATION_FIELD_COUNT
    action_nvec: list[int] = field(default_factory=lambda: list(ACTION_NVEC))
    simulation_dt: float = 1.0 / 60.0
    ## Free-form, never used for validation.
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayHeader:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        header = cls(**known)
        header.action_nvec = [int(value) for value in header.action_nvec]
        return header

    def validate(self, strict_contract: bool = True) -> None:
        if self.magic != REPLAY_MAGIC:
            raise ReplayError(f"not a SandboxAI replay (magic={self.magic!r})")
        if self.version not in READABLE_VERSIONS:
            raise ReplayIncompatibleError(
                f"replay format version {self.version} is not readable by this build "
                f"(supported: {sorted(READABLE_VERSIONS)})"
            )
        if self.detail not in DetailLevel.ALL:
            raise ReplayError(f"unknown detail level: {self.detail!r}")
        if not strict_contract:
            return
        if int(self.observation_dim) != OBSERVATION_FIELD_COUNT:
            raise ReplayIncompatibleError(
                f"replay was recorded with a {self.observation_dim}-float observation "
                f"contract; this build uses {OBSERVATION_FIELD_COUNT}"
            )
        if tuple(self.action_nvec) != tuple(ACTION_NVEC):
            raise ReplayIncompatibleError(
                f"replay action space {tuple(self.action_nvec)} != contract {tuple(ACTION_NVEC)}"
            )


@dataclass
class ReplayTick:
    """One simulation step as recorded."""

    tick: int
    action: list[int]
    reward: float = 0.0
    done: bool = False
    ## Only present in DETAILED recordings.
    observation: list[float] | None = None
    ## Optional, tiny, and only written when non-empty.
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def time(self) -> float:  # pragma: no cover - trivial; dt lives in the header
        raise AttributeError("use ReplayEpisode.time_of(tick) — dt belongs to the header")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "t": self.tick,
            "a": list(self.action),
            "r": round(float(self.reward), 6),
        }
        if self.done:
            record["d"] = True
        if self.observation is not None:
            record["o"] = [round(float(value), 5) for value in self.observation]
        if self.info:
            record["i"] = self.info
        return record

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> ReplayTick:
        return cls(
            tick=int(payload["t"]),
            action=[int(value) for value in payload.get("a", [])],
            reward=float(payload.get("r", 0.0)),
            done=bool(payload.get("d", False)),
            observation=(
                [float(value) for value in payload["o"]] if payload.get("o") is not None else None
            ),
            info=dict(payload.get("i", {})),
        )


@dataclass
class ReplayEvent:
    """One discrete thing that happened, anchored to a tick."""

    tick: int
    kind: str
    label: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"t": self.tick, "e": self.kind}
        if self.label:
            record["l"] = self.label
        if self.data:
            record["v"] = self.data
        return record

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> ReplayEvent:
        return cls(
            tick=int(payload["t"]),
            kind=str(payload["e"]),
            label=str(payload.get("l", "")),
            data=dict(payload.get("v", {})),
        )


@dataclass
class ReplayEpisode:
    """A loaded replay: header + ticks + events + result."""

    header: ReplayHeader
    ticks: list[ReplayTick] = field(default_factory=list)
    events: list[ReplayEvent] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ticks)

    @property
    def event_tick(self) -> int:
        """Index of the last recorded tick, i.e. the one a new event describes.

        The recorder's namesake reads its own cursor, which points at the
        *next* tick; a parsed episode has no cursor, so the same answer is
        the last index in `ticks`. Before the first tick the answer is 0.
        """
        return max(0, len(self.ticks) - 1)

    @property
    def detailed(self) -> bool:
        return self.header.detail == DetailLevel.DETAILED

    def time_of(self, tick_index: int) -> float:
        """Episode time (seconds) of a tick index."""
        return float(tick_index) * float(self.header.simulation_dt)

    @property
    def duration(self) -> float:
        return self.time_of(len(self.ticks))

    def actions(self) -> list[list[int]]:
        return [tick.action for tick in self.ticks]

    def rewards(self) -> list[float]:
        return [tick.reward for tick in self.ticks]

    def total_reward(self) -> float:
        return float(sum(tick.reward for tick in self.ticks))

    def observations(self) -> list[list[float]]:
        if not self.detailed:
            raise ReplayError(
                "observations were not recorded (detail=light); re-run the episode from "
                "the seed and action stream, or record with detail=detailed"
            )
        return [tick.observation or [] for tick in self.ticks]

    def events_of(self, kinds: Sequence[str] | None = None) -> list[ReplayEvent]:
        if kinds is None:
            return list(self.events)
        wanted = set(kinds)
        return [event for event in self.events if event.kind in wanted]

    def timeline(self, kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Event list with episode time attached, ready for a timeline UI."""
        return [
            {
                "tick": event.tick,
                "time": self.time_of(event.tick),
                "kind": event.kind,
                "label": event.label,
                "data": event.data,
            }
            for event in self.events_of(kinds)
        ]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for event in self.events:
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
        return {
            "version": self.header.version,
            "detail": self.header.detail,
            "seed": self.header.seed,
            "map": self.header.map_id,
            "scenario": self.header.scenario,
            "lighting": self.header.lighting,
            "enemy_count": self.header.enemy_count,
            "curriculum_level": self.header.curriculum_level,
            "policy_id": self.header.policy_id,
            "checkpoint": self.header.checkpoint,
            "ticks": len(self.ticks),
            "duration": self.duration,
            "total_reward": self.total_reward(),
            "events": len(self.events),
            "event_kinds": kinds,
            "result": dict(self.result),
        }


class ReplayRecorder:
    """Accumulates one episode and writes it as JSONL.

    Recording is explicitly opt-in and cheap when off: a caller that never
    constructs a recorder pays nothing, and a LIGHT recorder appends one
    small dict per tick. Nothing here reads simulation state directly —
    the caller pushes what it already has, which keeps the recorder unable
    to leak privileged information into a policy.
    """

    def __init__(self, header: ReplayHeader | None = None, detail: str = DetailLevel.LIGHT) -> None:
        if detail not in DetailLevel.ALL:
            raise ValueError(f"unknown detail level: {detail!r}")
        self.header = header or ReplayHeader()
        self.header.detail = detail
        self.header.magic = REPLAY_MAGIC
        self.header.version = REPLAY_FORMAT_VERSION
        self.ticks: list[ReplayTick] = []
        self.events: list[ReplayEvent] = []
        self.result: dict[str, Any] = {}
        self._tick: int = 0
        self._closed: bool = False

    # -- recording ---------------------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def event_tick(self) -> int:
        """Tick index an event recorded right now belongs to."""
        return max(0, self._tick - 1) if self.ticks else 0

    @property
    def detailed(self) -> bool:
        return self.header.detail == DetailLevel.DETAILED

    def start(self, seed: int | None = None, **header_updates: Any) -> None:
        """Begins a new episode, resetting tick/event state.

        Reusing a recorder across episodes is the normal case in a vector
        environment, so ``start`` must clear everything the previous
        episode left behind. A recorder that quietly keeps the old ticks
        produces replays that desync a few thousand steps in.
        """
        self.ticks.clear()
        self.events.clear()
        self.result = {}
        self._tick = 0
        self._closed = False
        if seed is not None:
            self.header.seed = int(seed)
        for key, value in header_updates.items():
            if not hasattr(self.header, key):
                raise AttributeError(f"ReplayHeader has no field {key!r}")
            setattr(self.header, key, value)
        self.event(
            "episode_start", label=self.header.map_id or "episode", data={"seed": self.header.seed}
        )

    def record_step(
        self,
        action: Sequence[int],
        reward: float = 0.0,
        observation: Sequence[float] | None = None,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Records one tick and returns its index."""
        if self._closed:
            raise ReplayError("cannot record into a finished replay; call start() first")
        stored_observation: list[float] | None = None
        if self.detailed and observation is not None:
            stored_observation = [float(value) for value in observation]
            if len(stored_observation) != int(self.header.observation_dim):
                raise ReplayError(
                    f"observation has {len(stored_observation)} values, header declares "
                    f"{self.header.observation_dim}"
                )
        self.ticks.append(
            ReplayTick(
                tick=self._tick,
                action=[int(value) for value in action],
                reward=float(reward),
                done=bool(done),
                observation=stored_observation,
                info=dict(info or {}),
            )
        )
        index = self._tick
        self._tick += 1
        return index

    def event(self, kind: str, label: str = "", data: dict[str, Any] | None = None) -> ReplayEvent:
        """Records a discrete event, anchored to the tick it describes.

        Callers record a step and *then* translate that step's events, by
        which point ``self._tick`` already points at the next, not yet
        recorded tick. Anchoring to it filed every event one tick after the
        thing it described, so ``events_at(n)`` never found them and "jump
        to next event" landed one tick late. Before the first step
        (``episode_start``) the answer is tick 0.
        """
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown replay event kind: {kind!r} (known: {EVENT_KINDS})")
        event = ReplayEvent(tick=self.event_tick, kind=kind, label=label, data=dict(data or {}))
        self.events.append(event)
        return event

    def record_events(self, events: dict[str, Any]) -> list[ReplayEvent]:
        """Translates one Godot step ``info["events"]`` dict into replay events.

        Only *things that happened* become events; the per-tick scalars
        (reward, positioning delta) already live on the tick record and
        are not duplicated here.
        """
        emitted: list[ReplayEvent] = []
        shot_result = str(events.get("shot_result", "none"))
        if events.get("shot_fired") or shot_result != "none":
            emitted.append(
                self.event(
                    "combat",
                    "shot",
                    {
                        "result": shot_result,
                        "fired": bool(events.get("shot_fired", False)),
                        "hit": bool(events.get("hit", False)),
                        "damage": float(events.get("damage_dealt", 0.0)),
                        "near_miss": bool(events.get("missed_shot", False)),
                        "useless": bool(events.get("useless_shot", False)),
                    },
                )
            )
        if float(events.get("damage_taken", 0.0)) > 0.0:
            emitted.append(
                self.event("combat", "damage_taken", {"amount": float(events["damage_taken"])})
            )
        if events.get("kill"):
            emitted.append(self.event("death", "enemy_killed", {}))
        if events.get("died"):
            emitted.append(self.event("death", "agent_died", {}))
        return emitted

    def finish(self, result: dict[str, Any] | None = None) -> ReplayEpisode:
        """Closes the episode and returns it as a loadable object."""
        self.result = dict(result or {})
        self.event(
            "episode_end", label=str(self.result.get("done_reason", "")), data=dict(self.result)
        )
        self._closed = True
        return self.episode()

    def episode(self) -> ReplayEpisode:
        return ReplayEpisode(
            header=ReplayHeader.from_dict(self.header.to_dict()),
            ticks=list(self.ticks),
            events=list(self.events),
            result=dict(self.result),
        )

    # -- persistence -------------------------------------------------------

    def iter_lines(self) -> Iterator[str]:
        yield json.dumps({"header": self.header.to_dict()}, separators=(",", ":"))
        for tick in self.ticks:
            yield json.dumps({"tick": tick.to_record()}, separators=(",", ":"))
        for event in self.events:
            yield json.dumps({"event": event.to_record()}, separators=(",", ":"))
        yield json.dumps({"result": self.result}, separators=(",", ":"))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for line in self.iter_lines():
                stream.write(line + "\n")
        return target

    def size_estimate_bytes(self) -> int:
        """Rough serialized size, for deciding whether DETAILED is affordable."""
        return sum(len(line) + 1 for line in self.iter_lines())


def save_episode(episode: ReplayEpisode, path: str | Path) -> Path:
    """Writes an already-loaded episode back out (round-trip / trimming)."""
    recorder = ReplayRecorder(header=episode.header, detail=episode.header.detail)
    recorder.ticks = list(episode.ticks)
    recorder.events = list(episode.events)
    recorder.result = dict(episode.result)
    return recorder.save(path)


def _upgrade_header(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrates an older header payload to the current field set.

    There is nothing to migrate at version 1; the hook exists so that the
    next format bump has an obvious place to live and old replays keep
    loading instead of being quietly dropped from READABLE_VERSIONS.
    """
    return payload


def load_replay(path: str | Path, strict_contract: bool = True) -> ReplayEpisode:
    """Loads a replay, rejecting corrupt and incompatible files loudly."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"replay not found: {source}")
    text = source.read_text(encoding="utf-8")
    return parse_replay(text.splitlines(), strict_contract=strict_contract)


def _decode_replay_line(raw: str, number: int) -> dict[str, Any] | None:
    """One JSONL line as a record dict, or None for a blank line."""
    line = raw.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"replay line {number} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayError(f"replay line {number} is not a JSON object")
    return payload


def _check_tick_sequence(ticks: list[ReplayTick], expected_action_width: int) -> None:
    """Ticks must be contiguous from 0 and match the header's action width.

    A gap means the recorder dropped a frame, which silently changes what a
    replay proves; refusing to load is better than analysing a hole.
    """
    for tick in ticks:
        if len(tick.action) != expected_action_width:
            raise ReplayError(
                f"tick {tick.tick} has a {len(tick.action)}-component action, "
                f"header declares {expected_action_width}"
            )
    for index, tick in enumerate(ticks):
        if tick.tick != index:
            raise ReplayError(
                f"replay ticks are not contiguous: expected {index}, found {tick.tick}"
            )


def parse_replay(lines: Iterable[str], strict_contract: bool = True) -> ReplayEpisode:
    header: ReplayHeader | None = None
    ticks: list[ReplayTick] = []
    events: list[ReplayEvent] = []
    result: dict[str, Any] = {}
    for number, raw in enumerate(lines, start=1):
        payload = _decode_replay_line(raw, number)
        if payload is None:
            continue
        if "header" in payload:
            if header is not None:
                raise ReplayError(f"replay line {number}: a second header")
            header = ReplayHeader.from_dict(_upgrade_header(payload["header"]))
            header.validate(strict_contract=strict_contract)
            continue
        if header is None:
            raise ReplayError("replay does not start with a header line")
        if "tick" in payload:
            ticks.append(ReplayTick.from_record(payload["tick"]))
        elif "event" in payload:
            events.append(ReplayEvent.from_record(payload["event"]))
        elif "result" in payload:
            result = dict(payload["result"])
        else:
            raise ReplayError(f"replay line {number}: unknown record {sorted(payload)}")
    if header is None:
        raise ReplayError("empty replay: no header")
    _check_tick_sequence(ticks, len(header.action_nvec))
    return ReplayEpisode(header=header, ticks=ticks, events=events, result=result)


def validate_replay(episode: ReplayEpisode, strict_contract: bool = True) -> list[str]:
    """Returns a list of problems; empty means the replay is usable.

    Separate from loading on purpose: loading refuses to produce a broken
    object, while this answers "is this replay *good*" (plausible rewards,
    events anchored inside the episode, a terminal tick) for a tool that
    wants to warn rather than fail.
    """
    problems: list[str] = []
    try:
        episode.header.validate(strict_contract=strict_contract)
    except ReplayError as exc:
        problems.append(str(exc))
    if not episode.ticks:
        problems.append("replay contains no ticks")
    nvec = tuple(episode.header.action_nvec)
    for tick in episode.ticks:
        for index, value in enumerate(tick.action):
            if index < len(nvec) and not 0 <= value < nvec[index]:
                problems.append(
                    f"tick {tick.tick}: action component {index} = {value} outside [0, {nvec[index]})"
                )
        if math.isnan(tick.reward) or math.isinf(tick.reward):
            problems.append(f"tick {tick.tick}: non-finite reward")
        if episode.detailed and tick.observation is None:
            problems.append(f"tick {tick.tick}: detailed replay without an observation")
    last_tick = len(episode.ticks) - 1
    for event in episode.events:
        if event.tick < 0 or event.tick > last_tick + 1:
            problems.append(
                f"event {event.kind} anchored at tick {event.tick}, outside the episode"
            )
        if event.kind not in EVENT_KINDS:
            problems.append(f"unknown event kind: {event.kind}")
    return problems


class ReplayPlayer:
    """Cursor over a loaded replay: pause, step, seek, speed, timeline.

    Playback is time-based rather than frame-based so a speed of 0.25 or 4
    behaves the same regardless of the caller's refresh rate. ``advance``
    takes wall-clock delta seconds and returns the ticks that were crossed,
    which is exactly what a 10 Hz Control Center refresh needs.
    """

    MIN_SPEED: float = 0.05
    MAX_SPEED: float = 16.0

    def __init__(self, episode: ReplayEpisode) -> None:
        self.episode = episode
        self.cursor: int = 0
        self.playing: bool = False
        self.speed: float = 1.0
        self._accumulator: float = 0.0

    # -- transport ---------------------------------------------------------

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def set_speed(self, speed: float) -> float:
        self.speed = min(max(float(speed), self.MIN_SPEED), self.MAX_SPEED)
        return self.speed

    def reset(self) -> None:
        """Rewinds to the first tick and stops. Speed is preserved."""
        self.cursor = 0
        self.playing = False
        self._accumulator = 0.0

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.episode.ticks)

    def step(self, count: int = 1) -> list[ReplayTick]:
        """Advances exactly ``count`` ticks regardless of speed/paused state."""
        crossed: list[ReplayTick] = []
        for _ in range(max(0, count)):
            if self.finished:
                break
            crossed.append(self.episode.ticks[self.cursor])
            self.cursor += 1
        return crossed

    def advance(self, delta_seconds: float) -> list[ReplayTick]:
        """Advances by wall-clock time when playing; a no-op when paused."""
        if not self.playing or self.finished:
            return []
        dt = float(self.episode.header.simulation_dt) or (1.0 / 60.0)
        self._accumulator += max(0.0, float(delta_seconds)) * self.speed
        count = int(self._accumulator / dt)
        if count <= 0:
            return []
        self._accumulator -= count * dt
        crossed = self.step(count)
        if self.finished:
            self.playing = False
        return crossed

    def seek(self, tick: int) -> int:
        self.cursor = max(0, min(int(tick), len(self.episode.ticks)))
        self._accumulator = 0.0
        return self.cursor

    def seek_time(self, seconds: float) -> int:
        dt = float(self.episode.header.simulation_dt) or (1.0 / 60.0)
        return self.seek(int(max(0.0, float(seconds)) / dt))

    # -- events ------------------------------------------------------------

    def next_event(self, kinds: Sequence[str] | None = IMPORTANT_EVENT_KINDS) -> ReplayEvent | None:
        for event in self.episode.events_of(kinds):
            if event.tick > self.cursor:
                return event
        return None

    def previous_event(
        self, kinds: Sequence[str] | None = IMPORTANT_EVENT_KINDS
    ) -> ReplayEvent | None:
        found: ReplayEvent | None = None
        for event in self.episode.events_of(kinds):
            if event.tick < self.cursor:
                found = event
            else:
                break
        return found

    def jump_to_next_event(
        self, kinds: Sequence[str] | None = IMPORTANT_EVENT_KINDS
    ) -> ReplayEvent | None:
        event = self.next_event(kinds)
        if event is not None:
            self.seek(event.tick)
        return event

    def jump_to_previous_event(
        self, kinds: Sequence[str] | None = IMPORTANT_EVENT_KINDS
    ) -> ReplayEvent | None:
        event = self.previous_event(kinds)
        if event is not None:
            self.seek(event.tick)
        return event

    def events_at(self, tick: int | None = None) -> list[ReplayEvent]:
        target = self.cursor if tick is None else int(tick)
        return [event for event in self.episode.events if event.tick == target]

    # -- presentation ------------------------------------------------------

    def current_tick(self) -> ReplayTick | None:
        if self.finished or not self.episode.ticks:
            return None
        return self.episode.ticks[self.cursor]

    def progress(self) -> float:
        total = len(self.episode.ticks)
        return (self.cursor / total) if total else 0.0

    def state(self) -> dict[str, Any]:
        """Flat dictionary for a UI. Presentation only; never fed back in."""
        tick = self.current_tick()
        return {
            "tick": self.cursor,
            "ticks_total": len(self.episode.ticks),
            "time": self.episode.time_of(self.cursor),
            "duration": self.episode.duration,
            "progress": self.progress(),
            "playing": self.playing,
            "speed": self.speed,
            "finished": self.finished,
            "action": list(tick.action) if tick else [],
            "reward": tick.reward if tick else 0.0,
            "observation": list(tick.observation) if (tick and tick.observation) else [],
            "events_here": [event.kind for event in self.events_at()],
            "policy_id": self.episode.header.policy_id,
            "checkpoint": self.episode.header.checkpoint,
            "seed": self.episode.header.seed,
            "map": self.episode.header.map_id,
            "scenario": self.episode.header.scenario,
            "lighting": self.episode.header.lighting,
        }


@dataclass
class DeterminismReport:
    """Result of re-running a replay against a live environment."""

    deterministic: bool
    ticks_compared: int
    first_divergence_tick: int = -1
    reward_mismatch: int = 0
    observation_mismatch: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_determinism(
    episode: ReplayEpisode,
    env_factory: Callable[[], Any],
    reward_tolerance: float = 1e-5,
    observation_tolerance: float = 1e-4,
) -> DeterminismReport:
    """Re-runs the recorded action stream and compares the outcome.

    ``env_factory`` must return an object with Gymnasium-style
    ``reset(seed=...) -> (observation, info)`` and
    ``step(action) -> (observation, reward, terminated, truncated, info)``.
    Taking a factory instead of an engine keeps this function testable
    without Godot, and keeps the check honest: it compares against a real
    environment run, never against the recording itself.

    Observations are only compared when the replay is DETAILED; a LIGHT
    replay can still prove reward-level determinism, which is what catches
    an unseeded RNG.
    """
    env = env_factory()
    report = DeterminismReport(deterministic=True, ticks_compared=0)
    try:
        env.reset(seed=episode.header.seed)
        for tick in episode.ticks:
            observation, reward, terminated, truncated, _info = env.step(tick.action)
            report.ticks_compared += 1
            diverged = False
            if abs(float(reward) - tick.reward) > reward_tolerance:
                report.reward_mismatch += 1
                diverged = True
            if episode.detailed and tick.observation is not None:
                values = list(observation)
                if len(values) != len(tick.observation):
                    report.observation_mismatch += 1
                    diverged = True
                else:
                    for recorded, live in zip(tick.observation, values):
                        if abs(float(live) - float(recorded)) > observation_tolerance:
                            report.observation_mismatch += 1
                            diverged = True
                            break
            if diverged and report.first_divergence_tick < 0:
                report.first_divergence_tick = tick.tick
                report.deterministic = False
            if bool(terminated or truncated):
                break
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()
    if report.deterministic:
        report.detail = f"{report.ticks_compared} ticks reproduced exactly"
    else:
        report.detail = (
            f"diverged at tick {report.first_divergence_tick} "
            f"({report.reward_mismatch} reward, {report.observation_mismatch} observation mismatches)"
        )
    return report
