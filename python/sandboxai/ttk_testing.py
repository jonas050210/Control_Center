"""Verified TTK Testing mechanics, evidence sources and calibration gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "EvidenceStatus",
    "MechanicEvidence",
    "TTK_TESTING_EVIDENCE",
    "calibration_required",
    "format_status",
    "status_summary",
    "verified_mechanics",
]


class EvidenceStatus(StrEnum):
    """Whether a mechanic may be represented as a current TTK Testing fact."""

    VERIFIED = "verified"
    CALIBRATION_REQUIRED = "calibration_required"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class MechanicEvidence:
    """One source-traceable game mechanic or an explicitly excluded feature."""

    mechanic: str
    status: EvidenceStatus
    implementation_rule: str
    source_url: str
    source_label: str
    notes: str = ""


OFFICIAL_EXPERIENCE_URL = "https://www.roblox.com/games/120189115846709/TTK-Testing"
OFFICIAL_DEVFORUM_URL = "https://devforum.roblox.com/t/ttk-our-very-early-tactical-fps/4664539"
OFFICIAL_WOUND_VIDEO_URL = "https://www.youtube.com/watch?v=fOpQt7dD4Ro"

## This table is deliberately conservative. It records what can be asserted
## from the official page/developer material checked on 2026-10-01, plus the
## user's explicit product decision that regular first-person presentation is
## wanted instead of a helmet-camera mode. A feature that is merely plausible
## in a tactical FPS belongs in CALIBRATION_REQUIRED, never VERIFIED.
TTK_TESTING_EVIDENCE: tuple[MechanicEvidence, ...] = (
    MechanicEvidence(
        mechanic="fire",
        status=EvidenceStatus.VERIFIED,
        implementation_rule="Expose a player-controlled primary-fire action.",
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Fire M1",
    ),
    MechanicEvidence(
        mechanic="aim",
        status=EvidenceStatus.VERIFIED,
        implementation_rule="Expose a player-controlled aim/ADS action; do not invent its numeric modifiers.",
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Aim M2",
    ),
    MechanicEvidence(
        mechanic="crouch",
        status=EvidenceStatus.VERIFIED,
        implementation_rule="Expose a player-controlled crouch action; calibrate speed, height and hitbox effects from evidence.",
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Crouch C",
    ),
    MechanicEvidence(
        mechanic="lean",
        status=EvidenceStatus.VERIFIED,
        implementation_rule="Expose left/right lean as explicit player actions; calibrate pose and collision effects from evidence.",
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Lean left/right Q/E",
    ),
    MechanicEvidence(
        mechanic="manual_weapon_swap",
        status=EvidenceStatus.VERIFIED,
        implementation_rule=(
            "Weapon selection must be an explicit player/policy action. Never auto-switch "
            "because a magazine is empty, an opponent is visible, or a weapon is on cooldown."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Swap weapon Number",
    ),
    MechanicEvidence(
        mechanic="wound_painting_and_bleeding",
        status=EvidenceStatus.VERIFIED,
        implementation_rule=(
            "Wounds/bleeding may be represented visually or logged as observed events, but do not "
            "assign damage-over-time, healing, or death rules until they are measured."
        ),
        source_url=OFFICIAL_WOUND_VIDEO_URL,
        source_label="Official developer video: TTK - wound painting/bleeding",
        notes="The public title confirms the feature family, not its hidden numerical rules.",
    ),
    MechanicEvidence(
        mechanic="pve_pvp_and_door_kicking_direction",
        status=EvidenceStatus.VERIFIED,
        implementation_rule=(
            "Keep PvE/PvP/door-kicking only as supported product directions; do not infer a map, "
            "mission script, AI behavior or breach timing from this high-level statement."
        ),
        source_url=OFFICIAL_DEVFORUM_URL,
        source_label="Official developer forum post",
    ),
    MechanicEvidence(
        mechanic="recoil_values_and_pattern",
        status=EvidenceStatus.CALIBRATION_REQUIRED,
        implementation_rule=(
            "Do not present any recoil curve, recovery rate, bloom angle or spread model as TTK Testing "
            "until screenshots or repeatable manual measurements support it."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official controls page does not specify recoil values",
    ),
    MechanicEvidence(
        mechanic="reload_behavior_and_timing",
        status=EvidenceStatus.CALIBRATION_REQUIRED,
        implementation_rule=(
            "Do not claim automatic/manual reload behavior, reload cancellation or timing without visible evidence."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official controls page does not specify reload behavior",
    ),
    MechanicEvidence(
        mechanic="weapon_slots_and_inventory",
        status=EvidenceStatus.CALIBRATION_REQUIRED,
        implementation_rule=(
            "Do not name or count weapon slots, weapons, magazines, attachments or fire modes without evidence."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official controls page documents swapping, not loadout contents",
    ),
    MechanicEvidence(
        mechanic="movement_physics",
        status=EvidenceStatus.CALIBRATION_REQUIRED,
        implementation_rule=(
            "Do not claim walk, sprint, jump, gravity, acceleration, stance, lean or ADS movement values "
            "without repeatable player-visible measurement."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official controls page does not publish movement physics",
    ),
    MechanicEvidence(
        mechanic="helmet_camera",
        status=EvidenceStatus.EXCLUDED,
        implementation_rule=(
            "Use normal first-person presentation in this project. Do not add a helmet-camera mode or "
            "hide the regular GUI behind one."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official page mentions Helmetcam; excluded by project decision",
        notes="The user explicitly excluded this presentation mode from the SandboxAI target.",
    ),
    MechanicEvidence(
        mechanic="automatic_weapon_switch",
        status=EvidenceStatus.EXCLUDED,
        implementation_rule=(
            "Never implement automatic primary-to-secondary switching. The official control is manual "
            "weapon selection, and no source verifies an auto-switch exception."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="Official experience controls: Swap weapon Number",
    ),
    MechanicEvidence(
        mechanic="invented_finish_or_pressure_reload_drills",
        status=EvidenceStatus.EXCLUDED,
        implementation_rule=(
            "Do not retain or introduce sidearm-finish, pressure-reload, faster-reload, or other named "
            "training drills as TTK Testing mechanics."
        ),
        source_url=OFFICIAL_EXPERIENCE_URL,
        source_label="No official control or feature source supports these drills",
    ),
)


def _with_status(status: EvidenceStatus) -> tuple[MechanicEvidence, ...]:
    return tuple(item for item in TTK_TESTING_EVIDENCE if item.status is status)


def verified_mechanics() -> tuple[MechanicEvidence, ...]:
    """Mechanics that may be described as officially verified current facts."""
    return _with_status(EvidenceStatus.VERIFIED)


def calibration_required() -> tuple[MechanicEvidence, ...]:
    """Mechanics that must remain explicitly uncalibrated until evidence arrives."""
    return _with_status(EvidenceStatus.CALIBRATION_REQUIRED)


def status_summary() -> dict[str, Any]:
    """Stable JSON-ready status used by the CLI and review tooling."""
    groups = {
        status.value: [asdict(item) for item in _with_status(status)] for status in EvidenceStatus
    }
    return {
        "target": "Roblox TTK Testing",
        "evidence_checked_on": "2026-10-01",
        "official_sources": {
            "experience": OFFICIAL_EXPERIENCE_URL,
            "developer_forum": OFFICIAL_DEVFORUM_URL,
            "wound_video": OFFICIAL_WOUND_VIDEO_URL,
        },
        "mechanics": groups,
        "rules": {
            "weapon_switch": "manual_only",
            "helmet_camera": "excluded",
            "unmeasured_values": "must_not_be_presented_as_ttk_testing_facts",
        },
    }


def format_status(summary: dict[str, Any] | None = None) -> str:
    """Human-readable evidence and calibration checklist."""
    payload = status_summary() if summary is None else summary
    lines = [
        f"{payload['target']} evidence status (checked {payload['evidence_checked_on']})",
        "",
    ]
    headings = {
        EvidenceStatus.VERIFIED.value: "Verified mechanics",
        EvidenceStatus.CALIBRATION_REQUIRED.value: "Needs screenshot/manual calibration",
        EvidenceStatus.EXCLUDED.value: "Excluded from this project",
    }
    for status in EvidenceStatus:
        key = status.value
        lines.append(f"{headings[key]}:")
        for item in payload["mechanics"][key]:
            lines.append(f"  - {item['mechanic']}: {item['implementation_rule']}")
        lines.append("")
    lines.append(
        "Rule: weapon switching is manual only; no automatic empty-magazine switch is allowed."
    )
    return "\n".join(lines)
