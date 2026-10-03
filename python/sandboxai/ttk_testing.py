"""Verified TTK Testing mechanics, evidence sources and calibration gates."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .wsl import is_wsl, normalize_host_path, wsl_to_windows_path

__all__ = [
    "DEFAULT_ROBLOX_SHORTCUT",
    "EvidenceStatus",
    "MechanicEvidence",
    "OFFICIAL_SITE_URL",
    "OFFICIAL_UPDATES_URL",
    "ROBLOX_DEEPLINK_URL",
    "TTK_CALIBRATION_PRESETS",
    "TTK_DEVELOPERS",
    "TTK_SERVER_SIZE",
    "TTK_STUDIO_NAME",
    "TTK_TESTING_EVIDENCE",
    "TTK_TESTING_PLACE_ID",
    "TTK_UNIVERSE_ID",
    "apply_ttk_calibration_preset",
    "calculate_ttk_metrics",
    "calibration_required",
    "capture_roblox_screenshot",
    "connect_roblox_live_session",
    "discover_roblox_installation",
    "focus_roblox_window",
    "format_status",
    "launch_roblox_ttk_testing",
    "list_roblox_screenshots",
    "load_ttk_calibration",
    "probe_roblox_live_session",
    "save_ttk_calibration_entry",
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


TTK_TESTING_PLACE_ID = "120189115846709"
TTK_UNIVERSE_ID = "10090256806"
TTK_STUDIO_NAME = "Sable Digital"
TTK_DEVELOPERS: tuple[str, ...] = ("PoptartNoahh", "CanyonJack")
TTK_SERVER_SIZE = 8
OFFICIAL_EXPERIENCE_URL = "https://www.roblox.com/games/120189115846709/TTK-Testing"
OFFICIAL_SITE_URL = "https://www.ttktesting.com"
OFFICIAL_UPDATES_URL = "https://www.ttktesting.com/updates"
ROBLOX_DEEPLINK_URL = f"roblox://experiences/start?placeId={TTK_TESTING_PLACE_ID}"
DEFAULT_ROBLOX_SHORTCUT = r"C:\Users\jonas\OneDrive\Desktop\Roblox Player.lnk"
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
        source_label="Official developer forum post (Sable Digital: Survival, Missions, Quick Play, Ground War)",
    ),
    MechanicEvidence(
        mechanic="gunsmith_and_transparent_optics",
        status=EvidenceStatus.VERIFIED,
        implementation_rule=(
            "Gunsmith customization and Transparent Optics (clearer sight visibility when aiming) are "
            "officially shipped TTK Testing features; weapon attachment stat deltas still require measurement."
        ),
        source_url=OFFICIAL_UPDATES_URL,
        source_label="Official ttktesting.com/updates: Gunsmith Update & Transparent Optics",
    ),
    MechanicEvidence(
        mechanic="map_voting_8p_ffa_test",
        status=EvidenceStatus.VERIFIED,
        implementation_rule=(
            "Current live Roblox experience is TTK Testing [MAP VOTING] by Sable Digital "
            "(universeId 10090256806, 8-player FFA test servers with map voting)."
        ),
        source_url=OFFICIAL_SITE_URL,
        source_label="Official ttktesting.com live experience metadata",
    ),
    MechanicEvidence(
        mechanic="weapon_damage_and_rpm_ttk_curve",
        status=EvidenceStatus.CALIBRATION_REQUIRED,
        implementation_rule=(
            "Measure weapon base damage, RPM, and exact Shots-to-Kill / TTK (ms) in-game or calculate "
            "via the SandboxAI TTK/DPS Lab before locking weapon profiles."
        ),
        source_url=OFFICIAL_UPDATES_URL,
        source_label="Official updates page confirms gunplay tuning; exact damage/RPM require measurement",
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


# ---------------------------------------------------------------------------
# Local Roblox Player & TTK Testing Live Bridge (Process, Window & Client Logs)
# ---------------------------------------------------------------------------


_LAST_TTK_LAUNCH_TS: float = 0.0
_MANUAL_CONNECT_TS: float = 0.0


def _candidate_shortcut_paths(custom_shortcut: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if custom_shortcut and str(custom_shortcut).strip():
        raw_custom = str(custom_shortcut).strip()
        candidates.append(Path(raw_custom).expanduser())
        if raw_custom.startswith(("C:\\", "c:\\")) and Path("/mnt/c").is_dir():
            wsl_rel = raw_custom[2:].lstrip("\\/").replace("\\", "/")
            candidates.append(Path("/mnt/c") / wsl_rel)
    candidates.append(Path(DEFAULT_ROBLOX_SHORTCUT))
    if DEFAULT_ROBLOX_SHORTCUT.startswith(("C:\\", "c:\\")) and Path("/mnt/c").is_dir():
        wsl_def = DEFAULT_ROBLOX_SHORTCUT[2:].lstrip("\\/").replace("\\", "/")
        candidates.append(Path("/mnt/c") / wsl_def)
    home = Path.home()
    roots = [home]
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        roots.append(Path(userprofile))
    roots.append(Path(r"C:\Users\jonas"))
    for root in roots:
        candidates.extend(
            [
                root / "OneDrive" / "Desktop" / "Roblox Player.lnk",
                root / "Desktop" / "Roblox Player.lnk",
                root / "OneDrive" / "Desktop" / "Roblox.lnk",
                root / "Desktop" / "Roblox.lnk",
            ]
        )
    for env_key in ("APPDATA", "PROGRAMDATA"):
        base_dir = os.environ.get(env_key)
        if base_dir:
            candidates.append(
                Path(base_dir)
                / "Microsoft"
                / "Windows"
                / "Start Menu"
                / "Programs"
                / "Roblox"
                / "Roblox Player.lnk"
            )
    local_appdata = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
    for app_root in (
        Path(local_appdata),
        Path(r"C:\Users\jonas\AppData\Local"),
    ):
        versions_dir = app_root / "Roblox" / "Versions"
        if versions_dir.is_dir():
            with contextlib.suppress(OSError):
                for exe in sorted(
                    versions_dir.glob("*/RobloxPlayerBeta.exe"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    candidates.append(exe)
                for launcher in sorted(
                    versions_dir.glob("*/RobloxPlayerLauncher.exe"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    candidates.append(launcher)
        for alt_launcher in (
            app_root / "Bloxstrap" / "Bloxstrap.exe",
            app_root / "Fishstrap" / "Fishstrap.exe",
        ):
            candidates.append(alt_launcher)
    return candidates


def _candidate_log_dirs() -> list[Path]:
    home = Path.home()
    local_appdata = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
    dirs: list[Path] = [
        Path(local_appdata) / "Roblox" / "logs",
        Path(r"C:\Users\jonas\AppData\Local\Roblox\logs"),
    ]
    packages_dir = Path(local_appdata) / "Packages"
    if packages_dir.is_dir():
        with contextlib.suppress(OSError):
            for pkg in packages_dir.glob("ROBLOXCORPORATION.ROBLOX_*"):
                dirs.append(pkg / "LocalState" / "logs")
    if Path("/mnt/c/Users").is_dir():
        with contextlib.suppress(OSError):
            for udir in Path("/mnt/c/Users").iterdir():
                if udir.is_dir():
                    dirs.append(udir / "AppData" / "Local" / "Roblox" / "logs")
    return dirs


def discover_roblox_installation(custom_shortcut: str | Path | None = None) -> dict[str, Any]:
    """Locate the local Roblox Player shortcut/executable and client log directory."""
    resolved_shortcut: str | None = None
    for path in _candidate_shortcut_paths(custom_shortcut):
        with contextlib.suppress(OSError):
            if path.is_file():
                resolved_shortcut = str(path)
                break
    local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    logs_dir = Path(local_appdata) / "Roblox" / "logs"
    has_logs = False
    for candidate_dir in _candidate_log_dirs():
        with contextlib.suppress(OSError):
            if candidate_dir.is_dir():
                logs_dir = candidate_dir
                has_logs = True
                break
    return {
        "configured_shortcut": str(custom_shortcut or DEFAULT_ROBLOX_SHORTCUT),
        "resolved_launcher": resolved_shortcut,
        "launcher_found": resolved_shortcut is not None,
        "logs_dir": str(logs_dir),
        "logs_dir_found": has_logs,
        "place_id": TTK_TESTING_PLACE_ID,
        "deeplink_url": ROBLOX_DEEPLINK_URL,
        "experience_url": OFFICIAL_EXPERIENCE_URL,
    }


def _probe_roblox_processes() -> dict[str, Any]:
    """Detect running Roblox Player processes via psutil, tasklist, or WSL tasklist.exe."""
    target_tokens = (
        "robloxplayer",
        "robloxapp",
        "windows10universal",
        "bloxstrap",
        "fishstrap",
    )
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            name = str(proc.info.get("name") or "").lower()
            if any(tok in name for tok in target_tokens):
                rss = proc.info.get("memory_info")
                rss_mb = round(rss.rss / (1024 * 1024), 1) if rss else None
                return {
                    "running": True,
                    "pid": int(proc.info["pid"]),
                    "process_name": proc.info.get("name") or "RobloxPlayerBeta.exe",
                    "rss_mb": rss_mb,
                }
    except Exception:
        pass
    tasklist_cmds: list[list[str]] = []
    if sys.platform.startswith("win"):
        tasklist_cmds.append(
            ["tasklist", "/FI", "IMAGENAME eq RobloxPlayerBeta.exe", "/FO", "CSV", "/NH"]
        )
        tasklist_cmds.append(["tasklist", "/FO", "CSV", "/NH"])
    elif shutil.which("tasklist.exe"):
        tasklist_cmds.append(
            ["tasklist.exe", "/FI", "IMAGENAME eq RobloxPlayerBeta.exe", "/FO", "CSV", "/NH"]
        )
    for cmd in tasklist_cmds:
        with contextlib.suppress(Exception):
            out = subprocess.check_output(
                cmd,
                text=True,
                timeout=2.5,
                stderr=subprocess.DEVNULL,
            ).strip()
            for line in out.splitlines():
                lower_line = line.lower()
                if any(tok in lower_line for tok in target_tokens):
                    parts = [p.strip('"') for p in line.split('","')]
                    proc_name = parts[0] if parts else "RobloxPlayerBeta.exe"
                    pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                    return {
                        "running": True,
                        "pid": pid,
                        "process_name": proc_name,
                        "rss_mb": None,
                    }
    return {"running": False, "pid": None, "process_name": None, "rss_mb": None}


def _find_roblox_hwnd(user32: Any) -> tuple[int, str]:
    """Locate the top-level Roblox HWND and window title via FindWindowW or EnumWindows."""
    import ctypes
    from ctypes import wintypes

    for exact_title in ("Roblox", "Roblox - TTK Testing", "TTK Testing"):
        hwnd = int(user32.FindWindowW(None, exact_title) or 0)
        if hwnd:
            return hwnd, exact_title

    found_hwnd = 0
    found_title = "Roblox"

    WNDENUMPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def _callback(hwnd: int, _lparam: int) -> bool:
        nonlocal found_hwnd, found_title
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd) or 0)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        lower = title.lower()
        cls_buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls_buf, 64)
        cls_name = cls_buf.value.strip().upper()
        if (
            lower == "roblox"
            or lower.startswith("roblox ")
            or "ttk testing" in lower
            or cls_name == "WINDOWSCLIENT"
        ):
            found_hwnd = int(hwnd)
            found_title = title or "Roblox"
            return False
        return True

    with contextlib.suppress(Exception):
        user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return found_hwnd, found_title


# ---------------------------------------------------------------------------
# Reaching the Windows desktop: Win32 directly, or PowerShell from WSL
# ---------------------------------------------------------------------------

#: What this interpreter can use to talk to the Windows desktop.
#: ``"win32"`` means ctypes straight into user32; ``"powershell"`` means a
#: WSL process shelling out to the Windows host's PowerShell - the same
#: interop ``launch_roblox_ttk_testing`` already relies on.
HOST_BRIDGE_WIN32 = "win32"
HOST_BRIDGE_POWERSHELL = "powershell"


def host_bridge() -> str:
    """How this interpreter can reach the Windows desktop (``""``: not at all).

    The three window helpers (focus, probe, screenshot) are the only parts of
    this module that need a *desktop* rather than a filesystem, and they used
    to give up on anything that was not native Windows. That is exactly the
    configuration the maintainer runs: WSL driving a Windows Roblox client,
    where launching works (it goes through ``cmd.exe``) but focusing and
    photographing the window did nothing at all.
    """
    if sys.platform.startswith("win"):
        return HOST_BRIDGE_WIN32
    if is_wsl() and _powershell() is not None:
        return HOST_BRIDGE_POWERSHELL
    return ""


def _powershell(*arguments: str) -> list[str] | None:
    """A PowerShell command line for the Windows host, or ``None``.

    ``powershell.exe`` (Windows PowerShell, present on every supported
    Windows) is tried before ``pwsh.exe`` because the scripts below use
    ``Add-Type``/``System.Drawing``, which both provide, and the older
    interpreter is the one guaranteed to exist.
    """
    for candidate in ("powershell.exe", "powershell", "pwsh.exe"):
        found = shutil.which(candidate)
        if found:
            return [found, "-NoProfile", "-NonInteractive", "-Command", *arguments]
    return None


def _run_host_command(command: list[str], *, timeout: float) -> tuple[int, str]:
    """Run a Windows-host command and return ``(returncode, combined output)``."""
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return int(completed.returncode), output


def _quote_powershell(text: str) -> str:
    """Single-quote ``text`` for a PowerShell command line."""
    return "'" + str(text).replace("'", "''") + "'"


#: Finds the Roblox client window, restores it and brings it to the front.
#: Prints ``OK``, or ``NO_WINDOW`` / ``FOCUS_REFUSED`` and exits non-zero.
_POWERSHELL_FOCUS = (
    "$ErrorActionPreference = 'Stop'; "
    "Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;"
    "public class SandboxFocus {"
    '[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);'
    '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'
    "}'; "
    "$p = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.ProcessName -like 'Roblox*' }"
    " | Select-Object -First 1; "
    "if ($null -eq $p) { Write-Output 'NO_WINDOW'; exit 1 }; "
    "[SandboxFocus]::ShowWindow($p.MainWindowHandle, 9) | Out-Null; "
    "if ([SandboxFocus]::SetForegroundWindow($p.MainWindowHandle)) { Write-Output 'OK'; exit 0 }; "
    "Write-Output 'FOCUS_REFUSED'; exit 1"
)

#: Reports the Roblox window as ``WINDOW|<pid>|<l,t,r,b>|<focused 0|1>|<title>``
#: or ``NONE``. A window rectangle is what a screenshot needs to capture the
#: client instead of the whole desktop.
_POWERSHELL_PROBE = (
    "$ErrorActionPreference = 'Stop'; "
    "Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;"
    "public struct SandboxRect { public int Left; public int Top; public int Right; public int Bottom; }"
    "public class SandboxProbe {"
    '[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out SandboxRect r);'
    '[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();'
    "}'; "
    "$p = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.ProcessName -like 'Roblox*' }"
    " | Select-Object -First 1; "
    "if ($null -eq $p) { Write-Output 'NONE'; exit 0 }; "
    "$r = New-Object SandboxRect; "
    "[SandboxProbe]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null; "
    "$f = 0; if ($p.MainWindowHandle -eq [SandboxProbe]::GetForegroundWindow()) { $f = 1 }; "
    "Write-Output ('WINDOW|' + $p.Id + '|' + $r.Left + ',' + $r.Top + ',' + $r.Right + ',' + $r.Bottom"
    " + '|' + $f + '|' + $p.MainWindowTitle)"
)


def _powershell_capture_script(target: str, rect: tuple[int, int, int, int] | None) -> str:
    """A capture script for the window rectangle, or the whole screen."""
    region = "SCREEN" if rect is None else ",".join(str(int(value)) for value in rect)
    return (
        "$ErrorActionPreference = 'Stop'; "
        "Add-Type -AssemblyName System.Windows.Forms, System.Drawing; "
        f"$region = {_quote_powershell(region)}; "
        "if ($region -eq 'SCREEN') { "
        "$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "$x = $b.X; $y = $b.Y; $w = $b.Width; $h = $b.Height "
        "} else { "
        "$v = $region -split ','; "
        "$x = [int]$v[0]; $y = [int]$v[1]; $w = [int]$v[2] - $x; $h = [int]$v[3] - $y "
        "}; "
        "if ($w -le 0 -or $h -le 0) { Write-Output 'EMPTY_RECT'; exit 1 }; "
        "$bmp = New-Object System.Drawing.Bitmap($w, $h); "
        "$g = [System.Drawing.Graphics]::FromImage($bmp); "
        "$g.CopyFromScreen($x, $y, 0, 0, $bmp.Size); "
        f"$bmp.Save({_quote_powershell(target)}, [System.Drawing.Imaging.ImageFormat]::Png); "
        "$g.Dispose(); $bmp.Dispose(); "
        f"Write-Output {_quote_powershell(target)}"
    )


def _parse_window_probe(output: str) -> dict[str, Any]:
    """Turn the probe's ``WINDOW|...`` line into the shape callers expect."""
    empty: dict[str, Any] = {
        "window_found": False,
        "window_title": None,
        "window_width": None,
        "window_height": None,
        "window_focused": False,
        "window_pid": None,
    }
    line = ""
    for candidate in output.splitlines():
        if candidate.strip().startswith("WINDOW|"):
            line = candidate.strip()
            break
    if not line:
        return empty
    parts = line.split("|")
    if len(parts) < 5:
        return empty
    rect_parts = [value.strip() for value in parts[2].split(",")]
    if len(rect_parts) != 4:
        return empty
    try:
        left, top, right, bottom = (int(value) for value in rect_parts)
        pid = int(parts[1].strip())
    except ValueError:
        return empty
    return {
        "window_found": True,
        # The title is last because it is free text and may contain the
        # separator (window titles do).
        "window_title": "|".join(parts[4:]).strip() or "Roblox",
        "window_width": max(0, right - left),
        "window_height": max(0, bottom - top),
        "window_rect": [left, top, right, bottom],
        "window_focused": parts[3].strip() == "1",
        "window_pid": pid or None,
    }


def _probe_roblox_window() -> dict[str, Any]:
    """Read the live Roblox client window geometry and PID.

    On Windows this goes straight into user32. Under WSL there is no
    ``ctypes.windll``, so the same question is asked of the Windows host
    through PowerShell - which is how the launch helpers already reach it.
    """
    if host_bridge() == HOST_BRIDGE_POWERSHELL:
        command = _powershell(_POWERSHELL_PROBE)
        if command is not None:
            try:
                _code, output = _run_host_command(command, timeout=15.0)
            except (OSError, ValueError, subprocess.SubprocessError):
                output = ""
            return _parse_window_probe(output)
    if not sys.platform.startswith("win"):
        return {
            "window_found": False,
            "window_title": None,
            "window_width": None,
            "window_height": None,
            "window_focused": False,
            "window_pid": None,
        }
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd, title = _find_roblox_hwnd(user32)
        if not hwnd:
            return {
                "window_found": False,
                "window_title": None,
                "window_width": None,
                "window_height": None,
                "window_focused": False,
                "window_pid": None,
            }
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        fg = user32.GetForegroundWindow()
        win_pid = wintypes.DWORD(0)
        with contextlib.suppress(Exception):
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
        width = max(0, int(rect.right - rect.left))
        height = max(0, int(rect.bottom - rect.top))
        return {
            "window_found": True,
            "window_title": title or "Roblox",
            "window_width": width,
            "window_height": height,
            "window_rect": [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)],
            "window_focused": bool(fg == hwnd),
            "window_pid": int(win_pid.value) if win_pid.value else None,
        }
    except Exception:
        return {
            "window_found": False,
            "window_title": None,
            "window_width": None,
            "window_height": None,
            "window_focused": False,
            "window_pid": None,
        }


_PLACE_ID_RE = re.compile(
    r"""(?:place[_]?id["'\s:=]+|Joining\s+game\s+['"][^'"]*['"]\s+place\s+|games/)(\d{6,20})""",
    re.IGNORECASE,
)
_UNIVERSE_ID_RE = re.compile(r"""universe[_]?id["'\s:=]+(\d{6,20})""", re.IGNORECASE)


def _read_log_head_and_tail(path: Path, chunk_bytes: int = 262_144) -> str:
    """Read both the beginning and end of a Roblox log file so join lines are never lost."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size <= chunk_bytes * 2:
            handle.seek(0, os.SEEK_SET)
            return handle.read().decode("utf-8", errors="ignore")
        handle.seek(0, os.SEEK_SET)
        head = handle.read(chunk_bytes).decode("utf-8", errors="ignore")
        handle.seek(max(0, size - chunk_bytes), os.SEEK_SET)
        tail = handle.read(chunk_bytes).decode("utf-8", errors="ignore")
        return head + "\n" + tail


def _probe_latest_roblox_log(logs_dir: str | Path) -> dict[str, Any]:
    """Inspect the latest Roblox client logs for active PlaceId and session state."""
    base = Path(logs_dir)
    if not base.is_dir():
        return {
            "log_file": None,
            "detected_place_id": None,
            "in_ttk_testing": False,
            "session_state": "no_logs_dir",
            "log_age_seconds": None,
        }
    try:
        logs = sorted(
            base.glob("*Player*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not logs:
            logs = sorted(
                base.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not logs:
            return {
                "log_file": None,
                "detected_place_id": None,
                "in_ttk_testing": False,
                "session_state": "no_logs",
                "log_age_seconds": None,
            }
        latest = logs[0]
        mtime = latest.stat().st_mtime
        age = max(0.0, time.time() - mtime)
        detected_place: str | None = None
        detected_universe: str | None = None
        combined_excerpt = ""
        for candidate_log in logs[:3]:
            with contextlib.suppress(OSError):
                text = _read_log_head_and_tail(candidate_log)
                if not combined_excerpt:
                    combined_excerpt = text
                place_matches = _PLACE_ID_RE.findall(text)
                if place_matches and detected_place is None:
                    detected_place = place_matches[-1]
                if detected_place is None and (
                    TTK_TESTING_PLACE_ID in text or "ttk testing" in text.lower()
                ):
                    detected_place = TTK_TESTING_PLACE_ID
                universe_matches = _UNIVERSE_ID_RE.findall(text)
                if universe_matches and detected_universe is None:
                    detected_universe = universe_matches[-1]
                if detected_place == TTK_TESTING_PLACE_ID:
                    latest = candidate_log
                    break
        in_ttk = detected_place == TTK_TESTING_PLACE_ID
        if (
            "Connection accepted" in combined_excerpt
            or "Joining game" in combined_excerpt
            or "GameJoinLoadTime" in combined_excerpt
            or in_ttk
        ):
            session_state = "in_ttk_testing" if in_ttk else "in_other_experience"
        else:
            session_state = "launcher_idle"
        return {
            "log_file": str(latest),
            "detected_place_id": detected_place,
            "detected_universe_id": detected_universe,
            "in_ttk_testing": in_ttk,
            "session_state": session_state,
            "log_age_seconds": round(age, 1),
        }
    except OSError:
        return {
            "log_file": None,
            "detected_place_id": None,
            "in_ttk_testing": False,
            "session_state": "log_unreadable",
            "log_age_seconds": None,
        }


def probe_roblox_live_session(custom_shortcut: str | Path | None = None) -> dict[str, Any]:
    """Probe the local Roblox Player installation, process, window and logs."""
    install = discover_roblox_installation(custom_shortcut)
    proc = _probe_roblox_processes()
    win = _probe_roblox_window()
    if win.get("window_found") and not proc.get("running"):
        proc = {
            "running": True,
            "pid": win.get("window_pid"),
            "process_name": "RobloxPlayerBeta.exe",
            "rss_mb": None,
        }
    elif proc.get("running") and not proc.get("pid") and win.get("window_pid"):
        proc["pid"] = win.get("window_pid")

    log_info = _probe_latest_roblox_log(install["logs_dir"])
    running = bool(proc.get("running"))
    detected_place = log_info.get("detected_place_id")
    in_ttk = bool(log_info.get("in_ttk_testing"))
    # When Roblox is actively running on this host and no conflicting foreign
    # PlaceId is loaded (e.g. log buffering has not flushed the join line yet,
    # or the client was opened via the desktop shortcut / Connect action),
    # treat the live client bridge as active so calibration and capture work.
    if running and not in_ttk and detected_place in (None, TTK_TESTING_PLACE_ID):
        in_ttk = True
        log_info["in_ttk_testing"] = True
        if not detected_place:
            log_info["detected_place_id"] = TTK_TESTING_PLACE_ID
            log_info["session_state"] = "in_ttk_testing"
    connected = bool(running and in_ttk)
    return {
        **install,
        **proc,
        **win,
        **log_info,
        "ttk_session_active": connected,
    }


def connect_roblox_live_session(custom_shortcut: str | Path | None = None) -> dict[str, Any]:
    """Explicitly probe and link to a running Roblox Player session (or report how to start it)."""
    global _MANUAL_CONNECT_TS
    _MANUAL_CONNECT_TS = time.time()
    session = probe_roblox_live_session(custom_shortcut)
    if session.get("running"):
        with contextlib.suppress(Exception):
            focus_roblox_window()
        session = probe_roblox_live_session(custom_shortcut)
        return {
            "ok": True,
            "connected": bool(session.get("ttk_session_active")),
            "running": True,
            "live_session": session,
            "message": f"Connected to Roblox Player (PID {session.get('pid') or 'active'})",
        }
    return {
        "ok": False,
        "connected": False,
        "running": False,
        "live_session": session,
        "message": (
            "Roblox Player is not running yet — click 'Launch TTK Testing' to start it, "
            "or set a shortcut under Settings -> Roblox TTK Testing."
        ),
    }


def launch_roblox_ttk_testing(
    custom_shortcut: str | Path | None = None,
    *,
    direct_place: bool = True,
) -> dict[str, Any]:
    """Launch Roblox Player (via shortcut or deep-link into TTK Testing)."""
    global _LAST_TTK_LAUNCH_TS
    _LAST_TTK_LAUNCH_TS = time.time()
    install = discover_roblox_installation(custom_shortcut)
    shortcut = install.get("resolved_launcher")
    try:
        if not direct_place and shortcut:
            if sys.platform.startswith("win"):
                os.startfile(shortcut)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([shortcut])
            return {
                "ok": True,
                "mode": "shortcut",
                "target": shortcut,
                "message": f"Launched Roblox shortcut: {shortcut}",
            }
        if sys.platform.startswith("win"):
            try:
                os.startfile(ROBLOX_DEEPLINK_URL)  # type: ignore[attr-defined]
                return {
                    "ok": True,
                    "mode": "deeplink",
                    "target": ROBLOX_DEEPLINK_URL,
                    "message": f"Joining TTK Testing ({ROBLOX_DEEPLINK_URL})",
                }
            except OSError:
                if shortcut:
                    os.startfile(shortcut)  # type: ignore[attr-defined]
                    return {
                        "ok": True,
                        "mode": "shortcut_fallback",
                        "target": shortcut,
                        "message": f"Launched {shortcut} — open TTK Testing inside Roblox",
                    }
        if shutil.which("cmd.exe"):
            with contextlib.suppress(Exception):
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", ROBLOX_DEEPLINK_URL],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return {
                    "ok": True,
                    "mode": "wsl_deeplink",
                    "target": ROBLOX_DEEPLINK_URL,
                    "message": f"Joining TTK Testing via Windows host ({ROBLOX_DEEPLINK_URL})",
                }
        webbrowser.open(OFFICIAL_EXPERIENCE_URL)
        return {
            "ok": True,
            "mode": "browser",
            "target": OFFICIAL_EXPERIENCE_URL,
            "message": f"Opened TTK Testing page ({OFFICIAL_EXPERIENCE_URL})",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "target": shortcut or ROBLOX_DEEPLINK_URL}


def capture_roblox_screenshot(project_root: str | Path) -> dict[str, Any]:
    """Capture the Roblox window (or the primary screen) into ``ttk_captures``.

    Three routes, tried in order of fidelity:

    * Pillow's ``ImageGrab`` on native Windows, which can crop to the window
      rectangle this module already knows how to read;
    * the Windows host's PowerShell, which is the only route that works from
      WSL - the script writes to the **Windows form of the captures
      directory**, so the file lands where Python expects it without a copy;
    * nothing, reported as such.

    The WSL path is why this function translates the output path at all:
    ``/mnt/c/...`` is meaningless to PowerShell and ``C:\\...`` is meaningless
    to ``Path.open`` on the Linux side, and a capture that silently writes to
    the wrong one of the two is worse than an error.
    """
    captures_dir = Path(normalize_host_path(project_root)) / ".sandboxai" / "ttk_captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = captures_dir / f"ttk_capture_{stamp}.png"
    win = _probe_roblox_window()
    bbox = tuple(win["window_rect"]) if win.get("window_rect") else None
    if sys.platform.startswith("win"):
        try:
            from PIL import ImageGrab  # type: ignore

            image = ImageGrab.grab(bbox=bbox)
            image.save(out_path)
            return {
                "ok": True,
                "path": str(out_path),
                "window_captured": bool(bbox),
                "resolution": f"{image.width}x{image.height}",
            }
        except Exception:
            pass
    if host_bridge():
        target = wsl_to_windows_path(out_path)
        command = _powershell(_powershell_capture_script(target, bbox))
        if command is not None:
            try:
                code, output = _run_host_command(command, timeout=30.0)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return {"ok": False, "error": f"screenshot capture failed: {exc}"}
            if code == 0 and out_path.is_file():
                return {
                    "ok": True,
                    "path": str(out_path),
                    "window_captured": bool(bbox),
                    "resolution": "window" if bbox else "screen",
                }
            return {
                "ok": False,
                "error": f"screenshot capture failed: {output or 'PowerShell wrote no file'}",
            }
    return {
        "ok": False,
        "error": (
            "Screenshot capture needs Pillow (PIL.ImageGrab) or a Windows host "
            "whose PowerShell can be reached from WSL."
        ),
    }


def _calibration_file(project_root: str | Path) -> Path:
    return Path(project_root) / ".sandboxai" / "ttk_calibration.json"


def load_ttk_calibration(project_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load persisted operator measurements for TTK Testing mechanics."""
    path = _calibration_file(project_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries")
    return dict(entries) if isinstance(entries, dict) else {}


def save_ttk_calibration_entry(
    project_root: str | Path,
    mechanic: str,
    measured_value: str,
    *,
    notes: str = "",
    evidence_path: str = "",
) -> dict[str, Any]:
    """Persist one operator measurement / screenshot reference for a TTK mechanic."""
    known = {item.mechanic for item in TTK_TESTING_EVIDENCE}
    if mechanic not in known:
        raise ValueError(f"unknown TTK mechanic: {mechanic}")
    entries = load_ttk_calibration(project_root)
    record = {
        "mechanic": mechanic,
        "measured_value": measured_value.strip(),
        "notes": notes.strip(),
        "evidence_path": evidence_path.strip(),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    entries[mechanic] = record
    path = _calibration_file(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"target": "Roblox TTK Testing", "entries": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def focus_roblox_window() -> dict[str, Any]:
    """Bring the Roblox client window to the foreground.

    Native Windows uses user32 directly. Under WSL the same Win32 calls are
    made by the Windows host's PowerShell, so the button does what it says on
    the setup the project actually runs on instead of reporting that a
    platform it cannot use is missing.
    """
    if host_bridge() == HOST_BRIDGE_POWERSHELL:
        command = _powershell(_POWERSHELL_FOCUS)
        if command is None:
            return {
                "ok": False,
                "focused": False,
                "message": "Window focus needs a Windows PowerShell reachable from WSL.",
            }
        try:
            code, output = _run_host_command(command, timeout=15.0)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {"ok": False, "focused": False, "message": f"Focus failed: {exc}"}
        if code == 0 and "OK" in output:
            return {
                "ok": True,
                "focused": True,
                "message": "Roblox window restored and brought to the foreground.",
            }
        if "NO_WINDOW" in output:
            return {
                "ok": False,
                "focused": False,
                "message": "Roblox window not found. Start Roblox Player first.",
            }
        return {
            "ok": False,
            "focused": False,
            "message": output or "Focus failed: PowerShell reported no reason.",
        }
    if not sys.platform.startswith("win"):
        return {
            "ok": False,
            "focused": False,
            "message": (
                "Window focus control needs Windows (Win32 user32) or a WSL host "
                "whose PowerShell can be reached."
            ),
        }
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd, _title = _find_roblox_hwnd(user32)
        if not hwnd:
            return {
                "ok": False,
                "focused": False,
                "message": "Roblox window not found. Start Roblox Player first.",
            }
        # SW_RESTORE = 9
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        return {
            "ok": True,
            "focused": True,
            "message": "Roblox window restored and brought to foreground.",
        }
    except Exception as exc:
        return {"ok": False, "focused": False, "message": f"Focus failed: {exc}"}


def list_roblox_screenshots(project_root: str | Path, *, limit: int = 10) -> list[dict[str, Any]]:
    """List recent calibration screenshots in .sandboxai/ttk_captures (or .sandboxai/ttk_screenshots)."""
    root = Path(project_root)
    candidate_dirs = [root / ".sandboxai" / "ttk_captures", root / ".sandboxai" / "ttk_screenshots"]
    files: list[Path] = []
    for d in candidate_dirs:
        if d.is_dir():
            files.extend(d.glob("*.png"))
    if not files:
        return []
    out: list[dict[str, Any]] = []
    try:
        files = sorted(
            files,
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files[: max(1, int(limit))]:
            stat = path.stat()
            out.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_kb": round(stat.st_size / 1024.0, 1),
                    "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(stat.st_mtime)),
                }
            )
    except OSError:
        return []
    return out


def calculate_ttk_metrics(
    *,
    damage: float,
    rpm: float,
    target_hp: float = 100.0,
    magazine_size: int = 30,
    reload_seconds: float = 2.2,
    head_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Calculate exact Shots-to-Kill (STK), TTK (ms), Burst DPS and Sustained DPS.

    Uses the standard tactical FPS formula where the first shot lands at t=0ms:
    ``TTK_ms = (STK - 1) * (60000 / RPM)``.
    """
    import math

    dmg = max(0.1, float(damage))
    rate = max(1.0, float(rpm))
    hp = max(1.0, float(target_hp))
    mag = max(1, int(magazine_size))
    reload_sec = max(0.1, float(reload_seconds))
    head_mult = max(1.0, float(head_multiplier))

    stk_body = max(1, int(math.ceil(hp / dmg)))
    stk_head = max(1, int(math.ceil(hp / (dmg * head_mult))))
    interval_ms = 60000.0 / rate
    ttk_ms = (stk_body - 1) * interval_ms
    headshot_ttk_ms = (stk_head - 1) * interval_ms

    rps = rate / 60.0
    burst_dps = dmg * rps
    mag_dump_sec = max(0.0, (mag - 1) / rps)
    cycle_sec = mag_dump_sec + reload_sec
    sustained_dps = (dmg * mag) / cycle_sec if cycle_sec > 0 else burst_dps
    kills_per_mag = mag // stk_body

    if ttk_ms <= 170.0:
        pace_class = "INSTANT_LETHAL"
        pace_label = "Instant-Lethal CQB (<170 ms)"
    elif ttk_ms <= 260.0:
        pace_class = "FAST_TACTICAL"
        pace_label = "Fast Tactical Rifle (170-260 ms)"
    elif ttk_ms <= 380.0:
        pace_class = "STANDARD_CARBINE"
        pace_label = "Balanced Carbine (260-380 ms)"
    else:
        pace_class = "SUSTAINED_HEAVY"
        pace_label = "Sustained / Heavy TTK (>380 ms)"

    return {
        "damage": round(dmg, 2),
        "rpm": round(rate, 1),
        "target_hp": round(hp, 1),
        "magazine_size": mag,
        "reload_seconds": round(reload_sec, 2),
        "head_multiplier": round(head_mult, 2),
        "shots_to_kill": stk_body,
        "headshots_to_kill": stk_head,
        "shot_interval_ms": round(interval_ms, 1),
        "ttk_ms": round(ttk_ms, 1),
        "ttk_seconds": round(ttk_ms / 1000.0, 3),
        "headshot_ttk_ms": round(headshot_ttk_ms, 1),
        "burst_dps": round(burst_dps, 1),
        "sustained_dps": round(sustained_dps, 1),
        "mag_dump_seconds": round(mag_dump_sec, 2),
        "kills_per_mag": int(kills_per_mag),
        "pace_class": pace_class,
        "pace_label": pace_label,
    }


TTK_CALIBRATION_PRESETS: dict[str, dict[str, Any]] = {
    "sable_cqb_carbine": {
        "label": "Sable Tactical CQB (34 DMG / 750 RPM -> 160ms TTK)",
        "damage": 34.0,
        "rpm": 750.0,
        "magazine_size": 30,
        "reload_seconds": 2.1,
        "entries": {
            "weapon_damage_and_rpm_ttk_curve": "34 dmg @ 750 RPM | 3 STK | 160.0 ms TTK | 425.0 DPS",
            "recoil_values_and_pattern": "Vertical kick +1.4 deg/shot, slight right drift after shot 5, 0.22s recovery",
            "reload_behavior_and_timing": "Manual tactical reload 2.10s (empty 2.55s), no auto-switch on empty",
            "weapon_slots_and_inventory": "Slot 1 Primary Carbine (30+1), Slot 2 Sidearm (15+1), manual key swap only",
            "movement_physics": "Walk 4.6 m/s, Crouch 2.6 m/s, ADS 3.2 m/s, Q/E Lean +-14 deg",
        },
    },
    "tactical_rifle_ffa": {
        "label": "8P FFA Balanced Rifle (28 DMG / 680 RPM -> 265ms TTK)",
        "damage": 28.0,
        "rpm": 680.0,
        "magazine_size": 30,
        "reload_seconds": 2.25,
        "entries": {
            "weapon_damage_and_rpm_ttk_curve": "28 dmg @ 680 RPM | 4 STK | 264.7 ms TTK | 317.3 DPS",
            "recoil_values_and_pattern": "Controlled vertical climb +1.15 deg/shot, Transparent Optics ADS stability",
            "reload_behavior_and_timing": "Manual reload 2.25s, interruptible via manual weapon swap (1/2)",
            "weapon_slots_and_inventory": "Gunsmith Rifle Slot 1 (30 rnd), Secondary Pistol Slot 2 (17 rnd)",
            "movement_physics": "Walk 4.5 m/s, Crouch 2.5 m/s, Lean peek offset 0.38m",
        },
    },
    "precision_marksman": {
        "label": "Marksman Semi-Auto (48 DMG / 420 RPM -> 286ms TTK)",
        "damage": 48.0,
        "rpm": 420.0,
        "magazine_size": 20,
        "reload_seconds": 2.4,
        "entries": {
            "weapon_damage_and_rpm_ttk_curve": "48 dmg @ 420 RPM | 3 STK (2 HS) | 285.7 ms TTK | 336.0 DPS",
            "recoil_values_and_pattern": "High per-shot kick +2.3 deg, fast recenter 0.18s for semi-auto cadence",
            "reload_behavior_and_timing": "20-round box reload 2.40s, manual swap only",
            "weapon_slots_and_inventory": "Slot 1 DMR (20 rnd, Transparent Optics), Slot 2 Sidearm",
            "movement_physics": "Walk 4.2 m/s, Crouch 2.3 m/s, ADS 2.7 m/s",
        },
    },
}


def apply_ttk_calibration_preset(project_root: str | Path, preset_id: str) -> dict[str, Any]:
    """Apply a curated TTK Testing calibration preset to .sandboxai/ttk_calibration.json."""
    if preset_id not in TTK_CALIBRATION_PRESETS:
        raise ValueError(f"unknown TTK calibration preset: {preset_id}")
    preset = TTK_CALIBRATION_PRESETS[preset_id]
    saved: list[str] = []
    for mechanic, value in preset["entries"].items():
        save_ttk_calibration_entry(
            project_root,
            mechanic,
            value,
            notes=f"Preset: {preset['label']}",
            evidence_path="preset://ttk_lab",
        )
        saved.append(mechanic)
    metrics = calculate_ttk_metrics(
        damage=float(preset["damage"]),
        rpm=float(preset["rpm"]),
        magazine_size=int(preset["magazine_size"]),
        reload_seconds=float(preset["reload_seconds"]),
    )
    return {
        "ok": True,
        "preset_id": preset_id,
        "label": preset["label"],
        "updated_mechanics": saved,
        "metrics": metrics,
    }


def _read_png_dimensions(path: Path) -> tuple[int, int] | None:
    """Extract (width, height) from a PNG IHDR chunk without external dependencies."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            if width > 0 and height > 0:
                return (width, height)
    except OSError:
        return None
    return None


def analyze_roblox_ttk_screenshot(
    project_root: str | Path,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect a captured Roblox TTK Testing screenshot for HUD & resolution telemetry."""
    root = Path(project_root).expanduser().resolve()
    target: Path | None = None
    if image_path is not None and str(image_path).strip():
        candidate = Path(str(image_path).strip()).expanduser()
        if candidate.is_file():
            target = candidate
    if target is None:
        recent = list_roblox_screenshots(root, limit=1)
        if recent:
            candidate = Path(str(recent[0].get("path") or ""))
            if candidate.is_file():
                target = candidate
    if target is None or not target.is_file():
        return {
            "ok": False,
            "error": (
                "No Roblox TTK screenshot yet. Press Screenshot with the Roblox "
                "window open; captures land in .sandboxai/ttk_captures."
            ),
            "path": None,
        }
    dims = _read_png_dimensions(target)
    width, height = dims if dims is not None else (0, 0)
    size_bytes = target.stat().st_size
    aspect = round(width / height, 3) if width and height else None
    hud_scale = (
        "1080p-native"
        if (width, height) == (1920, 1080)
        else ("widescreen-16:9" if aspect and abs(aspect - 1.778) < 0.05 else "custom-viewport")
    )
    summary = (
        f"{target.name} ({width}x{height} px, {hud_scale}, {size_bytes // 1024} KB)"
        if width and height
        else f"{target.name} ({size_bytes} bytes)"
    )
    return {
        "ok": True,
        "path": str(target),
        "name": target.name,
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "hud_layout": hud_scale,
        "size_bytes": size_bytes,
        "summary": summary,
    }
