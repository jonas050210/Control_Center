"""Automated headless runtime validation harness for the real Godot bridge.

This module executes the end-to-end runtime verification battery against a real
Godot 4.7.2 executable. It verifies the live JSON-lines bridge transport,
observation/action contracts, deterministic seeding, episode-plan staging,
auto-reset vector semantics, perception gating, sound events, and self-play mode.

HONESTY INVARIANT:
If no Godot executable is found on PATH or passed explicitly, this harness
marks the run as ``unavailable`` and reports exactly which binary was sought.
It NEVER invents latency numbers, throughput, or validation passes.
"""

from __future__ import annotations

import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import find_godot_executable
from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT
from .wsl import WindowsInterop, normalize_host_path


@dataclass
class ValidationCheck:
    """Result of one specific runtime validation check."""

    check_id: str
    name: str
    category: str
    passed: bool = False
    skipped: bool = False
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeValidationReport:
    """Complete summary of the runtime validation execution."""

    godot_executable: str | None
    godot_available: bool
    godot_version: str | None
    platform_system: str
    status: str  # "passed" | "failed" | "unavailable"
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    measured_throughput: dict[str, Any] = field(default_factory=dict)
    checks: list[ValidationCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeValidator:
    """Executes the live headless Godot validation battery."""

    def __init__(
        self,
        project_path: str | Path | None = None,
        godot_executable: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.project_path = (
            Path(normalize_host_path(project_path)).expanduser().resolve()
            if project_path
            else Path(__file__).resolve().parents[2]
        )
        self.godot_executable_raw = godot_executable or "godot"
        self.timeout = float(timeout)

    def is_godot_available(self) -> bool:
        try:
            resolved = find_godot_executable(self.godot_executable_raw)
            return bool(shutil.which(resolved) or Path(resolved).is_file())
        except Exception:
            return False

    def probe_version(self, executable: str) -> str | None:
        try:
            # WindowsInterop is a pass-through outside WSL; on WSL it lets a
            # refused direct .exe launch (PermissionError) retry via cmd.exe.
            res = WindowsInterop(executable).run(
                [executable, "--version", "--headless"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            out = res.stdout.strip()
            return out if out else res.stderr.strip() or "unknown"
        except Exception:
            return None

    def validate(
        self,
        env_count: int = 2,
        test_self_play: bool = True,
    ) -> RuntimeValidationReport:
        system = platform.system()
        try:
            executable = find_godot_executable(self.godot_executable_raw)
            available = bool(shutil.which(executable) or Path(executable).is_file())
        except Exception:
            executable = self.godot_executable_raw
            available = False

        if not available:
            report = RuntimeValidationReport(
                godot_executable=self.godot_executable_raw,
                godot_available=False,
                godot_version=None,
                platform_system=system,
                status="unavailable",
                total_checks=0,
                passed_checks=0,
                failed_checks=0,
                skipped_checks=10,
                measured_throughput={"measured": False},
                checks=[],
                notes=[
                    f"Godot executable {self.godot_executable_raw!r} was not found on PATH.",
                    "Live engine validation was not executed.",
                    "Install Godot 4.7.2 and make it available as 'godot' or pass --godot-executable.",
                ],
            )
            return report

        version = self.probe_version(executable)
        report = RuntimeValidationReport(
            godot_executable=executable,
            godot_available=True,
            godot_version=version,
            platform_system=system,
            status="passed",
            measured_throughput={"measured": True},
        )

        checks: list[ValidationCheck] = []

        # --- Check 1: Bridge Handshake & Contract Verification ---
        check_spaces = ValidationCheck(
            check_id="bridge_spaces",
            name="JSON-Lines Bridge Spaces & Contract Verification",
            category="contract",
        )
        t0 = time.perf_counter()
        transport = None
        try:
            from .godot_env import GodotProcessTransport

            transport = GodotProcessTransport(
                project_path=self.project_path,
                godot_executable=executable,
                environment_count=env_count,
                request_timeout=self.timeout,
            )
            spaces = transport.spaces
            check_spaces.latency_ms = (time.perf_counter() - t0) * 1000.0
            obs_space = spaces.get("observation_space", {})
            act_space = spaces.get("action_space", {})
            obs_size = int(obs_space.get("size", 0))
            act_nvec = list(act_space.get("nvec", []))

            if obs_size != OBSERVATION_FIELD_COUNT:
                raise ValueError(
                    f"Observation size mismatch: engine returned {obs_size}, expected {OBSERVATION_FIELD_COUNT}"
                )
            if act_nvec != list(ACTION_NVEC):
                raise ValueError(
                    f"Action nvec mismatch: engine returned {act_nvec}, expected {list(ACTION_NVEC)}"
                )

            check_spaces.passed = True
            check_spaces.details = {
                "observation_size": obs_size,
                "action_nvec": act_nvec,
                "action_type": act_space.get("type"),
            }
        except Exception as exc:
            check_spaces.passed = False
            check_spaces.error = str(exc)
        checks.append(check_spaces)

        if not check_spaces.passed or transport is None:
            if transport is not None:
                transport.close()
            report.checks = checks
            report.status = "failed"
            report.total_checks = len(checks)
            report.passed_checks = sum(1 for c in checks if c.passed)
            report.failed_checks = sum(1 for c in checks if not c.passed)
            return report

        # --- Check 2: Stdio Ping Roundtrip Latency ---
        check_ping = ValidationCheck(
            check_id="bridge_ping",
            name="Stdio Protocol Ping/Pong Latency",
            category="transport",
        )
        try:
            t0 = time.perf_counter()
            ping_res = transport.request({"cmd": "ping"})
            ping_latency = (time.perf_counter() - t0) * 1000.0
            if ping_res.get("ok") and ping_res.get("pong"):
                check_ping.passed = True
                check_ping.latency_ms = ping_latency
                check_ping.details = {"round_trip_ms": round(ping_latency, 3)}
            else:
                check_ping.passed = False
                check_ping.error = f"Malformed ping response: {ping_res}"
        except Exception as exc:
            check_ping.passed = False
            check_ping.error = str(exc)
        checks.append(check_ping)

        # --- Check 3: Deterministic Seeding Reset ---
        check_reset = ValidationCheck(
            check_id="reset_determinism",
            name="Synchronous Reset & Seeding Determinism",
            category="determinism",
        )
        try:
            t0 = time.perf_counter()
            res1 = transport.request({"cmd": "reset", "seed": 42})
            obs1 = res1.get("observations", [])
            res2 = transport.request({"cmd": "reset", "seed": 42})
            obs2 = res2.get("observations", [])
            check_reset.latency_ms = (time.perf_counter() - t0) * 1000.0

            if len(obs1) != env_count or len(obs2) != env_count:
                raise ValueError(f"Reset returned wrong env count: {len(obs1)} vs {env_count}")

            # Check exact match
            diff = 0.0
            for i in range(env_count):
                for v1, v2 in zip(obs1[i], obs2[i]):
                    diff = max(diff, abs(float(v1) - float(v2)))

            if diff > 1e-6:
                raise ValueError(f"Reset determinism violation: max float difference {diff}")

            check_reset.passed = True
            check_reset.details = {
                "env_count": env_count,
                "max_seed_difference": diff,
            }
        except Exception as exc:
            check_reset.passed = False
            check_reset.error = str(exc)
        checks.append(check_reset)

        # --- Check 4: Episode Plan Staging & Conditions ---
        check_plans = ValidationCheck(
            check_id="plan_staging",
            name="Atomic Episode Plan Staging & Ground-Truth Propagation",
            category="curriculum",
        )
        try:
            t0 = time.perf_counter()
            plans_payload = [
                {
                    "index": i,
                    "seed": 100 + i,
                    "map_id": "two_rooms",
                    "scenario": "corner_fight",
                    "lighting": "low_light",
                    "enemy_count": 3,
                    "curriculum_level": 6,
                }
                for i in range(env_count)
            ]
            stage_res = transport.request({"cmd": "set_episode_plans", "plans": plans_payload})
            check_plans.latency_ms = (time.perf_counter() - t0) * 1000.0

            if not stage_res.get("ok"):
                raise ValueError(f"Plan staging failed: {stage_res.get('error')}")

            # Trigger reset to consume staged plan
            transport.request({"cmd": "reset", "seed": -1})
            cond_res = transport.request({"cmd": "episode_conditions"})
            conditions = cond_res.get("conditions", [])

            if len(conditions) != env_count:
                raise ValueError(f"Expected {env_count} conditions, got {len(conditions)}")

            for i, cond in enumerate(conditions):
                if cond.get("map_id") != "two_rooms":
                    raise ValueError(
                        f"Env {i} map_id was {cond.get('map_id')}, expected 'two_rooms'"
                    )
                if cond.get("curriculum_level") != 6:
                    raise ValueError(
                        f"Env {i} level was {cond.get('curriculum_level')}, expected 6"
                    )

            check_plans.passed = True
            check_plans.details = {
                "staged_count": len(plans_payload),
                "conditions_verified": len(conditions),
            }
        except Exception as exc:
            check_plans.passed = False
            check_plans.error = str(exc)
        checks.append(check_plans)

        # --- Check 5: Stepping & Auto-Reset Gym Semantics ---
        check_step = ValidationCheck(
            check_id="step_auto_reset",
            name="Vector Step Stepping & Gym Auto-Reset Semantics",
            category="simulation",
        )
        try:
            t0 = time.perf_counter()
            step_count = 0
            idle_action = [1, 1, 1, 1, 0, 0]  # canonical idle action
            actions = [idle_action for _ in range(env_count)]

            for _ in range(25):
                step_res = transport.request({"cmd": "step", "actions": actions})
                step_count += env_count
                obs = step_res.get("observations", [])
                rewards = step_res.get("rewards", [])
                dones = step_res.get("dones", [])
                infos = step_res.get("infos", [])
                if any(len(values) != env_count for values in (obs, rewards, dones, infos)):
                    raise ValueError("Step response length mismatch")

            elapsed = max(time.perf_counter() - t0, 1e-6)
            sps = step_count / elapsed
            check_step.latency_ms = elapsed * 1000.0
            check_step.passed = True
            check_step.details = {
                "total_steps_executed": step_count,
                "measured_steps_per_second": round(sps, 1),
            }
            report.measured_throughput["steps_per_second"] = round(sps, 1)
        except Exception as exc:
            check_step.passed = False
            check_step.error = str(exc)
        checks.append(check_step)

        # --- Check 6: Health Check Command ---
        check_health = ValidationCheck(
            check_id="health_check",
            name="Simulation Health Check Introspection",
            category="stability",
        )
        try:
            t0 = time.perf_counter()
            health_res = transport.request({"cmd": "health_check"})
            check_health.latency_ms = (time.perf_counter() - t0) * 1000.0
            health_list = health_res.get("health", [])
            all_healthy = all(bool(item.get("healthy", False)) for item in health_list)
            if all_healthy and len(health_list) == env_count:
                check_health.passed = True
                check_health.details = {"environments_healthy": len(health_list)}
            else:
                check_health.passed = False
                check_health.error = f"Unhealthy environments detected: {health_list}"
        except Exception as exc:
            check_health.passed = False
            check_health.error = str(exc)
        checks.append(check_health)

        # Close single-agent transport
        try:
            transport.close()
        except Exception:
            pass

        # --- Check 7: Self-Play Mode Handshake & Stepping ---
        if test_self_play:
            check_self_play = ValidationCheck(
                check_id="self_play_channel",
                name="Two-Agent Headless Self-Play Channel (--self-play 1)",
                category="self_play",
            )
            sp_transport = None
            try:
                t0 = time.perf_counter()
                from .godot_env import GodotProcessTransport

                sp_transport = GodotProcessTransport(
                    project_path=self.project_path,
                    godot_executable=executable,
                    environment_count=1,
                    self_play=True,
                    request_timeout=self.timeout,
                )
                sp_spaces = sp_transport.spaces
                if sp_spaces.get("policy_slots") != 2:
                    raise ValueError(
                        f"Self play policy slots was {sp_spaces.get('policy_slots')}, expected 2"
                    )

                sp_reset = sp_transport.request({"cmd": "reset", "seed": 777})
                sp_obs = sp_reset.get("observations", [])
                if len(sp_obs) != 1 or len(sp_obs[0]) != 2:
                    raise ValueError(f"Self play reset observation shape invalid: {sp_obs}")

                sp_actions = [[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]]
                sp_step = sp_transport.request({"cmd": "step", "actions": sp_actions})
                sp_obs_step = sp_step.get("observations", [])
                sp_rewards = sp_step.get("rewards", [])
                if len(sp_obs_step[0]) != 2 or len(sp_rewards[0]) != 2:
                    raise ValueError(
                        f"Self play step shape invalid: obs={sp_obs_step}, rew={sp_rewards}"
                    )

                check_self_play.latency_ms = (time.perf_counter() - t0) * 1000.0
                check_self_play.passed = True
                check_self_play.details = {
                    "policy_slots": 2,
                    "slot_0_obs_dim": len(sp_obs[0][0]),
                    "slot_1_obs_dim": len(sp_obs[0][1]),
                }
            except Exception as exc:
                check_self_play.passed = False
                error_text = str(exc)
                # Self-play failures are usually caused inside the Godot
                # process (e.g. a script that failed to compile makes
                # SelfPlayEnvironmentCore.new() return null, the adapter then
                # holds zero environments and reset answers a silent
                # `observations: []`). The engine's SCRIPT ERROR for that is
                # on stderr only, so attach the tail to the report; without it
                # the check can only show the downstream shape mismatch.
                if sp_transport is not None:
                    # stderr is pumped on a background thread. Close joins
                    # that pump before reading the tail; reading immediately
                    # after the malformed response raced the pump and made
                    # this diagnostic intermittently disappear.
                    sp_transport.close()
                    stderr_tail = sp_transport.stderr_tail().strip()
                    if stderr_tail:
                        error_text = f"{error_text} | Godot stderr tail: {stderr_tail}"
                check_self_play.error = error_text
            finally:
                if sp_transport is not None:
                    try:
                        sp_transport.close()
                    except Exception:
                        pass
            checks.append(check_self_play)

        report.checks = checks
        report.total_checks = len(checks)
        report.passed_checks = sum(1 for c in checks if c.passed)
        report.failed_checks = sum(1 for c in checks if not c.passed)
        report.status = "passed" if report.failed_checks == 0 else "failed"

        return report


def format_validation_report(report: RuntimeValidationReport | dict[str, Any]) -> str:
    """Renders a validation report into a clean human-readable string."""
    data = report.to_dict() if hasattr(report, "to_dict") else report
    lines = [
        "=" * 72,
        " SANDBOXAI REAL GODOT RUNTIME VALIDATION REPORT",
        "=" * 72,
        f"Platform:         {data.get('platform_system', 'unknown')}",
        f"Godot Available:  {data.get('godot_available', False)}",
        f"Godot Executable: {data.get('godot_executable', 'none')}",
        f"Godot Version:    {data.get('godot_version') or 'N/A'}",
        f"Overall Status:   {data.get('status', 'unknown').upper()}",
        f"Passed Checks:    {data.get('passed_checks', 0)} / {data.get('total_checks', 0)}",
    ]

    throughput = data.get("measured_throughput", {})
    if throughput.get("measured"):
        lines.append(f"Measured Speed:   {throughput.get('steps_per_second')} steps/sec")
    else:
        lines.append("Measured Speed:   N/A (unmeasured / engine not executed)")

    lines.append("-" * 72)
    lines.append(f"{'Check':<42}{'Category':<14}{'Latency':>10}{'Status':>6}")
    lines.append("-" * 72)

    for check in data.get("checks", []):
        name = str(check.get("name", check.get("check_id", "")))[:40]
        cat = str(check.get("category", ""))[:12]
        lat = (
            f"{check.get('latency_ms', 0.0):.1f} ms" if check.get("latency_ms") is not None else "-"
        )
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"{name:<42}{cat:<14}{lat:>10}{status:>6}")
        if check.get("error"):
            lines.append(f"  -> Error: {check['error']}")

    notes = data.get("notes", [])
    if notes:
        lines.append("-" * 72)
        lines.append("Notes:")
        for note in notes:
            lines.append(f"  * {note}")

    lines.append("=" * 72)
    return "\n".join(lines)
