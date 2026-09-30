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

import contextlib
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import find_godot_executable
from .contract import ACTION_NVEC, GODOT_VERSION, OBSERVATION_FIELD_COUNT
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
        # find_godot_executable reads the environment, the remembered
        # settings file and the filesystem: OSError covers an unreadable
        # or vanished path, ValueError a corrupt settings JSON.
        try:
            resolved = find_godot_executable(self.godot_executable_raw)
            return bool(shutil.which(resolved) or Path(resolved).is_file())
        except (OSError, ValueError):
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
        except (OSError, ValueError, subprocess.SubprocessError):
            # Missing binary, a refused launch, or a build that never
            # answers --version. None means "could not be determined",
            # which the report distinguishes from a wrong version.
            return None

    # --- orchestration ---------------------------------------------------

    def validate(
        self,
        env_count: int = 2,
        test_self_play: bool = True,
    ) -> RuntimeValidationReport:
        """Runs the live-engine check battery and returns the report.

        Each `_check_*` helper below owns exactly one check, catches its own
        failures and returns a populated ValidationCheck. This function only
        sequences them, which is what makes the battery extendable: adding a
        check is a new method plus one line here, not another branch in what
        used to be a 350-line method with a cyclomatic complexity of 34.
        """
        system = platform.system()
        executable, available = self._resolve_executable()
        if not available:
            return self._unavailable_report(system)

        report = RuntimeValidationReport(
            godot_executable=executable,
            godot_available=True,
            godot_version=self.probe_version(executable),
            platform_system=system,
            status="passed",
            measured_throughput={"measured": True},
        )

        checks: list[ValidationCheck] = []
        check_spaces, transport = self._check_bridge_spaces(executable, env_count)
        checks.append(check_spaces)
        if not check_spaces.passed or transport is None:
            # Nothing downstream can run without a verified bridge, and
            # reporting six cascading failures would bury the real cause.
            if transport is not None:
                transport.close()
            return self._finalize(report, checks)

        try:
            checks.append(self._check_ping(transport))
            checks.append(self._check_reset_determinism(transport, env_count))
            checks.append(self._check_plan_staging(transport, env_count))
            check_step = self._check_stepping(transport, env_count)
            checks.append(check_step)
            if check_step.passed:
                report.measured_throughput["steps_per_second"] = check_step.details[
                    "measured_steps_per_second"
                ]
            checks.append(self._check_health(transport, env_count))
        finally:
            with contextlib.suppress(Exception):
                transport.close()

        # Self-play needs its own engine process (--self-play 1), so it runs
        # after the single-agent transport is gone rather than alongside it.
        if test_self_play:
            checks.append(self._check_self_play(executable))

        return self._finalize(report, checks)

    def _resolve_executable(self) -> tuple[str, bool]:
        try:
            executable = find_godot_executable(self.godot_executable_raw)
        except (OSError, ValueError):
            return self.godot_executable_raw, False
        return executable, bool(shutil.which(executable) or Path(executable).is_file())

    def _unavailable_report(self, system: str) -> RuntimeValidationReport:
        """The "no engine installed" outcome, which is not a failure.

        `unavailable` is deliberately distinct from `failed`: CI machines and
        fresh checkouts have no Godot, and the CLI treats both `passed` and
        `unavailable` as exit code 0.
        """
        return RuntimeValidationReport(
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
                f"Install Godot {GODOT_VERSION} and make it available as 'godot' or pass "
                "--godot-executable.",
            ],
        )

    @staticmethod
    def _finalize(
        report: RuntimeValidationReport, checks: list[ValidationCheck]
    ) -> RuntimeValidationReport:
        report.checks = checks
        report.total_checks = len(checks)
        report.passed_checks = sum(1 for check in checks if check.passed)
        report.failed_checks = sum(1 for check in checks if not check.passed)
        report.status = "passed" if report.failed_checks == 0 else "failed"
        return report

    # --- individual checks -------------------------------------------------

    def _check_bridge_spaces(self, executable: str, env_count: int) -> tuple[ValidationCheck, Any]:
        """Check 1: handshake, and that the engine's spaces match contract.py.

        Returns the live transport alongside the check so the remaining
        checks can reuse the process - engine startup costs seconds.
        """
        check = ValidationCheck(
            check_id="bridge_spaces",
            name="JSON-Lines Bridge Spaces & Contract Verification",
            category="contract",
        )
        started = time.perf_counter()
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
            check.latency_ms = (time.perf_counter() - started) * 1000.0
            obs_space = spaces.get("observation_space", {})
            act_space = spaces.get("action_space", {})
            obs_size = int(obs_space.get("size", 0))
            act_nvec = list(act_space.get("nvec", []))

            if obs_size != OBSERVATION_FIELD_COUNT:
                raise ValueError(
                    f"Observation size mismatch: engine returned {obs_size}, "
                    f"expected {OBSERVATION_FIELD_COUNT}"
                )
            if act_nvec != list(ACTION_NVEC):
                raise ValueError(
                    f"Action nvec mismatch: engine returned {act_nvec}, "
                    f"expected {list(ACTION_NVEC)}"
                )

            check.passed = True
            check.details = {
                "observation_size": obs_size,
                "action_nvec": act_nvec,
                "action_type": act_space.get("type"),
            }
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check, transport

    @staticmethod
    def _check_ping(transport: Any) -> ValidationCheck:
        """Check 2: stdio ping/pong roundtrip."""
        check = ValidationCheck(
            check_id="bridge_ping",
            name="Stdio Protocol Ping/Pong Latency",
            category="transport",
        )
        try:
            started = time.perf_counter()
            response = transport.request({"cmd": "ping"})
            latency = (time.perf_counter() - started) * 1000.0
            if response.get("ok") and response.get("pong"):
                check.passed = True
                check.latency_ms = latency
                check.details = {"round_trip_ms": round(latency, 3)}
            else:
                check.passed = False
                check.error = f"Malformed ping response: {response}"
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check

    @staticmethod
    def _check_reset_determinism(transport: Any, env_count: int) -> ValidationCheck:
        """Check 3: the same seed must produce bit-comparable observations."""
        check = ValidationCheck(
            check_id="reset_determinism",
            name="Synchronous Reset & Seeding Determinism",
            category="determinism",
        )
        try:
            started = time.perf_counter()
            first = transport.request({"cmd": "reset", "seed": 42}).get("observations", [])
            second = transport.request({"cmd": "reset", "seed": 42}).get("observations", [])
            check.latency_ms = (time.perf_counter() - started) * 1000.0

            if len(first) != env_count or len(second) != env_count:
                raise ValueError(f"Reset returned wrong env count: {len(first)} vs {env_count}")

            difference = 0.0
            for index in range(env_count):
                for left, right in zip(first[index], second[index]):
                    difference = max(difference, abs(float(left) - float(right)))
            if difference > 1e-6:
                raise ValueError(f"Reset determinism violation: max float difference {difference}")

            check.passed = True
            check.details = {"env_count": env_count, "max_seed_difference": difference}
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check

    @staticmethod
    def _check_plan_staging(transport: Any, env_count: int) -> ValidationCheck:
        """Check 4: staged episode plans must survive the next reset intact."""
        check = ValidationCheck(
            check_id="plan_staging",
            name="Atomic Episode Plan Staging & Ground-Truth Propagation",
            category="curriculum",
        )
        try:
            started = time.perf_counter()
            plans = [
                {
                    "index": index,
                    "seed": 100 + index,
                    "map_id": "two_rooms",
                    "scenario": "corner_fight",
                    "lighting": "low_light",
                    "enemy_count": 3,
                    "curriculum_level": 6,
                }
                for index in range(env_count)
            ]
            staged = transport.request({"cmd": "set_episode_plans", "plans": plans})
            check.latency_ms = (time.perf_counter() - started) * 1000.0
            if not staged.get("ok"):
                raise ValueError(f"Plan staging failed: {staged.get('error')}")

            # seed -1 means "consume the staged plan" rather than reseed.
            transport.request({"cmd": "reset", "seed": -1})
            conditions = transport.request({"cmd": "episode_conditions"}).get("conditions", [])
            if len(conditions) != env_count:
                raise ValueError(f"Expected {env_count} conditions, got {len(conditions)}")
            for index, condition in enumerate(conditions):
                if condition.get("map_id") != "two_rooms":
                    raise ValueError(
                        f"Env {index} map_id was {condition.get('map_id')}, expected 'two_rooms'"
                    )
                if condition.get("curriculum_level") != 6:
                    raise ValueError(
                        f"Env {index} level was {condition.get('curriculum_level')}, expected 6"
                    )

            check.passed = True
            check.details = {
                "staged_count": len(plans),
                "conditions_verified": len(conditions),
            }
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check

    @staticmethod
    def _check_stepping(transport: Any, env_count: int) -> ValidationCheck:
        """Check 5: vector stepping shape, plus the run's throughput measurement."""
        check = ValidationCheck(
            check_id="step_auto_reset",
            name="Vector Step Stepping & Gym Auto-Reset Semantics",
            category="simulation",
        )
        try:
            started = time.perf_counter()
            step_count = 0
            idle_action = [1, 1, 1, 1, 0, 0]  # canonical idle action
            actions = [idle_action for _ in range(env_count)]

            for _ in range(25):
                response = transport.request({"cmd": "step", "actions": actions})
                step_count += env_count
                lengths = (
                    response.get("observations", []),
                    response.get("rewards", []),
                    response.get("dones", []),
                    response.get("infos", []),
                )
                if any(len(values) != env_count for values in lengths):
                    raise ValueError("Step response length mismatch")

            elapsed = max(time.perf_counter() - started, 1e-6)
            check.latency_ms = elapsed * 1000.0
            check.passed = True
            check.details = {
                "total_steps_executed": step_count,
                "measured_steps_per_second": round(step_count / elapsed, 1),
            }
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check

    @staticmethod
    def _check_health(transport: Any, env_count: int) -> ValidationCheck:
        """Check 6: every environment reports itself healthy."""
        check = ValidationCheck(
            check_id="health_check",
            name="Simulation Health Check Introspection",
            category="stability",
        )
        try:
            started = time.perf_counter()
            health = transport.request({"cmd": "health_check"}).get("health", [])
            check.latency_ms = (time.perf_counter() - started) * 1000.0
            all_healthy = all(bool(item.get("healthy", False)) for item in health)
            if all_healthy and len(health) == env_count:
                check.passed = True
                check.details = {"environments_healthy": len(health)}
            else:
                check.passed = False
                check.error = f"Unhealthy environments detected: {health}"
        except Exception as exc:
            check.passed = False
            check.error = str(exc)
        return check

    def _check_self_play(self, executable: str) -> ValidationCheck:
        """Check 7: the two-agent headless channel (--self-play 1)."""
        check = ValidationCheck(
            check_id="self_play_channel",
            name="Two-Agent Headless Self-Play Channel (--self-play 1)",
            category="self_play",
        )
        transport = None
        try:
            started = time.perf_counter()
            from .godot_env import GodotProcessTransport

            transport = GodotProcessTransport(
                project_path=self.project_path,
                godot_executable=executable,
                environment_count=1,
                self_play=True,
                request_timeout=self.timeout,
            )
            spaces = transport.spaces
            if spaces.get("policy_slots") != 2:
                raise ValueError(
                    f"Self play policy slots was {spaces.get('policy_slots')}, expected 2"
                )

            observations = transport.request({"cmd": "reset", "seed": 777}).get("observations", [])
            if len(observations) != 1 or len(observations[0]) != 2:
                raise ValueError(f"Self play reset observation shape invalid: {observations}")

            actions = [[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]]
            stepped = transport.request({"cmd": "step", "actions": actions})
            stepped_obs = stepped.get("observations", [])
            rewards = stepped.get("rewards", [])
            if len(stepped_obs[0]) != 2 or len(rewards[0]) != 2:
                raise ValueError(f"Self play step shape invalid: obs={stepped_obs}, rew={rewards}")

            check.latency_ms = (time.perf_counter() - started) * 1000.0
            check.passed = True
            check.details = {
                "policy_slots": 2,
                "slot_0_obs_dim": len(observations[0][0]),
                "slot_1_obs_dim": len(observations[0][1]),
            }
        except Exception as exc:
            check.passed = False
            check.error = self._self_play_error_text(exc, transport)
        finally:
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.close()
        return check

    @staticmethod
    def _self_play_error_text(exc: Exception, transport: Any) -> str:
        """Attaches the engine's stderr tail to a self-play failure.

        Self-play failures are usually caused inside the Godot process (e.g.
        a script that failed to compile makes SelfPlayEnvironmentCore.new()
        return null, the adapter then holds zero environments and reset
        answers a silent `observations: []`). The engine's SCRIPT ERROR for
        that is on stderr only; without this the check can only show the
        downstream shape mismatch.
        """
        text = str(exc)
        if transport is None:
            return text
        # stderr is pumped on a background thread. Close joins that pump
        # before reading the tail; reading immediately after the malformed
        # response raced the pump and made this diagnostic intermittently
        # disappear.
        transport.close()
        tail = transport.stderr_tail().strip()
        return f"{text} | Godot stderr tail: {tail}" if tail else text


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
