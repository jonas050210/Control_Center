"""Single source of truth for hardware device-comparison measurement.

The Control Center's first-start hardware wizard needs to answer one
question honestly: *on this machine, which device configuration trains
this policy fastest?* There are three candidates worth comparing:

* **CPU** — PPO updates and rollout/evaluation inference both on the CPU.
* **CUDA** — both on the GPU. Only offered when a CUDA device is present.
* **Hybrid** — PPO updates on the GPU, but rollout/evaluation inference
  on the CPU (``device="cuda"``, ``inference_device="cpu"``). This removes
  the per-step host<->device round trip that makes CUDA slower for this
  small MLP while still keeping the batched optimizer update on the GPU.
  Only offered when a CUDA device is present.

This module owns the measurement *orchestration*, candidate selection,
profile persistence and safe fallback. The GUI (Godot operator scene and
the Tk desktop Control Center alike) must call in here rather than growing
a second, independent benchmark implementation — that is the whole point.

**It measures; it never invents.** A device that cannot be measured (no
CUDA, or the short training run failed) is recorded with an explicit
status and *no* throughput number, exactly like ``benchmark.py`` and
``benchmark_suites.py``. If every candidate fails, the profile falls back
to CPU defaults and says so. The real per-device measurement runs a short
PPO training slice through the existing :func:`sandboxai.ppo.train_ppo`
path, so it needs a working Godot bridge; the orchestration around it is
fully exercisable with an injected measurement function and no engine.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Bumped when the on-disk profile schema changes in a non-additive way.
HARDWARE_PROFILE_FORMAT = "sandboxai.hardware_profile/v1"

#: The number of environment steps the normal first-start measurement uses
#: per candidate. Small enough to finish quickly, large enough to average
#: out process-startup noise. Callers may override it (e.g. a quick retry).
DEFAULT_MEASUREMENT_STEPS = 5_000

#: The device labels the wizard compares, and the concrete
#: ``(device, inference_device)`` pair each maps onto.
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_HYBRID = "hybrid"

#: Measurement outcome states. Only ``measured`` carries a throughput.
STATUS_MEASURED = "measured"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_SETTINGS_DIR_NAME = ".sandboxai"
_PROFILE_FILE_NAME = "hardware_profile.json"


@dataclass(frozen=True)
class DeviceCandidate:
    """One device configuration to compare.

    ``label`` is the user-facing name (``cpu`` / ``hybrid`` / ``cuda``);
    ``device`` and ``inference_device`` are the concrete
    :class:`~sandboxai.config.TrainingConfig` fields it maps onto.
    """

    label: str
    device: str
    inference_device: str

    @property
    def description(self) -> str:
        if self.label == DEVICE_CPU:
            return "CPU — updates and inference on the CPU"
        if self.label == DEVICE_CUDA:
            return "CUDA — updates and inference on the GPU"
        if self.label == DEVICE_HYBRID:
            return "Hybrid — GPU updates, CPU inference (no per-step transfer)"
        return self.label


#: The three comparison candidates, in display order.
_ALL_CANDIDATES: tuple[DeviceCandidate, ...] = (
    DeviceCandidate(DEVICE_CPU, "cpu", "cpu"),
    DeviceCandidate(DEVICE_HYBRID, "cuda", "cpu"),
    DeviceCandidate(DEVICE_CUDA, "cuda", "cuda"),
)


def cuda_available() -> bool:
    """Whether a CUDA device is actually usable on this host.

    Defensive on purpose: a broken or CPU-only torch build reports
    ``False`` rather than raising, so candidate discovery never crashes
    the wizard.
    """
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def available_candidates(*, has_cuda: bool | None = None) -> list[DeviceCandidate]:
    """The device candidates that can actually be measured on this host.

    CPU is always available. Hybrid and CUDA are offered only when a CUDA
    device is present, so the wizard never shows a permanently-``n/a`` row
    for a device this machine does not have. Pass ``has_cuda`` to override
    detection (tests, or reusing an earlier probe).
    """
    if has_cuda is None:
        has_cuda = cuda_available()
    if has_cuda:
        return list(_ALL_CANDIDATES)
    return [candidate for candidate in _ALL_CANDIDATES if candidate.label == DEVICE_CPU]


@dataclass
class DeviceMeasurement:
    """The measured (or unmeasurable) throughput of one candidate."""

    label: str
    device: str
    inference_device: str
    status: str
    steps_per_second: float | None = None
    steps_completed: int | None = None
    wall_seconds: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_MEASURED and self.steps_per_second is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "device": self.device,
            "inference_device": self.inference_device,
            "status": self.status,
            "steps_per_second": self.steps_per_second,
            "steps_completed": self.steps_completed,
            "wall_seconds": self.wall_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceMeasurement:
        return cls(
            label=str(data["label"]),
            device=str(data["device"]),
            inference_device=str(data["inference_device"]),
            status=str(data.get("status", STATUS_FAILED)),
            steps_per_second=_opt_float(data.get("steps_per_second")),
            steps_completed=_opt_int(data.get("steps_completed")),
            wall_seconds=_opt_float(data.get("wall_seconds")),
            error=(str(data["error"]) if data.get("error") is not None else None),
        )


#: A measurement function turns one candidate into one measurement. The
#: default (:func:`default_measure`) runs a short training slice; tests and
#: dry runs inject their own.
MeasureFn = Callable[[DeviceCandidate], DeviceMeasurement]

#: A cancellation predicate: return ``True`` to stop before the next
#: candidate. The wizard checks it between candidates, so an in-flight
#: measurement always finishes rather than being killed mid-training.
CancelFn = Callable[[], bool]

#: A progress callback: called with each measurement as it completes.
ProgressFn = Callable[[DeviceMeasurement], None]


@dataclass
class HardwareProfile:
    """The persisted result of a hardware wizard run.

    ``selected_device`` is the user-facing label; ``device`` and
    ``inference_device`` are the concrete config overrides
    :meth:`config_overrides` hands back. ``fallback`` is ``True`` when no
    candidate could be measured and CPU defaults were assumed.
    """

    selected_device: str
    device: str
    inference_device: str
    measurement_steps: int
    measurements: list[DeviceMeasurement]
    godot_executable: str | None
    godot_available: bool
    host: dict[str, Any]
    fallback: bool
    note: str
    created_utc: str = ""
    format: str = HARDWARE_PROFILE_FORMAT

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def config_overrides(self) -> dict[str, str]:
        """The ``TrainingConfig`` fields this profile recommends."""
        return {"device": self.device, "inference_device": self.inference_device}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "created_utc": self.created_utc,
            "selected_device": self.selected_device,
            "device": self.device,
            "inference_device": self.inference_device,
            "measurement_steps": self.measurement_steps,
            "measurements": [m.to_dict() for m in self.measurements],
            "godot_executable": self.godot_executable,
            "godot_available": self.godot_available,
            "host": self.host,
            "fallback": self.fallback,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        measurements = [DeviceMeasurement.from_dict(item) for item in data.get("measurements", [])]
        return cls(
            selected_device=str(data.get("selected_device", DEVICE_CPU)),
            device=str(data.get("device", "cpu")),
            inference_device=str(data.get("inference_device", "cpu")),
            measurement_steps=int(data.get("measurement_steps", DEFAULT_MEASUREMENT_STEPS)),
            measurements=measurements,
            godot_executable=(
                str(data["godot_executable"]) if data.get("godot_executable") is not None else None
            ),
            godot_available=bool(data.get("godot_available", False)),
            host=dict(data.get("host", {})),
            fallback=bool(data.get("fallback", False)),
            note=str(data.get("note", "")),
            created_utc=str(data.get("created_utc", "")),
            format=str(data.get("format", HARDWARE_PROFILE_FORMAT)),
        )

    def save(self, path: Path | None = None) -> Path:
        """Persist the profile as JSON, returning the path written.

        Defaults to ``.sandboxai/hardware_profile.json`` at the repository
        root, next to the remembered Godot executable setting.
        """
        target = Path(path) if path is not None else profile_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def profile_path() -> Path:
    """Location of the machine-local hardware profile (repository root)."""
    return Path(__file__).resolve().parents[2] / _SETTINGS_DIR_NAME / _PROFILE_FILE_NAME


def load_profile(path: Path | None = None) -> HardwareProfile | None:
    """The persisted hardware profile, or ``None`` when unset/unreadable.

    Reading is deliberately defensive (BOM-tolerant, like every other JSON
    read in this package): a missing or corrupt profile never breaks the
    Control Center, it only means the wizard has not run yet.
    """
    target = Path(path) if path is not None else profile_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return HardwareProfile.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def measure_devices(
    candidates: list[DeviceCandidate],
    measure: MeasureFn,
    *,
    cancel: CancelFn | None = None,
    on_progress: ProgressFn | None = None,
) -> list[DeviceMeasurement]:
    """Measure each candidate in turn, honoring cancellation between them.

    A candidate whose measurement raises is recorded as ``failed`` with the
    exception text rather than aborting the whole comparison — one broken
    device must not deny the user the others. Cancellation is checked
    *before* each candidate, so an in-flight training slice always finishes;
    remaining candidates are recorded as ``cancelled``.
    """
    results: list[DeviceMeasurement] = []
    stopped = False
    for candidate in candidates:
        if stopped or (cancel is not None and cancel()):
            stopped = True
            measurement = DeviceMeasurement(
                label=candidate.label,
                device=candidate.device,
                inference_device=candidate.inference_device,
                status=STATUS_CANCELLED,
            )
        else:
            try:
                measurement = measure(candidate)
            except Exception as exc:  # measurement must never crash the wizard
                measurement = DeviceMeasurement(
                    label=candidate.label,
                    device=candidate.device,
                    inference_device=candidate.inference_device,
                    status=STATUS_FAILED,
                    error=str(exc),
                )
        results.append(measurement)
        if on_progress is not None:
            on_progress(measurement)
    return results


def select_best(measurements: list[DeviceMeasurement]) -> DeviceMeasurement | None:
    """The successfully-measured candidate with the highest throughput.

    Returns ``None`` when nothing could be measured, which is the signal
    for :func:`build_profile` to fall back to CPU defaults.
    """
    ranked = [m for m in measurements if m.ok]
    if not ranked:
        return None
    return max(ranked, key=lambda m: m.steps_per_second or 0.0)


def build_profile(
    measurements: list[DeviceMeasurement],
    *,
    measurement_steps: int,
    godot_executable: str | None,
    godot_available: bool,
    host: dict[str, Any] | None = None,
) -> HardwareProfile:
    """Turn a set of measurements into a persisted, actionable profile.

    The winner is the fastest measured candidate. If none could be
    measured (no engine, or every slice failed), the profile falls back to
    CPU defaults and records ``fallback=True`` with a human-readable note,
    never a fabricated throughput.
    """
    if host is None:
        from .manifest import host_snapshot

        host = host_snapshot()
    best = select_best(measurements)
    if best is None:
        note = (
            "No device could be measured; defaulting to CPU. "
            "Re-run the hardware wizard once a Godot bridge is reachable."
        )
        return HardwareProfile(
            selected_device=DEVICE_CPU,
            device="cpu",
            inference_device="cpu",
            measurement_steps=measurement_steps,
            measurements=measurements,
            godot_executable=godot_executable,
            godot_available=godot_available,
            host=host,
            fallback=True,
            note=note,
        )
    note = (
        f"Selected {best.label} at "
        f"{best.steps_per_second:.1f} steps/s over {measurement_steps} steps."
    )
    return HardwareProfile(
        selected_device=best.label,
        device=best.device,
        inference_device=best.inference_device,
        measurement_steps=measurement_steps,
        measurements=measurements,
        godot_executable=godot_executable,
        godot_available=godot_available,
        host=host,
        fallback=False,
        note=note,
    )


def default_measure(
    candidate: DeviceCandidate,
    *,
    project_path: str | Path,
    godot_executable: str,
    steps: int,
    environment_count: int = 4,
    env_workers: int = 1,
    enemy_count: int = 1,
    seed: int = 12345,
    output_root: str | Path | None = None,
) -> DeviceMeasurement:
    """Measure one candidate by timing a short real PPO training slice.

    This is the honest, engine-backed measurement: it builds a
    :class:`~sandboxai.config.TrainingConfig` for the candidate's device
    pair, runs ``steps`` timesteps through :func:`sandboxai.ppo.train_ppo`,
    and reports ``steps_completed / wall_seconds``. It needs a working
    Godot bridge (``godot_executable``); when the bridge is missing or the
    slice fails, the exception is surfaced to
    :func:`measure_devices`, which records a ``failed`` measurement.

    ``env_workers`` (default 1, the single-process bridge) lets callers
    measure a full topology — the benchmark pipeline validates its
    finalist (environment_count, env_workers) pairs through this same
    function instead of a second training-slice implementation.

    The training artifacts land in a throwaway directory that is removed
    afterwards — a measurement must not litter the user's run history.
    """
    from .config import TrainingConfig
    from .ppo import train_ppo

    with tempfile.TemporaryDirectory(prefix="sandboxai-hw-") as tmp:
        root = str(output_root) if output_root is not None else tmp
        config = TrainingConfig(
            environment_count=environment_count,
            env_workers=env_workers,
            enemy_count=enemy_count,
            total_training_steps=steps,
            checkpoint_frequency=max(steps * 2, 1),
            evaluation_frequency=max(steps * 2, 1),
            seed=seed,
            device=candidate.device,
            inference_device=candidate.inference_device,
            godot_executable=godot_executable,
            project_path=str(project_path),
            output_root=root,
            run_id=f"hwprofile_{candidate.label}",
            net_arch=(128, 128),
            skill_metrics=False,
            episode_log=False,
            replay_mode="off",
            curriculum_mode="fixed",
        ).validate()
        started = time.monotonic()
        result = train_ppo(config)
        elapsed = time.monotonic() - started

    completed = int(result.get("training_steps_completed") or result.get("timesteps") or 0)
    steps_per_second = (completed / elapsed) if elapsed > 0 and completed > 0 else None
    if steps_per_second is None:
        return DeviceMeasurement(
            label=candidate.label,
            device=candidate.device,
            inference_device=candidate.inference_device,
            status=STATUS_FAILED,
            error="training slice completed no measurable steps",
        )
    return DeviceMeasurement(
        label=candidate.label,
        device=candidate.device,
        inference_device=candidate.inference_device,
        status=STATUS_MEASURED,
        steps_per_second=steps_per_second,
        steps_completed=completed,
        wall_seconds=elapsed,
    )


def _make_default_measure(project_path: str | Path, godot_executable: str, steps: int) -> MeasureFn:
    """A :data:`MeasureFn` bound to one project/engine/step budget."""

    def measure(candidate: DeviceCandidate) -> DeviceMeasurement:
        return default_measure(
            candidate,
            project_path=project_path,
            godot_executable=godot_executable,
            steps=steps,
        )

    return measure


def run_hardware_wizard(
    *,
    project_path: str | Path,
    godot_executable: str | None = None,
    steps: int = DEFAULT_MEASUREMENT_STEPS,
    measure: MeasureFn | None = None,
    cancel: CancelFn | None = None,
    on_progress: ProgressFn | None = None,
    save: bool = True,
    save_path: Path | None = None,
    has_cuda: bool | None = None,
) -> HardwareProfile:
    """Detect Godot, measure the available devices and persist a profile.

    This is the one entry point a Control Center wizard should call. It:

    1. resolves the Godot executable (reusing ``config.find_godot_executable``);
    2. builds the candidate list, hiding CUDA/Hybrid when no GPU is present;
    3. measures each candidate (``measure`` defaults to the engine-backed
       :func:`default_measure`), honoring ``cancel`` between candidates;
    4. builds a profile — falling back to CPU defaults if nothing measured;
    5. persists it (unless ``save=False``).

    It never raises for a missing engine or a failed measurement: those
    become an honest ``fallback`` profile.
    """
    from .config import _resolve_executable, find_godot_executable

    raw = godot_executable or "godot"
    resolved = find_godot_executable(raw)
    godot_ok = _resolve_executable(resolved) is not None

    candidates = available_candidates(has_cuda=has_cuda)

    if measure is None and not godot_ok:
        # No engine and no injected measurement: skip straight to a
        # documented CPU fallback rather than launching a measurement that
        # can only fail.
        measurements = [
            DeviceMeasurement(
                label=candidate.label,
                device=candidate.device,
                inference_device=candidate.inference_device,
                status=STATUS_UNAVAILABLE,
                error="no Godot bridge is reachable to measure throughput",
            )
            for candidate in candidates
        ]
    else:
        measure_fn: MeasureFn = measure or _make_default_measure(project_path, resolved, steps)
        measurements = measure_devices(
            candidates, measure_fn, cancel=cancel, on_progress=on_progress
        )

    profile = build_profile(
        measurements,
        measurement_steps=steps,
        godot_executable=(resolved if godot_ok else raw),
        godot_available=godot_ok,
    )
    if save:
        profile.save(save_path)
    return profile


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
