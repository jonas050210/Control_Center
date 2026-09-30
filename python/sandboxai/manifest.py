"""Run manifest: what a result was produced BY.

Split out of :mod:`sandboxai.pipeline` (which is about orchestration) so
provenance has one obvious home. A manifest answers the questions you
actually ask three months later, when a checkpoint behaves differently
than the notes claim:

* which code — commit **and whether the tree was dirty**;
* which simulator — the Godot build that produced the trajectories, not
  the version the README recommends;
* which host — Python/OS/CPU/GPU, because throughput numbers and CUDA vs
  CPU sampling streams are host-dependent;
* which contract, seeds, curriculum, selection rule and hyperparameters.

The full hyperparameter set stays in ``config.json``; the manifest links
to it instead of duplicating every field. Everything here degrades
gracefully: a missing git binary, an absent Godot executable or a torch
build without CUDA produce ``None``/``false`` entries, never an
exception. A manifest must never be the reason a run fails to start.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: Bumped from v1: additive host/godot/code-provenance sections. Readers
#: that only look up known keys are unaffected.
MANIFEST_FORMAT = "sandboxai.run_manifest/v2"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Packages whose version changes can change results.
_TRACKED_PACKAGES = ("torch", "stable_baselines3", "gymnasium", "numpy")


def contract_fingerprint() -> dict[str, Any]:
    """The observation/action contract identity (replay-header style).

    Dimensions, not a version string: they are what actually decides
    compatibility, and they are what replay files stamp.
    """
    from .contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

    return {
        "observation_dim": OBSERVATION_FIELD_COUNT,
        "action_nvec": list(ACTION_NVEC),
        "observation_fields": "v3 (additive since v1; see docs/OBSERVATION_ACTION_CONTRACT.md)",
    }


def package_versions() -> dict[str, Any]:
    from . import __version__

    versions: dict[str, Any] = {"sandboxai": __version__}
    for package in _TRACKED_PACKAGES:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = None
    return versions


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def code_provenance() -> dict[str, Any]:
    """Commit, branch and — crucially — whether the tree was modified.

    A commit hash alone is a half-truth: training from a dirty tree is
    the single most common reason a "reproduction" does not reproduce.
    ``dirty`` is ``None`` when git could not be consulted at all, so an
    unknown state is never reported as clean.
    """
    try:
        commit = _git("rev-parse", "HEAD")
        if not commit:
            return {"commit": "", "branch": "", "dirty": None}
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "branch": "", "dirty": None}


def host_snapshot() -> dict[str, Any]:
    """Interpreter, OS, CPU and accelerator facts that affect results.

    Throughput comparisons across hosts are meaningless without this, and
    the CPU/CUDA split decides which RNG stream sampled the rollout.
    """
    logical = os.cpu_count() or 1
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "logical_cpus": logical,
        # WSL is a first-class target here (Windows host, Linux trainer),
        # and it changes process-spawn and path semantics.
        "wsl": "microsoft" in platform.release().lower(),
    }
    try:
        import torch  # type: ignore

        snapshot["torch_threads"] = int(torch.get_num_threads())
        cuda_available = bool(torch.cuda.is_available())
        snapshot["cuda_available"] = cuda_available
        if cuda_available:
            snapshot["cuda_device"] = str(torch.cuda.get_device_name(0))
            snapshot["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        # Deliberately broad: a broken or partially installed torch must
        # not stop a run from writing its manifest.
        snapshot["cuda_available"] = None
    return snapshot


def godot_snapshot(config: Any, probe: bool = True) -> dict[str, Any]:
    """Which simulator build actually produced the trajectories.

    Godot 4.7.2 is a project constraint, so recording the engine version
    that ran is worth one subprocess per run. ``probe=False`` skips the
    launch for callers that only want the configured path.
    """
    from .config import _resolve_executable, find_godot_executable

    raw = str(getattr(config, "godot_executable", "godot") or "godot")
    snapshot: dict[str, Any] = {"configured": raw, "resolved": raw, "version": None}
    if raw and raw != "godot":
        resolved = _resolve_executable(raw)
        if resolved:
            snapshot["resolved"] = resolved
    else:
        try:
            detected = find_godot_executable(raw)
            if _resolve_executable(detected):
                snapshot["resolved"] = detected
        except (OSError, ValueError):
            # No discoverable engine: the manifest keeps the configured
            # name and records no version, which is the honest answer.
            pass
    if not probe:
        return snapshot
    try:
        from .runtime_validation import RuntimeValidator

        if _resolve_executable(snapshot["resolved"]):
            validator = RuntimeValidator(godot_executable=snapshot["resolved"])
            snapshot["version"] = validator.probe_version(snapshot["resolved"])
    except (OSError, ValueError, ImportError):
        snapshot["version"] = None
    return snapshot


def build_manifest(
    config: Any,
    run_dir: Path,
    device: str,
    driver: Any | None,
    probe_godot: bool = True,
) -> dict[str, Any]:
    """Enough information to reproduce and interpret a run, and no more."""
    from .pipeline import EVAL_MASTER_SEED_SALT

    code = code_provenance()

    curriculum = {
        "mode": config.curriculum_mode,
        "adaptive": config.adaptive_curriculum,
    }
    training_distribution: dict[str, Any] = {}
    if config.curriculum_mode == "auto":
        curriculum["start_level"] = config.curriculum_start_level
        if driver is not None:
            curriculum["current"] = driver.curriculum_snapshot()
            training_distribution = driver.director.distribution.to_dict()
    else:
        curriculum["fixed_level"] = config.curriculum_level

    return {
        "format": MANIFEST_FORMAT,
        "experiment_id": config.experiment_id,
        "run_id": config.run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config.seed,
        "device": device,
        "contract": contract_fingerprint(),
        "godot_project": str(config.project),
        "godot": godot_snapshot(config, probe=probe_godot),
        "host": host_snapshot(),
        "curriculum": curriculum,
        "training_distribution": training_distribution,
        "evaluation": {
            "episodes": config.evaluation_episodes,
            "frequency": config.evaluation_frequency,
            "environment_count": config.evaluation_environment_count,
            "checkpoint_eval_environment_count": config.checkpoint_eval_environment_count,
            "inference_device": config.inference_device,
            "condition_eval": config.checkpoint_condition_eval,
            "generalization_eval": config.checkpoint_generalization_eval,
            "league_eval": config.checkpoint_league_eval,
            "eval_master_seed_salt": EVAL_MASTER_SEED_SALT,
        },
        "checkpoint_selection": config.checkpoint_selection_rule().as_dict(),
        "enabled_systems": {
            "skill_metrics": config.skill_metrics,
            "episode_log": config.episode_log,
            "replay_mode": config.replay_mode,
            "replay_detail": config.replay_detail if config.replay_mode != "off" else "off",
        },
        "hyperparameters": {
            key: getattr(config, key)
            for key in (
                "learning_rate",
                "rollout_length",
                "batch_size",
                "ppo_epochs",
                "gamma",
                "gae_lambda",
                "entropy_coefficient",
                "clip_range",
                "total_training_steps",
                "environment_count",
                "enemy_count",
            )
        },
        "rollout_schedule": config.rollout_schedule(),
        "parallelism": {
            "environment_count": config.environment_count,
            "env_workers": config.env_workers,
            "resolved_env_workers": config.resolved_env_workers(),
            "torch_threads": config.torch_threads,
            "resolved_torch_threads": config.resolved_torch_threads(),
        },
        "net_arch": list(config.net_arch),
        "config_file": "config.json",
        # Kept at the top level under its historical name so existing
        # readers keep working; `code` carries the dirty flag.
        "code_revision": code.get("commit", ""),
        "code": code,
        "package_versions": package_versions(),
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Writes ``run_manifest.json`` next to config.json / run_summary.json."""
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path
