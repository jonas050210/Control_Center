"""Background PPO training with curriculum, checkpointing, and frozen self-play."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import threading
import time
from typing import Any

from training.workers import CPU_JOB_LOCK, cpu_core_counts, make_vector_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    duration_seconds: float = 600.0
    total_timesteps: int = 1_000_000
    n_workers: int = 8
    envs_per_worker: int = 1
    map_name: str = "Dust"
    map_definition: Any | None = None
    curriculum: bool = True
    self_play: bool = True
    method: str = "Pure RL"
    vision_mode: str = "coarse_los"
    resume_checkpoint: str | None = None
    imitation_path: str | None = None
    max_envs: int = 24
    episode_seconds: float = 60.0
    # A curriculum must not outrun the policy: the opponent is only made harder
    # once the current phase is actually being won. Without this gate a 120k-step
    # run measurably collapsed (67.9% -> 0.0%) when the timer forced phase 3 + 4.
    curriculum_min_win_rate: float = 0.40
    # Episodes the "best checkpoint" watch averages over. A short window chases
    # single lucky kills; a long one averages a peak away.
    best_window_episodes: int = 50
    # Configurable, because the discount horizon is a real lever here: at 15
    # decisions per second (frame_skip 4) gamma=0.99 reaches ~100 steps while
    # gamma=0.995 reaches ~200. Measured on this machine the long horizon was not
    # the bottleneck, so the SB3 default stays - raise it for longer fights.
    gamma: float = 0.99
    frame_skip: int = 4
    # Perception curriculum: easy perception first, the honest full model later.
    # Measured: with `noisy` the kill rate rose to 27% in 100k steps, with the
    # final `coarse_los` model to 15% - the last phase still trains the real thing.
    vision_curriculum: bool = True
    # Automatic evaluation right after the run, so a result is never just a
    # training number that nobody verified.
    eval_after_training: bool = True
    eval_episodes: int = 8
    # The same rungs the curriculum climbs: without `mover` the report cannot say
    # whether the policy learned to track a moving target before it has to survive
    # return fire - exactly the rung the phase-2 step measured as the weak spot.
    eval_bots: tuple[str, ...] = ("stationary", "mover", "walker", "full")
    seed: int = 2026
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def __post_init__(self) -> None:
        self.duration_seconds = max(0.0, float(self.duration_seconds))
        self.total_timesteps = max(2_048, int(self.total_timesteps))
        self.n_workers = max(1, int(self.n_workers))
        self.envs_per_worker = max(1, int(self.envs_per_worker))
        self.max_envs = max(1, int(self.max_envs))
        # Shorter episodes mean more finished fights per minute: hiding until the
        # time limit is the worst case for reinforcement learning.
        self.episode_seconds = min(300.0, max(10.0, float(self.episode_seconds)))
        self.curriculum_min_win_rate = min(1.0, max(0.0, float(self.curriculum_min_win_rate)))
        self.gamma = min(0.9999, max(0.9, float(self.gamma)))
        self.frame_skip = max(1, min(8, int(self.frame_skip)))
        self.eval_episodes = max(1, min(50, int(self.eval_episodes)))
        self.best_window_episodes = max(20, min(200, int(self.best_window_episodes)))
        self.eval_bots = tuple(self.eval_bots) or ("stationary",)
        if not isinstance(self.eval_bots, tuple):
            self.eval_bots = tuple(self.eval_bots)
        self.models_dir = Path(self.models_dir)
        self.logs_dir = Path(self.logs_dir)
        from env.shooter_env import VISION_MODES

        if self.vision_mode not in VISION_MODES:
            raise ValueError(
                f"Unknown vision_mode {self.vision_mode!r}. Choose one of: {', '.join(VISION_MODES)}"
            )


class TrainingController:
    """Owns one non-blocking PPO training thread and exposes safe snapshots."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._csv_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.save_requested = threading.Event()
        self._status = "stopped"
        self._evaluation: dict[str, Any] | None = None
        self._logs: deque[str] = deque(maxlen=300)
        self._metrics: dict[str, Any] = {
            "timesteps": 0,
            "fps": 0.0,
            "episodes": 0,
            "win_rate": 0.0,
            "kill_rate": 0.0,
            "avg_reward": 0.0,
            "avg_ttk": 0.0,
            "headshot_pct": 0.0,
            "accuracy": 0.0,
            "elapsed": 0.0,
            "progress": 0.0,
            "effective_envs": 0,
        }
        self._started_at: float | None = None
        self._error: str | None = None
        self._latest_checkpoint: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self.stop_event.clear()
            self.resume_event.set()
            self.save_requested.clear()
            self._status = "starting"
            self._error = None
            self._evaluation = None
            self._started_at = time.monotonic()
            self._logs.clear()
            self._log_locked("Starting CPU PPO trainer.")
            self._thread = threading.Thread(target=self._run, name="neural-arena-ppo", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self._status in {"running", "starting"}:
                self.resume_event.clear()
                self._status = "paused"
                self._log_locked("Training paused; workers will stop at the next vector step.")

    def resume(self) -> None:
        with self._lock:
            if self._status == "paused":
                self.resume_event.set()
                self._status = "running"
                self._log_locked("Training resumed.")

    def stop(self) -> None:
        self.stop_event.set()
        self.resume_event.set()
        with self._lock:
            if self._status not in {"complete", "error", "stopped"}:
                self._status = "stopping"
                self._log_locked("Graceful stop requested.")

    def request_save(self) -> None:
        self.save_requested.set()
        self.log("Checkpoint requested; it will be saved at the next PPO callback.")

    def log(self, message: str) -> None:
        with self._lock:
            self._log_locked(message)

    def _log_locked(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{stamp}] {message}")

    def update(self, *, status: str | None = None, metrics: dict[str, Any] | None = None,
               error: str | None = None, checkpoint: str | None = None,
               evaluation: dict[str, Any] | None = None) -> None:
        with self._lock:
            if status is not None:
                self._status = status
            if metrics:
                self._metrics.update(metrics)
            if error is not None:
                self._error = error
            if checkpoint is not None:
                self._latest_checkpoint = checkpoint
            if evaluation is not None:
                self._evaluation = evaluation

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            alive = bool(self._thread and self._thread.is_alive())
            return {
                "status": self._status,
                "metrics": dict(self._metrics),
                "logs": list(self._logs),
                "error": self._error,
                "latest_checkpoint": self._latest_checkpoint,
                "evaluation": self._evaluation,
                "thread_alive": alive,
                "started_at": self._started_at,
                "paused": self._status == "paused",
                "stopping": self._status == "stopping",
            }

    def _run(self) -> None:
        if not CPU_JOB_LOCK.acquire(blocking=False):
            message = "Another CPU-heavy training or benchmark job is already active. Stop it before starting this run."
            self.update(status="error", error=message)
            self.log(message)
            return
        try:
            train_ppo(self.config, self)
            with self._lock:
                if self._status not in {"complete", "stopped", "error"}:
                    self._status = "stopped" if self.stop_event.is_set() else "complete"
                self._log_locked(f"Training thread exited with status: {self._status}.")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._error = message
                self._status = "error"
                self._log_locked(f"Training failed: {message}")
        finally:
            CPU_JOB_LOCK.release()


def linear_schedule(initial_value: float):
    """Stable-Baselines3-compatible linear learning-rate schedule."""
    def schedule(progress_remaining: float) -> float:
        return float(initial_value) * max(0.0, min(1.0, float(progress_remaining)))
    return schedule


def export_policy_snapshot(model: Any, vec_normalize: Any | None = None) -> dict[str, Any]:
    """Export the actor MLP as NumPy arrays for a torch-free env subprocess."""
    import numpy as np
    import torch
    from torch import nn

    policy_net = model.policy.mlp_extractor.policy_net
    output_layer = model.policy.action_net
    modules = [module for module in policy_net.modules() if isinstance(module, nn.Linear)]
    if isinstance(output_layer, nn.Linear):
        modules.append(output_layer)
    layers = []
    for module in modules:
        layers.append({
            "weight": module.weight.detach().cpu().numpy().astype(np.float32, copy=True),
            "bias": module.bias.detach().cpu().numpy().astype(np.float32, copy=True),
        })
    if vec_normalize is not None and getattr(vec_normalize, "obs_rms", None) is not None:
        obs_mean = np.asarray(vec_normalize.obs_rms.mean, dtype=np.float32).copy()
        obs_var = np.asarray(vec_normalize.obs_rms.var, dtype=np.float32).copy()
    else:
        obs_mean = np.zeros(int(model.observation_space.shape[0]), dtype=np.float32)
        obs_var = np.ones_like(obs_mean)
    from env.shooter_env import ACTION_NVECS
    return {"layers": layers, "obs_mean": obs_mean, "obs_var": obs_var,
            "action_nvec": tuple(ACTION_NVECS)}


VISION_CURRICULUM: dict[int, str] = {1: "noisy", 2: "coarse", 3: "coarse_los", 4: "coarse_los"}
"""Perception per curriculum phase: learn to shoot under easy perception first.

Phase 1-2 stay coarse but forgiving (noise / no cover gating), so the policy can
learn to aim and kill at all; phases 3-4 run the honest model that the arena uses.
"""


def vision_mode_for_phase(phase: int, fallback: str = "coarse_los") -> str:
    """Vision mode the trainer uses in a curriculum phase."""
    return VISION_CURRICULUM.get(min(4, max(1, int(phase))), fallback)


def _csv_columns(path: Path, fieldnames: list[str]) -> list[str]:
    """Column layout of a metrics CSV, widened when new fields show up.

    Metrics are appended for the lifetime of the project, so the columns grow.
    Earlier versions wrote the header only once and ran with
    ``extrasaction="ignore"`` – every field added later (``kill_rate``) was then
    silently dropped because the file already had an old header (measured: the
    dashboard series stayed empty although the trainer tracked the value).
    """
    if not path.exists() or path.stat().st_size == 0:
        return list(fieldnames)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            existing = [name for name in next(reader)]
        except StopIteration:
            return list(fieldnames)
    if set(existing) >= set(fieldnames):
        return existing
    columns = existing + [name for name in fieldnames if name not in existing]
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return columns


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        columns = _csv_columns(path, fieldnames)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def curriculum_window_ready(kill_history, gate: float, min_episodes: int = 20) -> bool:
    """Whether the last episodes of the *running* phase beat the gate.

    The window must contain episodes of this phase only: right after a switch the
    mixed window still holds the successes of the easier phase, which measured
    advanced the curriculum on stale numbers (36.7 % "kill rate" seven seconds
    after a step back, immediately followed by 0 %).
    """
    recent = list(kill_history)[-30:]
    if len(recent) < min_episodes:
        return False
    return (sum(float(value) for value in recent) / len(recent)) >= gate


def curriculum_next_phase(current_phase: int, ready: bool, progress_phase: int,
                          max_phase: int = 4) -> int:
    """One rung at a time - a jump skips the lesson of the skipped phase.

    The phase schedule follows the training progress, so late in a run it already
    asks for phase 3/4. Measured: after a step back to phase 1 the next satisfied
    window jumped *straight* to phase 3 (`walker`, shooting back) and produced
    0 % kills again - the ladder must be climbed step by step.
    """
    if not ready or progress_phase <= current_phase:
        return current_phase
    return min(int(max_phase), current_phase + 1)


def curriculum_backtrack_needed(*, current_phase: int, episodes_in_phase: int,
                                recent_kill_rate: float, gate: float,
                                min_episodes: int = 40) -> bool:
    """Whether the curriculum should undo a phase that cannot be won.

    Measured twice: after the scripted opponent started shooting back, both runs
    collapsed to 0 % wins with ~4 s episodes and never recovered within the
    budget. A curriculum that cannot step back throws a working policy away.
    """
    if current_phase <= 1 or episodes_in_phase < min_episodes:
        return False
    return recent_kill_rate < gate / 2.0


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    """Read the sidecar stamp of a checkpoint (empty dict when there is none)."""
    meta_path = checkpoint.with_name(f"{checkpoint.stem}_meta.json")
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _evaluate_checkpoint(
    config: TrainingConfig,
    job: TrainingController,
    checkpoint: Path,
    *,
    vision_mode: str,
    phase: int,
    seed: int,
    key: str | None = None,
    label: str = "",
) -> dict[str, Any] | None:
    """Play one checkpoint against every configured opponent.

    The report is written after *every* matchup and pushed into the job snapshot,
    so a crash (the sandbox can abort a process while it tears down its workers)
    cannot erase a verification that was already measured.
    """
    # ``train_ppo`` imports numpy locally, so the module-level helpers must not
    # assume it is available.
    import numpy as np

    from training.evaluation import evaluate, summarize

    meta = _checkpoint_metadata(checkpoint)
    episode_seconds = float(meta.get("episode_seconds") or config.episode_seconds)
    frame_skip = max(1, int(meta.get("frame_skip") or config.frame_skip))
    try:
        from training.evaluation import load_policy

        model, normalizer = load_policy(checkpoint)
    except Exception as exc:
        job.log(f"{label}Automatic evaluation failed to load {checkpoint.name}: {exc}")
        return None
    runs: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": checkpoint.name,
        "phase": phase,
        "vision_mode": vision_mode,
        "episodes_per_matchup": config.eval_episodes,
        "runs": runs,
    }
    try:
        for bot in config.eval_bots:
            result = evaluate(
                model=model, normalizer=normalizer, bot=bot, phase=phase,
                episodes=config.eval_episodes, vision_mode=vision_mode,
                map_name=config.map_name, episode_seconds=episode_seconds,
                frame_skip=frame_skip, seed=seed,
            )
            result["policy"] = str(checkpoint)
            runs.append(result)
            job.log("  " + (f"{label}" if label else "") + summarize(result))
            payload["win_rate_overall"] = float(np.mean([run["win_rate"] for run in runs]))
            payload["kill_rate_overall"] = float(np.mean([run["kill_rate"] for run in runs]))
            best = max(runs, key=lambda run: (run["kill_rate"], run["win_rate"]))
            payload["best"] = {"bot": best["bot"], "phase": best["phase"],
                               "win_rate": best["win_rate"], "kill_rate": best["kill_rate"]}
            _store_evaluation(config, job, payload, key)
    except Exception as exc:  # a failed matchup must not kill a finished run
        # Keep the reason next to the partial report: the run's log is a bounded
        # buffer, so a message written here can be gone by the time anyone reads
        # the report (measured: the evaluation stopped after the first opponent
        # and the log line was already rotated away).
        payload["error"] = f"{type(exc).__name__}: {exc}"
        job.log(f"{label}Automatic evaluation failed: {exc}")
        if not runs:
            return None
        _store_evaluation(config, job, payload, key)
        return payload
    return payload


def _store_evaluation(config: TrainingConfig, job: TrainingController,
                      payload: dict[str, Any], key: str | None) -> None:
    """Publish the (partial) evaluation to the panel and to ``models/``."""
    snapshot = dict(payload)
    if key:
        # A second checkpoint is measured under the *target* perception; it is
        # attached to the existing report instead of overwriting it.
        target = dict(payload)
        target.pop("target", None)
        target["runs"] = list(payload.get("runs") or [])
        current = dict(job.snapshot().get("evaluation") or {})
        current["target"] = target
        snapshot = current
    try:
        (config.models_dir / "best_model_eval.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8")
    except OSError as exc:
        job.log(f"Could not store the evaluation report: {exc}")
    job.update(evaluation=snapshot)


def _evaluate_best_checkpoint(config: TrainingConfig, job: TrainingController) -> None:
    """Verify the retained checkpoint by playing it - training numbers are not proof.

    A run can look fine internally and still lose every fight; this plays the
    best checkpoint against the bot and stores the result next to the model, so
    "does it actually work" is answered by matches, not by the loss curve.
    """
    checkpoint = config.models_dir / "best_model.zip"
    if not checkpoint.is_file():
        final_model = config.models_dir / "final_model.zip"
        if final_model.is_file():
            checkpoint = final_model
    if not checkpoint.is_file():
        # ``Path("")`` would silently mean "." and then looks like a file that
        # exists - only a real path from the job may be used as a fallback.
        candidate = job.snapshot().get("checkpoint")
        candidate_path = Path(candidate) if isinstance(candidate, str) and candidate else None
        if candidate_path is not None and candidate_path.is_file():
            checkpoint = candidate_path
    if not checkpoint.is_file():
        job.log("Automatic evaluation skipped: no checkpoint was written.")
        return
    meta = _checkpoint_metadata(checkpoint)
    phase = min(4, max(1, int(meta.get("phase", 4) or 4)))
    # The checkpoint was trained with *its* perception model and episode length -
    # measuring it under the final settings of the run would compare apples with
    # oranges (the vision curriculum starts with the forgiving `noisy` mode).
    vision_mode = str(meta.get("vision_mode") or config.vision_mode)
    job.log(f"Automatic evaluation of {checkpoint.name} "
            f"({config.eval_episodes} episodes per opponent, phase {phase}, "
            f"vision_mode={vision_mode}) ...")
    report = _evaluate_checkpoint(config, job, checkpoint, vision_mode=vision_mode,
                                 phase=phase, seed=config.seed)
    if report is None:
        return
    job.log(f"Evaluation summary: {report.get('win_rate_overall', 0.0):.0%} wins, "
            f"{report.get('kill_rate_overall', 0.0):.0%} kills (Ø over all opponents, "
            f"vision_mode={vision_mode}).")

    # With the perception curriculum the retained checkpoint may still be a
    # phase-1 model (forgiving `noisy` vision). The honest question - "does the
    # target perception work?" - is answered by the *final* model under the target
    # mode, so it is measured as well instead of being left to the reader.
    final_model = config.models_dir / "final_model.zip"
    if not final_model.is_file() or final_model == checkpoint:
        return
    final_meta = _checkpoint_metadata(final_model)
    final_mode = str(final_meta.get("vision_mode") or config.vision_mode)
    if final_mode == vision_mode:
        return
    final_phase = min(4, max(1, int(final_meta.get("phase", phase) or phase)))
    job.log(f"Target perception check: {final_model.name} with vision_mode={final_mode} ...")
    _evaluate_checkpoint(config, job, final_model, vision_mode=final_mode, phase=final_phase,
                         seed=config.seed + 1000, key="target", label="target ")


def train_ppo(config: TrainingConfig, job: TrainingController) -> None:
    """Build and train a CPU-only PPO policy, called by the background thread."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    physical, logical = cpu_core_counts()
    torch.set_num_threads(max(1, min(4, physical)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    vec_env_base, parallelism = make_vector_env(
        n_workers=config.n_workers,
        envs_per_worker=config.envs_per_worker,
        map_name=config.map_definition or config.map_name,
        seed=config.seed,
        curriculum=config.curriculum,
        initial_phase=1 if config.curriculum else 4,
        max_envs=config.max_envs,
        vision_mode=config.vision_mode,
        max_episode_seconds=config.episode_seconds,
        frame_skip=config.frame_skip,
    )
    job.update(metrics={"effective_envs": parallelism.effective_envs})
    if parallelism.effective_envs < config.n_workers * config.envs_per_worker:
        job.log(
            f"Requested {config.n_workers * config.envs_per_worker} envs; capped at "
            f"{parallelism.effective_envs} for CPU/memory safety."
        )

    resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    vec_normalize_path = None
    if resume_path:
        vec_normalize_path = resume_path.with_name(f"{resume_path.stem}_vecnormalize.pkl")
    if vec_normalize_path is not None and vec_normalize_path.exists():
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env_base)
        vec_env.training = True
        vec_env.norm_reward = True
        job.log(
            f"Loaded observation statistics from {vec_normalize_path.name} "
            f"(norm_obs={bool(vec_env.norm_obs)})."
        )
    else:
        # The perception observation is already bounded to [-1, 1] and its zeros
        # are meaningful ("never seen"), so running-statistic normalisation would
        # rescale exactly the values the policy has to interpret. Rewards are
        # still normalised.
        vec_env = VecNormalize(vec_env_base, norm_obs=False, norm_reward=True, clip_obs=10.0)

    model = None
    callback = None
    try:
        if (config.method.lower().startswith("imitation") and config.imitation_path
                and not Path(config.imitation_path).exists()):
            from training.imitation import train_behavior_cloning
            result = train_behavior_cloning(
                dataset_path=PROJECT_ROOT / "data" / "demos.csv",
                output_path=config.imitation_path,
            )
            torch.set_num_threads(max(1, min(4, physical)))
            job.log(
                f"Behavior cloning trained on {result['samples']} demos "
                f"(per-action accuracy {float(result['accuracy']):.1%})."
            )
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
            model = PPO.load(str(resume_path), env=vec_env, device="cpu")
            job.log(f"Resuming PPO policy from {resume_path.name}.")
        else:
            policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=linear_schedule(3e-4),
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=config.gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                seed=config.seed,
                verbose=0,
                device="cpu",
            )
            if config.imitation_path:
                from training.imitation import load_behavior_clone_into_policy
                load_behavior_clone_into_policy(model, config.imitation_path)
                job.log(f"Initialized the PPO actor from {Path(config.imitation_path).name}.")

        class ArenaTrainingCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.started_at = 0.0
                self.last_update = 0.0
                self.last_csv = 0.0
                self.base_timesteps = int(getattr(model, "num_timesteps", 0))
                self.next_checkpoint = (self.base_timesteps // 50_000 + 1) * 50_000
                self.current_phase = 1 if config.curriculum else 4
                self.wins: deque[float] = deque(maxlen=100)
                self.last_curriculum_log = 0.0
                self.kill_wins: deque[float] = deque(maxlen=100)
                self.episode_rewards: deque[float] = deque(maxlen=100)
                self.ttks: deque[float] = deque(maxlen=100)
                self.kill_ttks: deque[float] = deque(maxlen=100)
                self.accuracies: deque[float] = deque(maxlen=100)
                self.headshot_rates: deque[float] = deque(maxlen=100)
                self.episode_count = 0
                self.best_win_rate, self.best_kill_rate = self._load_best_score()
                self.best_seen_kill = self.best_kill_rate
                self.last_best_episode = 0
                self.vision_mode = config.vision_mode
                # Curriculum state: the gate currently in force and the episode
                # count when the running phase started (both drive the step-back).
                self.phase_gate = config.curriculum_min_win_rate
                self.phase_enter_episode = 0
                # Kill flags since the current phase started: the gate must not
                # judge a new phase by the successes of the previous one.
                self.phase_kills: list[float] = []
                self.metric_fields = [
                    "timestamp", "steps", "fps", "episodes", "win_rate", "avg_reward",
                    "avg_ttk", "headshot_pct", "accuracy", "elapsed", "map",
                    "kill_rate",
                ]
                self.event_fields = [
                    "timestamp", "episode", "map", "win", "draw", "killed", "ttk", "weapon",
                    "opponent_weapon", "distance", "shots_fired", "bullets_fired", "hits", "headshots",
                    "accuracy", "headshot_pct", "avg_kill_distance", "death_x", "death_y",
                    "kill_x", "kill_y",
                ]

            def _load_best_score(self) -> tuple[float, float]:
                """Return (kill rate, win rate) of the retained checkpoint.

                Checkpoints saved before the kill rate existed are scored with a
                kill rate of 0, so a new run has to produce a confirmed kill
                before it may overwrite an already useful model.
                """
                metadata = config.models_dir / "best_model.json"
                if metadata.exists():
                    try:
                        with metadata.open(encoding="utf-8") as handle:
                            payload = json.load(handle)
                        return (float(payload.get("best_kill_rate", 0.0)),
                                float(payload.get("best_win_rate", -1.0)))
                    except (OSError, ValueError, TypeError):
                        pass
                return -1.0, -1.0

            def _on_training_start(self) -> None:
                self.started_at = time.monotonic()
                self.last_update = self.started_at
                self.last_csv = self.started_at
                job.update(status="running")
                job.log(
                    f"PPO live: {parallelism.effective_envs} vector envs, CPU device, "
                    f"{config.map_name} map."
                )
                self.vision_mode = config.vision_mode
                if config.curriculum and config.vision_curriculum:
                    self.vision_mode = vision_mode_for_phase(1, config.vision_mode)
                    try:
                        self.training_env.env_method("set_vision_mode", self.vision_mode)
                        job.log(f"Perception curriculum: phase 1 trains with vision_mode="
                                f"{self.vision_mode}.")
                    except Exception as exc:
                        job.log(f"Perception curriculum notice: {exc}")
                if config.curriculum:
                    try:
                        self.training_env.env_method("set_curriculum_phase", 1)
                    except Exception as exc:
                        job.log(f"Curriculum initialization notice: {exc}")

            def _recent_win_rate(self, window: int = 30) -> float:
                recent = list(self.wins)[-window:]
                return float(np.mean(recent)) if recent else 0.0

            def _recent_kill_rate(self, window: int = 30, history=None) -> float:
                """Share of recent episodes that ended in a *confirmed* kill.

                Winning on health at the time limit is not the same as winning:
                it would reward hiding, so it never unlocks the next phase.
                ``history`` selects the episodes to judge - the curriculum passes
                the kill flags of the *running* phase (see
                :func:`curriculum_window_ready`).
                """
                recent = list((self.kill_wins if history is None else history))[-window:]
                return float(np.mean(recent)) if recent else 0.0

            def _curriculum_ready(self) -> bool:
                """True only when the running phase is demonstrably beaten."""
                if not config.curriculum or self.current_phase >= 4:
                    return False
                return curriculum_window_ready(self.phase_kills, self.phase_gate)

            def _curriculum_backtrack(self) -> bool:
                """Undo a phase that only produces defeats.

                Measured twice: after the opponent started shooting back (phase 3),
                both runs collapsed to 0 % wins with ~4 s episodes and never
                recovered inside the budget. A curriculum that cannot step back
                throws away a working policy, so the run returns to the last phase
                it could beat and asks for a *higher* kill rate before retrying.
                """
                if not config.curriculum:
                    return False
                if not curriculum_backtrack_needed(
                    current_phase=self.current_phase,
                    episodes_in_phase=self.episode_count - self.phase_enter_episode,
                    recent_kill_rate=self._recent_kill_rate(history=self.phase_kills),
                    gate=self.phase_gate,
                ):
                    return False
                back = self.current_phase - 1
                self.phase_gate = min(0.8, self.phase_gate + 0.1)
                self.current_phase = back
                self.phase_enter_episode = self.episode_count
                self.phase_kills.clear()
                job.log(
                    f"Curriculum step back to phase {back}/4: only "
                    f"{self._recent_kill_rate(history=self.phase_kills):.1%} confirmed "
                    f"kills in this phase (gate for the next attempt: "
                    f"{self.phase_gate:.0%})."
                )
                if config.vision_curriculum:
                    self.vision_mode = vision_mode_for_phase(back, config.vision_mode)
                    try:
                        self.training_env.env_method("set_vision_mode", self.vision_mode)
                    except Exception as exc:
                        job.log(f"Could not update one or more perception models: {exc}")
                try:
                    self.training_env.env_method("set_curriculum_phase", back)
                except Exception as exc:
                    job.log(f"Could not update one or more environment phases: {exc}")
                return True

            def _phase_from_progress(self) -> int:
                if not config.curriculum:
                    return 4
                relative = max(0, self.num_timesteps - self.base_timesteps)
                progress = min(0.999999, relative / max(1, config.total_timesteps))
                return min(4, int(progress * 4.0) + 1)

            def _on_step(self) -> bool:
                if job.stop_event.is_set():
                    job.update(status="stopped")
                    return False
                while not job.resume_event.wait(0.1):
                    if job.stop_event.is_set():
                        job.update(status="stopped")
                        return False
                if config.duration_seconds > 0 and time.monotonic() - self.started_at >= config.duration_seconds:
                    job.log("Selected training duration reached.")
                    job.update(status="complete")
                    return False

                infos = self.locals.get("infos", [])
                for info in infos:
                    episode_metrics = info.get("episode_metrics")
                    if not episode_metrics:
                        continue
                    self.episode_count += 1
                    win = float(episode_metrics.get("win", 0.0))
                    self.wins.append(win)
                    killed = bool(episode_metrics.get("killed"))
                    self.kill_wins.append(win if killed else 0.0)
                    self.phase_kills.append(1.0 if killed else 0.0)
                    monitor_episode = info.get("episode", {})
                    reward = float(monitor_episode.get("r", 0.0))
                    self.episode_rewards.append(reward)
                    self.ttks.append(float(episode_metrics.get("ttk", 0.0)))
                    if episode_metrics.get("killed"):
                        self.kill_ttks.append(float(episode_metrics.get("ttk", 0.0)))
                    self.accuracies.append(float(episode_metrics.get("accuracy", 0.0)))
                    self.headshot_rates.append(float(episode_metrics.get("headshot_pct", 0.0)))
                    event = {**episode_metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
                    _append_csv(config.logs_dir / "heatmap_events.csv", self.event_fields,
                                event, job._csv_lock)

                phase = self._phase_from_progress()
                ready = self._curriculum_ready()
                phase_kill_rate = self._recent_kill_rate(history=self.phase_kills)
                if phase > self.current_phase and not ready:
                    # ``now`` is only computed further down; the gate needs its
                    # own clock so the "hold" notice cannot crash the callback.
                    hold_now = time.monotonic()
                    if hold_now - self.last_curriculum_log >= 20.0:
                        self.last_curriculum_log = hold_now
                        # Name the real reason: too few episodes of this phase
                        # (the window must not judge the previous one) or a kill
                        # rate below the gate - the old wording claimed "below the
                        # gate" even when the rate was 100 % and only the count
                        # was missing.
                        job.log(
                            "Curriculum holds at phase "
                            f"{self.current_phase}/4 - {len(self.phase_kills)} episodes "
                            f"in this phase, kill rate {phase_kill_rate:.1%} "
                            f"(gate {config.curriculum_min_win_rate:.0%})."
                        )
                    phase = self.current_phase
                # One rung at a time: the schedule from the progress is only a
                # ceiling, so a satisfied window cannot jump over a phase.
                phase = curriculum_next_phase(self.current_phase, ready, phase)
                if phase != self.current_phase:
                    self.current_phase = phase
                    self.phase_enter_episode = self.episode_count
                    self.phase_kills.clear()
                    if config.vision_curriculum:
                        self.vision_mode = vision_mode_for_phase(phase, config.vision_mode)
                        try:
                            self.training_env.env_method("set_vision_mode", self.vision_mode)
                        except Exception as exc:
                            job.log(f"Could not update one or more perception models: {exc}")
                    # Evidence that opened the gate: the kill rate of the phase
                    # that was just left.
                    job.log(f"Curriculum advanced to phase {phase}/4 "
                            f"(kill rate {phase_kill_rate:.1%}, "
                            f"vision_mode={self.vision_mode}).")
                    try:
                        self.training_env.env_method("set_curriculum_phase", phase)
                    except Exception as exc:
                        job.log(f"Could not update one or more environment phases: {exc}")
                    if phase == 4 and config.self_play:
                        self._activate_self_play()
                else:
                    self._curriculum_backtrack()

                if job.save_requested.is_set():
                    job.save_requested.clear()
                    self._save_checkpoint("manual", force=True)

                now = time.monotonic()
                elapsed = max(1e-6, now - self.started_at)
                fps = max(0.0, (self.num_timesteps - self.base_timesteps) / elapsed)
                win_rate = float(np.mean(self.wins)) if self.wins else 0.0
                kill_rate = float(np.mean(self.kill_wins)) if self.kill_wins else 0.0
                avg_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
                # Time-to-kill only counts episodes that ended in a confirmed kill;
                # time-limit decisions and defeats are excluded.
                avg_ttk = float(np.mean(self.kill_ttks)) if self.kill_ttks else 0.0
                accuracy = float(np.mean(self.accuracies)) if self.accuracies else 0.0
                headshot_pct = float(np.mean(self.headshot_rates)) if self.headshot_rates else 0.0
                progress = min(1.0, max(0.0, (self.num_timesteps - self.base_timesteps)
                                        / max(1, config.total_timesteps)))
                metrics = {
                    "timesteps": int(self.num_timesteps),
                    "fps": fps,
                    "episodes": self.episode_count,
                    "win_rate": win_rate,
                    "kill_rate": kill_rate,
                    "avg_reward": avg_reward,
                    "avg_ttk": avg_ttk,
                    "headshot_pct": headshot_pct,
                    "accuracy": accuracy,
                    "elapsed": elapsed,
                    "progress": progress,
                    "effective_envs": parallelism.effective_envs,
                }
                if now - self.last_update >= 1.0:
                    self._update_best_watch()
                    job.update(metrics=metrics)
                    job.update(status="paused" if not job.resume_event.is_set() else "running")
                    job.log(
                        f"steps={self.num_timesteps:,} | {fps:,.0f} steps/s | "
                        f"episodes={self.episode_count} | win={win_rate:.1%} "
                        f"(kills {kill_rate:.1%})"
                    )
                    self.last_update = now
                if now - self.last_csv >= 2.0:
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "steps": self.num_timesteps,
                        "fps": round(fps, 4),
                        "episodes": self.episode_count,
                        "win_rate": round(win_rate, 6),
                        "kill_rate": round(kill_rate, 6),
                        "avg_reward": round(avg_reward, 6),
                        "avg_ttk": round(avg_ttk, 6),
                        "headshot_pct": round(headshot_pct, 6),
                        "accuracy": round(accuracy, 6),
                        "elapsed": round(elapsed, 3),
                        "map": config.map_name,
                    }
                    _append_csv(config.logs_dir / "training_metrics.csv", self.metric_fields,
                                row, job._csv_lock)
                    self.last_csv = now
                return True

            def _on_training_end(self) -> None:
                elapsed = max(1e-6, time.monotonic() - self.started_at) if self.started_at else 0.0
                fps = max(0.0, (self.num_timesteps - self.base_timesteps) / max(1e-6, elapsed))
                win_rate = float(np.mean(self.wins)) if self.wins else 0.0
                kill_rate = float(np.mean(self.kill_wins)) if self.kill_wins else 0.0
                avg_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
                # Time-to-kill only counts episodes that ended in a confirmed kill;
                # time-limit decisions and defeats are excluded.
                avg_ttk = float(np.mean(self.kill_ttks)) if self.kill_ttks else 0.0
                accuracy = float(np.mean(self.accuracies)) if self.accuracies else 0.0
                headshot_pct = float(np.mean(self.headshot_rates)) if self.headshot_rates else 0.0
                final_metrics = {
                    "timesteps": int(self.num_timesteps), "fps": fps,
                    "episodes": self.episode_count, "win_rate": win_rate,
                    "avg_reward": avg_reward, "avg_ttk": avg_ttk,
                    "headshot_pct": headshot_pct, "accuracy": accuracy,
                    "elapsed": elapsed, "progress": min(1.0, max(0.0,
                        (self.num_timesteps - self.base_timesteps) / max(1, config.total_timesteps))),
                    "effective_envs": parallelism.effective_envs,
                }
                job.update(metrics=final_metrics)
                _append_csv(config.logs_dir / "training_metrics.csv", self.metric_fields, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "steps": self.num_timesteps, "fps": round(fps, 4),
                    "episodes": self.episode_count, "win_rate": round(win_rate, 6),
                    "kill_rate": round(kill_rate, 6),
                    "avg_reward": round(avg_reward, 6), "avg_ttk": round(avg_ttk, 6),
                    "headshot_pct": round(headshot_pct, 6), "accuracy": round(accuracy, 6),
                    "elapsed": round(elapsed, 3), "map": config.map_name,
                }, job._csv_lock)

            def _activate_self_play(self) -> None:
                try:
                    best_path = config.models_dir / "best_model.zip"
                    frozen_model = PPO.load(str(best_path), device="cpu") if best_path.exists() else self.model
                    snapshot = export_policy_snapshot(frozen_model, self.training_env)
                    self.training_env.env_method("set_opponent_snapshot", snapshot)
                    source = best_path.name if best_path.exists() else "phase-three policy snapshot"
                    job.log(f"Frozen self-play opponent activated from {source}.")
                except Exception as exc:
                    job.log(f"Self-play snapshot could not be installed; phase-four bot remains active: {exc}")

            def _update_best_watch(self) -> None:
                """Retain the checkpoint with the highest recent kill rate.

                Kills come first: a win on health at the time limit rewards hiding
                and would keep a worse model. The rolling window is checked
                continuously instead of only at fixed step boundaries, because a
                50k boundary can fall into a bad phase and then the *only*
                checkpoint on disk is a worthless one.
                """
                # The window is a real trade-off: too short and single lucky kills
                # win, too long and a peak is averaged away. Any improvement over
                # the retained score qualifies, because the 15-episode spacing
                # already limits how often a checkpoint may be written.
                window = config.best_window_episodes
                recent = list(self.kill_wins)[-window:]
                if len(recent) < max(20, window * 3 // 5):
                    return
                kill = float(np.mean(recent))
                if kill <= self.best_seen_kill:
                    return
                if self.episode_count - self.last_best_episode < 15:
                    return
                self.best_seen_kill = kill
                self.last_best_episode = self.episode_count
                rate = float(np.mean(self.wins))
                self.best_kill_rate = max(self.best_kill_rate, kill)
                self.best_win_rate = max(self.best_win_rate, rate)
                self._save_checkpoint("best", force=True, win_rate=rate)
                job.log(f"New best checkpoint retained: {kill:.1%} kills in the last "
                        f"{len(recent)} episodes ({rate:.1%} wins).")

            def _save_checkpoint(self, kind: str, force: bool, win_rate: float | None = None) -> None:
                if not force:
                    return
                if kind == "best":
                    base = config.models_dir / "best_model"
                elif kind == "final":
                    base = config.models_dir / "final_model"
                else:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = config.models_dir / f"manual_{stamp}"
                model_path = base.with_suffix(".zip")
                self.model.save(str(base))
                normalizer_path = base.with_name(f"{base.name}_vecnormalize.pkl")
                self.training_env.save(str(normalizer_path))
                from env.shooter_env import OBSERVATION_VERSION

                metadata = {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "timesteps": int(self.num_timesteps),
                    "win_rate": win_rate if win_rate is not None else (
                        float(np.mean(self.wins)) if self.wins else 0.0
                    ),
                    "kill_rate": float(np.mean(self.kill_wins)) if self.kill_wins else 0.0,
                    "map": config.map_name,
                    "phase": self.current_phase,
                    # Policy checkpoints are only valid for the observation
                    # layout they were trained on (see server/policies.py).
                    "observation_version": OBSERVATION_VERSION,
                    "vision_mode": getattr(self, "vision_mode", config.vision_mode),
                    "vision_curriculum": bool(config.vision_curriculum),
                    "frame_skip": config.frame_skip,
                    "episode_seconds": config.episode_seconds,
                }
                with base.with_name(f"{base.name}_meta.json").open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)
                if kind == "best":
                    metadata["best_win_rate"] = float(metadata["win_rate"])
                    metadata["best_kill_rate"] = float(metadata["kill_rate"])
                    with (config.models_dir / "best_model.json").open("w", encoding="utf-8") as handle:
                        json.dump(metadata, handle, indent=2)
                job.update(checkpoint=str(model_path))
                job.log(f"Saved {kind} model and VecNormalize state to {model_path.name}.")

        callback = ArenaTrainingCallback()
        job.update(status="running")
        job.log(
            f"Observation normalisation: {bool(vec_env.norm_obs)} (structured perception "
            f"values are not rescaled by default)."
        )
        job.log(
            f"PPO configured: lr=3e-4 linear, n_steps=2048, batch=256, epochs=10, "
            f"gamma={config.gamma}, gae=0.95, clip=0.2, ent_coef=0.01."
        )
        job.log(
            f"Perception: vision_mode={config.vision_mode} "
            f"(the agent only sees bearing sectors, distance bands and line of sight)."
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            reset_num_timesteps=resume_path is None,
            progress_bar=False,
        )
        stopped = job.stop_event.is_set()
        final_metrics = dict(job.snapshot()["metrics"])
        final_metrics["timesteps"] = int(model.num_timesteps)
        if callback.started_at:
            final_metrics["elapsed"] = time.monotonic() - callback.started_at
            final_metrics["fps"] = max(0.0, (model.num_timesteps - callback.base_timesteps)
                                         / max(1e-6, final_metrics["elapsed"]))
        job.update(metrics=final_metrics)
        if callback is not None:
            # Always leave the end state of the run on disk: it is the model a
            # user would load next, and the automatic evaluation compares it with
            # the retained "best" checkpoint under the target perception.
            try:
                callback._save_checkpoint("final", force=True)
            except Exception as exc:
                job.log(f"Final checkpoint could not be written: {exc}")
        if config.eval_after_training and not stopped:
            # The status only flips to "complete" after the verification, so a
            # client that stops watching on "complete" cannot miss the report (and
            # the panel shows the run as busy while the checkpoint is playing).
            _evaluate_best_checkpoint(config, job)
        job.update(status="stopped" if stopped else "complete")
    finally:
        if vec_env is not None:
            vec_env.close()
