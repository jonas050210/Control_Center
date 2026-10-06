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
    # Configurable, because the discount horizon is a real lever here: at 15
    # decisions per second (frame_skip 4) gamma=0.99 reaches ~100 steps while
    # gamma=0.995 reaches ~200. Measured on this machine the long horizon was not
    # the bottleneck, so the SB3 default stays - raise it for longer fights.
    gamma: float = 0.99
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
               error: str | None = None, checkpoint: str | None = None) -> None:
        with self._lock:
            if status is not None:
                self._status = status
            if metrics:
                self._metrics.update(metrics)
            if error is not None:
                self._error = error
            if checkpoint is not None:
                self._latest_checkpoint = checkpoint

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            alive = bool(self._thread and self._thread.is_alive())
            return {
                "status": self._status,
                "metrics": dict(self._metrics),
                "logs": list(self._logs),
                "error": self._error,
                "latest_checkpoint": self._latest_checkpoint,
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


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


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
        job.log(f"Loaded observation statistics from {vec_normalize_path.name}.")
    else:
        vec_env = VecNormalize(vec_env_base, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
                if config.curriculum:
                    try:
                        self.training_env.env_method("set_curriculum_phase", 1)
                    except Exception as exc:
                        job.log(f"Curriculum initialization notice: {exc}")

            def _recent_win_rate(self, window: int = 30) -> float:
                recent = list(self.wins)[-window:]
                return float(np.mean(recent)) if recent else 0.0

            def _recent_kill_rate(self, window: int = 30) -> float:
                """Share of recent episodes that ended in a *confirmed* kill.

                Winning on health at the time limit is not the same as winning:
                it would reward hiding, so it never unlocks the next phase.
                """
                recent = list(self.kill_wins)[-window:]
                return float(np.mean(recent)) if recent else 0.0

            def _curriculum_ready(self) -> bool:
                """True only when the running phase is demonstrably beaten."""
                if not config.curriculum or self.current_phase >= 4:
                    return False
                recent = list(self.kill_wins)[-30:]
                if len(recent) < 20:
                    return False
                return float(np.mean(recent)) >= config.curriculum_min_win_rate

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
                if phase > self.current_phase and not self._curriculum_ready():
                    # ``now`` is only computed further down; the gate needs its
                    # own clock so the "hold" notice cannot crash the callback.
                    hold_now = time.monotonic()
                    if hold_now - self.last_curriculum_log >= 20.0:
                        self.last_curriculum_log = hold_now
                        job.log(
                            "Curriculum holds at phase "
                            f"{self.current_phase}/4 - kill rate "
                            f"{self._recent_kill_rate():.1%} is below the "
                            f"{config.curriculum_min_win_rate:.0%} gate."
                        )
                    phase = self.current_phase
                if phase != self.current_phase:
                    self.current_phase = phase
                    job.log(f"Curriculum advanced to phase {phase}/4 "
                            f"(kill rate {self._recent_kill_rate():.1%}).")
                    try:
                        self.training_env.env_method("set_curriculum_phase", phase)
                    except Exception as exc:
                        job.log(f"Could not update one or more environment phases: {exc}")
                    if phase == 4 and config.self_play:
                        self._activate_self_play()

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
                recent = list(self.kill_wins)[-30:]
                if len(recent) < 20:
                    return
                kill = float(np.mean(recent))
                if kill <= self.best_seen_kill + 0.05:
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
                    "vision_mode": config.vision_mode,
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
        if job.stop_event.is_set():
            job.update(status="stopped")
        elif job.snapshot()["status"] != "complete":
            job.update(status="complete")
        final_metrics = dict(job.snapshot()["metrics"])
        final_metrics["timesteps"] = int(model.num_timesteps)
        if callback.started_at:
            final_metrics["elapsed"] = time.monotonic() - callback.started_at
            final_metrics["fps"] = max(0.0, (model.num_timesteps - callback.base_timesteps)
                                         / max(1e-6, final_metrics["elapsed"]))
        job.update(metrics=final_metrics)
    finally:
        if vec_env is not None:
            vec_env.close()
