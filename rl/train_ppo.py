"""PPO reinforcement learning with optional BC warm start and checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from bc.models import BCVisionNetwork
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry
from sandbox.env import make_sandbox_env


class BCFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor matching SandboxAI's lightweight BC visual encoder."""

    def __init__(
        self,
        observation_space: Any,
        latent_dim: int = 256,
        channel_scales: tuple[int, int, int] = (16, 32, 32),
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim)
        network = BCVisionNetwork(
            latent_dim=latent_dim, channel_scales=channel_scales, use_temporal_gru=False
        )
        self.encoder = network.encoder
        self.projection = network.fc_proj

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # CnnPolicy normalizes uint8 images to [0,1] before this method.
        return self.projection(self.encoder(observations))


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        tracker: Optional[ExperimentTracker],
        telemetry: Optional[SystemTelemetry],
        checkpoint_path: Path,
        log_interval: int = 256,
    ) -> None:
        super().__init__(verbose=0)
        self.tracker = tracker
        self.telemetry = telemetry
        self.checkpoint_path = checkpoint_path
        self.log_interval = max(1, log_interval)
        self.started = time.perf_counter()
        self.initial_timesteps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self._running_returns: Optional[np.ndarray] = None
        self._running_lengths: Optional[np.ndarray] = None

    def _on_training_start(self) -> None:
        env_count = self.training_env.num_envs
        self.initial_timesteps = int(self.num_timesteps)
        self.started = time.perf_counter()
        self._running_returns = np.zeros(env_count, dtype=np.float64)
        self._running_lengths = np.zeros(env_count, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        if self._running_returns is not None and rewards.size:
            self._running_returns += rewards
            self._running_lengths += 1
            for idx, done in enumerate(dones):
                if done:
                    self.episode_returns.append(float(self._running_returns[idx]))
                    self.episode_lengths.append(int(self._running_lengths[idx]))
                    self._running_returns[idx] = 0.0
                    self._running_lengths[idx] = 0
        processed_timesteps = self.num_timesteps - self.initial_timesteps
        if processed_timesteps % self.log_interval < self.training_env.num_envs:
            elapsed = time.perf_counter() - self.started
            metrics = {
                "steps_per_sec": processed_timesteps / max(1e-9, elapsed),
                "episodes": len(self.episode_returns),
                "mean_episode_reward_20": float(np.mean(self.episode_returns[-20:]))
                if self.episode_returns
                else None,
                "mean_episode_length_20": float(np.mean(self.episode_lengths[-20:]))
                if self.episode_lengths
                else None,
            }
            if self.tracker:
                self.tracker.log_metrics(self.num_timesteps, metrics, phase="rl_train")
            if self.telemetry:
                self.telemetry.update_rl_status(
                    str(self.checkpoint_path), self.num_timesteps, metrics["steps_per_sec"]
                )
                actions = np.asarray(self.locals.get("actions", []))
                infos = self.locals.get("infos", [])
                action_payload = (
                    {"vector": actions[0].astype(int).tolist()} if actions.size else {}
                )
                raw_info = infos[0] if infos and isinstance(infos[0], dict) else {}
                info_payload = {
                    key: raw_info[key]
                    for key in ("step", "targets_hit", "kills", "accuracy", "health", "ammo")
                    if key in raw_info
                }
                self.telemetry.update_runtime(
                    fps=0.0,
                    steps_per_sec=metrics["steps_per_sec"],
                    episode_reward=self.episode_returns[-1] if self.episode_returns else 0.0,
                    action=action_payload,
                    info=info_payload,
                )
        return True


def _load_bc_network(checkpoint_path: Union[str, Path], device: torch.device) -> BCVisionNetwork:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    network = BCVisionNetwork(
        in_channels=int(config.get("in_channels", 3)),
        num_bins_x=int(config.get("num_bins_x", 21)),
        num_bins_y=int(config.get("num_bins_y", 21)),
        latent_dim=int(config.get("latent_dim", 256)),
        use_temporal_gru=bool(config.get("use_gru", False)),
        gru_hidden_dim=int(config.get("gru_hidden_dim", config.get("latent_dim", 256))),
        channel_scales=tuple(config.get("channel_scales", (16, 32, 32))),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    return network.to(device)


def warm_start_ppo_from_bc(model: PPO, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Copy the BC encoder and all compatible action heads into PPO.

    PPO still learns its own value function.  Temporal GRU state is not copied
    because SB3's standard PPO policy is feed-forward; the visual encoder and
    complete action distribution are transferred.
    """
    device = model.device
    bc_model = _load_bc_network(checkpoint_path, device)
    extractor = model.policy.features_extractor
    if not isinstance(extractor, BCFeatureExtractor):
        raise TypeError("PPO policy does not use BCFeatureExtractor")
    extractor.encoder.load_state_dict(bc_model.encoder.state_dict())
    extractor.projection.load_state_dict(bc_model.fc_proj.state_dict())

    heads = [
        bc_model.head_move_x,
        bc_model.head_move_y,
        bc_model.head_jump,
        bc_model.head_crouch,
        bc_model.head_sprint,
        bc_model.head_reload,
        bc_model.head_fire,
        bc_model.head_ads,
        bc_model.head_mouse_dx,
        bc_model.head_mouse_dy,
    ]
    action_net = model.policy.action_net
    expected = sum(head.out_features for head in heads)
    copied_heads = False
    if action_net.in_features == heads[0].in_features and action_net.out_features == expected:
        with torch.no_grad():
            action_net.weight.copy_(torch.cat([head.weight for head in heads], dim=0))
            action_net.bias.copy_(torch.cat([head.bias for head in heads], dim=0))
        copied_heads = True
    return {
        "source": str(checkpoint_path),
        "encoder_copied": True,
        "action_heads_copied": copied_heads,
        "temporal_gru_copied": False,
    }


def train_ppo_sandbox(
    timesteps: int = 10000,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    env_path: Optional[str] = None,
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "auto",
    bc_checkpoint: Optional[Union[str, Path]] = None,
    resume_path: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    checkpoint_freq: int = 5000,
    max_episode_steps: int = 500,
    experiment_root: Optional[Union[str, Path]] = None,
    state_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if timesteps < 1 or n_envs < 1:
        raise ValueError("timesteps and n_envs must be positive")
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    save_file = checkpoint_path / "ppo_sandbox.zip"
    params = {
        "timesteps": timesteps,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "backend": backend,
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
        "resume_path": str(resume_path) if resume_path else None,
    }
    tracker = ExperimentTracker("ppo", params, experiment_root) if experiment_root else None
    telemetry = SystemTelemetry(state_file) if state_file else None
    if telemetry:
        telemetry.update_stage("RL_TRAINING")

    def make_env(index: int):
        def factory():
            return make_sandbox_env(
                env_path=env_path,
                backend=backend,
                width=84,
                height=84,
                seed=seed + index * 1009,
                max_steps=max_episode_steps,
            )

        return factory

    env_factories = [make_env(index) for index in range(n_envs)]
    # DummyVecEnv is useful for one environment and deterministic debugging,
    # but it executes multiple environments serially. Use subprocess workers
    # for real throughput when the caller requests parallel environments.
    vector_env = (
        DummyVecEnv(env_factories)
        if n_envs == 1
        else SubprocVecEnv(env_factories, start_method="spawn")
    )
    warm_start: Optional[Dict[str, Any]] = None
    started = time.perf_counter()
    try:
        if resume_path:
            if not Path(resume_path).is_file():
                raise FileNotFoundError(f"PPO checkpoint to resume not found: {resume_path}")
            model = PPO.load(resume_path, env=vector_env, device=device)
        else:
            latent_dim = 256
            channel_scales = (16, 32, 32)
            if bc_checkpoint:
                checkpoint = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
                config = checkpoint.get("config", {})
                latent_dim = int(config.get("latent_dim", latent_dim))
                channel_scales = tuple(config.get("channel_scales", channel_scales))
            policy_kwargs = {
                "features_extractor_class": BCFeatureExtractor,
                "features_extractor_kwargs": {
                    "latent_dim": latent_dim,
                    "channel_scales": channel_scales,
                },
                # No actor MLP preserves direct compatibility with BC heads.
                "net_arch": {"pi": [], "vf": [128]},
                "normalize_images": True,
            }
            model = PPO(
                "CnnPolicy",
                vector_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                verbose=0,
                seed=seed,
                device=device,
            )
            if bc_checkpoint:
                if not Path(bc_checkpoint).is_file():
                    raise FileNotFoundError(f"BC warm-start checkpoint not found: {bc_checkpoint}")
                warm_start = warm_start_ppo_from_bc(model, bc_checkpoint)

        starting_timesteps = int(model.num_timesteps)
        metrics_callback = TrainingMetricsCallback(
            tracker, telemetry, save_file, log_interval=max(64, n_steps * n_envs)
        )
        callbacks: List[BaseCallback] = [metrics_callback]
        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1, checkpoint_freq // n_envs),
                    save_path=str(checkpoint_path / "ppo_intermediate"),
                    name_prefix="ppo",
                )
            )
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=resume_path is None,
            progress_bar=False,
        )
        model.save(str(save_file))
        elapsed = time.perf_counter() - started
        actual_this_run = int(model.num_timesteps) - starting_timesteps
        training_summary = {
            "timesteps": timesteps,
            "actual_timesteps_this_run": actual_this_run,
            "actual_model_timesteps": int(model.num_timesteps),
            "elapsed_sec": round(elapsed, 3),
            "steps_per_sec": round(actual_this_run / max(1e-9, elapsed), 2),
            "checkpoint_path": str(save_file),
            "warm_start": warm_start,
            "episodes_completed": len(metrics_callback.episode_returns),
            "mean_training_reward": round(
                float(np.mean(metrics_callback.episode_returns)), 5
            )
            if metrics_callback.episode_returns
            else None,
        }
        (checkpoint_path / "ppo_training.json").write_text(
            json.dumps(training_summary, indent=2), encoding="utf-8"
        )
        if tracker:
            tracker.add_artifact(save_file, "final_checkpoint")
            tracker.add_artifact(checkpoint_path / "ppo_training.json", "training_summary")
            tracker.finish(training_summary)
            training_summary["run_id"] = tracker.run_id
        if telemetry:
            telemetry.update_rl_status(
                str(save_file), int(model.num_timesteps), training_summary["steps_per_sec"]
            )
            telemetry.update_stage("IDLE")
        return training_summary
    except Exception as exc:
        if tracker:
            tracker.fail(exc)
        if telemetry:
            telemetry.update_stage("FAILED")
        raise
    finally:
        vector_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO in the SandboxAI arena")
    parser.add_argument("--timesteps", "-t", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", "-c", default="checkpoints")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env_path", default=None)
    parser.add_argument("--backend", choices=["auto", "python", "godot"], default="auto")
    parser.add_argument("--bc_checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=5000)
    parser.add_argument("--experiment_root", default="logs/experiments")
    parser.add_argument("--state_file", default="logs/system_state.json")
    args = parser.parse_args()
    result = train_ppo_sandbox(
        timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        env_path=args.env_path,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        bc_checkpoint=args.bc_checkpoint,
        resume_path=args.resume,
        backend=args.backend,
        checkpoint_freq=args.checkpoint_freq,
        experiment_root=args.experiment_root,
        state_file=args.state_file,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
