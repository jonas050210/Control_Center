"""Behavioral-cloning training with temporal vision, checkpoints, and metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bc.dataset import GameplayDataset
from bc.models import BCVisionNetwork
from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry

DISCRETE_HEADS = (
    "move_x",
    "move_y",
    "jump",
    "crouch",
    "sprint",
    "reload",
    "fire",
    "ads",
    "mouse_dx_bin",
    "mouse_dy_bin",
)


def _atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
    """Write checkpoints atomically so interruption cannot destroy the last model."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        prediction = logits.topk(min(k, logits.size(-1)), dim=-1).indices
        correct = prediction.eq(targets.view(-1, 1)).any(dim=-1)
        return float(correct.float().mean().item() * 100.0)


def _loss_and_batch_metrics(
    outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Movement/look are primary controls; sparse buttons receive a lower total
    # weight so idle-heavy demonstrations cannot swamp mouse learning.
    weights = {
        "move_x": 1.0,
        "move_y": 1.0,
        "mouse_dx_bin": 1.0,
        "mouse_dy_bin": 1.0,
        "fire": 0.65,
        "ads": 0.55,
        "jump": 0.35,
        "crouch": 0.35,
        "sprint": 0.45,
        "reload": 0.35,
    }
    loss = torch.zeros((), device=outputs["move_x"].device)
    metrics: Dict[str, float] = {}
    for head in DISCRETE_HEADS:
        head_loss = nn.functional.cross_entropy(
            outputs[head], targets[head], label_smoothing=0.01
        )
        loss = loss + weights[head] * head_loss
        metrics[f"{head}_acc"] = compute_top_k_accuracy(outputs[head], targets[head])
    # Continuous deltas are an auxiliary signal. Scaling prevents a rare large
    # flick from dominating all categorical heads.
    target_cont = torch.clamp(targets["mouse_continuous"] / 50.0, -3.0, 3.0)
    pred_cont = outputs["mouse_continuous"] / 50.0
    continuous_loss = nn.functional.smooth_l1_loss(pred_cont, target_cont)
    loss = loss + 0.1 * continuous_loss
    metrics["mouse_dx_top3"] = compute_top_k_accuracy(
        outputs["mouse_dx_bin"], targets["mouse_dx_bin"], 3
    )
    metrics["mouse_dy_top3"] = compute_top_k_accuracy(
        outputs["mouse_dy_bin"], targets["mouse_dy_bin"], 3
    )
    metrics["continuous_loss"] = float(continuous_loss.detach().item())
    return loss, metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for observations, targets in loader:
            observations = observations.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            batch_size = observations.shape[0]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs, _ = model(observations)
                    loss, batch_metrics = _loss_and_batch_metrics(outputs, targets)
            else:
                outputs, _ = model(observations)
                loss, batch_metrics = _loss_and_batch_metrics(outputs, targets)
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item()) * batch_size
            for key, value in batch_metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            examples += batch_size
    if examples == 0:
        raise ValueError("Cannot train/evaluate an empty DataLoader")
    return {key: value / examples for key, value in totals.items()}


def _public_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    result = {f"{prefix}_{key}": value for key, value in metrics.items()}
    # Stable aliases retained for existing reports.
    result[f"{prefix}_mouse_dx_top1"] = result[f"{prefix}_mouse_dx_bin_acc"]
    result[f"{prefix}_mouse_dy_top1"] = result[f"{prefix}_mouse_dy_bin_acc"]
    return result


def train_bc_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    return _public_metrics(
        _run_epoch(model, loader, device, optimizer, scaler, use_amp), "train"
    )


def evaluate_bc(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    return _public_metrics(_run_epoch(model, loader, device), "val")


def train_bc(
    data_dir: Union[str, Path],
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    target_h: int = 120,
    target_w: int = 160,
    seq_len: int = 1,
    use_gru: bool = False,
    device_str: Optional[str] = None,
    resume_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    augment: bool = True,
    latent_dim: int = 256,
    channel_scales: Tuple[int, int, int] = (16, 32, 32),
    num_workers: int = 0,
    experiment_root: Optional[Union[str, Path]] = None,
    state_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train and persist a BC model; optionally resume optimizer/model state."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    params = {
        "data_dir": str(data_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "resolution": [target_w, target_h],
        "seq_len": seq_len,
        "use_gru": use_gru,
        "seed": seed,
        "augment": augment,
        "device": str(device),
    }
    tracker = ExperimentTracker("bc", params, experiment_root) if experiment_root else None
    telemetry = SystemTelemetry(state_file) if state_file else None
    if telemetry:
        telemetry.update_stage("BC_TRAINING")

    print("=" * 60)
    print("  SandboxAI Behavioral Cloning Training")
    print("=" * 60)
    print(f"Device={device} AMP={use_amp}  Resolution={target_w}x{target_h}  Sequence={seq_len}")

    try:
        train_ds, val_ds = GameplayDataset.create_train_val_split(
            data_root=data_dir,
            val_ratio=val_ratio,
            seed=seed,
            target_size=(target_h, target_w),
            seq_len=seq_len,
            augment_train=augment,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_amp,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(batch_size, len(val_ds)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_amp,
        )
        print(
            f"Train={len(train_ds)} samples/{len(train_ds.sessions_data)} sessions, "
            f"Val={len(val_ds)} samples/{len(val_ds.sessions_data)} sessions"
        )

        resume_checkpoint: Optional[Dict[str, Any]] = None
        if resume_path:
            resume_file = Path(resume_path)
            if not resume_file.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file to resume from not found: {resume_path}"
                )
            resume_checkpoint = torch.load(resume_file, map_location=device, weights_only=False)
            resume_config = resume_checkpoint.get("config", {})
            expected_resume = {
                "target_h": target_h,
                "target_w": target_w,
                "seq_len": seq_len,
                "num_bins_x": int(train_ds.num_bins_x),
                "num_bins_y": int(train_ds.num_bins_y),
            }
            mismatches = {
                key: (expected_resume[key], resume_config[key])
                for key in expected_resume
                if key in resume_config and expected_resume[key] != resume_config[key]
            }
            if mismatches:
                raise ValueError(
                    "Resume checkpoint is incompatible with the requested dataset/model settings: "
                    + ", ".join(f"{key}={current!r} (checkpoint={saved!r})" for key, (current, saved) in mismatches.items())
                )
            latent_dim = int(resume_config.get("latent_dim", latent_dim))
            channel_scales = tuple(resume_config.get("channel_scales", channel_scales))
            use_gru = bool(resume_config.get("use_gru", use_gru))

        model = BCVisionNetwork(
            in_channels=3,
            num_bins_x=int(train_ds.num_bins_x),
            num_bins_y=int(train_ds.num_bins_y),
            latent_dim=latent_dim,
            use_temporal_gru=use_gru,
            gru_hidden_dim=latent_dim,
            channel_scales=channel_scales,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        start_epoch, best_val_loss = 1, float("inf")
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))

        metadata = train_ds.sessions_data[0]["metadata"]
        config = {
            "checkpoint_schema_version": 2,
            "in_channels": 3,
            "num_bins_x": int(train_ds.num_bins_x),
            "num_bins_y": int(train_ds.num_bins_y),
            "latent_dim": latent_dim,
            "gru_hidden_dim": latent_dim,
            "channel_scales": list(channel_scales),
            "target_h": target_h,
            "target_w": target_w,
            "seq_len": seq_len,
            "use_gru": use_gru,
            "mouse_binning_strategy": metadata.mouse_config.binning_strategy,
            "mouse_bin_edges_x": metadata.mouse_config.bin_edges_x,
            "mouse_bin_edges_y": metadata.mouse_config.bin_edges_y,
            "action_heads": list(DISCRETE_HEADS),
            "seed": seed,
        }
        history: List[Dict[str, Any]] = []
        history_file = checkpoint_path / "bc_history.json"
        if resume_checkpoint is not None and history_file.is_file():
            try:
                loaded_history = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(loaded_history, list):
                    history = [item for item in loaded_history if isinstance(item, dict)]
            except (OSError, json.JSONDecodeError):
                # A checkpoint remains usable even if its optional history
                # artifact was interrupted during a previous write.
                history = []
        for epoch in range(start_epoch, start_epoch + epochs):
            started = time.perf_counter()
            train_metrics = train_bc_epoch(
                model, train_loader, optimizer, device, scaler, use_amp
            )
            val_metrics = evaluate_bc(model, val_loader, device)
            scheduler.step()
            val_loss = val_metrics["val_loss"]
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            summary = {
                "epoch": epoch,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **train_metrics,
                **val_metrics,
                "best_val_loss": best_val_loss,
            }
            history.append(summary)
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            _atomic_torch_save(payload, checkpoint_path / "bc_latest.pt")
            if is_best:
                _atomic_torch_save(payload, checkpoint_path / "bc_best.pt")
            history_tmp = checkpoint_path / "bc_history.json.tmp"
            history_tmp.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(history_tmp, checkpoint_path / "bc_history.json")
            if tracker:
                tracker.log_metrics(epoch, summary, phase="train_val")
            if telemetry:
                telemetry.update_bc_status(
                    str(checkpoint_path / "bc_best.pt"), best_val_loss, summary
                )
            print(
                f"Epoch {epoch:03d} loss={train_metrics['train_loss']:.4f}/"
                f"{val_loss:.4f} fire={val_metrics['val_fire_acc']:.1f}% "
                f"look@3={val_metrics['val_mouse_dx_top3']:.1f}% "
                f"{('*BEST*' if is_best else '')}"
            )

        validation_ranges: List[Dict[str, Any]] = []
        for session_index, session in enumerate(val_ds.sessions_data):
            windows = [
                indices
                for indexed_session, indices in val_ds.samples_index
                if indexed_session == session_index
            ]
            if windows:
                validation_ranges.append(
                    {
                        "path": str(session["dir"]),
                        "context_start_step": min(window[0] for window in windows),
                        "start_step": min(window[-1] for window in windows),
                        "end_step": max(window[-1] for window in windows) + 1,
                    }
                )
        result = {
            "best_val_loss": best_val_loss,
            "epochs_completed": len(history),
            "history": history,
            "checkpoint_best": str(checkpoint_path / "bc_best.pt"),
            "checkpoint_latest": str(checkpoint_path / "bc_latest.pt"),
            "history_path": str(checkpoint_path / "bc_history.json"),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_sessions": [str(item["dir"]) for item in train_ds.sessions_data],
            "val_sessions": [str(item["dir"]) for item in val_ds.sessions_data],
            "validation_ranges": validation_ranges,
        }
        if tracker:
            tracker.add_artifact(checkpoint_path / "bc_best.pt", "best_checkpoint")
            tracker.add_artifact(checkpoint_path / "bc_history.json", "training_history")
            tracker.finish(result)
            result["run_id"] = tracker.run_id
        if telemetry:
            telemetry.update_stage("IDLE")
        return result
    except Exception as exc:
        if tracker:
            tracker.fail(exc)
        if telemetry:
            telemetry.update_stage("FAILED")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SandboxAI behavioral-cloning policy")
    parser.add_argument("--data_dir", "-d", default="datasets")
    parser.add_argument("--checkpoint_dir", "-c", default="checkpoints")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--gru", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--experiment_root", default="logs/experiments")
    parser.add_argument("--state_file", default="logs/system_state.json")
    args = parser.parse_args()
    train_bc(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        target_h=args.height,
        target_w=args.width,
        seq_len=args.seq_len,
        use_gru=args.gru,
        device_str=args.device,
        resume_path=args.resume,
        seed=args.seed,
        augment=args.augment,
        num_workers=args.num_workers,
        experiment_root=args.experiment_root,
        state_file=args.state_file,
    )


if __name__ == "__main__":
    main()
