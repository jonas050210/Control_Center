"""Behavioral Cloning Training Pipeline for SandboxAI (M1/M2).

Trains multi-task vision models from human gameplay demonstration datasets
with session-level splitting, AMP, checkpointing, and accuracy metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bc.dataset import GameplayDataset
from bc.models import BCVisionNetwork

logger = logging.getLogger("SandboxAI.BC.Train")


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Computes top-k accuracy percentage."""
    with torch.no_grad():
        _, pred_k = logits.topk(min(k, logits.size(-1)), dim=-1, largest=True, sorted=True)
        correct = pred_k.eq(targets.view(-1, 1).expand_as(pred_k))
        return float(correct.any(dim=-1).float().mean().item() * 100.0)


def train_bc_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    action_accuracies: Dict[str, List[float]] = {
        "move_x": [],
        "move_y": [],
        "fire": [],
        "ads": [],
        "jump": [],
        "mouse_dx_top1": [],
        "mouse_dx_top3": [],
        "mouse_dy_top1": [],
        "mouse_dy_top3": [],
    }

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    for obs, targets in loader:
        obs = obs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits_dict, _ = model(obs)
                loss_move_x = ce_loss(logits_dict["move_x"], targets["move_x"])
                loss_move_y = ce_loss(logits_dict["move_y"], targets["move_y"])
                loss_fire = ce_loss(logits_dict["fire"], targets["fire"])
                loss_ads = ce_loss(logits_dict["ads"], targets["ads"])
                loss_jump = ce_loss(logits_dict["jump"], targets["jump"])
                loss_crouch = ce_loss(logits_dict["crouch"], targets["crouch"])
                loss_sprint = ce_loss(logits_dict["sprint"], targets["sprint"])
                loss_reload = ce_loss(logits_dict["reload"], targets["reload"])
                loss_dx = ce_loss(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"])
                loss_dy = ce_loss(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"])
                loss_cont = mse_loss(logits_dict["mouse_continuous"], targets["mouse_continuous"])

                loss = (
                    loss_move_x
                    + loss_move_y
                    + 0.5 * (loss_fire + loss_ads + loss_jump + loss_crouch + loss_sprint + loss_reload)
                    + loss_dx
                    + loss_dy
                    + 0.1 * loss_cont
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_dict, _ = model(obs)
            loss_move_x = ce_loss(logits_dict["move_x"], targets["move_x"])
            loss_move_y = ce_loss(logits_dict["move_y"], targets["move_y"])
            loss_fire = ce_loss(logits_dict["fire"], targets["fire"])
            loss_ads = ce_loss(logits_dict["ads"], targets["ads"])
            loss_jump = ce_loss(logits_dict["jump"], targets["jump"])
            loss_crouch = ce_loss(logits_dict["crouch"], targets["crouch"])
            loss_sprint = ce_loss(logits_dict["sprint"], targets["sprint"])
            loss_reload = ce_loss(logits_dict["reload"], targets["reload"])
            loss_dx = ce_loss(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"])
            loss_dy = ce_loss(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"])
            loss_cont = mse_loss(logits_dict["mouse_continuous"], targets["mouse_continuous"])

            loss = (
                loss_move_x
                + loss_move_y
                + 0.5 * (loss_fire + loss_ads + loss_jump + loss_crouch + loss_sprint + loss_reload)
                + loss_dx
                + loss_dy
                + 0.1 * loss_cont
            )
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Compute metric batch accuracies
        with torch.no_grad():
            action_accuracies["move_x"].append(
                compute_top_k_accuracy(logits_dict["move_x"], targets["move_x"], k=1)
            )
            action_accuracies["move_y"].append(
                compute_top_k_accuracy(logits_dict["move_y"], targets["move_y"], k=1)
            )
            action_accuracies["fire"].append(
                compute_top_k_accuracy(logits_dict["fire"], targets["fire"], k=1)
            )
            action_accuracies["ads"].append(
                compute_top_k_accuracy(logits_dict["ads"], targets["ads"], k=1)
            )
            action_accuracies["jump"].append(
                compute_top_k_accuracy(logits_dict["jump"], targets["jump"], k=1)
            )
            action_accuracies["mouse_dx_top1"].append(
                compute_top_k_accuracy(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"], k=1)
            )
            action_accuracies["mouse_dx_top3"].append(
                compute_top_k_accuracy(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"], k=3)
            )
            action_accuracies["mouse_dy_top1"].append(
                compute_top_k_accuracy(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"], k=1)
            )
            action_accuracies["mouse_dy_top3"].append(
                compute_top_k_accuracy(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"], k=3)
            )

    num_batches = max(1, len(loader))
    metrics = {
        "train_loss": total_loss / num_batches,
        "train_move_x_acc": float(sum(action_accuracies["move_x"]) / len(action_accuracies["move_x"])),
        "train_move_y_acc": float(sum(action_accuracies["move_y"]) / len(action_accuracies["move_y"])),
        "train_fire_acc": float(sum(action_accuracies["fire"]) / len(action_accuracies["fire"])),
        "train_mouse_dx_top1": float(sum(action_accuracies["mouse_dx_top1"]) / len(action_accuracies["mouse_dx_top1"])),
        "train_mouse_dx_top3": float(sum(action_accuracies["mouse_dx_top3"]) / len(action_accuracies["mouse_dx_top3"])),
        "train_mouse_dy_top1": float(sum(action_accuracies["mouse_dy_top1"]) / len(action_accuracies["mouse_dy_top1"])),
        "train_mouse_dy_top3": float(sum(action_accuracies["mouse_dy_top3"]) / len(action_accuracies["mouse_dy_top3"])),
    }
    return metrics


def evaluate_bc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the BC model on validation set."""
    model.eval()
    total_loss = 0.0
    action_accuracies: Dict[str, List[float]] = {
        "move_x": [],
        "move_y": [],
        "fire": [],
        "ads": [],
        "jump": [],
        "mouse_dx_top1": [],
        "mouse_dx_top3": [],
        "mouse_dy_top1": [],
        "mouse_dy_top3": [],
    }

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for obs, targets in loader:
            obs = obs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            logits_dict, _ = model(obs)

            loss_move_x = ce_loss(logits_dict["move_x"], targets["move_x"])
            loss_move_y = ce_loss(logits_dict["move_y"], targets["move_y"])
            loss_fire = ce_loss(logits_dict["fire"], targets["fire"])
            loss_ads = ce_loss(logits_dict["ads"], targets["ads"])
            loss_jump = ce_loss(logits_dict["jump"], targets["jump"])
            loss_crouch = ce_loss(logits_dict["crouch"], targets["crouch"])
            loss_sprint = ce_loss(logits_dict["sprint"], targets["sprint"])
            loss_reload = ce_loss(logits_dict["reload"], targets["reload"])
            loss_dx = ce_loss(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"])
            loss_dy = ce_loss(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"])
            loss_cont = mse_loss(logits_dict["mouse_continuous"], targets["mouse_continuous"])

            loss = (
                loss_move_x
                + loss_move_y
                + 0.5 * (loss_fire + loss_ads + loss_jump + loss_crouch + loss_sprint + loss_reload)
                + loss_dx
                + loss_dy
                + 0.1 * loss_cont
            )
            total_loss += loss.item()

            action_accuracies["move_x"].append(
                compute_top_k_accuracy(logits_dict["move_x"], targets["move_x"], k=1)
            )
            action_accuracies["move_y"].append(
                compute_top_k_accuracy(logits_dict["move_y"], targets["move_y"], k=1)
            )
            action_accuracies["fire"].append(
                compute_top_k_accuracy(logits_dict["fire"], targets["fire"], k=1)
            )
            action_accuracies["ads"].append(
                compute_top_k_accuracy(logits_dict["ads"], targets["ads"], k=1)
            )
            action_accuracies["jump"].append(
                compute_top_k_accuracy(logits_dict["jump"], targets["jump"], k=1)
            )
            action_accuracies["mouse_dx_top1"].append(
                compute_top_k_accuracy(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"], k=1)
            )
            action_accuracies["mouse_dx_top3"].append(
                compute_top_k_accuracy(logits_dict["mouse_dx_bin"], targets["mouse_dx_bin"], k=3)
            )
            action_accuracies["mouse_dy_top1"].append(
                compute_top_k_accuracy(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"], k=1)
            )
            action_accuracies["mouse_dy_top3"].append(
                compute_top_k_accuracy(logits_dict["mouse_dy_bin"], targets["mouse_dy_bin"], k=3)
            )

    num_batches = max(1, len(loader))
    return {
        "val_loss": total_loss / num_batches,
        "val_move_x_acc": float(sum(action_accuracies["move_x"]) / len(action_accuracies["move_x"])),
        "val_move_y_acc": float(sum(action_accuracies["move_y"]) / len(action_accuracies["move_y"])),
        "val_fire_acc": float(sum(action_accuracies["fire"]) / len(action_accuracies["fire"])),
        "val_mouse_dx_top1": float(sum(action_accuracies["mouse_dx_top1"]) / len(action_accuracies["mouse_dx_top1"])),
        "val_mouse_dx_top3": float(sum(action_accuracies["mouse_dx_top3"]) / len(action_accuracies["mouse_dx_top3"])),
        "val_mouse_dy_top1": float(sum(action_accuracies["mouse_dy_top1"]) / len(action_accuracies["mouse_dy_top1"])),
        "val_mouse_dy_top3": float(sum(action_accuracies["mouse_dy_top3"]) / len(action_accuracies["mouse_dy_top3"])),
    }


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
) -> Dict[str, Any]:
    """Full Behavioral Cloning training run."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("============================================================")
    print("  SandboxAI Behavioral Cloning (BC) Training")
    print("============================================================")
    print(f"Device        : {device} (AMP: {use_amp})")
    print(f"Data Dir      : {Path(data_dir).resolve()}")
    print(f"Resolution    : {target_w}x{target_h} (seq_len={seq_len})")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Epochs        : {epochs}")
    print(f"Checkpoint Dir: {checkpoint_path.resolve()}")
    print("============================================================")

    # 1. Datasets & DataLoaders
    train_ds, val_ds = GameplayDataset.create_train_val_split(
        data_root=data_dir,
        val_ratio=val_ratio,
        target_size=(target_h, target_w),
        seq_len=seq_len,
    )
    print(f"[INFO] Train samples: {len(train_ds)} across {len(train_ds.sessions_data)} session(s)")
    print(f"[INFO] Val samples  : {len(val_ds)} across {len(val_ds.sessions_data)} session(s)")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
        drop_last=False,
    )

    # 2. Model, Optimizer, Scheduler
    model = BCVisionNetwork(
        in_channels=3,
        num_bins_x=train_ds.num_bins_x,
        num_bins_y=train_ds.num_bins_y,
        use_temporal_gru=use_gru,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume checkpoint if specified
    if resume_path and Path(resume_path).exists():
        print(f"[INFO] Resuming training from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # 3. Training Loop
    history: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        train_metrics = train_bc_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = evaluate_bc(model=model, loader=val_loader, device=device)
        scheduler.step()
        elapsed = time.time() - t0

        val_loss = val_metrics["val_loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_summary = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            **train_metrics,
            **val_metrics,
            "best_val_loss": round(best_val_loss, 4),
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:03d}/{start_epoch + epochs - 1:03d} [{elapsed:.1f}s] "
            f"Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_loss:.4f} | "
            f"Move X Acc: {val_metrics['val_move_x_acc']:.1f}% | "
            f"Move Y Acc: {val_metrics['val_move_y_acc']:.1f}% | "
            f"Fire Acc: {val_metrics['val_fire_acc']:.1f}% | "
            f"Mouse dX Top-3: {val_metrics['val_mouse_dx_top3']:.1f}% "
            f"{'(*BEST*)' if is_best else ''}"
        )

        # Save latest checkpoint
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "config": {
                "in_channels": 3,
                "num_bins_x": train_ds.num_bins_x,
                "num_bins_y": train_ds.num_bins_y,
                "latent_dim": model.latent_dim,
                "target_h": target_h,
                "target_w": target_w,
                "seq_len": seq_len,
                "use_gru": use_gru,
            },
        }
        torch.save(ckpt_payload, checkpoint_path / "bc_latest.pt")
        if is_best:
            torch.save(ckpt_payload, checkpoint_path / "bc_best.pt")

    print("\n[INFO] Training complete!")
    print(f"[INFO] Best Val Loss: {best_val_loss:.4f}")
    print(f"[INFO] Checkpoints saved to {checkpoint_path / 'bc_best.pt'}")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_best": str(checkpoint_path / "bc_best.pt"),
        "checkpoint_latest": str(checkpoint_path / "bc_latest.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Behavioral Cloning policy for SandboxAI")
    parser.add_argument("--data_dir", "-d", type=str, default="datasets", help="Path to datasets directory")
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation session split ratio")
    parser.add_argument("--width", type=int, default=160, help="Frame width")
    parser.add_argument("--height", type=int, default=120, help="Frame height")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence context window length")
    parser.add_argument("--gru", action="store_true", help="Use temporal GRU module")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

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
    )


if __name__ == "__main__":
    main()
