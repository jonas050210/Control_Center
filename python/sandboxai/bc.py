"""Small multi-head PyTorch behavior-cloning policy."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any

from .config import BCConfig
from .dataset import ACTION_NVECS, DemonstrationDataset

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class BehaviorCloningPolicy(nn.Module):
        def __init__(self, observation_dim: int, hidden_sizes: tuple[int, int] = (128, 128)) -> None:
            super().__init__()
            self.observation_dim = int(observation_dim)
            self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
            self.action_nvec = tuple(ACTION_NVECS)
            self.backbone = nn.Sequential(
                nn.Linear(self.observation_dim, self.hidden_sizes[0]),
                nn.Tanh(),
                nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                nn.Tanh(),
            )
            self.heads = nn.ModuleList([nn.Linear(self.hidden_sizes[1], size) for size in self.action_nvec])

        def forward(self, observations):
            features = self.backbone(observations)
            return tuple(head(features) for head in self.heads)

        @torch.no_grad()
        def predict(self, observations, deterministic: bool = True):
            tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations, dtype=torch.float32)
            logits = self.forward(tensor)
            if deterministic:
                values = [torch.argmax(logit, dim=-1) for logit in logits]
            else:
                values = [torch.distributions.Categorical(logits=logit).sample() for logit in logits]
            return torch.stack(values, dim=-1)

else:

    class BehaviorCloningPolicy:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("PyTorch is required for behavior cloning; install the training dependencies")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for behavior cloning; install torch and numpy")


def create_bc_policy(observation_dim: int, config: BCConfig | None = None) -> BehaviorCloningPolicy:
    _require_torch()
    config = (config or BCConfig()).validate()
    return BehaviorCloningPolicy(observation_dim, config.hidden_sizes)


def _metrics_from_logits(logits, actions) -> tuple[Any, float, float]:
    losses = [nn.functional.cross_entropy(logit, actions[:, index]) for index, logit in enumerate(logits)]
    loss = sum(losses)
    predictions = torch.stack([torch.argmax(logit, dim=-1) for logit in logits], dim=-1)
    component_accuracy = float((predictions == actions).float().mean().item())
    exact_accuracy = float((predictions == actions).all(dim=1).float().mean().item())
    return loss, component_accuracy, exact_accuracy


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_bc_checkpoint(path: str | Path, device: str = "cpu") -> BehaviorCloningPolicy:
    _require_torch()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = BehaviorCloningPolicy(
        int(checkpoint["observation_dim"]),
        tuple(int(value) for value in checkpoint["hidden_sizes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def train_behavior_cloning(
    dataset_path: str | Path,
    config: BCConfig | None = None,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    _require_torch()
    config = (config or BCConfig()).validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(config.seed)
    except ImportError as exc:
        raise RuntimeError("numpy is required for behavior cloning") from exc

    full = DemonstrationDataset.load(dataset_path)
    train_set, validation_set = full.split(config.validation_fraction, config.seed)
    train_arrays = train_set.arrays()
    validation_arrays = validation_set.arrays()
    device = config.resolved_device()
    model = BehaviorCloningPolicy(train_arrays[0].shape[1], config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch = 0
    best_validation = float("inf")
    patience_counter = 0

    if resume_checkpoint:
        checkpoint = torch.load(Path(resume_checkpoint), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_validation = float(checkpoint.get("best_validation_loss", best_validation))

    destination = Path(output_dir or Path(config.output_root) / "bc_runs" / "latest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps({**config.__dict__, "device": device}, indent=2) + "\n", encoding="utf-8")
    metrics_path = destination / "metrics.jsonl"
    loss_csv = destination / "loss.csv"
    if start_epoch == 0:
        metrics_path.write_text("", encoding="utf-8")
    with loss_csv.open("a", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=["epoch", "train_loss", "validation_loss", "component_accuracy", "exact_accuracy"])
        if loss_csv.stat().st_size == 0:
            writer.writeheader()
        train_observations = torch.as_tensor(train_arrays[0], dtype=torch.float32)
        train_actions = torch.as_tensor(train_arrays[1], dtype=torch.long)
        validation_observations = torch.as_tensor(validation_arrays[0], dtype=torch.float32)
        validation_actions = torch.as_tensor(validation_arrays[1], dtype=torch.long)
        generator = torch.Generator().manual_seed(config.seed)
        for epoch in range(start_epoch, config.epochs):
            model.train()
            permutation = torch.randperm(train_observations.shape[0], generator=generator)
            train_loss_total = 0.0
            train_batches = 0
            for batch_indices in permutation.split(config.batch_size):
                batch_observations = train_observations[batch_indices].to(device)
                batch_actions = train_actions[batch_indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _component, _exact = _metrics_from_logits(model(batch_observations), batch_actions)
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.item())
                train_batches += 1
            model.eval()
            with torch.no_grad():
                validation_loss, component_accuracy, exact_accuracy = _metrics_from_logits(
                    model(validation_observations.to(device)), validation_actions.to(device)
                )
            train_loss = train_loss_total / max(train_batches, 1)
            validation_value = float(validation_loss.item())
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_value,
                "component_accuracy": component_accuracy,
                "exact_accuracy": exact_accuracy,
            }
            writer.writerow(row)
            csv_stream.flush()
            with metrics_path.open("a", encoding="utf-8") as metrics_stream:
                metrics_stream.write(json.dumps(row) + "\n")
            checkpoint = {
                "format": "sandboxai.bc.v1",
                "epoch": epoch + 1,
                "observation_dim": model.observation_dim,
                "hidden_sizes": list(model.hidden_sizes),
                "action_nvec": list(model.action_nvec),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": min(best_validation, validation_value),
                "metrics": row,
                "dataset": str(dataset_path),
            }
            _atomic_torch_save(checkpoint, destination / "latest.pt")
            if validation_value < best_validation:
                best_validation = validation_value
                patience_counter = 0
                _atomic_torch_save(checkpoint, destination / "best.pt")
            else:
                patience_counter += 1
                if config.early_stopping_patience > 0 and patience_counter >= config.early_stopping_patience:
                    break
            if (epoch + 1) % config.checkpoint_frequency == 0:
                _atomic_torch_save(checkpoint, destination / f"epoch_{epoch + 1:05d}.pt")

    return {
        "output_dir": str(destination),
        "latest_checkpoint": str(destination / "latest.pt"),
        "best_checkpoint": str(destination / "best.pt"),
        "epochs": epoch + 1,
        "validation_loss": best_validation,
    }


def load_bc_into_sb3_policy(policy: Any, checkpoint_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Transfer weights only when the SB3 MLP layout is exactly compatible.

    This deliberately raises instead of silently claiming a warm start when
    a future SB3 version changes parameter names or dimensions.
    """
    _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model = BehaviorCloningPolicy(int(checkpoint["observation_dim"]), tuple(checkpoint["hidden_sizes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    policy_net = getattr(policy.mlp_extractor, "policy_net", None)
    action_net = getattr(policy, "action_net", None)
    if policy_net is None or action_net is None or len(policy_net) < 3:
        raise ValueError("SB3 policy layout is not compatible with the BC checkpoint")
    source_layers = [model.backbone[0], model.backbone[2]]
    target_layers = [policy_net[0], policy_net[2]]
    for source, target in zip(source_layers, target_layers):
        if source.weight.shape != target.weight.shape or source.bias.shape != target.bias.shape:
            raise ValueError("BC and PPO hidden-layer dimensions do not match; no weights transferred")
    output_rows = sum(model.action_nvec)
    if tuple(action_net.weight.shape) != (output_rows, model.hidden_sizes[-1]):
        raise ValueError("BC and PPO action-head dimensions do not match; no weights transferred")
    with torch.no_grad():
        for source, target in zip(source_layers, target_layers):
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
        offset = 0
        for head in model.heads:
            rows = head.out_features
            action_net.weight[offset : offset + rows].copy_(head.weight)
            action_net.bias[offset : offset + rows].copy_(head.bias)
            offset += rows
    return {"transferred": True, "observation_dim": model.observation_dim, "hidden_sizes": list(model.hidden_sizes)}
