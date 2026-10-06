"""Behavioral cloning utilities and a terminal-only demo recorder."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import sys
from typing import Any

import numpy as np

from env.shooter_env import (ACTION_NVECS, ACTION_SIZE, OBSERVATION_SIZE,
                             OBSERVATION_VERSION, ShooterEnv)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMOS = PROJECT_ROOT / "data" / "demos.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "behavior_clone.pt"


def _ordered_columns(fieldnames: list[str], prefix: str) -> list[str]:
    columns = [name for name in fieldnames if name.startswith(prefix)]
    return sorted(columns, key=lambda name: int(name[len(prefix):]) if name[len(prefix):].isdigit() else name)


def demo_meta_path(path: str | Path) -> Path:
    """Sidecar next to a demo CSV: ``data/demos.csv`` -> ``data/demos_meta.json``."""
    path = Path(path)
    return path.with_name(f"{path.stem}_meta.json")


def write_demo_meta(path: str | Path, vision_mode: str | None = None,
                    samples: int | None = None) -> Path:
    """Stamp a demo file with the observation layout it was recorded with."""
    meta_path = demo_meta_path(path)
    payload: dict[str, Any] = {"observation_version": OBSERVATION_VERSION}
    if vision_mode:
        payload["vision_mode"] = vision_mode
    if samples is not None:
        payload["samples"] = int(samples)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return meta_path


def archive_outdated_demos(path: str | Path) -> Path | None:
    """Move a demo CSV with an outdated layout aside before appending new rows.

    Appending would mix two column layouts in one file and then overwrite the
    sidecar with the new version - a combination the loader cannot detect any
    more. The old file is kept (renamed with its observation version), so nothing
    is lost.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    meta_path = demo_meta_path(path)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = int(meta.get("observation_version", OBSERVATION_VERSION))
    if version == OBSERVATION_VERSION:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}_v{version}_{stamp}{path.suffix}")
    path.replace(archived)
    meta_path.replace(demo_meta_path(archived))
    return archived


def load_demo_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load indexed state/action columns or JSON ``state,action`` pairs.

    A demo file recorded with an older observation layout is rejected: the
    columns would still line up, but state 5 would mean something completely
    different, and behavioural cloning would learn nonsense.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")
    meta_path = demo_meta_path(path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("observation_version", OBSERVATION_VERSION))
        if version != OBSERVATION_VERSION:
            raise ValueError(
                f"{path.name} was recorded with observation version {version}, this build "
                f"uses {OBSERVATION_VERSION}. Record new demonstrations "
                "(`python -m training.imitation record`) or delete the file."
            )
    states: list[list[float]] = []
    actions: list[list[int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Demo file has no CSV header: {path}")
        state_columns = _ordered_columns(reader.fieldnames, "state_")
        action_columns = _ordered_columns(reader.fieldnames, "action_")
        for row_number, row in enumerate(reader, start=2):
            if state_columns and action_columns:
                state = [float(row[column]) for column in state_columns]
                action = [int(float(row[column])) for column in action_columns]
            elif row.get("state") and row.get("action"):
                state = [float(value) for value in json.loads(row["state"])]
                action = [int(value) for value in json.loads(row["action"])]
            else:
                raise ValueError(
                    f"Row {row_number} needs state_0/action_0 columns or JSON state/action columns."
                )
            if len(state) != OBSERVATION_SIZE:
                raise ValueError(
                    f"Row {row_number} has {len(state)} state values; expected {OBSERVATION_SIZE}."
                )
            if len(action) != ACTION_SIZE:
                raise ValueError(
                    f"Row {row_number} has {len(action)} action values; expected {ACTION_SIZE}."
                )
            if any(value < 0 or value >= ACTION_NVECS[index] for index, value in enumerate(action)):
                raise ValueError(f"Row {row_number} contains an action category outside the action space.")
            states.append(state)
            actions.append(action)
    if not states:
        raise ValueError(f"No demonstrations found in {path}; record some with `python -m training.imitation record`. ")
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def train_behavior_cloning(
    dataset_path: str | Path = DEFAULT_DEMOS,
    output_path: str | Path = DEFAULT_OUTPUT,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    """Train a small categorical MLP and save actor weights for PPO warm-start."""
    import torch
    from torch import nn
    import torch.nn.functional as functional

    states, actions = load_demo_data(dataset_path)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class BehaviorCloner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(OBSERVATION_SIZE, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, sum(ACTION_NVECS)),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values)

    model = BehaviorCloner()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long)
    rng = torch.Generator().manual_seed(seed)
    batch_size = max(1, min(int(batch_size), len(states)))
    epochs = max(1, int(epochs))
    final_loss = 0.0
    final_accuracy = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(states), generator=rng)
        epoch_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(states), batch_size):
            indices = order[start:start + batch_size]
            logits = model(x[indices])
            offset = 0
            losses = []
            predictions = []
            for action_index, categories in enumerate(ACTION_NVECS):
                block = logits[:, offset:offset + categories]
                target = y[indices, action_index]
                losses.append(functional.cross_entropy(block, target))
                predictions.append(block.argmax(dim=1))
                offset += categories
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
            prediction_matrix = torch.stack(predictions, dim=1)
            correct += int((prediction_matrix == y[indices]).sum())
            seen += len(indices) * ACTION_SIZE
        final_loss = epoch_loss / max(1, len(states))
        final_accuracy = correct / max(1, seen)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "obs_size": OBSERVATION_SIZE,
        "action_nvec": tuple(ACTION_NVECS),
        "hidden_sizes": (128, 128),
        "epochs": epochs,
        "samples": len(states),
        "loss": final_loss,
        "accuracy": final_accuracy,
    }, output_path)
    return {
        "samples": len(states),
        "epochs": epochs,
        "loss": final_loss,
        "accuracy": final_accuracy,
        "output": str(output_path),
    }


def load_behavior_clone_into_policy(policy_or_model: Any, checkpoint_path: str | Path) -> None:
    """Copy compatible cloned actor weights into a Stable-Baselines3 PPO model."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("obs_size", -1)) != OBSERVATION_SIZE:
        raise ValueError("Behavior-cloning checkpoint observation size does not match this environment.")
    if tuple(checkpoint.get("action_nvec", ())) != tuple(ACTION_NVECS):
        raise ValueError("Behavior-cloning checkpoint action space does not match this environment.")
    model = policy_or_model if hasattr(policy_or_model, "policy") else None
    policy = model.policy if model is not None else policy_or_model
    state = checkpoint["state_dict"]
    actor_layers = policy.mlp_extractor.policy_net
    destination_layers = ((actor_layers[0], "network.0"),
                          (actor_layers[2], "network.2"),
                          (policy.action_net, "network.4"))
    with torch.no_grad():
        for destination, source_prefix in destination_layers:
            source_weight = state[f"{source_prefix}.weight"]
            source_bias = state[f"{source_prefix}.bias"]
            if tuple(destination.weight.shape) != tuple(source_weight.shape):
                raise ValueError(f"Actor layer mismatch at {source_prefix}; PPO uses a different architecture.")
            destination.weight.copy_(source_weight)
            destination.bias.copy_(source_bias)


def _read_terminal_keys() -> set[str]:
    """Read queued single-character keys without requiring a graphical window."""
    ready, _, _ = select.select([sys.stdin], [], [], 0.03)
    if not ready:
        return set()
    keys: set[str] = set()
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            break
        key = sys.stdin.read(1)
        if key:
            keys.add(key.lower())
    return keys


def record_demos(
    output_path: str | Path = DEFAULT_DEMOS,
    map_name: str = "Dust",
    weapon_name: str = "Pistol",
) -> int:
    """Record human policy examples through a raw terminal keyboard session."""
    if not sys.stdin.isatty():
        raise RuntimeError("Demo recording needs an interactive terminal. Run this command from a WSL terminal.")
    try:
        import termios
        import tty
    except ImportError as exc:
        raise RuntimeError("The raw-key recorder is intended for Linux/WSL terminals.") from exc

    import numpy as np
    import termios
    import tty

    env = ShooterEnv(map_name=map_name, weapon_name=weapon_name, curriculum=False,
                     opponent_mode="shooter", frame_skip=4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_fields = [f"state_{index}" for index in range(OBSERVATION_SIZE)]
    action_fields = [f"action_{index}" for index in range(ACTION_SIZE)]
    original_settings = termios.tcgetattr(sys.stdin.fileno())
    observation, _ = env.reset()
    rows = 0
    controls = (
        "W/S move, A/D strafe, J/L turn, I/K look, Space shoot, X sprint, "
        "C crouch, P prone, B/N lean, R reload, Q quit"
    )
    print("NEURAL ARENA demo recorder (headless)")
    print(controls)
    print("State/action pairs are saved as CSV. Press Q to finish.")
    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=state_fields + action_fields)
            writer.writeheader()
            tty.setraw(sys.stdin.fileno())
            while True:
                keys = _read_terminal_keys()
                if "q" in keys or "\x03" in keys:
                    break
                action = np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
                if "w" in keys:
                    action[0] = 2
                if "s" in keys:
                    action[0] = 0
                if "a" in keys:
                    action[1] = 0
                if "d" in keys:
                    action[1] = 2
                if "j" in keys:
                    action[2] = 0
                if "l" in keys:
                    action[2] = 2
                if "k" in keys:
                    action[3] = 0
                if "i" in keys:
                    action[3] = 2
                if " " in keys:
                    action[4] = 1
                if "x" in keys:
                    action[5] = 1
                if "c" in keys:
                    action[6] = 1
                if "p" in keys:
                    action[6] = 2
                if "b" in keys:
                    action[7] = 0
                if "n" in keys:
                    action[7] = 2
                if " " not in keys and "r" in keys:
                    action[9] = 1
                row = {**{field: float(value) for field, value in zip(state_fields, observation)},
                       **{field: int(value) for field, value in zip(action_fields, action)}}
                writer.writerow(row)
                handle.flush()
                observation, _, terminated, truncated, info = env.step(action)
                rows += 1
                if rows % 50 == 0:
                    player = info.get("player", {})
                    sys.stdout.write(
                        f"\rSamples: {rows:6d} | HP: {player.get('hp', 0):5.1f} | "
                        f"Ammo: {player.get('ammo', 0):3d} | Map: {map_name}   "
                    )
                    sys.stdout.flush()
                if terminated or truncated:
                    observation, _ = env.reset()
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_settings)
        env.close()
        print(f"\nSaved {rows} demonstrations to {output_path}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="NEURAL ARENA behavioral cloning utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record", help="record keyboard demonstrations in a terminal")
    record_parser.add_argument("--output", type=Path, default=DEFAULT_DEMOS)
    record_parser.add_argument("--map", dest="map_name", default="Dust")
    record_parser.add_argument("--weapon", dest="weapon_name", default="Pistol")
    train_parser = subparsers.add_parser("train", help="train the behavior-cloning MLP")
    train_parser.add_argument("--input", type=Path, default=DEFAULT_DEMOS)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train_parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.command == "record":
        record_demos(args.output, args.map_name, args.weapon_name)
    else:
        result = train_behavior_cloning(args.input, args.output, epochs=args.epochs)
        print(
            f"Trained on {result['samples']} samples for {result['epochs']} epochs; "
            f"loss={result['loss']:.4f}, per-action accuracy={result['accuracy']:.1%}. "
            f"Saved {result['output']}"
        )


if __name__ == "__main__":
    main()
