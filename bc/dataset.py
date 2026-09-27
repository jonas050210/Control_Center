"""PyTorch Dataset for SandboxAI Behavioral Cloning (M1/M2).

Loads versioned gameplay recording sessions, performs session-level train/validation
splitting, supports context sequence windows, and maps actions to training targets.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.schema import DatasetMetadata, DatasetSample

logger = logging.getLogger("SandboxAI.BC.Dataset")


class GameplayDataset(Dataset):
    """PyTorch Dataset for loading single-frame or multi-frame sequence demonstration data."""

    def __init__(
        self,
        session_dirs: Sequence[Union[str, Path]],
        target_size: Optional[Tuple[int, int]] = (120, 160),  # (H, W)
        seq_len: int = 1,
        stride: int = 1,
        num_bins_x: int = 21,
        num_bins_y: int = 21,
        transform: Optional[Any] = None,
    ) -> None:
        self.session_dirs = [Path(d) for d in session_dirs]
        self.target_size = target_size  # (H, W)
        self.seq_len = max(1, seq_len)
        self.stride = max(1, stride)
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.transform = transform

        # Indices list: (session_idx, list_of_step_indices_for_window)
        self.samples_index: List[Tuple[int, List[int]]] = []
        self.sessions_data: List[Dict[str, Any]] = []

        self._load_sessions()

    def _load_sessions(self) -> None:
        """Parses metadata and samples.jsonl for all session directories."""
        for s_idx, s_dir in enumerate(self.session_dirs):
            meta_path = s_dir / "metadata.json"
            samples_path = s_dir / "samples.jsonl"

            if not meta_path.exists() or not samples_path.exists():
                logger.warning(f"Skipping incomplete session directory: {s_dir}")
                continue

            metadata = DatasetMetadata.load(meta_path)
            samples: List[DatasetSample] = []
            with open(samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        samples.append(DatasetSample.from_dict(json.loads(line_str)))

            if len(samples) < self.seq_len:
                logger.warning(
                    f"Session {s_dir} has fewer samples ({len(samples)}) than seq_len ({self.seq_len}), skipping."
                )
                continue

            self.sessions_data.append(
                {
                    "dir": s_dir,
                    "metadata": metadata,
                    "samples": samples,
                }
            )

            # Build sequence index windows
            effective_idx = len(self.sessions_data) - 1
            num_samples = len(samples)
            for end_idx in range(self.seq_len - 1, num_samples, self.stride):
                window_indices = list(range(end_idx - self.seq_len + 1, end_idx + 1))
                self.samples_index.append((effective_idx, window_indices))

    def __len__(self) -> int:
        return len(self.samples_index)

    def _load_image(self, session_dir: Path, rel_path: str) -> torch.Tensor:
        """Loads and converts an image file into a normalized float tensor [3, H, W] in [0, 1]."""
        full_path = session_dir / rel_path
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            if self.target_size is not None:
                h, w = self.target_size
                if img.size != (w, h):
                    img = img.resize((w, h), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, C]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        session_idx, step_indices = self.samples_index[idx]
        session = self.sessions_data[session_idx]
        session_dir = session["dir"]
        samples_list: List[DatasetSample] = session["samples"]

        # Load frames for sequence window
        frame_tensors: List[torch.Tensor] = []
        for s_i in step_indices:
            sample = samples_list[s_i]
            frame_t = self._load_image(session_dir, sample.frame_file)
            frame_tensors.append(frame_t)

        if self.seq_len == 1:
            obs = frame_tensors[0]  # [3, H, W]
        else:
            obs = torch.stack(frame_tensors, dim=0)  # [T, 3, H, W]

        # Target actions are taken from the latest step in the window
        last_sample = samples_list[step_indices[-1]]
        act = last_sample.actions

        # Map discrete movement: -1, 0, +1 -> 0, 1, 2
        move_x_target = int(act.move_x + 1)
        move_y_target = int(act.move_y + 1)

        target_dict: Dict[str, torch.Tensor] = {
            "move_x": torch.tensor(move_x_target, dtype=torch.long),
            "move_y": torch.tensor(move_y_target, dtype=torch.long),
            "jump": torch.tensor(int(act.jump), dtype=torch.long),
            "crouch": torch.tensor(int(act.crouch), dtype=torch.long),
            "sprint": torch.tensor(int(act.sprint), dtype=torch.long),
            "reload": torch.tensor(int(act.reload), dtype=torch.long),
            "fire": torch.tensor(int(act.fire), dtype=torch.long),
            "ads": torch.tensor(int(act.ads), dtype=torch.long),
            "mouse_dx_bin": torch.tensor(
                max(0, min(act.mouse_dx_bin, self.num_bins_x - 1)), dtype=torch.long
            ),
            "mouse_dy_bin": torch.tensor(
                max(0, min(act.mouse_dy_bin, self.num_bins_y - 1)), dtype=torch.long
            ),
            "mouse_continuous": torch.tensor(
                [act.mouse_dx, act.mouse_dy], dtype=torch.float32
            ),
        }

        return obs, target_dict

    @classmethod
    def create_train_val_split(
        cls,
        data_root: Union[str, Path],
        val_ratio: float = 0.2,
        seed: int = 42,
        target_size: Optional[Tuple[int, int]] = (120, 160),
        seq_len: int = 1,
        stride: int = 1,
    ) -> Tuple[GameplayDataset, GameplayDataset]:
        """Splits recording sessions at the SESSION level (not frame level) for robust validation."""
        root = Path(data_root)
        session_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
        if not session_dirs:
            # Check if root itself is a session directory
            if (root / "metadata.json").exists():
                session_dirs = [root]

        if not session_dirs:
            raise ValueError(f"No valid session directories found in {data_root}")

        rng = random.Random(seed)
        shuffled = list(session_dirs)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            # Single session fallback: train and val use same session
            train_dirs = shuffled
            val_dirs = shuffled
        else:
            num_val = max(1, int(len(shuffled) * val_ratio))
            val_dirs = shuffled[:num_val]
            train_dirs = shuffled[num_val:]

        train_ds = cls(train_dirs, target_size=target_size, seq_len=seq_len, stride=stride)
        val_ds = cls(val_dirs, target_size=target_size, seq_len=seq_len, stride=stride)

        return train_ds, val_ds
