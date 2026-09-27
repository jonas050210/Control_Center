"""Versioned gameplay dataset loader for temporal behavioral cloning."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.schema import DatasetMetadata, DatasetSample, SCHEMA_VERSION

logger = logging.getLogger("SandboxAI.BC.Dataset")


class GameplayDataset(Dataset):
    """Load frame sequences without allowing windows to cross session boundaries.

    ``step_ranges`` maps resolved session paths to an inclusive-start,
    exclusive-end range.  It is used for a leakage-free chronological split
    when only one recording session is available.
    """

    def __init__(
        self,
        session_dirs: Sequence[Union[str, Path]],
        target_size: Optional[Tuple[int, int]] = (120, 160),
        seq_len: int = 1,
        stride: int = 1,
        num_bins_x: Optional[int] = None,
        num_bins_y: Optional[int] = None,
        transform: Optional[Any] = None,
        augment: bool = False,
        step_ranges: Optional[Mapping[str, Tuple[int, int]]] = None,
    ) -> None:
        self.session_dirs = [Path(directory) for directory in session_dirs]
        self.target_size = target_size
        self.seq_len = max(1, int(seq_len))
        self.stride = max(1, int(stride))
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.transform = transform
        self.augment = augment
        self.step_ranges = dict(step_ranges or {})
        self.samples_index: List[Tuple[int, List[int]]] = []
        self.sessions_data: List[Dict[str, Any]] = []
        self._load_sessions()
        if self.num_bins_x is None:
            self.num_bins_x = 21
        if self.num_bins_y is None:
            self.num_bins_y = 21

    @staticmethod
    def discover_sessions(data_root: Union[str, Path]) -> List[Path]:
        root = Path(data_root)
        if (root / "metadata.json").is_file() and (root / "samples.jsonl").is_file():
            return [root]
        if not root.exists():
            return []
        return sorted(
            metadata.parent
            for metadata in root.rglob("metadata.json")
            if (metadata.parent / "samples.jsonl").is_file()
        )

    def _range_for(self, session_dir: Path, count: int) -> Tuple[int, int]:
        keys = (str(session_dir.resolve()), str(session_dir), session_dir.name)
        for key in keys:
            if key in self.step_ranges:
                start, end = self.step_ranges[key]
                return max(0, int(start)), min(count, int(end))
        return 0, count

    def _load_sessions(self) -> None:
        expected_bins: Optional[Tuple[int, int]] = None
        for session_dir in self.session_dirs:
            meta_path = session_dir / "metadata.json"
            samples_path = session_dir / "samples.jsonl"
            if not meta_path.is_file() or not samples_path.is_file():
                logger.warning("Skipping incomplete session directory: %s", session_dir)
                continue
            metadata = DatasetMetadata.load(meta_path)
            if metadata.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                raise ValueError(
                    f"Unsupported schema {metadata.schema_version} in {session_dir}; expected {SCHEMA_VERSION}"
                )
            bins = (metadata.mouse_config.num_bins_x, metadata.mouse_config.num_bins_y)
            if expected_bins is None:
                expected_bins = bins
                self.num_bins_x = self.num_bins_x or bins[0]
                self.num_bins_y = self.num_bins_y or bins[1]
            elif bins != expected_bins:
                raise ValueError(
                    f"Mouse-bin mismatch: {session_dir} uses {bins}, expected {expected_bins}. "
                    "Re-bin sessions before combining them."
                )

            samples: List[DatasetSample] = []
            with samples_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = DatasetSample.from_dict(json.loads(line))
                    except Exception as exc:
                        raise ValueError(f"Malformed sample {samples_path}:{line_no}: {exc}") from exc
                    if sample.is_valid:
                        samples.append(sample)
            range_start, range_end = self._range_for(session_dir, len(samples))
            if range_end - range_start < self.seq_len:
                logger.warning(
                    "Session %s range [%d,%d) has fewer than seq_len=%d samples; skipping",
                    session_dir,
                    range_start,
                    range_end,
                    self.seq_len,
                )
                continue
            self.sessions_data.append(
                {"dir": session_dir, "metadata": metadata, "samples": samples}
            )
            session_idx = len(self.sessions_data) - 1
            for end_idx in range(range_start + self.seq_len - 1, range_end, self.stride):
                indices = list(range(end_idx - self.seq_len + 1, end_idx + 1))
                self.samples_index.append((session_idx, indices))

    def __len__(self) -> int:
        return len(self.samples_index)

    def _load_image(self, session_dir: Path, relative_path: str) -> torch.Tensor:
        full_path = (session_dir / relative_path).resolve()
        try:
            full_path.relative_to(session_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Frame path escapes session directory: {relative_path}") from exc
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            if self.target_size is not None:
                height, width = self.target_size
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        if self.augment:
            # Lightweight GPU-independent domain randomization suitable for an
            # 8 GB card. Geometry is intentionally unchanged so aiming labels
            # remain valid.
            brightness = 0.88 + torch.rand(()) * 0.24
            contrast = 0.88 + torch.rand(()) * 0.24
            channel_scale = 0.94 + torch.rand((3, 1, 1)) * 0.12
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            tensor = ((tensor - mean) * contrast + mean) * brightness * channel_scale
            if torch.rand(()) < 0.25:
                tensor = tensor + torch.randn_like(tensor) * 0.012
            tensor = tensor.clamp_(0.0, 1.0)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        session_idx, step_indices = self.samples_index[idx]
        session = self.sessions_data[session_idx]
        samples: List[DatasetSample] = session["samples"]
        frames = [
            self._load_image(session["dir"], samples[step_idx].frame_file)
            for step_idx in step_indices
        ]
        observation = frames[0] if self.seq_len == 1 else torch.stack(frames, dim=0)
        action = samples[step_indices[-1]].actions
        targets = {
            "move_x": torch.tensor(action.move_x + 1, dtype=torch.long),
            "move_y": torch.tensor(action.move_y + 1, dtype=torch.long),
            "jump": torch.tensor(int(action.jump), dtype=torch.long),
            "crouch": torch.tensor(int(action.crouch), dtype=torch.long),
            "sprint": torch.tensor(int(action.sprint), dtype=torch.long),
            "reload": torch.tensor(int(action.reload), dtype=torch.long),
            "fire": torch.tensor(int(action.fire), dtype=torch.long),
            "ads": torch.tensor(int(action.ads), dtype=torch.long),
            "mouse_dx_bin": torch.tensor(
                max(0, min(action.mouse_dx_bin, int(self.num_bins_x) - 1)), dtype=torch.long
            ),
            "mouse_dy_bin": torch.tensor(
                max(0, min(action.mouse_dy_bin, int(self.num_bins_y) - 1)), dtype=torch.long
            ),
            "mouse_continuous": torch.tensor(
                [action.mouse_dx, action.mouse_dy], dtype=torch.float32
            ),
        }
        return observation, targets

    @classmethod
    def create_train_val_split(
        cls,
        data_root: Union[str, Path],
        val_ratio: float = 0.2,
        seed: int = 42,
        target_size: Optional[Tuple[int, int]] = (120, 160),
        seq_len: int = 1,
        stride: int = 1,
        augment_train: bool = False,
    ) -> Tuple["GameplayDataset", "GameplayDataset"]:
        """Split by session, or chronologically with a context gap for one session."""
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")
        session_dirs = cls.discover_sessions(data_root)
        if not session_dirs:
            raise ValueError(f"No valid session directories found in {data_root}")
        rng = random.Random(seed)
        shuffled = list(session_dirs)
        rng.shuffle(shuffled)

        common = dict(target_size=target_size, seq_len=seq_len, stride=stride)
        if len(shuffled) > 1:
            num_val = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_ratio)))
            val_dirs, train_dirs = shuffled[:num_val], shuffled[num_val:]
            train_ds = cls(train_dirs, augment=augment_train, **common)
            val_ds = cls(val_dirs, augment=False, **common)
        else:
            session = shuffled[0]
            with (session / "samples.jsonl").open("r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
            split = max(seq_len, min(count - seq_len, int(count * (1.0 - val_ratio))))
            if count < seq_len * 2:
                # Tiny smoke datasets cannot support disjoint context windows.
                # Keep compatibility but make the leakage explicit to callers.
                logger.warning(
                    "Only %d samples are available; train/validation share the tiny session", count
                )
                train_ds = cls([session], augment=augment_train, **common)
                val_ds = cls([session], augment=False, **common)
            else:
                key = str(session.resolve())
                train_ds = cls(
                    [session], augment=augment_train, step_ranges={key: (0, split)}, **common
                )
                val_start = min(count - seq_len, split + max(0, seq_len - 1))
                val_ds = cls(
                    [session], augment=False, step_ranges={key: (val_start, count)}, **common
                )
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Train/validation split produced an empty dataset")
        return train_ds, val_ds
