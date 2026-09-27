"""Stateful real-time inference wrapper for behavioral-cloning checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bc.models import BCVisionNetwork
from data_pipeline.actions import MouseBinner
from data_pipeline.schema import ActionState
from sandbox.actions import SandboxAction


class BCPolicy:
    """Convert visual observations into the canonical FPS action contract."""

    def __init__(
        self,
        model: BCVisionNetwork,
        config: Dict[str, Any],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.config = dict(config)
        self.target_h = int(config.get("target_h", 120))
        self.target_w = int(config.get("target_w", 160))
        self.num_bins_x = int(config.get("num_bins_x", 21))
        self.num_bins_y = int(config.get("num_bins_y", 21))
        self.use_gru = bool(config.get("use_gru", False))
        self.seq_len = int(config.get("seq_len", 1))
        self.binner_x = MouseBinner(
            num_bins=self.num_bins_x,
            strategy=str(config.get("mouse_binning_strategy", "symmetric_log")),
            custom_edges=config.get("mouse_bin_edges_x"),
        )
        self.binner_y = MouseBinner(
            num_bins=self.num_bins_y,
            strategy=str(config.get("mouse_binning_strategy", "symmetric_log")),
            custom_edges=config.get("mouse_bin_edges_y"),
        )
        # The controlled sandbox has one stable 21-bin contract even when an
        # offline dataset was recorded with alternate bin edges/counts.
        self.env_binner_x = MouseBinner(num_bins=21, strategy="symmetric_log")
        self.env_binner_y = MouseBinner(num_bins=21, strategy="symmetric_log")
        self._hx: Optional[torch.Tensor] = None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "BCPolicy":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"BC checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Not a SandboxAI BC checkpoint: {path}")
        config = checkpoint.get("config", {})
        model = BCVisionNetwork(
            in_channels=int(config.get("in_channels", 3)),
            num_bins_x=int(config.get("num_bins_x", 21)),
            num_bins_y=int(config.get("num_bins_y", 21)),
            latent_dim=int(config.get("latent_dim", 256)),
            use_temporal_gru=bool(config.get("use_gru", False)),
            gru_hidden_dim=int(config.get("gru_hidden_dim", config.get("latent_dim", 256))),
            channel_scales=tuple(config.get("channel_scales", (16, 32, 32))),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=config, device=device)

    def reset(self) -> None:
        self._hx = None

    def preprocess_observation(
        self, observation: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize and resize HWC/CHW/single-batch RGB observations."""
        if isinstance(observation, Image.Image):
            image = observation.convert("RGB")
            if image.size != (self.target_w, self.target_h):
                image = image.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(observation, np.ndarray):
            array = np.asarray(observation, dtype=np.float32)
            if array.size and float(array.max()) > 1.0:
                array = array / 255.0
            if array.ndim == 3 and array.shape[-1] == 3:
                tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
            elif array.ndim == 3 and array.shape[0] == 3:
                tensor = torch.from_numpy(array.copy()).unsqueeze(0)
            elif array.ndim == 4 and array.shape[1] == 3:
                tensor = torch.from_numpy(array.copy())
            else:
                raise ValueError(f"Unsupported observation array shape: {array.shape}")
        elif isinstance(observation, torch.Tensor):
            tensor = observation.detach().float()
            if tensor.numel() and float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 4 or tensor.shape[1] != 3:
                raise ValueError(f"Unsupported observation tensor shape: {tuple(tensor.shape)}")
        else:
            raise TypeError(f"Unsupported observation type: {type(observation)!r}")
        if tensor.shape[-2:] != (self.target_h, self.target_w):
            tensor = F.interpolate(
                tensor, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
            )
        return tensor.to(self.device)

    def predict(
        self,
        observation: Union[Image.Image, np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[ActionState, Dict[str, Any]]:
        tensor = self.preprocess_observation(observation)
        with torch.inference_mode():
            logits, self._hx = self.model(tensor, self._hx)
            extracted = BCVisionNetwork.extract_discrete_actions(
                logits, deterministic=deterministic
            )
            confidences = {
                key: float(F.softmax(value, dim=-1).max(dim=-1).values[0].cpu())
                for key, value in logits.items()
                if key != "mouse_continuous"
            }

        values = {
            key: int(extracted[key][0])
            for key in (
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
        }
        dx = self.binner_x.dequantize(values["mouse_dx_bin"])
        dy = self.binner_y.dequantize(values["mouse_dy_bin"])
        keys = []
        if values["move_x"] < 0:
            keys.append("a")
        elif values["move_x"] > 0:
            keys.append("d")
        if values["move_y"] < 0:
            keys.append("s")
        elif values["move_y"] > 0:
            keys.append("w")
        for enabled, key in (
            (values["jump"], "space"),
            (values["crouch"], "c"),
            (values["sprint"], "shift"),
            (values["reload"], "r"),
        ):
            if enabled:
                keys.append(key)
        action_state = ActionState(
            **values,
            mouse_dx=dx,
            mouse_dy=dy,
            active_keys=keys,
            mouse_buttons={
                "left": values["fire"] == 1,
                "right": values["ads"] == 1,
                "middle": False,
            },
        )
        raw = {**values, "mouse_dx": dx, "mouse_dy": dy, "confidence": confidences}
        return action_state, raw

    def predict_env_action(
        self,
        observation: Union[Image.Image, np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action_state, raw = self.predict(observation, deterministic=deterministic)
        env_action = SandboxAction.from_action_state(action_state)
        canonical_dx = self.env_binner_x.discretize(action_state.mouse_dx)
        canonical_dy = self.env_binner_y.discretize(action_state.mouse_dy)
        env_action = SandboxAction(
            **{
                **env_action.to_dict(),
                "mouse_dx_bin": canonical_dx,
                "mouse_dy_bin": canonical_dy,
            }
        )
        raw["env_mouse_dx_bin"] = canonical_dx
        raw["env_mouse_dy_bin"] = canonical_dy
        return env_action.to_array(), raw
