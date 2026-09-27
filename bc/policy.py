"""Inference Policy wrapper for trained Behavioral Cloning models (M1/M2).

Takes raw frame observations, runs neural policy inference, and maps predicted logits
to canonical ActionState and environment action commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from bc.models import BCVisionNetwork
from data_pipeline.actions import MouseBinner
from data_pipeline.schema import ActionState


class BCPolicy:
    """Trained Behavioral Cloning policy wrapper for real-time inference in sandbox or evaluations."""

    def __init__(
        self,
        model: BCVisionNetwork,
        config: Dict[str, Any],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config

        self.target_h = config.get("target_h", 120)
        self.target_w = config.get("target_w", 160)
        self.num_bins_x = config.get("num_bins_x", 21)
        self.num_bins_y = config.get("num_bins_y", 21)
        self.use_gru = config.get("use_gru", False)

        self.binner_x = MouseBinner(num_bins=self.num_bins_x, strategy="symmetric_log")
        self.binner_y = MouseBinner(num_bins=self.num_bins_y, strategy="symmetric_log")

        self._hx: Optional[torch.Tensor] = None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> BCPolicy:
        """Loads a BCPolicy directly from a saved .pt checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        config = ckpt.get("config", {})

        latent_dim = config.get("latent_dim", 256)
        channel_scales = tuple(config.get("channel_scales", (16, 32, 32)))

        model = BCVisionNetwork(
            in_channels=config.get("in_channels", 3),
            num_bins_x=config.get("num_bins_x", 21),
            num_bins_y=config.get("num_bins_y", 21),
            latent_dim=latent_dim,
            use_temporal_gru=config.get("use_gru", False),
            channel_scales=channel_scales,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, config=config, device=device)

    def reset(self) -> None:
        """Resets recurrent hidden states for a new episode."""
        self._hx = None

    def preprocess_observation(
        self,
        observation: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Converts raw image input into normalized torch tensor [1, 3, H, W]."""
        if isinstance(observation, Image.Image):
            img = observation.convert("RGB")
            if img.size != (self.target_w, self.target_h):
                img = img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        elif isinstance(observation, np.ndarray):
            # Shape can be [H, W, 3] or [3, H, W]
            arr = observation.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3 and arr.shape[-1] == 3:
                # [H, W, 3] -> [3, H, W]
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[0] == 3:
                # [3, H, W] -> [1, 3, H, W]
                tensor = torch.from_numpy(arr).unsqueeze(0)
            elif arr.ndim == 4:
                # [B, 3, H, W]
                tensor = torch.from_numpy(arr)
            else:
                raise ValueError(f"Unsupported observation array shape: {observation.shape}")
        elif isinstance(observation, torch.Tensor):
            tensor = observation.float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported observation type: {type(observation)}")

        return tensor.to(self.device)

    def predict(
        self,
        observation: Union[Image.Image, np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[ActionState, Dict[str, Any]]:
        """Predicts player actions for a given screen observation.

        Returns:
            (action_state, raw_action_dict)
        """
        obs_tensor = self.preprocess_observation(observation)

        with torch.no_grad():
            logits_dict, self._hx = self.model(obs_tensor, self._hx)
            extracted = BCVisionNetwork.extract_discrete_actions(
                logits_dict, deterministic=deterministic
            )

        # Map predictions to ActionState
        move_x = int(extracted["move_x"][0])
        move_y = int(extracted["move_y"][0])
        jump = int(extracted["jump"][0])
        crouch = int(extracted["crouch"][0])
        sprint = int(extracted["sprint"][0])
        reload_act = int(extracted["reload"][0])
        fire = int(extracted["fire"][0])
        ads = int(extracted["ads"][0])
        dx_bin = int(extracted["mouse_dx_bin"][0])
        dy_bin = int(extracted["mouse_dy_bin"][0])

        # Dequantize mouse deltas back to continuous pixel rotations
        dx_cont = self.binner_x.dequantize(dx_bin)
        dy_cont = self.binner_y.dequantize(dy_bin)

        active_keys = []
        if move_x == -1:
            active_keys.append("a")
        elif move_x == 1:
            active_keys.append("d")
        if move_y == 1:
            active_keys.append("w")
        elif move_y == -1:
            active_keys.append("s")
        if jump == 1:
            active_keys.append("space")
        if crouch == 1:
            active_keys.append("c")
        if sprint == 1:
            active_keys.append("shift")
        if reload_act == 1:
            active_keys.append("r")

        action_state = ActionState(
            move_x=move_x,
            move_y=move_y,
            jump=jump,
            crouch=crouch,
            sprint=sprint,
            reload=reload_act,
            fire=fire,
            ads=ads,
            mouse_dx=dx_cont,
            mouse_dy=dy_cont,
            mouse_dx_bin=dx_bin,
            mouse_dy_bin=dy_bin,
            active_keys=active_keys,
            mouse_buttons={"left": fire == 1, "right": ads == 1, "middle": False},
        )

        raw_dict = {
            "move_x": move_x,
            "move_y": move_y,
            "jump": jump,
            "crouch": crouch,
            "sprint": sprint,
            "reload": reload_act,
            "fire": fire,
            "ads": ads,
            "mouse_dx_bin": dx_bin,
            "mouse_dy_bin": dy_bin,
            "mouse_dx": dx_cont,
            "mouse_dy": dy_cont,
        }

        return action_state, raw_dict
