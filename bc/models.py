"""Behavioral Cloning Model Architectures for SandboxAI (M1/M2).

Lightweight vision backbone (IMPALA-style residual CNN) with optional temporal GRU
and multi-head action prediction for discrete WASD, button actions, and mouse bin rotations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A standard residual convolutional block for IMPALA-style networks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class ConvSequence(nn.Module):
    """A downsampling convolution block followed by two residual blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class BCVisionNetwork(nn.Module):
    """Multi-task Behavioral Cloning neural network for tactical FPS imitation learning."""

    def __init__(
        self,
        in_channels: int = 3,
        num_bins_x: int = 21,
        num_bins_y: int = 21,
        latent_dim: int = 256,
        use_temporal_gru: bool = False,
        gru_hidden_dim: int = 256,
        channel_scales: Tuple[int, int, int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.latent_dim = latent_dim
        self.use_temporal_gru = use_temporal_gru
        self.gru_hidden_dim = gru_hidden_dim

        c1, c2, c3 = channel_scales
        self.encoder = nn.Sequential(
            ConvSequence(in_channels, c1),
            ConvSequence(c1, c2),
            ConvSequence(c2, c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        encoder_out_dim = c3 * 4 * 4
        self.fc_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.ReLU(),
        )

        if use_temporal_gru:
            self.gru = nn.GRU(
                input_size=latent_dim,
                hidden_size=gru_hidden_dim,
                batch_first=True,
            )
            head_in_dim = gru_hidden_dim
        else:
            self.gru = None
            head_in_dim = latent_dim

        # Multi-task Action Prediction Heads
        self.head_move_x = nn.Linear(head_in_dim, 3)  # 0=Left, 1=None, 2=Right
        self.head_move_y = nn.Linear(head_in_dim, 3)  # 0=Back, 1=None, 2=Forward
        self.head_jump = nn.Linear(head_in_dim, 2)  # Binary
        self.head_crouch = nn.Linear(head_in_dim, 2)  # Binary
        self.head_sprint = nn.Linear(head_in_dim, 2)  # Binary
        self.head_reload = nn.Linear(head_in_dim, 2)  # Binary
        self.head_fire = nn.Linear(head_in_dim, 2)  # Binary
        self.head_ads = nn.Linear(head_in_dim, 2)  # Binary
        self.head_mouse_dx = nn.Linear(head_in_dim, num_bins_x)  # Discretized horizontal
        self.head_mouse_dy = nn.Linear(head_in_dim, num_bins_y)  # Discretized vertical
        self.head_mouse_cont = nn.Linear(head_in_dim, 2)  # Continuous dx, dy regression

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, C, H, W] or sequence [B, T, C, H, W].
            hx: Optional recurrent hidden state for GRU.

        Returns:
            (logits_dict, new_hx)
        """
        is_sequence = x.dim() == 5
        if is_sequence:
            b, t, c, h, w = x.shape
            # Fold time into batch for CNN encoder
            x_flat = x.view(b * t, c, h, w)
            features = self.encoder(x_flat)
            latent = self.fc_proj(features)
            latent_seq = latent.view(b, t, -1)

            if self.gru is not None:
                gru_out, new_hx = self.gru(latent_seq, hx)
                rep = gru_out[:, -1, :]  # Take final timestep representation
            else:
                rep = latent_seq[:, -1, :]
                new_hx = None
        else:
            # Single frame [B, C, H, W]
            features = self.encoder(x)
            latent = self.fc_proj(features)

            if self.gru is not None:
                latent_unsq = latent.unsqueeze(1)  # [B, 1, D]
                gru_out, new_hx = self.gru(latent_unsq, hx)
                rep = gru_out.squeeze(1)
            else:
                rep = latent
                new_hx = None

        logits_dict = {
            "move_x": self.head_move_x(rep),
            "move_y": self.head_move_y(rep),
            "jump": self.head_jump(rep),
            "crouch": self.head_crouch(rep),
            "sprint": self.head_sprint(rep),
            "reload": self.head_reload(rep),
            "fire": self.head_fire(rep),
            "ads": self.head_ads(rep),
            "mouse_dx_bin": self.head_mouse_dx(rep),
            "mouse_dy_bin": self.head_mouse_dy(rep),
            "mouse_continuous": self.head_mouse_cont(rep),
        }

        return logits_dict, new_hx

    @staticmethod
    def extract_discrete_actions(
        logits_dict: Dict[str, torch.Tensor],
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Extracts canonical action space values from predicted head logits."""
        actions: Dict[str, Any] = {}

        for key, logits in logits_dict.items():
            if key == "mouse_continuous":
                actions["mouse_continuous"] = logits.detach().cpu().numpy()
                continue

            if deterministic:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()

            if key in ("move_x", "move_y"):
                # Map 0, 1, 2 back to -1, 0, 1
                actions[key] = preds - 1
            else:
                actions[key] = preds

        return actions
