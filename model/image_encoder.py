"""Swin-UNETR + Perceiver image encoder. Compresses 3D feature maps to fixed tokens."""
import os
import sys
from pathlib import Path

_project_root = Path(__file__).absolute().parent.parent
_paper1_model = _project_root.parent / "Tri-CAF-for-Stroke-Prediction" / "model"
if str(_paper1_model) not in sys.path:
    sys.path.insert(0, str(_paper1_model))

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_77sets_StrongCrossAttentionFusion import PretrainedSwin3D


class PerceiverCompressor(nn.Module):
    """Cross-attention compression: K learnable latents attend to flattened features."""

    def __init__(self, in_dim: int = 512, latent_dim: int = 256, n_latents: int = 32,
                 nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(latent_dim, nhead, dropout=dropout,
                                                 batch_first=True)
        self.proj_in = nn.Linear(in_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = features.shape
        spatial = features.flatten(2).permute(0, 2, 1)  # [B, S, C]
        spatial = self.proj_in(spatial)  # [B, S, latent_dim]
        latents = self.latents.expand(B, -1, -1)  # [B, K, latent_dim]
        compressed, _ = self.cross_attn(query=latents, key=spatial, value=spatial)
        return self.norm(compressed)  # [B, K, latent_dim]


class ImageEncoder(nn.Module):
    """Swin-UNETR backbone + optional Perceiver compression -> fixed token output."""

    def __init__(self, input_channels: int = 3, target_shape: tuple = (64, 160, 160),
                 n_latents: int = 32, latent_dim: int = 256,
                 weights_path: str | None = None, freeze_layers: int = 2,
                 use_perceiver: bool = True) -> None:
        super().__init__()
        self.use_perceiver = use_perceiver
        self.swin = PretrainedSwin3D(
            num_classes=latent_dim, input_channels=input_channels,
            weights_path=weights_path,
        )
        if use_perceiver:
            self.perceiver = PerceiverCompressor(
                in_dim=512, latent_dim=latent_dim, n_latents=n_latents,
            )
        else:
            self.perceiver = None
            self.gap_proj = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(1),
                nn.Linear(512, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        self.target_shape = target_shape
        self._apply_freeze_strategy(freeze_layers)

    def _apply_freeze_strategy(self, freeze_layers: int) -> None:
        for param in self.swin.swin.patch_embed.parameters():
            param.requires_grad = False
        if freeze_layers >= 1:
            for param in self.swin.swin.layers1.parameters():
                param.requires_grad = False
        if freeze_layers >= 2:
            for param in self.swin.swin.layers2.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.swin.extract_features(x)  # [B, 512, D, H, W]
        if self.use_perceiver:
            return self.perceiver(features)  # [B, K, latent_dim]
        else:
            return self.gap_proj(features).unsqueeze(1)  # [B, 1, latent_dim]
