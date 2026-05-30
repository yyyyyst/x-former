"""Multi-source radiomics encoder with per-sample dataset-specific projection."""
import torch
import torch.nn as nn


class RadiomicsEncoder(nn.Module):
    """Encodes variable-feature, variable-length radiomics sequences.

    - Per-sample dataset-specific linear projectors (handles mixed batches).
    - Shared temporal Transformer learns perfusion dynamics across datasets.
    - Optional lesion branch handles static shape features (disabled by default).
    """

    def __init__(self, proj_dim: int = 128, nhead: int = 4, nlayers: int = 2,
                 dropout: float = 0.1, use_lesion: bool = False) -> None:
        super().__init__()
        self.use_lesion = use_lesion
        self.proj_77sets = nn.Linear(16, proj_dim)
        self.proj_isle_brain = nn.Linear(15, proj_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=nhead, dim_feedforward=proj_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.norm = nn.LayerNorm(proj_dim)

        if use_lesion:
            self.lesion_mlp = nn.Sequential(
                nn.Linear(6, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.lesion_norm = nn.LayerNorm(proj_dim)

    def forward(self, radiomics: torch.Tensor, dataset_id: torch.Tensor,
                radiomics_mask: torch.Tensor | None = None,
                lesion_radiomics: torch.Tensor | None = None,
                has_lesion: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, F_padded = radiomics.shape
        device = radiomics.device

        # Per-sample projection: route each sample through its correct projector
        is_77 = (dataset_id == 0)  # [B]
        is_isle = (dataset_id == 1)  # [B]
        x = torch.zeros(B, T, self.proj_77sets.out_features, device=device)
        if is_77.any():
            x[is_77] = self.proj_77sets(radiomics[is_77]).float()
        if is_isle.any():
            # ISLE has 15 real features padded to 16; strip padding before projection
            x[is_isle] = self.proj_isle_brain(radiomics[is_isle, :, :15]).float()

        # Add CLS token and run temporal transformer
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.temporal_encoder(x)
        x = self.norm(x)

        brain_token = x.mean(dim=1, keepdim=True)  # [B, 1, proj_dim]

        # Lesion branch: compute for samples with actual lesion data
        lesion_token: torch.Tensor | None = None
        if self.use_lesion and lesion_radiomics is not None and has_lesion is not None:
            lesion_mask = has_lesion.view(B, 1, 1).float()
            lr = lesion_radiomics.to(device)
            lr_feat = self.lesion_mlp(lr)  # [B, N, proj_dim]
            lr_feat = lr_feat.mean(dim=1, keepdim=True)  # [B, 1, proj_dim]
            lesion_token = self.lesion_norm(lr_feat) * lesion_mask

        return brain_token, lesion_token
