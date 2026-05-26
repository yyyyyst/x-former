"""Prototype-based universal clinical encoder with feature masking for padded inputs."""
import torch
import torch.nn as nn


class PrototypeClinicalEncoder(nn.Module):
    """M learnable prototypes attend to clinical features via per-feature embedding.
    The feature_mask parameter allows heterogeneous feature counts in joint training.
    """

    def __init__(self, max_features: int = 35, n_prototypes: int = 8,
                 proto_dim: int = 128, n_datasets: int = 2,
                 use_dataset_embedding: bool = True) -> None:
        super().__init__()
        self.use_dataset_embedding = use_dataset_embedding
        self.feature_embed = nn.ModuleList([
            nn.Linear(1, proto_dim) for _ in range(max_features)
        ])
        self.prototypes = nn.Parameter(torch.randn(1, n_prototypes, proto_dim) * 0.02)
        if use_dataset_embedding:
            self.dataset_embed = nn.Embedding(n_datasets, proto_dim)
        self.norm = nn.LayerNorm(proto_dim)
        self.n_prototypes = n_prototypes
        self.proto_dim = proto_dim

    def forward(self, x: torch.Tensor, dataset_id: torch.Tensor,
                feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, F = x.shape
        D = self.proto_dim
        M = self.n_prototypes

        # Embed each feature position: [B, F] -> [B, F, D]
        feat_embeds = []
        for i in range(F):
            fi = x[:, i:i + 1]  # [B, 1]
            emb = self.feature_embed[i](fi)  # [B, D]
            feat_embeds.append(emb)
        feats = torch.stack(feat_embeds, dim=1)  # [B, F, D]

        # Compute prototype attention over all feature embeddings
        protos = self.prototypes.expand(B, -1, -1)  # [B, M, D]
        scores = torch.bmm(protos, feats.transpose(1, 2))  # [B, M, F]

        # Mask padded feature positions (score = -inf so softmax = 0)
        if feature_mask is not None:
            mask_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~feature_mask.unsqueeze(1), mask_val)

        attn = torch.softmax(scores / (D ** 0.5), dim=-1)  # [B, M, F]
        tokens = torch.bmm(attn, feats)  # [B, M, D]

        if self.use_dataset_embedding:
            ds_embed = self.dataset_embed(dataset_id).unsqueeze(1)  # [B, 1, D]
            tokens = tokens + ds_embed
        return self.norm(tokens)  # [B, M, D]
