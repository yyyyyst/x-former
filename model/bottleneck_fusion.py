"""Cross-modal information bottleneck transformer — the core innovation of X-Former."""
import torch
import torch.nn as nn


class BottleneckLayer(nn.Module):
    """One bottleneck layer: self-attn → compress to slots → expand back → FFN."""

    def __init__(self, dim: int = 256, n_slots: int = 8, nhead: int = 8,
                 ff_expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim

        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # Compress: all tokens → bottleneck slots (query = learnable slots)
        self.compress_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout,
                                                    batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        # Expand: bottleneck slots → all tokens
        self.expand_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout,
                                                  batch_first=True)
        self.norm3 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention on all tokens
        t = self.norm1(tokens)
        t = tokens + self.self_attn(t, t, t, need_weights=False)[0]

        # Compress to bottleneck
        t_norm = self.norm2(t)
        s_norm = self.norm2(slots)  # share norm for compression
        s = slots + self.compress_attn(query=s_norm, key=t_norm, value=t_norm,
                                        need_weights=False)[0]

        # Expand back to tokens
        t_norm2 = self.norm3(t)
        s_norm2 = self.norm3(s)
        t = t + self.expand_attn(query=t_norm2, key=s_norm2, value=s_norm2,
                                  need_weights=False)[0]

        # FFN on tokens only
        t = t + self.ffn(self.norm4(t))
        return t, s


class BottleneckFusion(nn.Module):
    """Cross-modal information bottleneck: compresses heterogeneous modality tokens
    into a small set of shared slots, forcing competition for limited capacity."""

    def __init__(self, dim: int = 256, n_slots: int = 8, n_layers: int = 4,
                 nhead: int = 8, ff_expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.layers = nn.ModuleList([
            BottleneckLayer(dim=dim, n_slots=n_slots, nhead=nhead,
                            ff_expansion=ff_expansion, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, dim] — concatenated tokens from all modalities
        B = tokens.shape[0]
        slots = self.slots.expand(B, -1, -1)  # [B, n_slots, dim]

        for layer in self.layers:
            tokens, slots = layer(tokens, slots)

        return self.norm(slots)  # [B, n_slots, dim]
