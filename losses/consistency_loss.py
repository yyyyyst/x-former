"""JS-divergence consistency loss with fp16-safe numerics."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Computes JS divergence between predictions with/without lesion radiomics.
    Uses log_softmax for fp16 numerical stability.
    """

    def forward(self, logits_with: torch.Tensor, logits_without: torch.Tensor) -> torch.Tensor:
        # Use log_softmax for fp16 safety (avoids softmax underflow -> log(0) = -inf)
        log_p_with = F.log_softmax(logits_with, dim=-1)
        log_p_without = F.log_softmax(logits_without, dim=-1)
        p_with = log_p_with.exp()
        p_without = log_p_without.exp()
        m = 0.5 * (p_with + p_without)
        log_m = torch.log(m.clamp(min=1e-8))
        js = 0.5 * (F.kl_div(log_m, p_with, reduction='batchmean', log_target=False) +
                     F.kl_div(log_m, p_without, reduction='batchmean', log_target=False))
        return js
