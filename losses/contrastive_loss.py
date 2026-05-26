"""Supervised contrastive loss (Khosla et al., 2020)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D], labels: [B]
        features = F.normalize(features, dim=1)
        # Clamp similarity to avoid fp16 overflow in exp (max safe: ~11 for fp16)
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]
        sim = sim.clamp(max=10.0)

        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        pos_mask.fill_diagonal_(False)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        exp_sim = torch.exp(sim)
        pos_exp = (exp_sim * pos_mask.float()).sum(dim=1)
        all_exp = exp_sim.sum(dim=1) - exp_sim.diagonal()
        # Clamp both numerator and denominator to avoid log(0)=inf for samples
        # with no positive pair in the batch (e.g. singleton class)
        loss = -torch.log(pos_exp.clamp(min=1e-8) / all_exp.clamp(min=1e-8))
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        return (loss / n_pos).mean()
