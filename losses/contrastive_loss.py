"""Supervised contrastive losses for cross-dataset representation learning."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        same_dataset_weight: float = 0.5,
        cross_dataset_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.same_dataset_weight = same_dataset_weight
        self.cross_dataset_weight = cross_dataset_weight

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        dataset_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted SupCon.

        When dataset_id is provided, same-label cross-dataset positives receive
        higher weight than same-dataset positives to align MRI/CT outcome space.
        """
        # features: [B, D], labels/dataset_id: [B]
        features = F.normalize(features.float(), dim=1)
        logits = torch.matmul(features, features.T) / self.temperature  # [B, B]

        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        pos_mask.fill_diagonal_(False)

        if pos_mask.sum() == 0:
            return features.sum() * 0.0

        logits_mask = torch.ones_like(pos_mask, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

        pos_weights = pos_mask.float()
        if dataset_id is not None:
            cross_dataset = dataset_id.unsqueeze(0) != dataset_id.unsqueeze(1)
            same_weights = torch.full_like(pos_weights, self.same_dataset_weight)
            cross_weights = torch.full_like(pos_weights, self.cross_dataset_weight)
            pos_weights = pos_weights * torch.where(cross_dataset, cross_weights, same_weights)

        pos_weight_sum = pos_weights.sum(dim=1)
        valid = pos_weight_sum > 0
        if not valid.any():
            return features.sum() * 0.0

        loss = -(pos_weights * log_prob).sum(dim=1) / pos_weight_sum.clamp(min=1e-8)
        return loss[valid].mean()
