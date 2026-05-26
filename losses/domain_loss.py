"""Domain adversarial loss with Gradient Reversal Layer (GRL)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversal(Function):
    """Gradient Reversal Layer: identity in forward, negates gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output.neg() * ctx.lambda_, None


class DomainLoss(nn.Module):
    """Domain adversarial loss. GRL should be applied BEFORE logits are computed.
    This module is kept for standalone use; in X-Former, GRL is applied internally."""

    def __init__(self, grl_lambda: float = 0.1) -> None:
        super().__init__()
        self.grl_lambda = grl_lambda

    def forward(self, domain_logits: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(domain_logits, dataset_id)
