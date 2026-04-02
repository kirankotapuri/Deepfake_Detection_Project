"""
Custom Loss Functions for Deepfake Detection
=============================================
Provides alternatives to standard CrossEntropyLoss for handling:
 - Class imbalance (common in deepfake datasets)
 - Hard examples (focal loss)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary / multi-class classification.

    Down-weights well-classified examples, focusing training on hard negatives.
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:  Class weighting scalar or tensor [num_classes].
                Use < 1 for the dominant class (real=0) to handle imbalance.
        gamma:  Focusing parameter. gamma=0 → standard CE. Typical: 2.0.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Logits [B, C]
            targets: Class indices [B]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt      = torch.exp(-ce_loss)                        # probability of correct class
        focal   = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float)
            else:
                alpha_t = self.alpha.to(inputs.device)[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with per-class weights for imbalanced datasets.

    Automatically computes weights from dataset class counts if not provided.

    Args:
        class_counts: [n_real, n_fake] — used to compute inverse-frequency weights.
        weight:       Manual weight tensor [num_classes]. Overrides class_counts.
    """

    def __init__(
        self,
        class_counts: list[int] | None = None,
        weight: torch.Tensor | None = None,
        num_classes: int = 2,
    ):
        super().__init__()
        if weight is not None:
            self.weight = weight
        elif class_counts is not None:
            total  = sum(class_counts)
            w      = [total / (num_classes * c) for c in class_counts]
            self.weight = torch.tensor(w, dtype=torch.float)
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        return F.cross_entropy(inputs, targets, weight=weight)


def get_loss_function(
    loss_type: str = "ce",
    class_counts: list[int] | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> nn.Module:
    """
    Factory function for loss selection.

    Args:
        loss_type:     'ce' (default), 'weighted_ce', or 'focal'.
        class_counts:  [n_real, n_fake] for weighted/focal losses.
        focal_gamma:   Gamma for focal loss.
        focal_alpha:   Alpha for focal loss.

    Returns:
        nn.Module loss function.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_counts=class_counts)

    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    raise ValueError(f"Unknown loss type: {loss_type!r}. Choose: ce | weighted_ce | focal")
