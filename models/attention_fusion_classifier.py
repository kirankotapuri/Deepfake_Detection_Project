"""
Improved fusion classifiers for multi-backbone deepfake detection.

Models:
1. ImprovedFusionClassifier  – ResNet + CLIP only (DINOv2 removed).
2. ImprovedFusionWithDINO    – All 3 backbones with constrained weights
   (CLIP=0.60, ResNet=0.30, DINOv2=0.10 init).
3. MultiBackboneAttentionFusionClassifier – Original attention-based (kept
   for comparison / backward compatibility).

All keep backbone extractors frozen and only train fusion heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ImprovedFusionClassifier – ResNet + CLIP only
# ---------------------------------------------------------------------------
class ImprovedFusionClassifier(nn.Module):
    """
    Weighted fusion of ResNet50 + CLIP only.
    DINOv2 removed because it consistently degrades performance.
    Scalar weights instead of MultiheadAttention to prevent overfitting
    on small datasets.
    """

    backbone_names: tuple[str, str] = ("resnet", "clip")

    def __init__(
        self,
        dim_resnet: int = 2048,
        dim_clip: int = 512,
        proj_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # Project both backbones to same dimension
        self.proj_resnet = nn.Sequential(
            nn.Linear(dim_resnet, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(dim_clip, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )

        # Single learnable weight for CLIP
        # Initialize to 0.65 (favor CLIP based on baseline knowledge)
        # ResNet weight = 1 - clip_weight automatically
        self.clip_weight = nn.Parameter(torch.tensor([0.65]))

        # Classifier on fused features
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        f_resnet: torch.Tensor,
        f_clip: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        # L2 normalize features BEFORE projection
        f_resnet = F.normalize(f_resnet, p=2, dim=1)
        f_clip = F.normalize(f_clip, p=2, dim=1)

        # Project to shared space
        r = self.proj_resnet(f_resnet)
        c = self.proj_clip(f_clip)

        # Weighted combination
        w_clip = torch.sigmoid(self.clip_weight)
        w_resnet = 1.0 - w_clip

        fused = w_clip * c + w_resnet * r

        logits = self.classifier(fused)

        if return_weights:
            return logits, {"clip": w_clip.item(), "resnet": w_resnet.item()}
        return logits


# ---------------------------------------------------------------------------
# ImprovedFusionWithDINO – All 3 backbones, constrained weights
# ---------------------------------------------------------------------------
class ImprovedFusionWithDINO(nn.Module):
    """
    Three backbone version WITH DINOv2 but with constrained weights.
    Use this if you want to keep DINOv2 but limit its influence.
    Initialises weights: CLIP=0.60, ResNet=0.30, DINOv2=0.10.
    """

    backbone_names: tuple[str, str, str] = ("resnet", "clip", "dinov2")

    def __init__(
        self,
        dim_resnet: int = 2048,
        dim_clip: int = 512,
        dim_dino: int = 768,
        proj_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.proj_resnet = nn.Sequential(
            nn.Linear(dim_resnet, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(dim_clip, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(dim_dino, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )

        # Initialise weights to reflect baseline knowledge
        # CLIP=0.60, ResNet=0.30, DINOv2=0.10
        self.raw_weights = nn.Parameter(
            torch.tensor([0.30, 0.60, 0.10])
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        f_resnet: torch.Tensor,
        f_clip: torch.Tensor,
        f_dino: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Normalize first
        f_resnet = F.normalize(f_resnet, p=2, dim=1)
        f_clip = F.normalize(f_clip, p=2, dim=1)
        f_dino = F.normalize(f_dino, p=2, dim=1)

        # Project
        r = self.proj_resnet(f_resnet)
        c = self.proj_clip(f_clip)
        d = self.proj_dino(f_dino)

        # Softmax weights (sum to 1)
        w = F.softmax(self.raw_weights, dim=0)

        # Weighted sum
        fused = w[0] * r + w[1] * c + w[2] * d

        logits = self.classifier(fused)

        if return_weights:
            return logits, w.detach()
        return logits


# ---------------------------------------------------------------------------
# Original attention-based model (backward compatibility)
# ---------------------------------------------------------------------------
class MultiBackboneAttentionFusionClassifier(nn.Module):
    """Original attention-based fusion (kept for comparison only)."""

    backbone_names: tuple[str, str, str] = ("resnet", "clip", "dinov2")

    def __init__(
        self,
        resnet_dim: int = 2048,
        clip_dim: int = 512,
        dinov2_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.proj_resnet = nn.Sequential(
            nn.Linear(resnet_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.proj_dinov2 = nn.Sequential(
            nn.Linear(dinov2_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _project(self, resnet_features, clip_features, dinov2_features):
        return torch.stack(
            (self.proj_resnet(resnet_features), self.proj_clip(clip_features), self.proj_dinov2(dinov2_features)),
            dim=1,
        )

    def forward(self, resnet_features, clip_features, dinov2_features, return_attention=False):
        tokens = self._project(resnet_features, clip_features, dinov2_features)
        attended, attention_weights = self.attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        fused_tokens = self.post_attention_norm(tokens + attended)
        pooled = fused_tokens.mean(dim=1)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attention_weights
        return logits

    @staticmethod
    def summarize_attention(attention_weights: torch.Tensor) -> torch.Tensor:
        if attention_weights.ndim != 4:
            raise ValueError("Expected shape [batch, heads, tgt_len, src_len].")
        return attention_weights.mean(dim=(0, 1, 2))
