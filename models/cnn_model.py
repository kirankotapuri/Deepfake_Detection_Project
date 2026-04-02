"""
Lightweight Custom CNN Feature Extractor
=========================================
A 5-block convolutional network designed as a lightweight alternative to
ResNet50 when training from scratch on small datasets.

Output: 512-dim feature vector per image.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFeatureExtractor(nn.Module):
    """
    Lightweight 5-block CNN backbone.

    Input:  [B, 3, 224, 224]
    Output: [B, 512]  (global average-pooled feature vector)

    Block progression (channels):
        3 → 32 → 64 → 128 → 256 → 512
    Spatial:  224 → 112 → 56 → 28 → 14 → 7 → 1 (gap)
    """

    def __init__(self, freeze: bool = False):
        super().__init__()
        self.feature_dim = 512
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # → 112×112
            ConvBlock(32,  64,  pool=True),   # →  56×56
            ConvBlock(64,  128, pool=True),   # →  28×28
            ConvBlock(128, 256, pool=True),   # →  14×14
            ConvBlock(256, 512, pool=True),   # →   7×7
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # →   1×1

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            [B, 512] feature vector
        """
        x = self.features(x)
        x = self.gap(x)
        return x.flatten(1)


def build_cnn_extractor(pretrained: bool = False, freeze: bool = False) -> CNNFeatureExtractor:
    """
    Factory function for CNNFeatureExtractor.

    Args:
        pretrained: Not used (custom architecture has no pretrained weights).
        freeze:     Whether to freeze all parameters after creation.
    """
    return CNNFeatureExtractor(freeze=freeze)
