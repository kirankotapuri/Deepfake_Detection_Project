"""
ResNet50 Feature Extractor for Deepfake Detection

Loads pretrained ResNet50, removes final classification layer,
freezes weights, and outputs feature vectors.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet50 backbone with final FC layer removed.
    Outputs 2048-dim feature vector per image.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """
        Args:
            pretrained: Use ImageNet pretrained weights.
            freeze: Freeze backbone parameters.
        """
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Feature tensor [B, 2048]
        """
        features = self.backbone(x)
        return features.flatten(1)
