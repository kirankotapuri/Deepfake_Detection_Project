"""
CLIP Feature Extractor for Deepfake Detection

Uses open_clip to load CLIP model and extract image embeddings.
All parameters are frozen.
"""

import torch
import torch.nn as nn

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


class CLIPFeatureExtractor(nn.Module):
    """
    CLIP vision encoder for feature extraction.
    Outputs embedding-dim feature vector (typically 512 or 768).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: CLIP architecture (e.g. ViT-B-32, ViT-L-14).
            pretrained: Pretrained weights (openai, laion2b, etc.).
            freeze: Freeze all parameters.
        """
        super().__init__()
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required. Install with: pip install open_clip_torch")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = model.visual
        self.feature_dim = self.visual.output_dim

        if freeze:
            for param in self.visual.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Feature tensor [B, feature_dim]
        """
        return self.visual(x)
