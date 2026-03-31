"""
DINOv2 Feature Extractor for Deepfake Detection

Loads pretrained DINOv2 model for self-supervised features.
Outputs feature embeddings (frozen).
"""

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Default DINOv2 model (ViT-base)
DINOV2_MODEL_ID = "facebook/dinov2-base"


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 vision transformer for feature extraction.
    Outputs 768-dim (base) or 1024-dim (large) features.
    """

    def __init__(
        self,
        model_id: str = DINOV2_MODEL_ID,
        freeze: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace model id (e.g. facebook/dinov2-base).
            freeze: Freeze all parameters.
        """
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        self.model = AutoModel.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, 224, 224] (normalized)

        Returns:
            Feature tensor [B, feature_dim] (CLS token output)
        """
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
