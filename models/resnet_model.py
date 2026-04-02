"""
Full ResNet Deepfake Detection Model (Extractor + Classifier)
==============================================================
Combines the frozen ResNet50 feature extractor with the MLP classifier
into a single end-to-end nn.Module for deployment / GradCAM use.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from models.resnet_feature_extractor import ResNetFeatureExtractor
from training.train_classifier import LinearClassifier


class ResNetDeepfakeDetector(nn.Module):
    """
    End-to-end deepfake detector:
        ResNet50 (frozen backbone)  →  MLP classifier head

    Useful for:
        - Single-call inference
        - GradCAM visualization (target_layer accessible via .extractor.backbone[7][-1])
        - ONNX / TorchScript export
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.extractor  = ResNetFeatureExtractor(pretrained=pretrained, freeze=freeze_backbone)
        self.classifier = LinearClassifier(
            input_dim=self.extractor.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            logits [B, num_classes]
        """
        features = self.extractor(x)
        return self.classifier(features)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class and fake probability.

        Returns:
            (pred_labels [B], fake_probs [B])
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
        return preds, probs[:, 1]

    def load_classifier_weights(self, checkpoint_path: str, device: torch.device | None = None) -> None:
        """Load classifier head weights from a training checkpoint."""
        device = device or torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["classifier_state"] if "classifier_state" in ckpt else ckpt
        self.classifier.load_state_dict(state)
        print(f"Loaded classifier weights from {checkpoint_path} "
              f"(epoch={ckpt.get('epoch', '?')}, acc={ckpt.get('accuracy', '?'):.4f})")
