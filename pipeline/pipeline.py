"""
Base Inference Pipeline
=======================
End-to-end single-image and batch inference pipeline.
Accepts a raw image path or numpy array, returns a prediction dict.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.resnet_feature_extractor import ResNetFeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from training.train_classifier import LinearClassifier
from preprocessing.dataset_loader import IMAGENET_MEAN, IMAGENET_STD

BACKBONE_DIM = {"resnet": 2048, "clip": 512, "dinov2": 768}

_TRANSFORM = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DeepfakeDetectionPipeline:
    """
    Ready-to-use inference pipeline.

    Usage
    -----
        pipeline = DeepfakeDetectionPipeline(
            backbone="resnet",
            checkpoint_path="checkpoints/cross/celebdf_resnet.pt",
        )
        result = pipeline.predict("path/to/face.jpg")
        # {"label": "fake", "confidence": 0.87, "fake_prob": 0.87}
    """

    def __init__(
        self,
        backbone: str = "resnet",
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
    ):
        self.backbone = backbone.lower()
        self.device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build extractor
        if self.backbone == "resnet":
            self.extractor = ResNetFeatureExtractor(pretrained=True, freeze=True)
        elif self.backbone == "clip":
            self.extractor = CLIPFeatureExtractor(freeze=True)
        elif self.backbone == "dinov2":
            self.extractor = DINOv2FeatureExtractor(freeze=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}. Choose: resnet | clip | dinov2")

        self.extractor = self.extractor.to(self.device)
        self.extractor.eval()

        # Build classifier
        input_dim = BACKBONE_DIM[self.backbone]
        self.classifier = LinearClassifier(input_dim).to(self.device)

        if checkpoint_path:
            self.load(checkpoint_path)

        self.classifier.eval()

    def load(self, checkpoint_path: str) -> None:
        """Load classifier weights from checkpoint."""
        ckpt  = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["classifier_state"] if "classifier_state" in ckpt else ckpt
        self.classifier.load_state_dict(state)

    def _preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Convert image to model input tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        return _TRANSFORM(image).unsqueeze(0).to(self.device)

    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> dict:
        """
        Run inference on a single image.

        Returns:
            {
                "label":      "real" | "fake",
                "confidence": float  (probability of predicted class),
                "fake_prob":  float  (probability of fake class),
                "real_prob":  float  (probability of real class),
            }
        """
        tensor = self._preprocess(image)
        with torch.no_grad():
            features = self.extractor(tensor)
            logits   = self.classifier(features)
            probs    = torch.softmax(logits, dim=1)[0]

        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label     = "fake" if fake_prob >= 0.5 else "real"
        confidence = fake_prob if label == "fake" else real_prob

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(real_prob, 4),
        }

    def predict_batch(
        self,
        images: list[Union[str, Path, np.ndarray, Image.Image]],
        batch_size: int = 32,
    ) -> list[dict]:
        """Run predict() over a list of images in batches."""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.cat([self._preprocess(img) for img in batch], dim=0)
            with torch.no_grad():
                features = self.extractor(tensors)
                logits   = self.classifier(features)
                probs    = torch.softmax(logits, dim=1)
            for j in range(len(batch)):
                fp = float(probs[j, 1])
                rp = float(probs[j, 0])
                lbl = "fake" if fp >= 0.5 else "real"
                results.append({
                    "label":      lbl,
                    "confidence": round(fp if lbl == "fake" else rp, 4),
                    "fake_prob":  round(fp, 4),
                    "real_prob":  round(rp, 4),
                })
        return results
