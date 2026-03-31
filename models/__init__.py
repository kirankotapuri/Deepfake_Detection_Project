"""Model feature extractors for deepfake detection."""

from models.resnet_feature_extractor import ResNetFeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor

__all__ = [
    "ResNetFeatureExtractor",
    "CLIPFeatureExtractor",
    "DINOv2FeatureExtractor",
]
