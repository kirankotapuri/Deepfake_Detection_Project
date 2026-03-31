"""
Inference Pipeline for Deepfake Detection

Runs predictions on new images/videos.
Outputs: fake probability, prediction label.
"""

import torch
from pathlib import Path
from PIL import Image

from torchvision import transforms
from preprocessing.dataset_loader import IMAGENET_MEAN, IMAGENET_STD


def load_model_for_inference(
    backbone: str,
    classifier_path: str,
    device: torch.device | None = None,
):
    """
    Load feature extractor and classifier.

    Returns:
        (extractor, classifier) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.resnet_feature_extractor import ResNetFeatureExtractor
    from models.clip_feature_extractor import CLIPFeatureExtractor
    from models.dinov2_feature_extractor import DINOv2FeatureExtractor
    from training.train_classifier import LinearClassifier

    if backbone == "resnet":
        extractor = ResNetFeatureExtractor(pretrained=True, freeze=True).to(device)
    elif backbone == "clip":
        extractor = CLIPFeatureExtractor(freeze=True).to(device)
    elif backbone == "dinov2":
        extractor = DINOv2FeatureExtractor(freeze=True).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    ckpt = torch.load(classifier_path, map_location=device)
    classifier = LinearClassifier(extractor.feature_dim, num_classes=2).to(device)
    classifier.load_state_dict(ckpt["classifier_state"])
    extractor.eval()
    classifier.eval()
    return extractor, classifier


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(
    image_path: str,
    extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    """
    Predict fake probability and label for single image.

    Args:
        image_path: Path to image.
        extractor: Feature extractor.
        classifier: Linear classifier.
        device: Device.

    Returns:
        (fake_probability, prediction_label) where label 0=real, 1=fake.
    """
    x = preprocess_image(image_path).to(device)
    with torch.no_grad():
        features = extractor(x)
        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)
    fake_prob = probs[0, 1].item()
    pred = 1 if fake_prob >= 0.5 else 0
    return fake_prob, pred
