"""
Explainability: GradCAM (ResNet50) + t-SNE (all 3 backbones)

Usage:
    python run_explainability.py

Outputs:
    results/gradcam_images/gradcam_real_<n>.png
    results/gradcam_images/gradcam_fake_<n>.png
    results/tsne_ResNet50.png
    results/tsne_CLIP.png
    results/tsne_DINOv2.png
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from explainability.gradcam import gradcam_resnet, save_gradcam
from explainability.tsne_visualization import tsne_visualize, plot_tsne
from models.resnet_feature_extractor import ResNetFeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from training.train_classifier import LinearClassifier

CKPT_DIR  = Path("checkpoints/cross")
RESULTS   = Path("results")
GRADCAM_DIR = RESULTS / "gradcam_images"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

TEST_DIR  = Path("dataset/CelebDF_quarter/test")
REAL_DIR  = TEST_DIR / "real"
FAKE_DIR  = TEST_DIR / "fake"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BACKBONE_CONFIGS = {
    "ResNet50": {
        "extractor_cls": ResNetFeatureExtractor,
        "extractor_kwargs": {},
        "input_dim": 2048,
        "ckpt": CKPT_DIR / "celebdf_resnet.pt",
    },
    "CLIP": {
        "extractor_cls": CLIPFeatureExtractor,
        "extractor_kwargs": {},
        "input_dim": 512,
        "ckpt": CKPT_DIR / "celebdf_clip.pt",
    },
    "DINOv2": {
        "extractor_cls": DINOv2FeatureExtractor,
        "extractor_kwargs": {},
        "input_dim": 768,
        "ckpt": CKPT_DIR / "celebdf_dinov2.pt",
    },
}


# ---------------------------------------------------------------------------
# Combined model for GradCAM
# ---------------------------------------------------------------------------
class FullResNetModel(nn.Module):
    """Wraps frozen ResNet extractor + MLP classifier into single forward pass."""

    def __init__(self, extractor: ResNetFeatureExtractor, classifier: LinearClassifier):
        super().__init__()
        self.extractor = extractor
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(x)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Helper: extract features from a folder of images
# ---------------------------------------------------------------------------
def extract_features(extractor: nn.Module, img_paths: list[Path], device: torch.device) -> np.ndarray:
    extractor.eval()
    feats = []
    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            feat = extractor(tensor).cpu().numpy()
            feats.append(feat)
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# 1) GradCAM on ResNet50 (CelebDF checkpoint)
# ---------------------------------------------------------------------------
def run_gradcam():
    print("\n=== GradCAM (ResNet50, trained on CelebDF) ===")
    cfg = BACKBONE_CONFIGS["ResNet50"]
    extractor = cfg["extractor_cls"](**cfg["extractor_kwargs"]).to(DEVICE)
    classifier = LinearClassifier(cfg["input_dim"]).to(DEVICE)
    ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
    classifier.load_state_dict(ckpt["classifier_state"])

    full_model = FullResNetModel(extractor, classifier).to(DEVICE)
    full_model.eval()

    # Target layer: last Bottleneck block of resnet layer4
    # backbone = Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
    target_layer = extractor.backbone[7][-1]

    real_images = sorted(REAL_DIR.glob("*.jpg"))
    fake_images = sorted(FAKE_DIR.glob("*.jpg"))
    random.seed(42)
    sample_real = random.sample(real_images, min(5, len(real_images)))
    sample_fake = random.sample(fake_images, min(5, len(fake_images)))

    for i, (img_path, label) in enumerate(
        [(p, "real") for p in sample_real] + [(p, "fake") for p in sample_fake]
    ):
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        heatmap = gradcam_resnet(full_model, target_layer, tensor)
        out_path = GRADCAM_DIR / f"gradcam_{label}_{i}.png"
        save_gradcam(str(img_path), heatmap, str(out_path))
        print(f"  Saved: {out_path.name}")

    print(f"GradCAM done. {len(sample_real)+len(sample_fake)} images saved to {GRADCAM_DIR}")


# ---------------------------------------------------------------------------
# 2) t-SNE for all 3 backbones
# ---------------------------------------------------------------------------
def run_tsne():
    print("\n=== t-SNE Visualization ===")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    fake_paths = sorted(FAKE_DIR.glob("*.jpg"))
    # Cap to 200 per class for speed
    random.seed(42)
    real_sample = random.sample(real_paths, min(200, len(real_paths)))
    fake_sample = random.sample(fake_paths, min(200, len(fake_paths)))

    all_paths = real_sample + fake_sample
    labels = np.array([0] * len(real_sample) + [1] * len(fake_sample))

    for backbone_name, cfg in BACKBONE_CONFIGS.items():
        print(f"\n  [{backbone_name}] Extracting features for {len(all_paths)} images...")
        extractor = cfg["extractor_cls"](**cfg["extractor_kwargs"]).to(DEVICE)
        extractor.eval()

        features = extract_features(extractor, all_paths, DEVICE)
        print(f"  Features shape: {features.shape}")

        print(f"  Running t-SNE...")
        embedding = tsne_visualize(features, labels, perplexity=30, n_iter=500, random_state=42)

        save_path = str(RESULTS / f"tsne_{backbone_name}.png")
        plot_tsne(
            embedding,
            labels,
            save_path,
            title=f"t-SNE: Real vs Fake ({backbone_name}, trained on CelebDF)",
        )
        print(f"  Saved: results/tsne_{backbone_name}.png")

    print("\nt-SNE done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_gradcam()
    run_tsne()
    print("\n=== Explainability complete ===")
    print(f"GradCAM images: {GRADCAM_DIR}")
    print(f"t-SNE plots:    {RESULTS}/tsne_*.png")
