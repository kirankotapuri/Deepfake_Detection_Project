"""
Diffusion Robustness Experiment  (Phase 5)

Tests whether models trained on GAN-based deepfakes can detect
diffusion-generated synthetic faces — directly measuring the "diffusion gap".

Expected result:
    Accuracy on diffusion images  <  accuracy on GAN datasets (DFDC / UADFV)
    → demonstrates that GAN-trained detectors fail on diffusion fakes.

Input dataset layout
--------------------
Option A — labelled (recommended):
    dataset/diffusion_faces/
        fake/   ← diffusion-generated faces (200–500 images)
        real/   ← real faces for contrast  (optional)

Option B — flat (unlabelled, all treated as fake):
    dataset/diffusion_faces/
        image1.jpg
        image2.png
        ...

Outputs
-------
    results/diffusion_results.csv
    results/diffusion_table.txt
    results/cm_diffusion_<backbone>.png    (only when real + fake both present)

Usage
-----
    # Basic — uses models trained on CelebDF
    python -m pipeline.diffusion_experiment \\
        --diffusion-dir dataset/diffusion_faces

    # Specify training dataset and backbones
    python -m pipeline.diffusion_experiment \\
        --diffusion-dir dataset/diffusion_faces \\
        --train-ds uadfv \\
        --backbones resnet dinov2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from evaluation.evaluate_model import evaluate, plot_confusion_matrix
from pipeline.cross_dataset_pipeline import get_feature_extractor
from preprocessing.dataset_loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DeepfakeDataset,
    get_dataloader,
)
from training.train_classifier import LinearClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Flat-fake dataset  (all images in a directory are labelled as fake=1)
# ---------------------------------------------------------------------------

class FlatFakeDataset(Dataset):
    """
    Loads every image in *root* and returns (tensor, 1) — label 1 = fake.
    Used when the diffusion image directory contains no real/ / fake/ subdirs.
    """

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.paths = [
            f for f in sorted(self.root.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 1  # label = fake


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_diffusion(
    extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    diffusion_dir: str,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, int]:
    """
    Run inference on diffusion images.

    Returns (metrics_dict, num_samples).
    metrics_dict has accuracy / precision / recall / f1 / auc / confusion_matrix.
    """
    ddir = Path(diffusion_dir)
    has_subdirs = any((ddir / cls).exists() for cls in ("real", "fake"))

    if has_subdirs:
        dataset = DeepfakeDataset(str(ddir), use_subdirs=True, augment=False)
    else:
        dataset = FlatFakeDataset(str(ddir))

    if len(dataset) == 0:
        raise ValueError(f"No images found in: {diffusion_dir}")

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = extractor(images)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy() if hasattr(labels, "numpy") else labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return evaluate(y_true, y_pred, y_prob), len(dataset)


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _format_diffusion_table(df: pd.DataFrame) -> str:
    sep = "=" * 80
    lines = [
        sep,
        "DIFFUSION ROBUSTNESS EXPERIMENT RESULTS",
        sep,
        f"{'Train DS':<12} {'Test DS':<12} {'Backbone':<10} "
        f"{'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}",
        "-" * 80,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Train Dataset']:<12} {row['Test Dataset']:<12} {row['Backbone']:<10} "
            f"{row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
            f"{row['Recall']:>10.4f} {row['F1']:>10.4f} {row['AUC']:>10.4f}"
        )
    lines += [
        sep,
        "NOTE: Lower accuracy on diffusion fakes vs GAN datasets",
        "      demonstrates the diffusion detectability gap.",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_diffusion_experiment(
    diffusion_dir: str,
    backbones: list[str] | None = None,
    train_ds: str = "celebdf",
    checkpoints_dir: str = "checkpoints/cross",
    results_dir: str = "results",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Run diffusion robustness experiment.

    Parameters
    ----------
    diffusion_dir : path to diffusion-generated face images
    backbones     : list of backbone names to evaluate (default: all three)
    train_ds      : which dataset the models were trained on  (celebdf | uadfv)
    checkpoints_dir : directory containing cross-evaluation checkpoints
    results_dir   : output directory
    batch_size    : inference batch size

    Returns
    -------
    DataFrame with one row per backbone.
    """
    if backbones is None:
        backbones = ["resnet", "clip", "dinov2"]

    backbone_labels = {"resnet": "ResNet50", "clip": "CLIP", "dinov2": "DINOv2"}

    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(diffusion_dir).exists():
        raise FileNotFoundError(f"Diffusion directory not found: {diffusion_dir}")

    records: list[dict] = []

    for backbone in backbones:
        ckpt_path = os.path.join(
            checkpoints_dir, f"{train_ds.lower()}_{backbone}.pt"
        )
        label = backbone_labels.get(backbone, backbone.upper())

        if not Path(ckpt_path).exists():
            print(
                f"[SKIP] {label}: checkpoint not found: {ckpt_path}\n"
                f"       Run cross-dataset training first:\n"
                f"       python -m pipeline.cross_dataset_pipeline"
            )
            continue

        print(f"\n[Diffusion] Loading {label} trained on {train_ds.upper()} …")
        extractor  = get_feature_extractor(backbone, device)
        classifier = LinearClassifier(extractor.feature_dim, num_classes=2).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt["classifier_state"])
        extractor.eval()
        classifier.eval()

        try:
            metrics, n_samples = _evaluate_diffusion(
                extractor, classifier, diffusion_dir, device, batch_size
            )
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            continue

        record = {
            "Train Dataset": train_ds.upper(),
            "Test Dataset":  "Diffusion",
            "Backbone":      label,
            "Accuracy":      round(metrics["accuracy"],  4),
            "Precision":     round(metrics["precision"], 4),
            "Recall":        round(metrics["recall"],    4),
            "F1":            round(metrics["f1"],        4),
            "AUC":           round(metrics["auc"],       4),
            "Num Samples":   n_samples,
        }
        records.append(record)

        print(
            f"  Acc={record['Accuracy']:.4f}  "
            f"P={record['Precision']:.4f}  "
            f"R={record['Recall']:.4f}  "
            f"F1={record['F1']:.4f}  "
            f"AUC={record['AUC']:.4f}  "
            f"(n={n_samples})"
        )

        if metrics.get("confusion_matrix") is not None:
            cm_path = os.path.join(results_dir, f"cm_diffusion_{backbone}.png")
            plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    if not records:
        print("\nNo diffusion results. "
              "Ensure checkpoints exist and diffusion_dir contains images.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = os.path.join(results_dir, "diffusion_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDiffusion CSV saved  →  {csv_path}")

    # Save table
    table_str = _format_diffusion_table(df)
    table_path = os.path.join(results_dir, "diffusion_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"Diffusion table saved →  {table_path}")
    print("\n" + table_str)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diffusion robustness experiment."
    )
    parser.add_argument(
        "--diffusion-dir", required=True,
        help="Directory containing diffusion-generated face images"
    )
    parser.add_argument(
        "--backbones", nargs="+",
        choices=["resnet", "clip", "dinov2"],
        default=["resnet", "clip", "dinov2"],
    )
    parser.add_argument(
        "--train-ds", default="celebdf",
        choices=["celebdf", "uadfv"],
        help="Which training dataset's models to load (default: celebdf)"
    )
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints/cross"
    )
    parser.add_argument(
        "--results-dir", default="results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32
    )
    args = parser.parse_args()

    run_diffusion_experiment(
        diffusion_dir=args.diffusion_dir,
        backbones=args.backbones,
        train_ds=args.train_ds,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
