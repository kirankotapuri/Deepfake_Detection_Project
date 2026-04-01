"""
Diffusion Detection Experiment  (Steps 4 – 8)
==============================================
Experiment 1  (Cross-domain):   Train → CelebDF    Test → StableDiffusion
Experiment 2  (Reverse):        Train → StableDiff  Test → CelebDF

For each experiment × 3 backbones (ResNet50, CLIP, DINOv2):
  ─ Confusion matrix
  ─ ROC curve (per backbone; combined multi-curve figure)
  ─ AUC + classification report
  ─ GradCAM heatmaps (ResNet50)
  ─ t-SNE feature plots

All outputs → results/diffusion_results/
Final CSV   → results/diffusion_results/diffusion_results.csv

Usage
-----
    # Prepare data first (once):
    python dataset/prepare_diffusion_dataset.py

    # Run experiment:
    python run_diffusion_experiment.py

    # Skip Experiment 2 (no retraining):
    python run_diffusion_experiment.py --skip-exp2
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from evaluation.evaluate_model import (
    evaluate,
    get_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_curves_multi,
)
from explainability.gradcam import gradcam_resnet, save_gradcam
from explainability.tsne_visualization import plot_tsne, tsne_visualize
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from models.resnet_feature_extractor import ResNetFeatureExtractor
from pipeline.cross_dataset_pipeline import get_feature_extractor
from preprocessing.dataset_loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DeepfakeDataset,
    get_dataloader,
)
from training.train_classifier import LinearClassifier, train_classifier

# ── Paths ──────────────────────────────────────────────────────────────────
DIFFUSION_QUARTER = Path("dataset/diffusion_quarter")
CELEBDF_QUARTER   = Path("dataset/CelebDF_quarter")
CROSS_CKPT_DIR    = Path("checkpoints/cross")
DIFF_CKPT_DIR     = Path("checkpoints/diffusion")
RESULTS_DIR       = Path("results/diffusion_results")
EXPLAIN_DIR       = RESULTS_DIR / "explainability"
GRADCAM_DIR       = EXPLAIN_DIR / "gradcam"
TSNE_DIR          = EXPLAIN_DIR / "tsne"

for d in (RESULTS_DIR, DIFF_CKPT_DIR, GRADCAM_DIR, TSNE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "experiment_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Backbone configs ───────────────────────────────────────────────────────
BACKBONE_CONFIGS = {
    "ResNet50": {"key": "resnet", "input_dim": 2048},
    "CLIP":     {"key": "clip",   "input_dim": 512},
    "DINOv2":   {"key": "dinov2", "input_dim": 768},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ===========================================================================
# Shared inference helper
# ===========================================================================

def run_inference(
    extractor: nn.Module,
    classifier: nn.Module,
    data_dir: str | Path,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run forward pass on data_dir (expects real/ + fake/ subdirs).

    Returns:
        (y_true, y_pred, y_prob)  — numpy arrays
    """
    dataset = DeepfakeDataset(str(data_dir), use_subdirs=True, augment=False)
    if len(dataset) == 0:
        log.warning(f"No samples found in {data_dir}")
        return np.array([]), np.array([]), np.array([])

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_labels, all_preds, all_probs = [], [], []

    extractor.eval()
    classifier.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = extractor(images)
            logits   = classifier(features)
            probs    = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def load_classifier(backbone_name: str, ckpt_path: Path, device: torch.device):
    """Load extractor + frozen classifier from checkpoint."""
    cfg = BACKBONE_CONFIGS[backbone_name]
    extractor  = get_feature_extractor(cfg["key"], device)
    classifier = LinearClassifier(cfg["input_dim"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(ckpt["classifier_state"])
    extractor.eval()
    classifier.eval()
    return extractor, classifier


def train_and_save(backbone_name: str, device: torch.device, save_path: Path):
    """Train a new classifier on diffusion_quarter/train, return (extractor, classifier)."""
    cfg = BACKBONE_CONFIGS[backbone_name]
    extractor  = get_feature_extractor(cfg["key"], device)
    classifier = LinearClassifier(cfg["input_dim"]).to(device)

    train_ds = DeepfakeDataset(str(DIFFUSION_QUARTER / "train"), use_subdirs=True, augment=True)
    val_ds   = DeepfakeDataset(str(DIFFUSION_QUARTER / "val"),   use_subdirs=True, augment=False)
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = get_dataloader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    log.info(f"  Training {backbone_name}: {len(train_ds)} train / {len(val_ds)} val")
    train_classifier(
        train_loader=train_loader,
        feature_extractor=extractor,
        classifier=classifier,
        device=device,
        num_epochs=20,
        lr=1e-4,
        save_path=str(save_path),
        val_loader=val_loader,
        patience=5,
    )
    # Reload best saved weights
    ckpt = torch.load(save_path, map_location=device)
    classifier.load_state_dict(ckpt["classifier_state"])
    extractor.eval()
    classifier.eval()
    return extractor, classifier


# ===========================================================================
# Step 6 — Evaluation outputs (CM + ROC + report)
# ===========================================================================

def generate_eval_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    backbone: str,
    experiment: str,
    prefix: str,
) -> dict:
    """Save CM, ROC curve, classification report. Return metrics dict."""
    metrics = evaluate(y_true, y_pred, y_prob)

    # Confusion matrix
    cm_path = str(RESULTS_DIR / f"cm_{prefix}_{backbone.lower().replace('50','')}.png")
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
    log.info(f"    CM saved: {Path(cm_path).name}")

    # ROC curve (single backbone)
    if len(np.unique(y_true)) > 1:
        roc_path = str(RESULTS_DIR / f"roc_{prefix}_{backbone.lower().replace('50','')}.png")
        plot_roc_curve(
            y_true, y_prob, roc_path,
            title=f"ROC — {experiment} ({backbone})",
            label=backbone,
        )
        log.info(f"    ROC saved: {Path(roc_path).name}")

    # Classification report
    report_str = get_classification_report(y_true, y_pred)
    report_path = RESULTS_DIR / f"report_{prefix}_{backbone.lower().replace('50','')}.txt"
    report_path.write_text(
        f"Experiment : {experiment}\nBackbone   : {backbone}\n\n{report_str}",
        encoding="utf-8",
    )
    log.info(f"    Report saved: {report_path.name}")

    return {
        "experiment": experiment,
        "backbone": backbone,
        "accuracy":  round(metrics["accuracy"],  4),
        "precision": round(metrics["precision"], 4),
        "recall":    round(metrics["recall"],    4),
        "f1":        round(metrics["f1"],        4),
        "auc":       round(metrics["auc"],       4),
        "y_true": y_true,
        "y_prob": y_prob,
    }


# ===========================================================================
# Step 7 — Explainability
# ===========================================================================

class _FullResNetModel(nn.Module):
    def __init__(self, extractor, classifier):
        super().__init__()
        self.extractor  = extractor
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.extractor(x))


def run_gradcam(extractor, classifier, test_dir: Path, prefix: str, n: int = 5):
    """Generate GradCAM heatmaps for n real + n fake images."""
    log.info(f"  GradCAM [{prefix}] ...")
    full_model  = _FullResNetModel(extractor, classifier).to(DEVICE)
    target_layer = extractor.backbone[7][-1]   # ResNet layer4 last block

    for label_str, label_dir in [("real", test_dir / "real"), ("fake", test_dir / "fake")]:
        imgs = sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.png"))
        random.seed(SEED)
        sample = random.sample(imgs, min(n, len(imgs)))
        for i, img_path in enumerate(sample):
            tensor = EVAL_TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            heatmap = gradcam_resnet(full_model, target_layer, tensor)
            out = GRADCAM_DIR / f"gradcam_{prefix}_{label_str}_{i}.png"
            save_gradcam(str(img_path), heatmap, str(out))
    log.info(f"    Saved {2*n} GradCAM images → {GRADCAM_DIR}")


def run_tsne(extractor, backbone_name: str, test_dir: Path, prefix: str, max_per_class: int = 150):
    """Generate t-SNE scatter for real vs fake features."""
    log.info(f"  t-SNE [{backbone_name} / {prefix}] ...")

    real_imgs = sorted((test_dir / "real").glob("*.jpg")) + sorted((test_dir / "real").glob("*.png"))
    fake_imgs = sorted((test_dir / "fake").glob("*.jpg")) + sorted((test_dir / "fake").glob("*.png"))
    random.seed(SEED)
    real_sample = random.sample(real_imgs, min(max_per_class, len(real_imgs)))
    fake_sample = random.sample(fake_imgs, min(max_per_class, len(fake_imgs)))

    all_paths = real_sample + fake_sample
    labels    = np.array([0] * len(real_sample) + [1] * len(fake_sample))

    extractor.eval()
    feats = []
    with torch.no_grad():
        for path in all_paths:
            tensor = EVAL_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            feat = extractor(tensor).cpu().numpy()
            feats.append(feat)
    features = np.concatenate(feats, axis=0)

    embedding = tsne_visualize(features, labels, perplexity=30, n_iter=500, random_state=SEED)
    save_path = str(TSNE_DIR / f"tsne_{prefix}_{backbone_name.lower().replace('50','')}.png")
    plot_tsne(
        embedding, labels, save_path,
        title=f"t-SNE: Real vs Diffusion Fake ({backbone_name}, {prefix})",
    )
    log.info(f"    Saved: {Path(save_path).name}")


# ===========================================================================
# Experiment 1 — CelebDF → StableDiffusion  (cross-domain)
# ===========================================================================

def experiment1(records: list[dict]) -> list[tuple]:
    """
    Load CelebDF-trained checkpoints, evaluate on diffusion_quarter/test.
    Returns list of (y_true, y_prob, label) for combined ROC.
    """
    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT 1 — Train: CelebDF  |  Test: StableDiffusion")
    log.info("=" * 70)

    test_dir  = DIFFUSION_QUARTER / "test"
    roc_curves = []

    for backbone in ("ResNet50", "CLIP", "DINOv2"):
        ckpt = CROSS_CKPT_DIR / f"celebdf_{BACKBONE_CONFIGS[backbone]['key']}.pt"
        if not ckpt.exists():
            log.warning(f"  Checkpoint not found: {ckpt} — skipping {backbone}")
            continue

        log.info(f"\n  [{backbone}]  Loading {ckpt.name} ...")
        extractor, classifier = load_classifier(backbone, ckpt, DEVICE)

        y_true, y_pred, y_prob = run_inference(extractor, classifier, test_dir, DEVICE)
        if len(y_true) == 0:
            continue

        result = generate_eval_outputs(
            y_true, y_pred, y_prob,
            backbone=backbone,
            experiment="CelebDF→StableDiffusion",
            prefix="exp1_celebdf_sd",
        )
        log.info(
            f"    Acc={result['accuracy']:.4f}  "
            f"P={result['precision']:.4f}  R={result['recall']:.4f}  "
            f"F1={result['f1']:.4f}  AUC={result['auc']:.4f}"
        )

        records.append({
            "Experiment":   "CelebDF→StableDiffusion",
            "Train Dataset": "CelebDF",
            "Test Dataset":  "StableDiffusion",
            "Backbone":      backbone,
            "Accuracy":      result["accuracy"],
            "Precision":     result["precision"],
            "Recall":        result["recall"],
            "F1":            result["f1"],
            "AUC":           result["auc"],
        })

        roc_curves.append((y_true, y_prob, backbone))

        # GradCAM only for ResNet50
        if backbone == "ResNet50":
            run_gradcam(extractor, classifier, test_dir, prefix="exp1")

        # t-SNE for all
        run_tsne(extractor, backbone, test_dir, prefix="exp1")

    # Combined ROC
    if roc_curves:
        plot_roc_curves_multi(
            roc_curves,
            str(RESULTS_DIR / "roc_exp1_celebdf_sd_all_backbones.png"),
            title="ROC — CelebDF trained → StableDiffusion test (all backbones)",
        )
        log.info(f"\n  Combined ROC saved: roc_exp1_celebdf_sd_all_backbones.png")

    return roc_curves


# ===========================================================================
# Experiment 2 — StableDiffusion → CelebDF  (reverse)
# ===========================================================================

def experiment2(records: list[dict]) -> list[tuple]:
    """
    Train models on diffusion_quarter/train, evaluate on CelebDF_quarter/test.
    Returns list of (y_true, y_prob, label) for combined ROC.
    """
    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT 2 — Train: StableDiffusion  |  Test: CelebDF")
    log.info("=" * 70)

    test_dir   = CELEBDF_QUARTER / "test"
    roc_curves = []

    for backbone in ("ResNet50", "CLIP", "DINOv2"):
        ckpt = DIFF_CKPT_DIR / f"diffusion_{BACKBONE_CONFIGS[backbone]['key']}.pt"

        if ckpt.exists():
            log.info(f"\n  [{backbone}]  Loading existing checkpoint {ckpt.name} ...")
            extractor, classifier = load_classifier(backbone, ckpt, DEVICE)
        else:
            log.info(f"\n  [{backbone}]  No checkpoint found — training from scratch ...")
            extractor, classifier = train_and_save(backbone, DEVICE, ckpt)

        y_true, y_pred, y_prob = run_inference(extractor, classifier, test_dir, DEVICE)
        if len(y_true) == 0:
            continue

        result = generate_eval_outputs(
            y_true, y_pred, y_prob,
            backbone=backbone,
            experiment="StableDiffusion→CelebDF",
            prefix="exp2_sd_celebdf",
        )
        log.info(
            f"    Acc={result['accuracy']:.4f}  "
            f"P={result['precision']:.4f}  R={result['recall']:.4f}  "
            f"F1={result['f1']:.4f}  AUC={result['auc']:.4f}"
        )

        records.append({
            "Experiment":   "StableDiffusion→CelebDF",
            "Train Dataset": "StableDiffusion",
            "Test Dataset":  "CelebDF",
            "Backbone":      backbone,
            "Accuracy":      result["accuracy"],
            "Precision":     result["precision"],
            "Recall":        result["recall"],
            "F1":            result["f1"],
            "AUC":           result["auc"],
        })

        roc_curves.append((y_true, y_prob, backbone))

        # GradCAM only for ResNet50
        if backbone == "ResNet50":
            run_gradcam(extractor, classifier, test_dir, prefix="exp2")

        # t-SNE for all
        run_tsne(extractor, backbone, test_dir, prefix="exp2")

    # Combined ROC
    if roc_curves:
        plot_roc_curves_multi(
            roc_curves,
            str(RESULTS_DIR / "roc_exp2_sd_celebdf_all_backbones.png"),
            title="ROC — StableDiffusion trained → CelebDF test (all backbones)",
        )
        log.info(f"\n  Combined ROC saved: roc_exp2_sd_celebdf_all_backbones.png")

    return roc_curves


# ===========================================================================
# Step 8 — Save final CSV + print table
# ===========================================================================

def save_results(records: list[dict]):
    """Save CSV and print formatted table."""
    if not records:
        log.warning("No results to save.")
        return

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "diffusion_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\n  Results CSV saved → {csv_path}")

    # Formatted text table
    cols = ["Experiment", "Backbone", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    sep  = "=" * 90

    lines = [sep, "DIFFUSION DETECTION EXPERIMENT RESULTS", sep]
    for exp_name in df["Experiment"].unique():
        sub = df[df["Experiment"] == exp_name]
        lines.append(f"\n  {exp_name}")
        lines.append("  " + "-" * 70)
        header = f"  {'Backbone':<12}" + "".join(f"  {m:<11}" for m in cols[2:])
        lines.append(header)
        lines.append("  " + "-" * 70)
        for _, row in sub.iterrows():
            vals = "".join(f"  {row[m]:<11.4f}" for m in cols[2:])
            lines.append(f"  {row['Backbone']:<12}{vals}")

    lines.append(f"\n{sep}")
    table = "\n".join(lines)
    print("\n" + table)

    table_path = RESULTS_DIR / "diffusion_table.txt"
    table_path.write_text(table, encoding="utf-8")
    log.info(f"  Table saved → {table_path}")

    # Save experiment config for reproducibility
    config = {
        "seed": SEED,
        "device": str(DEVICE),
        "diffusion_quarter": str(DIFFUSION_QUARTER),
        "celebdf_quarter": str(CELEBDF_QUARTER),
        "backbones": list(BACKBONE_CONFIGS.keys()),
        "experiments": ["CelebDF→StableDiffusion", "StableDiffusion→CelebDF"],
    }
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)
    log.info(f"  Config saved → {RESULTS_DIR / 'experiment_config.json'}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion detection experiments")
    parser.add_argument("--skip-exp2", action="store_true",
                        help="Skip Experiment 2 (no training on SD data)")
    args = parser.parse_args()

    # Verify data is prepared
    if not DIFFUSION_QUARTER.exists():
        log.error(
            f"diffusion_quarter not found at {DIFFUSION_QUARTER}. "
            "Run: python dataset/prepare_diffusion_dataset.py"
        )
        raise SystemExit(1)

    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            d = DIFFUSION_QUARTER / split / cls
            if not d.exists() or not any(d.iterdir()):
                log.error(f"Missing or empty: {d}")
                raise SystemExit(1)

    log.info("=" * 70)
    log.info("DIFFUSION DETECTION EXPERIMENT")
    log.info(f"  Seed   : {SEED}")
    log.info(f"  Device : {DEVICE}")
    log.info(f"  Output : {RESULTS_DIR}")
    log.info("=" * 70)

    records: list[dict] = []

    # ── Experiment 1 ──────────────────────────────────────────────────────
    experiment1(records)

    # ── Experiment 2 (optional) ───────────────────────────────────────────
    if not args.skip_exp2:
        experiment2(records)
    else:
        log.info("\n  [Experiment 2 skipped via --skip-exp2]")

    # ── Save all results ──────────────────────────────────────────────────
    save_results(records)

    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info(f"  Output directory: {RESULTS_DIR}")
    log.info(f"  Files generated:")
    for f in sorted(RESULTS_DIR.rglob("*")):
        if f.is_file():
            log.info(f"    {f.relative_to(RESULTS_DIR)}")
    log.info("=" * 70)
