"""
Cross-Dataset Generalization Pipeline

Trains one linear classifier per (training-dataset × backbone) pair and then
evaluates it on every other dataset — measuring how well features learned on
one distribution transfer to unseen distributions.

Two training scenarios × three backbones = 6 trained models:

    Scenario A (CelebDF training):
        Train  → dataset/CelebDF_quarter/train
        Test   → dataset/DFDC_quarter/test
        Test   → dataset/UADFV_quarter/test

    Scenario B (UADFV training):
        Train  → dataset/UADFV_quarter/train
        Test   → dataset/CelebDF_quarter/test
        Test   → dataset/DFDC_quarter/test

Outputs
-------
    checkpoints/cross/celebdf_resnet.pt
    checkpoints/cross/celebdf_clip.pt
    checkpoints/cross/celebdf_dinov2.pt
    checkpoints/cross/uadfv_resnet.pt
    checkpoints/cross/uadfv_clip.pt
    checkpoints/cross/uadfv_dinov2.pt

    results/cross_dataset_results.csv
    results/cross_dataset_table.txt    (human-readable comparison table)
    results/cm_<TrainDS>_<TestDS>_<backbone>.png   (one per combination)

Usage
-----
    # Run from project root:
    python -m pipeline.cross_dataset_pipeline

    # Only specified backbones:
    python -m pipeline.cross_dataset_pipeline --backbones resnet clip

    # Force re-train even if checkpoints exist:
    python -m pipeline.cross_dataset_pipeline --force-retrain
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.evaluate_model import evaluate, plot_confusion_matrix
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from models.resnet_feature_extractor import ResNetFeatureExtractor
from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader
from training.train_classifier import LinearClassifier, train_classifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_NAMES: list[str] = ["resnet", "clip", "dinov2"]

# Map backbone name → display label
BACKBONE_LABELS: dict[str, str] = {
    "resnet": "ResNet50",
    "clip":   "CLIP",
    "dinov2": "DINOv2",
}

# Training scenarios: training-dataset → test-datasets
TRAINING_SCENARIOS: dict[str, dict] = {
    "CelebDF": {
        "train_dir": "dataset/CelebDF_quarter/train",
        "val_dir":   "dataset/CelebDF_quarter/val",
        "test_datasets": {
            "DFDC":  "dataset/DFDC_quarter/test",
            "UADFV": "dataset/UADFV_quarter/test",
        },
    },
    "UADFV": {
        "train_dir": "dataset/UADFV_quarter/train",
        "val_dir":   "dataset/UADFV_quarter/val",
        "test_datasets": {
            "CelebDF": "dataset/CelebDF_quarter/test",
            "DFDC":    "dataset/DFDC_quarter/test",
        },
    },
}


# ---------------------------------------------------------------------------
# Feature-extractor factory
# ---------------------------------------------------------------------------

def get_feature_extractor(backbone: str, device: torch.device) -> torch.nn.Module:
    """Instantiate and return a frozen feature extractor on *device*."""
    if backbone == "resnet":
        return ResNetFeatureExtractor(pretrained=True, freeze=True).to(device)
    if backbone == "clip":
        return CLIPFeatureExtractor(freeze=True).to(device)
    if backbone == "dinov2":
        return DINOv2FeatureExtractor(freeze=True).to(device)
    raise ValueError(f"Unknown backbone: '{backbone}'")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_on_dataset(
    extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """
    Run inference on a test directory (expects real/ and fake/ subdirs).

    Returns a metrics dict from evaluation.evaluate_model.evaluate().
    """
    dataset = DeepfakeDataset(data_dir, use_subdirs=True, augment=False)
    if len(dataset) == 0:
        print(f"    WARNING: no samples found in {data_dir}")
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "auc": 0.0, "confusion_matrix": None,
        }

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
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    return evaluate(y_true, y_pred, y_prob)


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _format_table(df: pd.DataFrame) -> str:
    """Return a human-readable text table of results."""
    col_w = {"Train Dataset": 14, "Test Dataset": 12, "Backbone": 10,
             "Accuracy": 10, "Precision": 10, "Recall": 10, "F1": 10, "AUC": 10}
    sep = "=" * sum(col_w.values())
    lines = [sep, "CROSS-DATASET GENERALIZATION RESULTS", sep]
    header = (
        f"{'Train Dataset':<{col_w['Train Dataset']}}"
        f"{'Test Dataset':<{col_w['Test Dataset']}}"
        f"{'Backbone':<{col_w['Backbone']}}"
        f"{'Accuracy':>{col_w['Accuracy']}}"
        f"{'Precision':>{col_w['Precision']}}"
        f"{'Recall':>{col_w['Recall']}}"
        f"{'F1':>{col_w['F1']}}"
        f"{'AUC':>{col_w['AUC']}}"
    )
    lines.append(header)
    lines.append("-" * sum(col_w.values()))
    for _, row in df.iterrows():
        lines.append(
            f"{row['Train Dataset']:<{col_w['Train Dataset']}}"
            f"{row['Test Dataset']:<{col_w['Test Dataset']}}"
            f"{row['Backbone']:<{col_w['Backbone']}}"
            f"{row['Accuracy']:>{col_w['Accuracy']}.4f}"
            f"{row['Precision']:>{col_w['Precision']}.4f}"
            f"{row['Recall']:>{col_w['Recall']}.4f}"
            f"{row['F1']:>{col_w['F1']}.4f}"
            f"{row['AUC']:>{col_w['AUC']}.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cross_dataset_evaluation(
    backbones: list[str] | None = None,
    num_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 5,
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints/cross",
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Run the full cross-dataset generalization experiment.

    Parameters
    ----------
    backbones : list of backbone names (default: all three)
    num_epochs : max epochs per training run
    batch_size : batch size for training and evaluation
    lr : Adam learning rate
    patience : early-stopping patience (epochs)
    results_dir : directory for CSV / table / confusion matrices
    checkpoints_dir : directory for trained classifier checkpoints
    force_retrain : if True, ignore existing checkpoints and retrain

    Returns
    -------
    DataFrame with one row per (train_dataset, test_dataset, backbone).
    """
    if backbones is None:
        backbones = BACKBONE_NAMES

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load any records already saved from a previous (partial) run ────
    csv_path = os.path.join(results_dir, "cross_dataset_results.csv")
    if Path(csv_path).exists() and not force_retrain:
        existing_df = pd.read_csv(csv_path)
        records: list[dict] = existing_df.to_dict("records")
        print(f"Resuming: loaded {len(records)} existing result(s) from {csv_path}")
    else:
        records: list[dict] = []

    for train_ds, scenario in TRAINING_SCENARIOS.items():
        train_dir = scenario["train_dir"]
        val_dir   = scenario["val_dir"]
        test_datasets = scenario["test_datasets"]

        if not Path(train_dir).exists():
            print(
                f"[SKIP] Training scenario '{train_ds}': "
                f"train_dir not found → {train_dir}\n"
                f"       Run: python -m dataset.prepare_all_datasets"
            )
            continue

        for backbone in backbones:
            label = BACKBONE_LABELS.get(backbone, backbone.upper())
            ckpt_path = os.path.join(
                checkpoints_dir, f"{train_ds.lower()}_{backbone}.pt"
            )

            print(f"\n{'─'*60}")
            print(f"[Train] {train_ds}  ×  {label}")
            print(f"{'─'*60}")

            # ── Feature extractor ────────────────────────────────────────
            extractor = get_feature_extractor(backbone, device)
            classifier = LinearClassifier(
                extractor.feature_dim, num_classes=2
            ).to(device)

            # ── Training or checkpoint loading ───────────────────────────
            if Path(ckpt_path).exists() and not force_retrain:
                print(f"  Loading checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                classifier.load_state_dict(ckpt["classifier_state"])
            else:
                train_dataset = DeepfakeDataset(
                    train_dir, use_subdirs=True, augment=True
                )
                if len(train_dataset) == 0:
                    print(f"  SKIP: no samples in {train_dir}")
                    continue

                print(f"  Train samples : {len(train_dataset)}")
                train_loader = get_dataloader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    balance_classes=True,
                )

                val_loader = None
                if Path(val_dir).exists():
                    val_dataset = DeepfakeDataset(
                        val_dir, use_subdirs=True, augment=False
                    )
                    if len(val_dataset) > 0:
                        print(f"  Val samples   : {len(val_dataset)}")
                        val_loader = get_dataloader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            balance_classes=False,
                        )

                train_classifier(
                    train_loader=train_loader,
                    feature_extractor=extractor,
                    classifier=classifier,
                    device=device,
                    num_epochs=num_epochs,
                    lr=lr,
                    save_path=ckpt_path,
                    val_loader=val_loader,
                    patience=patience,
                )

                # Reload best checkpoint saved during training
                if Path(ckpt_path).exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                    classifier.load_state_dict(ckpt["classifier_state"])

            extractor.eval()
            classifier.eval()

            # ── Evaluate on every test dataset ───────────────────────────
            for test_ds, test_dir in test_datasets.items():
                if not Path(test_dir).exists():
                    print(
                        f"  [SKIP eval] '{test_ds}': {test_dir} not found. "
                        f"Run prepare_all_datasets.py"
                    )
                    continue

                # Skip if this (train_ds, test_ds, backbone) already recorded
                already_done = any(
                    r.get("Train Dataset") == train_ds
                    and r.get("Test Dataset") == test_ds
                    and r.get("Backbone") == label
                    for r in records
                )
                if already_done and not force_retrain:
                    print(f"  [SKIP — already in CSV] {train_ds} × {label} → {test_ds}")
                    continue

                print(f"  Evaluating → {test_ds} ({test_dir}) …")
                metrics = evaluate_on_dataset(
                    extractor, classifier, test_dir, device, batch_size
                )

                record = {
                    "Train Dataset": train_ds,
                    "Test Dataset":  test_ds,
                    "Backbone":      label,
                    "Accuracy":      round(metrics["accuracy"],  4),
                    "Precision":     round(metrics["precision"], 4),
                    "Recall":        round(metrics["recall"],    4),
                    "F1":            round(metrics["f1"],        4),
                    "AUC":           round(metrics["auc"],       4),
                }
                records.append(record)

                # ── Save incrementally after every result ────────────────
                pd.DataFrame(records).to_csv(csv_path, index=False)

                print(
                    f"    Acc={record['Accuracy']:.4f}  "
                    f"P={record['Precision']:.4f}  "
                    f"R={record['Recall']:.4f}  "
                    f"F1={record['F1']:.4f}  "
                    f"AUC={record['AUC']:.4f}"
                )

                # Confusion matrix
                if metrics.get("confusion_matrix") is not None:
                    cm_file = os.path.join(
                        results_dir,
                        f"cm_{train_ds}_{test_ds}_{backbone}.png",
                    )
                    plot_confusion_matrix(metrics["confusion_matrix"], cm_file)

    # ── Compile results ──────────────────────────────────────────────────
    if not records:
        print(
            "\nNo results generated. "
            "Ensure dataset preparation steps have been completed:\n"
            "  python -m dataset.standardize_datasets\n"
            "  python -m dataset.prepare_all_datasets"
        )
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # CSV
    csv_path = os.path.join(results_dir, "cross_dataset_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved  →  {csv_path}")

    # Text table
    table_str = _format_table(df)
    table_path = os.path.join(results_dir, "cross_dataset_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"Table saved →  {table_path}")
    print("\n" + table_str)

    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-dataset generalization experiment."
    )
    parser.add_argument(
        "--backbones", nargs="+",
        choices=BACKBONE_NAMES,
        default=BACKBONE_NAMES,
        help="Which backbone models to evaluate (default: all three)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Maximum training epochs per model (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience in epochs (default: 5)"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints/cross",
        help="Directory for trained classifier checkpoints (default: checkpoints/cross/)"
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Ignore existing checkpoints and retrain from scratch"
    )
    args = parser.parse_args()

    run_cross_dataset_evaluation(
        backbones=args.backbones,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
