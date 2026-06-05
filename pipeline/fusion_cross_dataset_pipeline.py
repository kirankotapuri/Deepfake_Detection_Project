"""
Standalone improved fusion experiment pipeline.

Supports two model variants:
  --model-type two_backbone   → ImprovedFusionClassifier  (ResNet + CLIP only)
  --model-type three_backbone → ImprovedFusionWithDINO    (all 3, constrained weights)

Writes results, checkpoints, confusion matrices, and comparison CSVs into
dedicated fusion-only directories so baseline results stay untouched.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force offline mode so HuggingFace uses cached weights instead of making
# network requests (avoids SSL cert errors on corporate networks).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

from evaluation.evaluate_model import evaluate, plot_confusion_matrix
from models.attention_fusion_classifier import (
    ImprovedFusionClassifier,
    ImprovedFusionWithDINO,
)
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from models.resnet_feature_extractor import ResNetFeatureExtractor
from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader
from training.train_fusion_classifier import train_fusion_classifier

TRAINING_SCENARIOS: dict[str, dict[str, object]] = {
    "CelebDF": {
        "train_dir": "dataset/CelebDF_quarter/train",
        "val_dir": "dataset/CelebDF_quarter/val",
        "test_datasets": {
            "DFDC": "dataset/DFDC_quarter/test",
            "UADFV": "dataset/UADFV_quarter/test",
        },
    },
    "UADFV": {
        "train_dir": "dataset/UADFV_quarter/train",
        "val_dir": "dataset/UADFV_quarter/val",
        "test_datasets": {
            "CelebDF": "dataset/CelebDF_quarter/test",
            "DFDC": "dataset/DFDC_quarter/test",
        },
    },
}


# -----------------------------------------------------------------------
# Frozen extractors
# -----------------------------------------------------------------------
def get_frozen_extractors(device: torch.device, use_three: bool = False) -> dict[str, torch.nn.Module]:
    """Load frozen backbones. Only loads DINOv2 when use_three=True."""
    extractors: dict[str, torch.nn.Module] = {
        "resnet": ResNetFeatureExtractor(pretrained=True, freeze=True).to(device),
        "clip": CLIPFeatureExtractor(freeze=True).to(device),
    }
    if use_three:
        extractors["dinov2"] = DINOv2FeatureExtractor(freeze=True).to(device)
    return extractors


# -----------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------
def _evaluate_fusion_on_dataset(
    extractors: dict[str, torch.nn.Module],
    fusion_classifier: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    batch_size: int = 32,
    use_three: bool = False,
) -> tuple[dict[str, object], dict[str, float]]:
    dataset = DeepfakeDataset(data_dir, use_subdirs=True, augment=False)
    if len(dataset) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "auc": 0.0, "confusion_matrix": None,
        }, {"resnet": 0.0, "clip": 0.0, "dinov2": 0.0}

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, y_prob = [], [], []
    weight_records: list[dict[str, float]] = []

    fusion_classifier.eval()
    for ext in extractors.values():
        ext.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            resnet_features = extractors["resnet"](images)
            clip_features = extractors["clip"](images)

            if use_three:
                dinov2_features = extractors["dinov2"](images)
                logits, w = fusion_classifier(
                    resnet_features, clip_features, dinov2_features, return_weights=True,
                )
                # w is a tensor [3] from softmax
                weight_records.append({
                    "resnet": float(w[0]),
                    "clip": float(w[1]),
                    "dinov2": float(w[2]),
                })
            else:
                logits, w_info = fusion_classifier(
                    resnet_features, clip_features, return_weights=True,
                )
                # w_info is a dict {"clip": float, "resnet": float}
                weight_records.append({
                    "resnet": w_info["resnet"],
                    "clip": w_info["clip"],
                    "dinov2": 0.0,
                })

            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    metrics = evaluate(np.array(y_true), np.array(y_pred), np.array(y_prob))

    # Average weights across all batches
    avg_w = {
        "resnet": np.mean([r["resnet"] for r in weight_records]),
        "clip": np.mean([r["clip"] for r in weight_records]),
        "dinov2": np.mean([r["dinov2"] for r in weight_records]),
    }
    return metrics, avg_w


# -----------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------
def _format_fusion_table(df: pd.DataFrame) -> str:
    widths = {"Train Dataset": 14, "Test Dataset": 12, "Accuracy": 10, "Precision": 10,
              "Recall": 10, "F1": 10, "AUC": 10, "ResNet W": 10, "CLIP W": 10, "DINO W": 10}
    sep = "=" * sum(widths.values())
    lines = [sep, "IMPROVED FUSION CROSS-DATASET RESULTS", sep]
    header = (f"{'Train Dataset':<{widths['Train Dataset']}}"
              f"{'Test Dataset':<{widths['Test Dataset']}}"
              f"{'Accuracy':>{widths['Accuracy']}}"
              f"{'Precision':>{widths['Precision']}}"
              f"{'Recall':>{widths['Recall']}}"
              f"{'F1':>{widths['F1']}}"
              f"{'AUC':>{widths['AUC']}}"
              f"{'ResNet W':>{widths['ResNet W']}}"
              f"{'CLIP W':>{widths['CLIP W']}}"
              f"{'DINO W':>{widths['DINO W']}}")
    lines += [header, "-" * sum(widths.values())]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Train Dataset']:<{widths['Train Dataset']}}"
            f"{row['Test Dataset']:<{widths['Test Dataset']}}"
            f"{row['Accuracy']:>{widths['Accuracy']}.4f}"
            f"{row['Precision']:>{widths['Precision']}.4f}"
            f"{row['Recall']:>{widths['Recall']}.4f}"
            f"{row['F1']:>{widths['F1']}.4f}"
            f"{row['AUC']:>{widths['AUC']}.4f}"
            f"{row['ResNet Weight']:>{widths['ResNet W']}.4f}"
            f"{row['CLIP Weight']:>{widths['CLIP W']}.4f}"
            f"{row['DINOv2 Weight']:>{widths['DINO W']}.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _build_comparison_table(fusion_df: pd.DataFrame, baseline_csv_path: str | Path) -> pd.DataFrame:
    baseline_path = Path(baseline_csv_path)
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline_df = pd.read_csv(baseline_path)
    baseline_best = (
        baseline_df.sort_values(["Train Dataset", "Test Dataset", "Accuracy"], ascending=[True, True, False])
        .groupby(["Train Dataset", "Test Dataset"], as_index=False).first()
    )
    comparison = fusion_df.merge(baseline_best, on=["Train Dataset", "Test Dataset"], suffixes=(" Fusion", " Baseline"))
    comparison = comparison.rename(columns={"Backbone": "Best Baseline Backbone"})
    comparison["Delta Accuracy"] = (comparison["Accuracy Fusion"] - comparison["Accuracy Baseline"]).round(4)
    comparison["Delta F1"] = (comparison["F1 Fusion"] - comparison["F1 Baseline"]).round(4)
    comparison["Delta AUC"] = (comparison["AUC Fusion"] - comparison["AUC Baseline"]).round(4)
    keep_cols = ["Train Dataset", "Test Dataset", "Best Baseline Backbone",
                 "Accuracy Baseline", "F1 Baseline", "AUC Baseline",
                 "Accuracy Fusion", "F1 Fusion", "AUC Fusion",
                 "Delta Accuracy", "Delta F1", "Delta AUC"]
    return comparison[keep_cols]


# -----------------------------------------------------------------------
# Main experiment driver
# -----------------------------------------------------------------------
def run_attention_fusion_cross_dataset_experiment(
    num_epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 8,
    results_dir: str = "results/fusion",
    checkpoints_dir: str = "checkpoints/fusion/cross",
    baseline_csv_path: str = "results/cross_dataset_results.csv",
    force_retrain: bool = False,
    model_type: str = "two_backbone",
    # kept for backward CLI compat:
    use_differential_lr: bool = True,
    use_warm_init: bool = False,
) -> pd.DataFrame:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    use_three = model_type == "three_backbone"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = "ImprovedFusionWithDINO" if use_three else "ImprovedFusionClassifier"
    print(f"Fusion experiment device: {device}")
    print(f"Model: {label} ({'3 backbones' if use_three else '2 backbones: ResNet+CLIP'})")
    print(f"Epochs: {num_epochs}, Patience: {patience}\n")

    records: list[dict[str, object]] = []
    csv_path = os.path.join(results_dir, "fusion_cross_dataset_results.csv")

    if Path(csv_path).exists() and not force_retrain:
        existing_df = pd.read_csv(csv_path)
        records = existing_df.to_dict("records")
        print(f"Resuming with {len(records)} saved result(s).")

    num_backbones = 3 if use_three else 2

    for train_ds, scenario in TRAINING_SCENARIOS.items():
        train_dir = scenario["train_dir"]
        val_dir = scenario["val_dir"]
        test_datasets = scenario["test_datasets"]
        ckpt_path = os.path.join(checkpoints_dir, f"{train_ds.lower()}_{model_type}.pt")

        if not Path(train_dir).exists():
            print(f"[SKIP] Missing train dir for {train_ds}: {train_dir}")
            continue

        print("-" * 72)
        print(f"Training scenario: {train_ds} ({label})")
        print("-" * 72)

        extractors = get_frozen_extractors(device, use_three=use_three)

        if use_three:
            fusion_classifier = ImprovedFusionWithDINO().to(device)
        else:
            fusion_classifier = ImprovedFusionClassifier().to(device)

        if Path(ckpt_path).exists() and not force_retrain:
            print(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            fusion_classifier.load_state_dict(ckpt["fusion_classifier_state"])
        else:
            train_dataset = DeepfakeDataset(train_dir, use_subdirs=True, augment=True)
            if len(train_dataset) == 0:
                print(f"  [SKIP] Empty training dataset: {train_dir}")
                continue

            train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, balance_classes=True)
            val_loader = None
            if Path(val_dir).exists():
                val_dataset = DeepfakeDataset(val_dir, use_subdirs=True, augment=False)
                if len(val_dataset) > 0:
                    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, balance_classes=False)

            print(f"  Train samples: {len(train_dataset)}")
            if val_loader is not None:
                print(f"  Val samples  : {len(val_dataset)}")

            train_fusion_classifier(
                train_loader=train_loader,
                extractors=extractors,
                fusion_classifier=fusion_classifier,
                device=device,
                num_epochs=num_epochs,
                lr=lr,
                save_path=ckpt_path,
                val_loader=val_loader,
                patience=patience,
                num_backbones=num_backbones,
            )

            if Path(ckpt_path).exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                fusion_classifier.load_state_dict(ckpt["fusion_classifier_state"])

        # ---------- Evaluate on each cross-dataset test set ----------
        for test_ds, test_dir in test_datasets.items():
            already_done = any(
                row.get("Train Dataset") == train_ds
                and row.get("Test Dataset") == test_ds
                and row.get("Fusion Model") == label
                for row in records
            )
            if already_done and not force_retrain:
                print(f"  [SKIP already recorded] {train_ds} -> {test_ds}")
                continue
            if not Path(test_dir).exists():
                print(f"  [SKIP missing test dir] {test_ds}: {test_dir}")
                continue

            print(f"  Evaluating -> {test_ds}")
            metrics, weights = _evaluate_fusion_on_dataset(
                extractors, fusion_classifier, test_dir, device, batch_size, use_three=use_three,
            )
            record = {
                "Train Dataset": train_ds,
                "Test Dataset": test_ds,
                "Fusion Model": label,
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Precision": round(float(metrics["precision"]), 4),
                "Recall": round(float(metrics["recall"]), 4),
                "F1": round(float(metrics["f1"]), 4),
                "AUC": round(float(metrics["auc"]), 4),
                "ResNet Weight": round(weights["resnet"], 4),
                "CLIP Weight": round(weights["clip"], 4),
                "DINOv2 Weight": round(weights["dinov2"], 4),
            }
            records.append(record)
            pd.DataFrame(records).to_csv(csv_path, index=False)

            if metrics.get("confusion_matrix") is not None:
                cm_path = os.path.join(results_dir, f"cm_fusion_{train_ds}_{test_ds}_{model_type}.png")
                plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

            print(f"    Acc={record['Accuracy']:.4f}  P={record['Precision']:.4f}  "
                  f"R={record['Recall']:.4f}  F1={record['F1']:.4f}  AUC={record['AUC']:.4f}")
            print(f"    Weights -> ResNet={record['ResNet Weight']:.4f}, "
                  f"CLIP={record['CLIP Weight']:.4f}, DINOv2={record['DINOv2 Weight']:.4f}")

    if not records:
        print("No fusion results generated.")
        return pd.DataFrame()

    fusion_df = pd.DataFrame(records)
    fusion_df.to_csv(csv_path, index=False)

    # Save weights
    weight_csv = os.path.join(results_dir, f"fusion_weights_{model_type}.csv")
    fusion_df[["Train Dataset", "Test Dataset", "ResNet Weight", "CLIP Weight", "DINOv2 Weight"]].to_csv(weight_csv, index=False)

    # Save text table
    table_path = os.path.join(results_dir, f"fusion_cross_dataset_table_{model_type}.txt")
    table_text = _format_fusion_table(fusion_df)
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(table_text)

    # Build comparison vs baseline
    comparison_df = _build_comparison_table(fusion_df, baseline_csv_path)
    if not comparison_df.empty:
        comp_csv = os.path.join(results_dir, f"baseline_vs_fusion_comparison_{model_type}.csv")
        comp_txt = os.path.join(results_dir, f"baseline_vs_fusion_comparison_{model_type}.txt")
        comparison_df.to_csv(comp_csv, index=False)
        with open(comp_txt, "w", encoding="utf-8") as fh:
            fh.write(comparison_df.to_string(index=False))
        print(f"\nComparison CSV  -> {comp_csv}")
        print(f"Comparison table -> {comp_txt}")

    print(f"\nFusion CSV  -> {csv_path}")
    print(f"Table       -> {table_path}")
    print(f"Weights     -> {weight_csv}")
    print("\n" + table_text)
    return fusion_df


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Improved fusion experiment")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--results-dir", default="results/fusion")
    parser.add_argument("--checkpoints-dir", default="checkpoints/fusion/cross")
    parser.add_argument("--baseline-csv", default="results/cross_dataset_results.csv")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--model-type",
        choices=["two_backbone", "three_backbone"],
        default="two_backbone",
        help="two_backbone = ResNet+CLIP only (recommended); three_backbone = all 3 with constrained DINOv2",
    )
    args = parser.parse_args()

    run_attention_fusion_cross_dataset_experiment(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        baseline_csv_path=args.baseline_csv,
        force_retrain=args.force_retrain,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
