"""
Re-evaluate all trained models on the corrected DFDC test set.

Run this script AFTER:
  1. dataset/DFDC/real  and  dataset/DFDC/fake  are fully populated
     (background re-standardization from standardize_datasets.py must finish)
  2. (optionally) main.py --mode cross_eval  has finished training
     so all checkpoints exist

What this script does:
  1. Rebuilds dataset/DFDC_quarter/test  with the corrected labels
  2. Evaluates every checkpoint in checkpoints/cross/ on that test set
  3. Updates  results/cross_dataset_results.csv  (replaces old DFDC rows)
  4. Re-writes  results/cross_dataset_table.txt

Usage:
    python re_eval_dfdc.py
    python re_eval_dfdc.py --ratio 0.25          # change sample fraction
    python re_eval_dfdc.py --checkpoints-dir checkpoints/cross
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.prepare_dataset import collect_images, copy_images
from evaluation.evaluate_model import evaluate, plot_confusion_matrix
from pipeline.cross_dataset_pipeline import get_feature_extractor, _format_table
from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader
from training.train_classifier import LinearClassifier


# ---------------------------------------------------------------------------
# Checkpoint name → (train_dataset, backbone)
# ---------------------------------------------------------------------------

def _parse_checkpoint_name(name: str) -> tuple[str, str] | None:
    """
    'celebdf_resnet.pt'  -> ('CelebDF', 'resnet')
    'uadfv_clip.pt'      -> ('UADFV',   'clip')
    """
    stem = Path(name).stem.lower()
    for ds in ("celebdf", "uadfv"):
        for bb in ("resnet", "clip", "dinov2"):
            if stem == f"{ds}_{bb}":
                return (ds.upper() if ds == "celebdf" else ds.upper(), bb)
    return None


BACKBONE_LABELS = {"resnet": "ResNet50", "clip": "CLIP", "dinov2": "DINOv2"}


# ---------------------------------------------------------------------------
# Rebuild DFDC_quarter/test
# ---------------------------------------------------------------------------

def rebuild_dfdc_quarter(
    dfdc_std_dir: str = "dataset/DFDC",
    target_dir:   str = "dataset/DFDC_quarter",
    ratio: float = 0.25,
    seed:  int   = 42,
) -> tuple[int, int]:
    """
    Sample *ratio* of DFDC/real and DFDC/fake into DFDC_quarter/test.

    Returns (n_real, n_fake) copied.
    """
    random.seed(seed)
    src_root = Path(dfdc_std_dir)
    dst_test  = Path(target_dir) / "test"

    for cls in ("real", "fake"):
        p = dst_test / cls
        if p.exists():
            shutil.rmtree(p)

    total_real = total_fake = 0
    for cls in ("real", "fake"):
        src = src_root / cls
        if not src.exists() or not any(src.iterdir()):
            print(f"  WARN: {src} is empty – run standardize_datasets.py first")
            continue
        images = collect_images(src)
        random.shuffle(images)
        keep   = max(1, int(len(images) * ratio))
        subset = images[:keep]
        n = copy_images(subset, dst_test / cls)
        print(f"  DFDC_quarter/test/{cls}: {n} images")
        if cls == "real":
            total_real = n
        else:
            total_fake = n

    return total_real, total_fake


# ---------------------------------------------------------------------------
# Evaluate one checkpoint on DFDC test
# ---------------------------------------------------------------------------

def _eval_on_dfdc(
    backbone:   str,
    ckpt_path:  str,
    dfdc_test:  str,
    device:     torch.device,
    batch_size: int = 32,
) -> dict | None:
    extractor  = get_feature_extractor(backbone, device)
    classifier = LinearClassifier(extractor.feature_dim, num_classes=2).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt["classifier_state"])
    except Exception as exc:
        print(f"  ERROR loading {ckpt_path}: {exc}")
        return None

    extractor.eval()
    classifier.eval()

    dataset = DeepfakeDataset(dfdc_test, use_subdirs=True, augment=False)
    if len(dataset) == 0:
        print(f"  ERROR: no samples in {dfdc_test}")
        return None

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats  = extractor(images)
            logits = classifier(feats)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    return evaluate(np.array(all_labels), np.array(all_preds), np.array(all_probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild DFDC_quarter and re-evaluate all models on DFDC."
    )
    parser.add_argument("--dfdc-std",        default="dataset/DFDC")
    parser.add_argument("--dfdc-quarter",    default="dataset/DFDC_quarter")
    parser.add_argument("--checkpoints-dir", default="checkpoints/cross")
    parser.add_argument("--results-dir",     default="results")
    parser.add_argument("--ratio",           type=float, default=0.25)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--batch-size",      type=int,   default=32)
    args = parser.parse_args()

    ckpt_dir    = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    dfdc_test   = str(Path(args.dfdc_quarter) / "test")

    # ── Validation ──────────────────────────────────────────────────────
    dfdc_real = Path(args.dfdc_std) / "real"
    dfdc_fake = Path(args.dfdc_std) / "fake"
    n_real_src = len(collect_images(dfdc_real)) if dfdc_real.exists() else 0
    n_fake_src = len(collect_images(dfdc_fake)) if dfdc_fake.exists() else 0
    if n_real_src == 0 or n_fake_src == 0:
        print(
            f"ERROR: DFDC/real={n_real_src}  DFDC/fake={n_fake_src}\n"
            f"The background standardize_dfdc job has not finished yet.\n"
            f"Run this script again once it completes."
        )
        return
    print(f"DFDC source: real={n_real_src}  fake={n_fake_src}")

    # ── Step 1: Rebuild DFDC_quarter ────────────────────────────────────
    print("\nRebuilding DFDC_quarter/test …")
    nr, nf = rebuild_dfdc_quarter(
        args.dfdc_std, args.dfdc_quarter, args.ratio, args.seed
    )
    print(f"Rebuilt: real={nr}  fake={nf}")
    if nr == 0 or nf == 0:
        print("ERROR: one class is empty — cannot compute meaningful metrics.")
        return

    # ── Step 2: Evaluate each checkpoint on new DFDC test ───────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    new_records: list[dict] = []
    for ckpt_file in sorted(ckpt_dir.glob("*.pt")):
        parsed = _parse_checkpoint_name(ckpt_file.name)
        if parsed is None:
            continue
        train_ds, backbone = parsed
        label = BACKBONE_LABELS.get(backbone, backbone.upper())
        print(f"\n  [{train_ds} × {label}]  →  DFDC")

        metrics = _eval_on_dfdc(backbone, str(ckpt_file), dfdc_test, device, args.batch_size)
        if metrics is None:
            continue

        record = {
            "Train Dataset": train_ds,
            "Test Dataset":  "DFDC",
            "Backbone":      label,
            "Accuracy":      round(metrics["accuracy"],  4),
            "Precision":     round(metrics["precision"], 4),
            "Recall":        round(metrics["recall"],    4),
            "F1":            round(metrics["f1"],        4),
            "AUC":           round(metrics["auc"],       4),
        }
        new_records.append(record)
        print(
            f"    Acc={record['Accuracy']:.4f}  "
            f"P={record['Precision']:.4f}  "
            f"R={record['Recall']:.4f}  "
            f"F1={record['F1']:.4f}  "
            f"AUC={record['AUC']:.4f}"
        )

        if metrics.get("confusion_matrix") is not None:
            cm_path = str(results_dir / f"cm_{train_ds}_DFDC_{backbone}.png")
            plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    if not new_records:
        print("\nNo results to update.")
        return

    # ── Step 3: Merge with existing CSV (replace old DFDC rows) ─────────
    csv_path = results_dir / "cross_dataset_results.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Drop any old DFDC rows
        existing = existing[existing["Test Dataset"] != "DFDC"]
        merged   = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
    else:
        merged = pd.DataFrame(new_records)

    # Sort by Train Dataset, Test Dataset, Backbone
    backbone_order = {"ResNet50": 0, "CLIP": 1, "DINOv2": 2}
    merged["_bb_order"] = merged["Backbone"].map(backbone_order).fillna(99)
    merged = merged.sort_values(["Train Dataset", "Test Dataset", "_bb_order"]).drop(columns=["_bb_order"])
    merged.to_csv(csv_path, index=False)
    print(f"\nUpdated CSV → {csv_path}")

    # ── Step 4: Re-generate text table ──────────────────────────────────
    table_str  = _format_table(merged)
    table_path = results_dir / "cross_dataset_table.txt"
    table_path.write_text(table_str)
    print(f"Updated table → {table_path}")
    print("\n" + table_str)


if __name__ == "__main__":
    main()
