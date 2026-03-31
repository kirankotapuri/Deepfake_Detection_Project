"""
Resume CelebDF evaluations with corrected DFDC data, then train+eval UADFV scenario.
Run: python resume_cross_eval.py
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.cross_dataset_pipeline import (
    get_feature_extractor, evaluate_on_dataset, _format_table, BACKBONE_LABELS
)
from training.train_classifier import LinearClassifier, train_classifier
from evaluation.evaluate_model import plot_confusion_matrix
from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader

RESULTS_DIR = Path("results")
CKPT_DIR    = Path("checkpoints/cross")
CSV_PATH    = RESULTS_DIR / "cross_dataset_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_records():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} existing records from {CSV_PATH}")
        return df.to_dict("records")
    return []


def already_done(records, train_ds, test_ds, backbone_label):
    return any(
        r.get("Train Dataset") == train_ds
        and r.get("Test Dataset") == test_ds
        and r.get("Backbone") == backbone_label
        for r in records
    )


def save_records(records):
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)


def eval_and_record(extractor, classifier, device, train_ds, test_ds, test_dir, backbone, label, records):
    if already_done(records, train_ds, test_ds, label):
        print(f"  [SKIP - already done] {train_ds} x {label} -> {test_ds}")
        return

    if not Path(test_dir).exists():
        print(f"  [SKIP - dir missing] {test_dir}")
        return

    print(f"  Evaluating -> {test_ds} ...", flush=True)
    m = evaluate_on_dataset(extractor, classifier, test_dir, device, batch_size=32)

    r = {
        "Train Dataset": train_ds,
        "Test Dataset":  test_ds,
        "Backbone":      label,
        "Accuracy":      round(m["accuracy"],  4),
        "Precision":     round(m["precision"], 4),
        "Recall":        round(m["recall"],    4),
        "F1":            round(m["f1"],        4),
        "AUC":           round(m["auc"],       4),
    }
    records.append(r)
    save_records(records)

    print(f"    Acc={r['Accuracy']:.4f}  P={r['Precision']:.4f}  "
          f"R={r['Recall']:.4f}  F1={r['F1']:.4f}  AUC={r['AUC']:.4f}")

    if m.get("confusion_matrix") is not None:
        plot_confusion_matrix(
            m["confusion_matrix"],
            str(RESULTS_DIR / f"cm_{train_ds}_{test_ds}_{backbone}.png")
        )


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    records = load_existing_records()

    # =========================================================
    # SCENARIO A: CelebDF train -> DFDC + UADFV test
    # =========================================================
    tests_celebdf = {
        "DFDC":  "dataset/DFDC_quarter/test",
        "UADFV": "dataset/UADFV_quarter/test",
    }

    for backbone in ["resnet", "clip", "dinov2"]:
        label     = BACKBONE_LABELS[backbone]
        ckpt_path = CKPT_DIR / f"celebdf_{backbone}.pt"

        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found")
            continue

        # Check if both evals already done
        if all(already_done(records, "CelebDF", td, label) for td in tests_celebdf):
            print(f"[SKIP - all done] CelebDF x {label}")
            continue

        print(f"\n{'='*50}")
        print(f"CelebDF x {label}")
        print(f"{'='*50}")
        extractor  = get_feature_extractor(backbone, device)
        classifier = LinearClassifier(extractor.feature_dim, 2).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt["classifier_state"])
        extractor.eval()
        classifier.eval()

        for test_ds, test_dir in tests_celebdf.items():
            eval_and_record(extractor, classifier, device,
                            "CelebDF", test_ds, test_dir, backbone, label, records)

    # =========================================================
    # SCENARIO B: UADFV train -> CelebDF + DFDC test
    # =========================================================
    train_dir_uadfv = "dataset/UADFV_quarter/train"
    val_dir_uadfv   = "dataset/UADFV_quarter/val"
    tests_uadfv = {
        "CelebDF": "dataset/CelebDF_quarter/test",
        "DFDC":    "dataset/DFDC_quarter/test",
    }

    if not Path(train_dir_uadfv).exists():
        print(f"\n[SKIP] UADFV train dir not found: {train_dir_uadfv}")
    else:
        for backbone in ["resnet", "clip", "dinov2"]:
            label     = BACKBONE_LABELS[backbone]
            ckpt_path = CKPT_DIR / f"uadfv_{backbone}.pt"

            print(f"\n{'='*50}")
            print(f"UADFV x {label}")
            print(f"{'='*50}")

            extractor  = get_feature_extractor(backbone, device)
            classifier = LinearClassifier(extractor.feature_dim, 2).to(device)

            # Train if checkpoint missing
            if not ckpt_path.exists():
                print(f"  Training UADFV x {label} ...")
                train_ds_obj = DeepfakeDataset(train_dir_uadfv, use_subdirs=True, augment=True)
                print(f"  Train samples: {len(train_ds_obj)}")
                train_loader = get_dataloader(
                    train_ds_obj, batch_size=16, shuffle=True,
                    num_workers=0, balance_classes=True,
                )
                val_loader = None
                if Path(val_dir_uadfv).exists():
                    val_ds_obj = DeepfakeDataset(val_dir_uadfv, use_subdirs=True, augment=False)
                    if len(val_ds_obj) > 0:
                        print(f"  Val samples:   {len(val_ds_obj)}")
                        val_loader = get_dataloader(
                            val_ds_obj, batch_size=16, shuffle=False,
                            num_workers=0, balance_classes=False,
                        )
                train_classifier(
                    train_loader=train_loader,
                    feature_extractor=extractor,
                    classifier=classifier,
                    device=device,
                    num_epochs=10,
                    lr=1e-4,
                    save_path=str(ckpt_path),
                    val_loader=val_loader,
                    patience=5,
                )
                if ckpt_path.exists():
                    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
                    classifier.load_state_dict(ckpt["classifier_state"])
                    print(f"  Best checkpoint reloaded.")
            else:
                print(f"  Loading checkpoint: {ckpt_path}")
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
                classifier.load_state_dict(ckpt["classifier_state"])

            extractor.eval()
            classifier.eval()

            for test_ds, test_dir in tests_uadfv.items():
                eval_and_record(extractor, classifier, device,
                                "UADFV", test_ds, test_dir, backbone, label, records)

    # =========================================================
    # Final summary table
    # =========================================================
    if records:
        df = pd.DataFrame(records)
        save_records(records)

        backbone_order = {"ResNet50": 0, "CLIP": 1, "DINOv2": 2}
        df["_ord"] = df["Backbone"].map(backbone_order).fillna(99)
        df = df.sort_values(["Train Dataset", "Test Dataset", "_ord"]).drop(columns=["_ord"])
        df.to_csv(CSV_PATH, index=False)

        table_str  = _format_table(df)
        table_path = RESULTS_DIR / "cross_dataset_table.txt"
        table_path.write_text(table_str)

        print(f"\n\n{'='*70}")
        print("FINAL CROSS-DATASET RESULTS")
        print(f"{'='*70}")
        print(table_str)
        print(f"\nCSV   -> {CSV_PATH}")
        print(f"Table -> {table_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
