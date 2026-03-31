"""
Deepfake Detection System — Main Entry Point

System Architecture
-------------------
    Image Input
          ↓
    Dataset Loader
          ↓
    Feature Extraction  (ResNet50 | CLIP | DINOv2)
          ↓
    Linear Classifier
          ↓
    Prediction
          ↓
    Explainability
       ├── GradCAM
       └── t-SNE

Available Modes
---------------
    prepare         Phase 1 — Standardize datasets + create 1/4 subsets
    train           Phase 3 — Train a single backbone on CelebDF (or any dir)
    evaluate        Evaluate a trained model on a test directory
    cross_eval      Phase 4 — Full cross-dataset generalization experiment
    diffusion       Phase 5 — Diffusion robustness experiment
    explain         Phase 6 — GradCAM heatmaps + t-SNE visualization
    full_pipeline   Phases 1–7 — Run everything in order (end-to-end)

Quick-start Examples
--------------------
    # Step 1: Standardize + prepare all datasets
    python main.py --mode prepare

    # Step 2: Run cross-dataset generalization experiment
    python main.py --mode cross_eval

    # Step 3: Test on diffusion images
    python main.py --mode diffusion --diffusion-dir dataset/diffusion_faces

    # Step 4: Explainability (GradCAM + t-SNE)
    python main.py --mode explain --backbone resnet --gradcam --tsne

    # All-in-one:
    python main.py --mode full_pipeline --diffusion-dir dataset/diffusion_faces

    # Train a single backbone manually:
    python main.py --mode train --backbone dinov2 \\
        --train-dir dataset/CelebDF_quarter/train \\
        --val-dir   dataset/CelebDF_quarter/val
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from utils.config_loader import load_config
from utils.logger import setup_logger


# ============================================================
# Mode: prepare
# ============================================================

def run_prepare(args) -> None:
    """Phase 1 — Standardize datasets and create 1/4 subsets."""
    from dataset.standardize_datasets import (
        standardize_celebdf,
        standardize_dfdc,
        standardize_uadfv,
    )
    from dataset.prepare_all_datasets import DATASET_CONFIGS, prepare_one

    root = Path(getattr(args, "root", None) or "dataset")
    ratio = getattr(args, "ratio", None) or 0.25
    seed  = getattr(args, "seed",  42)

    datasets = getattr(args, "datasets", None) or ["celebdf", "dfdc", "uadfv"]

    # Step 1 — Standardize raw datasets to flat real/fake structure
    print("=" * 60)
    print("PHASE 1A — STANDARDIZE DATASETS")
    print("=" * 60)
    if "celebdf" in datasets:
        standardize_celebdf(root)
    if "dfdc" in datasets:
        standardize_dfdc(root)
    if "uadfv" in datasets:
        standardize_uadfv(root)

    # Step 2 — Create 1/4 subsets
    print("\n" + "=" * 60)
    print("PHASE 1B — CREATE 1/4 DATASET SUBSETS")
    print("=" * 60)
    overwrite = getattr(args, "overwrite", False)
    for name in datasets:
        cfg = DATASET_CONFIGS[name]
        prepare_one(name, cfg, ratio, seed, overwrite)

    print("\nData preparation complete.")
    print("Next step: python main.py --mode cross_eval")


# ============================================================
# Mode: train
# ============================================================

def run_train(args):
    """Train mode: load dataset, extract features, train classifier, save model."""
    from pipeline.train_pipeline import run_train_pipeline

    config = load_config(args.config)
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.train_dir:
        config["dataset"]["train_dir"] = args.train_dir
    if args.val_dir:
        config["dataset"]["val_dir"] = args.val_dir

    log_file = os.path.join(config["paths"]["results_dir"], "train.log")
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    logger = setup_logger(log_file=log_file)
    logger.info("Starting training pipeline")

    save_path = run_train_pipeline(
        config=config,
        train_dir=args.train_dir or config["dataset"]["train_dir"],
        val_dir=args.val_dir or config["dataset"].get("val_dir"),
        backbone=config["model"]["backbone"],
    )
    logger.info(f"Model saved to {save_path}")


def run_evaluate(args) -> None:
    """Evaluate a trained model on a test directory."""
    import pandas as pd
    from pipeline.train_pipeline import get_feature_extractor
    from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader
    from training.train_classifier import LinearClassifier
    from evaluation.evaluate_model import evaluate, plot_confusion_matrix

    config = load_config(args.config)
    backbone = args.backbone or config["model"]["backbone"]
    test_dir = getattr(args, "test_dir", None) or config["dataset"]["test_dir"]
    classifier_path = getattr(args, "checkpoint", None) or os.path.join(
        config["paths"]["checkpoints_dir"], f"classifier_{backbone}.pt"
    )
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = get_feature_extractor(backbone, device)

    if not Path(classifier_path).exists():
        print(f"Checkpoint not found: {classifier_path}")
        print("Train first: python main.py --mode train")
        return

    ckpt = torch.load(classifier_path, map_location=device, weights_only=True)
    classifier = LinearClassifier(extractor.feature_dim, num_classes=2).to(device)
    classifier.load_state_dict(ckpt["classifier_state"])
    extractor.eval()
    classifier.eval()

    dataset = DeepfakeDataset(test_dir, use_subdirs=True)
    if len(dataset) == 0:
        print(f"No samples in {test_dir}. Skipping evaluation.")
        return

    loader = get_dataloader(dataset, batch_size=32, shuffle=False)
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

    metrics = evaluate(y_true, y_pred, y_prob)
    print(f"\n{'='*40}")
    print(f"Evaluation Results — {backbone.upper()}")
    print(f"{'='*40}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(results_dir, f"confusion_matrix_{backbone}.png"),
    )
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "confusion_matrix"}]).to_csv(
        os.path.join(results_dir, "metrics.csv"), index=False
    )
    print(f"\nResults saved to {results_dir}/")


# ============================================================
# Mode: cross_eval
# ============================================================

def run_cross_eval(args) -> None:
    """Phase 4 — Cross-dataset generalization experiment."""
    from pipeline.cross_dataset_pipeline import run_cross_dataset_evaluation

    config = load_config(args.config)
    training_cfg = config.get("training", {})

    run_cross_dataset_evaluation(
        backbones=getattr(args, "backbones", None) or ["resnet", "clip", "dinov2"],
        num_epochs=getattr(args, "epochs", None) or training_cfg.get("num_epochs", 10),
        batch_size=getattr(args, "batch_size", None) or training_cfg.get("batch_size", 16),
        lr=getattr(args, "lr", None) or training_cfg.get("learning_rate", 1e-4),
        patience=getattr(args, "patience", None) or training_cfg.get("early_stopping_patience", 5),
        results_dir=config["paths"].get("results_dir", "results"),
        checkpoints_dir=os.path.join(
            config["paths"].get("checkpoints_dir", "checkpoints"), "cross"
        ),
        force_retrain=getattr(args, "force_retrain", False),
    )


# ============================================================
# Mode: diffusion
# ============================================================

def run_diffusion(args) -> None:
    """Phase 5 — Diffusion robustness experiment."""
    from pipeline.diffusion_experiment import run_diffusion_experiment

    config = load_config(args.config)
    diffusion_dir = getattr(args, "diffusion_dir", None)
    if not diffusion_dir:
        diffusion_dir = config.get("paths", {}).get(
            "diffusion_dir", "dataset/diffusion_faces"
        )

    run_diffusion_experiment(
        diffusion_dir=diffusion_dir,
        backbones=getattr(args, "backbones", None) or ["resnet", "clip", "dinov2"],
        train_ds=getattr(args, "train_ds", None) or "celebdf",
        checkpoints_dir=os.path.join(
            config["paths"].get("checkpoints_dir", "checkpoints"), "cross"
        ),
        results_dir=config["paths"].get("results_dir", "results"),
        batch_size=getattr(args, "batch_size", None) or 32,
    )


# ============================================================
# Mode: explain
# ============================================================

def run_explain(args) -> None:
    """Phase 6 — GradCAM heatmaps and t-SNE visualization."""
    from pipeline.inference_pipeline import load_model_for_inference, preprocess_image
    from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader

    config = load_config(args.config)
    backbone = getattr(args, "backbone", None) or config["model"]["backbone"]
    classifier_path = getattr(args, "checkpoint", None) or os.path.join(
        config["paths"]["checkpoints_dir"], f"classifier_{backbone}.pt"
    )
    results_dir = config["paths"]["results_dir"]
    gradcam_dir = config["paths"].get(
        "gradcam_output", os.path.join(results_dir, "gradcam_images")
    )
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not Path(classifier_path).exists():
        # Also try cross-dataset checkpoint
        cross_ckpt = os.path.join(
            config["paths"].get("checkpoints_dir", "checkpoints"),
            "cross", f"celebdf_{backbone}.pt",
        )
        if Path(cross_ckpt).exists():
            classifier_path = cross_ckpt
            print(f"Using cross-dataset checkpoint: {classifier_path}")
        else:
            print(f"Checkpoint not found: {classifier_path}")
            print("Train first: python main.py --mode train  OR  --mode cross_eval")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor, classifier = load_model_for_inference(backbone, classifier_path, device)

    data_dir = getattr(args, "data_dir", None) or config["dataset"].get(
        "test_dir", "dataset/CelebDF_quarter/test"
    )

    do_gradcam = getattr(args, "gradcam", False)
    do_tsne    = getattr(args, "tsne",    False)
    if not do_gradcam and not do_tsne:
        do_gradcam = do_tsne = True

    # ── GradCAM ──────────────────────────────────────────────────────────
    if do_gradcam:
        if backbone == "resnet":
            from explainability.gradcam import gradcam_resnet, save_gradcam

            full_model   = torch.nn.Sequential(extractor, classifier)
            target_layer = extractor.backbone[-1]
            dataset      = DeepfakeDataset(data_dir, use_subdirs=True)
            n_samples    = min(getattr(args, "num_samples", 10), len(dataset))
            print(f"\nGenerating GradCAM for {n_samples} samples …")
            for i in range(n_samples):
                img_path, label = dataset.samples[i]
                x = preprocess_image(str(img_path)).to(device).requires_grad_(True)
                cam = gradcam_resnet(full_model, target_layer, x)
                out_name = f"gradcam_{i:04d}_{'fake' if label == 1 else 'real'}.jpg"
                save_gradcam(str(img_path), cam, os.path.join(gradcam_dir, out_name))
            print(f"GradCAM images saved → {gradcam_dir}/")
        else:
            print(
                f"GradCAM is only implemented for ResNet (got backbone='{backbone}'). "
                "Skipping GradCAM."
            )

    # ── t-SNE ─────────────────────────────────────────────────────────────
    if do_tsne:
        from explainability.tsne_visualization import tsne_visualize, plot_tsne

        dataset = DeepfakeDataset(data_dir, use_subdirs=True)
        if len(dataset) == 0:
            print(f"t-SNE: no samples found in {data_dir}")
        else:
            loader = get_dataloader(dataset, batch_size=32, shuffle=False)
            features_list, labels_list = [], []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    feats  = extractor(images)
                    features_list.append(feats.cpu().numpy())
                    labels_list.append(labels.numpy())

            all_features = np.vstack(features_list)
            all_labels   = np.concatenate(labels_list)
            n_max = min(1000, len(all_features))
            idx   = np.random.choice(len(all_features), n_max, replace=False)

            backbone_label = {"resnet": "ResNet50", "clip": "CLIP", "dinov2": "DINOv2"}.get(
                backbone, backbone
            )
            tsne_path = os.path.join(results_dir, f"tsne_{backbone}.png")
            print(f"\nRunning t-SNE on {n_max} samples …")
            emb = tsne_visualize(all_features[idx], all_labels[idx])
            plot_tsne(
                emb, all_labels[idx],
                tsne_path,
                title=f"t-SNE: {backbone_label} Features (Real vs Fake)",
            )
            print(f"t-SNE plot saved → {tsne_path}")


# ============================================================
# Mode: full_pipeline
# ============================================================

def run_full_pipeline(args) -> None:
    """
    End-to-end pipeline (Phases 1–7).

    1. Prepare datasets  (standardize + 1/4 subset)
    2. Cross-dataset evaluation  (CelebDF train, UADFV train)
    3. Diffusion robustness  (if diffusion_dir exists)
    4. Explainability  (GradCAM + t-SNE with ResNet)
    """
    print("\n" + "=" * 70)
    print("FULL PIPELINE — PHASES 1-7")
    print("=" * 70)

    # Phase 1 — Prepare
    print("\n>>> PHASE 1: DATA PREPARATION\n")
    run_prepare(args)

    # Phase 4 — Cross-dataset evaluation
    print("\n>>> PHASE 4: CROSS-DATASET GENERALIZATION\n")
    run_cross_eval(args)

    # Phase 5 — Diffusion (only if dir exists)
    diffusion_dir = getattr(args, "diffusion_dir", None)
    if not diffusion_dir:
        config = load_config(args.config)
        diffusion_dir = config.get("paths", {}).get(
            "diffusion_dir", "dataset/diffusion_faces"
        )
    if Path(diffusion_dir).exists():
        print("\n>>> PHASE 5: DIFFUSION ROBUSTNESS\n")
        run_diffusion(args)
    else:
        print(
            f"\n>>> PHASE 5 SKIPPED: '{diffusion_dir}' not found.\n"
            f"    To include: python main.py --mode diffusion "
            f"--diffusion-dir <path>"
        )

    # Phase 6 — Explainability
    print("\n>>> PHASE 6: EXPLAINABILITY\n")
    args.gradcam     = True
    args.tsne        = True
    args.num_samples = 20
    args.backbone    = "resnet"
    args.data_dir    = "dataset/CelebDF_quarter/test"
    run_explain(args)

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE")
    print("=" * 70)
    print("Outputs in results/:")
    print("  cross_dataset_results.csv   — generalization metrics (CSV)")
    print("  cross_dataset_table.txt     — formatted comparison table")
    print("  diffusion_results.csv       — diffusion robustness metrics")
    print("  cm_*.png                    — confusion matrices")
    print("  gradcam_images/             — GradCAM heatmaps")
    print("  tsne_resnet.png             — t-SNE feature distribution")


# ============================================================
# Argument parser + entry point
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "cross_eval",
                 "diffusion", "explain", "full_pipeline"],
        required=True,
        help="Pipeline mode to run",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument("--root",    default="dataset",
                        help="Dataset root dir (prepare mode)")
    parser.add_argument("--datasets", nargs="+",
                        choices=["celebdf", "dfdc", "uadfv"],
                        help="Datasets to prepare (default: all)")
    parser.add_argument("--ratio",   type=float, default=0.25,
                        help="Fraction to keep (default: 0.25)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prepared data")
    parser.add_argument("--train-dir",  dest="train_dir",
                        help="Training directory (train mode)")
    parser.add_argument("--val-dir",    dest="val_dir",
                        help="Validation directory (train mode)")
    parser.add_argument("--test-dir",   dest="test_dir",
                        help="Test directory (evaluate mode)")
    parser.add_argument("--data-dir",   dest="data_dir",
                        help="Data directory (explain mode)")
    parser.add_argument("--diffusion-dir", dest="diffusion_dir",
                        help="Diffusion images directory")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--backbone",  choices=["resnet", "clip", "dinov2"])
    parser.add_argument("--backbones", nargs="+",
                        choices=["resnet", "clip", "dinov2"],
                        default=["resnet", "clip", "dinov2"])
    parser.add_argument("--checkpoint", help="Path to classifier checkpoint")

    # ── Training hyper-parameters ─────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  dest="batch_size", type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--patience",    type=int,   default=None)
    parser.add_argument("--force-retrain", dest="force_retrain",
                        action="store_true")
    parser.add_argument("--train-ds",   dest="train_ds", default="celebdf",
                        choices=["celebdf", "uadfv"],
                        help="Trained-on dataset for diffusion mode (default: celebdf)")

    # ── Explain ───────────────────────────────────────────────────────────
    parser.add_argument("--gradcam",     action="store_true")
    parser.add_argument("--tsne",        action="store_true")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=10)

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    dispatch = {
        "prepare":       run_prepare,
        "train":         run_train,
        "evaluate":      run_evaluate,
        "cross_eval":    run_cross_eval,
        "diffusion":     run_diffusion,
        "explain":       run_explain,
        "full_pipeline": run_full_pipeline,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
