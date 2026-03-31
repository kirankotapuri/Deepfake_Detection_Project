"""
End-to-End Training Pipeline for Deepfake Detection

Steps:
  1. Load dataset (with optional augmentation + class balancing)
  2. Load feature extractor (ResNet / CLIP / DINOv2)
  3. Train MLP classifier with validation loop + early stopping
  4. Save best model
"""

import os

import torch
from torch.utils.data import DataLoader

from models.resnet_feature_extractor import ResNetFeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.dinov2_feature_extractor import DINOv2FeatureExtractor
from preprocessing.dataset_loader import DeepfakeDataset, get_dataloader
from training.train_classifier import LinearClassifier, train_classifier
from utils.config_loader import load_config
from utils.logger import setup_logger


def get_feature_extractor(backbone: str, device: torch.device):
    """Get feature extractor by name."""
    if backbone == "resnet":
        return ResNetFeatureExtractor(pretrained=True, freeze=True).to(device)
    if backbone == "clip":
        return CLIPFeatureExtractor(freeze=True).to(device)
    if backbone == "dinov2":
        return DINOv2FeatureExtractor(freeze=True).to(device)
    raise ValueError(f"Unknown backbone: {backbone}")


def run_train_pipeline(
    config: dict | None = None,
    train_dir: str | None = None,
    val_dir: str | None = None,
    backbone: str = "resnet",
    save_dir: str | None = None,
) -> str:
    """
    Run full training pipeline.

    Args:
        config: Config dict (loaded from file if None).
        train_dir: Override train directory.
        val_dir: Override validation directory (optional but recommended).
        backbone: resnet, clip, or dinov2.
        save_dir: Directory to save classifier and logs.

    Returns:
        Path to saved classifier.
    """
    if config is None:
        config = load_config()

    train_path = train_dir or config["dataset"]["train_dir"]
    val_path   = val_dir   or config["dataset"].get("val_dir")
    paths       = config["paths"]
    training_cfg = config["training"]
    image_size   = config["dataset"].get("image_size", 224)
    augment      = training_cfg.get("augment", True)
    balance      = training_cfg.get("balance_classes", True)
    patience     = training_cfg.get("early_stopping_patience", 5)

    save_dir = save_dir or paths["checkpoints_dir"]
    os.makedirs(paths["results_dir"], exist_ok=True)
    log_file = os.path.join(paths["results_dir"], "train.log")
    logger = setup_logger(log_file=log_file)
    logger.info(
        f"Training  backbone={backbone}  train={train_path}  "
        f"val={val_path}  augment={augment}  balance={balance}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    extractor = get_feature_extractor(backbone, device)
    feature_dim = extractor.feature_dim

    # ── Training dataset ────────────────────────────────────────────────────
    train_dataset = DeepfakeDataset(train_path, use_subdirs=True, augment=augment, image_size=image_size)
    if len(train_dataset) == 0:
        raise ValueError(f"No samples found in {train_path}. Expected real/ and fake/ subdirs.")

    train_loader = get_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 2),
        balance_classes=balance,
    )
    logger.info(f"Train samples: {len(train_dataset)}")

    # ── Validation dataset ──────────────────────────────────────────────────
    val_loader = None
    if val_path and os.path.isdir(val_path):
        val_dataset = DeepfakeDataset(val_path, use_subdirs=True, augment=False, image_size=image_size)
        if len(val_dataset) > 0:
            val_loader = get_dataloader(
                val_dataset,
                batch_size=training_cfg["batch_size"],
                shuffle=False,
                num_workers=training_cfg.get("num_workers", 2),
                balance_classes=False,
            )
            logger.info(f"Val samples: {len(val_dataset)}")
        else:
            logger.warning(f"val_dir exists but is empty: {val_path}")
    else:
        logger.warning("No val_dir configured — model selection on train accuracy.")

    # ── Classifier ──────────────────────────────────────────────────────────
    classifier = LinearClassifier(feature_dim, num_classes=2).to(device)
    save_path  = os.path.join(save_dir, f"classifier_{backbone}.pt")

    train_classifier(
        train_loader=train_loader,
        feature_extractor=extractor,
        classifier=classifier,
        device=device,
        num_epochs=training_cfg["num_epochs"],
        lr=training_cfg["learning_rate"],
        save_path=save_path,
        val_loader=val_loader,
        patience=patience,
    )

    logger.info(f"Training complete. Best model saved to {save_path}")
    return save_path

