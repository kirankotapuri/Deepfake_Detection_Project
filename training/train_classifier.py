"""
MLP Classifier Training for Deepfake Detection

Trains a two-hidden-layer MLP on top of frozen feature extractors.
Improvements over the original linear classifier:
  - Deeper network (2048 → 512 → 128 → 2) for better separation.
  - Batch normalisation + dropout to prevent over-fitting.
  - Validation loop tracks val accuracy each epoch.
  - ReduceLROnPlateau scheduler halves LR when val loss plateaus.
  - Early stopping saves time and avoids over-fitting.
  - Best model is saved based on VALIDATION accuracy (not training).
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearClassifier(nn.Module):
    """
    Two-hidden-layer MLP with Batch Norm and Dropout.

    Architecture: input_dim → 512 → BN → ReLU → Dropout
                             → 128 → BN → ReLU → Dropout
                             → num_classes
    """

    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    train_loader: DataLoader,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-4,
    save_path: str | None = None,
    val_loader: DataLoader | None = None,
    patience: int = 5,
) -> list[float]:
    """
    Train MLP classifier on extracted features with validation and early stopping.

    Args:
        train_loader: DataLoader yielding (images, labels).
        feature_extractor: Frozen backbone (ResNet, CLIP, DINOv2).
        classifier: MLP classifier to train.
        device: Device for training.
        num_epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        save_path: Path to save best classifier state (saved on best val accuracy).
        val_loader: Optional validation DataLoader. If provided, uses val accuracy
                    for best-model selection and early stopping.
        patience: Stop training after this many epochs without val improvement.

    Returns:
        List of training losses per epoch.
    """
    feature_extractor.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    losses: list[float] = []
    best_val_acc = 0.0
    no_improve_epochs = 0

    if len(train_loader) == 0:
        raise ValueError(
            f"Training DataLoader is empty. "
            f"The dataset has fewer samples than batch_size={train_loader.batch_size}. "
            "Reduce batch_size in config.yaml or add more images."
        )

    for epoch in range(num_epochs):
        # ── Training phase ─────────────────────────────────────────────
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = feature_extractor(images)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        losses.append(avg_loss)

        # ── Validation phase ────────────────────────────────────────────
        if val_loader is not None:
            val_loss, val_acc = _validate(val_loader, feature_extractor, classifier, criterion, device)
            scheduler.step(val_loss)
            print(
                f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            monitor_acc = val_acc
        else:
            scheduler.step(avg_loss)
            monitor_acc = train_acc

        # ── Save best model ─────────────────────────────────────────────
        if save_path and monitor_acc > best_val_acc:
            best_val_acc = monitor_acc
            no_improve_epochs = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "classifier_state": classifier.state_dict(),
                    "epoch": epoch,
                    "accuracy": monitor_acc,
                    "train_loss": avg_loss,
                },
                save_path,
            )
            print(f"  ✔ Best model saved (acc={best_val_acc:.4f})")
        else:
            no_improve_epochs += 1

        # ── Early stopping ───────────────────────────────────────────────
        if no_improve_epochs >= patience:
            print(f"  Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs).")
            break

    return losses


def _validate(
    val_loader: DataLoader,
    feature_extractor: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation pass and return (val_loss, val_accuracy)."""
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = feature_extractor(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = total_loss / len(val_loader)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc
