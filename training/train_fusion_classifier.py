"""
Training utilities for ImprovedFusionClassifier and ImprovedFusionWithDINO.

Key design decisions (matching user analysis):
- Projection LR   = 5e-5  (learn slowly – these are stable features)
- Weight LR        = 1e-3  (learn fast – only 1-3 parameters)
- Classifier LR    = 1e-4
- Default epochs   = 25
- Default patience = 8
- OneCycleLR scheduler
- Gradient clipping (max_norm=1.0)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.attention_fusion_classifier import (
    ImprovedFusionClassifier,
    ImprovedFusionWithDINO,
)


# -----------------------------------------------------------------------
# Feature extraction helpers (frozen backbones)
# -----------------------------------------------------------------------
def _extract_two_backbones(
    images: torch.Tensor,
    extractors: dict[str, nn.Module],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return extractors["resnet"](images), extractors["clip"](images)


def _extract_all_features(
    images: torch.Tensor,
    extractors: dict[str, nn.Module],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return (
            extractors["resnet"](images),
            extractors["clip"](images),
            extractors["dinov2"](images),
        )


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------
def _validate_fusion(
    val_loader: DataLoader,
    extractors: dict[str, nn.Module],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    use_three: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if use_three:
                features = _extract_all_features(images, extractors)
            else:
                features = _extract_two_backbones(images, extractors)
            logits = model(*features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(len(val_loader), 1), correct / max(total, 1)


# -----------------------------------------------------------------------
# Optimizer builder (user-specified differential LRs)
# -----------------------------------------------------------------------
def _build_optimizer(
    model: nn.Module,
    use_three: bool,
    steps_per_epoch: int,
    num_epochs: int,
):
    if use_three:
        # ImprovedFusionWithDINO
        params = [
            {"params": model.proj_resnet.parameters(), "lr": 5e-5},
            {"params": model.proj_clip.parameters(), "lr": 5e-5},
            {"params": model.proj_dino.parameters(), "lr": 5e-5},
            {"params": [model.raw_weights], "lr": 1e-3},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
        max_lrs = [5e-5, 5e-5, 5e-5, 1e-3, 1e-4]
    else:
        # ImprovedFusionClassifier
        params = [
            {"params": model.proj_resnet.parameters(), "lr": 5e-5},
            {"params": model.proj_clip.parameters(), "lr": 5e-5},
            {"params": [model.clip_weight], "lr": 1e-3},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
        max_lrs = [5e-5, 5e-5, 1e-3, 1e-4]

    optimizer = optim.Adam(params, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    return optimizer, scheduler


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------
def train_fusion_classifier(
    train_loader: DataLoader,
    extractors: dict[str, nn.Module],
    fusion_classifier: nn.Module,
    device: torch.device,
    num_epochs: int = 25,
    lr: float = 1e-4,           # kept for CLI compat but overridden by differential LR
    save_path: str | None = None,
    val_loader: DataLoader | None = None,
    patience: int = 8,
    # Legacy kwargs accepted but ignored for backward CLI compat:
    warm_init_resnet: str | None = None,
    warm_init_clip: str | None = None,
    use_differential_lr: bool = True,
    num_backbones: int = 2,
) -> list[float]:
    """Train ImprovedFusionClassifier or ImprovedFusionWithDINO."""

    use_three = num_backbones == 3

    for extractor in extractors.values():
        extractor.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _build_optimizer(
        fusion_classifier,
        use_three=use_three,
        steps_per_epoch=len(train_loader),
        num_epochs=num_epochs,
    )

    losses: list[float] = []
    best_monitor = 0.0
    no_improve_epochs = 0

    if len(train_loader) == 0:
        raise ValueError("Fusion training DataLoader is empty.")

    for epoch in range(num_epochs):
        fusion_classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Fusion epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if use_three:
                features = _extract_all_features(images, extractors)
            else:
                features = _extract_two_backbones(images, extractors)

            optimizer.zero_grad()
            logits = fusion_classifier(*features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(fusion_classifier.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        losses.append(avg_loss)

        if val_loader is not None:
            val_loss, val_acc = _validate_fusion(
                val_loader, extractors, fusion_classifier, criterion, device, use_three,
            )
            monitor_value = val_acc
            print(
                f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        else:
            monitor_value = train_acc
            print(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f} train_acc={train_acc:.4f}")

        if save_path and monitor_value > best_monitor:
            best_monitor = monitor_value
            no_improve_epochs = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"fusion_classifier_state": fusion_classifier.state_dict(), "epoch": epoch, "monitor_value": monitor_value},
                save_path,
            )
            print(f"  Saved best fusion checkpoint -> {save_path}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return losses


