"""
General Model Training Wrapper
================================
Wraps the full training loop for end-to-end training of the feature
extractor + classifier as a single unit (for fine-tuning scenarios).

For the standard frozen-backbone approach used in this project, use
training/train_classifier.py instead.
"""
from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-4,
    save_path: str | None = None,
    val_loader: DataLoader | None = None,
    patience: int = 5,
    loss_fn: nn.Module | None = None,
    freeze_backbone: bool = True,
) -> dict[str, list[float]]:
    """
    Train a full model (backbone + classifier) end-to-end.

    Args:
        model:           nn.Module with backbone + classifier (e.g. ResNetDeepfakeDetector).
        train_loader:    DataLoader yielding (images, labels).
        device:          Training device.
        num_epochs:      Maximum epochs.
        lr:              Initial learning rate.
        save_path:       Path to save best model checkpoint.
        val_loader:      Optional validation DataLoader for early stopping.
        patience:        Early stopping patience (epochs without improvement).
        loss_fn:         Loss function. Default: CrossEntropyLoss.
        freeze_backbone: If True, only classifier parameters receive gradients.

    Returns:
        History dict: {'train_loss': [...], 'train_acc': [...], 'val_acc': [...]}.
    """
    model = model.to(device)

    if freeze_backbone and hasattr(model, "extractor"):
        for p in model.extractor.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())

    optimizer = optim.Adam(trainable, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    criterion = loss_fn or nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        avg_loss  = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # Validation
        val_acc = 0.0
        if val_loader:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    v_correct += (model(images).argmax(1) == labels).sum().item()
                    v_total   += labels.size(0)
            val_acc = v_correct / v_total
            history["val_acc"].append(val_acc)
            scheduler.step(avg_loss)

        elapsed = time.time() - t0
        monitor_acc = val_acc if val_loader else train_acc
        print(
            f"  Epoch {epoch}/{num_epochs}: "
            f"loss={avg_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
        )

        if save_path and monitor_acc > best_val_acc:
            best_val_acc = monitor_acc
            no_improve   = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state":  model.state_dict(),
                "epoch":        epoch,
                "accuracy":     best_val_acc,
                "train_loss":   avg_loss,
            }, save_path)
            print(f"  Saved best model (val_acc={best_val_acc:.4f}) → {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs.")
                break

    return history
