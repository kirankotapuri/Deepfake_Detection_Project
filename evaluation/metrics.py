"""
Evaluation Metrics for Deepfake Detection
==========================================
Standalone metric computation functions decoupled from plotting.
All functions accept numpy arrays and return plain Python floats or dicts.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute full metric suite for binary deepfake detection.

    Args:
        y_true: Ground-truth labels  (0=real, 1=fake).
        y_pred: Predicted labels.
        y_prob: Predicted probability for the positive (fake) class.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc, ap.
    """
    metrics: dict[str, float] = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["ap"]  = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auc"] = 0.0
        metrics["ap"]  = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred)


def detection_rate_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """
    True Positive Rate (recall) at a fixed False Positive Rate.
    A standard forensics metric: e.g. TPR @ FPR=1%.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted probabilities for the fake class.
        target_fpr: The FPR operating point (default: 1%).

    Returns:
        TPR value at the closest FPR <= target_fpr.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    indices = np.where(fpr <= target_fpr)[0]
    if len(indices) == 0:
        return 0.0
    return float(tpr[indices[-1]])


def print_summary(metrics: dict[str, float], label: str = "") -> None:
    """Print a compact metrics summary line."""
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}"
        f"Acc={metrics.get('accuracy', 0):.4f}  "
        f"P={metrics.get('precision', 0):.4f}  "
        f"R={metrics.get('recall', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}  "
        f"AUC={metrics.get('auc', 0):.4f}"
    )
