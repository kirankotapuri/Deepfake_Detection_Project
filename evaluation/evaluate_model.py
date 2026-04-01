"""
Model Evaluation for Deepfake Detection

Computes accuracy, AUC, precision, recall, F1, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (for AUC).

    Returns:
        Dict with accuracy, auc, precision, recall, f1, confusion_matrix.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: tuple[str, ...] = ("Real", "Fake"),
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix array.
        save_path: Path to save figure.
        class_names: Class labels for axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=0)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
    label: str | None = None,
) -> float:
    """
    Plot ROC curve and save to file.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        save_path: Path to save the figure.
        title: Plot title.
        label: Legend label (defaults to backbone name).

    Returns:
        AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = float(roc_auc_score(y_true, y_prob))

    fig, ax = plt.subplots(figsize=(6, 5))
    curve_label = f"{label} (AUC={auc_score:.3f})" if label else f"AUC = {auc_score:.3f}"
    ax.plot(fpr, tpr, lw=2, label=curve_label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return auc_score


def plot_roc_curves_multi(
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: str,
    title: str = "ROC Curves",
) -> None:
    """
    Plot multiple ROC curves on one figure.

    Args:
        curves: List of (y_true, y_prob, label) tuples.
        save_path: Output file path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for y_true, y_prob, lbl in curves:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = float(roc_auc_score(y_true, y_prob))
        ax.plot(fpr, tpr, lw=2, label=f"{lbl} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = ("Real", "Fake"),
) -> str:
    """Return sklearn classification report string."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
