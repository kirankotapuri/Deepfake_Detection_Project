"""
Cross-Dataset Generalization Test

Computes generalization gap: AUC(Train Dataset) - AUC(Test Dataset).
Typical setup: Train on FaceForensics++, Test on Celeb-DF.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_generalization_gap(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
) -> dict[str, float]:
    """
    Compute AUC on train and test, and generalization gap.

    Args:
        y_true_train: Train labels.
        y_prob_train: Train predicted probabilities (positive class).
        y_true_test: Test labels (different dataset).
        y_prob_test: Test predicted probabilities.

    Returns:
        Dict with auc_train, auc_test, generalization_gap.
    """
    if len(np.unique(y_true_train)) < 2:
        auc_train = 0.0
    else:
        auc_train = float(roc_auc_score(y_true_train, y_prob_train))

    if len(np.unique(y_true_test)) < 2:
        auc_test = 0.0
    else:
        auc_test = float(roc_auc_score(y_true_test, y_prob_test))

    gap = auc_train - auc_test

    return {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "generalization_gap": gap,
    }
