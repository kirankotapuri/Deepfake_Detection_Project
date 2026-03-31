"""Evaluation module for deepfake detection."""

from evaluation.evaluate_model import evaluate, plot_confusion_matrix
from evaluation.cross_dataset_test import compute_generalization_gap

__all__ = ["evaluate", "plot_confusion_matrix", "compute_generalization_gap"]
