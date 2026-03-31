"""Pipeline module for deepfake detection."""

from pipeline.train_pipeline import run_train_pipeline, get_feature_extractor
from pipeline.inference_pipeline import load_model_for_inference, predict, preprocess_image

__all__ = [
    "run_train_pipeline",
    "get_feature_extractor",
    "load_model_for_inference",
    "predict",
    "preprocess_image",
]
