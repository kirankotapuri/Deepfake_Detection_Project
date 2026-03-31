"""
Configuration Loader for Deepfake Detection Pipeline

Loads settings from YAML config file.
"""

import os
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Default: config/config.yaml.

    Returns:
        Config dictionary with nested structure.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return _default_config()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    for section in ["dataset", "paths"]:
        if section in config and isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str) and not os.path.isabs(value):
                    config[section][key] = str(project_root / value)

    return config


def _default_config() -> dict:
    """Return default configuration when no file exists."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "dataset": {
            "train_dir": str(project_root / "dataset" / "faceforensics"),
            "test_dir": str(project_root / "dataset" / "faceforensics"),
            "celebdf_test_dir": str(project_root / "dataset" / "celebdf"),
            "dfdc_dir": str(project_root / "dataset" / "dfdc"),
            "image_size": 224,
        },
        "preprocessing": {"frame_rate": 1, "max_frames_per_video": 100, "face_size": 224},
        "model": {"backbone": "resnet", "num_classes": 2},
        "training": {"batch_size": 32, "num_epochs": 50, "learning_rate": 0.001, "num_workers": 0},
        "paths": {
            "results_dir": str(project_root / "results"),
            "checkpoints_dir": str(project_root / "checkpoints"),
            "gradcam_output": str(project_root / "results" / "gradcam_images"),
        },
    }
