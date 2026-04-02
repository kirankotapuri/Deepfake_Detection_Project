"""
Project Configuration Schema
=============================
Typed dataclass wrappers around config.yaml.
Use load_config() from utils/config_loader.py to obtain a plain dict,
or use ProjectConfig.from_dict() for attribute access.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from utils.config_loader import load_config


@dataclass
class DatasetConfig:
    celebdf_dir: str = "dataset/CelebDF"
    dfdc_dir: str = "dataset/DFDC"
    uadfv_dir: str = "dataset/UADFV"
    celebdf_quarter: str = "dataset/CelebDF_quarter"
    dfdc_quarter: str = "dataset/DFDC_quarter"
    uadfv_quarter: str = "dataset/UADFV_quarter"
    train_dir: str = "dataset/CelebDF_quarter/train"
    val_dir: str = "dataset/CelebDF_quarter/val"
    test_dir: str = "dataset/CelebDF_quarter/test"
    celebdf_test_dir: str = "dataset/Celeb_V2/Test"
    image_size: int = 224


@dataclass
class PreprocessingConfig:
    frame_rate: int = 1
    max_frames_per_video: int = 100
    face_size: int = 224


@dataclass
class ModelConfig:
    backbone: str = "resnet"   # resnet | clip | dinov2
    num_classes: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 0
    balance_classes: bool = True
    augment: bool = True
    early_stopping_patience: int = 5


@dataclass
class PathsConfig:
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    gradcam_output: str = "results/gradcam_images"
    diffusion_dir: str = "dataset/diffusion_faces"


@dataclass
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "ProjectConfig":
        """Load from YAML file and return a typed ProjectConfig."""
        raw = load_config(config_path)
        return cls(
            dataset=DatasetConfig(**{k: v for k, v in raw.get("dataset", {}).items()
                                     if k in DatasetConfig.__dataclass_fields__}),
            preprocessing=PreprocessingConfig(**{k: v for k, v in raw.get("preprocessing", {}).items()
                                                  if k in PreprocessingConfig.__dataclass_fields__}),
            model=ModelConfig(**{k: v for k, v in raw.get("model", {}).items()
                                  if k in ModelConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in raw.get("training", {}).items()
                                       if k in TrainingConfig.__dataclass_fields__}),
            paths=PathsConfig(**{k: v for k, v in raw.get("paths", {}).items()
                                  if k in PathsConfig.__dataclass_fields__}),
        )

    def ensure_dirs(self) -> None:
        """Create results and checkpoint directories if they don't exist."""
        for d in (self.paths.results_dir, self.paths.checkpoints_dir,
                  self.paths.gradcam_output):
            Path(d).mkdir(parents=True, exist_ok=True)
