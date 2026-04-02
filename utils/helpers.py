"""
General Utility Helpers
=======================
Seed management, timing, path utilities, and formatting helpers
used across the deepfake detection project.
"""
from __future__ import annotations
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value (default: 42, matches project convention).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic convolutions (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    label_str = f"[{label}] " if label else ""
    print(f"  {label_str}Elapsed: {elapsed:.2f}s")


def count_images(directory: str | Path, extensions: set[str] | None = None) -> int:
    """Count image files in a directory (recursive)."""
    extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(
        1 for f in Path(directory).rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )


def get_class_counts(dataset_root: str | Path) -> dict[str, int]:
    """
    Count images per class in a dataset with real/ and fake/ subdirs.

    Args:
        dataset_root: Path containing real/ and fake/ subdirectories.

    Returns:
        {'real': n, 'fake': n}
    """
    root = Path(dataset_root)
    counts: dict[str, int] = {}
    for cls in ("real", "fake"):
        d = root / cls
        counts[cls] = count_images(d) if d.exists() else 0
    return counts


def format_metrics(metrics: dict[str, float], precision: int = 4) -> str:
    """Format a metrics dict as a readable string."""
    parts = []
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        if key in metrics:
            parts.append(f"{key.capitalize()[:3]}={metrics[key]:.{precision}f}")
    return "  ".join(parts)


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parents. Return the path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and preferred, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def checkpoint_exists(checkpoint_dir: str | Path, name: str) -> tuple[bool, Path]:
    """
    Check if a named checkpoint file exists.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        name:           Checkpoint name without extension.

    Returns:
        (exists: bool, path: Path)
    """
    path = Path(checkpoint_dir) / f"{name}.pt"
    return path.exists(), path


def print_dataset_summary(dataset_root: str | Path, name: str = "") -> None:
    """Print real/fake counts for train/val/test splits."""
    root   = Path(dataset_root)
    label  = f" [{name}]" if name else ""
    print(f"Dataset Summary{label}: {root}")
    for split in ("train", "val", "test"):
        d = root / split
        if d.exists():
            counts = get_class_counts(d)
            total  = sum(counts.values())
            print(f"  {split:5}: real={counts['real']:4}  fake={counts['fake']:4}  total={total}")
