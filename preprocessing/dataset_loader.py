"""
PyTorch Dataset Loader for Deepfake Detection

Loads face images, applies transforms, and returns (image_tensor, label).
Supports directory-based structure: real/ and fake/ subdirectories.

Augmentation policy (train mode):
  - RandomHorizontalFlip               — mirrors faces
  - RandomRotation(±10°)               — slight head-tilt variation
  - ColorJitter(brightness, contrast)  — lighting variation
  - RandomGrayscale(p=0.05)            — occasional grayscale
  - RandomResizedCrop                  — small scale/aspect variation

Eval mode uses only Resize + CenterCrop + Normalize.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# Default normalisation for ImageNet-pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmented transform for training — increases generalisation."""
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation / test."""
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Expected structure:
        root/
            real/    -> label 0
            fake/    -> label 1
        OR a CSV/filelist with paths and labels.

    Returns (image_tensor, label) for each sample.
    """

    def __init__(
        self,
        root: str,
        transform: transforms.Compose | None = None,
        use_subdirs: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        augment: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            root: Root directory containing real/ and fake/ subdirs.
            transform: Custom torchvision transforms; overrides augment flag.
            use_subdirs: If True, expect real/ and fake/ subdirs.
            image_extensions: Valid image extensions.
            augment: If True and transform is None, use train-time augmentation.
            image_size: Target spatial size (224 is standard for ImageNet models).
        """
        self.root = Path(root)
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = get_train_transform(image_size)
        else:
            self.transform = get_eval_transform(image_size)

        self.samples: list[tuple[Path, int]] = []

        if use_subdirs:
            self._load_from_subdirs(image_extensions)
        else:
            self._load_flat(image_extensions)

    def _default_transform(self) -> transforms.Compose:
        """Kept for backwards compatibility; returns eval transform."""
        return get_eval_transform()

    def _load_from_subdirs(self, image_extensions: tuple[str, ...]) -> None:
        """Load paths from real/ and fake/ subdirectories."""
        for subdir, label in [("real", 0), ("fake", 1)]:
            sub_path = self.root / subdir
            if not sub_path.exists():
                continue
            for f in sub_path.rglob("*"):
                if f.suffix.lower() in image_extensions and f.is_file():
                    self.samples.append((f, label))

    def _load_flat(self, image_extensions: tuple[str, ...]) -> None:
        """Load from flat structure: infer label from path (real/fake)."""
        for f in self.root.rglob("*"):
            if f.suffix.lower() not in image_extensions or not f.is_file():
                continue
            path_lower = str(f).lower()
            label = 1 if "fake" in path_lower else 0
            self.samples.append((f, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    balance_classes: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for the dataset.

    Args:
        dataset: DeepfakeDataset instance.
        batch_size: Batch size.
        shuffle: Shuffle data (ignored when balance_classes=True).
        num_workers: Number of worker processes (0 = main process).
        balance_classes: If True, use WeightedRandomSampler so each class
                         is seen equally often per epoch — combats imbalance.

    Returns:
        DataLoader instance.
    """
    sampler = None
    if balance_classes and hasattr(dataset, "samples"):
        labels = [label for _, label in dataset.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        # Weight each sample so rarer class gets upsampled
        weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([weights[lbl] for lbl in labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive

    # Only drop the last incomplete batch when dataset is large enough that
    # doing so won't empty the loader entirely.
    n_samples = len(dataset)  # type: ignore[arg-type]
    drop_last = n_samples > batch_size * 2

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
