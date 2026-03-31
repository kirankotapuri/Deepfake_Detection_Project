"""
Face Extraction Module for Deepfake Detection Pipeline

Detects and crops faces from images using MTCNN (facenet-pytorch).
Resizes faces to 224x224 for compatibility with pretrained backbones.
"""

import os
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


# Default face size for pretrained models (ResNet, CLIP, DINOv2)
FACE_SIZE = 224


def load_mtcnn(device: str | torch.device | None = None) -> MTCNN:
    """
    Load MTCNN face detector.

    Args:
        device: Device to run MTCNN on (cuda/cpu).

    Returns:
        MTCNN model instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(
        image_size=FACE_SIZE,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
    )


def detect_and_crop_face(
    image_path: str,
    mtcnn: MTCNN,
    output_path: str | None = None,
    return_pil: bool = False,
) -> Image.Image | None:
    """
    Load image, detect face, crop and resize to 224x224.

    Args:
        image_path: Path to input image.
        mtcnn: MTCNN detector instance.
        output_path: If provided, save cropped face to this path.
        return_pil: If True, return PIL Image; else return None after saving.

    Returns:
        PIL Image of cropped face, or None if no face detected (or after save).
    """
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    # face is a tensor [C, H, W] when single face
    if face.dim() == 3:
        face_pil = _tensor_to_pil(face)
    else:
        # Multiple faces: take first
        face_pil = _tensor_to_pil(face[0])

    if output_path:
        face_pil.save(output_path)
        return output_path if not return_pil else face_pil

    return face_pil


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C,H,W] in [0,1] to PIL Image."""
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def process_image_directory(
    input_dir: str,
    output_dir: str,
    mtcnn: MTCNN | None = None,
    skip_existing: bool = True,
) -> list[str]:
    """
    Process all images in a directory: detect faces, crop, resize to 224x224, save.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save cropped faces.
        mtcnn: MTCNN instance (created if None).
        skip_existing: Skip images that already have output.

    Returns:
        List of paths to saved face images.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mtcnn is None:
        mtcnn = load_mtcnn()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in image_extensions and f.is_file()
    ]

    saved_paths = []
    for img_path in tqdm(image_files, desc="Extracting faces"):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{img_path.stem}_face.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out_path.exists():
            saved_paths.append(str(out_path))
            continue

        try:
            result = detect_and_crop_face(str(img_path), mtcnn, output_path=str(out_path), return_pil=False)
            if result is not None:
                saved_paths.append(str(out_path))
        except Exception as e:
            tqdm.write(f"Warning: Skipped {img_path}: {e}")

    return saved_paths
