"""
Preprocessing Coordinator
==========================
Orchestrates the full preprocessing pipeline:
  Video → Frames → Face crops → Normalized tensors

Ties together frame_extraction.py and face_extraction.py into a
single callable workflow with progress logging.
"""
from __future__ import annotations
import logging
from pathlib import Path

from preprocessing.frame_extraction import extract_frames
from preprocessing.face_extraction import process_image_directory as extract_faces

log = logging.getLogger(__name__)


def preprocess_video(
    video_path: str | Path,
    output_dir: str | Path,
    frame_rate: int = 1,
    max_frames: int = 100,
    face_size: int = 224,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Full pipeline: video → frames → face crops.

    Args:
        video_path:     Path to input .mp4 / .avi file.
        output_dir:     Root directory for outputs.
        frame_rate:     Frames to extract per second.
        max_frames:     Maximum number of frames to extract.
        face_size:      Output face image size (pixels).
        skip_existing:  Skip if output directory already has images.

    Returns:
        List of paths to extracted face images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    frames_dir = output_dir / "frames" / video_path.stem
    faces_dir  = output_dir / "faces"  / video_path.stem

    if skip_existing and faces_dir.exists() and any(faces_dir.iterdir()):
        existing = list(faces_dir.glob("*.jpg")) + list(faces_dir.glob("*.png"))
        log.debug(f"Skipping {video_path.name} — {len(existing)} faces already exist.")
        return existing

    # Step 1: extract frames
    log.info(f"Extracting frames from {video_path.name} ...")
    extract_frames(
        video_path=str(video_path),
        output_dir=str(frames_dir),
        frame_rate=frame_rate,
        max_frames=max_frames,
    )

    # Step 2: detect and crop faces
    log.info(f"Detecting faces in {video_path.name} ...")
    face_paths = extract_faces(
        frames_dir=str(frames_dir),
        output_dir=str(faces_dir),
        face_size=face_size,
    )

    log.info(f"  Done: {len(face_paths)} face images → {faces_dir}")
    return face_paths


def preprocess_dataset(
    video_dir: str | Path,
    output_dir: str | Path,
    label: str = "fake",
    frame_rate: int = 1,
    max_frames: int = 100,
    face_size: int = 224,
) -> dict[str, list[Path]]:
    """
    Preprocess all videos in a directory.

    Args:
        video_dir:  Directory containing .mp4 / .avi files.
        output_dir: Root directory for face outputs.
        label:      'real' or 'fake' — used to organise output subdirs.
        frame_rate: Frames per second to extract.
        max_frames: Cap on frames per video.
        face_size:  Face crop size.

    Returns:
        Dict mapping video stem → list of face image paths.
    """
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir) / label

    video_files = (
        list(video_dir.glob("*.mp4")) +
        list(video_dir.glob("*.avi")) +
        list(video_dir.glob("*.mov"))
    )
    if not video_files:
        log.warning(f"No video files found in {video_dir}")
        return {}

    log.info(f"Preprocessing {len(video_files)} videos from {video_dir} [{label}] ...")
    results: dict[str, list[Path]] = {}
    for i, video_path in enumerate(video_files, 1):
        log.info(f"  [{i}/{len(video_files)}] {video_path.name}")
        faces = preprocess_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_rate=frame_rate,
            max_frames=max_frames,
            face_size=face_size,
        )
        results[video_path.stem] = faces

    total_faces = sum(len(v) for v in results.values())
    log.info(f"Preprocessing complete: {total_faces} face images from {len(video_files)} videos.")
    return results
