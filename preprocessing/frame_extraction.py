"""
Frame Extraction Module for Deepfake Detection Pipeline

Extracts frames from video files at a specified frame rate.
Maintains folder structure for organized output.
"""

import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_rate: int = 1,
    max_frames: int | None = None,
) -> list[str]:
    """
    Extract frames from a video file at a specified frame rate.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        frame_rate: Extract 1 frame every N frames (1 = every frame).
        max_frames: Maximum number of frames to extract (None = no limit).

    Returns:
        List of paths to saved frame images.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video cannot be opened.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_paths = []
    frame_count = 0
    extracted_count = 0

    # Create progress bar
    pbar = tqdm(total=min(total_frames, max_frames or total_frames), desc="Extracting frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every Nth frame
            if frame_count % frame_rate == 0:
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(str(frame_path))
                extracted_count += 1
                pbar.update(1)

                if max_frames is not None and extracted_count >= max_frames:
                    break

            frame_count += 1
    finally:
        cap.release()
        pbar.close()

    return saved_paths


def extract_frames_from_directory(
    video_dir: str,
    output_base_dir: str,
    frame_rate: int = 1,
    max_frames_per_video: int | None = None,
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> dict[str, list[str]]:
    """
    Extract frames from all videos in a directory.
    Maintains folder structure: output_base_dir/video_name/frame_*.jpg

    Args:
        video_dir: Directory containing video files.
        output_base_dir: Base directory for extracted frames.
        frame_rate: Extract 1 frame every N frames.
        max_frames_per_video: Max frames per video (None = no limit).
        video_extensions: Valid video file extensions.

    Returns:
        Dict mapping video name to list of saved frame paths.
    """
    video_dir = Path(video_dir)
    output_base_dir = Path(output_base_dir)
    results = {}

    video_files = [
        f for f in video_dir.rglob("*")
        if f.suffix.lower() in video_extensions and f.is_file()
    ]

    for video_path in tqdm(video_files, desc="Processing videos"):
        # Preserve relative structure
        rel_path = video_path.relative_to(video_dir)
        video_output_dir = output_base_dir / rel_path.parent / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            saved_paths = extract_frames(
                str(video_path),
                str(video_output_dir),
                frame_rate=frame_rate,
                max_frames=max_frames_per_video,
            )
            results[str(video_path)] = saved_paths
        except (FileNotFoundError, ValueError) as e:
            tqdm.write(f"Warning: Skipped {video_path}: {e}")

    return results
