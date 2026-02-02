"""Frame extraction from video files using OpenCV."""

import cv2
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """Metadata about the source video."""
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str
    duration_seconds: float


def get_video_meta(video_path: str) -> VideoMeta:
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        codec=codec,
        duration_seconds=duration,
    )


def extract_frames(video_path: str, sample_fps: float = 2.0):
    """
    Generator that yields (frame_index, timestamp_seconds, frame_bgr) tuples.

    Args:
        video_path: Path to the input video.
        sample_fps: How many frames per second to sample. Set to 0 to extract
                     every frame.

    Yields:
        (frame_index, timestamp_sec, numpy_array_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback

    # If sample_fps is 0 or >= video fps, take every frame
    if sample_fps <= 0 or sample_fps >= video_fps:
        frame_interval = 1
    else:
        frame_interval = int(round(video_fps / sample_fps))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            yield (frame_idx, timestamp, frame)

        frame_idx += 1

    cap.release()
    logger.info("Extracted samples from %d total frames", frame_idx)
