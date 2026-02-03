"""ABOUTME: Smart frame sampling based on content change detection.
ABOUTME: Samples frames when visual content changes, not at fixed intervals.
"""

import cv2
import logging
import numpy as np
from typing import Generator

logger = logging.getLogger(__name__)


def sample_frames_smart(
    video_path: str,
    min_fps: float = 0.5,
    max_fps: float = 3.0,
    change_threshold: float = 0.02,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Sample frames from a video based on content changes.

    Instead of fixed-interval sampling, this detects when the screen
    content actually changes (scrolling, page switch, zoom) and samples
    at those transition points. Static screens produce fewer samples.

    Args:
        video_path: Path to the input video.
        min_fps: Minimum sample rate — ensures at least this many
                 frames per second even on static content.
        max_fps: Maximum sample rate — caps sampling during rapid
                 changes (e.g. fast scrolling).
        change_threshold: MSE threshold (0-1 normalized) to consider
                         a frame as "changed". Lower = more sensitive.

    Yields:
        (frame_index, timestamp_seconds, frame_bgr) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video FPS: {video_fps}")

    # Frame intervals for min/max sampling rates
    min_interval = max(1, int(round(video_fps / max_fps)))
    max_interval = max(1, int(round(video_fps / min_fps)))

    prev_gray = None
    prev_sampled_idx = -max_interval  # Force first frame to be sampled
    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / video_fps
        frames_since_sample = frame_idx - prev_sampled_idx

        # Always sample if we've hit the max interval (min_fps guarantee)
        force_sample = frames_since_sample >= max_interval

        # Check for content change if enough time has passed
        should_sample = False
        if frames_since_sample >= min_interval or force_sample:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for fast comparison
            small = cv2.resize(gray, (320, 240))

            if prev_gray is not None:
                # Normalized MSE between frames
                diff = np.mean((small.astype(float) - prev_gray.astype(float)) ** 2)
                mse_normalized = diff / (255.0 * 255.0)
                should_sample = mse_normalized > change_threshold or force_sample
            else:
                # Always sample the first frame
                should_sample = True

            if should_sample:
                prev_gray = small

        if should_sample:
            yield (frame_idx, timestamp, frame)
            prev_sampled_idx = frame_idx
            sampled_count += 1

        frame_idx += 1

    cap.release()
    logger.info(
        "Smart sampling: %d frames sampled from %d total (%.1f%%, effective %.1f fps)",
        sampled_count, total_frames,
        100 * sampled_count / max(1, total_frames),
        sampled_count / max(0.01, total_frames / video_fps),
    )
