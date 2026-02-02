"""Apply redaction masks to video frames and re-encode the output video.

Supports two redaction styles:
  - "blur": Gaussian blur over the sensitive region
  - "box": Solid black rectangle over the region
"""

import cv2
import logging
import numpy as np
from enum import Enum
from typing import Optional

from .tracker import TrackedRegion

logger = logging.getLogger(__name__)


class RedactStyle(str, Enum):
    BLUR = "blur"
    BOX = "box"


def redact_frame(
    frame_bgr: np.ndarray,
    regions: list[TrackedRegion],
    style: RedactStyle = RedactStyle.BLUR,
    blur_strength: int = 51,
    box_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Apply redaction to a single frame.

    Args:
        frame_bgr: The original frame (BGR, will NOT be modified in-place).
        regions: List of tracked regions to redact.
        style: Redaction style (blur or solid box).
        blur_strength: Kernel size for Gaussian blur (must be odd).
        box_color: BGR color for solid box redaction.

    Returns:
        A new frame with redactions applied.
    """
    result = frame_bgr.copy()
    h, w = result.shape[:2]

    for region in regions:
        # Clamp coordinates to frame bounds
        rx = max(0, region.x)
        ry = max(0, region.y)
        rx2 = min(w, region.x + region.w)
        ry2 = min(h, region.y + region.h)

        if rx2 <= rx or ry2 <= ry:
            continue

        if style == RedactStyle.BLUR:
            roi = result[ry:ry2, rx:rx2]
            # Ensure kernel size is odd
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            result[ry:ry2, rx:rx2] = blurred
        elif style == RedactStyle.BOX:
            cv2.rectangle(result, (rx, ry), (rx2, ry2), box_color, -1)

    return result


class VideoRedactor:
    """
    Reads the source video frame-by-frame, applies redaction masks
    from tracked regions, and writes the output MP4.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        regions: list[TrackedRegion],
        style: RedactStyle = RedactStyle.BLUR,
        blur_strength: int = 51,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.regions = regions
        self.style = style
        self.blur_strength = blur_strength

    def run(self) -> str:
        """
        Process the entire video and write the redacted output.

        Returns:
            Path to the output file.
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use mp4v codec for broad compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video: {self.output_path}")

        frame_idx = 0
        log_interval = max(1, total // 20)  # Log ~20 times

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Find which regions are active on this frame
            active = [
                r for r in self.regions
                if r.first_frame <= frame_idx <= r.last_frame
            ]

            if active:
                frame = redact_frame(
                    frame, active,
                    style=self.style,
                    blur_strength=self.blur_strength,
                )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % log_interval == 0:
                pct = int(100 * frame_idx / total) if total > 0 else 0
                logger.info("Redacting: %d/%d frames (%d%%)", frame_idx, total, pct)

        writer.release()
        cap.release()
        logger.info(
            "Redacted video written to %s (%d frames, %d regions applied)",
            self.output_path, frame_idx, len(self.regions),
        )
        return self.output_path
