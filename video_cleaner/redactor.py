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


# ── Affine transform helpers ────────────────────────────────────────


def _compose_affine(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: m1 applied after m2."""
    a1 = np.vstack([m1, [0, 0, 1]])
    a2 = np.vstack([m2, [0, 0, 1]])
    return (a1 @ a2)[:2, :]


def _invert_affine(m: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    full = np.vstack([m, [0, 0, 1]])
    inv = np.linalg.inv(full)
    return inv[:2, :]


def _transform_region(
    m: np.ndarray, x: int, y: int, w: int, h: int,
) -> tuple[int, int, int, int]:
    """Apply a 2x3 affine transform to a rectangle, return (x, y, w, h)."""
    # Transform top-left and bottom-right corners
    tl = m @ np.array([x, y, 1.0])
    br = m @ np.array([x + w, y + h, 1.0])
    nx = int(round(min(tl[0], br[0])))
    ny = int(round(min(tl[1], br[1])))
    nw = int(round(abs(br[0] - tl[0])))
    nh = int(round(abs(br[1] - tl[1])))
    return (nx, ny, max(1, nw), max(1, nh))


# ── Motion estimation ────────────────────────────────────────────────


class MotionEstimator:
    """Estimates global frame-to-frame motion using ORB feature matching.

    Computes a cumulative affine transform (translation + uniform scale)
    for each frame. Used during redaction to shift PII blur positions
    so they follow actual scroll/zoom movement between OCR sample points.
    """

    def __init__(self, downsample: int = 4):
        self.downsample = downsample
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_kps: Optional[list] = None
        self._prev_descs: Optional[np.ndarray] = None
        self._cumulative = np.eye(2, 3, dtype=np.float64)
        # Cumulative transform per frame: frame_idx -> 2x3 matrix
        self.transforms: dict[int, np.ndarray] = {}

    def update(self, frame_bgr: np.ndarray, frame_idx: int):
        """Process a frame and accumulate its motion transform."""
        small = cv2.resize(
            frame_bgr, None,
            fx=1.0 / self.downsample, fy=1.0 / self.downsample,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)

        if (
            self._prev_descs is not None
            and descs is not None
            and len(kps) >= 4
            and len(self._prev_kps) >= 4
        ):
            matches = self._matcher.match(self._prev_descs, descs)
            if len(matches) >= 4:
                src_pts = np.float32(
                    [self._prev_kps[m.queryIdx].pt for m in matches]
                )
                dst_pts = np.float32(
                    [kps[m.trainIdx].pt for m in matches]
                )
                # Scale matched points back to full resolution
                src_pts *= self.downsample
                dst_pts *= self.downsample
                m, _inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC,
                )
                if m is not None:
                    self._cumulative = _compose_affine(m, self._cumulative)

        self.transforms[frame_idx] = self._cumulative.copy()
        self._prev_kps = kps
        self._prev_descs = descs

    def get_compensated_position(
        self, region: TrackedRegion, frame_idx: int,
    ) -> tuple[int, int, int, int]:
        """Get motion-compensated (x, y, w, h) for a region at a frame.

        Finds the nearest OCR-sampled anchor frame, computes the delta
        transform from anchor to current frame, and applies it to the
        anchor's known position.
        """
        # Find nearest anchor frame in region.frame_positions
        if not region.frame_positions:
            return (region.x, region.y, region.w, region.h)

        anchor_frame = min(
            region.frame_positions.keys(),
            key=lambda f: abs(f - frame_idx),
        )
        anchor_pos = region.frame_positions[anchor_frame]

        # If we don't have transforms for both frames, fall back
        t_anchor = self.transforms.get(anchor_frame)
        t_current = self.transforms.get(frame_idx)
        if t_anchor is None or t_current is None:
            return region.get_position_at_frame(frame_idx)

        # delta = T_current @ inv(T_anchor)
        delta = _compose_affine(t_current, _invert_affine(t_anchor))
        return _transform_region(delta, *anchor_pos)


# ── Redaction ─────────────────────────────────────────────────────────


def redact_frame(
    frame_bgr: np.ndarray,
    regions: list[TrackedRegion],
    frame_index: int,
    motion: Optional[MotionEstimator] = None,
    style: RedactStyle = RedactStyle.BLUR,
    blur_strength: int = 51,
    box_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Apply redaction to a single frame.

    Args:
        frame_bgr: The original frame (BGR, will NOT be modified in-place).
        regions: List of tracked regions to redact.
        frame_index: Current frame index (for per-frame position lookup).
        motion: Optional motion estimator for scroll/zoom compensation.
        style: Redaction style (blur or solid box).
        blur_strength: Kernel size for Gaussian blur (must be odd).
        box_color: BGR color for solid box redaction.

    Returns:
        A new frame with redactions applied.
    """
    result = frame_bgr.copy()
    h, w = result.shape[:2]

    for region in regions:
        if motion is not None:
            px, py, pw, ph = motion.get_compensated_position(region, frame_index)
        else:
            px, py, pw, ph = region.get_position_at_frame(frame_index)

        # Clamp coordinates to frame bounds
        rx = max(0, px)
        ry = max(0, py)
        rx2 = min(w, px + pw)
        ry2 = min(h, py + ph)

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

        motion = MotionEstimator(downsample=4)
        frame_idx = 0
        log_interval = max(1, total // 20)  # Log ~20 times

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Build motion model for this frame
            motion.update(frame, frame_idx)

            # Find which regions are active on this frame
            active = [
                r for r in self.regions
                if r.first_frame <= frame_idx <= r.last_frame
            ]

            if active:
                frame = redact_frame(
                    frame, active,
                    frame_index=frame_idx,
                    motion=motion,
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
