"""OCR processing to extract text and bounding boxes from frames."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR to avoid slow import at startup
_reader = None


def _get_reader(gpu: bool = False):
    """Lazily initialize the EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Initializing EasyOCR reader (gpu=%s)...", gpu)
        _reader = easyocr.Reader(["en"], gpu=gpu)
        logger.info("EasyOCR reader ready.")
    return _reader


@dataclass
class OCRBox:
    """A detected text region with its bounding box."""
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    frame_index: int = 0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
        }


@dataclass
class FrameOCRResult:
    """All OCR detections for a single frame."""
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    boxes: list[OCRBox] = field(default_factory=list)


def ocr_frame(
    frame_bgr: np.ndarray,
    frame_index: int = 0,
    timestamp: float = 0.0,
    min_confidence: float = 0.3,
    gpu: bool = False,
) -> FrameOCRResult:
    """
    Run OCR on a single frame and return detected text boxes.

    Args:
        frame_bgr: OpenCV BGR image array.
        frame_index: Index of this frame in the video.
        timestamp: Timestamp in seconds.
        min_confidence: Minimum OCR confidence to keep a detection.
        gpu: Whether to use GPU for EasyOCR.

    Returns:
        FrameOCRResult with all detected text boxes.
    """
    reader = _get_reader(gpu=gpu)
    h, w = frame_bgr.shape[:2]

    results = reader.readtext(frame_bgr)

    boxes = []
    for (bbox, text, confidence) in results:
        if confidence < min_confidence:
            continue

        # EasyOCR returns [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        bx = max(0, min(xs))
        by = max(0, min(ys))
        bw = min(w, max(xs)) - bx
        bh = min(h, max(ys)) - by

        if bw > 0 and bh > 0:
            boxes.append(OCRBox(
                text=text.strip(),
                x=bx,
                y=by,
                w=bw,
                h=bh,
                confidence=confidence,
                frame_index=frame_index,
            ))

    logger.debug("Frame %d: found %d text regions", frame_index, len(boxes))
    return FrameOCRResult(
        frame_index=frame_index,
        timestamp=timestamp,
        frame_width=w,
        frame_height=h,
        boxes=boxes,
    )
