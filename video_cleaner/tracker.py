"""Track redaction boxes across frames for temporal consistency.

Uses simple IoU (Intersection over Union) matching to associate detected
PII regions across consecutive frames so masks don't flicker on/off.
"""

import logging
from dataclasses import dataclass, field

from .ocr import OCRBox
from .classifier import ClassifiedBox, PIICategory

logger = logging.getLogger(__name__)


@dataclass
class TrackedRegion:
    """A PII region tracked across multiple frames."""
    region_id: int
    category: PIICategory
    # Bounding box (union of all matched boxes, slightly padded)
    x: int
    y: int
    w: int
    h: int
    first_frame: int
    last_frame: int
    sample_text: str = ""
    hit_count: int = 1

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


def _iou(a: OCRBox, b_region: TrackedRegion) -> float:
    """Compute Intersection over Union between an OCRBox and a TrackedRegion."""
    xa = max(a.x, b_region.x)
    ya = max(a.y, b_region.y)
    xb = min(a.x2, b_region.x2)
    yb = min(a.y2, b_region.y2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = a.w * a.h
    area_b = b_region.w * b_region.h
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    """
    Tracks redaction regions across frames.

    For each new frame's classified boxes, matches them against existing
    tracked regions using IoU. Unmatched boxes start new tracked regions.
    Regions that haven't been seen for `max_gap` sampled frames are finalized.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_gap_frames: int = 10,
        padding: int = 5,
    ):
        self.iou_threshold = iou_threshold
        self.max_gap_frames = max_gap_frames
        self.padding = padding

        self._next_id = 0
        self._active: list[TrackedRegion] = []
        self._finalized: list[TrackedRegion] = []
        self._last_frame_seen: dict[int, int] = {}  # region_id -> last frame_index

    def update(self, frame_index: int, classified_boxes: list[ClassifiedBox]):
        """
        Process a new frame's classified boxes.

        Args:
            frame_index: The index of the current frame.
            classified_boxes: Boxes that were marked for redaction.
        """
        redact_boxes = [cb for cb in classified_boxes if cb.should_redact]

        matched_region_ids = set()

        for cb in redact_boxes:
            box = cb.box
            best_iou = 0.0
            best_region = None

            for region in self._active:
                if region.category != cb.category:
                    continue
                score = _iou(box, region)
                if score > best_iou:
                    best_iou = score
                    best_region = region

            if best_region is not None and best_iou >= self.iou_threshold:
                # Update existing region: expand bbox to union
                best_region.x = min(best_region.x, box.x)
                best_region.y = min(best_region.y, box.y)
                best_region.w = max(best_region.x2, box.x2) - best_region.x
                best_region.h = max(best_region.y2, box.y2) - best_region.y
                best_region.last_frame = frame_index
                best_region.hit_count += 1
                self._last_frame_seen[best_region.region_id] = frame_index
                matched_region_ids.add(best_region.region_id)
            else:
                # Start a new tracked region
                region = TrackedRegion(
                    region_id=self._next_id,
                    category=cb.category,
                    x=max(0, box.x - self.padding),
                    y=max(0, box.y - self.padding),
                    w=box.w + 2 * self.padding,
                    h=box.h + 2 * self.padding,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    sample_text=box.text[:80],
                )
                self._active.append(region)
                self._last_frame_seen[region.region_id] = frame_index
                self._next_id += 1

        # Finalize regions that haven't appeared in max_gap_frames
        still_active = []
        for region in self._active:
            last_seen = self._last_frame_seen.get(region.region_id, region.first_frame)
            if frame_index - last_seen > self.max_gap_frames:
                self._finalized.append(region)
            else:
                still_active.append(region)
        self._active = still_active

    def finalize(self) -> list[TrackedRegion]:
        """
        Finalize all remaining active regions and return the complete list.

        Call this after processing all frames.
        """
        all_regions = self._finalized + self._active
        self._finalized = []
        self._active = []
        logger.info("Finalized %d tracked redaction regions", len(all_regions))
        return all_regions

    def get_active_regions_for_frame(self, frame_index: int) -> list[TrackedRegion]:
        """Get all tracked regions that should be active at a given frame index."""
        return [
            r for r in (self._finalized + self._active)
            if r.first_frame <= frame_index <= r.last_frame
        ]
