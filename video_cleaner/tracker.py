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
    # Per-frame positions for scroll tracking: frame_index -> (x, y, w, h)
    frame_positions: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def get_position_at_frame(self, frame_index: int) -> tuple[int, int, int, int]:
        """Get interpolated (x, y, w, h) for a specific frame."""
        if frame_index in self.frame_positions:
            return self.frame_positions[frame_index]

        frames = sorted(self.frame_positions.keys())
        if not frames:
            return (self.x, self.y, self.w, self.h)

        if frame_index <= frames[0]:
            return self.frame_positions[frames[0]]
        if frame_index >= frames[-1]:
            return self.frame_positions[frames[-1]]

        # Linear interpolation between two nearest sampled frames
        before = max(f for f in frames if f <= frame_index)
        after = min(f for f in frames if f >= frame_index)

        if before == after:
            return self.frame_positions[before]

        t = (frame_index - before) / (after - before)
        bx, by, bw, bh = self.frame_positions[before]
        ax, ay, aw, ah = self.frame_positions[after]
        return (
            int(bx + t * (ax - bx)),
            int(by + t * (ay - by)),
            int(bw + t * (aw - bw)),
            int(bh + t * (ah - bh)),
        )


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
        self._text_index: dict[str, list[int]] = {}  # text -> list of region_ids

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

            # Fallback: text-based matching for scrolled content
            if (best_region is None or best_iou < self.iou_threshold) and box.text.strip():
                text_key = box.text.strip()[:80]
                for rid in self._text_index.get(text_key, []):
                    candidate = next(
                        (r for r in self._active if r.region_id == rid), None
                    )
                    if candidate is None:
                        continue
                    if candidate.category != cb.category:
                        continue
                    if candidate.region_id in matched_region_ids:
                        continue
                    best_region = candidate
                    best_iou = self.iou_threshold  # treat as matched
                    break

            if best_region is not None and best_iou >= self.iou_threshold:
                # Update existing region: expand bbox to union
                best_region.x = min(best_region.x, box.x)
                best_region.y = min(best_region.y, box.y)
                best_region.w = max(best_region.x2, box.x2) - best_region.x
                best_region.h = max(best_region.y2, box.y2) - best_region.y
                best_region.last_frame = frame_index
                best_region.hit_count += 1
                best_region.frame_positions[frame_index] = (
                    max(0, box.x - self.padding),
                    max(0, box.y - self.padding),
                    box.w + 2 * self.padding,
                    box.h + 2 * self.padding,
                )
                self._last_frame_seen[best_region.region_id] = frame_index
                matched_region_ids.add(best_region.region_id)
            else:
                # Start a new tracked region
                padded_x = max(0, box.x - self.padding)
                padded_y = max(0, box.y - self.padding)
                padded_w = box.w + 2 * self.padding
                padded_h = box.h + 2 * self.padding
                region = TrackedRegion(
                    region_id=self._next_id,
                    category=cb.category,
                    x=padded_x,
                    y=padded_y,
                    w=padded_w,
                    h=padded_h,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    sample_text=box.text[:80],
                    frame_positions={frame_index: (padded_x, padded_y, padded_w, padded_h)},
                )
                self._active.append(region)
                self._last_frame_seen[region.region_id] = frame_index
                # Index by text for scroll matching
                text_key = box.text.strip()[:80]
                if text_key:
                    self._text_index.setdefault(text_key, []).append(region.region_id)
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
