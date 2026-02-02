"""Main pipeline orchestrating the full video cleaning flow.

Steps:
1. Extract video metadata
2. Sample frames from the video
3. Run OCR on each sampled frame
4. Classify detected text as PII or safe
5. Track PII regions across frames
6. Re-encode the video with redaction masks applied
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .extractor import extract_frames, get_video_meta
from .ocr import ocr_frame
from .classifier import classify_boxes, ClassifiedBox
from .tracker import BoxTracker, TrackedRegion
from .redactor import VideoRedactor, RedactStyle

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    sample_fps: float = 2.0,
    redact_style: str = "blur",
    blur_strength: int = 51,
    min_ocr_confidence: float = 0.3,
    iou_threshold: float = 0.3,
    max_gap_frames: int = 30,
    padding: int = 8,
    gpu: bool = False,
    report_path: Optional[str] = None,
) -> str:
    """
    Run the full video cleaning pipeline.

    Args:
        input_path: Path to input MP4 file.
        output_path: Path for cleaned output. Defaults to <input>_cleaned.mp4.
        sample_fps: Frames per second to sample for OCR analysis.
        redact_style: "blur" or "box".
        blur_strength: Gaussian blur kernel size (odd number).
        min_ocr_confidence: Minimum OCR confidence threshold.
        iou_threshold: IoU threshold for tracking boxes across frames.
        max_gap_frames: Max frames a region can be unseen before finalizing.
        padding: Pixel padding around detected regions.
        gpu: Use GPU for OCR.
        report_path: Optional path to write a JSON detection report.

    Returns:
        Path to the cleaned output video.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = str(input_p.with_name(f"{input_p.stem}_cleaned{input_p.suffix}"))

    # Step 1: Get video metadata
    meta = get_video_meta(input_path)
    logger.info(
        "Video: %dx%d, %.1f fps, %d frames (%.1fs)",
        meta.width, meta.height, meta.fps, meta.total_frames, meta.duration_seconds,
    )

    # Convert sample_fps gap to frame indices based on actual video fps
    if sample_fps > 0 and meta.fps > 0:
        frame_interval = int(round(meta.fps / sample_fps))
    else:
        frame_interval = 1

    # Adjust max_gap_frames to account for sampling rate
    effective_max_gap = max_gap_frames * frame_interval

    # Step 2-4: Sample frames, OCR, classify
    tracker = BoxTracker(
        iou_threshold=iou_threshold,
        max_gap_frames=max_gap_frames,
        padding=padding,
    )

    all_detections: list[dict] = []
    frames_processed = 0
    t0 = time.time()

    logger.info("Scanning frames for PII (sample_fps=%.1f)...", sample_fps)

    for frame_idx, timestamp, frame_bgr in extract_frames(input_path, sample_fps):
        # OCR
        ocr_result = ocr_frame(
            frame_bgr,
            frame_index=frame_idx,
            timestamp=timestamp,
            min_confidence=min_ocr_confidence,
            gpu=gpu,
        )

        # Classify
        classified = classify_boxes(ocr_result.boxes)

        # Track
        tracker.update(frame_idx, classified)

        # Record detections for report
        for cb in classified:
            if cb.should_redact:
                all_detections.append({
                    "frame_index": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "text": cb.box.text,
                    "category": cb.category.value,
                    "pattern": cb.matched_pattern,
                    "box": cb.box.to_dict(),
                })

        frames_processed += 1

    scan_time = time.time() - t0
    logger.info(
        "Scan complete: %d frames analyzed in %.1fs, %d PII detections",
        frames_processed, scan_time, len(all_detections),
    )

    # Step 5: Finalize tracked regions
    regions = tracker.finalize()
    logger.info("Tracked %d unique PII regions across video", len(regions))

    if not regions:
        logger.info("No PII detected. Copying input to output without changes.")
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    # Expand region frame ranges to cover frames between samples
    # Since we only sampled every Nth frame, but the PII is likely present
    # in the frames between samples too.
    for region in regions:
        region.first_frame = max(0, region.first_frame - frame_interval)
        region.last_frame = min(
            meta.total_frames - 1,
            region.last_frame + frame_interval,
        )

    # Step 6: Redact and re-encode
    style = RedactStyle.BLUR if redact_style == "blur" else RedactStyle.BOX
    logger.info("Applying %s redaction to %d regions...", style.value, len(regions))

    t1 = time.time()
    redactor = VideoRedactor(
        input_path=input_path,
        output_path=output_path,
        regions=regions,
        style=style,
        blur_strength=blur_strength,
    )
    result_path = redactor.run()
    redact_time = time.time() - t1

    logger.info("Redaction complete in %.1fs. Output: %s", redact_time, result_path)

    # Optional: Write detection report
    if report_path:
        report = {
            "input": input_path,
            "output": output_path,
            "video": {
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "total_frames": meta.total_frames,
                "duration_seconds": round(meta.duration_seconds, 2),
            },
            "settings": {
                "sample_fps": sample_fps,
                "redact_style": redact_style,
                "min_ocr_confidence": min_ocr_confidence,
            },
            "stats": {
                "frames_analyzed": frames_processed,
                "scan_time_seconds": round(scan_time, 2),
                "redact_time_seconds": round(redact_time, 2),
                "total_detections": len(all_detections),
                "tracked_regions": len(regions),
            },
            "detections": all_detections,
            "regions": [
                {
                    "region_id": r.region_id,
                    "category": r.category.value,
                    "sample_text": r.sample_text,
                    "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                    "first_frame": r.first_frame,
                    "last_frame": r.last_frame,
                    "hit_count": r.hit_count,
                }
                for r in regions
            ],
        }
        Path(report_path).write_text(json.dumps(report, indent=2))
        logger.info("Detection report written to %s", report_path)

    return result_path
