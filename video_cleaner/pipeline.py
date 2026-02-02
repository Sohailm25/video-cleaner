"""Main pipeline orchestrating the full video cleaning flow.

Steps:
1. Extract video metadata
2. Sample frames from the video
3. Run OCR on each sampled frame
4. Classify detected text as PII or safe (regex pass)
5. (Optional) Run Kimi K2.5 semantic classifier on all OCR text
6. Merge regex + Kimi results (union: either flags → redact)
7. Track PII regions across frames
8. Re-encode the video with redaction masks applied
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, KimiConfig
from .extractor import extract_frames, get_video_meta
from .ocr import ocr_frame, OCRBox, FrameOCRResult
from .classifier import classify_boxes, merge_classifications, ClassifiedBox
from .kimi_classifier import classify_with_kimi
from .tracker import BoxTracker
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
    kimi_config: Optional[KimiConfig] = None,
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
        kimi_config: Optional Kimi K2.5 configuration. If enabled, runs
            semantic classification as a second pass after regex.

    Returns:
        Path to the cleaned output video.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = str(input_p.with_name(f"{input_p.stem}_cleaned{input_p.suffix}"))

    if kimi_config is None:
        kimi_config = KimiConfig(enabled=False)

    # ── Step 1: Video metadata ──────────────────────────────────────

    meta = get_video_meta(input_path)
    logger.info(
        "Video: %dx%d, %.1f fps, %d frames (%.1fs)",
        meta.width, meta.height, meta.fps, meta.total_frames, meta.duration_seconds,
    )

    if sample_fps > 0 and meta.fps > 0:
        frame_interval = int(round(meta.fps / sample_fps))
    else:
        frame_interval = 1

    # ── Step 2-3: Sample frames + OCR ───────────────────────────────

    all_ocr_results: list[FrameOCRResult] = []
    all_ocr_boxes_flat: list[OCRBox] = []
    frames_processed = 0
    t0 = time.time()

    logger.info("Scanning frames for PII (sample_fps=%.1f)...", sample_fps)

    for frame_idx, timestamp, frame_bgr in extract_frames(input_path, sample_fps):
        ocr_result = ocr_frame(
            frame_bgr,
            frame_index=frame_idx,
            timestamp=timestamp,
            min_confidence=min_ocr_confidence,
            gpu=gpu,
        )
        all_ocr_results.append(ocr_result)
        all_ocr_boxes_flat.extend(ocr_result.boxes)
        frames_processed += 1

    ocr_time = time.time() - t0
    logger.info(
        "OCR complete: %d frames, %d text regions in %.1fs",
        frames_processed, len(all_ocr_boxes_flat), ocr_time,
    )

    # ── Step 4: Regex classification ────────────────────────────────

    t_classify = time.time()
    regex_results = classify_boxes(all_ocr_boxes_flat)
    regex_redactions = sum(1 for r in regex_results if r.should_redact)
    logger.info("Regex classifier: %d/%d boxes flagged for redaction",
                regex_redactions, len(regex_results))

    # ── Step 5: Kimi K2.5 semantic classification (optional) ───────

    kimi_verdicts: dict[int, tuple[str, str]] = {}
    kimi_time = 0.0

    if kimi_config.enabled:
        logger.info("Running Kimi K2.5 semantic classifier...")
        t_kimi = time.time()
        kimi_verdicts = classify_with_kimi(kimi_config, all_ocr_boxes_flat)
        kimi_time = time.time() - t_kimi

        kimi_redact_count = sum(
            1 for v, _ in kimi_verdicts.values() if v == "REDACT"
        )
        logger.info(
            "Kimi classifier: %d/%d boxes analyzed, %d flagged for redaction (%.1fs)",
            len(kimi_verdicts), len(all_ocr_boxes_flat),
            kimi_redact_count, kimi_time,
        )

    # ── Step 6: Merge results ───────────────────────────────────────

    if kimi_verdicts:
        final_results = merge_classifications(
            regex_results, kimi_verdicts, all_ocr_boxes_flat,
        )
    else:
        final_results = regex_results

    total_redactions = sum(1 for r in final_results if r.should_redact)
    classify_time = time.time() - t_classify
    logger.info(
        "Classification complete: %d total redactions (regex=%d, kimi added=%d) in %.1fs",
        total_redactions, regex_redactions,
        total_redactions - regex_redactions, classify_time,
    )

    # ── Step 7: Track regions across frames ─────────────────────────

    tracker = BoxTracker(
        iou_threshold=iou_threshold,
        max_gap_frames=max_gap_frames,
        padding=padding,
    )

    # Feed classifications grouped by frame
    all_detections: list[dict] = []
    result_idx = 0
    for ocr_result in all_ocr_results:
        frame_count = len(ocr_result.boxes)
        frame_classified = final_results[result_idx:result_idx + frame_count]
        result_idx += frame_count

        tracker.update(ocr_result.frame_index, frame_classified)

        for cb in frame_classified:
            if cb.should_redact:
                all_detections.append({
                    "frame_index": ocr_result.frame_index,
                    "timestamp": round(ocr_result.timestamp, 2),
                    "text": cb.box.text,
                    "category": cb.category.value,
                    "pattern": cb.matched_pattern,
                    "source": cb.source,
                    "box": cb.box.to_dict(),
                })

    regions = tracker.finalize()
    logger.info("Tracked %d unique PII regions across video", len(regions))

    if not regions:
        logger.info("No PII detected. Copying input to output without changes.")
        shutil.copy2(input_path, output_path)
        return output_path

    # Expand region frame ranges to cover frames between samples
    for region in regions:
        region.first_frame = max(0, region.first_frame - frame_interval)
        region.last_frame = min(
            meta.total_frames - 1,
            region.last_frame + frame_interval,
        )

    # ── Step 8: Redact and re-encode ────────────────────────────────

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

    # ── Report ──────────────────────────────────────────────────────

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
                "kimi_enabled": kimi_config.enabled,
                "kimi_provider": kimi_config.provider.value if kimi_config.enabled else None,
            },
            "stats": {
                "frames_analyzed": frames_processed,
                "ocr_time_seconds": round(ocr_time, 2),
                "classify_time_seconds": round(classify_time, 2),
                "kimi_time_seconds": round(kimi_time, 2),
                "redact_time_seconds": round(redact_time, 2),
                "total_detections": len(all_detections),
                "regex_detections": regex_redactions,
                "kimi_detections": total_redactions - regex_redactions,
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
