"""ABOUTME: Kimi K2.5 vision-based PII classifier.
ABOUTME: Sends frame images directly to Kimi K2.5 vision API for PII detection with bounding boxes.
"""

import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .classifier import PIICategory, ClassifiedBox
from .config import KimiConfig
from .ocr import OCRBox

logger = logging.getLogger(__name__)


# ── System prompt for vision-based PII detection ──────────────────────

_VISION_SYSTEM_PROMPT = """\
You are a PII detection system analyzing screenshots from screen recordings.

For each image, identify ALL regions containing personally identifiable \
information (PII) and return their pixel-level bounding boxes.

REDACT these types of data:
- Full or partial person names (first + last, or clearly a name in context)
- Email addresses, phone numbers
- Physical / mailing addresses (street-level or more specific)
- SSNs, national ID numbers, passport numbers
- Credit card or bank account numbers
- Dates of birth (full date, not just a year)
- Medical record numbers, patient IDs, insurance policy/member IDs
- Employee IDs, student IDs, badge numbers
- API keys, tokens, secrets, passwords, private keys
- IP addresses that appear to identify a specific system
- Usernames, login credentials tied to a real person

DO NOT REDACT:
- UI labels, buttons, menus, section headers
- Medical conditions, diagnoses, symptoms (e.g. "Type 2 Diabetes")
- Medication names and dosages (e.g. "Lisinopril 10mg")
- Lab test results (e.g. "Positive", "Negative")
- Age or age ranges (e.g. "37 years old")
- Job titles, department names, company names
- Code syntax that is not secrets
- Public information (product names, public URLs)

Return ONLY a JSON array. Each element must have exactly these keys:
  "text": (string, the PII text you see)
  "category": (one of: "person_name", "email", "phone", "address", "ssn", \
"id_number", "credential", "medical_id", "credit_card", "date_of_birth")
  "bbox": [x_min, y_min, x_max, y_max] (integer pixel coordinates in the image)
  "reason": (string, 5-15 words explaining why)

Be precise with bounding boxes — they should tightly cover ONLY the PII text, \
not surrounding labels or UI elements.
If no PII is found, return an empty array: []
Do NOT include any text outside the JSON array. Do NOT wrap in markdown.\
"""


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class VisionDetection:
    """A PII region detected by Kimi K2.5 vision in a single frame."""
    text: str
    category: PIICategory
    x: int
    y: int
    w: int
    h: int
    frame_idx: int
    reason: str = ""


# ── Frame encoding ────────────────────────────────────────────────────

def _encode_frame(frame_bgr: np.ndarray, jpeg_quality: int = 85) -> str:
    """Encode a BGR frame as a base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    _, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Response parsing ──────────────────────────────────────────────────

def _parse_vision_response(text: str) -> Optional[list[dict]]:
    """Parse a JSON array from the vision model response."""
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find JSON array in text
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Extract individual objects
    objects = re.findall(r'\{[^{}]+\}', cleaned)
    if objects:
        parsed = []
        for obj_str in objects:
            try:
                parsed.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if parsed:
            return parsed

    logger.warning("Failed to parse vision response: %s...", text[:200])
    return None


def _validate_detection(item: dict, frame_w: int, frame_h: int) -> Optional[VisionDetection]:
    """Validate and convert a parsed detection dict to VisionDetection."""
    text = item.get("text", "")
    category_str = item.get("category", "")
    bbox = item.get("bbox")
    reason = item.get("reason", "")

    if not text or not bbox or not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    # Clamp to frame bounds
    x_min = max(0, min(x_min, frame_w - 1))
    y_min = max(0, min(y_min, frame_h - 1))
    x_max = max(x_min + 1, min(x_max, frame_w))
    y_max = max(y_min + 1, min(y_max, frame_h))

    w = x_max - x_min
    h = y_max - y_min

    if w < 2 or h < 2:
        return None

    category = PIICategory.from_kimi_category(category_str)
    if category == PIICategory.SAFE:
        return None

    return VisionDetection(
        text=text,
        category=category,
        x=x_min,
        y=y_min,
        w=w,
        h=h,
        frame_idx=0,  # Set by caller
        reason=reason,
    )


# ── API call ──────────────────────────────────────────────────────────

def _classify_single_frame(
    config: KimiConfig,
    frame_bgr: np.ndarray,
    frame_idx: int,
    jpeg_quality: int = 85,
    client=None,
) -> list[VisionDetection]:
    """Send a single frame to Kimi K2.5 vision API and parse detections."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.provider.base_url,
            timeout=config.timeout_seconds,
        )

    b64_frame = _encode_frame(frame_bgr, jpeg_quality)
    frame_h, frame_w = frame_bgr.shape[:2]

    messages = [
        {"role": "system", "content": _VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    f"Image dimensions: {frame_w}x{frame_h} pixels. "
                    f"Identify all PII in this screenshot and return bounding boxes "
                    f"using pixel coordinates within these dimensions."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"},
            },
        ]},
    ]

    last_error = None
    for attempt in range(config.max_retries):
        try:
            kwargs = {
                "model": config.provider.model_id,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                logger.warning("Vision API returned empty response for frame %d", frame_idx)
                continue

            items = _parse_vision_response(content.strip())
            if items is None:
                logger.warning("Failed to parse vision response for frame %d", frame_idx)
                return []

            detections = []
            for item in items:
                det = _validate_detection(item, frame_w, frame_h)
                if det is not None:
                    det.frame_idx = frame_idx
                    detections.append(det)

            return detections

        except Exception as e:
            last_error = e
            delay = config.retry_base_delay * (2 ** attempt)
            logger.warning(
                "Vision API failed for frame %d (attempt %d/%d): %s. Retry in %.1fs...",
                frame_idx, attempt + 1, config.max_retries, str(e)[:200], delay,
            )
            time.sleep(delay)

    logger.error("Vision API failed for frame %d after %d retries: %s",
                 frame_idx, config.max_retries, last_error)
    return []


# ── Parallel frame classification ────────────────────────────────────

def classify_frames_vision(
    config: KimiConfig,
    frames: list[tuple[int, float, np.ndarray]],
    jpeg_quality: int = 85,
) -> list[VisionDetection]:
    """Classify multiple frames in parallel using Kimi K2.5 vision API.

    Args:
        config: Kimi API configuration.
        frames: List of (frame_idx, timestamp, frame_bgr) tuples.
        jpeg_quality: JPEG encoding quality (1-100).

    Returns:
        All detections across all frames.
    """
    if not frames:
        return []

    from openai import OpenAI
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.provider.base_url,
        timeout=config.timeout_seconds,
    )

    logger.info("Vision: classifying %d frames (concurrency=%d, quality=%d)",
                len(frames), config.max_concurrent_calls, jpeg_quality)

    all_detections: list[VisionDetection] = []
    completed = 0

    def _process_frame(idx: int, frame_idx: int, frame_bgr: np.ndarray):
        return idx, _classify_single_frame(
            config, frame_bgr, frame_idx, jpeg_quality, client,
        )

    with ThreadPoolExecutor(max_workers=config.max_concurrent_calls) as executor:
        futures = {
            executor.submit(_process_frame, i, fidx, fbgr): i
            for i, (fidx, _ts, fbgr) in enumerate(frames)
        }

        for future in as_completed(futures):
            try:
                idx, detections = future.result()
                all_detections.extend(detections)
                completed += 1
                frame_idx = frames[idx][0]
                if detections:
                    logger.info(
                        "Vision frame %d/%d (frame_idx=%d): %d PII regions found",
                        completed, len(frames), frame_idx, len(detections),
                    )
                else:
                    logger.debug(
                        "Vision frame %d/%d (frame_idx=%d): no PII found",
                        completed, len(frames), frame_idx,
                    )
            except Exception as e:
                completed += 1
                logger.error("Vision frame processing failed: %s", e)

    logger.info("Vision classification complete: %d total detections across %d frames",
                len(all_detections), len(frames))
    return all_detections


# ── Bridge to existing tracker ────────────────────────────────────────

def vision_detections_to_classified_boxes(
    detections: list[VisionDetection],
) -> dict[int, list[ClassifiedBox]]:
    """Convert vision detections to ClassifiedBox format grouped by frame.

    Returns a dict mapping frame_idx -> list of ClassifiedBox, compatible
    with the existing BoxTracker.update() interface.
    """
    by_frame: dict[int, list[ClassifiedBox]] = {}

    for det in detections:
        box = OCRBox(
            text=det.text,
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            confidence=1.0,  # Vision model confidence is implicit
            frame_index=det.frame_idx,
        )
        cb = ClassifiedBox(
            box=box,
            category=det.category,
            should_redact=True,
            matched_pattern="",
            source="kimi_vision",
        )
        by_frame.setdefault(det.frame_idx, []).append(cb)

    return by_frame
