"""Kimi K2.5 semantic PII classifier.

Uses the Kimi K2.5 API as a second-pass classifier on OCR results.
The regex classifier handles obvious patterns (emails, phones, SSNs, etc.)
and this module catches contextual PII that regex misses:
  - Person names without explicit labels
  - Internal employee/patient IDs in org-specific formats
  - Sensitive context (e.g., "the password is hunter2" in a chat)
  - Partial addresses, account numbers, etc.

Design decisions:
  - Prompt-based JSON extraction (not response_format) because Moonshot's
    own API does not support json_schema mode (GitHub Issue #96).
  - Batches OCR results across frames to minimize API calls.
  - Union strategy: Kimi can only ADD redactions, never remove ones
    already flagged by regex.
  - Graceful degradation: API failures log warnings and skip, pipeline
    continues with regex-only results.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from .config import KimiConfig
from .ocr import OCRBox

logger = logging.getLogger(__name__)

# ── System prompt for PII classification ──────────────────────────────

_SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) detection system.

You will receive a JSON array of text detections from a screen recording. Each \
entry has an "id", the detected "text", and the "frame" number.

Your job: classify EACH entry as REDACT or SAFE.

REDACT if the text contains or appears to be:
- Full or partial person names (first + last, or clearly a name in context)
- Email addresses
- Phone numbers
- Physical / mailing addresses
- Social Security Numbers, national ID numbers
- Credit card or bank account numbers
- Dates of birth
- Medical record numbers, patient IDs
- Employee IDs, student IDs
- API keys, tokens, secrets, passwords, private keys
- IP addresses that appear to identify a specific system
- Usernames, user IDs tied to a real person
- Any other personally identifiable or confidential information

SAFE if the text is:
- Generic UI labels (buttons, menus, headers)
- Application chrome (timestamps, version numbers, generic status text)
- Code syntax that is not secrets (variable names, keywords, comments)
- Public information (company names, product names, URLs to public sites)

Respond with ONLY a JSON array. Each element must have exactly these keys:
  "id": (integer, matching the input id)
  "verdict": "REDACT" or "SAFE"
  "category": (string, e.g. "person_name", "email", "phone", "address", "id_number", "credential", "medical_id", "safe")
  "reason": (string, 5-15 words explaining why)

Do NOT include any text outside the JSON array. Do NOT wrap in markdown code blocks.\
"""

# ── Response parsing ──────────────────────────────────────────────────


def _parse_json_response(text: str) -> Optional[list[dict]]:
    """
    Parse a JSON array from the model response, handling common issues:
    - Markdown code fences
    - Leading/trailing text around the JSON
    - Partial JSON recovery
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find the JSON array in the text
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try to find individual JSON objects and reconstruct the array
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

    logger.warning("Failed to parse JSON from Kimi response: %s...", text[:200])
    return None


def _validate_verdict(item: dict) -> bool:
    """Check that a parsed verdict item has the required fields."""
    return (
        isinstance(item, dict)
        and "id" in item
        and "verdict" in item
        and item["verdict"] in ("REDACT", "SAFE")
    )


# ── OCR batch formatting ─────────────────────────────────────────────


@dataclass
class _IndexedBox:
    """An OCR box with an assigned batch ID for correlation."""
    batch_id: int
    box: OCRBox


def _format_batch(indexed_boxes: list[_IndexedBox]) -> str:
    """Format a batch of OCR boxes as a compact JSON array for the prompt."""
    items = []
    for ib in indexed_boxes:
        items.append({
            "id": ib.batch_id,
            "text": ib.box.text,
            "frame": ib.box.frame_index,
        })
    return json.dumps(items, ensure_ascii=False)


# ── API call with retry ──────────────────────────────────────────────


def _call_kimi_api(
    config: KimiConfig,
    user_content: str,
) -> Optional[str]:
    """
    Make a single chat completion call to the Kimi K2.5 API.

    Returns the assistant message content, or None on failure.
    Uses exponential backoff retry.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "openai package is required for Kimi integration. "
            "Install it with: pip install openai"
        )
        return None

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.provider.base_url,
        timeout=config.timeout_seconds,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error = None
    for attempt in range(config.max_retries):
        try:
            kwargs = {
                "model": config.provider.model_id,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }

            # Disable thinking mode for speed (instant mode)
            # Moonshot uses extra_body, Together/OpenRouter ignore it harmlessly
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                logger.warning("Kimi returned empty response (attempt %d)", attempt + 1)

        except Exception as e:
            last_error = e
            delay = config.retry_base_delay * (2 ** attempt)
            logger.warning(
                "Kimi API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1, config.max_retries, str(e)[:200], delay,
            )
            time.sleep(delay)

    logger.error("Kimi API call failed after %d retries: %s", config.max_retries, last_error)
    return None


# ── Public interface ──────────────────────────────────────────────────


def classify_with_kimi(
    config: KimiConfig,
    ocr_boxes: list[OCRBox],
) -> dict[int, tuple[str, str]]:
    """
    Send OCR results to Kimi K2.5 for semantic PII classification.

    Args:
        config: Kimi API configuration.
        ocr_boxes: List of OCR boxes (from all sampled frames in this batch).

    Returns:
        Dict mapping OCR box index (position in input list) to
        (verdict, category) where verdict is "REDACT" or "SAFE"
        and category is a string like "person_name", "email", etc.
        Boxes not in the dict were not classified (API failure).
    """
    if not config.enabled or not config.api_key:
        return {}

    if not ocr_boxes:
        return {}

    # Assign batch IDs and chunk into API calls
    indexed = [_IndexedBox(batch_id=i, box=b) for i, b in enumerate(ocr_boxes)]

    # Split into chunks respecting max_ocr_items_per_call
    chunks = []
    for i in range(0, len(indexed), config.max_ocr_items_per_call):
        chunks.append(indexed[i:i + config.max_ocr_items_per_call])

    results: dict[int, tuple[str, str]] = {}

    for chunk_idx, chunk in enumerate(chunks):
        batch_json = _format_batch(chunk)
        token_estimate = len(batch_json) // 3  # rough char-to-token ratio
        logger.info(
            "Kimi batch %d/%d: %d items (~%d input tokens)",
            chunk_idx + 1, len(chunks), len(chunk), token_estimate,
        )

        response_text = _call_kimi_api(config, batch_json)
        if response_text is None:
            logger.warning(
                "Kimi batch %d/%d failed, skipping %d items",
                chunk_idx + 1, len(chunks), len(chunk),
            )
            continue

        verdicts = _parse_json_response(response_text)
        if verdicts is None:
            logger.warning(
                "Kimi batch %d/%d returned unparseable response, skipping",
                chunk_idx + 1, len(chunks),
            )
            continue

        # Map verdicts back to input indices
        valid_count = 0
        for item in verdicts:
            if not _validate_verdict(item):
                continue
            batch_id = item["id"]
            if not isinstance(batch_id, int) or batch_id < 0 or batch_id >= len(indexed):
                continue
            verdict = item["verdict"]
            category = item.get("category", "unknown")
            results[batch_id] = (verdict, category)
            valid_count += 1

        logger.info(
            "Kimi batch %d/%d: %d/%d valid verdicts (%d REDACT)",
            chunk_idx + 1, len(chunks), valid_count, len(chunk),
            sum(1 for v, _ in results.values() if v == "REDACT"),
        )

    return results
