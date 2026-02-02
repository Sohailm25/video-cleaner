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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _make_client(config: KimiConfig):
    """Create a reusable OpenAI client for the Kimi API session."""
    from openai import OpenAI
    return OpenAI(
        api_key=config.api_key,
        base_url=config.provider.base_url,
        timeout=config.timeout_seconds,
    )


def _call_kimi_api(
    config: KimiConfig,
    user_content: str,
    client=None,
) -> Optional[str]:
    """
    Make a single chat completion call to the Kimi K2.5 API.

    Returns the assistant message content, or None on failure.
    Uses exponential backoff retry.
    """
    if client is None:
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

    Optimizations:
    1. Deduplication: identical text strings are sent once, verdict fanned out.
    2. Parallel batching: batches processed concurrently (up to max_concurrent_calls).

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

    # ── Deduplication ────────────────────────────────────────────────
    # Group boxes by text content. Send one representative per unique text.
    text_to_indices: dict[str, list[int]] = {}
    for i, box in enumerate(ocr_boxes):
        text_to_indices.setdefault(box.text, []).append(i)

    # Build representative list (first occurrence of each unique text)
    unique_texts: list[tuple[str, int]] = []  # (text, representative_index)
    for text, indices in text_to_indices.items():
        unique_texts.append((text, indices[0]))

    logger.info(
        "Kimi dedup: %d boxes -> %d unique texts (%.0f%% reduction)",
        len(ocr_boxes), len(unique_texts),
        (1 - len(unique_texts) / len(ocr_boxes)) * 100 if ocr_boxes else 0,
    )

    # Create _IndexedBox entries where batch_id = position in unique_texts
    representatives = [
        _IndexedBox(batch_id=uid, box=ocr_boxes[orig_idx])
        for uid, (_text, orig_idx) in enumerate(unique_texts)
    ]

    # ── Chunk into batches ───────────────────────────────────────────
    chunks: list[list[_IndexedBox]] = []
    for i in range(0, len(representatives), config.max_ocr_items_per_call):
        chunks.append(representatives[i:i + config.max_ocr_items_per_call])

    logger.info(
        "Kimi: %d batches of up to %d items, concurrency=%d",
        len(chunks), config.max_ocr_items_per_call, config.max_concurrent_calls,
    )

    # ── Shared client ────────────────────────────────────────────────
    try:
        client = _make_client(config)
    except Exception:
        logger.error("Failed to create OpenAI client for Kimi")
        return {}

    # ── Batch processor ──────────────────────────────────────────────
    unique_verdicts: dict[int, tuple[str, str]] = {}
    total_chunks = len(chunks)

    def _process_batch(chunk_idx: int, chunk: list[_IndexedBox]) -> dict[int, tuple[str, str]]:
        batch_json = _format_batch(chunk)
        token_estimate = len(batch_json) // 3
        logger.info(
            "Kimi batch %d/%d: %d items (~%d input tokens)",
            chunk_idx + 1, total_chunks, len(chunk), token_estimate,
        )

        response_text = _call_kimi_api(config, batch_json, client=client)
        if response_text is None:
            logger.warning(
                "Kimi batch %d/%d failed, skipping %d items",
                chunk_idx + 1, total_chunks, len(chunk),
            )
            return {}

        verdicts = _parse_json_response(response_text)
        if verdicts is None:
            logger.warning(
                "Kimi batch %d/%d returned unparseable response, skipping",
                chunk_idx + 1, total_chunks,
            )
            return {}

        batch_results: dict[int, tuple[str, str]] = {}
        valid_count = 0
        for item in verdicts:
            if not _validate_verdict(item):
                continue
            batch_id = item["id"]
            if not isinstance(batch_id, int) or batch_id < 0 or batch_id >= len(representatives):
                continue
            verdict = item["verdict"]
            category = item.get("category", "unknown")
            batch_results[batch_id] = (verdict, category)
            valid_count += 1

        logger.info(
            "Kimi batch %d/%d: %d/%d valid verdicts (%d REDACT)",
            chunk_idx + 1, total_chunks, valid_count, len(chunk),
            sum(1 for v, _ in batch_results.values() if v == "REDACT"),
        )
        return batch_results

    # ── Execute batches ──────────────────────────────────────────────
    max_workers = min(config.max_concurrent_calls, len(chunks))
    if max_workers <= 1:
        # Sequential fallback
        for chunk_idx, chunk in enumerate(chunks):
            unique_verdicts.update(_process_batch(chunk_idx, chunk))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_batch, idx, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                try:
                    unique_verdicts.update(future.result())
                except Exception as e:
                    cidx = futures[future]
                    logger.error("Kimi batch %d failed with exception: %s", cidx + 1, e)

    # ── Fan out verdicts to all original indices ─────────────────────
    results: dict[int, tuple[str, str]] = {}
    for uid, (text, _repr_idx) in enumerate(unique_texts):
        if uid in unique_verdicts:
            verdict = unique_verdicts[uid]
            for orig_idx in text_to_indices[text]:
                results[orig_idx] = verdict

    logger.info(
        "Kimi total: %d unique verdicts fanned out to %d box verdicts (%d REDACT)",
        len(unique_verdicts), len(results),
        sum(1 for v, _ in results.values() if v == "REDACT"),
    )

    return results
