"""PII classification using regex pattern matching.

Determines which OCR-detected text boxes contain sensitive information
that should be redacted. Also provides merge utilities for combining
regex results with Kimi K2.5 semantic classifications.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .ocr import OCRBox

logger = logging.getLogger(__name__)


class PIICategory(str, Enum):
    """Categories of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    AWS_KEY = "aws_key"
    PASSWORD = "password"
    SECRET_TOKEN = "secret_token"
    STREET_ADDRESS = "street_address"
    MRN = "medical_record_number"
    DATE_OF_BIRTH = "date_of_birth"
    PERSON_NAME_CONTEXT = "person_name_context"
    # Categories that can come from Kimi semantic classification
    PERSON_NAME = "person_name"
    ID_NUMBER = "id_number"
    CREDENTIAL = "credential"
    MEDICAL_ID = "medical_id"
    KIMI_PII = "kimi_pii"  # generic Kimi-detected PII
    SAFE = "safe"

    @classmethod
    def from_kimi_category(cls, kimi_cat: str) -> "PIICategory":
        """Map a Kimi-returned category string to a PIICategory."""
        mapping = {
            "person_name": cls.PERSON_NAME,
            "email": cls.EMAIL,
            "phone": cls.PHONE,
            "address": cls.STREET_ADDRESS,
            "id_number": cls.ID_NUMBER,
            "credential": cls.CREDENTIAL,
            "medical_id": cls.MEDICAL_ID,
            "ssn": cls.SSN,
            "credit_card": cls.CREDIT_CARD,
            "api_key": cls.API_KEY,
            "password": cls.PASSWORD,
            "ip_address": cls.IP_ADDRESS,
            "safe": cls.SAFE,
        }
        return mapping.get(kimi_cat.lower(), cls.KIMI_PII)


@dataclass
class ClassifiedBox:
    """An OCR box with a PII classification."""
    box: OCRBox
    category: PIICategory
    should_redact: bool
    matched_pattern: str = ""
    source: str = "regex"  # "regex" or "kimi"


# Compiled patterns for PII detection.
# Each entry: (PIICategory, compiled_regex, description)
_PATTERNS: list[tuple[PIICategory, re.Pattern, str]] = [
    # Email addresses
    (PIICategory.EMAIL,
     re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
     "email address"),

    # Phone numbers (various formats)
    (PIICategory.PHONE,
     re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
     "phone number"),

    # SSN
    (PIICategory.SSN,
     re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
     "social security number"),

    # Credit card numbers (basic detection)
    (PIICategory.CREDIT_CARD,
     re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
     "credit card number"),

    # IP addresses (v4)
    (PIICategory.IP_ADDRESS,
     re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
     "IP address"),

    # AWS access key IDs
    (PIICategory.AWS_KEY,
     re.compile(r'\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b'),
     "AWS access key"),

    # Generic API keys / tokens (long hex or alphanumeric strings with key-like prefixes)
    (PIICategory.API_KEY,
     re.compile(r'(?:api[_-]?key|token|secret|bearer)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}', re.IGNORECASE),
     "API key or token"),

    # Password fields
    (PIICategory.PASSWORD,
     re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*\S+', re.IGNORECASE),
     "password"),

    # Secret / private key material
    (PIICategory.SECRET_TOKEN,
     re.compile(r'(?:-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----)', re.IGNORECASE),
     "private key"),

    # Secret tokens (generic long base64-ish strings after common labels)
    (PIICategory.SECRET_TOKEN,
     re.compile(r'(?:secret|private[_-]?key)\s*[:=]\s*["\']?[A-Za-z0-9+/=_\-]{20,}', re.IGNORECASE),
     "secret token"),

    # Medical Record Number patterns
    (PIICategory.MRN,
     re.compile(r'\b(?:MRN|Medical\s*Record)\s*[:#]?\s*\d{5,}', re.IGNORECASE),
     "medical record number"),

    # Date of birth
    (PIICategory.DATE_OF_BIRTH,
     re.compile(r'(?:DOB|Date\s*of\s*Birth|born)\s*[:=]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', re.IGNORECASE),
     "date of birth"),

    # US Street address (basic: number + street name)
    (PIICategory.STREET_ADDRESS,
     re.compile(r'\b\d{1,5}\s+(?:[NSEW]\s+)?(?:\w+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Rd|Road|Ct|Court|Way|Pl|Place)\b', re.IGNORECASE),
     "street address"),

    # Labeled person name fields (e.g., "Patient: John Smith", "Name: Jane Doe")
    (PIICategory.PERSON_NAME_CONTEXT,
     re.compile(r'(?:patient|name|customer|user|client|employee|contact)\s*[:=]\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+', re.IGNORECASE),
     "person name with label"),
]


def classify_box(box: OCRBox) -> ClassifiedBox:
    """
    Classify a single OCR box as containing PII or being safe.

    Runs the text through all PII patterns and returns the first match.
    """
    text = box.text

    for category, pattern, description in _PATTERNS:
        if pattern.search(text):
            logger.debug(
                "REDACT: '%s' matched %s (%s)",
                text[:50], category.value, description
            )
            return ClassifiedBox(
                box=box,
                category=category,
                should_redact=True,
                matched_pattern=description,
            )

    return ClassifiedBox(
        box=box,
        category=PIICategory.SAFE,
        should_redact=False,
    )


def classify_boxes(boxes: list[OCRBox]) -> list[ClassifiedBox]:
    """Classify a list of OCR boxes."""
    return [classify_box(b) for b in boxes]


def merge_classifications(
    regex_results: list[ClassifiedBox],
    kimi_verdicts: dict[int, tuple[str, str]],
    all_boxes: list[OCRBox],
) -> list[ClassifiedBox]:
    """
    Merge regex classifications with Kimi K2.5 semantic verdicts.

    Union strategy: if EITHER regex or Kimi says REDACT, the box is redacted.
    Kimi can only add redactions, never remove ones already flagged by regex.

    Args:
        regex_results: Classifications from the regex classifier.
        kimi_verdicts: Dict mapping box index to (verdict, category) from Kimi.
            Index corresponds to position in all_boxes.
        all_boxes: The original flat list of OCR boxes that was sent to Kimi.

    Returns:
        Merged list of ClassifiedBox (same length as regex_results).
    """
    if not kimi_verdicts:
        return regex_results

    # Build a lookup from (frame_index, x, y, text) to box index in all_boxes
    # so we can correlate Kimi results back to regex results.
    box_key_to_kimi_idx: dict[tuple, int] = {}
    for idx, box in enumerate(all_boxes):
        key = (box.frame_index, box.x, box.y, box.text)
        box_key_to_kimi_idx[key] = idx

    merged = []
    kimi_additions = 0

    for cb in regex_results:
        box = cb.box
        key = (box.frame_index, box.x, box.y, box.text)
        kimi_idx = box_key_to_kimi_idx.get(key)

        if cb.should_redact:
            # Regex already flagged it — keep as-is
            merged.append(cb)
        elif kimi_idx is not None and kimi_idx in kimi_verdicts:
            verdict, kimi_cat = kimi_verdicts[kimi_idx]
            if verdict == "REDACT":
                category = PIICategory.from_kimi_category(kimi_cat)
                merged.append(ClassifiedBox(
                    box=box,
                    category=category,
                    should_redact=True,
                    matched_pattern=f"kimi: {kimi_cat}",
                    source="kimi",
                ))
                kimi_additions += 1
            else:
                merged.append(cb)
        else:
            merged.append(cb)

    if kimi_additions > 0:
        logger.info("Kimi added %d new redactions beyond regex", kimi_additions)

    return merged
