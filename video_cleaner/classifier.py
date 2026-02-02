"""PII classification using regex pattern matching.

Determines which OCR-detected text boxes contain sensitive information
that should be redacted.
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
    SAFE = "safe"


@dataclass
class ClassifiedBox:
    """An OCR box with a PII classification."""
    box: OCRBox
    category: PIICategory
    should_redact: bool
    matched_pattern: str = ""


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
