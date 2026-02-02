"""Centralized configuration for the video-cleaner pipeline.

Supports environment variables and explicit overrides for all settings,
including Kimi K2.5 API integration.
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class KimiProvider(str, Enum):
    """Supported API providers for Kimi K2.5."""
    MOONSHOT = "moonshot"       # api.moonshot.ai — no json_schema support
    TOGETHER = "together"       # api.together.xyz — has json_schema support
    OPENROUTER = "openrouter"   # openrouter.ai

    @property
    def base_url(self) -> str:
        return {
            KimiProvider.MOONSHOT: "https://api.moonshot.ai/v1",
            KimiProvider.TOGETHER: "https://api.together.xyz/v1",
            KimiProvider.OPENROUTER: "https://openrouter.ai/api/v1",
        }[self]

    @property
    def model_id(self) -> str:
        return {
            KimiProvider.MOONSHOT: "kimi-k2.5",
            KimiProvider.TOGETHER: "moonshotai/Kimi-K2.5",
            KimiProvider.OPENROUTER: "moonshotai/kimi-k2.5",
        }[self]


@dataclass
class KimiConfig:
    """Configuration for the Kimi K2.5 API classifier layer."""
    enabled: bool = False
    api_key: Optional[str] = None
    provider: KimiProvider = KimiProvider.MOONSHOT
    temperature: float = 0.6  # instant mode recommended
    max_tokens: int = 4096
    batch_size: int = 50          # max frames per API call
    max_ocr_items_per_call: int = 200  # max OCR boxes per API call
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 2.0  # exponential backoff base
    max_concurrent_calls: int = 4  # max parallel Kimi API requests

    def resolve(self):
        """Resolve API key from environment if not explicitly set."""
        if self.api_key:
            return

        env_map = {
            KimiProvider.MOONSHOT: "MOONSHOT_API_KEY",
            KimiProvider.TOGETHER: "TOGETHER_API_KEY",
            KimiProvider.OPENROUTER: "OPENROUTER_API_KEY",
        }
        env_var = env_map.get(self.provider, "MOONSHOT_API_KEY")
        self.api_key = os.environ.get(env_var)

        if not self.api_key:
            # Also check a generic fallback
            self.api_key = os.environ.get("KIMI_API_KEY")

        if self.enabled and not self.api_key:
            logger.warning(
                "Kimi integration enabled but no API key found. "
                "Set %s or KIMI_API_KEY environment variable, "
                "or pass --kimi-api-key. Falling back to regex-only.",
                env_var,
            )
            self.enabled = False


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Video I/O
    input_path: str = ""
    output_path: Optional[str] = None

    # Frame sampling
    sample_fps: float = 2.0

    # OCR
    min_ocr_confidence: float = 0.3
    gpu: bool = False

    # Redaction
    redact_style: str = "blur"
    blur_strength: int = 51
    padding: int = 8

    # Tracking
    iou_threshold: float = 0.3
    max_gap_frames: int = 30

    # Kimi K2.5 integration
    kimi: KimiConfig = field(default_factory=KimiConfig)

    # Reporting
    report_path: Optional[str] = None
