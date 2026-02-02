"""Command-line interface for video-cleaner."""

import argparse
import logging
import sys

from . import __version__
from .config import KimiConfig, KimiProvider
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="video-cleaner",
        description="Detect and redact PII from screen recordings.",
        epilog=(
            "Examples:\n"
            "  video-cleaner input.mp4\n"
            "  video-cleaner input.mp4 --style box\n"
            "  video-cleaner input.mp4 --kimi --report detections.json\n"
            "  video-cleaner input.mp4 --kimi --kimi-provider together\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to the input MP4 video file.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the cleaned output video. Defaults to <input>_cleaned.mp4.",
    )

    # ── Redaction options ───────────────────────────────────────────

    redact_group = parser.add_argument_group("redaction")
    redact_group.add_argument(
        "--style",
        choices=["blur", "box"],
        default="blur",
        help="Redaction style: gaussian blur or solid black box. Default: blur.",
    )
    redact_group.add_argument(
        "--blur-strength",
        type=int,
        default=51,
        help="Blur kernel size (odd number, higher = stronger). Default: 51.",
    )
    redact_group.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Pixel padding around detected PII regions. Default: 8.",
    )

    # ── OCR options ─────────────────────────────────────────────────

    ocr_group = parser.add_argument_group("OCR")
    ocr_group.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frames per second to sample for OCR analysis. Default: 2.0.",
    )
    ocr_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum OCR confidence threshold (0-1). Default: 0.3.",
    )
    ocr_group.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for OCR (requires CUDA).",
    )

    # ── Tracking options ────────────────────────────────────────────

    track_group = parser.add_argument_group("tracking")
    track_group.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracking boxes across frames. Default: 0.3.",
    )

    # ── Kimi K2.5 options ───────────────────────────────────────────

    kimi_group = parser.add_argument_group(
        "Kimi K2.5",
        "Enable semantic PII classification via the Kimi K2.5 API. "
        "Runs as a second pass after regex to catch contextual PII.",
    )
    kimi_group.add_argument(
        "--kimi",
        action="store_true",
        help="Enable Kimi K2.5 semantic classifier (requires API key).",
    )
    kimi_group.add_argument(
        "--kimi-api-key",
        default=None,
        help=(
            "API key for Kimi K2.5. If not set, reads from environment: "
            "MOONSHOT_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY, "
            "or KIMI_API_KEY."
        ),
    )
    kimi_group.add_argument(
        "--kimi-provider",
        choices=["moonshot", "together", "openrouter"],
        default="moonshot",
        help=(
            "API provider. moonshot = api.moonshot.ai (default), "
            "together = api.together.xyz, openrouter = openrouter.ai."
        ),
    )
    kimi_group.add_argument(
        "--kimi-batch-size",
        type=int,
        default=200,
        help="Max OCR items per Kimi API call. Default: 200.",
    )
    kimi_group.add_argument(
        "--kimi-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each Kimi API call. Default: 60.",
    )
    kimi_group.add_argument(
        "--kimi-concurrency",
        type=int,
        default=4,
        help="Max parallel Kimi API calls. Default: 4.",
    )

    # ── Output options ──────────────────────────────────────────────

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--report",
        default=None,
        help="Path to write a JSON detection report.",
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    output_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build Kimi config
    kimi_config = KimiConfig(
        enabled=args.kimi,
        api_key=args.kimi_api_key,
        provider=KimiProvider(args.kimi_provider),
        max_ocr_items_per_call=args.kimi_batch_size,
        timeout_seconds=args.kimi_timeout,
        max_concurrent_calls=args.kimi_concurrency,
    )
    kimi_config.resolve()

    try:
        output = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            sample_fps=args.sample_fps,
            redact_style=args.style,
            blur_strength=args.blur_strength,
            min_ocr_confidence=args.min_confidence,
            iou_threshold=args.iou_threshold,
            padding=args.padding,
            gpu=args.gpu,
            report_path=args.report,
            kimi_config=kimi_config,
        )
        print(f"\nCleaned video saved to: {output}")
    except Exception as e:
        logging.getLogger(__name__).error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
