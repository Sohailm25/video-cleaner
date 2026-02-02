"""Command-line interface for video-cleaner."""

import argparse
import logging
import sys

from . import __version__
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="video-cleaner",
        description="Detect and redact PII from screen recordings.",
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
    parser.add_argument(
        "--style",
        choices=["blur", "box"],
        default="blur",
        help="Redaction style: gaussian blur or solid black box. Default: blur.",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=51,
        help="Blur kernel size (odd number, higher = stronger). Default: 51.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frames per second to sample for OCR analysis. Default: 2.0.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum OCR confidence threshold (0-1). Default: 0.3.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracking boxes across frames. Default: 0.3.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Pixel padding around detected PII regions. Default: 8.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for OCR (requires CUDA).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write a JSON detection report.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
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
        )
        print(f"\nCleaned video saved to: {output}")
    except Exception as e:
        logging.getLogger(__name__).error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
