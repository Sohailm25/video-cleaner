# video-cleaner

Automatically detect and redact PII (emails, phone numbers, API keys, passwords, SSNs, etc.) from screen recordings.

## How it works

1. **Frame sampling** — Extracts frames at a configurable rate (default 2 fps)
2. **OCR** — Runs EasyOCR to detect all text with pixel-accurate bounding boxes
3. **PII classification** — Regex pattern matching identifies sensitive text (emails, phone numbers, SSNs, credit cards, API keys, passwords, addresses, medical record numbers, etc.)
4. **Temporal tracking** — IoU-based tracker associates PII regions across frames so masks stay stable and don't flicker
5. **Redaction** — Re-encodes the full video with blur or solid-box masks over every detected PII region

## Install

```bash
pip install -e .
```

## Usage

```bash
# Basic usage — blur redaction (default)
video-cleaner input.mp4

# Solid black box redaction
video-cleaner input.mp4 --style box

# Custom output path + detection report
video-cleaner input.mp4 -o cleaned.mp4 --report detections.json

# Verbose logging, higher sample rate
video-cleaner input.mp4 -v --sample-fps 5

# GPU-accelerated OCR
video-cleaner input.mp4 --gpu
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `<input>_cleaned.mp4` | Output file path |
| `--style` | `blur` | `blur` or `box` |
| `--blur-strength` | `51` | Gaussian blur kernel size |
| `--sample-fps` | `2.0` | Frames per second to analyze |
| `--min-confidence` | `0.3` | OCR confidence threshold |
| `--padding` | `8` | Pixel padding around detections |
| `--gpu` | off | Use CUDA for OCR |
| `--report` | none | Write JSON detection report |
| `-v` | off | Verbose logging |

## What it detects

- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- AWS access keys
- API keys and tokens
- Passwords
- Private keys / secret tokens
- Medical record numbers
- Dates of birth
- Street addresses
- Labeled person names (e.g., "Patient: John Smith")

## Architecture

```
input.mp4
    │
    ├─► Frame Extraction (OpenCV, sampled at N fps)
    │
    ├─► OCR (EasyOCR) ─► bounding boxes + text
    │
    ├─► PII Classifier (regex patterns) ─► REDACT / SAFE
    │
    ├─► Box Tracker (IoU matching across frames)
    │
    └─► Video Redactor (OpenCV re-encode with blur/box masks)
            │
            └─► cleaned.mp4
```
