# video-cleaner

Automatically detect and redact PII (emails, phone numbers, API keys, passwords, SSNs, etc.) from screen recordings.

## How it works

1. **Frame sampling** — Extracts frames at a configurable rate (default 2 fps)
2. **OCR** — Runs EasyOCR to detect all text with pixel-accurate bounding boxes
3. **PII classification (regex)** — Pattern matching catches obvious PII (emails, phones, SSNs, credit cards, API keys, passwords, addresses, etc.)
4. **PII classification (Kimi K2.5, optional)** — Semantic AI classifier catches contextual PII that regex misses (unlabeled names, org-specific IDs, sensitive context). Union strategy: either layer flagging = redaction.
5. **Temporal tracking** — IoU-based tracker associates PII regions across frames so masks stay stable and don't flicker
6. **Redaction** — Re-encodes the full video with blur or solid-box masks over every detected PII region

## Install

```bash
pip install -e .
```

## Usage

```bash
# Basic usage — regex-only, blur redaction (default)
video-cleaner input.mp4

# Solid black box redaction
video-cleaner input.mp4 --style box

# Enable Kimi K2.5 semantic classifier (catches more PII)
video-cleaner input.mp4 --kimi

# Kimi via Together AI (has better structured output support)
video-cleaner input.mp4 --kimi --kimi-provider together

# Custom output path + detection report
video-cleaner input.mp4 -o cleaned.mp4 --report detections.json

# Verbose logging, higher sample rate
video-cleaner input.mp4 -v --sample-fps 5

# GPU-accelerated OCR
video-cleaner input.mp4 --gpu

# Full pipeline: Kimi + report + box redaction
video-cleaner input.mp4 --kimi --report detections.json --style box
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
| `--kimi` | off | Enable Kimi K2.5 semantic classifier |
| `--kimi-api-key` | env var | API key (or set MOONSHOT_API_KEY / TOGETHER_API_KEY) |
| `--kimi-provider` | `moonshot` | `moonshot`, `together`, or `openrouter` |
| `--kimi-batch-size` | `200` | Max OCR items per API call |
| `--kimi-timeout` | `60` | API call timeout in seconds |

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

With `--kimi` enabled, additionally catches:
- Unlabeled person names (contextual detection)
- Org-specific internal IDs
- Sensitive content in chat/messaging windows
- Any PII the model can infer from context

## Architecture

```
input.mp4
    │
    ├─► Frame Extraction (OpenCV, sampled at N fps)
    │
    ├─► OCR (EasyOCR) ─► bounding boxes + text
    │       │
    │       ├─► Regex Classifier ─► high-confidence REDACT / SAFE
    │       │
    │       └─► Kimi K2.5 Classifier (optional, API) ─► semantic REDACT / SAFE
    │               │
    │               └─► Merge (union: either flags → REDACT)
    │
    ├─► Box Tracker (IoU matching across frames)
    │
    └─► Video Redactor (OpenCV re-encode with blur/box masks)
            │
            └─► cleaned.mp4
```

## Kimi K2.5 Integration

The optional `--kimi` flag enables a second-pass semantic classifier powered by
[Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) (Moonshot AI, Jan 2026).

**How it works:** After EasyOCR extracts text + bounding boxes and regex handles
obvious patterns, the full OCR text is batched and sent to Kimi K2.5 for semantic
analysis. Kimi classifies each text region as REDACT or SAFE with a category and
reason. Results are merged using a union strategy — if either regex or Kimi flags
a region, it gets redacted.

**Why two layers:** Regex is fast, deterministic, and works offline. Kimi catches
contextual PII that no regex can (e.g., a person's name without a "Name:" label,
sensitive context in chat messages). Together they provide high coverage.

**Supported providers:**
- `moonshot` — Moonshot AI direct (`api.moonshot.ai`). Default.
- `together` — Together AI (`api.together.xyz`). Better structured output.
- `openrouter` — OpenRouter (`openrouter.ai`).

**Graceful degradation:** If the API is unreachable, rate-limited, or returns
unparseable responses, the pipeline continues with regex-only results and logs
warnings. The video always gets processed.
