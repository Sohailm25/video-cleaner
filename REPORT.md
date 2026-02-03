# Video-Cleaner: Project Report

**From empty repo to fully functioning PII detection and redaction pipeline for screen recordings.**

Repository: `Sohailm25/video-cleaner`
Branch: `main`
Date: February 2, 2026
Codebase: 2,687 lines across 15 Python files

---

## 1. Objective

Build an autonomous system that accepts an MP4 screen recording as input, detects all personally identifiable information (PII) and protected health information (PHI), masks detected regions with blur or black boxes, and outputs a cleaned video. The user should be able to dump a video in and expect a clean video out with zero manual intervention.

---

## 2. Architecture

The pipeline supports three modes of operation, each building on the previous:

### Mode 1: Regex-Only (OCR Pipeline)

```
Input MP4
    |
    v
[1] Frame Extraction (OpenCV, configurable sample rate)
    |
    v
[2] OCR (EasyOCR, pixel-accurate bounding boxes)
    |
    v
[3] Regex Classification (14 pattern categories)
    |
    v
[4] IoU Temporal Tracking (associate regions across frames)
    |
    v
[5] Motion-Compensated Redaction (ORB features + affine transforms)
    |
    v
Output MP4 + JSON Detection Report
```

### Mode 2: OCR + Kimi Semantic (`--kimi`)

Same as Mode 1 but adds semantic classification after regex:

```
[3] Regex Classification
    |
    v
[4] Kimi K2.5 Semantic Classification (text-only, contextual PII)
    |
    v
[5] Union Merge (regex + Kimi, additive only)
    |
    v
[6] Tracking → [7] Motion-Compensated Redaction
```

### Mode 3: Vision-First (`--vision`)

Bypasses OCR entirely — sends frame images directly to Kimi K2.5:

```
Input MP4
    |
    v
[1] Smart Frame Sampling (content change detection, not fixed interval)
    |
    v
[2] Kimi K2.5 Vision API (sends JPEG frames, gets PII bounding boxes)
    |
    v
[3] IoU Temporal Tracking
    |
    v
[4] Motion-Compensated Redaction (ORB features + affine transforms)
    |
    v
Output MP4 + JSON Detection Report
```

### Key Design Decisions

- **EasyOCR for text detection**: Deterministic, pixel-accurate bounding boxes. Runs on CPU (GPU optional). Chosen over Tesseract for better accuracy on screen text.
- **Two-pass classification**: Regex handles structured patterns (emails, SSNs, phones). Kimi K2.5 handles semantic/contextual PII (person names, garbled text, contextual IDs).
- **Union merge strategy**: Either classifier flagging = redaction. Kimi can only ADD redactions, never remove regex-flagged ones. This ensures high recall at the cost of some over-redaction.
- **Prompt-based JSON from Kimi**: Moonshot's API does not support `response_format` with `json_schema` (GitHub Issue #96). We use a carefully crafted system prompt requesting JSON arrays and a multi-strategy parser (direct parse, strip markdown fences, regex extraction, individual object extraction).
- **IoU-based temporal tracking**: Associates PII regions across frames using Intersection-over-Union with text-based fallback matching for scroll scenarios.
- **ORB motion estimation**: Frame-to-frame global motion tracked via ORB feature matching + `cv2.estimateAffinePartial2D`. Captures scroll (translation) and zoom (uniform scale). Blur boxes follow content movement in real-time.
- **PII vs attributes distinction**: The Kimi prompt explicitly separates identifiers (names, SSNs, emails — REDACT) from attributes (medical conditions, medications, test results — SAFE). Data that describes a person but cannot identify them alone is not redacted.
- **Vision mode**: Kimi K2.5 is natively multimodal (trained on 15T mixed vision+text tokens, 92.3% OCRBench). Vision mode leverages this by sending full screenshot context instead of fragmented OCR text, eliminating the OCR bottleneck entirely.
- **Smart frame sampling**: Content change detection via normalized MSE between frames. Static screens produce 1 sample, scroll events produce more at transition points. Min/max FPS bounds guarantee coverage without over-sampling.

---

## 3. Development Timeline

### Commit 1: Initial Implementation (`783da78`)

Built the full MVP pipeline from scratch:
- `extractor.py` — Frame extraction with configurable sample FPS
- `ocr.py` — EasyOCR wrapper with lazy initialization
- `classifier.py` — 14 regex patterns covering emails, phones, SSNs, credit cards, API keys, passwords, IP addresses, dates of birth, medical record numbers, street addresses
- `tracker.py` — IoU-based cross-frame region tracking
- `redactor.py` — Blur and box redaction with OpenCV
- `pipeline.py` — 8-phase orchestrator
- `cli.py` — Full argparse interface
- `setup.py` — Package with entry point

### Commit 2: Kimi K2.5 Integration (`67f674e`)

Added semantic classification as a second pass:
- `config.py` — Multi-provider support (Moonshot, Together AI, OpenRouter), environment variable resolution
- `kimi_classifier.py` — API client with prompt-based JSON extraction, batching, retry with exponential backoff, multi-strategy response parsing
- Updated `classifier.py` with `merge_classifications()` and expanded PII categories
- Updated `pipeline.py` with phased execution: OCR -> regex -> Kimi -> merge -> track -> redact

### Commit 3: Performance Optimization (`74052ce`)

Three optimizations to Kimi API usage:
1. **Text deduplication**: Send each unique text string once, fan out verdicts
2. **Skip regex-flagged**: Don't re-classify items already caught by regex
3. **Parallel batching**: `ThreadPoolExecutor` with configurable concurrency

### Commit 4: Test Infrastructure (`d66049e`)

Synthetic test video generator producing a 10-second screen recording with three scenes containing diverse PII types.

### Session 2: Scroll/Zoom Tracking + PII Precision + Vision Mode

Three major improvements in one session:

**A. Motion-Compensated Redaction** — Blur boxes now follow content during scrolling and zooming. Uses ORB feature matching to estimate global frame-to-frame motion (translation + uniform scale), composing cumulative affine transforms. Delta transforms applied to known PII positions at OCR-sampled frames to compute position at any intermediate frame.

- `redactor.py` — Added `MotionEstimator` class with ORB detection, BFMatcher, `estimateAffinePartial2D`, cumulative transform composition
- `tracker.py` — Added `frame_positions` dict for per-frame position storage, text-based fallback matching for scroll scenarios, `get_position_at_frame()` interpolation

**B. PII Precision Tuning** — Tightened the Kimi system prompt to distinguish identifiers (data that can identify a specific person) from attributes (data that describes a person but cannot identify them alone).

- Removed catch-all "Any other personally identifiable or confidential information"
- Added explicit SAFE list: medical conditions, medications, test results, ages, job titles, allergy information, descriptive labels
- Result: `medical_id` regions dropped 31 → 3, `id_number` dropped 64 → 39, `person_name` dropped 55 → 39

**C. Vision-First Pipeline (`--vision` mode)** — New pipeline mode that sends frame images directly to Kimi K2.5's vision API instead of OCR → text classification. Eliminates the 252-second OCR step entirely.

- `vision_classifier.py` — Kimi K2.5 vision API client, base64 JPEG frame encoding, bbox response parsing, parallel frame classification
- `vision_sampler.py` — Smart frame sampling via content change detection (MSE-based), min/max FPS bounds
- `pipeline.py` — Added `run_vision_pipeline()` with smart sampling → vision classification → tracking → redaction
- `cli.py` — Added `--vision`, `--vision-quality`, `--vision-max-fps`, `--vision-min-fps`, `--vision-change-threshold` flags
- `config.py` — Added vision mode configuration fields

---

## 4. Blockers, Issues, and Resolutions

### Issue 1: Moonshot API Lacks `json_schema` Support

**Problem**: The Moonshot API (api.moonshot.ai) does not support OpenAI's `response_format` parameter with `json_schema` mode. This was documented in GitHub Issue #96 on the Moonshot repository. Without structured output, the model could return JSON wrapped in markdown fences, with trailing text, or in unpredictable formats.

**Resolution**: Built a multi-fallback JSON parser (`_parse_json_response`) with four strategies:
1. Direct `json.loads()` on the full response
2. Strip markdown code fences (` ```json ... ``` `)
3. Regex search for `[...]` array pattern
4. Extract individual `{...}` objects and reconstruct array

This handles every response format variation we encountered in testing.

### Issue 2: First Kimi API Run Timed Out

**Problem**: Initial batch size of 200 OCR items per API call caused timeouts at the default 60-second limit. The model needed more time to process large batches and generate 200 JSON verdict objects.

**Resolution**: Reduced batch size to 50 items per call and increased timeout to 120 seconds. All subsequent runs completed successfully with 200 OK responses on every batch.

### Issue 3: OCR Text Fragmentation Defeating Regex

**Problem**: EasyOCR splits rendered text across multiple bounding boxes based on visual spacing. This caused regex misses:
- `"Sarah Johnson"` detected as `"Name:"` and `"Sarah Johnson"` in separate boxes — regex can't match names anyway
- `"412-55-7890"` (SSN) OCR'd as `"412-55_7890"` — underscore instead of dash breaks the SSN regex
- `"sarah.johnson@email.com"` OCR'd as `"sarah johnson@email com"` — spaces break email regex
- `"4532-8821-0099-7766"` (credit card) OCR'd as `"4532-8821_0099_7766"` — OCR character substitution
- Street addresses split across 3+ boxes: `"1234"`, `"Oak"`, `"Drive"`

**Resolution**: This is the core reason the Kimi K2.5 semantic classifier was added. Kimi understands context — it knows `"412-55_7890"` next to an `"SSN"` label is a social security number regardless of character garbling. It knows `"Margaret E Thompson"` is a person name. It catches what regex structurally cannot. The vision mode further eliminates this issue by letting Kimi see the full screenshot.

### Issue 4: Kimi API Performance (98 Minutes for 32s Video)

**Problem**: A 32-second real screen recording produced 9,423 OCR text regions across 64 sampled frames. Sending all of them to Kimi in sequential batches of 50 required 189 API calls at ~30 seconds each = 98 minutes. Cost: $1.22.

**Root cause analysis**:
1. Same text appears identically in 30+ consecutive frames (static screen content)
2. Regex-flagged items sent to Kimi unnecessarily (union merge makes this redundant)
3. All API calls sequential (no parallelism)

**Resolution**: Three optimizations implemented:
- **Text deduplication**: 9,423 boxes → 612 unique texts (93% reduction)
- **Skip regex-flagged**: 459 items excluded before Kimi
- **Parallel batching**: 4 concurrent API calls via ThreadPoolExecutor

Result: 189 batches → 13 batches, 98 min → 1.6 min (60x speedup), $1.22 → ~$0.08.

### Issue 5: macOS Screen Recording Codec Incompatibility

**Problem**: macOS screen recordings (.mov files) use HEVC (H.265) codec which OpenCV's default build cannot read. `cv2.VideoCapture` returned `isOpened() = False`.

**Resolution**: Pre-convert with FFmpeg: `ffmpeg -i input.mov -c:v libx264 -preset fast -crf 18 -an output.mp4`. This transcodes to H.264 which OpenCV handles natively.

### Issue 6: Blur Boxes Drift During Scroll and Zoom

**Problem**: PII positions are only known at OCR-sampled frames (2fps = every ~27 frames at 53fps video). Between samples, linear interpolation of (x, y, w, h) doesn't match actual scroll dynamics or zoom transforms. During scrolling, blur boxes would momentarily drift off sensitive text, briefly exposing PII.

**Root cause**: Scroll is a content-dependent translation — speed varies with scroll acceleration. Zoom scales coordinates around a center point. Interpolating x/y/w/h independently is geometrically wrong for both cases.

**Resolution**: Added `MotionEstimator` class using ORB feature matching + `cv2.estimateAffinePartial2D`. This 4-DOF affine model (translation + uniform scale) captures both scroll and zoom. Frame-to-frame motion is composed into cumulative transforms. For each PII region, the delta transform between the nearest anchor frame and the current frame is computed and applied to the known position, yielding the correct position at every frame.

Performance: ~12ms/frame additional at 1/4 resolution (864x558), negligible impact on total pipeline time.

### Issue 7: Over-Redaction of Medical Conditions and Attributes

**Problem**: The original Kimi prompt included a catch-all `"Any other personally identifiable or confidential information"`. This caused Kimi to flag medical conditions ("Type 2 Diabetes"), medications ("Lisinopril 10mg"), test results ("Positive", "Negative"), ages ("37 years old"), and descriptive labels ("Fingerprint (Right Index)"). These are confidential but NOT identifiers — they cannot identify a specific individual alone.

**Resolution**: Tightened the Kimi system prompt with a key principle: "REDACT data that IDENTIFIES a person. Mark as SAFE data that DESCRIBES a person but cannot identify them alone." Added explicit SAFE list for medical conditions, medications, test results, ages, job titles, allergy information. Removed the catch-all.

Result on the confidential test recording:

| Category | Before | After | Change |
|----------|--------|-------|--------|
| medical_id regions | 31 | 3 | -28 |
| id_number regions | 64 | 39 | -25 |
| person_name regions | 55 | 39 | -16 |

Items correctly dropped: "Type 2 Diabetes", "Hypertension", "Anxiety disorder", "Lisinopril 10mg", "Metformin 500mg 2x daily", "Positive", "Negative", "37 years old", "Fingerprint (Right Index)", "EpiPen carried for shellfish", "Plano Elementary School", "Google (2017-2023)".

---

## 5. Test Results

### Confidential Screen Recording (31.1s, 3456x2234, 53fps)

A screen recording of a confidential employee document containing diverse PII: names, SSNs, emails, phone numbers, addresses, employee IDs, medical records, bank accounts.

#### Three-Way Pipeline Comparison

| Metric | Regex Only | OCR + Kimi | Vision |
|--------|-----------|------------|--------|
| Total detections | 306 | 738 | 849 |
| Tracked regions | 74 | 183 | 180 |
| Frames analyzed | 62 | 62 | 63 |
| OCR time | 252s | 252s | **0s** (skipped) |
| Kimi API time | 0s | 106s | 214s |
| Redaction time | 34s | 45s | 39s |
| **Total time** | **~286s** | **~403s** | **~256s** |

#### Detection Coverage by Category (Tracked Regions)

| Category | Regex | OCR+Kimi | Vision |
|----------|-------|----------|--------|
| person_name | 0 | 39 | 39 |
| id_number | 0 | 39 | 45 |
| credential | 0 | 7 | 17 |
| kimi_pii | 0 | 9 | 21 |
| medical_id | 0 | 3 | 7 |
| credit_card | 1 | 1 | 11 |
| email | 14 | 19 | 11 |
| phone | 32 | 37 | 14 |
| ssn | 16 | 16 | 6 |
| street_address | 7 | 9 | 9 |
| ip_address | 4 | 4 | 0 |

**Key insights**:
- **Vision mode is fastest overall**: Eliminates the 252s OCR step, cutting total time by ~37% vs OCR+Kimi despite longer API calls.
- **Vision finds more credentials and IDs**: 17 credential regions vs 7, 45 id_number regions vs 39. Full visual context helps Kimi recognize IDs it missed as isolated text strings.
- **OCR+regex better at structured patterns**: Regex catches more SSNs (16 vs 6) and phones (32 vs 14) because it matches exact patterns regardless of visual layout. Vision mode relies on Kimi's judgment for these.
- **Complementary strengths**: The ideal approach may be combining both — vision for contextual detection + regex for pattern guarantees.

### Synthetic Test Video (10s, 1280x720, 30fps)

Three scenes: patient dashboard, email client, terminal with secrets.

| Metric | Regex-Only | Regex + Kimi |
|--------|-----------|--------------|
| Total detections | 56 | 179 |
| Unique PII items | 8 | 40 |
| Tracked regions | 8 | 40 |
| Processing time | ~19s | ~301s |

**Regex caught**: DOB, MRN, phone, email, API key, password, 2x IP address.

**Kimi additionally caught**: Person name ("Sarah Johnson"), SSN with OCR garble ("412-55_7890"), insurance ID, employee ID, credit card with garbled chars, fragmented email, JWT token, AWS key, DB connection string, address fragments, username in status bar.

---

## 6. Optimization Deep Dive

### Kimi API: Before vs After

**Before Optimization**:
```
9,423 OCR boxes
    |
    v (all sent to Kimi, sequentially)
189 API calls x ~30s each = 5,890s
    |
    v
1,232 detections
```

**After Optimization**:
```
9,423 OCR boxes
    |
    v (remove 459 regex-flagged)
8,964 non-flagged boxes
    |
    v (deduplicate by text)
612 unique texts (93% reduction)
    |
    v (13 batches, 4 parallel)
97.5s total
    |
    v (fan out verdicts to all 8,964 boxes)
1,372 detections
```

| Optimization | Items Reduced | Time Saved |
|-------------|--------------|------------|
| Text deduplication | 9,423 → 612 (93%) | ~95% of Kimi time |
| Skip regex-flagged | 612 → ~580 | ~5% additional |
| Parallel calls (4x) | Same items, 4x throughput | ~75% of remaining time |
| **Combined** | **9,423 → 612, 4x parallel** | **98.3% total (60x speedup)** |

### Vision Mode: Architectural Optimization

Vision mode eliminates the OCR bottleneck entirely:

```
                    OCR+Kimi Pipeline          Vision Pipeline
                    ─────────────────          ───────────────
Frame sampling:     Fixed 2fps (62 frames)     Smart sampling (63 frames)
OCR:                252s (CPU EasyOCR)         0s (skipped)
Classification:     0.01s regex + 106s Kimi    214s Kimi vision
Redaction:          45s                        39s
──────────────────────────────────────────────────────────────
Total:              ~403s                      ~256s (37% faster)
```

---

## 7. Current Limitations

1. **Output codec**: Uses `mp4v` (MPEG-4 Part 2) which produces larger files than H.264. FFmpeg post-processing would improve compression.
2. **No automatic MOV→MP4 conversion**: macOS HEVC recordings require manual FFmpeg conversion before processing.
3. **Vision bbox precision**: Kimi K2.5's bounding boxes are approximate — they may not tightly frame individual PII text. Future work could add OCR-based bbox refinement on vision-detected regions.
4. **Vision mode misses structured patterns**: Without regex, SSNs and phone numbers in non-obvious formats may be missed. A hybrid mode (vision + regex post-pass) would combine both strengths.
5. **Kimi API dependency**: Both Kimi and vision modes require internet access and API key. Regex-only mode works offline but misses person names and contextual PII.
6. **API cost for vision**: Vision mode sends full images (~300-500KB JPEG per frame), consuming more input tokens than text-only classification.

---

## 8. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Video I/O | OpenCV 4.8+ | Frame extraction and re-encoding |
| OCR | EasyOCR 1.7+ | Text detection with bounding boxes |
| Regex | Python `re` | Structured PII pattern matching |
| LLM (text) | Kimi K2.5 (Moonshot AI) | Semantic PII classification |
| LLM (vision) | Kimi K2.5 (Moonshot AI) | Visual PII detection with bounding boxes |
| Motion estimation | OpenCV ORB + RANSAC | Frame-to-frame scroll/zoom tracking |
| API Client | OpenAI Python SDK | OpenAI-compatible API calls |
| Parallelism | `concurrent.futures` | Parallel Kimi API calls |
| Codec conversion | FFmpeg | HEVC to H.264 pre-processing |

**Kimi K2.5 model details**: 1 trillion parameter MoE model, 32B active parameters. Released January 27, 2026. 92.3% OCRBench score (best-in-class). Native multimodal — trained on 15T mixed vision+text tokens. OpenAI-compatible API at `api.moonshot.ai/v1`. Multi-provider support: Moonshot (native), Together AI, OpenRouter.

---

## 9. File Structure

```
video-cleaner/
  video_cleaner/
    __init__.py           (3 lines)    Package init, version
    __main__.py           (5 lines)    python -m entry point
    cli.py                (255 lines)  Argument parsing, config wiring, mode routing
    config.py             (115 lines)  KimiConfig, KimiProvider, PipelineConfig
    extractor.py          (87 lines)   Frame extraction from video
    ocr.py                (123 lines)  EasyOCR wrapper
    classifier.py         (242 lines)  Regex patterns + merge logic
    kimi_classifier.py    (399 lines)  Kimi K2.5 text API client + dedup + parallel
    vision_classifier.py  (362 lines)  Kimi K2.5 vision API client + bbox parsing
    vision_sampler.py     (100 lines)  Smart frame sampling via content change detection
    tracker.py            (226 lines)  IoU + text-based cross-frame tracking
    redactor.py           (283 lines)  Blur/box redaction + ORB motion compensation
    pipeline.py           (487 lines)  Pipeline orchestrator (OCR + vision modes)
  tests/
    generate_test_video.py (91 lines)  Synthetic test video generator
  setup.py                (21 lines)   Package metadata + dependencies
```

Total: **2,687 lines of Python** (+815 from initial implementation).
