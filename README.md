# Hakko FM-203 Firmware Extraction from Video

Extract the firmware of a **Hakko FM-203** soldering station from a YouTube video of the Xeltek SuperPro 6100N programmer displaying the hex dump of the MCU's flash memory.

## Target

| Parameter | Value |
|-----------|-------|
| MCU | Renesas R5F21258SNFP (R8C/24, 52-pin TQFP) |
| Program ROM | 64 KB (`$04000`-`$13FFF`) in two 32 KB erase blocks |
| Programmer | Xeltek SuperPro 6100N with DX3063 adapter |
| Video | 93,093 frames at 30fps (20,070 pre-extracted for pipeline; full video available for gap recovery) |
| Reference | 256 verified bytes at `$0FF70`-`$1006F` from screenshot |

## Pipeline

```
download_video.py        Download video & extract frames
        |
calibrate_grid.py        Detect cell geometry (14x28px cells, byte positions)
        |
extract_pipeline.py      Main extraction: kNN OCR + multi-frame voting
        |
postprocess_firmware.py  Merge, fill gaps, produce binary
        |
firmware_review_tool/    Flask app for human-assisted review
```

### How it works

1. **Training** -- Tesseract reads addresses on frames showing the known reference region. Matching rows provide labeled digit samples for a kNN classifier (67-dim structural features, including 8↔6 discriminative features).
2. **Extraction** -- Every frame is OCR'd: read 10-digit address, read 16 hex bytes. Duplicate/transitional frames are skipped.
3. **Voting** -- Each address+byte gets multiple observations across frames. Weighted majority vote picks the best reading.
4. **Post-processing** -- Reference data overlay, OCR misread fixes (4/9 swap, C/D two-phase correction), gap filling, binary output.

## Scripts

| Script | Purpose |
|--------|---------|
| `download_video.py` | Download video and extract PNG frames |
| `calibrate_grid.py` | Calibrate character grid geometry -> `grid_calibration.json` |
| `template_matcher.py` | Cell extraction, feature extraction, reference data loading |
| `extract_pipeline.py` | Full extraction pipeline with FastKNNClassifier |
| `postprocess_firmware.py` | Merge extractions, produce firmware binary |
| `analyze_reference.py` | Verify reference transcription against screenshot |
| `measure_reference_geometry.py` | Measure reference screenshot geometry (row/byte positions) |
| `fix_49_misread.py` | Post-hoc fix for 4/9 OCR address confusion |
| `fix_d_c_misread.py` | Post-hoc fix for C/D OCR address confusion (Phase 1: neighbor context + Phase 2: anchor monotonicity) |
| `fullvideo_gap_recovery.py` | Strategy 7: scan full video for gap addresses |
| `diagnose.py` | Project status diagnostic |

## Review Tool

`firmware_review_tool/` contains a Flask web app for human-assisted byte verification. Shows frame crops alongside kNN readings, allows manual correction, and tracks review progress.

## Setup

```bash
sudo apt install git-lfs tesseract-ocr
git lfs install
git lfs pull            # fetch crop_index.json and fast_knn_classifier.npz
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`crop_index.json` and `fast_knn_classifier.npz` are stored via Git LFS. A fresh clone with LFS support provides these automatically — no need for the 40 GB `frames/` directory or 70+ minutes of regeneration to use the review tool.

## Reference Data

`reference/reference_screenshot.png` -- a screenshot of the Xeltek UI showing 256 bytes (`$0FF70`-`$1006F`). Manually transcribed in `reference/reference_transcription.txt`. Used for:

- Training label generation (matching video frames to known addresses)
- Post-extraction overlay (guaranteed-correct bytes)
- Accuracy validation

## Dead Ends

### Reference PNG as OCR training data (tested, not viable)

Attempted using the 512 digit cells from the reference screenshot to train a separate classifier and ensemble it with the video classifier. The domain gap between a clean GUI screenshot and camera-captured video frames is too large -- the reference-only classifier achieved only 1.8% accuracy on video frames, and a 70/30 ensemble hurt accuracy (94.7% vs 99.7% video-only). See `reference_png_ocr_training_notes.md` for full results.

## Accuracy

The video-only kNN classifier achieves ~99.7% accuracy on reference-visible frames. Remaining confusions (8/6, D/C) are rare and handled by post-hoc correction scripts.

See `ocr_accuracy_improvement_strategies.md` for proposed strategies to push accuracy higher.

## Coverage

Current coverage: **4,923 / 5,120 addresses (96.2% automated)**, 155 missing lines.

### Post-extraction fixes applied

1. **C↔D address misread** (`fix_d_c_misread.py`): Two-phase correction using neighbor context (Phase 1) and anchor-based monotonicity (Phase 2). Recovered 215 lines in `$0D050`–`$0DF70` and 7 addresses in `$10D80`–`$10DF0`.

2. **Full-video gap recovery** (`fullvideo_gap_recovery.py`): Strategy 7 — scans `full_video.mp4` directly for addresses missing from the pre-extracted frames. The pipeline normally processes 20,070 pre-extracted frames, but the full video has 93,093 frames. By estimating which video timestamps correspond to each gap region and running the existing kNN classifier, this recovered **115 of 270 missing addresses**, improving coverage from 94.7% to 96.2%. The remaining 155 gaps are unrecoverable — the video jumps over those addresses between consecutive frames.

### Largest remaining gaps

1. **`$047E0`–`$04990`** (28 addrs) — Dense code in Block 1, video jumps over this region
2. **`$11D40`–`$11F50`** — Partially recovered, remaining addrs skipped by video
3. **Scattered single-address gaps** — 46 groups of 1-4 addresses each throughout ROM
