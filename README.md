# Hakko FM-203 Firmware Extraction from Video

Extract the firmware of a **Hakko FM-203** soldering station from a YouTube video of the Xeltek SuperPro 6100N programmer displaying the hex dump of the MCU's flash memory.

## Target

| Parameter | Value |
|-----------|-------|
| MCU | Renesas R5F21258SNFP (R8C/24, 52-pin TQFP) |
| Program ROM | 64 KB (`$04000`-`$13FFF`) in two 32 KB erase blocks |
| Programmer | Xeltek SuperPro 6100N with DX3063 adapter |
| Video | 20,070 frames of scrolling hex dump |
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
4. **Post-processing** -- Reference data overlay, OCR misread fixes (4/9, C/D swaps), gap filling, binary output.

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
| `fix_d_c_misread.py` | Post-hoc fix for C/D OCR address confusion |
| `diagnose.py` | Project status diagnostic |

## Review Tool

`firmware_review_tool/` contains a Flask web app for human-assisted byte verification. Shows frame crops alongside kNN readings, allows manual correction, and tracks review progress.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Tesseract OCR installed (`apt install tesseract-ocr`).

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
