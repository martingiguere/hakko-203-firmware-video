# Investigation: Using Reference PNG for OCR Training

## Status: TESTED — Domain gap too large, not viable

> **Conclusion (2026-03-07):** Phase 1 was fully implemented and tested.
> The domain gap between reference screenshot and video frames is too large
> for reference-based training to help. **Do not revisit** unless the feature
> extraction approach fundamentally changes (e.g., domain-adaptive normalization).

## What it offers

The reference PNG (`reference/reference_screenshot.png`) contains 256 bytes across 16 rows — that's **512 hex digit samples** with perfect ground truth labels from the transcription. Add address digits and you get ~672 total samples.

## Phase 1 Results (A/B Test)

Implemented in `extract_reference_samples.py` and `test_reference_classifier.py`.

### Test setup
- Extracted 512 digit cells from reference PNG, resized to 14x28 with cubic interpolation
- Built separate reference-only FastKNNClassifier (`ref_classifier_data.npz`)
- Tested on 608 digit cells from 4 held-out video frames showing reference addresses
- Compared three configs: video-only (A), reference-only (B), ensemble 70/30 (C)

### Results

| Config | Accuracy | Notes |
|--------|----------|-------|
| A: Video-only | **99.7%** (606/608) | Only 2 errors (8->6, D->C) |
| B: Reference-only | **1.8%** (11/608) | Classifies nearly everything as '7' |
| C: Ensemble 70/30 | **94.7%** (576/608) | Ensemble actively hurts accuracy |

### Why reference-only fails so badly

The reference-only classifier achieves only 1.8% accuracy on video frames — worse than random (6.25% for 16 classes). It classifies almost every digit as '7'. This indicates the feature vectors from resized reference cells occupy a completely different region of feature space than video cells, despite identical resize dimensions. The domain gap manifests in:

- Anti-aliasing differences (crisp GUI rendering vs camera blur)
- Contrast/gamma (pure black-on-white screenshot vs gray-on-gray video)
- Sub-pixel positioning and interpolation artifacts from resize

### Why ensemble hurts

Even with 30% weight, the reference classifier's confident-but-wrong predictions corrupt the video classifier's correct predictions. The ensemble drops from 99.7% to 94.7% — a net loss of 32 correct classifications.

## Phase 2 verdict

**Do not pursue.** The domain gap is too fundamental for sample mixing to help. Reference samples would be treated as outliers/noise by the kNN classifier, degrading accuracy rather than improving it.

## What the reference data IS useful for

- **Ground truth validation** — already used via `load_reference()` and `find_rows_with_known_data()` to validate OCR accuracy and build training labels from video frames showing reference addresses
- **Post-hoc correction** — reference data is overlaid directly onto extracted firmware as guaranteed-correct bytes (see `run_extraction()`)
- **These existing uses are the right approach** — they avoid the domain gap problem entirely

## Original analysis

### The problems (confirmed by testing)

1. **Feature vector incompatibility (solved)** — Resize to 14x28 before feature extraction. This works mechanically but doesn't solve the domain gap.

2. **Domain mismatch (confirmed as fatal)** — The reference is a native Windows GUI screenshot; video frames are camera captures of a monitor. The feature space distributions do not overlap.

3. **Imbalanced distribution (moot)** — The reference data is dominated by `4E FC 00 00` patterns, but this is irrelevant given the domain gap makes all reference samples unusable.

## Files

| File | Purpose |
|------|---------|
| `extract_reference_samples.py` | Extracts 512 labeled digit cells from reference PNG, saves `ref_classifier_data.npz` |
| `test_reference_classifier.py` | A/B test comparing video-only vs reference-only vs ensemble |
| `ref_classifier_data.npz` | Reference-only classifier (512 samples, 16 classes) |
