# OCR Accuracy Improvement Strategies

## Current State (2026-03-07)

- **Per-digit accuracy**: ~99.7% on reference-visible frames
- **Classifier**: FastKNNClassifier, 67-dim structural features, k=7, weighted inverse-distance voting
- **Known confusions**: 8<->6 (most common remaining), D<->C, 4<->9
- **Post-hoc fixes applied**: `fix_49_misread.py` (826 frames relocated), `fix_d_c_misread.py` (7,514 frames relocated — Phase 1 neighbor + Phase 2 monotonicity)
- **Training**: 2-pass (Tesseract-labeled Pass 1, kNN-detected Pass 2 with 80 frames), max 500-800 samples/class

---

## Strategy 1: Confusion-Aware Per-Byte Voting Penalties

**Problem**: Multi-frame voting (`extract_pipeline.py:697-725`) weights by `byte_conf * addr_conf` but doesn't account for known confusion pairs. When `8` and `6` are close in feature space, both get high confidence -- the vote picks whichever appears more, which may be wrong if training data is imbalanced.

**Approach**: In the voting step, when the top two candidates are a known confusion pair (8/6, C/D, 4/9):
- Require a wider margin for the winner to be accepted
- Apply a discriminative check (e.g., center-bar strength for 8 vs 6)
- Alternatively, add a second-pass "confusion audit" that flags and re-examines ambiguous bytes

**Where to modify**: `run_extraction()` voting loop in `extract_pipeline.py:697-725`

**Impact**: Medium. Targets residual ~0.3% error on data bytes.

**Requires retrain**: No.

---

## Strategy 2: Targeted Discriminative Features for 8<->6 ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07. Added 3 features to `extract_features()` (64→67 dim):
1. **Upper-right quarter density**: `mean(ink[:h//4, w//2:])` — `8` has ink here (upper loop), `6` doesn't
2. **Bottom-left vs top-right asymmetry**: `mean(ink[y23:, :w//3]) - mean(ink[:y3, 2*w//3:])` — large positive for `6`, near-zero for `8`
3. **Right-side vertical balance**: `mean(ink[:mid_y, x23:]) - mean(ink[mid_y:, x23:])` — negative for `6` (bottom-heavy right side), near-zero for `8`

**Where modified**: `extract_features()` in `template_matcher.py`, after the 3x3 grid block.

**Impact**: High. Directly targets the most common remaining confusion pair.

**Requires retrain**: Yes (`--rebuild`). Invalidates `fast_knn_classifier.npz`.

---

## Strategy 3: Temporal Consistency Voting (Sliding Window)

**Problem**: Each frame is processed independently, then votes are aggregated globally per address. But consecutive frames showing the same address should agree. If frame N reads `A6` and frames N-1, N+1, N+2 all read `A8` for the same byte, `A8` is almost certainly correct.

**Approach**: Before global voting in `run_extraction()`, group observations by address and sort by frame number. Apply a local sliding window: if a reading disagrees with its temporal neighbors (same address, consecutive frames) and the disagreement involves a known confusion pair, override with the neighbors' consensus.

**Where to modify**: New step between frame processing and global voting in `extract_pipeline.py:697`, or as a post-processing pass on `all_observations`.

**Impact**: High for bytes seen in only a few frames. The temporal signal is currently wasted by global voting.

**Requires retrain**: No.

---

## Strategy 4: Increase Training Sample Diversity and Balance ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07.
- Pass 2 sampling increased from 30→80 frames for broader diversity
- Added per-class balance reporting with warnings for classes below 100 samples

**Where modified**: `extract_pipeline.py` — Pass 2 frame sampling loop and `FastKNNClassifier.build_from_samples()`.

**Impact**: Medium-high. Addresses root cause of class imbalance rather than patching symptoms post-hoc.

**Requires retrain**: Yes (`--rebuild`).

---

## Strategy 5: Row-Position-Aware Classification

**Problem**: Top rows (0-1) have reduced contrast (~200 vs ~255 for middle rows), per `ocr_9_4_confusion_fix_prompt.md`. The classifier treats all row positions equally, but edge rows are systematically harder to classify correctly.

**Approach**: Two options:
1. **Preprocessing**: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each cell before feature extraction in `extract_features()`. Currently uses simple min-max normalization (line 125-129) which doesn't recover lost contrast.
2. **Voting**: Track row position with each observation, down-weight top/bottom row observations (rows 0-1 and 11-12) in multi-frame voting.

**Where to modify**:
- Option A: `extract_features()` in `template_matcher.py:109-186` (add CLAHE before normalization)
- Option B: `process_frame()` in `extract_pipeline.py:463-506` (record row position) and voting loop at line 697

**Impact**: Medium. Specifically helps the errors concentrated in edge rows.

**Requires retrain**: Option A requires retrain. Option B does not.

---

## Strategy 6: Global Monotonicity Address Correction ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07. Added as Phase 2 in `fix_d_c_misread.py`:
- **Anchor trajectory**: Built from 57,745 addresses with no C/D digits (unambiguous ground truth)
- **Interpolation**: For each C/D-containing address, estimate expected address via inverse-distance weighted median of nearby anchors (±500 frames)
- **Swap detection**: If a C↔D swap brings the address ≥`$800` closer to expected, generate a move
- **Results**: 7,325 frames moved (vs 388 from Phase 1 alone), coverage 93.7% → 94.7%
- **Key fix**: `$0DB00`–`$0DC40` region (frames 14290–14340) now correctly mapped to `$0Dxxx` addresses

**Where modified**: `fix_d_c_misread.py` — added `build_anchor_trajectory()`, `estimate_expected_address()`, `detect_monotonicity_moves()`, `deduplicate_moves()`, and updated `main()` to run both phases.

**Impact**: Very high. Recovered 208 lines in the `$0D050`–`$0DF70` range, reducing the `$0D` gap from 242 to ~34 lines.

**Requires retrain**: No. Post-hoc address correction only.

---

## Recommended Implementation Order

1. **Strategy 2 + Strategy 4** ✅ DONE
   - Added 3 discriminative features for 8<->6 (64→67 dim)
   - Expanded Pass 2 training from 30→80 frames
   - Added per-class balance warnings

2. **Strategy 6** ✅ DONE
   - Anchor-based monotonicity correction (Phase 2 in `fix_d_c_misread.py`)
   - Recovered 208 lines in `$0D050`–`$0DF70`, coverage 93.7% → 94.7%

3. **Strategy 3** (no retrain needed)
   - Add temporal consistency as a post-vote correction
   - Can be applied to existing extraction results

4. **Strategy 1** (no retrain needed)
   - Add confusion-aware voting penalties
   - Fine-tuning step after Strategies 2-4 establish a better baseline

5. **Strategy 5** (optional, based on remaining error analysis)
   - Only if edge-row errors remain significant after Strategies 2-4

---

## Key Constants Reference

```
ADDR_X_START = 289          # x-pixel of first address digit
address_char_spacing = 13.92 # pixels between address digit centers
CELL_W = 14, CELL_H = 28   # character cell dimensions
first_row_center_y = 272    # y-pixel of row 0 center
row_height = 28             # pixels between row centers
visible_rows = 13           # rows per frame
Feature dim = 67            # after Strategy 2 (was 64)
k = 7                       # kNN neighborhood size
max_per_class = 500-800     # training sample cap
```

## Key Files

| File | Relevance |
|------|-----------|
| `extract_pipeline.py` | FastKNNClassifier, voting, frame processing |
| `template_matcher.py` | `extract_features()` (67-dim), `extract_cell()`, calibration |
| `grid_calibration.json` | Grid geometry constants |
| `fast_knn_classifier.npz` | Trained kNN model (invalidated by feature changes) |
| `fix_49_misread.py` | Prior art: post-hoc address confusion fix |
| `fix_d_c_misread.py` | Post-hoc address confusion fix: Phase 1 (±10 neighbor heuristic) + Phase 2 (anchor-based monotonicity for systematic blocks) |
| `ocr_9_4_confusion_fix_prompt.md` | Detailed 4/9 confusion analysis and root causes |
