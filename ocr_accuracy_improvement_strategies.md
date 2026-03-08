# OCR Accuracy Improvement Strategies

## Current State (2026-03-08)

- **Per-digit accuracy**: ~99.7% on reference-visible frames
- **Classifier**: FastKNNClassifier, 67-dim structural features, k=7, weighted inverse-distance voting
- **Known confusions**: 8<->6 (most common remaining), D<->C, 4<->9
- **Post-hoc fixes applied**: `fix_address_trajectory.py` (Strategy 8: unified trajectory correction, 2,794 frames relocated across C/D, 4/9, 8/6 confusion pairs)
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
- **Swap detection**: Position-adaptive threshold `max(swap_magnitude // 2, 0x80)` — handles C/D swaps at any digit position:
  - Position 1 (`$0Cxxx` → `$0Dxxx`): magnitude `$1000`, threshold `$800`
  - Position 2 (`$10Cxx` → `$10Dxx`): magnitude `$100`, threshold `$80`
  - Position 3+ disabled by floor (magnitude `$10`, threshold `$80` > improvement) — too noisy
- **Results**: Coverage 93.7% → 94.7% → 95.1% (position-adaptive threshold recovered 7 additional `$10Dxx` addresses)
- **Key fixes**: `$0DB00`–`$0DC40` region correctly mapped to `$0Dxxx`; `$10D80`–`$10DF0` recovered from `$10Cxx` misreads

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
   - Position-adaptive threshold enables position-2 C/D swaps (`$10Cxx` → `$10Dxx`)
   - Coverage 93.7% → 94.7% → 95.1%

3. **Strategy 7** ✅ DONE
   - Full-video gap recovery (`fullvideo_gap_recovery.py`)
   - Recovered 115/270 missing addresses, coverage 94.7% → 96.2%
   - 155 remaining gaps are unrecoverable (video jumps over them)

4. **Strategy 8** ✅ DONE
   - Global address trajectory correction (`fix_address_trajectory.py`): unified replacement for fix_d_c_misread.py + fix_49_misread.py
   - Moved 2,794 frames across 587 address pairs (C/D, 4/9, 8/6 confusions)
   - Coverage 96.8% → 96.6% (net loss from removing incorrectly-populated addresses)

5. **All-FF duplicate detection fix** ✅ DONE (2026-03-08)
   - `is_frame_different()` now also compares address column (threshold 5,000) alongside byte data (threshold 20,000)
   - Catches scrolling through all-FF regions where byte data is identical but address changed
   - Pipeline re-run + post-processing: coverage **96.6% → 97.0%** (4,966/5,120), 154 lines still missing

5. **Strategy 3** (no retrain needed)
   - Add temporal consistency as a post-vote correction
   - Can be applied to existing extraction results

6. **Strategy 1** (no retrain needed)
   - Add confusion-aware voting penalties
   - Fine-tuning step after Strategies 2-4 establish a better baseline

7. **Strategy 5** (optional, based on remaining error analysis)
   - Only if edge-row errors remain significant after Strategies 2-4

---

## Strategy 7: Full-Video Frame Extraction for Gap Recovery ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07. Script: `fullvideo_gap_recovery.py`.

**Results**:
- Recovered **115 of 270** missing addresses
- Coverage improved from **94.7% → 96.2%** (4,923/5,120)
- **155 addresses remain unrecoverable** — the video jumps over them between consecutive frames
- Scanned 2,304 video frames across 7 search windows (563 unique frames, ~160s)
- Saved row crops and updated `crop_index.json` for review tool compatibility
- Video frames stored in separate `video_frames` array with `"v"`-prefixed dict keys (e.g., `"v35040"`) and crop PNGs named `frame_v35040.png`, distinct from extracted frames in `frames` array

**Approach** (as implemented):
1. Load `firmware_merged.txt` to find missing ROM addresses; group into gap regions
2. Use `crop_index.json` anchor data to estimate video frame ranges via linear interpolation
3. Read frames from `full_video.mp4` in memory (no full-frame PNGs), skip duplicates with `is_frame_different()`
4. Run `process_frame()` on each unique frame; also check C/D swap candidates for addresses with C or D digits
5. Save row crops (~13 KB each) to `crops/<addr>/`, update `crop_index.json` with readings/confidences (atomic writes via tmp+rename)
6. Update `extracted_firmware.txt`, rebuild downstream files, reset review state

**Frame mapping**: `video_frame = extracted_frame + 24629`. Video has hex data in frames 24400-44730. Frame numbering uses `"v"` prefix for video frames in `crop_index.json` keys/filenames (see `frame_utils.py`).

**Impact**: Very high. Recovered nearly half of all missing addresses with no OCR algorithm changes.

**Requires retrain**: No.

---

## Strategy 8: Global Address Trajectory Correction (Unified) ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07. Script: `fix_address_trajectory.py`. Supersedes `fix_d_c_misread.py` and `fix_49_misread.py`.

**Results**:
- Moved **2,794 frames** across **587 unique source→dest address pairs**
- Top confusion types corrected: 9→4 (792), D→C (719), 8→6 (394), C→D (291), plus compound swaps
- Coverage 96.8% → 96.6% (net loss: 71 incorrectly-populated addresses emptied, 5 new addresses created)
- **Idempotent**: re-running produces 0 additional moves

**How it works**:
1. **Anchor trajectory**: Builds per-frame-median trajectory from 1,019 anchor addresses (no C/D/4/9/8/6 digits) — 3,328 extracted + 92 video anchor points
2. **Breakpoint detection**: Finds 14 scroll-direction reversals using smoothed running median, creating 15 monotone segments
3. **Expected address estimation**: Inverse-distance-weighted median of nearby per-frame-median anchors (±500 frame radius) within the same segment
4. **Swap candidate generation**: For each non-anchor address, generates all combinations of C↔D, 4↔9, 8↔6 swaps at each digit position
5. **Selection criteria**: Candidate must (a) reduce distance to expected by ≥ adaptive threshold `max(swap_magnitude // 2, 0x80)` AND (b) reduce distance by ≥50% (prevents false moves when both original and candidate are far from expected)
6. **Execution**: Moves frames, crop PNGs, readings, confidences; recomputes byte consensus; rebuilds downstream files; resets review state

**Key design decisions**:
- 0↔8 excluded from swap map (too many false positives, not a documented confusion pair)
- Per-frame-median aggregation prevents bias from multiple anchor rows per frame
- Weighted median (not linear interpolation) avoids artifacts from video jumps between sparse anchor frames
- 50% relative improvement threshold prevents false moves in regions far from any anchor

**Where modified**: New file `fix_address_trajectory.py` — standalone post-hoc correction script.

**Impact**: Very high. Unified approach handles compound errors (e.g., C→D + 8→6 in same address) that piecemeal scripts missed.

**Requires retrain**: No. Post-hoc address correction only.

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
| `fix_address_trajectory.py` | Strategy 8: unified global trajectory address correction (C/D, 4/9, 8/6) |
| `fix_49_misread.py` | Prior art: post-hoc 4/9 address confusion fix (superseded by Strategy 8) |
| `fix_d_c_misread.py` | Prior art: post-hoc C/D address confusion fix (superseded by Strategy 8) |
| `fullvideo_gap_recovery.py` | Strategy 7: scan full video for gap addresses, save crops, update firmware |
| `ocr_9_4_confusion_fix_prompt.md` | Detailed 4/9 confusion analysis and root causes |
