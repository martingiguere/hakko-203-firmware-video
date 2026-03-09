# OCR Accuracy Improvement Strategies

## Current State (2026-03-09)

- **Per-digit accuracy**: ~99.7% on reference-visible frames
- **Classifier**: FastKNNClassifier, 67-dim structural features, k=7, weighted inverse-distance voting
- **Known confusions**: 8<->6 (most common remaining), D<->C, 4<->9
- **Post-hoc fixes applied**: `fix_address_trajectory.py` (Strategy 8: two-phase — Phase 1 manual trajectory plausibility moved 2,897 frames, Phase 2 confusion-pair refinement moved 107 frames; total 3,004 frames relocated)
- **Training**: 2-pass (Tesseract-labeled Pass 1, kNN-detected Pass 2 with 80 frames), max 500-800 samples/class
- **Coverage**: 4,951/5,120 (96.7%)

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
1. ~~**Preprocessing**: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each cell before feature extraction in `extract_features()`.~~ **REJECTED** — see Option A results below.
2. **Voting**: Track row position with each observation, down-weight top/bottom row observations (rows 0-1 and 11-12) in multi-frame voting.

### Option A: CLAHE Preprocessing — REJECTED (2026-03-09)

**Tested via** `test_clahe_ab.py`: A/B comparison on 20 reference-visible frames (15 train, 5 test, 896 test digits).

**Results**: CLAHE **significantly hurts** accuracy:
- Overall: 96.76% → 81.25% (**-139 digits, -15.5%**)
- Edge rows: 100% → 89.58% (-30 digits)
- Middle rows: 95.23% → 77.30% (-109 digits)
- 8/6 pair: 92.2% → 59.8% (-33 digits)
- D/C pair: 93.3% → 50.0% (-26 digits)
- '6' accuracy collapsed to 12% (misread as '0' or '8')
- '2' accuracy collapsed to 34% (misread as '8')

**Root cause**: CLAHE's local histogram equalization amplifies noise and distorts the subtle stroke-shape differences that the 67-dim structural features rely on. The existing min-max normalization already handles contrast differences adequately. CLAHE creates artifacts that make structurally distinct digits (e.g., 2 vs 8, 6 vs 0) appear similar in feature space.

**Conclusion**: Option A is not viable for this classifier architecture. The kNN's structural features are sensitive to stroke shape, not absolute contrast — CLAHE changes shapes, not just contrast.

### Option B: Edge-Row Vote Down-Weighting — REJECTED (2026-03-09)

**Tested via** `test_strategy5b.py`: Used existing classifier on 20 reference-visible frames (295 observations across 16 reference addresses). Tested edge weights 1.0, 0.75, 0.5, 0.25, 0.1, 0.0.

**Results**: No effect at any weight. Byte accuracy stayed at 94.53% (242/256) for all weights 0.1–1.0. At weight 0.0 (completely discard edge rows), accuracy dropped slightly to 94.14% (241/256, -1 byte).

**Why it doesn't help**: Only 4 of 16 reference addresses had mixed edge+middle observations. With ~20 observations per address, middle-row votes already dominate the consensus — edge-row votes aren't numerous or wrong enough to swing the outcome.

**Conclusion**: Strategy 5 is closed. Neither option improves accuracy. The remaining errors are not caused by edge-row contrast issues influencing the vote; they stem from fundamental kNN separation limits on the confusion pairs.

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

4. **Strategy 8** ✅ DONE (updated 2026-03-09)
   - Two-phase address trajectory correction (`fix_address_trajectory.py`):
     - **Phase 1** (new): Manual trajectory plausibility check using `manual_trajectory.py` — detects ALL misassigned frames (not just confusable chars), moved 2,897 frames (837 re-read, 2,060 trajectory fallback). Only active where waypoint spacing ≤ 500 frames.
     - **Phase 2**: Confusion-pair refinement (C/D, 4/9, 8/6) — moved 107 additional frames after Phase 1 cleanup
   - Total: 3,004 moves. Coverage 97.0% → 96.7% (net loss from removing incorrectly-populated addresses)
   - Fixed known F6419 misassignment: `$03E90` → `$05E70`

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

7. **Strategy 5** ❌ REJECTED (2026-03-09)
   - Option A (CLAHE): -15.5% accuracy, destroys structural features
   - Option B (edge-row down-weighting): no effect, middle rows already dominate votes

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

## Strategy 8: Two-Phase Address Trajectory Correction ✅ IMPLEMENTED

**Status**: Implemented 2026-03-07, updated 2026-03-09 with Phase 1 (manual trajectory). Script: `fix_address_trajectory.py` + `manual_trajectory.py`. Supersedes `fix_d_c_misread.py` and `fix_49_misread.py`.

**Results** (2026-03-09 run):
- **Phase 1**: 2,897 frames moved (837 via constrained re-read, 2,060 via trajectory fallback)
- **Phase 2**: 107 additional frames moved (newly detectable after Phase 1 cleanup)
- **Total**: 3,004 moves across 776 source / 1,076 destination addresses
- Coverage 97.0% → 96.7% (net loss from removing incorrectly-populated addresses)
- Fixed known F6419 misassignment: `$03E90` → `$05E70`

**How it works**:

*Phase 1 — Manual trajectory plausibility check (new)*:
1. **Manual trajectory**: `manual_trajectory.py` stores 60+ manually confirmed waypoints (frame, top_address) covering F1–F20070. `interpolate_trajectory(frame)` returns expected (top, bottom) via piecewise linear interpolation.
2. **Plausibility check**: For each (addr, frame) pair in crop_index, check if addr falls within [expected_top - margin, expected_bottom + margin]. Margin = 0x300 (48 rows). Only operates where waypoint spacing ≤ 500 frames — Pass 1 (F1–F1728, spacing 1,727) is skipped because linear interpolation is unreliable with only 2 waypoints.
3. **Constrained re-read**: For each implausible frame, reload the PNG and re-OCR addresses with a trajectory-constrained filter (only keep reads near the expected address). `validate_address_sequence()` then picks the best anchor from filtered reads.
4. **Trajectory fallback**: If re-read fails, assign addresses based on trajectory interpolation. No frames are dropped — maximum data recovery.

*Phase 2 — Confusion-pair refinement (existing)*:
1. **Anchor trajectory**: Builds per-frame-median trajectory from anchor addresses (no C/D/4/9/8/6 digits)
2. **Breakpoint detection**: Finds scroll-direction reversals using smoothed running median, creating monotone segments
3. **Expected address estimation**: Inverse-distance-weighted median of nearby anchors (±500 frame radius) within same segment
4. **Swap candidate generation**: For each non-anchor address, generates all combinations of C↔D, 4↔9, 8↔6 swaps
5. **Selection criteria**: Candidate must reduce distance to expected by ≥ adaptive threshold AND ≥50%

*Shared execution*:
6. **Execution**: Moves frames, crop PNGs, readings, confidences; recomputes byte consensus; rebuilds downstream files; resets review state

**Key design decisions**:
- Phase 1 only applies where manual trajectory waypoints are dense (spacing ≤ 500 frames) — avoids false positives from interpolation error in sparse regions
- Phase 1 catches ALL misassignments (not just confusable chars) — e.g., `$03E90` → `$05E70` where no characters are confusable
- Phase 2 runs after Phase 1 on the corrected crop_index, catching subtle confusion-pair errors Phase 1's margin can't detect
- 0↔8 excluded from swap map (too many false positives)
- `--phase1-only` and `--phase2-only` flags for selective execution

**Where modified**: `fix_address_trajectory.py` (added Phase 1 functions + modified main flow), new file `manual_trajectory.py`.

**Impact**: Very high. Phase 1 fixes gross misassignments that were invisible to the confusion-pair approach.

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
| `fix_address_trajectory.py` | Strategy 8: two-phase address correction (manual trajectory + confusion pairs) |
| `manual_trajectory.py` | Manually confirmed scroll trajectory waypoints + `interpolate_trajectory()` |
| `fix_49_misread.py` | Prior art: post-hoc 4/9 address confusion fix (superseded by Strategy 8) |
| `fix_d_c_misread.py` | Prior art: post-hoc C/D address confusion fix (superseded by Strategy 8) |
| `fullvideo_gap_recovery.py` | Strategy 7: scan full video for gap addresses, save crops, update firmware |
| `ocr_9_4_confusion_fix_prompt.md` | Detailed 4/9 confusion analysis and root causes |
