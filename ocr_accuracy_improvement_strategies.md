# OCR Accuracy Improvement Strategies

## Current State (2026-03-07)

- **Per-digit accuracy**: ~99.7% on reference-visible frames
- **Classifier**: FastKNNClassifier, 67-dim structural features, k=7, weighted inverse-distance voting
- **Known confusions**: 8<->6 (most common remaining), D<->C, 4<->9
- **Post-hoc fixes applied**: `fix_49_misread.py` (826 frames relocated), `fix_d_c_misread.py` (Phase 1 neighbor + Phase 2 monotonicity with position-adaptive threshold)
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

3. **Strategy 7 — Full-Video Frame Extraction for Gap Recovery** ⬅️ NEXT
   - See details below
   - Highest expected impact: purely additive, targets all 270 missing addresses
   - No OCR changes needed — uses existing kNN classifier on previously unprocessed frames

4. **Strategy 3** (no retrain needed)
   - Add temporal consistency as a post-vote correction
   - Can be applied to existing extraction results

5. **Strategy 1** (no retrain needed)
   - Add confusion-aware voting penalties
   - Fine-tuning step after Strategies 2-4 establish a better baseline

6. **Strategy 5** (optional, based on remaining error analysis)
   - Only if edge-row errors remain significant after Strategies 2-4

---

## Strategy 7: Full-Video Frame Extraction for Gap Recovery ⬅️ NEXT

**Problem**: The `frames/` directory contains only 20,070 pre-extracted frames, but the full video (`full_video.mp4`) has 93,093 frames at 30fps. Approximately 73,000 video frames are never processed by the pipeline. The mapping between extracted frame numbers and video frame numbers is non-linear and unpredictable. Many of the 270 missing addresses are visible in these unprocessed frames.

**Evidence**: Address `$0CDC0` was missing because no pre-extracted frame captured it with a correct address OCR. But the data was clearly present in the full video at YouTube timestamp 21:03 (video frames 37910-37916). Running the pipeline's kNN classifier on raw video frames successfully recovered the data in one session.

**Approach**:
1. Build a video-frame→address mapping from `crop_index.json` anchor data (frame numbers that map to known addresses) to estimate which video timestamps correspond to each gap region
2. For each gap, extract frames from `full_video.mp4` at the estimated timestamps using OpenCV (`cap.set(CAP_PROP_POS_FRAMES, n)`)
3. Run the existing kNN classifier (`fast_knn_classifier.npz`) on each frame via `process_frame()` — frames are 1080x1920 grayscale, matching the pre-extracted frames exactly
4. Collect observations for missing addresses and add to `extracted_firmware.txt`
5. Run `postprocess_firmware.py` to merge

**Where to modify**: New script (e.g., `extract_from_video.py`) or extend `extract_pipeline.py` with a `--video` mode.

**Impact**: Very high. Targets all 270 missing addresses without any OCR algorithm changes. Purely additive — no risk of corrupting existing data.

**Requires retrain**: No.

**Caution**: The OCR may misclassify addresses containing both C and D (as seen with `$0CDC0` → `$0CCC0`). Cross-check OCR'd addresses against the expected scroll position. The monotonic scroll assumption and anchor interpolation from `fix_d_c_misread.py` can help validate.

**Estimated video-to-address mapping**: The video scrolls through `$00000`–`$13FFF` over ~51 minutes (~3,100 seconds). Rough rate: ~26 addresses/second, but non-uniform (pauses, speed variations). The existing `crop_index.json` maps extracted frame numbers to addresses; these can be converted to approximate video timestamps using the non-linear frame mapping.

**Video resolution**: The source YouTube video maxes out at 1080p (format 137, H.264). The existing `full_video.mp4` and pre-extracted frames are already at maximum available quality (1920×1080). No higher resolution is available.

**Disk space and review tool integration**:
- Full-frame PNGs are ~2.1 MB each (20,070 existing frames = 40 GB). Extracting all 73,000 remaining frames as PNGs would require ~150 GB — not feasible (38 GB free).
- **However**, Strategy 7 only needs ~1,400 targeted frames (~2.8 GB if saved as full PNGs). This fits comfortably.
- **Recommended approach**: Read frames from `full_video.mp4` in memory via OpenCV (no full-frame PNGs), but save row-level crop PNGs to `crops/<addr>/` and update `crop_index.json` with readings and confidences. This preserves review tool compatibility:
  - Row crops are ~13 KB each. Estimated ~1,400 frames × 14 rows = ~250 MB of crops.
  - `crop_index.json` entries enable the review tool to show per-frame readings, confidence scores, and visual crops for human verification.
  - The review tool's `/api/frame/<n>` endpoint won't have full-frame PNGs for Strategy 7 frames, but this endpoint is rarely used — the row crops at `/api/crop/<addr>/<frame>` are the primary visual reference.
- **Frame numbering**: Strategy 7 frames come from video frame numbers (0–93092), not the pre-extracted frame numbers (1–20070). The crop filenames and crop_index entries should use the video frame numbers to avoid collisions (existing crops use extracted frame numbers which max out at 20070).

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
