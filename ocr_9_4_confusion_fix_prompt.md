# Prompt: Fix 9↔4 OCR Address Confusion in Hakko FM-203 Firmware Extraction

## Context

This project extracts firmware from video frames of a Xeltek SuperPro 6100N "Edit Buffer" hex dump scrolling through memory for a Hakko FM-203 soldering station (Renesas R5F21258SNFP MCU, 80 KB buffer at $00000-$13FFF).

The extraction pipeline uses a kNN classifier on 14x28 pixel character cells with 67-dimensional feature vectors. Address reading is done by `read_address_from_row()` in `extract_pipeline.py`, with auto-correction in `auto_correct_address()`.

## The Bug

The kNN classifier systematically confuses digits **'9' and '4'** in address fields. This causes frames to be assigned to wrong addresses in `crops/crop_index.json`.

### Concrete example at address 0x04490

Address `04490` in crop_index.json has 13 frames that don't belong:
- **Frames 9825, 9827, 9828, 9829**: Actually show address `0x09490` (digit 6 of the 10-digit address `000009490x` is '9' misread as '4')
- **Frames 10021, 10029-10039**: Actually show address `0x09990` (digits 6 AND 7 are '9' misread as '4', making `0000004490` instead of `0000009990`)

### Root cause analysis (verified)

1. **Marginal kNN separation**: For frame 9825 row 1, digit 6 kNN votes are `9:0.53, 4:0.47` -- essentially a coin flip. Even a correctly-OCR'd '9' from the same frame is closer to the '4' centroid than the '9' centroid (distances 0.302 vs 0.312).

2. **Training data imbalance**: 523 training samples for '4' vs only 216 for '9', biasing kNN toward '4'.

3. **Reduced contrast on top rows**: Rows 0-1 in many frames have contrast ~200 vs ~255 for middle rows, degrading already-marginal digit distinction.

4. **`auto_correct_address()` (line 287-312)** only handles `8→0` and `9→1` corrections for leading digits. It has zero handling for 4↔9 confusion.

5. **`validate_address_sequence()` (line 382-432)** can anchor on wrong addresses when the first rows of a frame are misread.

6. **The problem is widespread**: The entire `04xxx` address block from frames 9700+ is likely misread `09xxx`. Many `044xx` addresses from frames in the 5xxx range also overlap with `094xx` frames, suggesting bidirectional confusion.

## Prior art

See `fix_d_c_misread.py` for an existing fix script that corrected a similar systematic D→C address misclassification. That script:
- Detected contamination boundaries from crop_index.json
- Moved frame data between address entries (including crop PNGs)
- Recomputed byte consensus
- Updated extracted_firmware.txt
- Rebuilt downstream files (postprocess, gap precompute)
- Reset review state for affected addresses

## Files to read before making changes

| File | Purpose |
|------|---------|
| `extract_pipeline.py` | Main extraction: `FastKNNClassifier`, `read_address_from_row()`, `auto_correct_address()`, `validate_address_sequence()`, `process_frame()` |
| `template_matcher.py` | Feature extraction: `extract_features()` (67-dim), `extract_cell()`, calibration constants (`CAL`, `ADDR_X_START=289`, `CELL_W=14`, `CELL_H=28`) |
| `grid_calibration.json` | Grid geometry: `address_char_spacing=13.92`, `row_height=28`, `first_row_center_y=272`, `visible_rows=13` |
| `crops/crop_index.json` | Frame assignments: `{addr: {frames: [...], readings: {...}, confidences: {...}}}` |
| `frame_moves.json` | Ledger of manual frame reassignments |
| `fix_d_c_misread.py` | Prior art for systematic address misclassification fix |
| `postprocess_firmware.py` | Downstream rebuild after corrections |
| `SPEC.md` | Hardware spec, address ranges, ROM layout |
| `reference/reference_transcription.txt` | Ground truth data for $0FF70-$1006F |

## Implementation approaches (pick any, in any order)

### Approach A: Intra-frame address majority voting (recommended first)

Modify `validate_address_sequence()` or add a post-processing step in `process_frame()`.

**Logic**: Each frame shows 13 visible rows. The addresses should form a contiguous sequence with 0x10 spacing. When most rows agree on the upper address nibbles (e.g., 10 out of 13 rows read `09xxx`), correct outlier rows that read `04xxx` by replacing '4' with '9' where it produces a consistent sequence.

**Key insight**: The confusion is specifically between '4' and '9'. When an address is inconsistent with the majority AND swapping 4↔9 in a specific digit makes it consistent, apply the swap.

**Where to add**: In `validate_address_sequence()` around line 395-432 in `extract_pipeline.py`. The anchor-based approach already tries this but doesn't account for 4↔9 specifically. Add logic: when `addr_diff > 0x08`, check if replacing any '4' digit with '9' (or vice versa) in the outlier address produces `expected_addr`.

### Approach B: Post-hoc crop_index.json cleanup script

Similar to `fix_d_c_misread.py`. Write a standalone script that:

1. Scans crop_index.json for addresses where frames appear in temporally inconsistent positions (e.g., `04xxx` addresses with frames from the 9700-10100 range, when the video should be in the `09xxx` region by then)
2. For each suspicious frame, determines the correct address by:
   - Looking at what addresses neighboring frames (+-5 frame numbers) are assigned to
   - Checking if replacing '4' with '9' in any digit of the address produces a plausible correction
3. Moves frames to corrected addresses (update crop_index.json entries, move crop PNG files)
4. Logs all moves to frame_moves.json
5. Rebuilds downstream files

**Temporal heuristic**: The video scrolls monotonically from $00000 to $13FFF across ~20,070 frames. By frame ~5000, we're past $04xxx. By frame ~9700, we're in the $09xxx range. Any `04xxx` address assigned to frames >9000 is almost certainly a misread `09xxx`.

### Approach C: Improve auto_correct_address()

Add 4↔9 correction rules to `auto_correct_address()` at line 287. The function already knows the valid address range ($00000-$13FFF) and the 10-digit format. New logic:

- Digit 5 (first significant digit after leading zeros): can only be '0' or '1'. Already handled.
- Digit 6: if it's '4', check if '9' is more consistent with the frame's address sequence context. The value range is $0xxxx to $1xxxx, so digit 6 can be 0-3 (for $0xxxx) or 0-3 (for $1xxxx). Wait -- digit 6 can actually be 0-F since address goes up to $13FFF. So '4' and '9' are both valid at this position. The correction needs context (other rows in the frame) rather than a static rule.

This approach alone is insufficient -- it needs to be combined with A or B for context.

### Approach D: Retrain classifier with balanced 9/4 samples

1. Load `fast_knn_classifier.npz`
2. Count samples per class (currently: '4'=523, '9'=216)
3. Upsample '9' training data by extracting more labeled '9' cells from frames where the address is known (e.g., addresses containing '9' that are confirmed correct by sequence consistency)
4. Optionally add synthetic augmentation (small shifts, noise)
5. Rebuild and save the classifier
6. Re-run extraction pipeline with `--rebuild` flag

### Approach E: Add targeted 4-vs-9 discriminative feature

In `extract_features()` in `template_matcher.py` (line 109-186), add one or two features that specifically distinguish '4' from '9':

- **Bottom-half ink distribution**: '9' has ink curving into the center-bottom; '4' has a vertical stroke on the right with an open bottom-left. Measure `ink[bottom_half, :center] - ink[bottom_half, center:]` (a '4' has more right-side ink in the bottom half).
- **Horizontal crossbar position**: '4' has a prominent horizontal bar in the middle; '9' has a more curved transition. The existing `center_bar` feature (line 159-163) partially captures this but isn't discriminative enough.

**Warning**: Adding features changes the feature dimension from 64 to 65+, which invalidates `fast_knn_classifier.npz`. The classifier must be retrained after this change (`--rebuild` flag).

### Approach F: Confidence-gated contextual override

In `read_address_from_row()` (line 315-342):
- After classifying all 10 address digits, check if any digit classified as '4' has confidence below 0.65
- If so, flag it as ambiguous and return both the '4' and '9' interpretations
- In `process_frame()`, use the address sequence context (other rows) to select the correct interpretation

## Testing

After any fix, verify:
1. `crops/crop_index.json` address `04490` no longer contains frames 9825-9829 or 10021-10039
2. Address `09490` gains frames 9825-9829 (if not already there via other rows)
3. Address `09990` gains frames 10021-10039 (if not already there)
4. Run the review tool and check addresses near 094xx and 099xx for data consistency
5. Check that legitimate `04xxx` addresses (from frames 5000-5700) are not affected by the fix

## Key constants

```
ADDR_X_START = 289          # x-pixel of first address digit
address_char_spacing = 13.92 # pixels between address digit centers
CELL_W = 14, CELL_H = 28   # character cell dimensions
first_row_center_y = 272    # y-pixel of row 0 center
row_height = 28             # pixels between row centers
visible_rows = 13           # rows per frame
ADDR_MIN = 0x00000          # valid address range
ADDR_MAX = 0x13FFF
```

## Python environment

Always use `source venv/bin/activate` before running Python commands.
