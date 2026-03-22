# Architecture Gaps: Frame vs Frame.Row Conflation

**Date**: 2026-03-21
**Status**: Investigation complete, fixes pending

---

## Executive Summary

Each video frame contains ~12–16 address rows. The data stores (frame_assignments.json, crop_index.json) correctly model this as per-row data. However, the review app's move API, the UI, the pipeline's address validator, and the firmware output path all conflate "frame" with "frame.row" in different ways, creating data integrity issues.

**Key metrics:**
- 3,123 frames (15.6%) have address discontinuities (gaps > 0x10 between consecutive rows)
- 81 frames have address reversals (mid-scroll artifacts)
- ~48% of byte observations come from frames with discontinuities
- ~57% of addresses have at least one observation from a discontinuous frame
- frame_moves.json has 22,089 entries — NO row_y field in any of them

---

## 1. Move API: Single-Address Moves on Multi-Address Frames

### The Problem

`apply_single_move()` (`app.py:157–215`) moves ONE frame from ONE address to ONE address. But each frame appears at ~12–16 addresses in crop_index.json (one per visible row). Moving a frame from one address leaves it at all other addresses.

### Code

```python
# app.py:157-215 — apply_single_move()
def apply_single_move(frame, from_addr, to_addr):
    src = crop_index.get(from_addr)
    dst = crop_index.setdefault(to_addr, {...})

    # Moves readings, confidences, crop PNG, frame array entry
    # from src[from_addr] to dst[to_addr]
    #
    # Does NOT touch the same frame at other addresses
```

### Concrete Example

Frame 6679 contains 15 rows at addresses `$03FB0`–`$040A0`. User moves frame 6679 from `$03FB0` to `$05FB0`:

```
Before:
  $03FB0: frames=[6679], readings={'6679': [...]}
  $03FC0: frames=[6679], readings={'6679': [...]}
  ...
  $040A0: frames=[6679], readings={'6679': [...]}

After move:
  $03FB0: DELETED (empty after move)
  $03FC0: frames=[6679], readings={'6679': [...]}   ← UNCHANGED
  $03FD0: frames=[6679], readings={'6679': [...]}   ← UNCHANGED
  ...
  $05FB0: frames=[6679], readings={'6679': [...]}   ← NEW
```

Frame 6679 now appears at 15 addresses total (`$03FC0`–`$040A0` + `$05FB0`) with inconsistent offsets.

### Who Handles Multi-Row Moves Correctly

**Only `fix_address_trajectory.py` Phase 1** (`build_phase1_moves()`, line 811–863) preserves row offsets:

```python
# fix_address_trajectory.py:854-856
row_offset = (addr_val - old_top) // 0x10
new_addr_val = new_top_val + row_offset * 0x10
new_addr = f"{new_addr_val:05X}"
```

This generates one move per row, preserving the 0x10 spacing from the old top address. Example: frame 690 generates 12 separate moves, all with the same offset applied.

**Phase 2** (`detect_trajectory_moves()`, line 300–390) does NOT coordinate rows — each address is evaluated independently based on confusable character swaps.

### frame_moves.json Schema

```json
{
  "frame": 6679,
  "from_addr": "03FB0",
  "to_addr": "05FB0",
  "timestamp": "2026-03-04T21:47:52.659380+00:00"
}
```

Extended (from pipeline scripts):
```json
{
  "frame": "14263",
  "from_addr": "0DA80",
  "to_addr": "0DA70",
  "timestamp": "2026-03-21T19:09:16.329637+00:00",
  "strategy": "outlier_vote"
}
```

**No `row_y` field in any move record.** Row identity is implicit in the from/to address pair.

### Chained Move Bug (TODO #7h)

Append-only ledger + sequential replay = ghost frames:

```
Move 1: Frame 8154, $08000 → $08010   (user move)
Move 2: Frame 8154, $07FF0 → $08000   (trajectory fix)

Replay:
  1. Move 8154 from $08000 → $08010    ✓
  2. Try move 8154 from $07FF0 → $08000 — frame not at $07FF0 → SKIP

Result: Frame 8154 in $08010.frames AND $08000.frames but with
        no readings → blank in review tool
```

No undo mechanism exists. The ledger is append-only.

---

## 2. validate_address_sequence: Assumes Contiguous Rows

### The Problem

`validate_address_sequence()` (`extract_pipeline.py:432–496`) assumes all rows in a frame form a single contiguous address sequence. It picks the best anchor and forces all rows to fit `anchor + row_offset * 0x10`. It never detects or handles:

- Address reversals within a frame
- Gaps > 0x10 between consecutive rows
- Split-scroll artifacts (top rows from new position, bottom rows from old)

### Code

```python
# extract_pipeline.py:432-496
def validate_address_sequence(addresses_with_rows):
    for anchor_addr, anchor_row_y, anchor_conf in addresses_with_rows:
        for addr, row_y, conf in addresses_with_rows:
            row_offset = round((row_y - anchor_row_y) / row_h)
            expected_addr = anchor_addr + row_offset * 0x10

            addr_diff = abs(addr - expected_addr)
            if addr_diff == 0:
                score += conf * 2          # exact match
            elif addr_diff <= 0x08:
                score += conf              # 0/8 confusion
            else:
                score += conf * 0.1        # large mismatch — penalized but INCLUDED

        # Winner: highest score
```

**Critical flaw:** Mismatched rows get confidence scaled to 0.3, but are still **included in the returned sequence** with forced-corrected addresses. Their byte data is then read and added to the voting pool.

### Split-Scroll Example: Frame 673

```
Row  0: y=216  addr=$0C340  Δ=+0x00  (anchor)
Row  1: y=244  addr=$0C350  Δ=+0x10  ✓
...
Row  7: y=412  addr=$0C3B0  Δ=+0x10  ✓
Row  8: y=440  addr=$0C3A0  Δ=-0x10  ✗ REVERSAL (mid-scroll artifact)
Row  9: y=468  addr=$0C3D0  Δ=+0x30  ✗ GAP ($0C3C0 never visible)
Row 10: y=496  addr=$0C3E0  Δ=+0x10  ✓
...
Row 14: y=608  addr=$0C420  Δ=+0x10  ✓
```

What the validator does:
- Picks anchor $0C340 (best score on rows 0–7)
- Row 8: expected $0C3C0, got $0C3A0 → corrected to $0C3C0, confidence × 0.3
- Row 9: expected $0C3D0, got $0C3D0 → exact match
- **Result:** Row 8's bytes (which visually show $0C3A0 data) are assigned to address $0C3C0

### Scale

| Category | Count | % of all frames |
|----------|-------|-----------------|
| Frames with gaps > 0x10 | 3,042 | 15.2% |
| Frames with reversals | 81 | 0.4% |
| Frames with multiple issues | 358 | 1.8% |
| **Total affected frames** | **3,123** | **15.6%** |

### Byte Voting Contamination

`process_frame()` (`extract_pipeline.py:499–545`) reads hex bytes for every validated row and appends them to `all_observations[addr]`:

```python
# extract_pipeline.py:728-734
for addr_int, row_y, hex_bytes, byte_confs, addr_conf in results:
    all_observations[addr_int].append({
        'bytes': hex_bytes,
        'byte_confs': byte_confs,
        'addr_conf': addr_conf,
        'frame': frame_name,
    })
```

No filtering for split-scroll. All observations enter the weighted majority vote:

```python
# extract_pipeline.py:768-776
for obs in observations:
    weight = obs['byte_confs'][byte_idx] * obs['addr_conf']
    votes[byte_val] += weight
```

A split-scroll row contributes weight ≈ 0.27 × 0.09 ≈ 0.024. Weak, but for addresses with few observations, enough to shift the vote.

### Phase 1 Trajectory Fix Limitations

`fix_address_trajectory.py` Phase 1 detects **frame-level** misassignment (entire frame at wrong address range). It does NOT detect **row-level** splits within a correctly-placed frame.

Additionally:
- Pass 1 (F1–F1728) has waypoint spacing 1727 → **skipped entirely** (threshold is 500)
- A frame where the top half is at the correct trajectory position but the bottom half shows stale data is never flagged

---

## 3. Review UI: No Row Context

### Crop Structure

Each crop PNG is a **single row** from one frame:
- 836 × 28 pixels (address digits + 16 bytes)
- Cropped at `(x=284..1120, y=row_y-17..row_y+11)` — see `precompute.py:48-52, 172-178`

The crop correctly shows one row. The issue is that the UI provides no context about the row's position within its frame.

### What's Shown vs What's Available

| Information | Shown in UI? | Available in data? | Where? |
|-------------|-------------|-------------------|--------|
| Frame number | Yes | Yes | crop_index.json |
| 16 bytes + confidence | Yes | Yes | crop_index.json |
| Row position (row_y) | **No** | Yes | frame_assignments.json |
| Is split-scroll frame? | **No** | Derivable | frame_assignments.json |
| How many addresses in this frame? | **No** | Yes | frame_assignments.json |
| Address OCR confidence | **No** | Yes | frame_assignments.json |

### API Response

`/api/frames/<addr>` (`app.py:569–584`):

```python
return jsonify({
    "address": addr,
    "frames": frames,
    "video_frames": video_frames,
    "readings": readings,            # {frame_str: [16 bytes]}
    "knn_confidences": knn_confidences,  # {frame_str: [16 floats]}
    "has_ref": addr in ref_addresses,
})
```

**No `row_y` transmitted.** The UI has no way to know which row within the frame produced this crop.

### Move Flow (Mobile)

```javascript
// mobile.html:1014-1025
async function doSingleMove(frameKey, rawAddr) {
    const toAddr = rawAddr.trim().toUpperCase().replace(/^0X/, '').padStart(5, '0');
    const result = await api('POST', '/api/move_frame', {
        frame: frameKey,
        from_addr: state.currentAddr,  // Current address being reviewed
        to_addr: toAddr,               // User-entered destination
    });
}
```

The user sees "Frame 673" at address `$0C340` and clicks Move. They enter `$0C350`. The API moves frame 673 from `$0C340` to `$0C350`. The user has no indication that frame 673 also appears at `$0C350, $0C360, ..., $0C420` — those entries are untouched.

### Accept/Reject Granularity

- **Address-scoped** — marks all 16 bytes of an address as accepted/rejected
- Not per-frame, not per-row, not per-byte
- A user accepting address `$0C340` accepts the consensus vote from all contributing frames, including contaminated split-scroll observations

---

## 4. Data Flow: Frame Moves Don't Reach Firmware Output

### Two Divergent Output Paths

```
PATH A: Pipeline Output (ignores frame moves)
  extract_pipeline.py
    → extracted_firmware.txt (voted bytes, with [N obs] counts)
    → postprocess_firmware.py (reads extracted_firmware.txt directly)
    → firmware_merged.txt
    → ff_fill.py
    → hakko_fm203_full.bin / hakko_fm203.bin

PATH B: Review App Export (includes frame moves)
  precompute.py
    → loads frame_assignments.json
    → replays frame_moves.json onto crop_index.json
  app.py
    → loads crop_index.json (has moves applied)
    → review_state.json (user edits)
    → /api/export/binary (uses review_state)
```

**`postprocess_firmware.py` never reads `frame_moves.json`** — it only reads `extracted_firmware.txt`. Any corrections made via the review app's move UI are invisible to the pipeline output.

### Pipeline Step Order

```python
# extract_pipeline.py:994-1001
post_steps = [
    ('Precompute (crop index)',          'firmware_review_tool/precompute.py'),
    ('Address trajectory correction',    'fix_address_trajectory.py'),
    ('Outlier vote correction',          'fix_outlier_votes.py'),
    ('Post-processing (merge/binary)',   'postprocess_firmware.py'),
    ('FF-fill & FF-forced override',     'ff_fill.py --heuristic'),
    ('Gap context precompute',           'firmware_review_tool/precompute_gaps.py'),
]
```

Note: precompute runs FIRST, then trajectory fix generates new frame_moves, then postprocess runs. But postprocess reads `extracted_firmware.txt` (which was produced by the extraction step before any post-pipeline corrections). The trajectory fix moves are only reflected in crop_index.json (via the NEXT precompute run), not in the current pipeline's firmware output.

### Staleness & Refresh

`app.py` detects when `firmware_merged.txt` is newer than `review_state.json` and shows a "Refresh from merged" banner. When the user clicks refresh:
- review_state is overwritten from firmware_merged.txt
- **Any consensus corrections from frame moves in crop_index are lost**
- User-edited bytes (source="user") are preserved

### frame_assignments.json Update Chain

| Script | Reads | Writes |
|--------|-------|--------|
| extract_pipeline.py | — | frame_assignments.json (initial) |
| precompute.py | frame_assignments.json | crop_index.json |
| fix_address_trajectory.py | frame_assignments.json | frame_assignments.json (updated), frame_moves.json |
| fullvideo_gap_recovery.py | — | frame_moves.json, frame_assignments.json (appended) |
| postprocess_firmware.py | extracted_firmware.txt | firmware_merged.txt |
| app.py (move_frame) | — | frame_moves.json, crop_index.json (in-memory + disk) |

**frame_assignments.json is never updated by review app moves.** It represents the pipeline's view. crop_index.json diverges from it after user moves.

---

## 5. Recommendations

### 5.1 Move API: Add Multi-Row Awareness

The move API should either:
- **Option A**: Accept a "move all rows" flag that computes offset from the frame's top address and moves all rows with preserved 0x10 spacing (like Phase 1 does)
- **Option B**: Accept `row_y` to make single-row moves explicit and warn the user about orphaned rows

### 5.2 validate_address_sequence: Detect Split-Scroll

Add post-validation checks:
1. Detect gaps > 0x10 between consecutive rows in the returned sequence
2. Detect address reversals
3. When found, split the frame into separate address groups
4. Validate each group independently
5. Or: cross-check each row against the manual trajectory and reject rows outside the expected screen range

### 5.3 Review UI: Surface Row Context

- Add `row_y` to the `/api/frames/<addr>` response
- Show a visual indicator for split-scroll frames (e.g., row position badge, "1 of 15 rows" label)
- When moving a frame, offer "move all rows (preserve offsets)" vs "move this row only"
- Show address OCR confidence per frame

### 5.4 Data Flow: Unify Output Paths

Either:
- Make `postprocess_firmware.py` read `frame_moves.json` and adjust its voting accordingly
- Or make `postprocess_firmware.py` read from `crop_index.json` (which has moves applied) instead of `extracted_firmware.txt`
- Ensure pipeline step order is correct: trajectory fix → precompute → postprocess

### 5.5 crop_index.json: Store row_y

Add `row_y` to crop_index entries so downstream consumers (API, UI, move logic) can reason about row position without loading frame_assignments.json.

---

## 6. File Reference

| File | Lines | Role |
|------|-------|------|
| `extract_pipeline.py` | 432–496 | `validate_address_sequence()` — anchor-based validator |
| `extract_pipeline.py` | 499–545 | `process_frame()` — reads all rows, no split-scroll filter |
| `extract_pipeline.py` | 725–739 | Observation collection + frame_assignments save |
| `extract_pipeline.py` | 763–787 | Weighted majority byte voting |
| `extract_pipeline.py` | 994–1001 | Post-pipeline step order |
| `firmware_review_tool/app.py` | 157–215 | `apply_single_move()` — single-address frame move |
| `firmware_review_tool/app.py` | 218–244 | `recompute_consensus_for_addr()` |
| `firmware_review_tool/app.py` | 569–584 | `/api/frames/<addr>` — no row_y in response |
| `firmware_review_tool/app.py` | 861–911 | `/api/move_frame` endpoint |
| `firmware_review_tool/app.py` | 914–964 | `/api/move_frames` batch endpoint |
| `firmware_review_tool/precompute.py` | 134–178 | crop_index build from frame_assignments |
| `firmware_review_tool/precompute.py` | 245–311 | `apply_frame_moves()` — replays ledger |
| `firmware_review_tool/templates/mobile.html` | 1014–1025 | `doSingleMove()` — no row context |
| `fix_address_trajectory.py` | 300–390 | Phase 2: per-address confusion swap (no row coordination) |
| `fix_address_trajectory.py` | 811–863 | Phase 1: multi-row offset-preserving moves (CORRECT) |
| `postprocess_firmware.py` | 57+ | `load_extraction()` — reads extracted_firmware.txt only |
| `manual_trajectory.py` | — | 77 waypoints, `interpolate_trajectory()` |
| `frame_moves.json` | — | 22,089 entries, no row_y field |
| `frame_assignments.json` | — | Per-row (addr, row_y, conf) — authoritative |
| `crop_index.json` | — | Per-address frame mapping — no row_y |
