#!/usr/bin/env python3
"""
Strategy 8: Global Address Trajectory Correction.

Unified replacement for fix_d_c_misread.py + fix_49_misread.py.  Uses the full
video trajectory (piecewise-monotone through anchor frames) to correct ALL
address OCR confusions in one pass.

Confusion pairs: C<->D, 4<->9, 8<->6, 0<->8

Steps:
  1. Build raw frame->address trajectory from crop_index.json
  2. Build anchor trajectory (addresses with no confusable chars)
  3. Detect breakpoints (scroll direction reversals) and segment trajectory
  4. For each non-anchor address+frame, generate swap candidates and pick
     the one closest to the expected address from trajectory interpolation
  5. Execute moves + rebuild downstream
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from itertools import product

from frame_utils import (
    is_video_frame_key, video_frame_key, crop_filename, parse_frame_key,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
EXTRACTED_FW_PATH = os.path.join(PROJECT_ROOT, 'extracted_firmware.txt')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')
MERGED_FW_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')

# Characters that participate in OCR confusion pairs
CONFUSABLE_CHARS = set('CD4986')

# Swap map: for each confusable char, the list of chars it can be confused with
# Only documented confusion pairs: C<->D, 4<->9, 8<->6
# (0<->8 excluded: too many false positives, not a documented confusion)
SWAP_MAP = {
    'C': ['D'],
    'D': ['C'],
    '4': ['9'],
    '9': ['4'],
    '8': ['6'],
    '6': ['8'],
}

# Breakpoint detection: require this many consecutive anchors confirming reversal
BREAKPOINT_CONFIRM = 20

# Trajectory interpolation radius (frames)
ANCHOR_RADIUS = 500


# ── Step 1-2: Build trajectories ─────────────────────────────────────────────

def load_crop_index():
    with open(CROP_INDEX_PATH) as f:
        return json.load(f)


def save_crop_index(crop_index):
    with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f, indent=2)
    print(f"  Saved {CROP_INDEX_PATH}")


def has_confusable(addr):
    """Check if address string contains any confusable character."""
    return any(c in CONFUSABLE_CHARS for c in addr)


def build_anchor_trajectory(crop_index, use_extracted=True):
    """
    Build ground-truth trajectory from addresses with NO confusable characters.
    Returns sorted list of (frame, addr_median) where addr_median is the
    median of all anchor addresses visible at that frame.

    Aggregates per-frame first to avoid bias from multiple rows per frame.
    """
    import statistics

    # Collect raw (frame, addr_value) pairs
    raw = {}  # frame -> list of addr values
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        if has_confusable(addr):
            continue
        addr_val = int(addr, 16)
        if use_extracted:
            for f in entry.get('frames', []):
                raw.setdefault(f, []).append(addr_val)
        else:
            for vf in entry.get('video_frames', []):
                raw.setdefault(vf, []).append(addr_val)

    # Aggregate: one point per frame using median address
    anchors = []
    for f in sorted(raw):
        anchors.append((f, statistics.median(raw[f])))

    return anchors


def detect_breakpoints(anchors):
    """
    Detect major direction-change breakpoints by finding peaks and valleys
    in the trajectory.  A breakpoint is where the address reaches a global
    or significant local extremum.

    Uses a smoothed trajectory (running median over BREAKPOINT_CONFIRM points)
    to ignore single-frame noise.

    Returns list of frame numbers where breakpoints occur.
    """
    n = len(anchors)
    if n < BREAKPOINT_CONFIRM * 3:
        return []

    # Smooth with running median
    half_w = BREAKPOINT_CONFIRM // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        vals = sorted(a[1] for a in anchors[lo:hi])
        smoothed.append(vals[len(vals) // 2])

    # Find direction changes in smoothed trajectory
    # Compare chunks of BREAKPOINT_CONFIRM size to detect sustained reversals
    chunk = BREAKPOINT_CONFIRM
    breakpoints = []
    prev_dir = None

    for i in range(0, n - chunk, chunk // 2):
        start_val = smoothed[i]
        end_val = smoothed[min(i + chunk, n - 1)]
        diff = end_val - start_val

        if abs(diff) < 0x100:  # ignore tiny changes
            continue

        current_dir = 1 if diff > 0 else -1
        if prev_dir is not None and current_dir != prev_dir:
            # Direction changed — breakpoint is at the boundary
            bp_idx = i + chunk // 4
            breakpoints.append(anchors[min(bp_idx, n - 1)][0])

        prev_dir = current_dir

    return breakpoints


def build_segments(anchors, breakpoints):
    """
    Split anchor trajectory into monotone segments at breakpoints.
    Returns list of segments, each a sorted list of (frame, addr_median).
    """
    if not breakpoints:
        return [anchors]

    segments = []
    bp_set = sorted(breakpoints)
    seg_start = 0

    for bp_frame in bp_set:
        seg_end = 0
        for i, (f, _) in enumerate(anchors):
            if f >= bp_frame:
                seg_end = i
                break
        if seg_end > seg_start:
            segments.append(anchors[seg_start:seg_end])
        seg_start = seg_end

    if seg_start < len(anchors):
        segments.append(anchors[seg_start:])

    return segments


def estimate_expected_address(frame, segment, radius=ANCHOR_RADIUS):
    """
    Estimate expected address at a given frame using inverse-distance-weighted
    median of per-frame-median anchor points within ±radius.

    Each anchor point is one per-frame median, so each frame gets equal weight
    regardless of how many anchor rows are visible.
    """
    import bisect
    frames_list = [s[0] for s in segment]

    # Check for exact match first
    idx = bisect.bisect_left(frames_list, frame)
    if idx < len(segment) and segment[idx][0] == frame:
        return int(segment[idx][1])

    # Gather nearby anchor frames within radius
    nearby = []
    weights = []

    # Search backward
    for i in range(idx - 1, -1, -1):
        d = frame - segment[i][0]
        if d > radius:
            break
        nearby.append(segment[i][1])
        weights.append(1.0 / d)

    # Search forward
    for i in range(idx, len(segment)):
        d = segment[i][0] - frame
        if d > radius:
            break
        if d == 0:
            continue
        nearby.append(segment[i][1])
        weights.append(1.0 / d)

    if not nearby:
        return None

    # Weighted median
    paired = sorted(zip(nearby, weights))
    total_w = sum(weights)
    cumulative = 0.0
    for val, w in paired:
        cumulative += w
        if cumulative >= total_w / 2.0:
            return int(val)
    return int(paired[-1][0])


def find_segment_for_frame(frame, segments):
    """Find which segment a frame belongs to (by frame range)."""
    for seg in segments:
        if not seg:
            continue
        if seg[0][0] <= frame <= seg[-1][0]:
            return seg
    if segments and frame < segments[0][0][0]:
        return segments[0]
    if segments and frame > segments[-1][-1][0]:
        return segments[-1]
    return None


# ── Step 3: Generate swap candidates ─────────────────────────────────────────

def generate_swap_candidates(addr):
    """
    Generate all addresses reachable by swapping confusable characters.
    Each position with a confusable char can stay or swap to any alternative.
    Includes single and multi-position swaps. Excludes original address.
    """
    chars = list(addr)
    # For each position, list of possible replacements (including original)
    options = []
    for c in chars:
        if c in SWAP_MAP:
            options.append([c] + SWAP_MAP[c])
        else:
            options.append([c])

    candidates = set()
    for combo in product(*options):
        candidate = ''.join(combo)
        if candidate != addr:
            candidates.add(candidate)

    return candidates


# ── Step 4: Detect moves ─────────────────────────────────────────────────────

def detect_trajectory_moves(crop_index, segments_extracted, segments_video):
    """
    For each address containing confusable characters, for each frame,
    check if a swap candidate is closer to the expected trajectory address.

    Returns list of (source_addr, frame_id, dest_addr) tuples.
    frame_id is int for extracted frames, 'vNNNNN' string for video frames.
    """
    moves = []
    skipped_no_segment = 0
    skipped_no_expected = 0

    for addr in sorted(crop_index):
        if addr == 'ref_addresses':
            continue
        if not has_confusable(addr):
            continue

        entry = crop_index[addr]
        addr_val = int(addr, 16)
        candidates = generate_swap_candidates(addr)
        if not candidates:
            continue

        # Pre-compute candidate values
        cand_vals = {c: int(c, 16) for c in candidates}

        # Check extracted frames
        for frame in entry.get('frames', []):
            seg = find_segment_for_frame(frame, segments_extracted)
            if seg is None:
                skipped_no_segment += 1
                continue

            expected = estimate_expected_address(frame, seg)
            if expected is None:
                skipped_no_expected += 1
                continue

            current_dist = abs(addr_val - expected)
            best_swap = None
            best_dist = current_dist

            for cand, cand_val in cand_vals.items():
                dist = abs(cand_val - expected)
                swap_magnitude = abs(addr_val - cand_val)
                adaptive_threshold = max(swap_magnitude // 2, 0x80)
                if (dist < best_dist
                        and (current_dist - dist) >= adaptive_threshold
                        and dist < current_dist * 0.5):
                    best_dist = dist
                    best_swap = cand

            if best_swap is not None:
                moves.append((addr, frame, best_swap))

        # Check video frames
        for vf in entry.get('video_frames', []):
            seg = find_segment_for_frame(vf, segments_video)
            if seg is None:
                skipped_no_segment += 1
                continue

            expected = estimate_expected_address(vf, seg)
            if expected is None:
                skipped_no_expected += 1
                continue

            current_dist = abs(addr_val - expected)
            best_swap = None
            best_dist = current_dist

            for cand, cand_val in cand_vals.items():
                dist = abs(cand_val - expected)
                swap_magnitude = abs(addr_val - cand_val)
                adaptive_threshold = max(swap_magnitude // 2, 0x80)
                if (dist < best_dist
                        and (current_dist - dist) >= adaptive_threshold
                        and dist < current_dist * 0.5):
                    best_dist = dist
                    best_swap = cand

            if best_swap is not None:
                moves.append((addr, video_frame_key(vf), best_swap))

    if skipped_no_segment:
        print(f"  Skipped {skipped_no_segment} frames (no segment)")
    if skipped_no_expected:
        print(f"  Skipped {skipped_no_expected} frames (no nearby anchors)")

    return moves


# ── Step 5: Execute moves ────────────────────────────────────────────────────

def execute_moves(crop_index, moves):
    """
    Move frames from source to destination address entries.
    Moves readings, confidences, crop PNGs.
    """
    affected_src = set()
    affected_dst = set()
    frames_moved = 0
    crops_moved = 0
    crops_missing = 0
    entries_emptied = 0

    # Group moves by (source, dest) for efficiency
    grouped = {}
    for src, frame, dest in moves:
        grouped.setdefault((src, dest), []).append(frame)

    for (src_key, dst_key), frame_list in sorted(grouped.items()):
        affected_src.add(src_key)
        affected_dst.add(dst_key)

        src_entry = crop_index.get(src_key)
        if src_entry is None:
            continue

        # Create destination entry if needed
        if dst_key not in crop_index:
            crop_index[dst_key] = {
                'frames': [],
                'video_frames': [],
                'readings': {},
                'confidences': {},
            }
        dst_entry = crop_index[dst_key]
        dst_entry.setdefault('video_frames', [])

        for frame in frame_list:
            # Determine if this is a video frame (string starting with 'v')
            if isinstance(frame, str) and frame.startswith('v'):
                is_video = True
                frame_int = int(frame[1:])
                frame_str = frame
                arr_key = 'video_frames'
            else:
                is_video = False
                frame_int = frame
                frame_str = str(frame)
                arr_key = 'frames'

            src_entry.setdefault(arr_key, [])
            if frame_int not in src_entry.get(arr_key, []):
                continue  # already moved or not present

            # Move readings and confidences
            if frame_str in src_entry.get('readings', {}):
                dst_entry.setdefault('readings', {})[frame_str] = \
                    src_entry['readings'].pop(frame_str)
            if frame_str in src_entry.get('confidences', {}):
                dst_entry.setdefault('confidences', {})[frame_str] = \
                    src_entry['confidences'].pop(frame_str)

            # Add frame to destination
            if frame_int not in dst_entry[arr_key]:
                dst_entry[arr_key].append(frame_int)

            # Remove frame from source
            src_entry[arr_key].remove(frame_int)

            # Move crop PNG
            src_dir = os.path.join(CROPS_DIR, src_key.lower())
            dst_dir = os.path.join(CROPS_DIR, dst_key.lower())
            png_name = crop_filename(frame_int, is_video=is_video)
            src_path = os.path.join(src_dir, png_name)
            dst_path = os.path.join(dst_dir, png_name)

            if os.path.exists(src_path):
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src_path, dst_path)
                crops_moved += 1
            else:
                crops_missing += 1

            frames_moved += 1

        # Sort destination frames
        dst_entry['frames'] = sorted(dst_entry['frames'])
        dst_entry['video_frames'] = sorted(dst_entry['video_frames'])

        # Clean up empty source entries
        if not src_entry.get('frames') and not src_entry.get('video_frames'):
            del crop_index[src_key]
            entries_emptied += 1
            src_dir = os.path.join(CROPS_DIR, src_key.lower())
            if os.path.isdir(src_dir) and not os.listdir(src_dir):
                os.rmdir(src_dir)

    print(f"  Frames moved: {frames_moved}")
    print(f"  Crops moved: {crops_moved} (missing: {crops_missing})")
    print(f"  Source addresses affected: {len(affected_src)}")
    print(f"  Source entries emptied (removed): {entries_emptied}")
    print(f"  Destination addresses created/updated: {len(affected_dst)}")

    return affected_src, affected_dst


def log_frame_moves(moves):
    """Append move records to frame_moves.json ledger."""
    existing = {"moves": []}
    if os.path.exists(FRAME_MOVES_PATH):
        with open(FRAME_MOVES_PATH) as f:
            existing = json.load(f)

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for src, frame, dest in moves:
        existing["moves"].append({
            "frame": frame,
            "from_addr": src,
            "to_addr": dest,
            "timestamp": ts,
            "strategy": "trajectory",
        })

    with open(FRAME_MOVES_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    print(f"  Logged {len(moves)} moves to {FRAME_MOVES_PATH}")


# ── Step 6: Recompute byte consensus ─────────────────────────────────────────

def weighted_majority_vote(readings, confidences):
    """
    For each byte position, do weighted majority voting across frames.
    Returns list of 16 hex strings and total observation count.
    """
    if not readings:
        return None, 0

    result = []
    for byte_idx in range(16):
        votes = Counter()
        for frame_str, byte_list in readings.items():
            if byte_idx < len(byte_list):
                weight = 1.0
                if frame_str in confidences:
                    conf_list = confidences[frame_str]
                    if byte_idx < len(conf_list):
                        weight = conf_list[byte_idx]
                votes[byte_list[byte_idx]] += weight
        if votes:
            result.append(votes.most_common(1)[0][0])
        else:
            result.append('FF')

    return result, len(readings)


def update_extracted_firmware(crop_index, affected_src, affected_dst):
    """Recompute byte consensus for affected addresses, update extracted_firmware.txt."""
    all_affected = affected_src | affected_dst

    lines_by_addr = {}
    header_lines = []
    with open(EXTRACTED_FW_PATH) as f:
        for line in f:
            stripped = line.rstrip('\n')
            if stripped.lstrip().startswith('#') or not stripped.strip():
                header_lines.append(stripped)
                continue
            parts = stripped.split(':')
            if len(parts) >= 2:
                addr_str = parts[0].strip().upper()
                lines_by_addr[addr_str] = stripped

    updated = 0
    added = 0
    for addr_key in sorted(all_affected):
        if addr_key in crop_index:
            entry = crop_index[addr_key]
            readings = entry.get('readings', {})
            confidences = entry.get('confidences', {})
            consensus, obs_count = weighted_majority_vote(readings, confidences)

            if consensus and obs_count > 0:
                hex_str = ' '.join(consensus)
                new_line = f"{addr_key}: {hex_str}  [{obs_count} obs]"
                if addr_key in lines_by_addr:
                    updated += 1
                else:
                    added += 1
                lines_by_addr[addr_key] = new_line
        else:
            if addr_key in lines_by_addr:
                existing = lines_by_addr[addr_key]
                if '[REF]' not in existing:
                    del lines_by_addr[addr_key]

    with open(EXTRACTED_FW_PATH, 'w') as f:
        for h in header_lines:
            f.write(h + '\n')
        for addr_str in sorted(lines_by_addr.keys(),
                                key=lambda x: int(x, 16)):
            f.write(lines_by_addr[addr_str] + '\n')

    print(f"  Updated {updated} lines, added {added} lines in extracted_firmware.txt")


# ── Step 7: Rebuild downstream ───────────────────────────────────────────────

def rebuild_downstream():
    """Run postprocess_firmware.py and precompute_gaps.py."""
    print("\n  Running postprocess_firmware.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'postprocess_firmware.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: postprocess_firmware.py failed:\n{result.stderr}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Coverage', 'Final data', 'Hex dump',
                                      'Source map', 'binary', 'MATCH']):
            print(f"    {line}")

    print("\n  Running precompute_gaps.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'firmware_review_tool/precompute_gaps.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: precompute_gaps.py failed:\n{result.stderr}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Missing', 'Addresses with', 'Unreachable',
                                      'Total context', 'Done']):
            print(f"    {line}")


# ── Step 8: Reset review state ───────────────────────────────────────────────

def reset_review_state(affected_src, affected_dst):
    """Reset review status for all affected addresses."""
    if not os.path.exists(REVIEW_STATE_PATH):
        print("  review_state.json not found -- skipping")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    if 'lines' not in state:
        print("  review_state.json has no 'lines' -- skipping")
        return

    merged_bytes = {}
    if os.path.exists(MERGED_FW_PATH):
        with open(MERGED_FW_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split(':')
                if len(parts) >= 2:
                    addr_str = parts[0].strip().upper()
                    rest = parts[1].strip()
                    rest = re.sub(r'\[.*?\]', '', rest).strip()
                    hex_bytes = rest.split()
                    if len(hex_bytes) == 16:
                        merged_bytes[addr_str] = hex_bytes

    all_affected = affected_src | affected_dst
    reset_count = 0

    for addr_key in sorted(all_affected):
        if addr_key in state['lines']:
            if addr_key in merged_bytes:
                state['lines'][addr_key]['bytes'] = merged_bytes[addr_key]
                state['lines'][addr_key]['source'] = 'merged'
            state['lines'][addr_key]['status'] = 'unreviewed'
            state['lines'][addr_key]['edited_positions'] = []
            reset_count += 1
        elif addr_key in merged_bytes:
            state['lines'][addr_key] = {
                'status': 'unreviewed',
                'bytes': merged_bytes[addr_key],
                'source': 'merged',
                'edited_positions': [],
            }
            reset_count += 1

    with open(REVIEW_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print(f"  Reset {reset_count} addresses in review_state.json")


# ── Reporting ────────────────────────────────────────────────────────────────

def summarize_moves(moves):
    """Print a summary of planned moves by confusion type."""
    by_pair = Counter()
    for src, frame, dest in moves:
        # Determine which chars changed
        swaps = []
        for i, (s, d) in enumerate(zip(src, dest)):
            if s != d:
                swaps.append(f"{s}->{d}")
        key = ', '.join(swaps) if swaps else '?'
        by_pair[key] += 1

    print(f"\n  Move summary by confusion type:")
    for pair, count in by_pair.most_common():
        print(f"    {pair}: {count} frames")

    # Unique source->dest pairs
    pairs = set((src, dest) for src, _, dest in moves)
    print(f"  Unique source->dest address pairs: {len(pairs)}")

    # Show a sample of moves
    sample = moves[:10]
    if sample:
        print(f"\n  Sample moves (first {len(sample)}):")
        for src, frame, dest in sample:
            print(f"    {src} -> {dest}  (frame {frame})")
    if len(moves) > 10:
        print(f"    ... and {len(moves) - 10} more")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Strategy 8: Global address trajectory correction')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview moves without modifying any files')
    args = parser.parse_args()

    print("=" * 60)
    print("Strategy 8: Global Address Trajectory Correction")
    print("=" * 60)

    # Load data
    crop_index = load_crop_index()
    total_addrs = sum(1 for k in crop_index if k != 'ref_addresses')
    anchor_count = sum(1 for k in crop_index
                       if k != 'ref_addresses' and not has_confusable(k))
    non_anchor_count = total_addrs - anchor_count
    print(f"\n  Total addresses: {total_addrs}")
    print(f"  Anchor addresses (no confusable chars): {anchor_count}")
    print(f"  Non-anchor addresses (may need correction): {non_anchor_count}")

    # Build extracted frame trajectory
    print("\nStep 1: Build anchor trajectories")
    anchors_extracted = build_anchor_trajectory(crop_index, use_extracted=True)
    anchors_video = build_anchor_trajectory(crop_index, use_extracted=False)
    print(f"  Extracted anchor points: {len(anchors_extracted)}")
    print(f"  Video anchor points: {len(anchors_video)}")

    # Detect breakpoints
    print("\nStep 2: Detect breakpoints")
    bp_extracted = detect_breakpoints(anchors_extracted)
    bp_video = detect_breakpoints(anchors_video)
    print(f"  Extracted trajectory breakpoints: {len(bp_extracted)}")
    if bp_extracted:
        for bp in bp_extracted:
            print(f"    Frame {bp}")
    print(f"  Video trajectory breakpoints: {len(bp_video)}")
    if bp_video:
        for bp in bp_video:
            print(f"    Frame {bp}")

    # Build segments
    segments_extracted = build_segments(anchors_extracted, bp_extracted)
    segments_video = build_segments(anchors_video, bp_video)
    print(f"  Extracted segments: {len(segments_extracted)}")
    for i, seg in enumerate(segments_extracted):
        if seg:
            direction = "increasing" if seg[-1][1] >= seg[0][1] else "decreasing"
            print(f"    Segment {i}: frames {seg[0][0]}-{seg[-1][0]}, "
                  f"addr ${int(seg[0][1]):05X}-${int(seg[-1][1]):05X} ({direction})")
    if segments_video:
        print(f"  Video segments: {len(segments_video)}")
        for i, seg in enumerate(segments_video):
            if seg:
                direction = "increasing" if seg[-1][1] >= seg[0][1] else "decreasing"
                print(f"    Segment {i}: frames {seg[0][0]}-{seg[-1][0]}, "
                      f"addr ${seg[0][1]:05X}-${seg[-1][1]:05X} ({direction})")

    # Detect moves
    print("\nStep 3: Detect trajectory corrections")
    moves = detect_trajectory_moves(crop_index, segments_extracted, segments_video)
    print(f"  Total moves detected: {len(moves)}")

    if not moves:
        print("\nNo corrections needed. Nothing to do.")
        return

    summarize_moves(moves)

    if args.dry_run:
        print("\n  [DRY RUN] No files modified.")
        return

    # Execute moves
    print("\nStep 4: Execute moves")
    affected_src, affected_dst = execute_moves(crop_index, moves)

    # Log moves
    log_frame_moves(moves)

    # Save crop_index
    print("\nSaving updated crop_index.json...")
    save_crop_index(crop_index)

    # Recompute consensus
    print("\nStep 5: Recompute byte consensus")
    update_extracted_firmware(crop_index, affected_src, affected_dst)

    # Rebuild downstream
    print("\nStep 6: Rebuild downstream files")
    rebuild_downstream()

    # Reset review state
    print("\nStep 7: Reset review state")
    reset_review_state(affected_src, affected_dst)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Source addresses affected: {len(affected_src)}")
    print(f"  Destination addresses affected: {len(affected_dst)}")

    # Verification: check known problem regions
    crop_index_check = load_crop_index()

    # Check $0D region
    d_range = [k for k in crop_index_check
               if k != 'ref_addresses' and k.startswith('0D')
               and 0x0D050 <= int(k, 16) <= 0x0DF70]
    print(f"  $0D050-$0DF70 entries in crop_index: {len(d_range)}")

    # Check $04xxx/$09xxx
    four_addrs = [k for k in crop_index_check
                  if k != 'ref_addresses' and k.startswith('04')]
    nine_addrs = [k for k in crop_index_check
                  if k != 'ref_addresses' and k.startswith('09')]
    print(f"  $04xxx entries: {len(four_addrs)}, $09xxx entries: {len(nine_addrs)}")

    # Coverage from firmware_merged.txt
    if os.path.exists(MERGED_FW_PATH):
        line_count = 0
        with open(MERGED_FW_PATH) as f:
            for line in f:
                if not line.startswith('#') and line.strip() and ':' in line:
                    line_count += 1
        print(f"  firmware_merged.txt lines: {line_count}/5120 "
              f"({line_count/5120*100:.1f}%)")

    print("=" * 60)


if __name__ == '__main__':
    main()
