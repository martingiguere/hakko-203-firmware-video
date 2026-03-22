#!/usr/bin/env python3
"""
Strategy 8: Global Address Trajectory Correction.

Two-phase correction of address misassignments in crop_index.json:

Phase 1 — Manual trajectory plausibility check (NEW):
  Uses the manually confirmed scroll trajectory (manual_trajectory.py) to
  detect frames whose crop_index address falls outside the expected screen
  position.  Re-reads the frame with trajectory constraints to find the
  correct address; falls back to trajectory interpolation if re-read fails.
  Catches ALL misassignments, not just confusion pairs.

Phase 2 — Confusion-pair refinement (existing):
  Rebuilds anchor trajectory from the corrected crop_index and applies
  confusion-pair swaps (C<->D, 4<->9, 8<->6) for subtle corrections
  the manual trajectory margin can't catch.

Steps:
  1. Load crop_index + manual trajectory
  2. Phase 1: detect implausible frames, re-read with constraints, move
  3. Phase 2: rebuild anchor trajectory, detect confusion-pair swaps, move
  4. Execute all moves, rebuild downstream
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

import cv2
import numpy as np

from frame_utils import (
    is_video_frame_key, video_frame_key, crop_filename, parse_frame_key,
)
from manual_trajectory import interpolate_trajectory, get_waypoint_spacing

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
                'row_ys': {},
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

            # Move readings, confidences, and row_ys
            if frame_str in src_entry.get('readings', {}):
                dst_entry.setdefault('readings', {})[frame_str] = \
                    src_entry['readings'].pop(frame_str)
            if frame_str in src_entry.get('confidences', {}):
                dst_entry.setdefault('confidences', {})[frame_str] = \
                    src_entry['confidences'].pop(frame_str)
            if frame_str in src_entry.get('row_ys', {}):
                dst_entry.setdefault('row_ys', {})[frame_str] = \
                    src_entry['row_ys'].pop(frame_str)

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


def log_frame_moves(moves, strategy="trajectory"):
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
            "strategy": strategy,
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
    """Reset review status for affected addresses.

    Only clears the status to 'unreviewed' — does NOT update byte data.
    Byte data gets updated when postprocess regenerates firmware_merged.txt
    and the app detects staleness on next startup.
    """
    if not os.path.exists(REVIEW_STATE_PATH):
        print("  review_state.json not found -- skipping")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    if 'lines' not in state:
        print("  review_state.json has no 'lines' -- skipping")
        return

    all_affected = affected_src | affected_dst
    reset_count = 0

    for addr_key in sorted(all_affected):
        if addr_key in state['lines']:
            state['lines'][addr_key]['status'] = 'unreviewed'
            state['lines'][addr_key]['edited_positions'] = []
            reset_count += 1

    with open(REVIEW_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print(f"  Reset {reset_count} addresses in review_state.json")


# ── Phase 1: Manual trajectory plausibility ───────────────────────────────────

# Margin around expected screen range for plausibility check.
# Applied when waypoints are close enough for reliable interpolation.
PLAUSIBILITY_MARGIN = 0x300

# Maximum waypoint spacing (in frames) for Phase 1 to apply.
# Beyond this, trajectory interpolation is unreliable (scroll speed varies
# too much for linear interpolation to be meaningful).
# 500 frames = ~17 seconds at 30fps.  The densely-sampled regions
# (oscillation zone, paused regions) all have spacing < 500.
MAX_WAYPOINT_SPACING = 500


def detect_implausible_frames(crop_index):
    """Find (addr, frame, expected_top) tuples where addr is outside
    the expected screen range from the manual trajectory.

    Only checks extracted frames (not video frames, which don't have PNGs
    for re-reading).  Skips frames in trajectory regions where waypoints
    are too far apart for reliable interpolation.
    """
    implausible = []
    no_trajectory = 0
    skipped_sparse = 0

    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        addr_val = int(addr, 16)

        for frame in entry.get('frames', []):
            result = interpolate_trajectory(frame)
            if result is None:
                no_trajectory += 1
                continue

            spacing = get_waypoint_spacing(frame)
            if spacing is not None and spacing > MAX_WAYPOINT_SPACING:
                skipped_sparse += 1
                continue

            top, bottom = result
            lo = max(0, top - PLAUSIBILITY_MARGIN)
            hi = bottom + PLAUSIBILITY_MARGIN
            if addr_val < lo or addr_val > hi:
                implausible.append((addr, frame, top))

    if no_trajectory:
        print(f"  Skipped {no_trajectory} frames (no trajectory data)")
    if skipped_sparse:
        print(f"  Skipped {skipped_sparse} frames (waypoint spacing > "
              f"{MAX_WAYPOINT_SPACING})")
    return implausible


def constrained_reread(classifier, frame_num, expected_top):
    """Re-read addresses from a frame PNG with trajectory constraints.

    Returns the best address (as 5-char hex string) for the top row,
    or None if re-read fails entirely.
    """
    from template_matcher import CAL, ADDR_X_START, CELL_H
    from extract_pipeline import read_address_from_row, validate_address_sequence

    frame_path = os.path.join(PROJECT_ROOT, 'frames',
                              f'frame_{frame_num:05d}.png')
    if not os.path.exists(frame_path):
        return None

    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']

    # Read all addresses from the frame
    raw_reads = []
    for row_idx in range(-2, num_rows):
        row_y = y_first + row_idx * row_h
        if row_y < CELL_H or row_y > img.shape[0] - CELL_H:
            continue
        addr_str, addr_conf = read_address_from_row(
            classifier, img, row_y, ADDR_X_START
        )
        try:
            addr_int = int(addr_str, 16)
            if 0x00000 <= addr_int <= 0x13FFF:
                raw_reads.append((addr_int, row_y, addr_conf))
        except ValueError:
            pass

    if not raw_reads:
        return None

    # Filter: keep only reads within expected range (screen span + margin)
    filtered = [(a, y, c) for a, y, c in raw_reads
                if expected_top - 0x100 <= a <= expected_top + 0x200]

    if len(filtered) >= 2:
        validated = validate_address_sequence(filtered)
    elif len(raw_reads) >= 2:
        # Not enough filtered reads — try full set but validate
        validated = validate_address_sequence(raw_reads)
    else:
        return None

    if not validated:
        return None

    # Find the topmost validated row and compute the implied top address
    validated.sort(key=lambda x: x[1])  # sort by row_y
    top_addr_int, top_row_y, _ = validated[0]

    # How many rows above y_first is this row?
    row_offset = round((top_row_y - y_first) / row_h)
    screen_top = top_addr_int - row_offset * 0x10

    # Check plausibility against trajectory
    if abs(screen_top - expected_top) <= PLAUSIBILITY_MARGIN:
        return f"{screen_top:05X}"

    # Fallback: use trajectory directly
    return None


def build_frame_to_addrs(crop_index):
    """Pre-build frame → sorted list of addr_vals index for fast lookup."""
    frame_to_addrs = {}
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        addr_val = int(addr, 16)
        for f in entry.get('frames', []):
            frame_to_addrs.setdefault(f, []).append(addr_val)
    for f in frame_to_addrs:
        frame_to_addrs[f].sort()
    return frame_to_addrs


def build_phase1_moves(crop_index, implausible, classifier):
    """For each implausible (addr, frame, expected_top), try constrained
    re-read to find the correct address.  Returns moves list.
    """
    moves = []
    reread_success = 0
    trajectory_fallback = 0
    skipped = 0

    # Pre-build frame→addrs index once
    frame_to_addrs = build_frame_to_addrs(crop_index)

    # Group by frame for efficiency (avoid re-loading same frame)
    by_frame = {}
    for addr, frame, expected_top in implausible:
        by_frame.setdefault(frame, []).append((addr, expected_top))

    total_frames = len(by_frame)
    for idx, frame in enumerate(sorted(by_frame)):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Processing frame {idx+1}/{total_frames}...")

        items = by_frame[frame]
        expected_top = items[0][1]

        # Try constrained re-read once per frame
        new_top_hex = constrained_reread(classifier, frame, expected_top)

        # Get all addrs assigned to this frame (from pre-built index)
        frame_addrs = frame_to_addrs.get(frame, [])
        if not frame_addrs:
            skipped += len(items)
            continue
        old_top = frame_addrs[0]

        for addr, _ in items:
            addr_val = int(addr, 16)

            if new_top_hex is not None:
                new_top_val = int(new_top_hex, 16)
            else:
                new_top_val = expected_top

            row_offset = (addr_val - old_top) // 0x10
            new_addr_val = new_top_val + row_offset * 0x10
            new_addr = f"{new_addr_val:05X}"

            if new_addr != addr:
                moves.append((addr, frame, new_addr))
                if new_top_hex is not None:
                    reread_success += 1
                else:
                    trajectory_fallback += 1

    print(f"  Re-read success: {reread_success} moves")
    print(f"  Trajectory fallback: {trajectory_fallback} moves")
    print(f"  Skipped: {skipped}")
    return moves


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
    parser.add_argument('--phase1-only', action='store_true',
                        help='Only run Phase 1 (manual trajectory)')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Only run Phase 2 (confusion pairs)')
    args = parser.parse_args()

    run_phase1 = not args.phase2_only
    run_phase2 = not args.phase1_only

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

    all_moves = []
    all_affected_src = set()
    all_affected_dst = set()

    # ── Phase 1: Manual trajectory plausibility ──────────────────────────
    if run_phase1:
        print("\n" + "─" * 60)
        print("Phase 1: Manual Trajectory Plausibility Check")
        print("─" * 60)

        print("\nStep 1.1: Detect implausible frames")
        implausible = detect_implausible_frames(crop_index)
        print(f"  Implausible frames detected: {len(implausible)}")

        if implausible:
            # Show sample
            sample = implausible[:10]
            print(f"  Sample (first {len(sample)}):")
            for addr, frame, expected_top in sample:
                result = interpolate_trajectory(frame)
                exp_range = f"${expected_top:05X}-${result[1]:05X}" if result else "?"
                print(f"    {addr} @ F{frame} — expected {exp_range}")
            if len(implausible) > 10:
                print(f"    ... and {len(implausible) - 10} more")

            print("\nStep 1.2: Constrained re-read + trajectory fallback")
            # Load classifier for re-reading
            from extract_pipeline import FastKNNClassifier
            classifier = FastKNNClassifier()
            classifier_path = os.path.join(PROJECT_ROOT,
                                           'fast_knn_classifier.npz')
            classifier.load(classifier_path)

            phase1_moves = build_phase1_moves(crop_index, implausible,
                                              classifier)
            print(f"  Phase 1 moves: {len(phase1_moves)}")

            if phase1_moves:
                summarize_moves(phase1_moves)

                if not args.dry_run:
                    print("\nStep 1.3: Execute Phase 1 moves (in memory)")
                    src, dst = execute_moves(crop_index, phase1_moves)
                    all_affected_src |= src
                    all_affected_dst |= dst
                    # Defer save_crop_index and log_frame_moves until both phases complete

                all_moves.extend(phase1_moves)
        else:
            print("  No implausible frames found.")

    # ── Phase 2: Confusion-pair refinement ───────────────────────────────
    if run_phase2:
        print("\n" + "─" * 60)
        print("Phase 2: Confusion-Pair Refinement")
        print("─" * 60)

        # Phase 1 may have modified crop_index in memory — use it directly
        # (no reload from disk needed since we deferred saves)

        print("\nStep 2.1: Build anchor trajectories")
        anchors_extracted = build_anchor_trajectory(crop_index,
                                                    use_extracted=True)
        anchors_video = build_anchor_trajectory(crop_index,
                                                use_extracted=False)
        print(f"  Extracted anchor points: {len(anchors_extracted)}")
        print(f"  Video anchor points: {len(anchors_video)}")

        print("\nStep 2.2: Detect breakpoints")
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

        segments_extracted = build_segments(anchors_extracted, bp_extracted)
        segments_video = build_segments(anchors_video, bp_video)
        print(f"  Extracted segments: {len(segments_extracted)}")
        for i, seg in enumerate(segments_extracted):
            if seg:
                direction = ("increasing" if seg[-1][1] >= seg[0][1]
                             else "decreasing")
                print(f"    Segment {i}: frames {seg[0][0]}-{seg[-1][0]}, "
                      f"addr ${int(seg[0][1]):05X}-"
                      f"${int(seg[-1][1]):05X} ({direction})")
        if segments_video:
            print(f"  Video segments: {len(segments_video)}")
            for i, seg in enumerate(segments_video):
                if seg:
                    direction = ("increasing" if seg[-1][1] >= seg[0][1]
                                 else "decreasing")
                    print(f"    Segment {i}: frames {seg[0][0]}-"
                          f"{seg[-1][0]}, addr ${seg[0][1]:05X}-"
                          f"${seg[-1][1]:05X} ({direction})")

        print("\nStep 2.3: Detect confusion-pair corrections")
        phase2_moves = detect_trajectory_moves(crop_index,
                                               segments_extracted,
                                               segments_video)
        print(f"  Phase 2 moves detected: {len(phase2_moves)}")

        if phase2_moves:
            summarize_moves(phase2_moves)

            if not args.dry_run:
                print("\nStep 2.4: Execute Phase 2 moves (in memory)")
                src, dst = execute_moves(crop_index, phase2_moves)
                all_affected_src |= src
                all_affected_dst |= dst
                # Defer save_crop_index and log_frame_moves until finalize

            all_moves.extend(phase2_moves)

    # ── Finalize ─────────────────────────────────────────────────────────
    total_moves = len(all_moves)
    print(f"\n  Total moves across all phases: {total_moves}")

    if not all_moves:
        print("\nNo corrections needed. Nothing to do.")
        return

    if args.dry_run:
        print("\n  [DRY RUN] No files modified.")
        return

    # Atomic save: all file writes happen together after both phases complete
    print("\nStep 4: Save all changes")
    save_crop_index(crop_index)
    log_frame_moves(all_moves, strategy="trajectory_correction")

    # Update frame_assignments.json if it exists
    assignments_path = os.path.join(PROJECT_ROOT, 'frame_assignments.json')
    if os.path.exists(assignments_path):
        import json as _json
        with open(assignments_path) as f:
            frame_assignments = _json.load(f)
        # Update assignments for moved frames
        for src_addr, frame_key, dst_addr in all_moves:
            # Find frame_name from frame_key
            frame_int, is_video = parse_frame_key(str(frame_key))
            if is_video:
                continue  # video frames not in frame_assignments
            frame_name = f"frame_{frame_int:05d}.png"
            if frame_name in frame_assignments:
                for a in frame_assignments[frame_name]:
                    if f"{a['addr']:05X}" == src_addr:
                        a['addr'] = int(dst_addr, 16)
        with open(assignments_path, 'w') as f:
            _json.dump(frame_assignments, f)
        print(f"  Updated {assignments_path}")

    # Recompute consensus
    print("\nStep 5: Recompute byte consensus")
    update_extracted_firmware(crop_index, all_affected_src, all_affected_dst)

    # Rebuild downstream
    print("\nStep 6: Rebuild downstream files")
    rebuild_downstream()

    # Reset review state
    print("\nStep 7: Reset review state")
    reset_review_state(all_affected_src, all_affected_dst)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Source addresses affected: {len(all_affected_src)}")
    print(f"  Destination addresses affected: {len(all_affected_dst)}")

    # Verification: check known problem regions
    crop_index_check = load_crop_index()

    d_range = [k for k in crop_index_check
               if k != 'ref_addresses' and k.startswith('0D')
               and 0x0D050 <= int(k, 16) <= 0x0DF70]
    print(f"  $0D050-$0DF70 entries in crop_index: {len(d_range)}")

    four_addrs = [k for k in crop_index_check
                  if k != 'ref_addresses' and k.startswith('04')]
    nine_addrs = [k for k in crop_index_check
                  if k != 'ref_addresses' and k.startswith('09')]
    print(f"  $04xxx entries: {len(four_addrs)}, "
          f"$09xxx entries: {len(nine_addrs)}")

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
