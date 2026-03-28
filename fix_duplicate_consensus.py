#!/usr/bin/env python3
"""
Fix addresses where ALL frames carry stolen data from another address.

When the OCR misreads address digits for every frame at an address, all
frames agree on the wrong data. The consensus is a perfect duplicate of
another address's consensus. This script detects such duplicates by
comparing consensus byte data across all ROM addresses, then moves the
frames from the stolen address to the source.

Layered guards prevent false merges:
  - Exact 16/16 byte match required
  - Minimum address distance 0x300
  - ROM lookup table region ($10070-$10920) exempt
  - Trajectory validation: don't steal from trajectory-confirmed addresses
  - Asymmetric frame count required (stolen < 25% of destination)
  - Single pass only (no cascade)

Key difference from fix_byte_agreement.py:
  - fix_byte_agreement: minority outlier frames disagree → move outliers
  - This script: ALL frames agree on wrong data → detect via cross-address
    consensus comparison

Runs after fix_byte_agreement.py and before postprocess_firmware.py.

Usage:
    python3 fix_duplicate_consensus.py              # single pass
    python3 fix_duplicate_consensus.py --dry-run    # detect only
"""

import argparse
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from fix_address_trajectory import (
    load_crop_index, save_crop_index, execute_moves,
    log_frame_moves, update_extracted_firmware,
)
from frame_utils import parse_frame_key
from manual_trajectory import interpolate_trajectory
from memory_map_utils import load_memory_map, is_ff_forced, load_accepted_addresses

FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

# Thresholds
MIN_FRAMES = 4           # minimum frames for consensus
MATCH_THRESHOLD = 16     # exact 16/16 byte match required (was 14)
MIN_DISTANCE = 0x300     # minimum address distance (was 0x30)
ROM_START = 0x04000
ROM_END = 0x107F0

# ROM lookup table region: genuinely repetitive calibration data, never merge
LOOKUP_TABLE_START = 0x10070
LOOKUP_TABLE_END = 0x10920

# Trajectory validation
TRAJECTORY_MARGIN = 0x400     # margin for interpolation error
MIN_TRAJECTORY_CONFIRMED = 2  # min frames confirmed at an address to protect it

# Frame count asymmetry: stolen must have < this fraction of destination's frames
MAX_STOLEN_RATIO = 0.25


def build_all_consensus(crop_index, mmap, accepted_addrs=None):
    """Build weighted consensus for all ROM addresses with enough frames.

    Returns dict of addr -> (consensus_list, frame_count, is_ff_forced).
    Includes ff-forced addresses (they can be sources of stolen data).
    Includes accepted addresses (they can be move destinations).
    Skips all-FF consensus and non-ROM addresses.
    """
    if accepted_addrs is None:
        accepted_addrs = set()
    result = {}
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses' or not isinstance(entry, dict):
            continue
        addr_int = int(addr, 16)
        if addr_int < ROM_START or addr_int > ROM_END:
            continue

        readings = entry.get('readings', {})
        if len(readings) < MIN_FRAMES:
            continue

        confidences = entry.get('confidences', {})
        consensus = []
        for i in range(16):
            votes = Counter()
            for fk, bl in readings.items():
                if i < len(bl) and bl[i] != '--':
                    w = 1.0
                    if fk in confidences:
                        cl = confidences[fk]
                        if i < len(cl):
                            w = cl[i]
                    votes[bl[i]] += w
            consensus.append(votes.most_common(1)[0][0] if votes else '--')

        # Skip all-FF consensus
        if all(b == 'FF' for b in consensus):
            continue

        result[addr] = (consensus, len(readings), is_ff_forced(mmap, addr_int))

    return result


def is_in_lookup_table(addr_key):
    """Check if address falls in the ROM lookup table region."""
    addr_int = int(addr_key, 16)
    return LOOKUP_TABLE_START <= addr_int <= LOOKUP_TABLE_END


def count_trajectory_confirmed_frames(crop_index, addr_key):
    """Count how many frames at this address are trajectory-confirmed.

    A frame is 'confirmed' if interpolate_trajectory() places the address
    within the expected screen range ± TRAJECTORY_MARGIN.

    Returns (confirmed_count, checked_count, total_frames).
    Video frames (v-prefixed) are skipped (no trajectory data).
    """
    entry = crop_index.get(addr_key, {})
    readings = entry.get('readings', {})
    addr_int = int(addr_key, 16)

    confirmed = 0
    checked = 0
    for fk in readings:
        if str(fk).startswith('v'):
            continue
        try:
            frame_int = int(fk)
        except ValueError:
            continue
        result = interpolate_trajectory(frame_int)
        if result is None:
            continue
        top, bot = result
        checked += 1
        if (top - TRAJECTORY_MARGIN) <= addr_int <= (bot + TRAJECTORY_MARGIN):
            confirmed += 1

    return confirmed, checked, len(readings)


def find_duplicate_pairs(addr_consensus, crop_index, accepted_addrs=None):
    """Find address pairs whose consensus matches at >= MATCH_THRESHOLD bytes.

    Returns list of (stolen_addr, source_addr, match_score) where stolen
    is the address whose frames should be moved to source.

    Layered guards (any one blocks the merge):
      - Both ff-forced: skip
      - Distance < MIN_DISTANCE: skip
      - Either in lookup table region: skip
      - Both accepted: skip
      - Frame count ratio >= MAX_STOLEN_RATIO (too symmetric): skip
      - Stolen has >= MIN_TRAJECTORY_CONFIRMED confirmed frames: skip
      - Source has checked frames but zero confirmed: skip
    """
    if accepted_addrs is None:
        accepted_addrs = set()
    pairs = []
    addrs = sorted(addr_consensus.keys())

    stats = {
        'total_compared': 0,
        'below_threshold': 0,
        'both_ff': 0,
        'too_close': 0,
        'lookup_table': 0,
        'both_accepted': 0,
        'symmetric_frames': 0,
        'trajectory_stolen_confirmed': 0,
        'trajectory_source_unconfirmed': 0,
    }

    for i, a1 in enumerate(addrs):
        c1, n1, ff1 = addr_consensus[a1]
        for a2 in addrs[i + 1:]:
            c2, n2, ff2 = addr_consensus[a2]
            stats['total_compared'] += 1

            if ff1 and ff2:
                stats['both_ff'] += 1
                continue

            dist = abs(int(a1, 16) - int(a2, 16))
            if dist < MIN_DISTANCE:
                stats['too_close'] += 1
                continue

            # Exempt lookup table region (genuinely repetitive calibration data)
            if is_in_lookup_table(a1) or is_in_lookup_table(a2):
                stats['lookup_table'] += 1
                continue

            matches = sum(1 for k in range(16)
                          if c1[k] != '--' and c2[k] != '--' and c1[k] == c2[k])
            if matches < MATCH_THRESHOLD:
                stats['below_threshold'] += 1
                continue

            a1_accepted = a1 in accepted_addrs
            a2_accepted = a2 in accepted_addrs
            if a1_accepted and a2_accepted:
                stats['both_accepted'] += 1
                continue

            # Determine stolen vs source
            if ff1 and not ff2:
                stolen, source = a1, a2
            elif ff2 and not ff1:
                stolen, source = a2, a1
            elif a2_accepted:
                stolen, source = a1, a2
            elif a1_accepted:
                stolen, source = a2, a1
            else:
                # Neither ff-forced nor accepted: require strong asymmetry
                if n1 == n2:
                    stats['symmetric_frames'] += 1
                    continue
                if n1 < n2:
                    if n1 / n2 >= MAX_STOLEN_RATIO:
                        stats['symmetric_frames'] += 1
                        continue
                    stolen, source = a1, a2
                else:
                    if n2 / n1 >= MAX_STOLEN_RATIO:
                        stats['symmetric_frames'] += 1
                        continue
                    stolen, source = a2, a1

            # Trajectory validation: don't steal from trajectory-confirmed addresses
            stolen_conf, stolen_chk, _ = count_trajectory_confirmed_frames(
                crop_index, stolen)
            if stolen_conf >= MIN_TRAJECTORY_CONFIRMED:
                stats['trajectory_stolen_confirmed'] += 1
                continue

            source_conf, source_chk, _ = count_trajectory_confirmed_frames(
                crop_index, source)
            if source_chk > 0 and source_conf == 0:
                stats['trajectory_source_unconfirmed'] += 1
                continue

            pairs.append((stolen, source, matches))

    # Print rejection stats
    print(f"\n  Pair filtering stats:")
    print(f"    Pairs compared: {stats['total_compared']}")
    print(f"    Below match threshold ({MATCH_THRESHOLD}/16): {stats['below_threshold']}")
    print(f"    Too close (< ${MIN_DISTANCE:X}): {stats['too_close']}")
    print(f"    Lookup table exempt: {stats['lookup_table']}")
    print(f"    Both ff-forced: {stats['both_ff']}")
    print(f"    Both accepted: {stats['both_accepted']}")
    print(f"    Symmetric frame counts (>= {MAX_STOLEN_RATIO:.0%}): {stats['symmetric_frames']}")
    print(f"    Trajectory: stolen confirmed: {stats['trajectory_stolen_confirmed']}")
    print(f"    Trajectory: source unconfirmed: {stats['trajectory_source_unconfirmed']}")
    print(f"    Pairs accepted: {len(pairs)}")

    return pairs


def detect_duplicate_moves(crop_index, mmap):
    """Detect stolen addresses and build move list.

    Returns list of (src_addr, frame_key, dst_addr) tuples.
    Skips accepted addresses (user-verified data).
    """
    accepted_addrs, _ = load_accepted_addresses()
    addr_consensus = build_all_consensus(crop_index, mmap, accepted_addrs)
    print(f"  ROM addresses with consensus: {len(addr_consensus)}")

    pairs = find_duplicate_pairs(addr_consensus, crop_index, accepted_addrs)
    print(f"  Duplicate pairs found: {len(pairs)}")

    # Deduplicate: a stolen address might match multiple sources.
    # Pick the source with the highest match score (then most frames).
    best_source = {}
    for stolen, source, score in pairs:
        _, source_n, _ = addr_consensus[source]
        prev = best_source.get(stolen)
        if prev is None or score > prev[1] or (score == prev[1] and source_n > addr_consensus[prev[0]][1]):
            best_source[stolen] = (source, score)

    print(f"  Unique stolen addresses: {len(best_source)}")

    # Build moves: move ALL frames from stolen to source
    moves = []
    for stolen_addr, (source_addr, score) in best_source.items():
        entry = crop_index.get(stolen_addr, {})
        readings = entry.get('readings', {})
        for fk in readings:
            moves.append((stolen_addr, fk, source_addr))

    return moves


def update_frame_assignments(moves):
    """Update frame_assignments.json for moved frames."""
    if not os.path.exists(FRAME_ASSIGNMENTS_PATH):
        return

    with open(FRAME_ASSIGNMENTS_PATH) as f:
        assignments = json.load(f)

    updated = 0
    for src_addr, frame_key, dst_addr in moves:
        frame_int, is_video = parse_frame_key(str(frame_key))
        if is_video:
            continue
        frame_name = f"frame_{frame_int:05d}.png"
        if frame_name in assignments:
            dst_int = int(dst_addr, 16)
            for a in assignments[frame_name]:
                if f"{a['addr']:05X}" == src_addr:
                    a['addr'] = dst_int
                    updated += 1

    with open(FRAME_ASSIGNMENTS_PATH, 'w') as f:
        json.dump(assignments, f)
    print(f"  Updated {updated} entries in frame_assignments.json")


def summarize_moves(moves):
    """Print move summary by digit substitution type."""
    subs = Counter()
    stolen_addrs = Counter()
    for src, fk, dst in moves:
        stolen_addrs[src] += 1
        for pos in range(5):
            if src[pos] != dst[pos]:
                subs[f"{src[pos]}->{dst[pos]} at pos {pos}"] += 1

    print(f"\n  Digit substitution breakdown:")
    for sub, cnt in subs.most_common(15):
        print(f"    {sub}: {cnt} frame moves")

    print(f"\n  Top stolen addresses (all frames moved):")
    for addr, cnt in stolen_addrs.most_common(10):
        dst = None
        for s, f, d in moves:
            if s == addr:
                dst = d
                break
        print(f"    ${addr} -> ${dst}: {cnt} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Fix addresses with 100% stolen frame data via consensus comparison')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect duplicates but do not modify files')
    parser.add_argument('--loop', action='store_true',
                        help='[DEPRECATED] Ignored. Script now runs a single pass only.')
    args = parser.parse_args()

    if args.loop:
        print("WARNING: --loop is deprecated and ignored. Running single pass.")

    mmap = load_memory_map()

    print(f"\n{'=' * 60}")
    print(f"Duplicate Consensus Correction")
    print(f"{'=' * 60}")

    crop_index = load_crop_index()
    total_addrs = sum(1 for k in crop_index if k != 'ref_addresses')
    print(f"\n  Total addresses in crop_index: {total_addrs}")

    print(f"\nStep 1: Detect duplicate consensuses")
    moves = detect_duplicate_moves(crop_index, mmap)

    if not moves:
        print("\nNo duplicate consensus moves detected.")
        print("Done.")
        return

    summarize_moves(moves)

    if args.dry_run:
        print(f"\n  [DRY RUN] {len(moves)} moves detected, no files modified.")
        print("Done.")
        return

    print(f"\nStep 2: Execute moves")
    affected_src, affected_dst = execute_moves(crop_index, moves)

    print(f"\nStep 3: Save and update downstream")
    save_crop_index(crop_index)
    log_frame_moves(moves, strategy="duplicate_consensus")
    update_frame_assignments(moves)
    update_extracted_firmware(crop_index, affected_src, affected_dst)

    print(f"\n  Source addresses affected: {len(affected_src)}")
    print(f"  Destination addresses affected: {len(affected_dst)}")
    print(f"  Total moves: {len(moves)}")
    print("Done.")


if __name__ == '__main__':
    main()
