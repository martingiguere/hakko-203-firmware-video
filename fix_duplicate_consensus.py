#!/usr/bin/env python3
"""
Fix addresses where ALL frames carry stolen data from another address.

When the OCR misreads address digits for every frame at an address, all
frames agree on the wrong data. The consensus is a perfect duplicate of
another address's consensus. This script detects such duplicates by
comparing consensus byte data across all ROM addresses, then moves the
frames from the stolen address to the source (the one with more frames).

Key difference from fix_byte_agreement.py:
  - fix_byte_agreement: minority outlier frames disagree → move outliers
  - This script: ALL frames agree on wrong data → detect via cross-address
    consensus comparison

Runs after fix_byte_agreement.py and before postprocess_firmware.py.

Usage:
    python3 fix_duplicate_consensus.py              # single pass
    python3 fix_duplicate_consensus.py --dry-run    # detect only
    python3 fix_duplicate_consensus.py --loop       # iterate until convergence
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
from memory_map_utils import load_memory_map, is_ff_forced

FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

# Thresholds
MIN_FRAMES = 4           # minimum frames for consensus
MATCH_THRESHOLD = 14     # bytes matching to consider duplicate
MIN_DISTANCE = 0x30      # minimum address distance (skip neighbors)
ROM_START = 0x04000
ROM_END = 0x107F0


def build_all_consensus(crop_index, mmap):
    """Build weighted consensus for all ROM addresses with enough frames.

    Returns dict of addr -> (consensus_list, frame_count, is_ff_forced).
    Includes ff-forced addresses (they can be sources of stolen data).
    Skips all-FF consensus and non-ROM addresses.
    """
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


def find_duplicate_pairs(addr_consensus):
    """Find address pairs whose consensus matches at >= MATCH_THRESHOLD bytes.

    Returns list of (stolen_addr, source_addr, match_score) where stolen
    is the address with fewer frames (its data came from source).
    FF-forced addresses are always considered stolen (their data gets
    overwritten with FF anyway — moving frames OUT is purely beneficial).
    Skips pairs closer than MIN_DISTANCE, and pairs where both are ff-forced.
    """
    pairs = []
    addrs = sorted(addr_consensus.keys())

    for i, a1 in enumerate(addrs):
        c1, n1, ff1 = addr_consensus[a1]
        for a2 in addrs[i + 1:]:
            c2, n2, ff2 = addr_consensus[a2]

            # Skip if both ff-forced (nowhere to move to)
            if ff1 and ff2:
                continue

            dist = abs(int(a1, 16) - int(a2, 16))
            if dist < MIN_DISTANCE:
                continue

            matches = sum(1 for k in range(16)
                          if c1[k] != '--' and c2[k] != '--' and c1[k] == c2[k])
            if matches < MATCH_THRESHOLD:
                continue

            # FF-forced is always the stolen one
            if ff1 and not ff2:
                pairs.append((a1, a2, matches))
            elif ff2 and not ff1:
                pairs.append((a2, a1, matches))
            else:
                # Neither ff-forced: fewer frames = stolen
                if n1 == n2:
                    continue
                if n1 < n2:
                    pairs.append((a1, a2, matches))
                else:
                    pairs.append((a2, a1, matches))

    return pairs


def detect_duplicate_moves(crop_index, mmap):
    """Detect stolen addresses and build move list.

    Returns list of (src_addr, frame_key, dst_addr) tuples.
    """
    addr_consensus = build_all_consensus(crop_index, mmap)
    print(f"  ROM addresses with consensus: {len(addr_consensus)}")

    pairs = find_duplicate_pairs(addr_consensus)
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
                        help='Run iteratively until no more moves found')
    args = parser.parse_args()

    mmap = load_memory_map()
    iteration = 0
    total_moves = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"Duplicate Consensus Correction{f' (iteration {iteration})' if args.loop else ''}")
        print(f"{'=' * 60}")

        crop_index = load_crop_index()
        total_addrs = sum(1 for k in crop_index if k != 'ref_addresses')
        print(f"\n  Total addresses in crop_index: {total_addrs}")

        print(f"\nStep 1: Detect duplicate consensuses")
        moves = detect_duplicate_moves(crop_index, mmap)

        if not moves:
            print("\nNo duplicate consensus moves detected.")
            break

        summarize_moves(moves)

        if args.dry_run:
            print(f"\n  [DRY RUN] {len(moves)} moves detected, no files modified.")
            break

        print(f"\nStep 2: Execute moves")
        affected_src, affected_dst = execute_moves(crop_index, moves)

        print(f"\nStep 3: Save and update downstream")
        save_crop_index(crop_index)
        log_frame_moves(moves, strategy="duplicate_consensus")
        update_frame_assignments(moves)
        update_extracted_firmware(crop_index, affected_src, affected_dst)

        print(f"\n  Source addresses affected: {len(affected_src)}")
        print(f"  Destination addresses affected: {len(affected_dst)}")
        total_moves += len(moves)
        print(f"  Total moves this run: {total_moves}")

        if not args.loop:
            break

    if args.loop and iteration > 1:
        print(f"\nConverged after {iteration} iterations. Total moves: {total_moves}")

    print("Done.")


if __name__ == '__main__':
    main()
