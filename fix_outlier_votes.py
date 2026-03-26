#!/usr/bin/env python3
"""
Fix outlier frames whose byte data belongs to an adjacent address.

Scroll-transition frames can be assigned to the correct address but
contain byte data from a neighboring row (off by ±1 or ±2 rows).
This script detects such outliers by comparing each frame's readings
to the per-byte consensus, then moves frames whose data matches a
neighbor's consensus better.

Runs after fix_address_trajectory.py and before postprocess_firmware.py.

Usage:
    python3 fix_outlier_votes.py            # normal run
    python3 fix_outlier_votes.py --dry-run  # detect only, no changes
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
    weighted_majority_vote,
)
from frame_utils import parse_frame_key

FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

# Thresholds
MIN_FRAMES = 4          # minimum frames per address for consensus
OUTLIER_THRESHOLD = 8   # >50% bytes disagree = outlier
MATCH_THRESHOLD = 12    # 75%+ bytes must match neighbor to move
NEIGHBOR_OFFSETS = [-2, -1, 1, 2]  # ±1 and ±2 rows


def build_consensus(readings, confidences):
    """Build per-byte weighted consensus for an address.

    Returns list of 16 hex strings (majority vote per byte).
    """
    result = []
    for i in range(16):
        votes = Counter()
        for fk, byte_list in readings.items():
            if i < len(byte_list) and byte_list[i] != '--':
                weight = 1.0
                if fk in confidences:
                    conf_list = confidences[fk]
                    if i < len(conf_list):
                        weight = conf_list[i]
                votes[byte_list[i]] += weight
        result.append(votes.most_common(1)[0][0] if votes else '--')
    return result


def count_matches(reading, consensus):
    """Count how many bytes in reading match consensus."""
    return sum(1 for i in range(min(16, len(reading)))
               if reading[i] != '--' and consensus[i] != '--'
               and reading[i] == consensus[i])


def detect_outlier_moves(crop_index):
    """Detect outlier frames and find better neighbor assignments.

    Returns list of (source_addr, frame_key, dest_addr) tuples.
    Skips accepted addresses (user-verified data).
    """
    from memory_map_utils import load_accepted_addresses
    accepted_addrs, _ = load_accepted_addresses()

    # Pre-build consensus for all addresses with enough frames
    addr_consensus = {}
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses' or not isinstance(entry, dict):
            continue
        if addr in accepted_addrs:
            continue
        readings = entry.get('readings', {})
        if len(readings) < MIN_FRAMES:
            continue
        confidences = entry.get('confidences', {})
        addr_consensus[addr] = build_consensus(readings, confidences)

    print(f"  Addresses with consensus: {len(addr_consensus)}")

    # Find outliers and match to neighbors
    moves = []
    total_outliers = 0

    for addr, consensus in addr_consensus.items():
        entry = crop_index[addr]
        readings = entry.get('readings', {})
        addr_int = int(addr, 16)

        for fk, reading in readings.items():
            # Count disagreements with own consensus
            current_matches = count_matches(reading, consensus)
            disagreements = 16 - current_matches

            if disagreements < OUTLIER_THRESHOLD:
                continue

            total_outliers += 1

            # Check neighbor consensuses
            best_neighbor = None
            best_score = current_matches

            for offset in NEIGHBOR_OFFSETS:
                neighbor_int = addr_int + offset * 0x10
                neighbor_addr = f"{neighbor_int:05X}"
                if neighbor_addr not in addr_consensus:
                    continue

                n_consensus = addr_consensus[neighbor_addr]
                n_matches = count_matches(reading, n_consensus)

                if n_matches >= MATCH_THRESHOLD and n_matches > best_score:
                    best_score = n_matches
                    best_neighbor = (neighbor_addr, offset, n_matches)

            if best_neighbor:
                dest_addr, offset, score = best_neighbor
                moves.append((addr, fk, dest_addr))

    print(f"  Total outlier frames: {total_outliers}")
    print(f"  Moves to neighbors: {len(moves)}")

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
    """Print move summary by offset direction."""
    offsets = Counter()
    for src, fk, dst in moves:
        diff = (int(dst, 16) - int(src, 16)) // 0x10
        offsets[diff] += 1

    print(f"\n  Move direction breakdown:")
    for off, cnt in sorted(offsets.items()):
        print(f"    {off:+d} row: {cnt} frames")

    # Show top source addresses
    src_counts = Counter(src for src, _, _ in moves)
    print(f"\n  Top source addresses:")
    for addr, cnt in src_counts.most_common(10):
        print(f"    ${addr}: {cnt} outlier frames moved")


def main():
    parser = argparse.ArgumentParser(
        description='Fix outlier frames by byte-consensus neighbor matching')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect outliers but do not modify files')
    args = parser.parse_args()

    print("=" * 60)
    print("Outlier Vote Correction")
    print("=" * 60)

    crop_index = load_crop_index()
    total_addrs = sum(1 for k in crop_index if k != 'ref_addresses')
    print(f"\n  Total addresses in crop_index: {total_addrs}")

    print(f"\nStep 1: Detect outlier frames and neighbor matches")
    moves = detect_outlier_moves(crop_index)

    if not moves:
        print("\nNo outlier moves detected.")
        return

    summarize_moves(moves)

    if args.dry_run:
        print(f"\n  [DRY RUN] {len(moves)} moves detected, no files modified.")
        return

    print(f"\nStep 2: Execute moves")
    affected_src, affected_dst = execute_moves(crop_index, moves)

    print(f"\nStep 3: Save and update downstream")
    save_crop_index(crop_index)
    log_frame_moves(moves, strategy="outlier_vote")
    update_frame_assignments(moves)
    update_extracted_firmware(crop_index, affected_src, affected_dst)

    print(f"\n  Source addresses affected: {len(affected_src)}")
    print(f"  Destination addresses affected: {len(affected_dst)}")
    print("Done.")


if __name__ == '__main__':
    main()
