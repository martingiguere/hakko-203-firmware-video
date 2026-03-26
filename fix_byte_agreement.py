#!/usr/bin/env python3
"""
Fix misassigned frames using byte-data agreement with confusion-mapped addresses.

When the OCR misreads a digit in the address column, a frame-row gets assigned
to the wrong address. This script detects outliers (frames whose byte data
disagrees with their address's consensus) and searches for a better home by
applying OCR confusion substitutions to the address digits and picking the
alternative where byte data agrees best.

Key difference from fix_outlier_votes.py:
  - fix_outlier_votes checks ±1/±2 ROW neighbors (address offset ±0x10/0x20)
  - This script checks OCR CONFUSION alternatives (e.g., $0C870 → $08870,
    $0D870, $06870, etc.)

Runs after fix_outlier_votes.py and before postprocess_firmware.py.

Usage:
    python3 fix_byte_agreement.py            # normal run
    python3 fix_byte_agreement.py --dry-run  # detect only, no changes
"""

import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations, product

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from fix_address_trajectory import (
    load_crop_index, save_crop_index, execute_moves,
    log_frame_moves, update_extracted_firmware,
    weighted_majority_vote,
)
from frame_utils import parse_frame_key
from memory_map_utils import load_memory_map, is_ff_forced, load_accepted_addresses

FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

# Expanded OCR confusion map based on empirical kNN classifier analysis.
# For each hex digit, the digits it gets misread as in the address column.
CONFUSION_MAP = {
    'A': ['8', '4'],
    'B': ['8'],
    'C': ['0', '6', 'D'],
    'D': ['0', 'C', '8'],
    'E': ['8', '6'],
    'F': ['5', '7'],
    '0': ['C', 'D', '8'],
    '1': ['7'],
    '2': [],
    '3': [],
    '4': ['A', '9'],
    '5': ['F'],
    '6': ['8', 'E'],
    '7': ['F', '9', '1'],
    '8': ['A', 'B', '6', 'E', '0', 'D'],
    '9': ['4', '7'],
}

# Thresholds
MIN_FRAMES = 4            # minimum frames per address for consensus
OUTLIER_THRESHOLD = 8     # >=8 bytes disagree with consensus = outlier
MATCH_THRESHOLD = 12      # >=12/16 bytes must match destination to move
MAX_OUTLIER_RATIO = 0.30  # skip address if >30% of frames are outliers


def generate_confusion_alternatives(addr, num_digits):
    """Generate addresses reachable by exactly num_digits OCR confusion substitutions.

    Substitutes digits at positions 0-3 (position 4 is always '0' since
    addresses are 0x10-aligned). Returns a set excluding the original.
    """
    positions = range(4)
    alternatives = set()
    for pos_combo in combinations(positions, num_digits):
        alt_lists = []
        for p in pos_combo:
            digit = addr[p].upper()
            alts = CONFUSION_MAP.get(digit, [])
            if not alts:
                break
            alt_lists.append((p, alts))
        else:
            if len(alt_lists) == num_digits:
                for combo in product(*(a for _, a in alt_lists)):
                    new_addr = list(addr)
                    for (p, _), alt_digit in zip(alt_lists, combo):
                        new_addr[p] = alt_digit
                    alternatives.add(''.join(new_addr))
    alternatives.discard(addr)
    return alternatives


def build_consensus(readings, confidences):
    """Build per-byte weighted consensus for an address."""
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


def is_all_ff(reading):
    """Check if a reading is entirely FF (would false-match any FF region)."""
    return all(b == 'FF' for b in reading[:16])


def detect_confusion_moves(crop_index, mmap):
    """Detect outlier frames whose bytes match a confusion-mapped address.

    Returns list of (source_addr, frame_key, dest_addr) tuples.
    Skips accepted addresses (user-verified data).
    """
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

    print(f"  Addresses with consensus (>={MIN_FRAMES} frames): {len(addr_consensus)}")

    moves = []
    total_outliers = 0
    skipped_ff = 0
    skipped_ratio = 0
    resolved_by_level = {1: 0, 2: 0, 3: 0}

    # Collect all outliers first
    all_outliers = []  # (addr, fk, reading, consensus)

    for addr, consensus in addr_consensus.items():
        entry = crop_index[addr]
        readings = entry.get('readings', {})

        if is_all_ff(consensus):
            continue

        outliers = []
        for fk, reading in readings.items():
            disagreements = 16 - count_matches(reading, consensus)
            if disagreements >= OUTLIER_THRESHOLD:
                if is_all_ff(reading):
                    skipped_ff += 1
                    continue
                outliers.append((fk, reading))

        if not outliers:
            continue

        if len(outliers) / len(readings) > MAX_OUTLIER_RATIO:
            skipped_ratio += len(outliers)
            continue

        total_outliers += len(outliers)
        for fk, reading in outliers:
            all_outliers.append((addr, fk, reading, consensus))

    # Cascade: try single-digit, then double, then triple
    remaining = all_outliers
    for level in (1, 2, 3):
        still_remaining = []
        for addr, fk, reading, consensus in remaining:
            current_score = count_matches(reading, consensus)
            alternatives = generate_confusion_alternatives(addr, level)
            alt_with_consensus = [(a, addr_consensus[a]) for a in alternatives
                                  if a in addr_consensus]

            best_alt = None
            best_score = current_score

            for alt_addr, alt_consensus in alt_with_consensus:
                alt_int = int(alt_addr, 16)
                if alt_int > 0x13FFF:
                    continue
                if is_ff_forced(mmap, alt_int):
                    continue

                score = count_matches(reading, alt_consensus)
                if score >= MATCH_THRESHOLD and score > best_score:
                    best_score = score
                    best_alt = alt_addr

            if best_alt:
                moves.append((addr, fk, best_alt))
                resolved_by_level[level] += 1
            else:
                still_remaining.append((addr, fk, reading, consensus))

        remaining = still_remaining

    total_resolved = sum(resolved_by_level.values())
    print(f"  Total outlier frames: {total_outliers}")
    print(f"  Skipped (all-FF readings): {skipped_ff}")
    print(f"  Skipped (outlier ratio >{MAX_OUTLIER_RATIO:.0%}): {skipped_ratio}")
    print(f"  Resolved (single-digit): {resolved_by_level[1]}")
    print(f"  Resolved (double-digit): {resolved_by_level[2]}")
    print(f"  Resolved (triple-digit): {resolved_by_level[3]}")
    print(f"  Total resolved: {total_resolved}")
    print(f"  Unresolved: {len(remaining)}")

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
    # Analyze which digit positions and substitutions are most common
    subs = Counter()
    for src, fk, dst in moves:
        for pos in range(5):
            if src[pos] != dst[pos]:
                subs[f"{src[pos]}->{dst[pos]} at pos {pos}"] += 1

    print(f"\n  Digit substitution breakdown:")
    for sub, cnt in subs.most_common(15):
        print(f"    {sub}: {cnt} moves")

    # Top source addresses
    src_counts = Counter(src for src, _, _ in moves)
    print(f"\n  Top source addresses:")
    for addr, cnt in src_counts.most_common(10):
        print(f"    ${addr}: {cnt} outlier frames moved")

    # Top destination addresses
    dst_counts = Counter(dst for _, _, dst in moves)
    print(f"\n  Top destination addresses:")
    for addr, cnt in dst_counts.most_common(10):
        print(f"    ${addr}: {cnt} frames received")


def main():
    parser = argparse.ArgumentParser(
        description='Fix misassigned frames by byte-agreement with confusion-mapped addresses')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect moves but do not modify files')
    parser.add_argument('--loop', action='store_true',
                        help='Run iteratively until no more moves found')
    args = parser.parse_args()

    mmap = load_memory_map()
    iteration = 0
    total_moves = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"Byte Agreement Address Correction{f' (iteration {iteration})' if args.loop else ''}")
        print(f"{'=' * 60}")

        crop_index = load_crop_index()
        total_addrs = sum(1 for k in crop_index if k != 'ref_addresses')
        print(f"\n  Total addresses in crop_index: {total_addrs}")

        print(f"\nStep 1: Detect outlier frames and confusion matches")
        moves = detect_confusion_moves(crop_index, mmap)

        if not moves:
            print("\nNo byte-agreement moves detected.")
            break

        summarize_moves(moves)

        if args.dry_run:
            print(f"\n  [DRY RUN] {len(moves)} moves detected, no files modified.")
            break

        print(f"\nStep 2: Execute moves")
        affected_src, affected_dst = execute_moves(crop_index, moves)

        print(f"\nStep 3: Save and update downstream")
        save_crop_index(crop_index)
        log_frame_moves(moves, strategy="byte_agreement")
        update_frame_assignments(moves)
        update_extracted_firmware(crop_index, affected_src, affected_dst)

        print(f"\n  Source addresses affected: {len(affected_src)}")
        print(f"  Destination addresses affected: {len(affected_dst)}")
        total_moves += len(moves)

        if not args.loop:
            break

    if args.loop and iteration > 1:
        print(f"\nConverged after {iteration} iterations. Total moves: {total_moves}")

    print("Done.")


if __name__ == '__main__':
    main()
