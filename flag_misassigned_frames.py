#!/usr/bin/env python3
"""
Flag addresses with misassigned frames for manual review.

Detects ROM addresses where multiple frames have byte data that strongly
disagrees with the address consensus (8+ of 16 bytes differ). These are
frames assigned to the wrong address by OCR errors — their data belongs
to a different address.

This script only flags addresses in review_state.json for manual review.
It does NOT move frames or modify crop_index.json.

Usage:
    python3 flag_misassigned_frames.py              # flag addresses
    python3 flag_misassigned_frames.py --dry-run    # detect only
    python3 flag_misassigned_frames.py --threshold 6 --min-wrong 3
"""

# TODO: Future auto-move improvement
# For each wrong frame, the correct destination could be found automatically:
#   1. Use interpolate_trajectory(frame) to get expected screen position
#   2. Search all addresses whose consensus matches the frame's reading (>=12/16 bytes)
#   3. If trajectory + byte match agree on a single destination → auto-move
#   4. Call apply_single_move() / log_frame_moves(strategy="misassigned_fix")
#   5. Update frame_assignments, extracted_firmware, recompute consensus
# This would use the same move infrastructure as fix_byte_agreement.py but
# with trajectory validation instead of OCR confusion pairs.
# Not implemented yet — flag-only for now, user reviews and moves manually.

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from memory_map_utils import load_memory_map, is_ff_forced, load_accepted_addresses

# Thresholds
DISAGREEMENT_THRESHOLD = 8  # bytes that must disagree to mark frame as wrong
MIN_WRONG_FRAMES = 2        # minimum wrong frames to flag an address
MIN_FRAMES = 4              # minimum total frames for consensus
ROM_START = 0x04000
ROM_END = 0x107F0

CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')


def build_consensus(readings, confidences):
    """Build weighted majority vote consensus for 16 byte positions."""
    result = []
    for i in range(16):
        votes = Counter()
        for fk, rd in readings.items():
            if i < len(rd) and rd[i] != '--':
                w = 1.0
                if fk in confidences:
                    cl = confidences[fk]
                    if i < len(cl):
                        w = cl[i]
                votes[rd[i]] += w
        result.append(votes.most_common(1)[0][0] if votes else '--')
    return result


def find_misassigned_addresses(crop_index, mmap, accepted_addrs,
                                disagreement_threshold, min_wrong):
    """Find addresses with multiple frames that disagree with consensus.

    Returns list of (addr, total_frames, wrong_count, wrong_pct, wrong_frames)
    sorted by wrong_count descending.
    """
    results = []

    for addr_str, entry in crop_index.items():
        if not isinstance(entry, dict) or addr_str == 'ref_addresses':
            continue
        addr_int = int(addr_str, 16)
        if addr_int < ROM_START or addr_int > ROM_END:
            continue
        if is_ff_forced(mmap, addr_int):
            continue
        if addr_str in accepted_addrs:
            continue

        readings = entry.get('readings', {})
        if len(readings) < MIN_FRAMES:
            continue

        confidences = entry.get('confidences', {})
        consensus = build_consensus(readings, confidences)

        # Skip all-FF consensus (erased/empty regions)
        if all(b == 'FF' for b in consensus):
            continue

        # Find frames that disagree strongly
        wrong_frames = []
        for fk, rd in readings.items():
            diffs = sum(1 for i in range(16)
                        if i < len(rd) and rd[i] != '--' and rd[i] != consensus[i])
            if diffs >= disagreement_threshold:
                wrong_frames.append(fk)

        if len(wrong_frames) >= min_wrong:
            pct = len(wrong_frames) / len(readings) * 100
            results.append((addr_str, len(readings), len(wrong_frames), pct, wrong_frames))

    results.sort(key=lambda x: -x[2])
    return results


def flag_review_state(results):
    """Flag addresses in review_state.json for manual review.

    Only flags addresses not already accepted/edited.
    Returns count of newly flagged addresses.
    """
    with open(REVIEW_STATE_PATH) as f:
        rs = json.load(f)

    lines = rs.get('lines', {})
    flagged = 0

    for addr, total, wrong, pct, _ in results:
        line = lines.get(addr)
        if not line:
            continue
        if line.get('status') in ('accepted', 'edited'):
            continue
        line['status'] = 'flagged'
        line['notes'] = f'{wrong}/{total} frames misassigned ({pct:.0f}%)'
        flagged += 1

    rs['last_saved'] = datetime.now(timezone.utc).isoformat()
    with open(REVIEW_STATE_PATH, 'w') as f:
        json.dump(rs, f)

    return flagged


def main():
    parser = argparse.ArgumentParser(
        description='Flag addresses with misassigned frames for manual review')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and report only, do not flag')
    parser.add_argument('--threshold', type=int, default=DISAGREEMENT_THRESHOLD,
                        help=f'Min byte disagreements to mark frame as wrong (default {DISAGREEMENT_THRESHOLD})')
    parser.add_argument('--min-wrong', type=int, default=MIN_WRONG_FRAMES,
                        help=f'Min wrong frames to flag address (default {MIN_WRONG_FRAMES})')
    args = parser.parse_args()

    mmap = load_memory_map()
    accepted_addrs, _ = load_accepted_addresses()

    print(f"\n{'=' * 60}")
    print(f"Flag Misassigned Frames")
    print(f"{'=' * 60}")
    print(f"\n  Threshold: {args.threshold}/16 bytes disagree")
    print(f"  Min wrong frames: {args.min_wrong}")

    # Step 1: Load & detect
    print(f"\nStep 1: Load crop_index and detect misassigned frames")
    with open(CROP_INDEX_PATH) as f:
        crop_index = json.load(f)

    total_addrs = sum(1 for k, v in crop_index.items()
                      if isinstance(v, dict) and k != 'ref_addresses')
    print(f"  Total addresses in crop_index: {total_addrs}")
    print(f"  Accepted (skipped): {len(accepted_addrs)}")

    results = find_misassigned_addresses(
        crop_index, mmap, accepted_addrs, args.threshold, args.min_wrong)

    total_wrong = sum(w for _, _, w, _, _ in results)
    print(f"\n  Addresses with misassigned frames: {len(results)}")
    print(f"  Total wrong frames: {total_wrong}")

    if not results:
        print("\nNo misassigned frames detected.")
        print("Done.")
        return

    # Step 2: Summarize
    print(f"\nStep 2: Summary")
    print(f"\n  {'Address':>8} {'Frames':>6} {'Wrong':>5} {'%':>5}")
    print(f"  {'-' * 28}")
    for addr, total, wrong, pct, _ in results[:40]:
        print(f"  ${addr}  {total:5d} {wrong:5d} {pct:4.0f}%")
    if len(results) > 40:
        print(f"  ... and {len(results) - 40} more")

    if args.dry_run:
        print(f"\n  [DRY RUN] {len(results)} addresses detected, no files modified.")
        print("Done.")
        return

    # Step 3: Flag review_state
    print(f"\nStep 3: Flag addresses in review_state.json")
    flagged = flag_review_state(results)
    print(f"  Flagged: {flagged}")
    print(f"  Skipped (accepted/edited): {len(results) - flagged}")
    print("Done.")


if __name__ == '__main__':
    main()
