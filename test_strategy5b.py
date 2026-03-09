#!/usr/bin/env python3
"""
A/B test: Strategy 5B — Edge-row vote down-weighting.

Processes test frames through the existing classifier, collects
observations with row position, then compares multi-frame voting
accuracy with and without edge-row down-weighting.

Edge rows (0, 1, 11, 12) have reduced contrast (~200 vs ~255 for
middle rows). This test checks whether down-weighting their votes
improves the final consensus accuracy against reference ground truth.
"""

import cv2
import numpy as np
import os
from collections import defaultdict, Counter

from template_matcher import (
    CAL, CELL_H, ADDR_X_START,
    BYTE_POSITIONS, BYTE_DIGIT_SPACING,
    extract_cell, load_reference, find_rows_with_known_data,
)
from extract_pipeline import (
    FastKNNClassifier, process_frame, read_address_from_row,
    read_hex_bytes_from_row, validate_address_sequence,
    ADDR_MIN, ADDR_MAX,
)

TEST_FRAME_DIR = 'test_frames'
CLASSIFIER_PATH = 'fast_knn_classifier.npz'
EDGE_ROWS = {0, 1, 11, 12}


def process_frame_with_row(classifier, img):
    """Like process_frame() but also returns row index for each result."""
    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START

    addr_results = []
    for row_idx in range(-2, num_rows):
        row_y = y_first + row_idx * row_h
        if row_y < CELL_H or row_y > img.shape[0] - CELL_H:
            continue

        addr_str, addr_conf = read_address_from_row(
            classifier, img, row_y, addr_x
        )

        try:
            addr_int = int(addr_str, 16)
            if ADDR_MIN <= addr_int <= ADDR_MAX:
                addr_results.append((addr_int, row_y, addr_conf))
        except ValueError:
            pass

    validated_addrs = validate_address_sequence(addr_results)

    results = []
    for addr_int, row_y, addr_conf in validated_addrs:
        row_idx = round((row_y - y_first) / row_h)
        hex_bytes, byte_confs, avg_conf = read_hex_bytes_from_row(
            classifier, img, row_y
        )
        if (ADDR_MIN <= addr_int <= ADDR_MAX - 0x0F and
                addr_int % 0x10 == 0):
            results.append((addr_int, hex_bytes, byte_confs, addr_conf, row_idx))

    return results


def vote(observations, edge_weight=1.0):
    """Multi-frame voting per byte, with optional edge-row down-weighting.

    Args:
        observations: list of dicts with 'bytes', 'byte_confs', 'addr_conf', 'row_idx'
        edge_weight: multiplier for edge-row observations (1.0 = no change)

    Returns:
        list of 16 voted hex byte strings
    """
    final_bytes = []
    for byte_idx in range(16):
        votes = Counter()
        for obs in observations:
            if byte_idx < len(obs['bytes']):
                byte_val = obs['bytes'][byte_idx]
                if byte_val == "--":
                    continue
                weight = obs['byte_confs'][byte_idx] * obs['addr_conf']
                if obs['row_idx'] in EDGE_ROWS:
                    weight *= edge_weight
                votes[byte_val] += weight

        if votes:
            final_bytes.append(votes.most_common(1)[0][0])
        else:
            final_bytes.append('FF')

    return final_bytes


def main():
    print("=== A/B Test: Strategy 5B — Edge-Row Vote Down-Weighting ===\n")

    # Load classifier
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"ERROR: {CLASSIFIER_PATH} not found. Run extract_pipeline.py first.")
        return

    clf = FastKNNClassifier()
    clf.load(CLASSIFIER_PATH)

    # Load reference ground truth
    ref_data = load_reference()
    ref_addrs = set(ref_data.keys())
    print(f"Reference: {len(ref_data)} addresses "
          f"(${min(ref_data):05X}-${max(ref_data)+0xF:05X})")

    # Load test frames
    if not os.path.isdir(TEST_FRAME_DIR):
        print(f"ERROR: {TEST_FRAME_DIR}/ not found")
        return

    frame_files = sorted(f for f in os.listdir(TEST_FRAME_DIR) if f.endswith('.png'))
    print(f"Test frames: {len(frame_files)}")

    # Process all frames, collect observations with row position
    print("\nProcessing frames...")
    all_observations = defaultdict(list)
    row_distribution = Counter()
    total_rows_processed = 0

    for fname in frame_files:
        path = os.path.join(TEST_FRAME_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        results = process_frame_with_row(clf, img)
        for addr_int, hex_bytes, byte_confs, addr_conf, row_idx in results:
            all_observations[addr_int].append({
                'bytes': hex_bytes,
                'byte_confs': byte_confs,
                'addr_conf': addr_conf,
                'row_idx': row_idx,
                'frame': fname,
            })
            row_distribution[row_idx] += 1
            total_rows_processed += 1

    # Filter to reference addresses only (where we have ground truth)
    ref_observations = {a: obs for a, obs in all_observations.items()
                        if a in ref_addrs}

    print(f"Total observations: {total_rows_processed}")
    print(f"Unique addresses: {len(all_observations)}")
    print(f"Reference addresses with observations: {len(ref_observations)}")

    print(f"\nRow position distribution of reference observations:")
    for row_idx in sorted(row_distribution):
        tag = " <-- EDGE" if row_idx in EDGE_ROWS else ""
        print(f"  Row {row_idx:2d}: {row_distribution[row_idx]:4d} observations{tag}")

    if not ref_observations:
        print("ERROR: No reference addresses found in test frames!")
        return

    # Count how many reference addresses have mixed edge+middle observations
    mixed_count = 0
    for addr, obs_list in ref_observations.items():
        rows = {o['row_idx'] for o in obs_list}
        has_edge = bool(rows & EDGE_ROWS)
        has_middle = bool(rows - EDGE_ROWS)
        if has_edge and has_middle:
            mixed_count += 1

    print(f"\nAddresses with both edge and middle row observations: {mixed_count}")
    print(f"(These are the addresses where down-weighting can change the vote)")

    # Test multiple edge weights
    edge_weights = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    print(f"\n{'=' * 70}")
    print(f"{'Weight':>8} {'Correct':>8} {'Total':>6} {'Accuracy':>9} {'ByteCorr':>9} {'ByteTot':>8} {'ByteAcc':>9}")
    print(f"{'-' * 70}")

    best_weight = 1.0
    best_byte_correct = 0

    for ew in edge_weights:
        lines_correct = 0
        lines_total = 0
        bytes_correct = 0
        bytes_total = 0

        for addr, obs_list in ref_observations.items():
            voted_bytes = vote(obs_list, edge_weight=ew)
            expected_bytes = ref_data[addr]
            lines_total += 1
            line_ok = True

            for i in range(min(16, len(voted_bytes))):
                bytes_total += 1
                if voted_bytes[i].upper() == expected_bytes[i].upper():
                    bytes_correct += 1
                else:
                    line_ok = False

            if line_ok:
                lines_correct += 1

        label = "baseline" if ew == 1.0 else f"ew={ew}"
        print(f"  {label:>6} {lines_correct:>8} {lines_total:>6} "
              f"{lines_correct/lines_total*100:>8.2f}% "
              f"{bytes_correct:>9} {bytes_total:>8} "
              f"{bytes_correct/bytes_total*100:>8.2f}%")

        if bytes_correct > best_byte_correct:
            best_byte_correct = bytes_correct
            best_weight = ew

    # Detailed comparison: baseline vs best
    print(f"\n{'=' * 70}")

    if best_weight == 1.0:
        print("VERDICT: Down-weighting does NOT help. Baseline is best or tied.")
    else:
        print(f"VERDICT: Best edge weight = {best_weight}")

        # Show which bytes changed
        print(f"\nBytes that changed (baseline vs ew={best_weight}):")
        for addr in sorted(ref_observations.keys()):
            obs_list = ref_observations[addr]
            baseline_bytes = vote(obs_list, edge_weight=1.0)
            best_bytes = vote(obs_list, edge_weight=best_weight)
            expected = ref_data[addr]

            for i in range(16):
                if baseline_bytes[i] != best_bytes[i]:
                    exp = expected[i]
                    b_ok = "OK" if baseline_bytes[i].upper() == exp.upper() else "WRONG"
                    w_ok = "OK" if best_bytes[i].upper() == exp.upper() else "WRONG"
                    rows_for_addr = [o['row_idx'] for o in obs_list]
                    print(f"  ${addr:05X}[{i:2d}]: {baseline_bytes[i]}({b_ok}) -> "
                          f"{best_bytes[i]}({w_ok})  expected={exp}  "
                          f"rows={rows_for_addr}")


if __name__ == '__main__':
    main()
