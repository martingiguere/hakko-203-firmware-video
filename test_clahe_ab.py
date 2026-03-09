#!/usr/bin/env python3
"""
A/B test: CLAHE preprocessing impact on digit classification accuracy.

Builds two classifiers from the same training data:
  A: Baseline (current min-max normalization only)
  B: CLAHE applied to each cell before feature extraction

Tests both against ground-truth reference data on held-out frames.
Reports per-digit accuracy and confusion matrices, with focus on
known confusion pairs (8/6, 4/9, D/C) and edge vs middle rows.
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import defaultdict, Counter

from template_matcher import (
    CAL, CELL_W, CELL_H, ADDR_X_START,
    BYTE_POSITIONS, BYTE_DIGIT_SPACING,
    extract_cell, is_blank_cell, extract_features,
    load_reference, find_rows_with_known_data,
)
from extract_pipeline import FastKNNClassifier

# Frame directory for test frames
TEST_FRAME_DIR = 'test_frames'


def extract_features_clahe(cell, clip_limit=2.0, tile_grid=(2, 4)):
    """extract_features() with CLAHE preprocessing applied first.

    Applies CLAHE to the raw cell after the blank check but before
    min-max normalization and feature computation.
    """
    if is_blank_cell(cell):
        return None

    # Apply CLAHE on raw uint8 cell
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cell = clahe.apply(cell)

    # Rest is identical to extract_features()
    h, w = cell.shape
    c = cell.astype(np.float32)
    mn, mx = c.min(), c.max()
    if mx > mn:
        c = (c - mn) / (mx - mn)
    else:
        c = np.zeros_like(c)
    ink = 1.0 - c

    features = []

    # 1. Horizontal profile (h values)
    features.extend(np.mean(ink, axis=1))
    # 2. Vertical profile (w values)
    features.extend(np.mean(ink, axis=0))

    mid_y, mid_x = h // 2, w // 2
    y3, y23 = h // 3, 2 * h // 3
    x3, x23 = w // 3, 2 * w // 3

    # 3. Quadrant densities (2x2)
    features.append(np.mean(ink[:mid_y, :mid_x]))
    features.append(np.mean(ink[:mid_y, mid_x:]))
    features.append(np.mean(ink[mid_y:, :mid_x]))
    features.append(np.mean(ink[mid_y:, mid_x:]))

    # 4. Center region
    features.append(np.mean(ink[y3:y23, x3:x23]))
    features.append(np.mean(ink[y3:y23, :x3]))
    features.append(np.mean(ink[y3:y23, x23:]))

    # 5. Top/bottom, left/right asymmetry
    features.append(np.mean(ink[:mid_y, :]) - np.mean(ink[mid_y:, :]))
    features.append(np.mean(ink[:, :mid_x]) - np.mean(ink[:, mid_x:]))

    # 6. Center bar detection (distinguishes 0 vs 8)
    center_row = ink[max(0, mid_y - 1):mid_y + 2, :]
    center_bar = np.mean(center_row[:, x3:x23])
    above = np.mean(ink[max(0, mid_y - 4):max(1, mid_y - 1), x3:x23])
    below = np.mean(ink[mid_y + 2:min(h, mid_y + 5), x3:x23])
    features.append(center_bar - (above + below) / 2)

    # 7. Vertical symmetry
    left_half = ink[:, :mid_x]
    right_half = np.fliplr(ink[:, mid_x:mid_x + left_half.shape[1]])
    if left_half.shape == right_half.shape:
        features.append(np.mean(np.abs(left_half - right_half)))
    else:
        features.append(0.0)

    # 8. Corner features
    features.append(np.mean(ink[:y3, x23:]))    # top-right
    features.append(np.mean(ink[y23:, :x3]))    # bottom-left

    # 9. 3x3 sub-grid densities
    for yi in range(3):
        for xi in range(3):
            ys = yi * h // 3
            ye = (yi + 1) * h // 3
            xs = xi * w // 3
            xe = (xi + 1) * w // 3
            features.append(np.mean(ink[ys:ye, xs:xe]))

    # 10. 8-vs-6 discriminative features
    features.append(np.mean(ink[:h//4, w//2:]))
    features.append(np.mean(ink[y23:, :w//3]) - np.mean(ink[:y3, 2*w//3:]))
    features.append(np.mean(ink[:mid_y, x23:]) - np.mean(ink[mid_y:, x23:]))

    return np.array(features, dtype=np.float32)


def build_classifier(samples_dict, use_clahe=False, max_per_class=500):
    """Build a FastKNNClassifier, optionally with CLAHE preprocessing.

    When use_clahe=True, monkey-patches extract_features during build
    so the training features use CLAHE preprocessing.
    """
    import template_matcher

    if use_clahe:
        original_fn = template_matcher.extract_features
        template_matcher.extract_features = extract_features_clahe

    clf = FastKNNClassifier()
    clf.build_from_samples(samples_dict, max_per_class=max_per_class)

    if use_clahe:
        template_matcher.extract_features = original_fn

    return clf


def classify_batch_clahe(clf, cells, k=7):
    """Classify cells using CLAHE-preprocessed features against a
    CLAHE-trained classifier."""
    feat_dim = clf.feature_matrix.shape[1]
    raw_features = [extract_features_clahe(c) for c in cells]
    blank_mask = [f is None for f in raw_features]

    features = np.stack([
        f if f is not None else np.zeros(feat_dim, dtype=np.float32)
        for f in raw_features
    ]).astype(np.float32)

    a_sq = np.sum(features ** 2, axis=1, keepdims=True)
    ab = features @ clf.feature_matrix.T
    distances = a_sq + clf._b_sq[np.newaxis, :] - 2 * ab

    results = []
    for i in range(len(cells)):
        if blank_mask[i]:
            results.append((None, 0.0))
            continue

        row = distances[i]
        top_k_idx = np.argpartition(row, k)[:k]
        top_k_labels = clf.labels[top_k_idx]
        top_k_dists = row[top_k_idx]

        votes = defaultdict(float)
        for label, dist in zip(top_k_labels, top_k_dists):
            weight = 1.0 / (dist + 1e-6)
            votes[label] += weight

        best_label = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[best_label] / total if total > 0 else 0
        results.append((clf.idx_to_char[best_label], confidence))

    return results


def print_confusion_matrix(confusion):
    """Print confusion matrix with ANSI colors."""
    chars = sorted(set(list(confusion.keys()) +
                      [c for v in confusion.values() for c in v.keys()]))
    print(f"{'':>5}", end='')
    for c in chars:
        print(f" {c:>3}", end='')
    print("  |  Tot  Acc")
    print("-" * (6 + 4 * len(chars) + 14))

    for true_char in chars:
        if true_char not in confusion:
            continue
        row = confusion[true_char]
        total = sum(row.values())
        correct_count = row.get(true_char, 0)
        acc = correct_count / total if total > 0 else 0
        print(f"  {true_char:>3}", end='')
        for pred_char in chars:
            count = row.get(pred_char, 0)
            if count > 0 and pred_char == true_char:
                print(f" \033[92m{count:>3}\033[0m", end='')
            elif count > 0:
                print(f" \033[91m{count:>3}\033[0m", end='')
            else:
                print(f"   .", end='')
        print(f"  | {total:>4} {acc:.0%}")


def main():
    print("=== A/B Test: CLAHE Preprocessing Impact ===\n")

    # Load reference data
    ref_data = load_reference()
    print(f"Reference: {len(ref_data)} addresses "
          f"(${min(ref_data):05X}-${max(ref_data)+0xF:05X})")

    # Check test frames exist
    if not os.path.isdir(TEST_FRAME_DIR):
        print(f"ERROR: {TEST_FRAME_DIR}/ not found")
        return

    frame_files = sorted(f for f in os.listdir(TEST_FRAME_DIR) if f.endswith('.png'))
    print(f"Test frames available: {len(frame_files)}")

    if not frame_files:
        print("ERROR: No PNG frames found in test_frames/")
        return

    # Split into train (75%) and test (25%)
    n_train = max(1, int(len(frame_files) * 0.75))
    train_files = frame_files[:n_train]
    test_files = frame_files[n_train:]
    print(f"Train: {len(train_files)} frames, Test: {len(test_files)} frames")

    # Build training samples from train frames
    print("\nBuilding training samples...")
    samples = defaultdict(list)
    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']

    for fname in train_files:
        path = os.path.join(TEST_FRAME_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        matches = find_rows_with_known_data(img, ref_data)
        for addr, row_y in matches:
            hex_bytes = ref_data[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING
                samples[byte_str[0]].append(extract_cell(img, row_y, hi_x))
                samples[byte_str[1]].append(extract_cell(img, row_y, lo_x))

    total_samples = sum(len(v) for v in samples.values())
    print(f"Training samples: {total_samples} across {len(samples)} classes")

    if total_samples == 0:
        print("ERROR: No training samples collected!")
        return

    # Build both classifiers
    print("\n--- Building Baseline classifier (no CLAHE) ---")
    clf_baseline = build_classifier(samples, use_clahe=False)

    print("\n--- Building CLAHE classifier ---")
    clf_clahe = build_classifier(samples, use_clahe=True)

    # Collect test cells with row position info
    print("\nCollecting test cells...")
    test_cells = []
    test_ground_truth = []
    test_row_indices = []

    for fname in test_files:
        path = os.path.join(TEST_FRAME_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        matches = find_rows_with_known_data(img, ref_data)
        for addr, row_y in matches:
            # Determine row index (0-12)
            row_idx = round((row_y - y_first) / row_h)
            hex_bytes = ref_data[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING
                test_cells.append(extract_cell(img, row_y, hi_x))
                test_ground_truth.append(byte_str[0])
                test_row_indices.append(row_idx)
                test_cells.append(extract_cell(img, row_y, lo_x))
                test_ground_truth.append(byte_str[1])
                test_row_indices.append(row_idx)

    print(f"Test cells: {len(test_cells)} digits")

    if not test_cells:
        print("ERROR: No test cells collected!")
        return

    # Classify with both
    print("\nClassifying...")
    baseline_results = clf_baseline.classify_batch(test_cells)
    clahe_results = classify_batch_clahe(clf_clahe, test_cells)

    # Overall accuracy
    def compute_accuracy(results, ground_truth):
        correct = sum(1 for (p, _), gt in zip(results, ground_truth) if p == gt)
        return correct, len(ground_truth)

    b_correct, b_total = compute_accuracy(baseline_results, test_ground_truth)
    c_correct, c_total = compute_accuracy(clahe_results, test_ground_truth)

    print(f"\n{'=' * 60}")
    print(f"OVERALL ACCURACY")
    print(f"  Baseline: {b_correct}/{b_total} ({b_correct/b_total*100:.2f}%)")
    print(f"  CLAHE:    {c_correct}/{c_total} ({c_correct/c_total*100:.2f}%)")
    print(f"  Delta:    {(c_correct - b_correct):+d} digits")

    # Edge vs middle row breakdown
    edge_rows = {0, 1, 11, 12}
    print(f"\n{'=' * 60}")
    print(f"EDGE ROWS (0, 1, 11, 12) vs MIDDLE ROWS")

    for label, row_set in [("Edge", edge_rows), ("Middle", set(range(2, 11)))]:
        idx = [i for i, r in enumerate(test_row_indices) if r in row_set]
        if not idx:
            print(f"  {label}: no test cells")
            continue
        b_ok = sum(1 for i in idx if baseline_results[i][0] == test_ground_truth[i])
        c_ok = sum(1 for i in idx if clahe_results[i][0] == test_ground_truth[i])
        n = len(idx)
        print(f"  {label:6s}: Baseline {b_ok}/{n} ({b_ok/n*100:.2f}%), "
              f"CLAHE {c_ok}/{n} ({c_ok/n*100:.2f}%), "
              f"delta {(c_ok - b_ok):+d}")

    # Known confusion pairs
    print(f"\n{'=' * 60}")
    print(f"KNOWN CONFUSION PAIRS")
    pairs = [('8', '6'), ('4', '9'), ('D', 'C')]
    for a, b in pairs:
        idx = [i for i, gt in enumerate(test_ground_truth) if gt in (a, b)]
        if not idx:
            print(f"  {a}/{b}: no test cells")
            continue
        b_ok = sum(1 for i in idx if baseline_results[i][0] == test_ground_truth[i])
        c_ok = sum(1 for i in idx if clahe_results[i][0] == test_ground_truth[i])
        n = len(idx)
        print(f"  {a}/{b}: Baseline {b_ok}/{n} ({b_ok/n*100:.1f}%), "
              f"CLAHE {c_ok}/{n} ({c_ok/n*100:.1f}%), "
              f"delta {(c_ok - b_ok):+d}")

    # Confusion matrices
    for label, results in [("BASELINE", baseline_results), ("CLAHE", clahe_results)]:
        confusion = defaultdict(lambda: defaultdict(int))
        for (pred, _), gt in zip(results, test_ground_truth):
            if pred is None:
                pred = '?'
            confusion[gt][pred] += 1
        print(f"\n{'=' * 60}")
        print(f"Confusion Matrix: {label}")
        print_confusion_matrix(confusion)

    # Verdict
    print(f"\n{'=' * 60}")
    delta_pct = (c_correct - b_correct) / b_total * 100
    if c_correct > b_correct:
        print(f"VERDICT: CLAHE HELPS (+{c_correct - b_correct} digits, +{delta_pct:.2f}%)")
    elif c_correct == b_correct:
        print(f"VERDICT: CLAHE has NO EFFECT (same accuracy)")
    else:
        print(f"VERDICT: CLAHE HURTS ({c_correct - b_correct} digits, {delta_pct:.2f}%)")


if __name__ == '__main__':
    main()
