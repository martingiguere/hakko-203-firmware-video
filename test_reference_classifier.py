#!/usr/bin/env python3
"""
A/B test: video-only vs reference-only vs ensemble classifier.

Compares classification accuracy of three configurations on held-out video
frames that show reference addresses:
  A: Video-only classifier (fast_knn_classifier.npz)
  B: Reference-only classifier (ref_classifier_data.npz)
  C: Ensemble — 70% video weight, 30% reference weight
"""

import cv2
import numpy as np
import os
from collections import defaultdict

from template_matcher import (
    load_reference, find_rows_with_known_data, extract_cell,
    BYTE_POSITIONS, BYTE_DIGIT_SPACING, extract_features,
)
from extract_pipeline import FastKNNClassifier, find_reference_frames


VIDEO_CLASSIFIER_PATH = 'fast_knn_classifier.npz'
REF_CLASSIFIER_PATH = 'ref_classifier_data.npz'


def classify_ensemble(video_clf, ref_clf, cells, video_weight=0.7, ref_weight=0.3, k=7):
    """Classify cells using weighted ensemble of two classifiers.

    For each cell, classifies with both classifiers, then combines
    confidence-weighted votes.

    Returns list of (char, confidence) tuples.
    """
    video_results = video_clf.classify_batch(cells, k=k)
    ref_results = ref_clf.classify_batch(cells, k=k)

    ensemble_results = []
    for (v_char, v_conf), (r_char, r_conf) in zip(video_results, ref_results):
        if v_char is None and r_char is None:
            ensemble_results.append((None, 0.0))
            continue

        votes = defaultdict(float)
        if v_char is not None:
            votes[v_char] += v_conf * video_weight
        if r_char is not None:
            votes[r_char] += r_conf * ref_weight

        best_char = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[best_char] / total if total > 0 else 0
        ensemble_results.append((best_char, confidence))

    return ensemble_results


def evaluate_config(name, results, ground_truth):
    """Evaluate a classifier configuration against ground truth.

    Args:
        name: Configuration name (for printing)
        results: list of (predicted_char, confidence)
        ground_truth: list of expected_char

    Returns:
        (accuracy, total, correct, confusion_dict)
    """
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = len(ground_truth)

    for (pred, conf), expected in zip(results, ground_truth):
        if pred is None:
            pred = '?'
        confusion[expected][pred] += 1
        if pred == expected:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, total, correct, confusion


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
    print("=== A/B Test: Video vs Reference vs Ensemble Classifier ===\n")

    # Load classifiers
    if not os.path.exists(VIDEO_CLASSIFIER_PATH):
        print(f"ERROR: Video classifier not found: {VIDEO_CLASSIFIER_PATH}")
        print("Run extract_pipeline.py first to build it.")
        return

    if not os.path.exists(REF_CLASSIFIER_PATH):
        print(f"ERROR: Reference classifier not found: {REF_CLASSIFIER_PATH}")
        print("Run extract_reference_samples.py first to build it.")
        return

    video_clf = FastKNNClassifier()
    video_clf.load(VIDEO_CLASSIFIER_PATH)

    ref_clf = FastKNNClassifier()
    ref_clf.load(REF_CLASSIFIER_PATH)

    # Load ground truth
    ref_data = load_reference()
    print(f"Reference: {len(ref_data)} addresses "
          f"(${min(ref_data):05X}-${max(ref_data)+0xF:05X})")

    # Find test frames
    print("\nFinding reference-visible test frames...")
    frame_paths = find_reference_frames(ref_data)
    if not frame_paths:
        print("ERROR: No reference-visible frames found!")
        return

    # Use ~5 evenly-spaced frames for testing
    n_test = min(5, len(frame_paths))
    step = max(1, len(frame_paths) // n_test)
    test_paths = [frame_paths[i * step] for i in range(n_test)
                  if i * step < len(frame_paths)]
    print(f"Using {len(test_paths)} test frames")

    # Collect all test cells and ground truth
    all_cells = []
    all_ground_truth = []
    frames_used = 0

    for frame_path in test_paths:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        matches = find_rows_with_known_data(img, ref_data)
        if not matches:
            continue

        frames_used += 1
        for addr, row_y in matches:
            hex_bytes = ref_data[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING

                # High nybble
                cell_hi = extract_cell(img, row_y, hi_x)
                all_cells.append(cell_hi)
                all_ground_truth.append(byte_str[0])

                # Low nybble
                cell_lo = extract_cell(img, row_y, lo_x)
                all_cells.append(cell_lo)
                all_ground_truth.append(byte_str[1])

    print(f"Collected {len(all_cells)} digit cells from {frames_used} frames\n")

    if not all_cells:
        print("ERROR: No test cells collected!")
        return

    # Classify with each configuration
    print("Classifying with each configuration...")
    video_results = video_clf.classify_batch(all_cells)
    ref_results = ref_clf.classify_batch(all_cells)
    ensemble_results = classify_ensemble(video_clf, ref_clf, all_cells)

    # Evaluate each
    configs = [
        ("A: Video-only", video_results),
        ("B: Reference-only", ref_results),
        ("C: Ensemble (70/30)", ensemble_results),
    ]

    print("\n" + "=" * 60)
    best_acc = 0
    best_name = ""

    for name, results in configs:
        acc, total, correct, confusion = evaluate_config(
            name, results, all_ground_truth
        )
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.1%} ({correct}/{total})")
        print_confusion_matrix(confusion)

        if acc > best_acc:
            best_acc = acc
            best_name = name

    # Summary
    print("\n" + "=" * 60)
    print(f"\nBest configuration: {best_name} ({best_acc:.1%})")

    # Per-character comparison
    print("\nPer-character accuracy comparison:")
    print(f"{'Char':>6} {'Video':>8} {'Ref':>8} {'Ensemble':>8} {'Winner':>10}")
    print("-" * 44)

    char_set = sorted(set(all_ground_truth))
    for char in char_set:
        indices = [i for i, gt in enumerate(all_ground_truth) if gt == char]
        if not indices:
            continue

        v_correct = sum(1 for i in indices if video_results[i][0] == char)
        r_correct = sum(1 for i in indices if ref_results[i][0] == char)
        e_correct = sum(1 for i in indices if ensemble_results[i][0] == char)
        n = len(indices)

        v_acc = v_correct / n
        r_acc = r_correct / n
        e_acc = e_correct / n

        accs = {'Video': v_acc, 'Ref': r_acc, 'Ensemble': e_acc}
        winner = max(accs, key=accs.get)
        if v_acc == r_acc == e_acc:
            winner = "tie"

        print(f"  '{char}' {v_acc:>7.0%} {r_acc:>7.0%} {e_acc:>7.0%} {winner:>10}")

    # Verdict
    v_acc_total = sum(1 for (p, _), gt in zip(video_results, all_ground_truth)
                      if p == gt) / len(all_ground_truth)
    e_acc_total = sum(1 for (p, _), gt in zip(ensemble_results, all_ground_truth)
                      if p == gt) / len(all_ground_truth)

    print(f"\nVerdict: Ensemble {'HELPS' if e_acc_total > v_acc_total else 'MATCHES' if e_acc_total == v_acc_total else 'HURTS'} "
          f"vs video-only ({e_acc_total:.1%} vs {v_acc_total:.1%})")

    r_acc_total = sum(1 for (p, _), gt in zip(ref_results, all_ground_truth)
                      if p == gt) / len(all_ground_truth)
    if r_acc_total < 0.5:
        print(f"WARNING: Reference-only accuracy is low ({r_acc_total:.1%}) — "
              f"domain gap may be too large for Phase 2")


if __name__ == '__main__':
    main()
