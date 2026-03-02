#!/usr/bin/env python3
"""
Precompute gap context crops for missing firmware addresses.

For addresses where the standard precompute found no data,
this script discovers candidate frames via neighboring addresses and generates
multi-row context crops so a human reviewer can attempt manual transcription.

Input:
  crops/crop_index.json    (from precompute.py)
  firmware_merged.txt      (current best hex dump)
  frames/                  (video frames)
  grid_calibration.json    (grid geometry)
  fast_knn_classifier.npz  (kNN model)

Output:
  crops/gap/<address>/<frame_NNNNN>.png   (9-row context crops)
  crops/gap_context_index.json            (gap frame discovery index)
"""

import sys
import os
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from extract_pipeline import (
    FastKNNClassifier,
    read_address_from_row,
    validate_address_sequence,
    is_frame_different,
    CAL,
)
from template_matcher import ADDR_X_START

FRAMES_DIR = os.path.join(PROJECT_ROOT, 'frames')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
GAP_CROPS_DIR = os.path.join(CROPS_DIR, 'gap')
CROP_INDEX_PATH = os.path.join(CROPS_DIR, 'crop_index.json')
GAP_INDEX_PATH = os.path.join(CROPS_DIR, 'gap_context_index.json')
MERGED_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
KNN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'fast_knn_classifier.npz')

BASE_ADDR = 0x00000
END_ADDR = 0x13FF0
MAX_CANDIDATES = 8
CONTEXT_ROWS = 9  # +/-4 rows around target
EXTRA_ROWS_ABOVE = 2

# Crop geometry (matches precompute.py)
CROP_X_START = 270
CROP_X_END = 1100
ROW_HEIGHT = 28


def load_merged_addresses():
    """Load addresses present in firmware_merged.txt."""
    addrs = set()
    with open(MERGED_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    addr = int(parts[0].strip(), 16)
                    if BASE_ADDR <= addr <= END_ADDR and addr % 0x10 == 0:
                        addrs.add(addr)
                except ValueError:
                    continue
    return addrs


def find_missing_addresses(merged_addrs):
    """Find all expected addresses not present in merged data."""
    expected = set(range(BASE_ADDR, END_ADDR + 1, 0x10))
    return sorted(expected - merged_addrs)


def group_into_gaps(missing_addrs):
    """Group contiguous missing addresses into gaps."""
    if not missing_addrs:
        return []
    gaps = []
    gap_start = missing_addrs[0]
    gap_addrs = [missing_addrs[0]]
    for i in range(1, len(missing_addrs)):
        if missing_addrs[i] == missing_addrs[i - 1] + 0x10:
            gap_addrs.append(missing_addrs[i])
        else:
            gaps.append((gap_start, gap_addrs[-1], list(gap_addrs)))
            gap_start = missing_addrs[i]
            gap_addrs = [missing_addrs[i]]
    gaps.append((gap_start, gap_addrs[-1], list(gap_addrs)))
    return gaps


def find_candidate_frames(missing_addr, crop_index, neighbor_range=8):
    """
    Find candidate frames for a missing address by searching crop_index
    for neighboring addresses within +/-neighbor_range rows (+/-0x80 bytes).

    Returns list of (frame_num, score, neighbor_addr, row_offset) tuples.
    """
    candidates = {}  # frame_num -> (best_score, neighbor_addr, row_offset)

    for offset in range(-neighbor_range, neighbor_range + 1):
        if offset == 0:
            continue
        neighbor = missing_addr + offset * 0x10
        neighbor_key = f"{neighbor:05X}"
        if neighbor_key not in crop_index:
            continue

        distance = abs(offset)
        proximity_score = (neighbor_range + 1 - distance) ** 2

        for frame_num in crop_index[neighbor_key]["frames"]:
            if frame_num not in candidates or candidates[frame_num][0] < proximity_score:
                candidates[frame_num] = (proximity_score, neighbor, offset)

    return [(f, s, n, o) for f, (s, n, o) in sorted(
        candidates.items(), key=lambda x: -x[1][0]
    )]


def compute_sharpness(img, y_center, row_h, x_start=CROP_X_START, x_end=CROP_X_END):
    """Compute Laplacian-based sharpness of the target row region."""
    half_h = int(row_h * 2)
    y1 = max(0, int(y_center) - half_h)
    y2 = min(img.shape[0], int(y_center) + half_h)
    roi = img[y1:y2, x_start:x_end]
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    variance = lap.var()
    # Normalize to 0-1 range (typical variance range 0-500)
    return min(1.0, variance / 500.0)


def locate_target_row(classifier, img, missing_addr, row_h, y_first, addr_x, num_rows):
    """
    Locate the Y-coordinate of a missing address in a frame by detecting
    nearby addresses as anchors and computing the offset.
    """
    anchors = []
    for row_idx in range(-EXTRA_ROWS_ABOVE, num_rows + 2):
        row_y = y_first + row_idx * row_h
        if row_y < 0 or row_y >= img.shape[0]:
            continue
        addr_str, conf = read_address_from_row(classifier, img, row_y, addr_x)
        try:
            addr_int = int(addr_str, 16)
            if BASE_ADDR <= addr_int <= END_ADDR and addr_int % 0x10 == 0:
                anchors.append((addr_int, row_y, conf))
        except ValueError:
            continue

    if not anchors:
        return None

    # Validate address sequence
    validated = validate_address_sequence(anchors)

    # Use validated anchors to estimate target position
    estimates = []
    for addr_int, row_y, conf in validated:
        row_diff = (missing_addr - addr_int) / 0x10
        target_y = row_y + row_diff * row_h
        if 0 <= target_y < img.shape[0]:
            estimates.append((target_y, conf))

    if not estimates:
        return None

    # Weighted average by confidence
    total_weight = sum(c for _, c in estimates)
    if total_weight == 0:
        return estimates[0][0]
    target_y = sum(y * c for y, c in estimates) / total_weight
    return target_y


def generate_context_crop(img, target_y, row_h, x_start=CROP_X_START, x_end=CROP_X_END):
    """
    Generate a 9-row context crop centered on target_y with CLAHE enhancement.
    """
    half_rows = CONTEXT_ROWS // 2
    crop_height = int(CONTEXT_ROWS * row_h) + 2
    y_center = int(round(target_y))
    y1 = max(0, y_center - int(half_rows * row_h) - 1)
    y2 = min(img.shape[0], y1 + crop_height)
    x2 = min(img.shape[1], x_end)

    crop = img[y1:y2, x_start:x2]
    if crop.size == 0:
        return None

    # Apply CLAHE for better human readability
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop = clahe.apply(crop)

    return crop


def precompute_gaps():
    print("=== Gap Context Precompute ===")
    print()

    # Load crop index
    if not os.path.exists(CROP_INDEX_PATH):
        print(f"ERROR: {CROP_INDEX_PATH} not found. Run precompute.py first.")
        sys.exit(1)

    with open(CROP_INDEX_PATH) as f:
        crop_index = json.load(f)
    print(f"Loaded crop index: {len(crop_index)} addresses")

    # Load merged addresses
    merged_addrs = load_merged_addresses()
    print(f"Merged firmware: {len(merged_addrs)} addresses")

    # Find missing addresses
    missing_addrs = find_missing_addresses(merged_addrs)
    print(f"Missing addresses: {len(missing_addrs)}")

    if not missing_addrs:
        print("No gaps to process.")
        return

    # Group into gaps
    gaps = group_into_gaps(missing_addrs)
    print(f"Gaps: {len(gaps)}")

    # Load kNN classifier
    print("\nLoading kNN classifier...")
    classifier = FastKNNClassifier()
    classifier.load(KNN_MODEL_PATH)

    # Grid params
    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START

    os.makedirs(GAP_CROPS_DIR, exist_ok=True)

    gap_index = {}
    total_crops = 0
    frames_loaded = {}  # cache loaded frames
    start_time = time.time()

    for gap_id, (gap_start, gap_end, gap_addrs) in enumerate(gaps):
        for missing_addr in gap_addrs:
            addr_key = f"{missing_addr:05X}"

            entry = {
                "gap_id": gap_id,
                "gap_start": f"{gap_start:05X}",
                "gap_end": f"{gap_end:05X}",
                "gap_size": len(gap_addrs),
                "candidates": [],
            }

            # Find candidate frames
            candidates = find_candidate_frames(missing_addr, crop_index)

            if not candidates:
                gap_index[addr_key] = entry
                continue

            # Score and rank candidates
            scored = []
            for frame_num, proximity_score, neighbor_addr, row_offset in candidates[:MAX_CANDIDATES * 2]:
                # Load frame (cached)
                if frame_num not in frames_loaded:
                    frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_num:05d}.png')
                    if not os.path.exists(frame_path):
                        continue
                    frames_loaded[frame_num] = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

                img = frames_loaded[frame_num]
                if img is None:
                    continue

                # Locate target row
                target_y = locate_target_row(
                    classifier, img, missing_addr, row_h, y_first, addr_x, num_rows
                )
                if target_y is None:
                    continue

                # Check bounds
                if target_y < row_h or target_y >= img.shape[0] - row_h:
                    continue

                # Compute sharpness
                quality = compute_sharpness(img, target_y, row_h)

                # Composite score: 60% proximity, 40% quality
                composite = 0.6 * proximity_score + 0.4 * (quality * 100)
                scored.append((frame_num, composite, quality, target_y,
                               neighbor_addr, row_offset))

            # Keep top candidates
            scored.sort(key=lambda x: -x[1])
            scored = scored[:MAX_CANDIDATES]

            for frame_num, score, quality, target_y, anchor_addr, anchor_offset in scored:
                img = frames_loaded[frame_num]

                # Generate context crop
                crop = generate_context_crop(img, target_y, row_h)
                if crop is None:
                    continue

                # Save crop
                addr_lower = addr_key.lower()
                crop_dir = os.path.join(GAP_CROPS_DIR, addr_lower)
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f'frame_{frame_num:05d}.png')
                crop_saved = cv2.imwrite(crop_path, crop)

                entry["candidates"].append({
                    "frame": int(frame_num),
                    "score": float(round(score, 1)),
                    "quality": float(round(quality, 2)),
                    "target_y": float(round(target_y, 1)),
                    "anchor_addr": f"{anchor_addr:05X}",
                    "anchor_offset_rows": int(-anchor_offset),
                    "crop_saved": bool(crop_saved),
                })

                if crop_saved:
                    total_crops += 1

            gap_index[addr_key] = entry

    elapsed = time.time() - start_time

    # Write gap context index
    with open(GAP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(gap_index, f, indent=2)

    # Stats
    addrs_with_crops = sum(1 for e in gap_index.values() if e["candidates"])
    addrs_without = sum(1 for e in gap_index.values() if not e["candidates"])

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Missing addresses processed: {len(gap_index)}")
    print(f"  Addresses with context crops: {addrs_with_crops}")
    print(f"  Unreachable addresses: {addrs_without}")
    print(f"  Total context crops saved: {total_crops}")
    print(f"  Frames loaded: {len(frames_loaded)}")
    print(f"  Gap index written to {GAP_INDEX_PATH}")

    # Free frame cache
    frames_loaded.clear()


if __name__ == '__main__':
    precompute_gaps()
