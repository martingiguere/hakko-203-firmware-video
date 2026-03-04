#!/usr/bin/env python3
"""
Precompute frame crops and kNN readings for the firmware review tool.

For each of 20,070 video frames, identifies visible addresses and:
  - Crops the full row (address + hex bytes) as a PNG
  - Runs kNN byte classification on all 16 bytes
  - Builds a crop_index.json mapping addresses to frames and readings

Output:
  crops/<addr_lower>/frame_NNNNN.png
  crops/crop_index.json
"""

import sys
import os
import json
import time

# Must set up paths before importing extract_pipeline (it reads grid_calibration.json at import)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from extract_pipeline import (
    FastKNNClassifier,
    read_address_from_row,
    read_hex_bytes_from_row,
    validate_address_sequence,
    is_frame_different,
)
from template_matcher import CAL, ADDR_X_START, BYTE_POSITIONS, BYTE_DIGIT_SPACING

FRAMES_DIR = os.path.join(PROJECT_ROOT, 'frames')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
CROP_INDEX_PATH = os.path.join(CROPS_DIR, 'crop_index.json')
KNN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'fast_knn_classifier.npz')

MAX_FRAMES_PER_ADDRESS = 20
TOTAL_FRAMES = 20070

# Row crop geometry
ROW_HALF_HEIGHT = 14  # 28px tall crop centered on row center
CROP_X_START = 270
CROP_X_END = 1100  # 830px wide crop

BASE_ADDR = 0x00000
END_ADDR = 0x13FF0
MAX_ADDR_DIFF = 0x10000  # Max allowable difference between raw OCR read and validated address
EXTRA_ROWS_ABOVE = 2


def precompute():
    print("=== Firmware Review Tool: Precompute ===")
    print()

    # Load classifier
    print("Loading kNN classifier...")
    classifier = FastKNNClassifier()
    classifier.load(KNN_MODEL_PATH)

    # Grid params
    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START

    # Prepare output directory
    os.makedirs(CROPS_DIR, exist_ok=True)

    # Index: addr_upper -> {frames: [int], readings: {frame_str: [bytes]}, confidences: {frame_str: [floats]}}
    crop_index = {}
    # Track frame count per address to enforce cap
    addr_frame_count = {}

    prev_img = None
    skipped = 0
    processed = 0
    total_crops = 0
    start_time = time.time()

    for frame_num in range(TOTAL_FRAMES):
        frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_num:05d}.png')
        if not os.path.exists(frame_path):
            continue

        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Skip identical frames
        if not is_frame_different(img, prev_img):
            skipped += 1
            prev_img = img
            continue

        prev_img = img
        processed += 1

        # Read addresses from all visible rows (+ extra above)
        addr_results = []
        for row_idx in range(-EXTRA_ROWS_ABOVE, num_rows):
            row_y = y_first + row_idx * row_h
            if row_y < 0 or row_y >= img.shape[0]:
                continue
            addr_str, addr_conf = read_address_from_row(classifier, img, row_y, addr_x)
            try:
                addr_int = int(addr_str, 16)
                addr_results.append((addr_int, row_y, addr_conf))
            except ValueError:
                continue

        # Validate and correct address sequence
        validated = validate_address_sequence(addr_results)

        for (raw_addr, _, _), (addr_int, row_y, conf) in zip(addr_results, validated):
            # Skip rows where raw OCR read is wildly different from validated address
            if abs(raw_addr - addr_int) > MAX_ADDR_DIFF:
                continue
            if addr_int < BASE_ADDR or addr_int > END_ADDR:
                continue
            if addr_int % 0x10 != 0:
                continue

            addr_upper = f"{addr_int:05X}"
            addr_lower = addr_upper.lower()

            # Enforce frame cap
            count = addr_frame_count.get(addr_upper, 0)
            if count >= MAX_FRAMES_PER_ADDRESS:
                continue

            # Crop the full row
            y_center = int(round(row_y))
            y1 = max(0, y_center - ROW_HALF_HEIGHT)
            y2 = min(img.shape[0], y_center + ROW_HALF_HEIGHT)
            x1 = CROP_X_START
            x2 = min(img.shape[1], CROP_X_END)
            crop = img[y1:y2, x1:x2]

            # Read kNN bytes for this row
            hex_bytes, confidences, _avg_conf = read_hex_bytes_from_row(
                classifier, img, row_y
            )

            # Save crop PNG
            crop_dir = os.path.join(CROPS_DIR, addr_lower)
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f'frame_{frame_num:05d}.png')
            cv2.imwrite(crop_path, crop)

            # Update index
            if addr_upper not in crop_index:
                crop_index[addr_upper] = {"frames": [], "readings": {}, "confidences": {}}
            crop_index[addr_upper]["frames"].append(frame_num)
            crop_index[addr_upper]["readings"][str(frame_num)] = [
                b.upper() if b != "--" else "--" for b in hex_bytes
            ]
            crop_index[addr_upper]["confidences"][str(frame_num)] = [
                round(float(c), 3) for c in confidences
            ]

            addr_frame_count[addr_upper] = count + 1
            total_crops += 1

        # Progress bar
        if frame_num % 200 == 0 or frame_num == TOTAL_FRAMES - 1:
            elapsed = time.time() - start_time
            pct = (frame_num + 1) / TOTAL_FRAMES * 100
            rate = (frame_num + 1) / elapsed if elapsed > 0 else 0
            eta = (TOTAL_FRAMES - frame_num - 1) / rate if rate > 0 else 0
            print(
                f"\r  Frame {frame_num + 1:5d}/{TOTAL_FRAMES} "
                f"({pct:5.1f}%) | {processed} unique | "
                f"{total_crops} crops | {rate:.0f} fps | ETA {eta:.0f}s   ",
                end="", flush=True,
            )

    print()
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Frames processed (unique): {processed}")
    print(f"  Frames skipped (identical): {skipped}")
    print(f"  Total crops saved: {total_crops}")
    print(f"  Addresses with crops: {len(crop_index)}")

    # Sort frames lists
    for addr in crop_index:
        crop_index[addr]["frames"].sort()

    # Write crop index
    with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f)
    print(f"  Crop index written to {CROP_INDEX_PATH}")
    print(f"  Index size: {os.path.getsize(CROP_INDEX_PATH) / 1024:.0f} KB")


if __name__ == '__main__':
    precompute()
