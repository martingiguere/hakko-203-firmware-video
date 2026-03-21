#!/usr/bin/env python3
"""
Extract labeled training samples from review-tool-accepted lines.

Reads review_state.json for user-confirmed byte values, then extracts
digit cells from the corresponding video frames to produce supplementary
training data for the kNN classifier.

Input:
    review_state.json        — accepted/edited lines with confirmed bytes
    crops/crop_index.json    — frame-to-address mapping
    frame_assignments.json   — row_y positions per frame (from extract_pipeline)
    frames/                  — source frame PNGs

Output:
    review_training_samples.npz — per-character cell arrays for Pass 3 training

Usage:
    python3 build_review_training.py
"""

import cv2
import json
import numpy as np
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from template_matcher import (
    extract_cell, is_blank_cell, BYTE_POSITIONS, BYTE_DIGIT_SPACING,
)

REVIEW_STATE_PATH = 'review_state.json'
CROP_INDEX_PATH = os.path.join('crops', 'crop_index.json')
FRAME_ASSIGNMENTS_PATH = 'frame_assignments.json'
FRAMES_DIR = 'frames'
OUTPUT_PATH = 'review_training_samples.npz'

MAX_FRAMES_PER_ADDR = 5  # Limit to avoid over-representing one region


def main():
    print("=== Build Review Training Samples ===\n")

    # Load review state
    if not os.path.exists(REVIEW_STATE_PATH):
        print("ERROR: review_state.json not found")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    lines = state.get('lines', {})
    confirmed = {addr: line for addr, line in lines.items()
                 if line.get('status') in ('accepted', 'edited')
                 and any(b != '--' for b in line.get('bytes', []))}

    print(f"Confirmed lines (accepted/edited): {len(confirmed)}")
    if not confirmed:
        print("No confirmed lines found. Accept or edit lines in the review tool first.")
        return

    # Load crop index
    with open(CROP_INDEX_PATH) as f:
        crop_index = json.load(f)

    # Load frame assignments (for row_y lookup)
    frame_assignments = None
    if os.path.exists(FRAME_ASSIGNMENTS_PATH):
        with open(FRAME_ASSIGNMENTS_PATH) as f:
            frame_assignments = json.load(f)
        print(f"Frame assignments loaded: {len(frame_assignments)} frames")
    else:
        print("WARNING: frame_assignments.json not found — will skip row_y lookup")

    # Extract labeled cells
    samples = defaultdict(list)
    addrs_used = 0
    frames_loaded = 0
    cells_extracted = 0

    for addr, line in sorted(confirmed.items()):
        confirmed_bytes = line['bytes']
        if all(b == '--' for b in confirmed_bytes):
            continue

        entry = crop_index.get(addr)
        if not entry or not isinstance(entry, dict):
            continue

        # Get frames for this address (extracted only — video frames
        # don't have entries in frame_assignments)
        frame_nums = entry.get('frames', [])
        if not frame_nums:
            continue

        # Limit frames per address
        if len(frame_nums) > MAX_FRAMES_PER_ADDR:
            # Evenly spaced subsample
            indices = np.linspace(0, len(frame_nums) - 1,
                                  MAX_FRAMES_PER_ADDR, dtype=int)
            frame_nums = [frame_nums[i] for i in indices]

        addr_int = int(addr, 16)
        addrs_used += 1

        for frame_num in frame_nums:
            frame_name = f"frame_{frame_num:05d}.png"
            frame_path = os.path.join(FRAMES_DIR, frame_name)

            # Find row_y for this address on this frame
            row_y = None
            if frame_assignments and frame_name in frame_assignments:
                for a in frame_assignments[frame_name]:
                    if a['addr'] == addr_int:
                        row_y = a['row_y']
                        break

            if row_y is None:
                continue

            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            frames_loaded += 1

            for byte_idx in range(16):
                byte_str = confirmed_bytes[byte_idx].upper()
                if byte_str == '--' or len(byte_str) != 2:
                    continue

                hi_char = byte_str[0]
                lo_char = byte_str[1]

                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING

                hi_cell = extract_cell(img, row_y, hi_x)
                lo_cell = extract_cell(img, row_y, lo_x)

                if not is_blank_cell(hi_cell):
                    samples[hi_char].append(hi_cell)
                    cells_extracted += 1

                if not is_blank_cell(lo_cell):
                    samples[lo_char].append(lo_cell)
                    cells_extracted += 1

    print(f"\nAddresses used: {addrs_used}")
    print(f"Frames loaded: {frames_loaded}")
    print(f"Cells extracted: {cells_extracted}")

    if not samples:
        print("No cells extracted. Check that frame_assignments.json exists.")
        return

    # Save as .npz
    arrays = {}
    print(f"\nPer-digit sample counts:")
    for char in sorted(samples.keys()):
        arr = np.stack(samples[char])
        arrays[char] = arr
        print(f"  '{char}': {len(samples[char])}")

    np.savez_compressed(OUTPUT_PATH, **arrays)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSaved to {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"Use 'python3 extract_pipeline.py --rebuild' to retrain the classifier")


if __name__ == '__main__':
    main()
