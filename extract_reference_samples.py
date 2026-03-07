#!/usr/bin/env python3
"""
Extract labeled digit cells from the reference screenshot PNG.

Detects rows and byte positions in reference/reference_screenshot.png using
the same approach as measure_reference_geometry.py, extracts individual digit
cells, resizes to 14x28 (video cell dimensions), labels with ground truth
from the reference transcription, and saves as a FastKNNClassifier.
"""

import cv2
import numpy as np
from collections import defaultdict

from template_matcher import load_reference, extract_features
from extract_pipeline import FastKNNClassifier

IMAGE_PATH = 'reference/reference_screenshot.png'
OUTPUT_PATH = 'ref_classifier_data.npz'

# Video cell dimensions (target size for resized reference digits)
TARGET_W = 14
TARGET_H = 28


def detect_row_y_ranges(img_gray, img_color):
    """Detect text band y-ranges for each data row.

    Returns list of (y_start, y_end) for up to 16 data rows.
    """
    h, w = img_gray.shape

    # Detect green header bars to find where data starts
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    row_green = np.sum(green_mask > 0, axis=1)
    green_rows = np.where(row_green > w * 0.05)[0]

    if len(green_rows) > 0:
        header_bottom = int(green_rows[-1]) + 1
        # Find HEX bar x-range
        header_top = int(green_rows[0])
        header_mask = green_mask[header_top:header_bottom, :]
        col_green = np.sum(header_mask > 0, axis=0)
        min_coverage = max((header_bottom - header_top) * 0.15, 3)
        green_cols = np.where(col_green > min_coverage)[0]
        bars = []
        if len(green_cols) > 0:
            bar_start = green_cols[0]
            for i in range(1, len(green_cols)):
                if green_cols[i] - green_cols[i - 1] > 15:
                    bars.append((int(bar_start), int(green_cols[i - 1])))
                    bar_start = green_cols[i]
            bars.append((int(bar_start), int(green_cols[-1])))
        hex_x0, hex_x1 = bars[1] if len(bars) >= 2 else (190, 756)
    else:
        header_bottom = 0
        hex_x0, hex_x1 = 190, 756

    # Scan for text bands (avg intensity < 254)
    row_y_ranges = []
    in_text = False
    text_start = 0

    for y in range(header_bottom, int(h * 0.70)):
        avg = np.mean(img_gray[y, hex_x0:hex_x1].astype(float))
        if avg < 254 and not in_text:
            in_text = True
            text_start = y
        elif avg >= 254 and in_text:
            in_text = False
            text_end = y - 1
            if text_end - text_start + 1 >= 8:
                row_y_ranges.append((text_start, text_end))

    if in_text:
        text_end = int(h * 0.70) - 1
        if text_end - text_start + 1 >= 8:
            row_y_ranges.append((text_start, text_end))

    # Keep only first 16 data rows
    return row_y_ranges[:16]


def detect_byte_positions(img_gray, row_y_range):
    """Detect byte left edges from ink-run analysis on a row.

    Returns list of 16 byte left-edge x positions.
    """
    ys, ye = row_y_range
    row_strip = np.mean(img_gray[ys:ye + 1, :].astype(float), axis=0)

    # Find ink runs in the hex data area
    ink_runs = []
    in_ink = False
    ink_start = 0
    for x in range(170, 770):
        if row_strip[x] < 250 and not in_ink:
            in_ink = True
            ink_start = x
        elif row_strip[x] >= 250 and in_ink:
            in_ink = False
            ink_runs.append((ink_start, x - 1))
    if in_ink:
        ink_runs.append((ink_start, 769))

    # Map ink runs to bytes using the known pattern for row 0
    # Row 0: "4E FC 00 00 4E FC 00 00-4E FC 00 00 4E FC 00 00"
    # 4E -> 1 merged run, FC -> 1 merged run, 00 -> 2 separate runs
    # = 6 runs per 4-byte group, dash at run 12
    byte_run_map = [
        ([0], 0), ([1], 1), ([2, 3], 2), ([4, 5], 3),
        ([6], 4), ([7], 5), ([8, 9], 6), ([10, 11], 7),
        # Dash at run 12
        ([13], 8), ([14], 9), ([15, 16], 10), ([17, 18], 11),
        ([19], 12), ([20], 13), ([21, 22], 14), ([23, 24], 15),
    ]

    byte_left_edges = []
    for run_indices, _byte_idx in byte_run_map:
        if run_indices[0] < len(ink_runs):
            byte_left_edges.append(ink_runs[run_indices[0]][0])

    return byte_left_edges


def extract_reference_samples():
    """Extract labeled digit cells from the reference screenshot.

    Returns dict[char] -> list[cell] where each cell is a 28x14 uint8 array.
    """
    img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

    print(f"Loaded {IMAGE_PATH}: {img_gray.shape[1]}x{img_gray.shape[0]}")

    # Detect geometry
    row_y_ranges = detect_row_y_ranges(img_gray, img_color)
    print(f"Detected {len(row_y_ranges)} text rows")

    # Use hardcoded byte positions (from measure_reference_geometry.py output)
    # These are stable for the reference screenshot
    byte_left_edges = [193, 228, 264, 300, 337, 372, 408, 444,
                       481, 516, 552, 588, 625, 660, 696, 732]
    byte_stride = 35.9333
    char_width = byte_stride / 3.0  # ~11.98px

    # Load ground truth
    ref_data = load_reference()
    addresses = sorted(ref_data.keys())
    print(f"Ground truth: {len(addresses)} addresses")

    if len(row_y_ranges) < len(addresses):
        print(f"WARNING: Only {len(row_y_ranges)} rows detected, "
              f"expected {len(addresses)}")

    samples = defaultdict(list)
    total_cells = 0

    for row_idx, (ys, ye) in enumerate(row_y_ranges):
        if row_idx >= len(addresses):
            break

        addr = addresses[row_idx]
        hex_bytes = ref_data[addr]
        text_height = ye - ys + 1

        for byte_idx in range(16):
            byte_str = hex_bytes[byte_idx]
            byte_left = byte_left_edges[byte_idx]

            for digit_pos in range(2):
                digit_char = byte_str[digit_pos]
                # Digit x position: byte_left + digit_pos * char_width
                digit_x = byte_left + digit_pos * char_width

                # Extract at native resolution
                x1 = int(round(digit_x))
                x2 = int(round(digit_x + char_width))
                y1 = ys
                y2 = ye + 1  # inclusive -> exclusive

                # Clamp to image bounds
                x1 = max(0, x1)
                x2 = min(img_gray.shape[1], x2)
                y1 = max(0, y1)
                y2 = min(img_gray.shape[0], y2)

                cell_native = img_gray[y1:y2, x1:x2]

                # Resize to video cell dimensions (14x28)
                cell_resized = cv2.resize(cell_native,
                                          (TARGET_W, TARGET_H),
                                          interpolation=cv2.INTER_CUBIC)

                samples[digit_char].append(cell_resized)
                total_cells += 1

    return samples, total_cells


def main():
    print("=== Extract Reference Samples ===\n")

    samples, total_cells = extract_reference_samples()

    print(f"\nExtracted {total_cells} digit cells "
          f"(expected 512 = 16 rows x 16 bytes x 2 digits)")
    print(f"\nCharacter distribution:")
    for char in sorted(samples.keys()):
        print(f"  '{char}': {len(samples[char]):4d} samples")

    # Build and save classifier
    print(f"\nBuilding FastKNNClassifier...")
    classifier = FastKNNClassifier()
    classifier.build_from_samples(samples, max_per_class=500)
    classifier.save(OUTPUT_PATH)

    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
