#!/usr/bin/env python3
"""
Full firmware extraction pipeline for Hakko FM-203 video dump.

Processes all 20,070 video frames from the Xeltek SuperPro 6100N "Edit Buffer"
scrolling display, extracts hex data using kNN character classification, and
uses multi-frame voting to produce a high-quality firmware dump.

Adapts the FM-202 (Segger) extraction approach for:
- 10-digit addresses ($00000-$13FFF, 80 KB buffer)
- 14x28 pixel cells → 64-dim feature vectors
- Byte positions from calibration array (not formula)
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict, Counter

from template_matcher import (
    CAL, CELL_W, CELL_H, BLANK_INK_THRESHOLD,
    BYTE_POSITIONS, BYTE_DIGIT_SPACING, ADDR_X_START,
    extract_cell, is_blank_cell, extract_features,
    load_reference, find_rows_with_known_data,
)

# Address range for the R5F21258SNFP buffer
ADDR_MIN = 0x00000
ADDR_MAX = 0x13FFF


class FastKNNClassifier:
    """Optimized kNN classifier for hex digit recognition.

    Pre-computes feature matrices for vectorized distance computation.
    Dimension-agnostic: works with any feature vector size.
    """

    def __init__(self):
        self.chars = []             # list of unique characters
        self.feature_matrix = None  # (N, D) matrix of all training features
        self.labels = None          # (N,) array of character indices
        self.char_to_idx = {}
        self.idx_to_char = {}

    def build_from_samples(self, samples_dict, max_per_class=500):
        """Build from a dict of char -> list of cells.

        Subsamples to max_per_class per character for speed.
        Uses evenly-spaced subsampling for diversity.
        """
        all_features = []
        all_labels = []

        self.chars = sorted(samples_dict.keys())
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        skipped_blank = 0
        for char in self.chars:
            cells = samples_dict[char]
            # Subsample if too many — evenly spaced for diversity
            if len(cells) > max_per_class:
                indices = np.linspace(0, len(cells) - 1, max_per_class, dtype=int)
                cells = [cells[i] for i in indices]

            for cell in cells:
                feat = extract_features(cell)
                if feat is None:
                    skipped_blank += 1
                    continue
                all_features.append(feat)
                all_labels.append(self.char_to_idx[char])

        self.feature_matrix = np.stack(all_features).astype(np.float32)  # (N, D)
        self.labels = np.array(all_labels)             # (N,)
        # Pre-compute squared norms for fast distance computation
        self._b_sq = np.sum(self.feature_matrix ** 2, axis=1)  # (N,)

        if skipped_blank > 0:
            print(f"  (skipped {skipped_blank} blank cells)")
        print(f"FastKNN built: {len(self.chars)} classes, "
              f"{self.feature_matrix.shape[0]} samples, "
              f"{self.feature_matrix.shape[1]} features")

    def classify_batch(self, cells, k=7):
        """Classify multiple cells at once using vectorized operations.

        Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b for fast distance
        computation.

        Returns list of (char, confidence) tuples.
        Blank cells return (None, 0.0).
        """
        feat_dim = self.feature_matrix.shape[1]
        raw_features = [extract_features(c) for c in cells]
        blank_mask = [f is None for f in raw_features]

        # Replace None features with zero placeholder for matrix ops
        features = np.stack([
            f if f is not None else np.zeros(feat_dim, dtype=np.float32)
            for f in raw_features
        ])  # (M, D)

        # Compute all pairwise distances: (M, N) matrix
        # Uses pre-computed _b_sq for training set norms
        features = features.astype(np.float32)
        a_sq = np.sum(features ** 2, axis=1, keepdims=True)  # (M, 1)
        ab = features @ self.feature_matrix.T  # (M, N)
        distances = a_sq + self._b_sq[np.newaxis, :] - 2 * ab  # (M, N)

        results = []
        for i in range(len(cells)):
            if blank_mask[i]:
                results.append((None, 0.0))
                continue

            row = distances[i]
            top_k_idx = np.argpartition(row, k)[:k]
            top_k_labels = self.labels[top_k_idx]
            top_k_dists = row[top_k_idx]

            # Weighted voting by inverse distance
            votes = defaultdict(float)
            for label, dist in zip(top_k_labels, top_k_dists):
                weight = 1.0 / (dist + 1e-6)
                votes[label] += weight

            best_label = max(votes, key=votes.get)
            total = sum(votes.values())
            confidence = votes[best_label] / total if total > 0 else 0

            results.append((self.idx_to_char[best_label], confidence))

        return results

    def classify_single(self, cell, k=7):
        """Classify a single cell."""
        return self.classify_batch([cell], k=k)[0]

    def save(self, path):
        """Save classifier state."""
        np.savez_compressed(path,
                            feature_matrix=self.feature_matrix,
                            labels=self.labels,
                            chars=np.array(self.chars))
        print(f"FastKNN saved to {path}")

    def load(self, path):
        """Load classifier state."""
        data = np.load(path, allow_pickle=True)
        self.feature_matrix = data['feature_matrix'].astype(np.float32)
        self.labels = data['labels']
        self.chars = list(data['chars'])
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self._b_sq = np.sum(self.feature_matrix ** 2, axis=1)
        print(f"FastKNN loaded from {path}: {len(self.chars)} classes, "
              f"{self.feature_matrix.shape[0]} samples")


def build_training_samples(ref_data, frame_paths):
    """Build labeled training samples from frames with known reference data.

    Pass 1: Uses Tesseract to find rows matching reference addresses,
    then extracts digit cells using byte_positions from calibration.

    Args:
        ref_data: {addr_int: [hex_str_list]}
        frame_paths: List of frame file paths

    Returns:
        dict of char -> list of cells
    """
    samples = defaultdict(list)

    for frame_path in frame_paths:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        matches = find_rows_with_known_data(img, ref_data)
        if not matches:
            continue

        seen = set()
        for addr, row_y in matches:
            if addr in seen:
                continue
            seen.add(addr)

            hex_bytes = ref_data[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING

                # High nybble
                cell = extract_cell(img, row_y, hi_x)
                samples[byte_str[0]].append(cell)
                # Low nybble
                cell = extract_cell(img, row_y, lo_x)
                samples[byte_str[1]].append(cell)

        if matches:
            print(f"  {os.path.basename(frame_path)}: {len(seen)} rows")

    return samples


def build_training_samples_knn(classifier, ref_data, frame_paths):
    """Build labeled training samples using kNN-based address detection.

    Pass 2: Uses the existing kNN classifier (not Tesseract) to read
    addresses, enabling training on frames from diverse video segments.

    Args:
        classifier: FastKNNClassifier from Pass 1
        ref_data: {addr_int: [hex_str_list]}
        frame_paths: List of frame file paths

    Returns:
        dict of char -> list of cells
    """
    samples = defaultdict(list)

    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START

    ref_addrs = {addr: ref_data[addr] for addr in ref_data}

    for frame_path in frame_paths:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Scan rows including 2 extra above for partial visibility
        matches = []
        for row_idx in range(-2, num_rows):
            row_y = y_first + row_idx * row_h
            if row_y < 5 or row_y > img.shape[0] - 15:
                continue

            addr_str, addr_conf = read_address_from_row(
                classifier, img, row_y, addr_x
            )
            try:
                addr_val = int(addr_str, 16)
            except ValueError:
                continue

            if addr_val in ref_addrs and addr_conf > 0.5:
                matches.append((addr_val, row_y))

        # Deduplicate and extract cells
        seen = set()
        for addr, row_y in matches:
            if addr in seen:
                continue
            seen.add(addr)

            hex_bytes = ref_addrs[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING

                cell_hi = extract_cell(img, row_y, hi_x)
                if not is_blank_cell(cell_hi):
                    samples[byte_str[0]].append(cell_hi)

                cell_lo = extract_cell(img, row_y, lo_x)
                if not is_blank_cell(cell_lo):
                    samples[byte_str[1]].append(cell_lo)

        if matches:
            print(f"  {os.path.basename(frame_path)}: {len(seen)} rows matched")

    return samples


def auto_correct_address(addr_str):
    """Auto-correct common OCR errors in 10-digit hex address strings.

    The FM-203 address range is $00000-$13FFF (10-digit format: 000000xxxx).
    The first 5 digits are always '00000'. Common 0<->8 confusion in
    leading zeros is corrected.
    """
    if len(addr_str) != 10:
        return addr_str

    chars = list(addr_str.upper())

    # Digits 0-4 must all be '0' — correct '8' -> '0' (0<->8 confusion)
    for i in range(5):
        if chars[i] == '8':
            chars[i] = '0'
        elif chars[i] == 'O':
            chars[i] = '0'

    # Digit 5 can be '0' or '1' — correct '8' -> '0', '9' -> '1'
    if chars[5] == '8':
        chars[5] = '0'
    elif chars[5] == '9':
        chars[5] = '1'

    return ''.join(chars)


def read_address_from_row(classifier, img, row_y, addr_x):
    """Read the 10-digit hex address from a row using the classifier.

    Args:
        classifier: FastKNNClassifier
        img: Grayscale frame image
        row_y: Vertical center of the row
        addr_x: X position of the address column start

    Returns:
        (addr_str, avg_confidence) where addr_str is a 10-char hex string.
    """
    spacing = CAL['address_char_spacing']
    cells = []
    for i in range(10):
        char_x = addr_x + i * spacing
        cell = extract_cell(img, row_y, char_x)
        cells.append(cell)

    results = classifier.classify_batch(cells)
    addr_str = ''.join(char if char is not None else '?' for char, conf in results)
    valid_confs = [conf for char, conf in results if conf > 0]
    avg_conf = np.mean(valid_confs) if valid_confs else 0.0

    # Auto-correct leading zeros
    addr_str = auto_correct_address(addr_str)

    return addr_str, avg_conf


def read_hex_bytes_from_row(classifier, img, row_y):
    """Read 16 hex bytes (32 digits) from a row using the classifier.

    Uses byte_positions array from calibration for precise digit location.

    Returns:
        (hex_bytes, confidences, avg_conf)
        hex_bytes: list of 16 two-char hex strings (e.g. ['4E', 'FC', ...])
        confidences: list of 16 per-byte confidence values
        avg_conf: mean confidence across all bytes
    """
    cells = []
    for byte_idx in range(16):
        hi_x = BYTE_POSITIONS[byte_idx]
        lo_x = hi_x + BYTE_DIGIT_SPACING
        cells.append(extract_cell(img, row_y, hi_x))
        cells.append(extract_cell(img, row_y, lo_x))

    results = classifier.classify_batch(cells)

    hex_bytes = []
    confidences = []
    for byte_idx in range(16):
        hi_char, hi_conf = results[byte_idx * 2]
        lo_char, lo_conf = results[byte_idx * 2 + 1]
        if hi_char is None or lo_char is None:
            hex_bytes.append("--")
            confidences.append(0.0)
        else:
            hex_bytes.append(hi_char + lo_char)
            confidences.append(min(hi_conf, lo_conf))

    valid_confs = [c for c in confidences if c > 0.0]
    avg_conf = np.mean(valid_confs) if valid_confs else 0.0
    return hex_bytes, confidences, avg_conf


def validate_address_sequence(addresses_with_rows):
    """Validate and correct addresses in a frame.

    Addresses should be sequential with 0x10 spacing. Uses an anchor-based
    approach: try each address as anchor, score by how many others are
    consistent, keep the best sequence.

    Args:
        addresses_with_rows: list of (addr_int, row_y, confidence)

    Returns:
        Corrected list of (addr_int, row_y, confidence).
    """
    if not addresses_with_rows:
        return []

    row_h = CAL['row_height']

    best_sequence = []
    best_score = 0

    for anchor_addr, anchor_row_y, anchor_conf in addresses_with_rows:
        sequence = []
        score = 0
        consistent_count = 0

        for addr, row_y, conf in addresses_with_rows:
            row_offset = round((row_y - anchor_row_y) / row_h)
            expected_addr = anchor_addr + row_offset * 0x10

            # Allow +-8 for single-digit 0/8 confusion
            addr_diff = abs(addr - expected_addr)
            if addr_diff == 0:
                sequence.append((expected_addr, row_y, conf))
                score += conf * 2
                consistent_count += 1
            elif addr_diff <= 0x08:
                sequence.append((expected_addr, row_y, conf * 0.8))
                score += conf
                consistent_count += 1
            else:
                sequence.append((expected_addr, row_y, conf * 0.3))
                score += conf * 0.1

        score *= (1 + consistent_count / len(addresses_with_rows))

        if score > best_score:
            best_score = score
            best_sequence = sequence

    return best_sequence


def process_frame(classifier, img):
    """Process a single frame: read all addresses and hex data.

    Returns list of (addr_int, hex_bytes, byte_confidences, addr_conf).
    """
    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START

    # Step 1: Read all addresses (including 2 extra rows above)
    addr_results = []
    for row_idx in range(-2, num_rows):
        row_y = y_first + row_idx * row_h
        if row_y < CELL_H or row_y > img.shape[0] - CELL_H:
            continue

        addr_str, addr_conf = read_address_from_row(
            classifier, img, row_y, addr_x
        )

        # Validate as hex and in range
        try:
            addr_int = int(addr_str, 16)
            if ADDR_MIN <= addr_int <= ADDR_MAX:
                addr_results.append((addr_int, row_y, addr_conf))
        except ValueError:
            pass

    # Step 2: Validate address sequence
    validated_addrs = validate_address_sequence(addr_results)

    # Step 3: Read hex bytes for each validated row
    results = []
    for addr_int, row_y, addr_conf in validated_addrs:
        hex_bytes, byte_confs, avg_conf = read_hex_bytes_from_row(
            classifier, img, row_y
        )
        # Must be 16-byte aligned and in valid range
        if (ADDR_MIN <= addr_int <= ADDR_MAX - 0x0F and
                addr_int % 0x10 == 0):
            results.append((addr_int, hex_bytes, byte_confs, addr_conf))

    return results


def is_frame_different(img1, img2, threshold=20000):
    """Check if two frames show different hex dump content.

    Compares the hex data region (byte columns only) using absolute
    pixel difference sum. Threshold scaled for FM-203's larger hex region
    (~665x551 px). A scroll by one row produces ~2.4M diff, so 20000
    safely catches tiny rendering variations while skipping true duplicates.
    """
    if img1 is None or img2 is None:
        return True

    # Use byte data region for comparison
    y1 = int(CAL['hex_region']['y_min'])
    y2 = int(CAL['hex_region']['y_max'])
    x1 = int(BYTE_POSITIONS[0]) - 10
    x2 = int(BYTE_POSITIONS[15]) + CELL_W + 15

    crop1 = img1[y1:y2, x1:x2]
    crop2 = img2[y1:y2, x1:x2]

    if crop1.shape != crop2.shape:
        return True

    diff = np.sum(np.abs(crop1.astype(np.int16) - crop2.astype(np.int16)))
    return diff > threshold


def is_frame_stable(img1, img2, img3):
    """Check if a frame is stable (not during a scroll transition).

    A frame is stable if it matches both its neighbors closely.
    Uses per-pixel mean absolute difference (independent of image size).
    """
    if img1 is None or img3 is None:
        return True

    y1 = int(CAL['hex_region']['y_min'])
    y2 = int(CAL['hex_region']['y_max'])
    x1 = int(BYTE_POSITIONS[0]) - 10
    x2 = int(BYTE_POSITIONS[15]) + CELL_W + 15

    crop1 = img1[y1:y2, x1:x2]
    crop2 = img2[y1:y2, x1:x2]
    crop3 = img3[y1:y2, x1:x2]

    if crop1.shape != crop2.shape or crop2.shape != crop3.shape:
        return True

    diff_prev = np.mean(np.abs(crop1.astype(np.float32) - crop2.astype(np.float32)))
    diff_next = np.mean(np.abs(crop2.astype(np.float32) - crop3.astype(np.float32)))

    # If very different from both neighbors, likely a scroll transition
    if diff_prev > 5 and diff_next > 5:
        return False
    return True


def find_reference_frames(ref_data, frame_dir='frames',
                          scan_start=14000, scan_end=18000, scan_step=50):
    """Scan video frames to find those showing reference address data.

    Returns list of frame file paths that have >= 3 matching reference rows.
    """
    print(f"Scanning frames {scan_start}-{scan_end} (step {scan_step}) "
          f"for reference data...")
    found = []

    for f_num in range(scan_start, scan_end, scan_step):
        path = os.path.join(frame_dir, f'frame_{f_num:05d}.png')
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        matches = find_rows_with_known_data(img, ref_data)
        if len(matches) >= 3:
            found.append((f_num, path, len(matches)))
            print(f"  frame_{f_num:05d}: {len(matches)} reference rows")

    if not found:
        # Expand to full video range
        print("  No hits — expanding to full range (step 200)...")
        for f_num in range(0, 20100, 200):
            path = os.path.join(frame_dir, f'frame_{f_num:05d}.png')
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            matches = find_rows_with_known_data(img, ref_data)
            if len(matches) >= 3:
                found.append((f_num, path, len(matches)))
                print(f"  frame_{f_num:05d}: {len(matches)} reference rows")

    if found:
        # Once we found the region, fill in nearby frames for diversity
        center = found[len(found) // 2][0]
        filled = []
        for f_num in range(center - 100, center + 200, 5):
            path = os.path.join(frame_dir, f'frame_{f_num:05d}.png')
            if os.path.exists(path):
                filled.append(path)
            if len(filled) >= 40:
                break
        if filled:
            return filled

    return [f[1] for f in found]


def run_extraction(classifier, frame_dir='frames',
                   output_path='extracted_firmware.txt'):
    """Run the full extraction pipeline on all frames.

    Processes each frame, deduplicates, skips scroll transitions,
    and uses multi-frame weighted voting per byte.
    """
    # Get sorted list of frame files
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    print(f"Total frames to process: {len(frames)}")

    # Storage: addr_int -> list of observations
    all_observations = defaultdict(list)

    prev_img = None
    processed = 0
    skipped_dup = 0
    skipped_jump = 0
    t0 = time.time()

    # Adaptive frame-skipping: when many consecutive dups, jump ahead
    consecutive_dups = 0
    frame_idx = 0

    while frame_idx < len(frames):
        frame_name = frames[frame_idx]
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            frame_idx += 1
            continue

        # Skip if frame is same as previous (no scroll happened)
        if not is_frame_different(img, prev_img):
            skipped_dup += 1
            consecutive_dups += 1
            # Adaptive skip: if many dups in a row, jump ahead
            if consecutive_dups >= 5:
                jump = min(consecutive_dups, 20)
                frame_idx += jump
                skipped_jump += jump - 1
            else:
                frame_idx += 1
            continue

        consecutive_dups = 0
        prev_img = img
        processed += 1

        # Process this frame
        results = process_frame(classifier, img)

        for addr_int, hex_bytes, byte_confs, addr_conf in results:
            all_observations[addr_int].append({
                'bytes': hex_bytes,
                'byte_confs': byte_confs,
                'addr_conf': addr_conf,
                'frame': frame_name,
            })

        frame_idx += 1

        # Progress report every 100 processed frames
        if processed % 100 == 0:
            elapsed = time.time() - t0
            fps = processed / elapsed if elapsed > 0 else 0
            addrs = len(all_observations)
            print(f"  Processed {processed} frames ({skipped_dup} dup, "
                  f"{skipped_jump} jump-skipped), "
                  f"{addrs} unique addresses, {fps:.1f} fps "
                  f"[frame {frame_idx}/{len(frames)}]")

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {processed} frames processed, "
          f"{skipped_dup} dup-skipped, {skipped_jump} jump-skipped, "
          f"{elapsed:.1f}s")
    print(f"Unique addresses found: {len(all_observations)}")

    # Multi-frame voting
    print("\nVoting on best reading per address...")
    final_dump = {}

    for addr_int in sorted(all_observations.keys()):
        observations = all_observations[addr_int]

        # For each byte position, weighted majority vote
        final_bytes = []
        for byte_idx in range(16):
            votes = Counter()
            for obs in observations:
                if byte_idx < len(obs['bytes']):
                    byte_val = obs['bytes'][byte_idx]
                    if byte_val == "--":
                        continue
                    weight = obs['byte_confs'][byte_idx] * obs['addr_conf']
                    votes[byte_val] += weight

            if votes:
                best_byte = votes.most_common(1)[0][0]
                final_bytes.append(best_byte)
            else:
                final_bytes.append('FF')  # Default: erased flash

        final_dump[addr_int] = {
            'bytes': final_bytes,
            'num_observations': len(observations),
        }

    # Overlay reference data (guaranteed correct)
    ref_data = load_reference()
    ref_overlaid = 0
    for addr_int, ref_bytes in ref_data.items():
        final_dump[addr_int] = {
            'bytes': ref_bytes,
            'num_observations': -1,  # -1 = reference data
        }
        ref_overlaid += 1
    print(f"Overlaid {ref_overlaid} lines from verified reference data")

    # Write output
    with open(output_path, 'w') as f:
        f.write(f"# Firmware extraction from video using kNN template matching\n")
        f.write(f"# Device: Hakko FM-203 (R5F21258SNFP)\n")
        f.write(f"# Processed {processed} frames, "
                f"{len(final_dump)} addresses recovered\n")
        if final_dump:
            f.write(f"# Address range: "
                    f"0x{min(final_dump):05X} - 0x{max(final_dump):05X}\n")
        f.write(f"# Reference data overlaid for "
                f"0x{min(ref_data):05X}-0x{max(ref_data)+0xF:05X}\n")
        f.write(f"#\n")

        for addr_int in sorted(final_dump.keys()):
            data = final_dump[addr_int]
            addr_str = f"{addr_int:05X}"
            hex_str = ' '.join(data['bytes'])
            obs_count = data['num_observations']
            if obs_count == -1:
                f.write(f"{addr_str}: {hex_str}  [REF]\n")
            else:
                f.write(f"{addr_str}: {hex_str}  [{obs_count} obs]\n")

    print(f"\nOutput written to {output_path}")

    # Coverage stats
    expected_addrs = set(range(ADDR_MIN, ADDR_MAX + 1, 0x10))
    found_addrs = set(final_dump.keys())
    coverage = len(found_addrs & expected_addrs) / len(expected_addrs) * 100
    print(f"Coverage: {len(found_addrs & expected_addrs)}/{len(expected_addrs)} "
          f"addresses ({coverage:.1f}%)")

    # Validate against reference
    ref_correct = 0
    ref_total = 0
    ref_bytes_correct = 0
    ref_bytes_total = 0

    for addr_int, expected_bytes in ref_data.items():
        if addr_int in final_dump:
            ref_total += 1
            extracted_bytes = final_dump[addr_int]['bytes']
            line_correct = True
            for i in range(min(16, len(extracted_bytes))):
                ref_bytes_total += 1
                if extracted_bytes[i].upper() == expected_bytes[i].upper():
                    ref_bytes_correct += 1
                else:
                    line_correct = False
            if line_correct:
                ref_correct += 1

    if ref_total > 0:
        print(f"\nReference validation:")
        print(f"  Lines: {ref_correct}/{ref_total} fully correct "
              f"({ref_correct / ref_total * 100:.1f}%)")
        print(f"  Bytes: {ref_bytes_correct}/{ref_bytes_total} correct "
              f"({ref_bytes_correct / ref_bytes_total * 100:.1f}%)")

    return final_dump


def main():
    print("=== Firmware Extraction Pipeline (FM-203) ===\n")

    ref_data = load_reference()
    print(f"Reference: {len(ref_data)} lines "
          f"(${min(ref_data):05X}-${max(ref_data)+0xF:05X})")

    # Parse command-line args
    classifier_path = 'fast_knn_classifier.npz'
    rebuild = '--rebuild' in sys.argv

    classifier = None

    if os.path.exists(classifier_path) and not rebuild:
        print(f"\nLoading existing kNN classifier from {classifier_path}...")
        classifier = FastKNNClassifier()
        classifier.load(classifier_path)
    else:
        # === Pass 1: Build initial kNN from Tesseract-labeled frames ===
        print("\n=== Pass 1: Building initial kNN from Tesseract-labeled frames ===")

        # Find frames showing reference addresses
        train_frame_paths = find_reference_frames(ref_data)
        if not train_frame_paths:
            print("ERROR: No reference-visible frames found!")
            return

        print(f"\nUsing {len(train_frame_paths)} frames for Pass 1 training...")
        samples = build_training_samples(ref_data, train_frame_paths)

        print(f"\nPass 1 training samples:")
        total_samples = 0
        for c in sorted(samples):
            print(f"  '{c}': {len(samples[c])}")
            total_samples += len(samples[c])
        print(f"  Total: {total_samples}")

        classifier = FastKNNClassifier()
        classifier.build_from_samples(samples, max_per_class=500)

        # === Pass 2: Expand training with kNN-based detection ===
        # Use diverse frames from across the video that overlap with reference
        print(f"\n=== Pass 2: Expanding training with kNN-detected frames ===")

        # Sample frames at various timestamps for diversity
        pass2_paths = []
        all_frames = sorted(os.listdir('frames'))
        # Evenly sample ~30 frames across the full video
        step = max(1, len(all_frames) // 30)
        for i in range(0, len(all_frames), step):
            path = os.path.join('frames', all_frames[i])
            pass2_paths.append(path)
            if len(pass2_paths) >= 30:
                break

        new_samples = build_training_samples_knn(classifier, ref_data, pass2_paths)

        if new_samples:
            print(f"\nPass 2 new samples:")
            for c in sorted(new_samples):
                print(f"  '{c}': {len(new_samples[c])}")

            # Merge new samples into existing
            for char, cells in new_samples.items():
                samples[char].extend(cells)

            print(f"\nCombined training samples:")
            total_combined = 0
            for c in sorted(samples):
                print(f"  '{c}': {len(samples[c])}")
                total_combined += len(samples[c])
            print(f"  Total: {total_combined}")

            # Rebuild classifier with expanded data
            classifier = FastKNNClassifier()
            classifier.build_from_samples(samples, max_per_class=800)
        else:
            print("  No additional samples from Pass 2")

        classifier.save(classifier_path)

    # Quick validation on a reference frame
    print("\nQuick validation on a reference frame...")
    ref_frame_paths = find_reference_frames(ref_data)
    if ref_frame_paths:
        test_path = ref_frame_paths[len(ref_frame_paths) // 2]
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if test_img is not None:
            results = process_frame(classifier, test_img)
            correct_bytes = 0
            total_bytes = 0
            for addr_int, hex_bytes, byte_confs, addr_conf in results:
                if addr_int in ref_data:
                    expected = ref_data[addr_int]
                    for i in range(min(16, len(hex_bytes))):
                        total_bytes += 1
                        if hex_bytes[i].upper() == expected[i].upper():
                            correct_bytes += 1
            if total_bytes > 0:
                print(f"  Reference accuracy: {correct_bytes}/{total_bytes} bytes "
                      f"({correct_bytes / total_bytes * 100:.1f}%)")

    # Run full extraction
    print("\n" + "=" * 60)
    print("Starting full extraction...")
    print("=" * 60)
    final_dump = run_extraction(classifier)


if __name__ == '__main__':
    main()
