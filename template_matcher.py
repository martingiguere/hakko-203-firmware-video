#!/usr/bin/env python3
"""
Hex digit recognition for Xeltek SuperPro 6100N firmware dump video.

Adapts the FM-202 (Segger) template matching approach for the FM-203's
larger cells (14x28 pixels, producing 64-dim feature vectors).

Combines:
1. Template matching (averaged per-class templates)
2. kNN classification on raw pixel cells
3. kNN classification on structural features (64-dim)
4. Ensemble voting across all three methods
"""

import cv2
import numpy as np
import json
import os
from collections import defaultdict, Counter

# Load calibration (relative to project root — scripts run from there)
with open('grid_calibration.json') as f:
    CAL = json.load(f)

# Cell geometry from calibration
CELL_W = CAL['cell_width']   # 14
CELL_H = CAL['cell_height']  # 28

# Blank cell detection threshold.
# 14x28=392 pixels; real characters have total ink ~30-80; blank cells < 1.0.
# Scaled from FM-202's 3.0 for the larger cell area (392 vs 126 pixels).
BLANK_INK_THRESHOLD = 6.0

# Byte positions from calibration (byte_index -> x_position)
BYTE_POSITIONS = {bp['byte']: bp['x'] for bp in CAL['byte_positions']}
BYTE_DIGIT_SPACING = CAL['byte_digit_spacing']  # 13.92

# Corrected address x-start: the calibration detected a UI vertical separator
# line at x=271 instead of the actual first digit at x=289.  Pixel profile
# analysis shows constant ink at 271 across all rows (UI element) while real
# address characters begin at ~289 with 13.92px spacing.
ADDR_X_START = 289


def load_reference():
    """Load verified reference transcription.

    Wraps analyze_reference.load_transcription() and converts integer byte
    values to uppercase hex strings.

    Returns {addr_int: [hex_str_list]}
        e.g. {0x0FF70: ['4E', 'FC', '00', '00', ...]}
    """
    from analyze_reference import load_transcription
    raw = load_transcription('reference/reference_transcription.txt')
    ref = {}
    for addr, byte_vals in raw.items():
        ref[addr] = [f'{b:02X}' for b in byte_vals]
    return ref


def extract_cell(img, row_y, char_x, cell_h=CELL_H, cell_w=CELL_W,
                 y_off=None, x_off=-1):
    """Extract a character cell from the image.

    Args:
        img: Grayscale image
        row_y: Vertical center of the text row
        char_x: Horizontal position of character left edge
        cell_h: Cell height (default 28)
        cell_w: Cell width (default 14)
        y_off: Vertical offset from row_y to top of cell (default: -cell_h//2)
        x_off: Horizontal offset from char_x to left of cell (default: -1)

    Returns:
        cell: (cell_h, cell_w) uint8 array
    """
    if y_off is None:
        y_off = -(cell_h // 2)

    y1 = int(row_y + y_off)
    y2 = y1 + cell_h
    x1 = int(char_x + x_off)
    x2 = x1 + cell_w

    h, w = img.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)

    cell = img[y1:y2, x1:x2]
    if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
        padded = np.full((cell_h, cell_w), 255, dtype=np.uint8)
        ph = min(cell.shape[0], cell_h)
        pw = min(cell.shape[1], cell_w)
        padded[:ph, :pw] = cell[:ph, :pw]
        cell = padded
    return cell


def is_blank_cell(cell, threshold=BLANK_INK_THRESHOLD):
    """Check if a cell is blank (no meaningful character content).

    Uses raw pixel ink density BEFORE normalization.
    """
    ink = (255.0 - cell.astype(np.float32)) / 255.0
    return ink.sum() < threshold


def extract_features(cell):
    """Extract structural features from a character cell.

    For 28x14 cells produces a 64-dim vector:
        h_profile(28) + v_profile(14) + quadrants(4) + center(3) +
        asymmetry(2) + center_bar(1) + symmetry(1) + corners(2) +
        3x3_grid(9) = 64

    Dimension auto-adapts to cell size via h,w = cell.shape.
    Returns None if the cell is blank.
    """
    if is_blank_cell(cell):
        return None

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

    return np.array(features, dtype=np.float32)


class HexDigitClassifier:
    """Multi-strategy hex digit classifier.

    Uses a combination of template matching, kNN on raw pixels,
    and kNN on structural features with ensemble voting.
    """

    def __init__(self):
        self.samples = defaultdict(list)          # char -> list of raw cells
        self.feature_samples = defaultdict(list)  # char -> list of feature vectors
        self.templates = {}                       # char -> averaged template
        self.median_templates = {}                # char -> median template

    def add_sample(self, char, cell):
        """Add a labeled training sample."""
        self.samples[char].append(cell.copy())
        feat = extract_features(cell)
        self.feature_samples[char].append(feat)

    def build(self):
        """Build templates and prepare for classification."""
        for char, cells in self.samples.items():
            if not cells:
                continue
            stack = np.stack(cells, axis=0).astype(np.float32)
            self.templates[char] = np.mean(stack, axis=0).astype(np.uint8)
            self.median_templates[char] = np.median(stack, axis=0).astype(np.uint8)

        print(f"Classifier built: {len(self.templates)} characters, "
              f"{sum(len(v) for v in self.samples.values())} total samples")
        for c in sorted(self.templates):
            print(f"  '{c}': {len(self.samples[c])} samples")

    def classify(self, cell, method='ensemble'):
        """Classify a character cell. Returns (predicted_char, confidence)."""
        if method == 'template':
            return self._classify_template(cell)
        elif method == 'knn':
            return self._classify_knn(cell, k=5)
        elif method == 'feature_knn':
            return self._classify_feature_knn(cell, k=7)
        elif method == 'ensemble':
            return self._classify_ensemble(cell)
        return self._classify_template(cell)

    def _classify_template(self, cell):
        """Template matching using normalized cell."""
        cell_f = cell.astype(np.float32)
        mn, mx = cell_f.min(), cell_f.max()
        if mx > mn:
            cell_f = (cell_f - mn) / (mx - mn) * 255

        best_char = None
        best_score = float('inf')
        scores = {}
        for char, tmpl in self.templates.items():
            tmpl_f = tmpl.astype(np.float32)
            tmn, tmx = tmpl_f.min(), tmpl_f.max()
            if tmx > tmn:
                tmpl_f = (tmpl_f - tmn) / (tmx - tmn) * 255
            dist = np.sum((cell_f - tmpl_f) ** 2)
            scores[char] = dist
            if dist < best_score:
                best_score = dist
                best_char = char

        sorted_scores = sorted(scores.values())
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            confidence = 1.0 - sorted_scores[0] / sorted_scores[1]
        else:
            confidence = 1.0
        return best_char, confidence

    def _classify_knn(self, cell, k=5):
        """kNN classification on normalized raw pixel values."""
        cell_f = cell.astype(np.float32).flatten()
        mn, mx = cell_f.min(), cell_f.max()
        if mx > mn:
            cell_f = (cell_f - mn) / (mx - mn)

        distances = []
        for char, cells in self.samples.items():
            for sample in cells:
                sample_f = sample.astype(np.float32).flatten()
                smn, smx = sample_f.min(), sample_f.max()
                if smx > smn:
                    sample_f = (sample_f - smn) / (smx - smn)
                dist = np.sum((cell_f - sample_f) ** 2)
                distances.append((dist, char))

        distances.sort(key=lambda x: x[0])
        votes = Counter()
        for dist, char in distances[:k]:
            weight = 1.0 / (dist + 1e-6)
            votes[char] += weight

        best_char = votes.most_common(1)[0][0]
        total_weight = sum(votes.values())
        confidence = votes[best_char] / total_weight if total_weight > 0 else 0
        return best_char, confidence

    def _classify_feature_knn(self, cell, k=7):
        """kNN classification on structural features."""
        feat = extract_features(cell)
        if feat is None:
            return None, 0.0

        distances = []
        for char, feat_list in self.feature_samples.items():
            for sample_feat in feat_list:
                if sample_feat is not None:
                    dist = np.sum((feat - sample_feat) ** 2)
                    distances.append((dist, char))

        distances.sort(key=lambda x: x[0])
        votes = Counter()
        for dist, char in distances[:k]:
            weight = 1.0 / (dist + 1e-6)
            votes[char] += weight

        best_char = votes.most_common(1)[0][0]
        total_weight = sum(votes.values())
        confidence = votes[best_char] / total_weight if total_weight > 0 else 0
        return best_char, confidence

    def _classify_ensemble(self, cell):
        """Ensemble of multiple classification methods with weighted voting."""
        tmpl_char, tmpl_conf = self._classify_template(cell)
        knn_char, knn_conf = self._classify_knn(cell, k=5)
        feat_char, feat_conf = self._classify_feature_knn(cell, k=7)

        votes = Counter()
        votes[tmpl_char] += 1.0 * tmpl_conf
        votes[knn_char] += 1.5 * knn_conf
        if feat_char is not None:
            votes[feat_char] += 1.5 * feat_conf

        best_char = votes.most_common(1)[0][0]
        total = sum(votes.values())
        confidence = votes[best_char] / total if total > 0 else 0
        return best_char, confidence

    def save(self, path='classifier_data.npz'):
        """Save classifier data."""
        save_dict = {}
        for char, cells in self.samples.items():
            save_dict[f'samples_{char}'] = np.stack(cells)
        np.savez_compressed(path, **save_dict)
        print(f"Classifier data saved to {path}")

    def load(self, path='classifier_data.npz'):
        """Load classifier data."""
        data = np.load(path)
        self.samples.clear()
        self.feature_samples.clear()
        for key in data.files:
            char = key.replace('samples_', '')
            cells = data[key]
            for i in range(cells.shape[0]):
                self.add_sample(char, cells[i])
        self.build()
        print(f"Classifier loaded from {path}")


def find_rows_with_known_data(img, ref_data):
    """Find rows matching reference addresses using Tesseract OCR.

    Reads 10-digit hex addresses from each visible row, matches the last
    5 significant digits against reference data.

    Args:
        img: Grayscale frame image
        ref_data: {addr_int: [hex_str_list]} from load_reference()

    Returns:
        List of (addr_int, row_y) with deduplication and sequential validation.
    """
    import pytesseract
    from PIL import Image

    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']
    addr_x = ADDR_X_START
    addr_digits = CAL['address_digits']  # 10
    addr_spacing = CAL['address_char_spacing']

    # Width of address field: 10 digits x 13.92px + margin
    addr_width = int(addr_digits * addr_spacing) + 20

    ref_addr_set = set(ref_data.keys())
    # Build lookup by last 4 hex chars for fuzzy matching
    ref_by_suffix = {}
    for addr in ref_addr_set:
        suffix = f'{addr:05X}'[-4:]
        ref_by_suffix.setdefault(suffix, []).append(addr)

    candidates = []  # (addr_int, row_y, row_idx, match_quality)
    for row_idx in range(num_rows):
        row_y = y_first + row_idx * row_h

        # Crop address region
        y1 = int(row_y - CELL_H // 2 - 2)
        y2 = int(row_y + CELL_H // 2 + 2)
        x1 = max(0, addr_x - 5)
        x2 = min(img.shape[1], addr_x + addr_width)

        if y2 > img.shape[0] or y1 < 0:
            continue

        addr_crop = img[y1:y2, x1:x2]

        # Scale up 3x for better Tesseract recognition on small text
        scale = 3
        addr_crop_big = cv2.resize(addr_crop, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

        pil_crop = Image.fromarray(addr_crop_big)
        text = pytesseract.image_to_string(
            pil_crop,
            config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFabcdef'
        ).strip().replace(' ', '').upper()

        # Strip trailing non-hex chars (Xeltek shows 'H' suffix)
        clean = ''
        for ch in text:
            if ch in '0123456789ABCDEF':
                clean += ch
            else:
                break

        if len(clean) < 4:
            continue

        # Try exact match
        try:
            addr_val = int(clean, 16)
        except ValueError:
            continue

        if addr_val in ref_addr_set:
            candidates.append((addr_val, row_y, row_idx, len(clean)))
            continue

        # Try matching the last 4-5 significant hex digits
        matched = False
        for ref_addr in ref_addr_set:
            ref_hex = f'{ref_addr:05X}'
            if len(clean) >= 5 and clean[-5:] == ref_hex:
                candidates.append((ref_addr, row_y, row_idx, 5))
                matched = True
                break
            elif len(clean) >= 4 and clean[-4:] == ref_hex[-4:]:
                candidates.append((ref_addr, row_y, row_idx, 4))
                matched = True
                break

    # Deduplicate: keep best quality match per address
    seen_addrs = {}
    for addr, row_y, row_idx, quality in candidates:
        if addr not in seen_addrs or quality > seen_addrs[addr][2]:
            seen_addrs[addr] = (row_y, row_idx, quality)

    result = [(addr, info[0]) for addr, info in sorted(seen_addrs.items(),
              key=lambda x: x[1][1])]

    # Validate sequential ordering (addresses should increase by 0x10 per row)
    validated = []
    for i, (addr, row_y) in enumerate(result):
        valid = True
        if i > 0:
            prev_addr = validated[-1][0]
            prev_row_y = validated[-1][1]
            expected_diff = round((row_y - prev_row_y) / row_h) * 0x10
            actual_diff = addr - prev_addr
            if abs(actual_diff - expected_diff) > 0x10:
                valid = False
        if valid:
            validated.append((addr, row_y))

    return validated


def build_classifier_from_frames(ref_data, frame_paths):
    """Build classifier from frames with known reference data.

    Uses byte_positions array from calibration to locate each hex digit
    precisely, rather than computing positions from a formula.

    Args:
        ref_data: {addr_int: [hex_str_list]} from load_reference()
        frame_paths: List of frame file paths to process

    Returns:
        HexDigitClassifier with labeled samples from all matched rows.
    """
    classifier = HexDigitClassifier()

    for frame_path in frame_paths:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        matches = find_rows_with_known_data(img, ref_data)
        if not matches:
            continue

        n_samples = 0
        for addr, row_y in matches:
            hex_bytes = ref_data[addr]
            for byte_idx in range(16):
                byte_str = hex_bytes[byte_idx]
                hi_x = BYTE_POSITIONS[byte_idx]
                lo_x = hi_x + BYTE_DIGIT_SPACING

                # High nybble
                cell = extract_cell(img, row_y, hi_x)
                classifier.add_sample(byte_str[0], cell)
                n_samples += 1

                # Low nybble
                cell = extract_cell(img, row_y, lo_x)
                classifier.add_sample(byte_str[1], cell)
                n_samples += 1

        print(f"  {os.path.basename(frame_path)}: {len(matches)} rows, "
              f"{n_samples} digit samples")

    classifier.build()
    return classifier


def test_classifier(img, ref_data, classifier, method='ensemble', verbose=False):
    """Test classifier accuracy on a frame with known reference data."""
    matches = find_rows_with_known_data(img, ref_data)

    total = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for addr, row_y in matches:
        expected_bytes = ref_data[addr]
        for byte_idx in range(16):
            byte_str = expected_bytes[byte_idx]
            hi_x = BYTE_POSITIONS[byte_idx]
            lo_x = hi_x + BYTE_DIGIT_SPACING

            for digit_pos, char_x in enumerate([hi_x, lo_x]):
                expected_char = byte_str[digit_pos]
                cell = extract_cell(img, row_y, char_x)
                predicted, conf = classifier.classify(cell, method=method)

                total += 1
                if predicted == expected_char:
                    correct += 1
                elif verbose:
                    print(f"  WRONG: ${addr:05X}[{byte_idx}][{digit_pos}] "
                          f"{expected_char}->{predicted} (conf={conf:.3f})")

                confusion[expected_char][predicted] += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, total, correct, confusion


def print_confusion(confusion):
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
                print(f"   ·", end='')
        print(f"  | {total:>4} {acc:.0%}")


def main():
    """Standalone test: scan for reference frames, build classifier, evaluate."""
    print("=== Hex Digit Classifier (FM-203) ===\n")

    ref_data = load_reference()
    print(f"Reference: {len(ref_data)} address lines "
          f"(${min(ref_data):05X}-${max(ref_data)+0xF:05X})")

    frame_dir = 'frames'
    if not os.path.isdir(frame_dir):
        print(f"Error: {frame_dir} not found")
        return

    # Scan for frames showing reference addresses
    print("\nScanning for reference-visible frames...")
    train_frames = []
    test_frame = None

    for f_num in range(15000, 18000, 10):
        path = os.path.join(frame_dir, f'frame_{f_num:05d}.png')
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        matches = find_rows_with_known_data(img, ref_data)
        if len(matches) >= 3:
            if test_frame is None and len(train_frames) >= 8:
                test_frame = path
                print(f"  Hold-out test: {path} ({len(matches)} rows)")
            else:
                train_frames.append(path)
                print(f"  Train: {path} ({len(matches)} rows)")
            if len(train_frames) >= 20:
                break

    if not train_frames:
        print("No training frames found in range 15000-18000.")
        print("Trying broader scan (0-20070, step 200)...")
        for f_num in range(0, 20100, 200):
            path = os.path.join(frame_dir, f'frame_{f_num:05d}.png')
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            matches = find_rows_with_known_data(img, ref_data)
            if len(matches) >= 3:
                train_frames.append(path)
                print(f"  Train: {path} ({len(matches)} rows)")
                if len(train_frames) >= 15:
                    break

    if not train_frames:
        print("ERROR: No reference-visible frames found!")
        return

    print(f"\nBuilding classifier from {len(train_frames)} training frames...")
    classifier = build_classifier_from_frames(ref_data, train_frames)

    # Test on held-out frame
    if test_frame:
        test_img = cv2.imread(test_frame, cv2.IMREAD_GRAYSCALE)
        print(f"\nTesting on held-out frame: {test_frame}")

        for method in ['template', 'knn', 'feature_knn', 'ensemble']:
            acc, total, correct_count, confusion = test_classifier(
                test_img, ref_data, classifier, method=method
            )
            print(f"\n--- {method} ---")
            print(f"Accuracy: {acc:.1%} ({correct_count}/{total})")
            print_confusion(confusion)

    classifier.save()
    print("\nDone.")


if __name__ == '__main__':
    main()
