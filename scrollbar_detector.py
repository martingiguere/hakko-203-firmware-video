#!/usr/bin/env python3
"""
Scrollbar thumb position detector for Xeltek SuperPro Edit Buffer window.

Detects the vertical scrollbar thumb position in video frames and estimates
the top visible address. Used as a soft constraint in validate_address_sequence()
to boost correct anchors and penalize implausible ones.

Calibration: linear mapping from thumb center Y pixel to address,
fitted against 41 manual trajectory waypoint frames.
  addr = slope * thumb_center_y + intercept
  Mean error: $01F4 (~500 bytes), max error: $04C0
  Zero errors > $1000 — safely disambiguates C/D confusion ($0Cxxx vs $0Dxxx)
"""

import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

# Load scrollbar geometry from calibration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_ROOT, 'grid_calibration.json')) as f:
    _CAL = json.load(f)

_SB = _CAL.get('scrollbar', {})
SB_X_MIN = _SB.get('x_min', 1375)
SB_X_MAX = _SB.get('x_max', 1395)
SB_TRACK_Y_MIN = _SB.get('track_y_min', 187)
SB_TRACK_Y_MAX = _SB.get('track_y_max', 563)
SB_SLOPE = _SB.get('slope', 217.9)
SB_INTERCEPT = _SB.get('intercept', -40780)
SB_MIN_THUMB_HEIGHT = _SB.get('min_thumb_height', 8)


def detect_scrollbar_address(img):
    """Detect scrollbar thumb position and estimate the top visible address.

    Args:
        img: Grayscale frame image (1080x1920)

    Returns:
        (estimated_addr, confidence) or (None, 0.0) if detection fails.
        confidence: 0.0-1.0 based on thumb clarity (height).
    """
    if img is None or img.shape[0] < SB_TRACK_Y_MAX or img.shape[1] < SB_X_MAX:
        return None, 0.0

    # Extract scrollbar track strip
    track_strip = img[SB_TRACK_Y_MIN:SB_TRACK_Y_MAX, SB_X_MIN:SB_X_MAX]
    row_avg = track_strip.mean(axis=1).astype(float)

    # Smooth to reduce noise
    smoothed = uniform_filter1d(row_avg, size=3)

    # Track background: the bright empty area (90th percentile)
    track_bg = np.percentile(smoothed, 90)

    if track_bg < 180:
        # Track not bright enough — likely occluded or wrong region
        return None, 0.0

    # Find thumb: contiguous band darker than track but brighter than content
    thumb_mask = (smoothed > 150) & (smoothed < track_bg - 15)

    # Find the largest contiguous run
    best_start, best_end, best_len = 0, 0, 0
    in_thumb = False
    start = 0
    for i in range(len(thumb_mask)):
        if thumb_mask[i] and not in_thumb:
            start = i
            in_thumb = True
        elif not thumb_mask[i] and in_thumb:
            if i - start > best_len:
                best_start, best_end, best_len = start, i, i - start
            in_thumb = False
    if in_thumb and len(thumb_mask) - start > best_len:
        best_start = start
        best_end = len(thumb_mask)
        best_len = best_end - start

    if best_len < SB_MIN_THUMB_HEIGHT:
        return None, 0.0

    # Compute thumb center in absolute frame coordinates
    center_y = SB_TRACK_Y_MIN + (best_start + best_end) / 2.0

    # Map to address
    estimated_addr = int(SB_SLOPE * center_y + SB_INTERCEPT)
    estimated_addr = max(0, min(estimated_addr, 0x13FF0))
    # Align to 0x10
    estimated_addr = (estimated_addr // 0x10) * 0x10

    # Confidence based on thumb height
    # 30px = 1.0 (clear, steady frame), 10px = 0.5, 8px = 0.3
    if best_len >= 25:
        confidence = 1.0
    elif best_len >= 15:
        confidence = 0.7
    else:
        confidence = 0.3 + 0.4 * (best_len - SB_MIN_THUMB_HEIGHT) / (15 - SB_MIN_THUMB_HEIGHT)
        confidence = max(0.3, min(confidence, 0.7))

    return estimated_addr, confidence


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import cv2
    from manual_trajectory import MANUAL_TRAJECTORY

    print("=== Scrollbar Detector Test ===\n")

    detected = 0
    total = 0
    errors = []

    for frame_num, expected_addr in MANUAL_TRAJECTORY:
        if expected_addr is None or expected_addr < 0:
            continue

        fname = os.path.join(PROJECT_ROOT, 'frames', f'frame_{frame_num:05d}.png')
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        total += 1
        addr, conf = detect_scrollbar_address(img)

        if addr is not None:
            detected += 1
            error = abs(addr - expected_addr)
            errors.append(error)
            status = "OK" if error < 0x1000 else "BAD"
            print(f"  F{frame_num:5d}: expected ${expected_addr:05X}, "
                  f"detected ${addr:05X} (err=${error:04X}, conf={conf:.2f}) [{status}]")
        else:
            print(f"  F{frame_num:5d}: expected ${expected_addr:05X}, "
                  f"NOT DETECTED")

    print(f"\nDetected: {detected}/{total} ({detected/total*100:.0f}%)")
    if errors:
        print(f"Mean error: ${int(np.mean(errors)):04X}")
        print(f"Max error: ${int(max(errors)):04X}")
        print(f"Errors > $1000: {sum(1 for e in errors if e > 0x1000)}")
