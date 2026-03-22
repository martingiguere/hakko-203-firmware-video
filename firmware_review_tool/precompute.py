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
import shutil

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
from frame_utils import is_video_frame_key, parse_frame_key, crop_filename

FRAMES_DIR = os.path.join(PROJECT_ROOT, 'frames')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
CROP_INDEX_PATH = os.path.join(CROPS_DIR, 'crop_index.json')
KNN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'fast_knn_classifier.npz')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')
FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

TOTAL_FRAMES = len([f for f in os.listdir(os.path.join(PROJECT_ROOT, 'frames'))
                     if f.endswith('.png')]) if os.path.isdir(os.path.join(PROJECT_ROOT, 'frames')) else 0

# Row crop geometry — asymmetric to center on ink (characters sit above row_y center)
CROP_Y_ABOVE = 17   # pixels above row_y (ink starts at ~row_y-15)
CROP_Y_BELOW = 11   # pixels below row_y (ink ends at ~row_y+2)
CROP_X_START = 284   # 5px margin before first address digit (ADDR_X_START=289)
CROP_X_END = 1120    # 10px margin after last byte digit (ends at ~1108)

BASE_ADDR = 0x00000
END_ADDR = 0x13FF0
MAX_ADDR_DIFF = 0x10000  # Max allowable difference between raw OCR read and validated address
EXTRA_ROWS_ABOVE = 2

REF_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'reference', 'reference_screenshot.png')
REF_CROPS_DIR = os.path.join(CROPS_DIR, 'ref')
REF_FIRST_ROW_Y = 127.0
REF_ROW_SPACING = 24.0
REF_NUM_ROWS = 16
REF_FIRST_ADDR = 0x0FF70
REF_CROP_X1 = 45
REF_CROP_X2 = 760
REF_CROP_HALF_H = 12


def load_frame_assignments():
    """Load shared frame assignments from extract_pipeline, if available."""
    if os.path.exists(FRAME_ASSIGNMENTS_PATH):
        with open(FRAME_ASSIGNMENTS_PATH) as f:
            data = json.load(f)
        print(f"Loaded frame assignments: {len(data)} frames")
        return data
    return None


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

    # Load shared frame assignments (from extract_pipeline.py)
    shared_assignments = load_frame_assignments()
    use_shared = shared_assignments is not None
    if use_shared:
        print("Using shared frame assignments (consistent with extraction)")
    else:
        print("WARNING: No frame_assignments.json — falling back to independent address validation")

    # Prepare output directory
    os.makedirs(CROPS_DIR, exist_ok=True)

    # Index: addr_upper -> {frames: [int], readings: {frame_str: [bytes]}, confidences: {frame_str: [floats]}, row_ys: {frame_str: float}}
    crop_index = {}
    prev_img = None
    skipped = 0
    processed = 0
    total_crops = 0
    start_time = time.time()

    for frame_num in range(TOTAL_FRAMES):
        frame_name = f'frame_{frame_num:05d}.png'
        frame_path = os.path.join(FRAMES_DIR, frame_name)
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

        # Get validated address assignments
        if use_shared and frame_name in shared_assignments:
            # Use pre-computed assignments from extract_pipeline
            validated_rows = []
            for a in shared_assignments[frame_name]:
                addr_int = a['addr']
                row_y = a['row_y']
                conf = a['conf']
                if BASE_ADDR <= addr_int <= END_ADDR and addr_int % 0x10 == 0:
                    validated_rows.append((addr_int, row_y, conf))
        else:
            # Fallback: independent address validation (legacy behavior)
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
            validated_full = validate_address_sequence(addr_results)
            validated_rows = []
            for (raw_addr, _, _), (addr_int, row_y, conf) in zip(addr_results, validated_full):
                if abs(raw_addr - addr_int) > MAX_ADDR_DIFF:
                    continue
                if addr_int < BASE_ADDR or addr_int > END_ADDR:
                    continue
                if addr_int % 0x10 != 0:
                    continue
                validated_rows.append((addr_int, row_y, conf))

        for addr_int, row_y, conf in validated_rows:

            addr_upper = f"{addr_int:05X}"
            addr_lower = addr_upper.lower()

            # Crop the full row (asymmetric — shifted up to center on ink)
            y_center = int(round(row_y))
            y1 = max(0, y_center - CROP_Y_ABOVE)
            y2 = min(img.shape[0], y_center + CROP_Y_BELOW)
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
                crop_index[addr_upper] = {"frames": [], "readings": {}, "confidences": {}, "row_ys": {}}
            crop_index[addr_upper]["frames"].append(frame_num)
            crop_index[addr_upper]["readings"][str(frame_num)] = [
                b.upper() if b != "--" else "--" for b in hex_bytes
            ]
            crop_index[addr_upper]["confidences"][str(frame_num)] = [
                round(float(c), 3) for c in confidences
            ]
            crop_index[addr_upper].setdefault("row_ys", {})[str(frame_num)] = round(float(row_y), 1)

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

    # Apply frame moves ledger (survives re-runs)
    applied = apply_frame_moves(crop_index)
    if applied:
        print(f"  Applied {applied} frame moves from frame_moves.json")

    ref_addresses = precompute_ref_crops()
    if ref_addresses:
        crop_index["ref_addresses"] = ref_addresses

    # Write crop index
    with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f)
    print(f"  Crop index written to {CROP_INDEX_PATH}")
    print(f"  Index size: {os.path.getsize(CROP_INDEX_PATH) / 1024:.0f} KB")


def apply_frame_moves(crop_index):
    """Replay frame_moves.json onto the freshly-generated crop_index.

    Handles both extracted frames (integer) and video frames ("vNNNNN" string).
    """
    if not os.path.exists(FRAME_MOVES_PATH):
        return 0
    with open(FRAME_MOVES_PATH) as f:
        data = json.load(f)
    moves = data.get("moves", [])
    applied = 0
    skipped = 0
    for move in moves:
        frame = move["frame"]
        from_addr = move["from_addr"]
        to_addr = move["to_addr"]

        # Determine frame type
        if isinstance(frame, str) and frame.startswith('v'):
            is_video = True
            frame_int = int(frame[1:])
            frame_str = frame
            arr_key = 'video_frames'
        else:
            is_video = False
            frame_int = int(frame)
            frame_str = str(frame_int)
            arr_key = 'frames'

        if from_addr not in crop_index:
            print(f"  WARNING: frame_move skipped — {from_addr} not in crop_index (frame {frame})")
            skipped += 1
            continue
        src = crop_index[from_addr]
        if frame_int not in src.get(arr_key, []):
            print(f"  WARNING: frame_move skipped — frame {frame} not at {from_addr}")
            skipped += 1
            continue
        # Move readings + confidences + row_ys
        dst = crop_index.setdefault(to_addr, {"frames": [], "video_frames": [], "readings": {}, "confidences": {}, "row_ys": {}})
        dst.setdefault(arr_key, [])
        if frame_str in src.get("readings", {}):
            dst["readings"][frame_str] = src["readings"].pop(frame_str)
        if frame_str in src.get("confidences", {}):
            dst["confidences"][frame_str] = src["confidences"].pop(frame_str)
        if frame_str in src.get("row_ys", {}):
            dst.setdefault("row_ys", {})[frame_str] = src["row_ys"].pop(frame_str)
        src[arr_key].remove(frame_int)
        if frame_int not in dst[arr_key]:
            dst[arr_key].append(frame_int)
            dst[arr_key].sort()
        # Move crop PNG
        png_name = crop_filename(frame_int, is_video=is_video)
        src_png = os.path.join(CROPS_DIR, from_addr.lower(), png_name)
        dst_dir = os.path.join(CROPS_DIR, to_addr.lower())
        dst_png = os.path.join(dst_dir, png_name)
        if os.path.exists(src_png):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_png, dst_png)
        # Clean up empty source
        if not src.get("frames") and not src.get("video_frames") and not src.get("readings"):
            del crop_index[from_addr]
            src_dir_path = os.path.join(CROPS_DIR, from_addr.lower())
            if os.path.isdir(src_dir_path) and not os.listdir(src_dir_path):
                os.rmdir(src_dir_path)
        applied += 1
    if skipped:
        print(f"  Frame moves: applied {applied}, skipped {skipped}")

    # Clean up ghost frames: frames in the array but with no readings
    # (caused by chained moves where a later move re-adds the frame number
    # but the readings were already moved elsewhere)
    ghosts_removed = 0
    for addr, entry in list(crop_index.items()):
        if addr == 'ref_addresses':
            continue
        readings = entry.get("readings", {})
        for arr_key in ("frames", "video_frames"):
            clean = []
            for frame_int in entry.get(arr_key, []):
                if arr_key == "video_frames":
                    frame_str = f"v{frame_int}"
                else:
                    frame_str = str(frame_int)
                if frame_str in readings:
                    clean.append(frame_int)
                else:
                    ghosts_removed += 1
            entry[arr_key] = clean
    if ghosts_removed:
        print(f"  Ghost frames removed: {ghosts_removed}")

    return applied


def precompute_ref_crops():
    """Crop and resize reference screenshot rows to match video crop dimensions."""
    if not os.path.exists(REF_IMAGE_PATH):
        print(f"WARNING: Reference screenshot not found at {REF_IMAGE_PATH}")
        return []

    ref_img = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print(f"WARNING: Could not load reference screenshot")
        return []

    os.makedirs(REF_CROPS_DIR, exist_ok=True)

    target_w = CROP_X_END - CROP_X_START  # 836
    target_h = CROP_Y_ABOVE + CROP_Y_BELOW  # 28

    ref_addresses = []
    for row_idx in range(REF_NUM_ROWS):
        addr_int = REF_FIRST_ADDR + row_idx * 0x10
        addr_upper = f"{addr_int:05X}"
        addr_lower = addr_upper.lower()

        row_y = REF_FIRST_ROW_Y + row_idx * REF_ROW_SPACING
        y1 = int(round(row_y - REF_CROP_HALF_H))
        y2 = int(round(row_y + REF_CROP_HALF_H))
        y1 = max(0, y1)
        y2 = min(ref_img.shape[0], y2)

        crop = ref_img[y1:y2, REF_CROP_X1:REF_CROP_X2]
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

        crop_path = os.path.join(REF_CROPS_DIR, f'{addr_lower}.png')
        cv2.imwrite(crop_path, resized)
        ref_addresses.append(addr_upper)

    print(f"  Reference crops: {len(ref_addresses)} rows saved to {REF_CROPS_DIR}")
    return ref_addresses


if __name__ == '__main__':
    if '--ref-only' in sys.argv:
        ref_addresses = precompute_ref_crops()
        if ref_addresses and os.path.exists(CROP_INDEX_PATH):
            with open(CROP_INDEX_PATH, encoding='utf-8') as f:
                idx = json.load(f)
            idx["ref_addresses"] = ref_addresses
            with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
                json.dump(idx, f)
            print(f"Updated {CROP_INDEX_PATH} with ref_addresses")
    else:
        precompute()
