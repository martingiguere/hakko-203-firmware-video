#!/usr/bin/env python3
"""
Relocate misassigned frames from FF-forced addresses to correct ROM addresses.

Scans crop_index.json for frames at FF-forced addresses that have non-FF byte
data (indicating misassignment). Uses the scrollbar thumb position to estimate
where each frame really belongs, then moves it to the correct ROM address.

Runs as a pipeline step after trajectory fix and before outlier vote correction.
"""

import cv2
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from fix_address_trajectory import execute_moves, log_frame_moves, load_crop_index, save_crop_index
from scrollbar_detector import detect_scrollbar_address
from memory_map_utils import load_memory_map, is_ff_forced
from template_matcher import CAL

FRAMES_DIR = 'frames'
FIRST_ROW_Y = CAL['first_row_center_y']  # 272.0
ROW_HEIGHT = CAL['row_height']            # 28.0
MIN_SCROLLBAR_CONFIDENCE = 0.5


def main():
    print("=== FF-Forced Frame Relocation ===\n")

    mmap = load_memory_map()
    crop_index = load_crop_index()

    # Collect candidates: frames at FF-forced addresses with non-FF data
    candidates = []
    for addr_str, entry in crop_index.items():
        if addr_str == 'ref_addresses':
            continue
        try:
            addr = int(addr_str, 16)
        except ValueError:
            continue

        if not is_ff_forced(mmap, addr):
            continue

        readings = entry.get('readings', {})
        row_ys = entry.get('row_ys', {})

        for frame_key, reading in readings.items():
            ff_count = sum(1 for b in reading if b == 'FF')
            if ff_count >= 14:
                continue  # genuinely FF, not misassigned

            row_y = row_ys.get(frame_key)
            if row_y is None:
                continue

            # Determine frame number and type
            if frame_key.startswith('v'):
                frame_int = int(frame_key[1:])
                is_video = True
            else:
                frame_int = int(frame_key)
                is_video = False

            candidates.append((addr_str, frame_key, frame_int, is_video, row_y))

    print(f"Candidates (non-FF data at FF-forced addresses): {len(candidates)}")

    if not candidates:
        print("No candidates found.")
        return

    # For each candidate, detect scrollbar position and compute destination
    moves = []
    no_scrollbar = 0
    still_ff = 0
    frames_loaded = {}

    for addr_str, frame_key, frame_int, is_video, row_y in candidates:
        # Load frame image (cache to avoid reloading for multiple rows)
        if frame_int not in frames_loaded:
            if is_video:
                # Video frames not in frames/ directory
                frames_loaded[frame_int] = None
                continue
            frame_path = os.path.join(FRAMES_DIR, f'frame_{frame_int:05d}.png')
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frames_loaded[frame_int] = img
        else:
            img = frames_loaded[frame_int]

        if img is None:
            continue

        # Detect scrollbar position
        sb_addr, sb_conf = detect_scrollbar_address(img)
        if sb_addr is None or sb_conf < MIN_SCROLLBAR_CONFIDENCE:
            no_scrollbar += 1
            continue

        # Compute actual address for this row based on scrollbar estimate
        # scrollbar gives the top visible address; row_y determines offset
        row_offset = round((row_y - FIRST_ROW_Y) / ROW_HEIGHT)
        dest_addr = sb_addr + row_offset * 0x10

        # Validate destination
        if dest_addr < 0 or dest_addr > 0x13FF0 or dest_addr % 0x10 != 0:
            continue
        if is_ff_forced(mmap, dest_addr):
            still_ff += 1
            continue

        dest_str = f"{dest_addr:05X}"
        if dest_str == addr_str:
            continue  # already at correct address (unlikely but check)

        frame_id = frame_key if is_video else frame_int
        moves.append((addr_str, frame_id, dest_str))

    print(f"Moves to execute: {len(moves)}")
    print(f"  No scrollbar detected: {no_scrollbar}")
    print(f"  Scrollbar points to FF-forced: {still_ff}")

    if not moves:
        print("No moves to execute.")
        return

    # Execute moves
    execute_moves(crop_index, moves)

    # Log moves
    log_frame_moves(moves, strategy="ff_forced_relocation")

    # Save
    save_crop_index(crop_index)
    print(f"\nRelocated {len(moves)} frame-rows from FF-forced to ROM addresses")


if __name__ == '__main__':
    main()
