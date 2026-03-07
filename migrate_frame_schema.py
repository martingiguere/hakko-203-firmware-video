#!/usr/bin/env python3
"""One-time migration: separate video frames from extracted frames in crop_index.json."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')

from frame_utils import migrate_crop_index, migrate_frame_moves, MAX_EXTRACTED_FRAME


def main():
    print("=== Migrate Frame Schema ===\n")

    # Load crop_index
    with open(CROP_INDEX_PATH) as f:
        crop_index = json.load(f)

    # Count before
    total_addrs = len([k for k in crop_index if k != 'ref_addresses' and isinstance(crop_index[k], dict)])
    addrs_with_video = 0
    video_frame_count = 0
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses' or not isinstance(entry, dict):
            continue
        big = [f for f in entry.get('frames', []) if f > MAX_EXTRACTED_FRAME]
        if big:
            addrs_with_video += 1
            video_frame_count += len(big)

    print(f"Before: {total_addrs} addresses, {addrs_with_video} with video frames, "
          f"{video_frame_count} video frame refs to migrate")

    if video_frame_count == 0:
        print("Nothing to migrate (already done or no video frames).")
        return

    # Migrate crop_index
    migrated = migrate_crop_index(crop_index, CROPS_DIR)
    print(f"Migrated {migrated} video frame references in crop_index")

    # Save
    tmp_path = CROP_INDEX_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f)
    os.replace(tmp_path, CROP_INDEX_PATH)
    print(f"Saved {CROP_INDEX_PATH}")

    # Migrate frame_moves.json
    moves_migrated = migrate_frame_moves(FRAME_MOVES_PATH)
    print(f"Migrated {moves_migrated} frame move entries")

    # Verify
    with open(CROP_INDEX_PATH) as f:
        ci = json.load(f)
    violations = 0
    for addr, entry in ci.items():
        if addr == 'ref_addresses' or not isinstance(entry, dict):
            continue
        for f in entry.get('frames', []):
            if f > MAX_EXTRACTED_FRAME:
                violations += 1
    print(f"\nVerification: {violations} frames > {MAX_EXTRACTED_FRAME} still in 'frames' arrays "
          f"({'FAIL' if violations else 'OK'})")

    # Count video_frames
    vf_count = sum(len(e.get('video_frames', []))
                   for e in ci.values() if isinstance(e, dict))
    print(f"Total video_frames entries: {vf_count}")


if __name__ == '__main__':
    main()
