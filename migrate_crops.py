#!/usr/bin/env python3
"""
One-time migration: move crop PNGs from address-based to frame-based storage.

Before: crops/<address>/frame_NNNNN.png  (address-based, duplicated across addresses)
After:  crops/frames/<frame_id>/<row_y>.png  (frame-based, deduplicated)

Uses crop_index.json to look up row_y for each frame at each address.
Skips crops that already exist at the destination (dedup).

Usage:
    python3 migrate_crops.py              # migrate
    python3 migrate_crops.py --dry-run    # preview only
"""

import argparse
import json
import os
import re
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
FRAMES_DIR = os.path.join(CROPS_DIR, 'frames')
CROP_INDEX_PATH = os.path.join(CROPS_DIR, 'crop_index.json')

SKIP_DIRS = {'ref', 'gap', 'frames', 'crop_index.json', 'gap_context_index.json'}


def main():
    parser = argparse.ArgumentParser(description='Migrate crops from address-based to frame-based storage')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no file changes')
    args = parser.parse_args()

    print("=== Crop Storage Migration ===\n")

    # Load crop_index for row_y lookups
    print("Loading crop_index.json...")
    with open(CROP_INDEX_PATH) as f:
        ci = json.load(f)

    # Build reverse map: (addr, frame_key) -> row_y
    row_y_map = {}
    for addr, entry in ci.items():
        if addr == 'ref_addresses' or not isinstance(entry, dict):
            continue
        for fk, ry in entry.get('row_ys', {}).items():
            row_y_map[(addr.lower(), fk)] = int(ry)

    print(f"  row_y mappings: {len(row_y_map)}")

    # Scan existing address-based crop directories
    moved = 0
    skipped_exists = 0
    skipped_no_row_y = 0
    errors = 0
    bytes_saved = 0
    dirs_to_clean = []

    addr_dirs = sorted(d for d in os.listdir(CROPS_DIR)
                       if os.path.isdir(os.path.join(CROPS_DIR, d)) and d not in SKIP_DIRS)

    print(f"  Address directories to process: {len(addr_dirs)}\n")

    for addr_dir in addr_dirs:
        addr_path = os.path.join(CROPS_DIR, addr_dir)
        pngs = [f for f in os.listdir(addr_path) if f.endswith('.png')]
        if not pngs:
            dirs_to_clean.append(addr_path)
            continue

        for png in pngs:
            src_path = os.path.join(addr_path, png)

            # Parse frame number and video flag from filename
            m = re.match(r'frame_(v?)(\d+)\.png', png)
            if not m:
                continue
            is_video = m.group(1) == 'v'
            frame_num = int(m.group(2))
            frame_key = f"v{frame_num}" if is_video else str(frame_num)
            frame_id = f"v{frame_num}" if is_video else str(frame_num)

            # Look up row_y
            row_y = row_y_map.get((addr_dir, frame_key))
            if row_y is None:
                skipped_no_row_y += 1
                continue

            # Destination path
            dst_dir = os.path.join(FRAMES_DIR, frame_id)
            dst_path = os.path.join(dst_dir, f'{row_y}.png')

            if os.path.exists(dst_path):
                # Dedup: same crop already migrated from another address dir
                skipped_exists += 1
                bytes_saved += os.path.getsize(src_path)
                if not args.dry_run:
                    os.remove(src_path)
                continue

            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src_path, dst_path)
            moved += 1

        # Mark dir for cleanup
        dirs_to_clean.append(addr_path)

    # Clean up empty directories
    cleaned = 0
    if not args.dry_run:
        for d in dirs_to_clean:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                cleaned += 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Results:")
    print(f"  Crops moved: {moved}")
    print(f"  Duplicates removed: {skipped_exists} ({bytes_saved / 1024 / 1024:.0f} MB saved)")
    print(f"  Skipped (no row_y): {skipped_no_row_y}")
    print(f"  Empty dirs cleaned: {cleaned}")
    if errors:
        print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
