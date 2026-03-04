#!/usr/bin/env python3
"""
Fix D→C address misclassification in 0x0D000–0x0DFFF range.

The kNN classifier systematically misreads hex digit D as C at address digit
position 1 during frames ~13368–14493. This causes:
  1. Gap: addresses at $0D050–$0DF70 have zero frame data (filed under 0x0C)
  2. Contamination: ~559 frame observations from 0x0D rows are mixed into
     0x0C entries, corrupting byte consensus for 28+ addresses

The correction is +0x1000: any contaminated 0x0C frame's true address is
0x0C_addr + 0x1000.

Steps:
  1. Detect contamination boundaries from crop_index.json
  2. Move frame data from 0x0C → 0x0D entries (including crop PNGs)
  3. Recompute byte consensus → update extracted_firmware.txt
  4. Rebuild downstream files (postprocess, gap precompute)
  5. Reset review state for affected addresses
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
EXTRACTED_FW_PATH = os.path.join(PROJECT_ROOT, 'extracted_firmware.txt')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')
MERGED_FW_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')


def load_crop_index():
    with open(CROP_INDEX_PATH) as f:
        return json.load(f)


def save_crop_index(crop_index):
    with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f, indent=2)
    print(f"  Saved {CROP_INDEX_PATH}")


# ── Step 1: Detect contamination boundaries ─────────────────────────────────

def detect_boundaries(crop_index):
    """Find the frame range where D→C misreads occur.

    Uses 0x0D040 (last correctly-read address before the gap) to find the
    contamination start, and 0x0DF80 (first correctly-read address after the
    gap) for the end. This avoids 0x0D000-0x0D030 which have mixed early/late
    frames, and avoids scanning 0x0E/0x0F ranges.
    """
    # Find the last pre-gap D address (search backwards from 0D040)
    contam_start = None
    for addr_int in range(0x0D040, 0x0CFFF, -0x10):
        key = f"{addr_int:05X}"
        if key in crop_index:
            contam_start = max(crop_index[key]['frames'])
            print(f"  Using {key} (max frame {contam_start}) as start boundary")
            break
    if contam_start is None:
        print("ERROR: Could not find lower contamination boundary")
        sys.exit(1)

    # Min frame in 0x0DF80–0x0DFF0 (first post-gap D addresses)
    contam_end = float('inf')
    for addr_int in range(0x0DF80, 0x0E000, 0x10):
        key = f"{addr_int:05X}"
        if key in crop_index:
            contam_end = min(contam_end, min(crop_index[key]['frames']))

    if contam_end == float('inf'):
        print("ERROR: Could not find upper contamination boundary")
        sys.exit(1)

    print(f"  Contamination window: frame {contam_start} < f < {contam_end}")
    return contam_start, contam_end


# ── Step 2: Move frame data from 0x0C → 0x0D entries ────────────────────────

def move_contaminated_frames(crop_index, contam_start, contam_end):
    """
    For each 0x0C entry, split frames into clean/contaminated.
    Move contaminated frames to the corrected 0x0D address (+0x1000).
    Also move crop PNGs.
    """
    affected_0c = set()
    affected_0d = set()
    frames_moved = 0
    crops_moved = 0
    crops_missing = 0
    entries_emptied = 0

    c_keys = sorted(k for k in crop_index if k.startswith('0C')
                    and k != 'ref_addresses')

    for c_key in c_keys:
        entry = crop_index[c_key]
        frames = entry['frames']

        # Split into clean and contaminated
        clean_frames = [f for f in frames if not (contam_start < f < contam_end)]
        contam_frames = [f for f in frames if contam_start < f < contam_end]

        if not contam_frames:
            continue

        c_addr = int(c_key, 16)
        d_addr = c_addr + 0x1000
        d_key = f"{d_addr:05X}"
        affected_0c.add(c_key)
        affected_0d.add(d_key)

        # Create 0x0D entry if needed
        if d_key not in crop_index:
            crop_index[d_key] = {
                'frames': [],
                'readings': {},
                'confidences': {},
            }

        d_entry = crop_index[d_key]

        for frame in contam_frames:
            frame_str = str(frame)

            # Move readings and confidences
            if frame_str in entry.get('readings', {}):
                d_entry['readings'][frame_str] = entry['readings'].pop(frame_str)
            if frame_str in entry.get('confidences', {}):
                d_entry['confidences'][frame_str] = entry['confidences'].pop(frame_str)

            # Add frame to 0x0D entry
            if frame not in d_entry['frames']:
                d_entry['frames'].append(frame)

            # Move crop PNG
            c_dir = os.path.join(CROPS_DIR, c_key.lower())
            d_dir = os.path.join(CROPS_DIR, d_key.lower())
            crop_name = f"frame_{frame:05d}.png"
            src_path = os.path.join(c_dir, crop_name)
            dst_path = os.path.join(d_dir, crop_name)

            if os.path.exists(src_path):
                os.makedirs(d_dir, exist_ok=True)
                shutil.move(src_path, dst_path)
                crops_moved += 1
            else:
                crops_missing += 1

            frames_moved += 1

        # Update 0x0C entry: keep only clean frames
        entry['frames'] = clean_frames

        # Sort 0x0D frames
        d_entry['frames'] = sorted(d_entry['frames'])

        # Remove empty 0x0C entries
        if not clean_frames:
            del crop_index[c_key]
            entries_emptied += 1
            # Remove empty crop directory
            c_dir = os.path.join(CROPS_DIR, c_key.lower())
            if os.path.isdir(c_dir) and not os.listdir(c_dir):
                os.rmdir(c_dir)

    print(f"  Frames moved: {frames_moved}")
    print(f"  Crops moved: {crops_moved} (missing: {crops_missing})")
    print(f"  0x0C entries cleaned: {len(affected_0c)}")
    print(f"  0x0C entries emptied (removed): {entries_emptied}")
    print(f"  0x0D entries created/updated: {len(affected_0d)}")

    return affected_0c, affected_0d


# ── Step 3: Recompute byte consensus → update extracted_firmware.txt ─────────

def weighted_majority_vote(readings, confidences):
    """
    For each byte position, do weighted majority voting across frames.
    Weight = per-byte kNN confidence.
    Returns list of 16 hex strings and total observation count.
    """
    if not readings:
        return None, 0

    result = []
    for byte_idx in range(16):
        votes = Counter()
        for frame_str, byte_list in readings.items():
            if byte_idx < len(byte_list):
                weight = 1.0
                if frame_str in confidences:
                    conf_list = confidences[frame_str]
                    if byte_idx < len(conf_list):
                        weight = conf_list[byte_idx]
                votes[byte_list[byte_idx]] += weight
        if votes:
            result.append(votes.most_common(1)[0][0])
        else:
            result.append('FF')

    obs_count = len(readings)
    return result, obs_count


def update_extracted_firmware(crop_index, affected_0c, affected_0d):
    """
    Recompute byte consensus for all affected addresses and update
    extracted_firmware.txt.
    """
    all_affected = affected_0c | affected_0d

    # Load existing extracted_firmware.txt
    lines_by_addr = {}  # addr_str -> line text
    header_lines = []
    with open(EXTRACTED_FW_PATH) as f:
        for line in f:
            stripped = line.rstrip('\n')
            if stripped.lstrip().startswith('#') or not stripped.strip():
                header_lines.append(stripped)
                continue
            parts = stripped.split(':')
            if len(parts) >= 2:
                addr_str = parts[0].strip().upper()
                lines_by_addr[addr_str] = stripped

    # Recompute consensus for affected addresses
    updated = 0
    added = 0
    for addr_key in sorted(all_affected):
        if addr_key in crop_index:
            entry = crop_index[addr_key]
            readings = entry.get('readings', {})
            confidences = entry.get('confidences', {})
            consensus, obs_count = weighted_majority_vote(readings, confidences)

            if consensus and obs_count > 0:
                hex_str = ' '.join(consensus)
                new_line = f"{addr_key}: {hex_str}  [{obs_count} obs]"
                if addr_key in lines_by_addr:
                    updated += 1
                else:
                    added += 1
                lines_by_addr[addr_key] = new_line
        else:
            # Entry was removed (emptied 0x0C) — check if it had [REF] data
            if addr_key in lines_by_addr:
                existing = lines_by_addr[addr_key]
                if '[REF]' not in existing:
                    # Remove non-REF line for emptied entry
                    del lines_by_addr[addr_key]

    # Write back
    with open(EXTRACTED_FW_PATH, 'w') as f:
        for h in header_lines:
            f.write(h + '\n')
        for addr_str in sorted(lines_by_addr.keys(),
                                key=lambda x: int(x, 16)):
            f.write(lines_by_addr[addr_str] + '\n')

    print(f"  Updated {updated} lines, added {added} lines in extracted_firmware.txt")


# ── Step 4: Rebuild downstream files ────────────────────────────────────────

def rebuild_downstream():
    """Run postprocess_firmware.py and precompute_gaps.py."""
    print("\n  Running postprocess_firmware.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'postprocess_firmware.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: postprocess_firmware.py failed:\n{result.stderr}")
        sys.exit(1)
    # Print summary lines
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Coverage', 'Final data', 'Hex dump',
                                      'Source map', 'binary', 'MATCH']):
            print(f"    {line}")

    print("\n  Running precompute_gaps.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'firmware_review_tool/precompute_gaps.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: precompute_gaps.py failed:\n{result.stderr}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Missing', 'Addresses with', 'Unreachable',
                                      'Total context', 'Done']):
            print(f"    {line}")


# ── Step 5: Reset review state for affected addresses ────────────────────────

def reset_review_state(affected_0c, affected_0d):
    """Reset review status for all affected addresses."""
    if not os.path.exists(REVIEW_STATE_PATH):
        print("  review_state.json not found — skipping")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    if 'lines' not in state:
        print("  review_state.json has no 'lines' — skipping")
        return

    # Load current firmware_merged.txt for updated bytes
    merged_bytes = {}
    with open(MERGED_FW_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                addr_str = parts[0].strip().upper()
                rest = parts[1].strip()
                # Remove annotations
                rest = re.sub(r'\[.*?\]', '', rest).strip()
                hex_bytes = rest.split()
                if len(hex_bytes) == 16:
                    merged_bytes[addr_str] = hex_bytes

    all_affected = affected_0c | affected_0d
    reset_count = 0

    for addr_key in sorted(all_affected):
        if addr_key in state['lines']:
            if addr_key in merged_bytes:
                state['lines'][addr_key]['bytes'] = merged_bytes[addr_key]
                state['lines'][addr_key]['source'] = 'merged'
            state['lines'][addr_key]['status'] = 'unreviewed'
            state['lines'][addr_key]['edited_positions'] = []
            reset_count += 1
        elif addr_key in merged_bytes:
            # New address not yet in review state
            state['lines'][addr_key] = {
                'status': 'unreviewed',
                'bytes': merged_bytes[addr_key],
                'source': 'merged',
                'edited_positions': [],
            }
            reset_count += 1

    with open(REVIEW_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print(f"  Reset {reset_count} addresses in review_state.json")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fix D→C Address Misclassification (0x0D000–0x0DFFF)")
    print("=" * 60)

    # Step 1
    print("\nStep 1: Detect contamination boundaries")
    crop_index = load_crop_index()
    contam_start, contam_end = detect_boundaries(crop_index)

    # Step 2
    print("\nStep 2: Move contaminated frames from 0x0C → 0x0D")
    affected_0c, affected_0d = move_contaminated_frames(
        crop_index, contam_start, contam_end
    )

    if not affected_0c and not affected_0d:
        print("\nNo contaminated frames found. Nothing to do.")
        return

    # Save crop_index before recomputing consensus
    print("\nSaving updated crop_index.json...")
    save_crop_index(crop_index)

    # Step 3
    print("\nStep 3: Recompute byte consensus")
    update_extracted_firmware(crop_index, affected_0c, affected_0d)

    # Step 4
    print("\nStep 4: Rebuild downstream files")
    rebuild_downstream()

    # Step 5
    print("\nStep 5: Reset review state")
    reset_review_state(affected_0c, affected_0d)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Affected 0x0C addresses: {len(affected_0c)}")
    print(f"  Affected 0x0D addresses: {len(affected_0d)}")

    # Quick verification: check 0x0D050–0x0DF70 gap
    crop_index_check = load_crop_index()
    d_range = [k for k in crop_index_check
               if k.startswith('0D') and 0x0D050 <= int(k, 16) <= 0x0DF70]
    print(f"  0x0D050–0x0DF70 entries now in crop_index: {len(d_range)}")

    # Check merged firmware for those addresses
    merged_d_count = 0
    if os.path.exists(MERGED_FW_PATH):
        with open(MERGED_FW_PATH) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        addr = int(parts[0].strip(), 16)
                        if 0x0D050 <= addr <= 0x0DF70:
                            merged_d_count += 1
                    except ValueError:
                        pass
    print(f"  0x0D050–0x0DF70 lines in firmware_merged.txt: {merged_d_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
